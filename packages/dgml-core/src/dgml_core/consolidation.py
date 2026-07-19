# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vision-LLM adjudicator for clustering consolidation (§5).

:class:`~clustering.consolidation.Scenario.consolidate` in the framework
selects the low-confidence tail and applies verdicts, but stays LLM-free.
This module supplies the missing half: a concrete
:class:`~clustering.consolidation.Adjudicator` that asks the configured vision
model to reconsider each borderline document against its nearest candidate
clusters, exactly the assign-vs-create question
:mod:`dgml_core.classification` already poses — but pre-seeded with the
embedding-derived candidates so the model adjudicates rather than searches
blind.

Two modes:

- **reassign** (default): one constrained tool call per document — *"which of
  these candidate types does this belong to, or is it novel?"* Run twice with
  the candidate order rotated; the confidence is the model's self-report scaled
  by the two-attempt agreement (§5.6 / §4.4), an ordinal signal.
- **repartition**: one batch grouping call over the whole selected subset (a
  contested region), mapping each document to the existing type its group
  matched or a novel bucket.

Everything is soft-fail: a per-document LLM error drops just that document's
verdict (its assignment is left untouched), and a total failure is caught by
the framework's :func:`~clustering.consolidation.consolidate` guard.
"""

from __future__ import annotations

import io
import json
from collections import Counter
from typing import TYPE_CHECKING, Any

from clustering.consolidation import AdjudicationRequest, AdjudicationVerdict

from .classification import ClassificationConfig, _resolve_api_key
from .llm import LLMConfig, call_with_tools
from .usage import OPERATION_CLUSTER
from .utils import image_to_data_url

if TYPE_CHECKING:
    from clustering.data.datasets import DocumentDataset

_TOOL_ADJUDICATE = "adjudicate"
_TOOL_REGROUP = "regroup_documents"
# Sentinel choice meaning "none of the candidates — a genuinely new type".
_NOVEL = "__novel__"
_TEXT_SNIPPET_CHARS = 800


class LLMAdjudicator:
    """A litellm-backed :class:`~clustering.consolidation.Adjudicator`.

    Construct with the workspace :class:`ClassificationConfig` (model + api-key
    precedence reused verbatim) and call it with the dataset + the framework's
    :class:`~clustering.consolidation.AdjudicationRequest` list.
    """

    def __init__(
        self,
        config: ClassificationConfig,
        *,
        attempts: int = 2,
        debug: bool = False,
    ) -> None:
        self.config = config
        self.attempts = max(1, attempts)
        self.debug = debug

    # ── Adjudicator protocol ─────────────────────────────────────────────
    def __call__(
        self,
        dataset: DocumentDataset,
        requests: list[AdjudicationRequest],
        *,
        mode: str,
        batch_size: int,
    ) -> dict[str, AdjudicationVerdict]:
        if not requests:
            return {}
        if mode == "repartition":
            return self._repartition(dataset, requests, batch_size)
        # "reassign" and "auto" both use the per-document path; "auto" would
        # route genuinely contested *regions* to repartition, but a per-doc
        # candidate pick is the safe, bounded default.
        return self._reassign(dataset, requests)

    # ── reassign: one decision per document ──────────────────────────────
    def _reassign(
        self, dataset: DocumentDataset, requests: list[AdjudicationRequest]
    ) -> dict[str, AdjudicationVerdict]:
        verdicts: dict[str, AdjudicationVerdict] = {}
        for req in requests:
            verdict = self._adjudicate_one(dataset, req)
            if verdict is not None:
                verdicts[req.doc_id] = verdict
        return verdicts

    def _adjudicate_one(
        self, dataset: DocumentDataset, req: AdjudicationRequest
    ) -> AdjudicationVerdict | None:
        try:
            record = dataset[req.doc_index]
        except Exception:
            return None

        picks: list[str] = []
        self_confs: list[float] = []
        rationale: str | None = None
        for attempt in range(self.attempts):
            # Rotate the candidate order between attempts so agreement reflects
            # a real robustness check rather than fixed position bias.
            choices = _rotate(req.candidate_labels, attempt) + [_NOVEL]
            content = _reassign_content(record, req)
            try:
                result = call_with_tools(
                    self._llm_config([req.doc_id]),
                    messages=[{"role": "user", "content": content}],
                    tools=[_adjudicate_tool(choices)],
                    tool_choice="required",
                )
            except Exception:
                continue
            choice, self_conf, rat = _parse_adjudicate(result.response)
            if choice is None:
                continue
            picks.append(choice)
            if self_conf is not None:
                self_confs.append(self_conf)
            rationale = rationale or rat

        if not picks:
            return None
        modal, count = Counter(picks).most_common(1)[0]
        agreement = count / len(picks)
        base = sum(self_confs) / len(self_confs) if self_confs else agreement
        confidence = max(0.0, min(1.0, base * agreement))
        assignment = None if modal == _NOVEL else modal
        return AdjudicationVerdict(
            assignment=assignment, confidence=confidence, rationale=rationale
        )

    # ── repartition: one batch grouping call ─────────────────────────────
    def _repartition(
        self,
        dataset: DocumentDataset,
        requests: list[AdjudicationRequest],
        batch_size: int,
    ) -> dict[str, AdjudicationVerdict]:
        batch = requests[: max(1, batch_size)]
        existing = sorted({c for req in batch for c in req.candidate_labels})
        labels: dict[str, AdjudicationRequest] = {}
        content: list[dict[str, Any]] = [{"type": "text", "text": _repartition_prompt(existing)}]
        for n, req in enumerate(batch, start=1):
            tag = f"doc_{n}"
            labels[tag] = req
            content.append({"type": "text", "text": f"=== Document {tag} ==="})
            try:
                record = dataset[req.doc_index]
            except Exception:
                continue
            url = _image_url(record)
            if url is not None:
                content.append({"type": "image_url", "image_url": {"url": url}})
            elif record.text:
                content.append({"type": "text", "text": record.text[:_TEXT_SNIPPET_CHARS]})

        try:
            result = call_with_tools(
                self._llm_config([req.doc_id for req in batch]),
                messages=[{"role": "user", "content": content}],
                tools=[_regroup_tool(existing)],
                tool_choice="required",
            )
            groups = _extract_groups(result.response)
        except Exception:
            return {}

        return _map_groups_to_verdicts(groups, labels)

    def _llm_config(self, doc_ids: list[str]) -> LLMConfig:
        return LLMConfig(
            model=self.config.model,
            api_key=_resolve_api_key(self.config),
            max_tokens=None,
            debug=self.debug,
            operation=OPERATION_CLUSTER,
            context={"consolidation": doc_ids},
        )


# ── helpers ─────────────────────────────────────────────────────────────────
def _rotate(items: list[str], by: int) -> list[str]:
    if not items:
        return []
    k = by % len(items)
    return items[k:] + items[:k]


def _image_url(record: Any) -> str | None:
    image = getattr(record, "image", None)
    if image is None:
        return None
    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return image_to_data_url(buf.getvalue())
    except Exception:
        return None


def _reassign_content(record: Any, req: AdjudicationRequest) -> list[dict[str, Any]]:
    lines = [
        "You are re-checking a borderline document classification made by an "
        "automated clustering pipeline.",
        "",
        "Candidate document types this document might belong to:",
        *(f"  - {c}" for c in req.candidate_labels),
        "",
        f'The pipeline tentatively labeled it "{req.current_label}".',
        "",
        "Two documents share a type only if the same structured questions could "
        "be answered from each (same extraction schema) — use document type, not "
        "topic. Decide which candidate it truly belongs to. If it fits none of "
        f"them, choose `{_NOVEL}` (a genuinely new type).",
        "",
        f"Call `{_TOOL_ADJUDICATE}` with your choice, a confidence from 0.0 to "
        "1.0, and a one-line rationale.",
    ]
    content: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
    if getattr(record, "text", ""):
        content.append({"type": "text", "text": record.text[:_TEXT_SNIPPET_CHARS]})
    url = _image_url(record)
    if url is not None:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content


def _adjudicate_tool(choices: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _TOOL_ADJUDICATE,
            "description": (
                "Decide which candidate document type the document belongs to, "
                "or that it is a genuinely new type."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "choice": {
                        "type": "string",
                        "enum": choices,
                        "description": (
                            f"The candidate type it belongs to, or '{_NOVEL}' if none fit."
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "How sure you are of this choice, 0.0 to 1.0.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One sentence justifying the choice.",
                    },
                },
                "required": ["choice"],
                "additionalProperties": False,
            },
        },
    }


def _parse_adjudicate(response: Any) -> tuple[str | None, float | None, str | None]:
    """Pull ``(choice, confidence, rationale)`` from an ``adjudicate`` call."""
    try:
        call = response.choices[0].message.tool_calls[0]
        raw = call.function.arguments
    except (AttributeError, IndexError, TypeError):
        return None, None, None
    try:
        args = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    except (json.JSONDecodeError, TypeError):
        return None, None, None
    choice = args.get("choice")
    if not isinstance(choice, str) or not choice:
        return None, None, None
    conf = args.get("confidence")
    confidence = (
        float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) else None
    )
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))
    rationale = args.get("rationale") if isinstance(args.get("rationale"), str) else None
    return choice, confidence, rationale


def _repartition_prompt(existing: list[str]) -> str:
    lines = [
        "You are re-partitioning a small set of documents that an automated "
        "clustering pipeline was unsure about.",
        "",
        "Group them by document type (same type ⇒ the same structured questions "
        "could be answered from each). For each group, either set "
        "`existing_label` to one of the known types below if it matches, or give "
        "the group a short new `name` if it is a genuinely new type.",
    ]
    if existing:
        lines.append("")
        lines.append("Known types:")
        lines.extend(f"  - {c}" for c in existing)
    lines.append("")
    lines.append(f"Call `{_TOOL_REGROUP}` exactly once with all groups.")
    return "\n".join(lines)


def _regroup_tool(existing: list[str]) -> dict[str, Any]:
    group_props: dict[str, Any] = {
        "members": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
            "description": "Document labels (e.g. 'doc_1') in this group.",
        },
        "name": {
            "type": "string",
            "description": "For a NEW type: a short 2-5 word name. Omit if using existing_label.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "How sure you are of this grouping, 0.0 to 1.0.",
        },
    }
    if existing:
        group_props["existing_label"] = {
            "type": "string",
            "enum": existing,
            "description": "An existing type this group matches (instead of name).",
        }
    return {
        "type": "function",
        "function": {
            "name": _TOOL_REGROUP,
            "description": "Partition the attached documents into same-type groups.",
            "parameters": {
                "type": "object",
                "properties": {
                    "groups": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": group_props,
                            "required": ["members"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["groups"],
                "additionalProperties": False,
            },
        },
    }


def _extract_groups(response: Any) -> list[Any]:
    call = response.choices[0].message.tool_calls[0]
    raw = call.function.arguments
    args = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    groups = args.get("groups")
    return groups if isinstance(groups, list) else []


def _map_groups_to_verdicts(
    groups: list[Any], labels: dict[str, AdjudicationRequest]
) -> dict[str, AdjudicationVerdict]:
    verdicts: dict[str, AdjudicationVerdict] = {}
    for group in groups:
        if not isinstance(group, dict):
            continue
        members = group.get("members")
        if not isinstance(members, list):
            continue
        existing_label = group.get("existing_label")
        assignment = existing_label if isinstance(existing_label, str) and existing_label else None
        conf_raw = group.get("confidence")
        confidence = (
            max(0.0, min(1.0, float(conf_raw)))
            if isinstance(conf_raw, (int, float)) and not isinstance(conf_raw, bool)
            else None
        )
        for member in members:
            req = labels.get(str(member))
            if req is None:
                continue
            verdicts[req.doc_id] = AdjudicationVerdict(
                assignment=assignment, confidence=confidence, rationale=None
            )
    return verdicts
