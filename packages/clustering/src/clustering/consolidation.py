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

"""LLM adjudication of the low-confidence tail (§5).

The embedding pipeline does the bulk of the work cheaply and confidently; a
handful of documents sit on a cluster boundary or in a noise bucket where the
statistics are weak. This module reconsiders *only* those — selecting the
least-confident assignments, offering each its nearest candidate clusters, and
asking an LLM the one question it is good at: *does this document belong to one
of these, or is it genuinely new?* Cost scales with uncertainty, not corpus
size.

The heavy lifting stays here in the framework and is pure Python / torch:

- :func:`select_low_confidence_tail` — budgeted selection over the per-document
  confidence already on the :class:`~clustering.scenarios.base.ScenarioResult`.
- :func:`candidate_clusters` — the ``candidates_k`` nearest existing clusters to
  a document, by manifold distance to each cluster's centroid.
- :func:`consolidate` — assemble requests, call the injected adjudicator, and
  merge verdicts back through :meth:`Scenario.refine`.

The LLM call itself is **not** here: an :class:`Adjudicator` is injected by the
caller (dgml-core supplies a litellm-backed one), so this package keeps its
Apache-2.0, no-LLM-dependency contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch

from clustering.scenarios.base import ScenarioResult

if TYPE_CHECKING:
    from clustering.data.datasets import DocumentDataset
    from clustering.scenarios.base import Scenario


@dataclass(frozen=True)
class AdjudicationRequest:
    """One document handed to the adjudicator for a second opinion.

    ``candidate_labels`` are the nearest existing cluster/category names (most
    likely first); the adjudicator picks one of them, or returns ``None`` to
    flag the document as genuinely novel.
    """

    doc_id: str
    doc_index: int
    current_label: str | None
    candidate_labels: list[str]


@dataclass(frozen=True)
class AdjudicationVerdict:
    """The adjudicator's decision for one document.

    ``assignment`` is a cluster/category name to (re)assign to, or ``None`` for
    a *novel* verdict (open a new bucket). ``confidence`` is the adjudicator's
    ordinal confidence in ``[0, 1]`` (or ``None``); ``rationale`` is a short
    free-text justification for audit.
    """

    assignment: str | None
    confidence: float | None = None
    rationale: str | None = None


class Adjudicator(Protocol):
    """Callback that reconsiders a batch of low-confidence documents.

    Implemented in the caller's layer (dgml-core wraps the vision LLM) so the
    framework stays LLM-free. Must be soft-failing in spirit — but callers of
    :func:`consolidate` are also guarded, so a raised exception degrades to
    "no change" rather than aborting the run.
    """

    def __call__(
        self,
        dataset: DocumentDataset,
        requests: list[AdjudicationRequest],
        *,
        mode: str,
        batch_size: int,
    ) -> dict[str, AdjudicationVerdict]: ...


# Predicted-label conventions that mean "not in a real cluster" — HDBSCAN-style
# noise buckets and the unassigned sentinel.
_NOISE_SUFFIX = "_noise"


def _is_noise(label: str | None) -> bool:
    return label is None or label.endswith(_NOISE_SUFFIX)


def _conf_or_low(confidence: list[float | None], i: int) -> float:
    """Confidence of doc ``i``, treating missing / ``None`` as maximally uncertain."""
    if i >= len(confidence):
        return -1.0
    c = confidence[i]
    return -1.0 if c is None else float(c)


# Below this max-min confidence gap the signal is treated as flat: a
# bottom-quantile cut over near-tied values selects an essentially arbitrary
# set of documents, so the quantile strategy is suppressed (see
# :func:`select_low_confidence_tail`). Set at 1e-3 deliberately: an
# *uncalibrated* softmax peak over well-separated clusters saturates to ~1.0
# with only float-noise variation (measured ~1e-4 across a real 96-doc corpus)
# — technically non-zero but not a rankable signal. A healthy signal (e.g. the
# 'auto'-temperature S1 confidence, or nearest-prototype confidence in S2-S5)
# spreads one to two orders of magnitude wider, so this floor never suppresses
# a genuine ranking.
_MIN_CONFIDENCE_SPREAD = 1e-3


def _confidence_spread(confidence: list[float | None], n: int) -> float:
    """max-min of the per-document confidence (``None`` ⇒ ``-1``) over ``n`` docs.

    ``0.0`` when every document scored identically — the degenerate case a
    quantile cut cannot rank."""
    if n <= 0:
        return 0.0
    vals = [_conf_or_low(confidence, i) for i in range(n)]
    return max(vals) - min(vals)


def select_low_confidence_tail(result: ScenarioResult, cfg: Any) -> list[int]:
    """Indices of the documents to adjudicate, per the selector config (§5.2).

    ``cfg`` is a ``ConsolidationSelectorConfig``. The active ``strategy`` picks
    the tail; ``include_noise`` unions in every noise / unassigned document;
    the union is then sorted least-confident-first and capped at ``max_docs``
    so LLM cost is bounded regardless of how uncertain the run is.
    """
    n = len(result.doc_ids)
    conf = result.confidence
    selected: set[int] = set()

    strategy = cfg.strategy
    if strategy == "confidence" and cfg.confidence_threshold is not None:
        thr = float(cfg.confidence_threshold)
        selected.update(i for i in range(n) if _conf_or_low(conf, i) < thr)
    elif strategy == "margin" and cfg.margin_threshold is not None and result.scores is not None:
        selected.update(_margin_selected(result, float(cfg.margin_threshold)))
    else:
        # Default / fallback: bottom-quantile by confidence. (Also the landing
        # spot when 'margin' is requested but no per-class scores exist.)
        #
        # Guard the degenerate case: when the confidence signal has no spread
        # (every document ~tied — e.g. an uncalibrated softmax that saturated
        # at 1.0), a *partial* quantile cut selects an arbitrary set and lets
        # the adjudicator perturb confidently-correct assignments for no
        # reason. Suppress it — only genuine noise (via ``include_noise``)
        # still enters the tail. A full cut (``quantile >= 1``) selects every
        # document deterministically, so it is exempt: nothing arbitrary about
        # "adjudicate all".
        q = float(cfg.quantile)
        if q >= 1.0 or _confidence_spread(conf, n) >= _MIN_CONFIDENCE_SPREAD:
            selected.update(_quantile_selected(result, q))

    if cfg.include_noise:
        selected.update(i for i in range(n) if _is_noise(result.predictions[i]))

    ordered = sorted(selected, key=lambda i: _conf_or_low(conf, i))
    max_docs = int(cfg.max_docs)
    return ordered[:max_docs] if max_docs >= 0 else ordered


def _quantile_selected(result: ScenarioResult, quantile: float) -> list[int]:
    n = len(result.doc_ids)
    if n == 0 or quantile <= 0.0:
        return []
    q = min(1.0, quantile)
    k = max(1, round(n * q))
    ordered = sorted(range(n), key=lambda i: _conf_or_low(result.confidence, i))
    return ordered[:k]


def _margin_selected(result: ScenarioResult, margin_threshold: float) -> list[int]:
    scores = result.scores
    assert scores is not None  # guarded by caller
    if int(scores.shape[0]) == 0 or int(scores.shape[-1]) < 2:
        return []
    top2 = torch.topk(scores, k=2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).tolist()
    return [i for i, m in enumerate(margin) if float(m) < margin_threshold]


def candidate_clusters(
    scenario: Scenario,
    result: ScenarioResult,
    index: int,
    k: int,
) -> list[str]:
    """The ``k`` nearest existing cluster labels to document ``index``.

    Cluster centroids are the manifold-mean of each (non-noise) cluster's
    member embeddings; candidates are ranked by manifold distance from the
    document to those centroids. Returns fewer than ``k`` labels when the run
    has fewer clusters, and an empty list when there are none.
    """
    labels = result.predictions
    members: dict[str, list[int]] = {}
    for i, lbl in enumerate(labels):
        if not _is_noise(lbl) and lbl is not None:
            members.setdefault(lbl, []).append(i)
    if not members:
        return []

    names = list(members)
    centroids = torch.stack(
        [
            scenario.manifold.expmap0(
                result.embeddings[torch.tensor(members[name])].mean(dim=0).unsqueeze(0)
            ).squeeze(0)
            for name in names
        ],
        dim=0,
    )
    query = result.embeddings[index].unsqueeze(0)
    dist = scenario.manifold.pairwise_dist(query, centroids).squeeze(0)
    order = sorted(range(len(names)), key=lambda j: float(dist[j].item()))
    return [names[j] for j in order[: max(0, k)]]


def _next_novel_index(labels: list[str | None]) -> int:
    """One past the highest existing ``unknown_<n>`` index, so minted novel
    buckets don't collide with emergent clusters already in the result."""
    highest = -1
    for lbl in labels:
        if lbl and lbl.startswith("unknown_"):
            suffix = lbl[len("unknown_") :]
            if suffix.isdigit():
                highest = max(highest, int(suffix))
    return highest + 1


def consolidate(
    scenario: Scenario,
    result: ScenarioResult,
    unknown_dataset: DocumentDataset,
    adjudicator: Adjudicator,
) -> ScenarioResult:
    """Run the consolidation pass (see :meth:`Scenario.consolidate`).

    No-op (returns ``result`` unchanged) when consolidation is disabled or the
    tail is empty. Soft-fails: any adjudicator error is swallowed and the
    original result returned with a note in metadata, matching dgml-core's
    never-raise clustering philosophy (§5.6).
    """
    cfg = scenario.config.scenario.consolidation
    if not cfg.enabled:
        return result

    tail = select_low_confidence_tail(result, cfg.selector)
    if not tail:
        sel = cfg.selector
        note = "empty tail"
        if (
            sel.strategy in ("quantile", "margin")
            and float(sel.quantile) < 1.0
            and _confidence_spread(result.confidence, len(result.doc_ids)) < _MIN_CONFIDENCE_SPREAD
        ):
            note = (
                "no confidence spread — every document scored ~equally, so a "
                "bottom-quantile selection would be arbitrary; skipped (raise "
                "the confidence signal's resolution or use an absolute "
                "confidence_threshold to adjudicate anyway)"
            )
        return _with_meta(result, {"enabled": True, "n_selected": 0, "note": note})

    requests = [
        AdjudicationRequest(
            doc_id=result.doc_ids[i],
            doc_index=i,
            current_label=result.predictions[i],
            candidate_labels=candidate_clusters(scenario, result, i, cfg.candidates_k),
        )
        for i in tail
    ]

    try:
        verdicts = adjudicator(unknown_dataset, requests, mode=cfg.mode, batch_size=cfg.batch_size)
    except Exception as exc:
        return _with_meta(
            result,
            {
                "enabled": True,
                "n_selected": len(tail),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )

    return _apply_verdicts(scenario, result, unknown_dataset, requests, verdicts, cfg)


def _apply_verdicts(
    scenario: Scenario,
    result: ScenarioResult,
    unknown_dataset: DocumentDataset,
    requests: list[AdjudicationRequest],
    verdicts: dict[str, AdjudicationVerdict],
    cfg: Any,
) -> ScenarioResult:
    """Merge adjudicator verdicts into the result (or record them for review)."""
    n = len(result.doc_ids)
    idx_of = {req.doc_id: req.doc_index for req in requests}
    novel_counter = _next_novel_index(result.predictions)

    corrections: dict[str, str] = {}
    new_conf = list(result.confidence)
    review = list(result.review) if result.review else [False] * n
    if len(review) < n:
        review.extend([False] * (n - len(review)))
    records: list[dict[str, Any]] = []
    n_reassigned = 0
    n_novel = 0

    for doc_id, verdict in verdicts.items():
        i = idx_of.get(doc_id)
        if i is None:
            continue
        if verdict.assignment is None:
            new_label = f"unknown_{novel_counter}"
            novel_counter += 1
            n_novel += 1
        else:
            new_label = verdict.assignment
            n_reassigned += 1
        records.append(
            {
                "doc_id": doc_id,
                "from": result.predictions[i],
                "to": new_label,
                "confidence": verdict.confidence,
                "rationale": verdict.rationale,
            }
        )
        if cfg.apply == "auto":
            corrections[doc_id] = new_label
            if verdict.confidence is not None:
                new_conf[i] = verdict.confidence
            review[i] = False
        else:  # suggest — leave labels, flag for human review
            review[i] = True

    meta = {
        "enabled": True,
        "mode": cfg.mode,
        "apply": cfg.apply,
        "consolidated_by": "llm",
        "model": cfg.model,
        "n_selected": len(requests),
        "n_reassigned": n_reassigned,
        "n_novel": n_novel,
        "verdicts": records,
    }

    if cfg.apply == "auto" and corrections:
        # ``refine`` applies the {doc_id: label} corrections and copies
        # confidence/review verbatim; overlay our confidence + review updates.
        refined = scenario.refine(result, corrections, unknown_dataset)
        refined.confidence[:] = new_conf
        refined.review[:] = review
        return _with_meta(refined, meta)

    # suggest mode: labels unchanged, only the review queue + provenance move.
    return _replace(result, confidence=new_conf, review=review, meta=meta)


def _with_meta(result: ScenarioResult, consolidation_meta: dict[str, Any]) -> ScenarioResult:
    return ScenarioResult(
        run_id=result.run_id,
        scenario_name=result.scenario_name,
        doc_ids=result.doc_ids,
        embeddings=result.embeddings,
        predictions=list(result.predictions),
        confidence=list(result.confidence),
        true_labels=list(result.true_labels),
        scores=result.scores,
        class_names=list(result.class_names) if result.class_names else None,
        metadata={**result.metadata, "consolidation": consolidation_meta},
        review=list(result.review),
    )


def _replace(
    result: ScenarioResult,
    *,
    confidence: list[float | None],
    review: list[bool],
    meta: dict[str, Any],
) -> ScenarioResult:
    return ScenarioResult(
        run_id=result.run_id,
        scenario_name=result.scenario_name,
        doc_ids=result.doc_ids,
        embeddings=result.embeddings,
        predictions=list(result.predictions),
        confidence=confidence,
        true_labels=list(result.true_labels),
        scores=result.scores,
        class_names=list(result.class_names) if result.class_names else None,
        metadata={**result.metadata, "consolidation": meta},
        review=review,
    )
