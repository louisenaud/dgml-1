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

"""Tests for the dgml-core consolidation layer (§4.4 / §5):

- two-attempt agreement confidence in ``propose_new_docset_for_files``
- the litellm-backed :class:`LLMAdjudicator` (reassign + repartition)

``litellm.completion`` is patched with hand-built OpenAI-shaped stubs, exactly
as ``test_classification`` / ``test_llm_clustering`` do.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from clustering.consolidation import AdjudicationRequest
from clustering.data.datasets import DocumentDataset, DocumentRecord
from dgml_core.classification import ClassificationConfig, propose_new_docset_for_files
from dgml_core.consolidation import LLMAdjudicator
from dgml_core.models import FileRecord
from dgml_core.storage import Workspace, write_json_atomic
from PIL import Image

from .conftest import make_fake_png

DEFAULT_TEST_MODEL = "gemini/gemini-3.1-flash-lite"


# ---------------------------------------------------------------------------
# Response stubs
# ---------------------------------------------------------------------------
def _tool_response(name: str, args: dict[str, Any]) -> SimpleNamespace:
    call = SimpleNamespace(function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(tool_calls=[call], content=None),
                finish_reason="tool_calls",
            )
        ]
    )


def _create_response(name: str) -> SimpleNamespace:
    return _tool_response(
        "create_new_docset",
        {"name": name, "description": f"{name} docs", "key_questions": ["q1?", "q2?", "q3?"]},
    )


def _adjudicate_response(choice: str, confidence: float = 0.8) -> SimpleNamespace:
    return _tool_response(
        "adjudicate", {"choice": choice, "confidence": confidence, "rationale": "because"}
    )


def _regroup_response(groups: list[dict[str, Any]]) -> SimpleNamespace:
    return _tool_response("regroup_documents", {"groups": groups})


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _seed(workspace: Workspace, file_id: str) -> None:
    record = FileRecord(
        id=file_id,
        original_path=f"/fake/{file_id}.pdf",
        original_filename=f"{file_id}.pdf",
        sha256="0" * 64,
        added_at="2026-01-01T00:00:00Z",
        page_count=1,
        text_mode="digital",
    )
    workspace.file_dir(file_id).mkdir(parents=True, exist_ok=True)
    write_json_atomic(workspace.file_json_path(file_id), record.to_json())
    pages = workspace.file_pages_dir(file_id)
    pages.mkdir(parents=True, exist_ok=True)
    (pages / "page_1.png").write_bytes(make_fake_png(8, 8))


class _MemDataset(DocumentDataset):
    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> DocumentRecord:
        return DocumentRecord(
            doc_id=f"doc_{index}",
            label=None,
            image=Image.new("RGB", (8, 8), color=(index * 20 % 255, 0, 0)),
            text=f"document {index}",
            thumbnail_path=None,
        )


def _config() -> ClassificationConfig:
    return ClassificationConfig(model=DEFAULT_TEST_MODEL)


# ---------------------------------------------------------------------------
# propose_new_docset_for_files — two-attempt agreement
# ---------------------------------------------------------------------------
def test_single_attempt_has_no_confidence(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path / "ws")
    ws.init()
    _seed(ws, "a")
    with patch("litellm.completion", return_value=_create_response("Invoice")):
        decision = propose_new_docset_for_files(ws, ["a"], config=_config())
    assert decision.new_name == "Invoice"
    assert decision.confidence is None  # single attempt ⇒ no agreement signal


def test_two_attempts_agree_confidence_one(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path / "ws")
    ws.init()
    _seed(ws, "a")
    with patch("litellm.completion", return_value=_create_response("Invoice")):
        decision = propose_new_docset_for_files(ws, ["a"], config=_config(), attempts=2)
    assert decision.new_name == "Invoice"
    assert decision.confidence == 1.0  # both attempts proposed the same name


def test_two_attempts_disagree_confidence_half(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path / "ws")
    ws.init()
    _seed(ws, "a")
    with patch(
        "litellm.completion",
        side_effect=[_create_response("Invoice"), _create_response("Receipt")],
    ):
        decision = propose_new_docset_for_files(ws, ["a"], config=_config(), attempts=2)
    # Case-insensitive modal share of a 2-way split is 1/2.
    assert decision.confidence == 0.5


def test_agreement_is_case_insensitive(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path / "ws")
    ws.init()
    _seed(ws, "a")
    with patch(
        "litellm.completion",
        side_effect=[_create_response("PILOT Agreement"), _create_response("pilot   agreement")],
    ):
        decision = propose_new_docset_for_files(ws, ["a"], config=_config(), attempts=2)
    assert decision.confidence == 1.0


# ---------------------------------------------------------------------------
# LLMAdjudicator — reassign
# ---------------------------------------------------------------------------
def test_adjudicator_reassigns_to_candidate() -> None:
    ds = _MemDataset(2)
    adj = LLMAdjudicator(_config(), attempts=2)
    requests = [
        AdjudicationRequest(
            doc_id="doc_0", doc_index=0, current_label="B", candidate_labels=["A", "B"]
        )
    ]
    with patch("litellm.completion", return_value=_adjudicate_response("A", 0.9)):
        verdicts = adj(ds, requests, mode="reassign", batch_size=40)
    assert set(verdicts) == {"doc_0"}
    v = verdicts["doc_0"]
    assert v.assignment == "A"
    # Both attempts agree (rotation only reorders candidates) ⇒ agreement 1.0,
    # confidence = mean self-report (0.9) x 1.0.
    assert v.confidence is not None and abs(v.confidence - 0.9) < 1e-6
    assert v.rationale == "because"


def test_adjudicator_novel_verdict() -> None:
    ds = _MemDataset(1)
    adj = LLMAdjudicator(_config(), attempts=1)
    requests = [
        AdjudicationRequest(doc_id="doc_0", doc_index=0, current_label="A", candidate_labels=["A"])
    ]
    with patch("litellm.completion", return_value=_adjudicate_response("__novel__", 0.6)):
        verdicts = adj(ds, requests, mode="reassign", batch_size=40)
    assert verdicts["doc_0"].assignment is None  # novel


def test_adjudicator_soft_fails_per_document() -> None:
    ds = _MemDataset(1)
    adj = LLMAdjudicator(_config(), attempts=1)
    requests = [
        AdjudicationRequest(doc_id="doc_0", doc_index=0, current_label="A", candidate_labels=["A"])
    ]
    with patch("litellm.completion", side_effect=RuntimeError("down")):
        verdicts = adj(ds, requests, mode="reassign", batch_size=40)
    assert verdicts == {}  # failed doc simply drops out; no raise


def test_adjudicator_empty_requests() -> None:
    adj = LLMAdjudicator(_config())
    assert adj(_MemDataset(0), [], mode="reassign", batch_size=40) == {}


# ---------------------------------------------------------------------------
# LLMAdjudicator — repartition
# ---------------------------------------------------------------------------
def test_adjudicator_repartition_maps_groups() -> None:
    ds = _MemDataset(3)
    adj = LLMAdjudicator(_config())
    requests = [
        AdjudicationRequest(
            doc_id=f"doc_{i}", doc_index=i, current_label="x", candidate_labels=["A"]
        )
        for i in range(3)
    ]
    groups = [
        {"members": ["doc_1", "doc_2"], "existing_label": "A", "confidence": 0.9},
        {"members": ["doc_3"], "name": "NewType", "confidence": 0.5},
    ]
    with patch("litellm.completion", return_value=_regroup_response(groups)):
        verdicts = adj(ds, requests, mode="repartition", batch_size=40)
    assert verdicts["doc_0"].assignment == "A"  # doc_1 label ⇒ first request
    assert verdicts["doc_1"].assignment == "A"
    assert verdicts["doc_2"].assignment is None  # NewType ⇒ novel
