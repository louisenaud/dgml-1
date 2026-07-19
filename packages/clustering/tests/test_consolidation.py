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

"""Tests for the consolidation pass (§5): selector, candidates, apply.

The LLM adjudicator is faked (a plain callable matching the ``Adjudicator``
protocol), so these exercise the framework's pure selection / candidate /
merge logic without any provider dependency.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from clustering.config.schema import Config, ConsolidationSelectorConfig, ManifoldConfig
from clustering.consolidation import (
    AdjudicationRequest,
    AdjudicationVerdict,
    candidate_clusters,
    select_low_confidence_tail,
)
from clustering.data.datasets import DocumentDataset, DocumentRecord
from clustering.manifolds import build_manifold
from clustering.scenarios import build_scenario
from clustering.scenarios.base import ScenarioResult
from PIL import Image

_DIM = 8


def _result(preds: list[str | None], conf: list[float | None], emb: torch.Tensor) -> ScenarioResult:
    return ScenarioResult(
        run_id="r",
        scenario_name="s1",
        doc_ids=[f"d{i}" for i in range(len(preds))],
        embeddings=emb,
        predictions=preds,
        confidence=conf,
        true_labels=[None] * len(preds),
    )


# ── selector ────────────────────────────────────────────────────────────────
def test_quantile_selects_least_confident_plus_noise() -> None:
    res = _result(
        ["cluster_0", "cluster_0", "cluster_1", "cluster_noise"],
        [0.9, 0.2, 0.8, 0.0],
        torch.randn(4, _DIM),
    )
    cfg = ConsolidationSelectorConfig(strategy="quantile", quantile=0.5, include_noise=True)
    selected = set(select_low_confidence_tail(res, cfg))
    assert 1 in selected  # 0.2 — lowest real cluster
    assert 3 in selected  # noise bucket
    assert 0 not in selected  # 0.9 — confident, kept


def test_confidence_threshold_strategy() -> None:
    res = _result(["a", "b", "c"], [0.95, 0.4, 0.1], torch.randn(3, _DIM))
    cfg = ConsolidationSelectorConfig(
        strategy="confidence", confidence_threshold=0.5, include_noise=False
    )
    assert set(select_low_confidence_tail(res, cfg)) == {1, 2}


def test_max_docs_caps_the_tail() -> None:
    res = _result(["a"] * 10, [i / 10 for i in range(10)], torch.randn(10, _DIM))
    cfg = ConsolidationSelectorConfig(strategy="quantile", quantile=1.0, max_docs=3)
    selected = select_low_confidence_tail(res, cfg)
    assert len(selected) == 3
    # Cap keeps the *least* confident first.
    assert selected == [0, 1, 2]


def test_none_confidence_is_maximally_uncertain() -> None:
    res = _result(["a", "b", "c"], [None, 0.9, 0.8], torch.randn(3, _DIM))
    cfg = ConsolidationSelectorConfig(strategy="quantile", quantile=0.34, include_noise=False)
    assert select_low_confidence_tail(res, cfg) == [0]


# ── degenerate-tail guard: a flat confidence signal must not be sliced ────────
def test_flat_confidence_suppresses_partial_quantile() -> None:
    # Every document tied at 1.0 (the saturated-softmax case). A partial
    # bottom-quantile cut would pick an arbitrary subset, so it is suppressed.
    res = _result(["a", "b", "c", "d"], [1.0, 1.0, 1.0, 1.0], torch.randn(4, _DIM))
    cfg = ConsolidationSelectorConfig(strategy="quantile", quantile=0.5, include_noise=False)
    assert select_low_confidence_tail(res, cfg) == []


def test_flat_real_confidence_still_adjudicates_noise() -> None:
    # All real clusters tie at 1.0 but a noise bucket sits at 0.0. The noise
    # doc gives the column genuine spread, so the guard does not fire here —
    # and the noise document must always be adjudicated via include_noise.
    res = _result(["a", "b", "cluster_noise"], [1.0, 1.0, 0.0], torch.randn(3, _DIM))
    cfg = ConsolidationSelectorConfig(strategy="quantile", quantile=0.5, include_noise=True)
    assert 2 in select_low_confidence_tail(res, cfg)


def test_fully_flat_with_noise_flag_selects_only_noise() -> None:
    # When *every* document (noise included) ties, the quantile cut is
    # suppressed and only the noise-flagged documents are adjudicated.
    res = _result(["a", "b", "cluster_noise"], [1.0, 1.0, 1.0], torch.randn(3, _DIM))
    cfg = ConsolidationSelectorConfig(strategy="quantile", quantile=0.5, include_noise=True)
    # index 2 is a *_noise label ⇒ swept in by include_noise; the tied reals are not.
    assert set(select_low_confidence_tail(res, cfg)) == {2}


def test_near_tied_saturated_confidence_is_suppressed() -> None:
    # The real demo failure mode: an uncalibrated softmax that saturated to
    # ~1.0 with only float-noise variation (~1e-4). That is not a rankable
    # signal, so the partial quantile cut must still be suppressed.
    res = _result(
        ["a", "b", "c", "d"],
        [1.0000, 1.0001, 0.9999, 1.0002],
        torch.randn(4, _DIM),
    )
    cfg = ConsolidationSelectorConfig(strategy="quantile", quantile=0.25, include_noise=False)
    assert select_low_confidence_tail(res, cfg) == []


def test_flat_confidence_full_quantile_is_exempt() -> None:
    # quantile >= 1.0 means "adjudicate everything" — deterministic, not
    # arbitrary — so the guard does not apply even with a flat signal.
    res = _result(["a", "b", "c"], [1.0, 1.0, 1.0], torch.randn(3, _DIM))
    cfg = ConsolidationSelectorConfig(strategy="quantile", quantile=1.0, include_noise=False)
    assert set(select_low_confidence_tail(res, cfg)) == {0, 1, 2}


def test_flat_confidence_absolute_threshold_still_works() -> None:
    # The 'confidence' strategy is an absolute cutoff, not a ranking, so it is
    # unaffected by the spread guard: a threshold above the tied value selects
    # everything; below it selects nothing.
    res = _result(["a", "b", "c"], [1.0, 1.0, 1.0], torch.randn(3, _DIM))
    hit = ConsolidationSelectorConfig(
        strategy="confidence", confidence_threshold=1.5, include_noise=False
    )
    miss = ConsolidationSelectorConfig(
        strategy="confidence", confidence_threshold=0.5, include_noise=False
    )
    assert set(select_low_confidence_tail(res, hit)) == {0, 1, 2}
    assert select_low_confidence_tail(res, miss) == []


# ── candidate assembly ────────────────────────────────────────────────────────
def test_candidate_clusters_orders_by_distance() -> None:
    manifold = build_manifold(ManifoldConfig(name="euclidean", dim=2, curvature=0.0))
    scenario = SimpleNamespace(manifold=manifold)
    emb = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0], [0.2, 0.0]], dtype=torch.float32
    )
    # Two clusters: 'near' at the origin, 'far' around x=10; doc 4 is near the origin.
    res = _result(["near", "near", "far", "far", "near"], [0.9] * 5, emb)
    cands = candidate_clusters(scenario, res, index=4, k=2)  # type: ignore[arg-type]
    assert cands == ["near", "far"]


def test_candidate_clusters_ignores_noise() -> None:
    manifold = build_manifold(ManifoldConfig(name="euclidean", dim=2, curvature=0.0))
    scenario = SimpleNamespace(manifold=manifold)
    emb = torch.tensor([[0.0, 0.0], [5.0, 0.0], [2.5, 0.0]], dtype=torch.float32)
    res = _result(["cluster_noise", "real", None], [0.0, 0.9, None], emb)
    cands = candidate_clusters(scenario, res, index=2, k=3)  # type: ignore[arg-type]
    assert cands == ["real"]  # noise + None excluded


# ── end-to-end consolidate via a fake adjudicator ─────────────────────────────
class _MemDataset(DocumentDataset):
    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> DocumentRecord:
        return DocumentRecord(
            doc_id=f"doc_{index}",
            label=None,
            image=Image.new("RGB", (8, 8), color=(index * 9 % 255, 0, 0)),
            text=f"document {index}",
            thumbnail_path=None,
        )


def _s1_config(apply: str, **selector: Any) -> Config:
    raw: dict[str, Any] = {
        "scenario": {
            "name": "s1",
            "k_clusters": 2,
            "cluster_algorithm": "kmeans",
            "consolidation": {
                "enabled": True,
                "apply": apply,
                "candidates_k": 2,
                "selector": {"strategy": "quantile", "quantile": 1.0, **selector},
            },
        },
        "encoder_text": {"name": "dummy", "model_id": "dummy", "embedding_dim": _DIM},
        "encoder_image": {"name": "dummy", "model_id": "dummy", "embedding_dim": _DIM},
        "fusion": {"name": "late_concat", "output_dim": 2 * _DIM},
        "manifold": {"name": "euclidean", "dim": 2 * _DIM},
        "training": {"epochs": 0},
        "logger": {"name": "none"},
        "corpus": {"root": "."},
        "device": "cpu",
        "seed": 0,
    }
    return Config.model_validate(raw)


def _novel_adjudicator(
    dataset: DocumentDataset,
    requests: list[AdjudicationRequest],
    *,
    mode: str,
    batch_size: int,
) -> dict[str, AdjudicationVerdict]:
    """Verdict: reassign to the first candidate if any, else declare novel."""
    out: dict[str, AdjudicationVerdict] = {}
    for req in requests:
        if req.candidate_labels:
            out[req.doc_id] = AdjudicationVerdict(
                assignment=req.candidate_labels[0], confidence=0.77, rationale="closest"
            )
        else:
            out[req.doc_id] = AdjudicationVerdict(assignment=None, confidence=0.4, rationale="new")
    return out


def test_disabled_consolidation_is_noop() -> None:
    cfg = _s1_config("auto")
    # Turn it off via a fresh config with enabled False.
    raw = cfg.model_dump()
    raw["scenario"]["consolidation"]["enabled"] = False
    scenario = build_scenario(Config.model_validate(raw))
    ds = _MemDataset(6)
    result = scenario.fit_predict(ds)
    same = scenario.consolidate(result, ds, _novel_adjudicator)
    assert same.predictions == result.predictions
    assert "consolidation" not in same.metadata


def test_consolidate_auto_applies_reassignments() -> None:
    scenario = build_scenario(_s1_config("auto"))
    ds = _MemDataset(6)
    result = scenario.fit_predict(ds)
    consolidated = scenario.consolidate(result, ds, _novel_adjudicator)

    meta = consolidated.metadata["consolidation"]
    assert meta["enabled"] is True
    assert meta["consolidated_by"] == "llm"
    assert meta["n_selected"] >= 1
    # auto mode writes the verdict confidence onto reassigned docs.
    assert len(consolidated.predictions) == len(result.predictions)
    assert 0.77 in [c for c in consolidated.confidence if c is not None]


def test_consolidate_suggest_leaves_labels_but_flags_review() -> None:
    scenario = build_scenario(_s1_config("suggest"))
    ds = _MemDataset(6)
    result = scenario.fit_predict(ds)
    consolidated = scenario.consolidate(result, ds, _novel_adjudicator)

    # suggest mode: labels unchanged, but selected docs are flagged for review.
    assert consolidated.predictions == result.predictions
    assert any(consolidated.review)
    assert consolidated.metadata["consolidation"]["apply"] == "suggest"


def test_consolidate_noops_on_flat_confidence_with_note() -> None:
    # End-to-end: a flat confidence column + a partial quantile selector must
    # leave the partition untouched and explain why in metadata — this is the
    # guard that stops consolidation from *degrading* an already-good run.
    scenario = build_scenario(_s1_config("auto", quantile=0.2, include_noise=False))
    ds = _MemDataset(6)
    result = scenario.fit_predict(ds)
    flat = ScenarioResult(
        run_id=result.run_id,
        scenario_name=result.scenario_name,
        doc_ids=result.doc_ids,
        embeddings=result.embeddings,
        predictions=result.predictions,
        confidence=[1.0] * len(result.doc_ids),
        true_labels=result.true_labels,
    )
    consolidated = scenario.consolidate(flat, ds, _novel_adjudicator)

    assert consolidated.predictions == flat.predictions
    meta = consolidated.metadata["consolidation"]
    assert meta["n_selected"] == 0
    assert "no confidence spread" in meta["note"]


def test_consolidate_soft_fails_on_adjudicator_error() -> None:
    scenario = build_scenario(_s1_config("auto"))
    ds = _MemDataset(6)
    result = scenario.fit_predict(ds)

    def _boom(dataset: Any, requests: Any, *, mode: str, batch_size: int) -> Any:
        raise RuntimeError("provider down")

    consolidated = scenario.consolidate(result, ds, _boom)
    # No raise; labels intact; the error is recorded in metadata.
    assert consolidated.predictions == result.predictions
    assert "error" in consolidated.metadata["consolidation"]
