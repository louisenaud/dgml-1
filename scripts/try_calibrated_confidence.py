#!/usr/bin/env python3
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

"""Try the calibrated-confidence + abstain + consolidation improvements (§4.4/§5).

A self-contained, offline demo (no PDFs, no API key, no model downloads) of the
three things just added:

1. **Calibration** — temperature / Platt scaling + a distribution-free conformal
   abstain gate, shown on controlled logits so the numbers are interpretable:
   raw softmax peak vs calibrated confidence, and which rows the conformal gate
   routes to review at a target coverage.
2. **Abstain / review queue, end-to-end** — a real S3 (few-shot) scenario run
   over a small in-memory labeled corpus, with calibration fit on the support
   set by leave-one-out. Prints each document's calibrated confidence and its
   ``review`` flag, plus the calibration provenance the run recorded.
3. **Consolidation** — the LLM adjudication pass over the low-confidence tail.
   By default a built-in *offline* adjudicator stands in for the vision model so
   the demo always runs; pass ``--use-llm --model ...`` (with the matching API
   key) to exercise the real :class:`LLMAdjudicator`.

Because the framework's dummy encoder is a content hash (deterministic but not
semantically separable), the *confidence values* in sections 2-3 are not
meaningful classification quality — they demonstrate the machinery and the
output contract, not model accuracy. Section 1 uses controlled logits, so its
numbers are real.

It is intentionally NOT part of the public ``dgml`` CLI — a demo / debug tool.

Usage:
    uv run python scripts/try_calibrated_confidence.py
    uv run python scripts/try_calibrated_confidence.py --section calibration
    uv run python scripts/try_calibrated_confidence.py --calibration platt --coverage 0.8
    uv run python scripts/try_calibrated_confidence.py --consolidate-apply auto
    uv run python scripts/try_calibrated_confidence.py --use-llm   # needs --model + API key
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Demo calibrated confidence + abstain + consolidation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--section",
        choices=["all", "calibration", "scenario"],
        default="all",
        help="Which demo to run.",
    )
    p.add_argument("--classes", type=int, default=3, help="Number of synthetic categories.")
    p.add_argument("--support-per-class", type=int, default=4, help="Labeled support docs/class.")
    p.add_argument("--query-per-class", type=int, default=3, help="Query docs/class.")
    p.add_argument(
        "--calibration",
        choices=["none", "temperature", "platt"],
        default="temperature",
        help="Parametric calibration method.",
    )
    p.add_argument(
        "--coverage",
        type=float,
        default=0.9,
        help="Conformal target coverage in (0,1); <=0 disables the conformal gate.",
    )
    p.add_argument(
        "--abstain-threshold",
        type=float,
        default=None,
        help="Absolute calibrated-confidence floor; below it a doc abstains.",
    )
    p.add_argument(
        "--consolidate-apply",
        choices=["suggest", "auto"],
        default="suggest",
        help="Consolidation: emit verdicts for review, or auto-write them.",
    )
    p.add_argument(
        "--consolidate-quantile",
        type=float,
        default=0.34,
        help="Bottom fraction of confidences sent to LLM adjudication.",
    )
    p.add_argument(
        "--use-llm",
        action="store_true",
        help="Use the real LLMAdjudicator (needs --model + API key) instead of the "
        "built-in offline adjudicator.",
    )
    p.add_argument("--model", default="gemini/gemini-3.1-flash-lite", help="Adjudication model.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true", help="Also dump the raw result dicts as JSON.")
    return p.parse_args(argv)


# ── Section 1: calibration on controlled logits ─────────────────────────────
def demo_calibration(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from clustering.calibration import fit_calibrator

    k = max(2, args.classes)
    # Labeled calibration logits: each class sits a clear margin above the rest,
    # with noise — so temperature has something real to fit and most rows are
    # correct (mirrors a well-separated support set).
    g = torch.Generator().manual_seed(args.seed)
    rows, labels = [], []
    for c in range(k):
        base = torch.zeros(60, k)
        base[:, c] = 6.0
        rows.append(base + 1.5 * torch.randn(60, k, generator=g))
        labels += [c] * 60
    logits = torch.cat(rows, dim=0)
    labels_t = torch.tensor(labels)

    coverage = args.coverage if args.coverage and args.coverage > 0 else None
    cal = fit_calibrator(
        logits,
        labels_t,
        method=args.calibration,
        coverage=coverage,
        abstain_threshold=args.abstain_threshold,
    )

    print("── 1. Calibration (controlled logits) " + "─" * 30)
    print(f"  method            : {cal.method}")
    print(f"  temperature       : {cal.temperature:.4f}")
    if cal.method == "platt":
        print(f"  platt (a, b)      : ({cal.platt_a:.4f}, {cal.platt_b:.4f})")
    print(f"  conformal coverage: {cal.coverage}")
    print(f"  conformal q̂       : {cal.conformal_threshold}")
    print(f"  fit on            : {cal.n_calibration} labeled rows")

    # Probe rows spanning confident → ambiguous, and show raw vs calibrated.
    probes = torch.tensor(
        [[8.0] + [0.0] * (k - 1), [3.0] + [0.0] * (k - 1), [0.4, 0.2] + [0.0] * (k - 2)]
    )
    raw_peak = torch.softmax(probes, dim=-1).amax(dim=-1)
    cal_conf, abstain = cal.apply(probes)
    print(f"\n  {'probe':<22} {'raw':>6} {'calibrated':>11} {'review?':>8}")
    print(f"  {'-' * 22} {'-' * 6} {'-' * 11} {'-' * 8}")
    names = ["clear winner", "mild winner", "near tie"]
    for name, r, c, a in zip(
        names, raw_peak.tolist(), cal_conf.tolist(), abstain.tolist(), strict=True
    ):
        print(f"  {name:<22} {r:>6.2f} {c:>11.2f} {('REVIEW' if a else '-'):>8}")
    return {"calibration": cal.as_dict()}


# ── Section 2+3: real S3 scenario run + consolidation ───────────────────────
def _build_dataset_classes(args: argparse.Namespace) -> Any:
    from clustering.data.datasets import DocumentDataset, DocumentRecord
    from PIL import Image

    class _Corpus(DocumentDataset):
        def __init__(self, specs: list[tuple[str, str | None]]) -> None:
            # specs: list of (text, label)
            self._specs = specs

        def __len__(self) -> int:
            return len(self._specs)

        def __getitem__(self, index: int) -> DocumentRecord:
            text, label = self._specs[index]
            return DocumentRecord(
                doc_id=f"doc_{index}",
                label=label,
                image=Image.new("RGB", (8, 8), color=(index * 17 % 255, 40, 90)),
                text=text,
                thumbnail_path=None,
            )

    return _Corpus


def _s3_config(args: argparse.Namespace, categories: list[str]) -> Any:
    from clustering.config.schema import Config

    dim = 16
    coverage = args.coverage if args.coverage and args.coverage > 0 else None
    raw: dict[str, Any] = {
        "scenario": {
            "name": "s3",
            "known_categories": categories,
            "n_shots": args.support_per_class,
            "cluster_algorithm": "kmeans",
            "calibration": {
                "method": args.calibration,
                "coverage": coverage,
                "abstain_threshold": args.abstain_threshold,
            },
            "consolidation": {
                "enabled": True,
                "apply": args.consolidate_apply,
                "candidates_k": min(3, len(categories)),
                "mode": "reassign",
                "model": args.model,
                "selector": {
                    "strategy": "quantile",
                    "quantile": args.consolidate_quantile,
                    "include_noise": True,
                },
            },
        },
        "encoder_text": {"name": "dummy", "model_id": "dummy", "embedding_dim": dim},
        "encoder_image": {"name": "dummy", "model_id": "dummy", "embedding_dim": dim},
        "fusion": {"name": "late_concat", "output_dim": 2 * dim},
        "manifold": {"name": "euclidean", "dim": 2 * dim},
        "training": {"epochs": 0},
        "logger": {"name": "none"},
        "corpus": {"root": "."},
        "device": "cpu",
        "seed": args.seed,
    }
    return Config.model_validate(raw)


def _offline_adjudicator(dataset: Any, requests: list[Any], *, mode: str, batch_size: int) -> Any:
    """Stand-in for the vision LLM: reassign to the nearest candidate (the one
    the embedding structure already suggested), else declare novel. Lets the
    consolidation pass run end-to-end with no network."""
    from clustering.consolidation import AdjudicationVerdict

    out: dict[str, AdjudicationVerdict] = {}
    for req in requests:
        if req.candidate_labels:
            out[req.doc_id] = AdjudicationVerdict(
                assignment=req.candidate_labels[0], confidence=0.66, rationale="nearest candidate"
            )
        else:
            out[req.doc_id] = AdjudicationVerdict(
                assignment=None, confidence=0.33, rationale="no candidate fits"
            )
    return out


def demo_scenario(args: argparse.Namespace) -> dict[str, Any]:
    from clustering.scenarios import build_scenario

    categories = [f"type_{chr(ord('A') + i)}" for i in range(max(2, args.classes))]
    corpus_cls = _build_dataset_classes(args)

    support_specs = [
        (f"{cat} support example {i}", cat)
        for cat in categories
        for i in range(args.support_per_class)
    ]
    query_specs: list[tuple[str, str | None]] = [
        (f"{cat} query document {i}", None)
        for cat in categories
        for i in range(args.query_per_class)
    ]
    support = corpus_cls(support_specs)
    query = corpus_cls(query_specs)

    scenario = build_scenario(_s3_config(args, categories))
    result = scenario.fit_predict(query, support)

    print("\n── 2. Abstain / review queue (real S3 run) " + "─" * 25)
    print("  (dummy encoder ⇒ synthetic embeddings; confidences show the")
    print("   mechanism + output contract, not classification quality)")
    cal_meta = result.metadata.get("calibration")
    print(f"  calibration       : {cal_meta}")
    review = result.review if result.review else [False] * len(result.doc_ids)
    print(f"\n  {'doc':<8} {'predicted':<10} {'confidence':>10} {'review?':>8}")
    print(f"  {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 8}")
    for doc_id, pred, conf, rev in zip(
        result.doc_ids, result.predictions, result.confidence, review, strict=True
    ):
        c = "n/a" if conf is None else f"{conf:.2f}"
        print(f"  {doc_id:<8} {pred!s:<10} {c:>10} {('REVIEW' if rev else '-'):>8}")
    n_review = int(sum(bool(r) for r in review))
    print(f"\n  review queue: {n_review} / {len(result.doc_ids)} document(s) flagged")

    # ── Consolidation over the low-confidence tail ──────────────────────
    print("\n── 3. Consolidation of the low-confidence tail " + "─" * 21)
    if args.use_llm:
        from dgml_core.classification import ClassificationConfig
        from dgml_core.consolidation import LLMAdjudicator

        adjudicator: Any = LLMAdjudicator(ClassificationConfig(model=args.model), attempts=2)
        print(f"  adjudicator: real LLMAdjudicator (model={args.model})")
    else:
        adjudicator = _offline_adjudicator
        print("  adjudicator: built-in offline stand-in (pass --use-llm for the real model)")

    consolidated = scenario.consolidate(result, query, adjudicator)
    meta = consolidated.metadata.get("consolidation", {})
    print(f"  apply mode        : {meta.get('apply')}")
    print(f"  selected (tail)   : {meta.get('n_selected')}")
    print(f"  reassigned        : {meta.get('n_reassigned')}")
    print(f"  novel verdicts    : {meta.get('n_novel')}")
    if meta.get("error"):
        print(f"  (soft-failed: {meta['error']})")
    for v in meta.get("verdicts", [])[:12]:
        conf = "n/a" if v["confidence"] is None else f"{v['confidence']:.2f}"
        print(f"    {v['doc_id']:<8} {v['from']!s:<10} → {v['to']!s:<12} conf={conf}")

    return {
        "calibration": cal_meta,
        "review_count": n_review,
        "consolidation": meta,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        import clustering  # noqa: F401
    except ImportError as exc:  # pragma: no cover — clustering extra missing
        raise SystemExit(
            f"error: could not import the clustering framework ({exc}). "
            "Install the clustering extra: `uv sync`."
        ) from exc

    out: dict[str, Any] = {}
    if args.section in ("all", "calibration"):
        out["calibration_demo"] = demo_calibration(args)
    if args.section in ("all", "scenario"):
        out["scenario_demo"] = demo_scenario(args)

    if args.json:
        print("\n--- raw JSON ---")
        print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
