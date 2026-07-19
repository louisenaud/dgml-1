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

"""Clustering metrics for a ``dgml cluster`` run — before vs after consolidation.

Reads the JSON envelopes emitted by ``dgml cluster`` (the ``assignments`` map:
``{file_id: {docset, confidence, review, is_new}}``) and reports, per side:

* **Descriptive** — documents, #DocSets, singletons, largest cluster, review
  queue, mean/median/min assignment confidence.
* **External** (vs. ground truth, when a label map is available) — Adjusted
  Rand Index, Normalized / Adjusted Mutual Information, homogeneity,
  completeness, V-measure, and cluster purity. These are the standard measures
  of *how well the discovered clusters recover the known categories*.

Ground truth is taken from ``--labels`` (a ``{relative/path: label}`` JSON map,
matched by filename) or, failing that, from the parent folder of each file's
original path — the corpus is organized one folder per class.

Internal geometric metrics (silhouette, Davies-Bouldin) are intentionally
omitted: they need the document embeddings, which the CLI ``cluster`` output
does not carry.

Usage:
    uv run python scripts/clustering_metrics.py \
        --before cluster_before.json --after cluster_after.json \
        --files files.json [--labels /path/to/labels.json]

    # single run (no comparison):
    uv run python scripts/clustering_metrics.py --before cluster.json --files files.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def _load(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build_truth(files: list[dict[str, Any]], labels_path: str | None) -> dict[str, str]:
    """Map ``file_id -> ground-truth label``.

    Prefers a ``labels.json`` map (keyed by relative path, matched on basename);
    falls back to the parent-folder name of each file's stored original path.
    """
    labels_map: dict[str, str] = {}
    by_basename: dict[str, str] = {}
    if labels_path and Path(labels_path).is_file():
        labels_map = _load(labels_path)
        for rel, label in labels_map.items():
            by_basename[Path(rel).name] = label

    truth: dict[str, str] = {}
    for rec in files:
        fid = rec.get("id")
        if not fid:
            continue
        original = rec.get("original_path") or rec.get("original_filename") or ""
        base = Path(original).name
        label = by_basename.get(base)
        if label is None:
            # One-folder-per-class layout: the parent directory is the class.
            label = Path(original).parent.name or "?"
        truth[fid] = label
    return truth


def _aligned(cluster: dict[str, Any], truth: dict[str, str]) -> tuple[list[str], list[str]]:
    """Ground-truth and predicted labels aligned over docs present in both."""
    gt: list[str] = []
    pred: list[str] = []
    for fid, det in (cluster.get("assignments", {}) or {}).items():
        if fid in truth and det.get("docset") is not None:
            gt.append(truth[fid])
            pred.append(det["docset"])
    return gt, pred


def purity(gt: list[str], pred: list[str]) -> float:
    """Fraction of docs in the majority true-class of their assigned cluster."""
    if not gt:
        return 0.0
    clusters: dict[str, list[str]] = {}
    for g, p in zip(gt, pred, strict=True):
        clusters.setdefault(p, []).append(g)
    correct = sum(Counter(members).most_common(1)[0][1] for members in clusters.values())
    return correct / len(gt)


def external_metrics(gt: list[str], pred: list[str]) -> dict[str, Any]:
    """Standard external cluster-validity scores (needs scikit-learn)."""
    if not gt:
        return {}
    from sklearn import metrics as m

    homogeneity, completeness, v_measure = m.homogeneity_completeness_v_measure(gt, pred)
    return {
        "n_true_classes": len(set(gt)),
        "n_pred_clusters": len(set(pred)),
        "ari": m.adjusted_rand_score(gt, pred),
        "nmi": m.normalized_mutual_info_score(gt, pred),
        "ami": m.adjusted_mutual_info_score(gt, pred),
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
        "purity": purity(gt, pred),
    }


def descriptive_metrics(cluster: dict[str, Any]) -> dict[str, Any]:
    """Shape/confidence stats derived straight from the cluster envelope."""
    assigns = cluster.get("assignments", {}) or {}
    sizes: dict[str, int] = {}
    confs: list[float] = []
    reviews = 0
    for det in assigns.values():
        sizes[det.get("docset")] = sizes.get(det.get("docset"), 0) + 1
        conf = det.get("confidence")
        if conf is not None:
            confs.append(float(conf))
        if det.get("review"):
            reviews += 1
    size_vals = sorted(sizes.values(), reverse=True)
    return {
        "documents": len(assigns),
        "docsets": len(sizes),
        "new_docsets": cluster.get("n_new_clusters", 0),
        "assigned_existing": cluster.get("n_assigned_existing", 0),
        "singletons": sum(1 for v in size_vals if v == 1),
        "largest_cluster": size_vals[0] if size_vals else 0,
        "review_queue": len(cluster.get("review_queue", []) or []) or reviews,
        "failed": len(cluster.get("failed_file_ids", []) or []),
        "mean_conf": statistics.fmean(confs) if confs else None,
        "median_conf": statistics.median(confs) if confs else None,
        "min_conf": min(confs) if confs else None,
    }


def compute(cluster: dict[str, Any], truth: dict[str, str]) -> dict[str, Any]:
    out = descriptive_metrics(cluster)
    gt, pred = _aligned(cluster, truth)
    out.update(external_metrics(gt, pred))
    out["_labeled_docs"] = len(gt)
    return out


# ── Reporting ────────────────────────────────────────────────────────────────
_ROWS: list[tuple[str, str | None, str]] = [
    # (label, key, kind) — kind: "int" | "float"
    ("documents clustered", "documents", "int"),
    ("# DocSets (clusters)", "docsets", "int"),
    ("  new DocSets created", "new_docsets", "int"),
    ("  assigned to existing", "assigned_existing", "int"),
    ("singleton clusters", "singletons", "int"),
    ("largest cluster size", "largest_cluster", "int"),
    ("review queue size", "review_queue", "int"),
    ("failed / unclusterable", "failed", "int"),
    ("mean confidence", "mean_conf", "float"),
    ("median confidence", "median_conf", "float"),
    ("min confidence", "min_conf", "float"),
    ("— external vs ground truth —", None, "header"),
    ("labeled documents", "_labeled_docs", "int"),
    ("# true classes", "n_true_classes", "int"),
    ("Adjusted Rand Index", "ari", "float"),
    ("Normalized MI", "nmi", "float"),
    ("Adjusted MI", "ami", "float"),
    ("homogeneity", "homogeneity", "float"),
    ("completeness", "completeness", "float"),
    ("V-measure", "v_measure", "float"),
    ("purity", "purity", "float"),
]


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def report(before: dict[str, Any], after: dict[str, Any] | None) -> None:
    label_w = max(len(lbl) for lbl, _, _ in _ROWS)
    if after is None:
        print(f"  {'metric'.ljust(label_w)}   {'value':>8}")
        print(f"  {'-' * label_w}   {'-' * 8}")
        for lbl, key, kind in _ROWS:
            if kind == "header" or key is None:
                print(f"\n  {lbl}")
                continue
            print(f"  {lbl.ljust(label_w)}   {_fmt(before.get(key)):>8}")
        return

    print(f"  {'metric'.ljust(label_w)}   {'before':>8}   {'after':>8}   delta")
    print(f"  {'-' * label_w}   {'-' * 8}   {'-' * 8}   -----")
    for lbl, key, kind in _ROWS:
        if kind == "header" or key is None:
            print(f"\n  {lbl}")
            continue
        b, a = before.get(key), after.get(key)
        if b is None or a is None:
            delta = ""
        elif kind == "float":
            delta = f"{a - b:+.3f}"
        else:
            delta = f"{a - b:+d}"
        print(f"  {lbl.ljust(label_w)}   {_fmt(b):>8}   {_fmt(a):>8}   {delta}")


def reassignments(before_json: dict[str, Any], after_json: dict[str, Any]) -> None:
    ba = before_json.get("assignments", {}) or {}
    aa = after_json.get("assignments", {}) or {}
    moved = [
        (fid, ba[fid].get("docset"), aa[fid].get("docset"))
        for fid in sorted(ba.keys() & aa.keys())
        if ba[fid].get("docset") != aa[fid].get("docset")
    ]
    print(f"\n  Consolidation reassignments: {len(moved)} document(s) changed DocSet")
    for fid, b_ds, a_ds in moved[:25]:
        print(f"    {fid}:  {b_ds}  →  {a_ds}")
    if len(moved) > 25:
        print(f"    … and {len(moved) - 25} more")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--before", required=True, help="cluster JSON (the 'before' / only run).")
    p.add_argument("--after", default=None, help="cluster JSON with consolidation on.")
    p.add_argument("--files", required=True, help="`dgml file list` JSON (for ground truth).")
    p.add_argument("--labels", default=None, help="Optional {relpath: label} JSON map.")
    p.add_argument("--json", action="store_true", help="Also dump raw metric dicts as JSON.")
    args = p.parse_args(argv)

    files = _load(args.files).get("files", [])
    truth = build_truth(files, args.labels)

    before_json = _load(args.before)
    after_json = _load(args.after) if args.after else None
    before = compute(before_json, truth)
    after = compute(after_json, truth) if after_json else None

    report(before, after)
    if after_json is not None:
        reassignments(before_json, after_json)

    if args.json:
        print("\n--- raw JSON ---")
        print(json.dumps({"before": before, "after": after}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
