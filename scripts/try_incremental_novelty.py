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

"""Try incremental clustering with the novelty gate on a folder of documents.

This is a hands-on demo of three things that fit together:

1. **Incremental clustering** — new documents are assigned to the nearest
   *existing* DocSet rather than clustered from scratch (scenarios S2/S3).
2. **The novelty gate** — a document that doesn't fit any existing DocSet
   well enough is routed to a fresh ``unknown_N`` bucket instead of being
   force-fit. Here we drive the *quantile* gate (``scenario.threshold_quantile``),
   the corpus-robust default: the ~``(1 - q)`` most-distant incoming docs are
   treated as novel. Without a gate, incremental mode assigns *everything* to
   its nearest DocSet and nothing is ever novel.
3. **The per-document confidence signal** — every assignment now carries a
   real confidence in ``[0, 1]`` (nearest-prototype softmax peak), so you can
   see how sure each placement is instead of a column of ``null``.

Flow (a temporary, throwaway workspace by default):

    seed split ──fresh clustering──▶ initial DocSets
    rest split ──incremental clustering + novelty gate──▶ assign-or-flag-novel

The script prints, per incoming document, the DocSet it landed in, the
assignment confidence, and whether the novelty gate flagged it as new.

It is intentionally NOT part of the public ``dgml`` CLI surface — it is a
demo / debugging tool.

Requirements:
  - the ``clustering`` extra: ``uv sync`` (installs torch, scikit-learn, …)
  - Ghostscript on PATH (page rendering at ingest); PDFs are read digitally.
  - An API key for the classification model used to *name* new DocSets
    (default ``gemini/gemini-3.1-flash-lite`` ⇒ ``GEMINI_API_KEY``). Naming
    failures are reported, not fatal — the clustering itself still runs.

Usage:
    uv run python scripts/try_incremental_novelty.py \
        --files-dir /path/to/samples/4-Infrastructure-Funds/files \
        [--quantile 0.8] [--seed-fraction 0.5] \
        [--model gemini/gemini-3.1-flash-lite] \
        [--workspace /tmp/demo-ws] [--keep] [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

# Ingestible source extensions (case-insensitive), mirroring `dgml file add`.
_INGESTIBLE = {".pdf", ".docx", ".doc", ".xlsx", ".xls"}

# Default corpus — the folder named in the request. Override with --files-dir.
_DEFAULT_FILES_DIR = (
    "/Users/louisenaud/src/ineviam/dgml/dgml-spec/samples/4-Infrastructure-Funds/files"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Demo incremental clustering + novelty gate + confidence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--files-dir",
        type=Path,
        default=Path(_DEFAULT_FILES_DIR),
        help="Directory of source documents to cluster.",
    )
    p.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace directory. Default: a temporary one, deleted on exit "
        "unless --keep is given.",
    )
    p.add_argument(
        "--quantile",
        type=float,
        default=0.8,
        help="Novelty gate: scenario.threshold_quantile in (0, 1). Keeps the "
        "closest q of incoming docs as known; the rest are flagged novel.",
    )
    p.add_argument(
        "--seed-fraction",
        type=float,
        default=0.5,
        help="Fraction of the corpus used to seed initial DocSets via a fresh "
        "clustering pass. The remainder is clustered incrementally.",
    )
    p.add_argument(
        "--model",
        default="gemini/gemini-3.1-flash-lite",
        help="Vision LLM used to name new DocSets (needs the matching API key).",
    )
    p.add_argument(
        "--seed-algorithm",
        default=None,
        help="Optional cluster_algorithm override for the fresh seed pass "
        "(e.g. hdbscan, leiden, kmeans). Default: the bundled config's choice.",
    )
    p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
    p.add_argument("--keep", action="store_true", help="Keep the workspace after finishing.")
    p.add_argument("--json", action="store_true", help="Emit the raw result dicts as JSON.")
    return p.parse_args(argv)


def _discover_files(files_dir: Path, *, recursive: bool) -> list[Path]:
    """Every ingestible source file in ``files_dir``, sorted for determinism."""
    if not files_dir.is_dir():
        raise SystemExit(f"error: --files-dir is not a directory: {files_dir}")
    walker = files_dir.rglob("*") if recursive else files_dir.glob("*")
    found = sorted(p for p in walker if p.is_file() and p.suffix.lower() in _INGESTIBLE)
    if not found:
        raise SystemExit(
            f"error: no ingestible files ({', '.join(sorted(_INGESTIBLE))}) found in {files_dir}"
        )
    return found


def _split(files: list[Path], seed_fraction: float) -> tuple[list[Path], list[Path]]:
    """Split into (seed, incremental), guaranteeing at least one seed file and
    (when possible) at least one incremental file so both phases have work."""
    n = len(files)
    if n == 1:
        return files, []
    seed_n = max(1, min(n - 1, math.ceil(n * seed_fraction)))
    return files[:seed_n], files[seed_n:]


def _write_config(config_path: Path, *, model: str, quantile: float, algorithm: str | None) -> None:
    """Seed <workspace>/config.json with the classification model (for naming
    new DocSets) and the clustering novelty gate."""
    scenario: dict[str, Any] = {"threshold_quantile": quantile}
    if algorithm:
        scenario["cluster_algorithm"] = algorithm
    config = {
        "classification": {"model": model},
        "clustering": {"scenario": scenario},
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _ingest(store: Any, paths: list[Path], conflict: Any) -> dict[str, str]:
    """Add each file; return {file_id: filename}. Reports render failures."""
    from dgml_core.errors import DgmlError

    ids: dict[str, str] = {}
    for path in paths:
        try:
            result = store.add(path, on_conflict=conflict)
        except DgmlError as exc:
            print(f"  ! skipped {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        ids[result.record.id] = path.name
        if getattr(result, "page_render_error", None):
            print(
                f"  ! {path.name}: page render issue ({result.page_render_error}); "
                "it won't be clusterable",
                file=sys.stderr,
            )
    return ids


def _fmt_conf(conf: float | None) -> str:
    return "  n/a" if conf is None else f"{conf:5.2f}"


def _print_assignments(
    title: str, assignments: dict[str, dict[str, Any]], names: dict[str, str]
) -> None:
    print(f"\n{title}")
    if not assignments:
        print("  (none)")
        return
    width = max((len(names.get(fid, fid)) for fid in assignments), default=8)
    header = f"  {'document'.ljust(width)}  novel  conf   docset"
    print(header)
    print(f"  {'-' * width}  -----  -----  ------")
    # Novel (gate-flagged) first, then by ascending confidence.
    for fid, detail in sorted(
        assignments.items(),
        key=lambda kv: (not kv[1].get("is_new"), kv[1].get("confidence") or 0.0),
    ):
        name = names.get(fid, fid).ljust(width)
        novel = " NEW " if detail.get("is_new") else "  -  "
        conf = _fmt_conf(detail.get("confidence"))
        print(f"  {name}  {novel}  {conf}  {detail.get('docset')}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not 0.0 < args.quantile < 1.0:
        raise SystemExit(f"error: --quantile must be in (0, 1); got {args.quantile}")

    try:
        from dgml_core.clustering import clustering
        from dgml_core.docsets import DocSetStore
        from dgml_core.errors import IncrementalWithoutClusters
        from dgml_core.files import ConflictPolicy, FileStore
        from dgml_core.storage import Workspace
    except ImportError as exc:  # pragma: no cover — clustering extra missing
        raise SystemExit(
            f"error: could not import dgml_core clustering stack ({exc}). "
            "Install the clustering extra: `uv sync` (or pip install dgml[clustering])."
        ) from exc

    files = _discover_files(args.files_dir, recursive=args.recursive)
    seed_files, incr_files = _split(files, args.seed_fraction)

    made_temp = args.workspace is None
    ws_root = Path(tempfile.mkdtemp(prefix="dgml-novelty-")) if made_temp else args.workspace
    ws = Workspace(root=ws_root)
    ws.init()
    _write_config(
        ws.config_path, model=args.model, quantile=args.quantile, algorithm=args.seed_algorithm
    )

    print(f"Corpus:     {args.files_dir}  ({len(files)} files)")
    print(f"Workspace:  {ws_root}")
    print(
        f"Novelty gate: scenario.threshold_quantile = {args.quantile} "
        f"(≈ closest {args.quantile:.0%} kept as known, rest flagged novel)"
    )

    store = FileStore(ws)

    # ── Phase 1: seed initial DocSets with a fresh clustering pass ──────────
    print(f"\n[1/2] Seeding DocSets from {len(seed_files)} file(s) (fresh clustering)…")
    seed_names = _ingest(store, seed_files, ConflictPolicy.SKIP)
    seed_result = clustering(ws, mode="fresh", method="embedding")
    docsets = DocSetStore(ws).list_all()
    if not docsets:
        print(
            "\nNo DocSets were created from the seed split — nothing to assign into.\n"
            "The seed corpus may be too small/uniform, or naming (LLM) failed "
            "(check the API key for --model). Try a larger --seed-fraction.",
            file=sys.stderr,
        )
        if seed_result.get("failed_file_ids"):
            print(f"  seed failed_file_ids: {seed_result['failed_file_ids']}", file=sys.stderr)
        if not made_temp or args.keep:
            print(f"  workspace kept at {ws_root}", file=sys.stderr)
        elif made_temp:
            shutil.rmtree(ws_root, ignore_errors=True)
        return 1
    print(f"  → {len(docsets)} DocSet(s): {', '.join(sorted(d.name for d in docsets))}")

    # ── Phase 2: incremental clustering of the rest, with the novelty gate ──
    print(f"\n[2/2] Incrementally clustering {len(incr_files)} new file(s) with the novelty gate…")
    incr_result: dict[str, Any]
    if not incr_files:
        print("  (no incremental split — increase the corpus or lower --seed-fraction)")
        incr_result = {"assignments": {}, "n_assigned_existing": 0, "n_new_clusters": 0}
        incr_names: dict[str, str] = {}
    else:
        incr_names = _ingest(store, incr_files, ConflictPolicy.SKIP)
        try:
            incr_result = clustering(ws, mode="incremental", method="embedding")
        except IncrementalWithoutClusters as exc:
            raise SystemExit(f"error: {exc}") from exc

    # ── Report ──────────────────────────────────────────────────────────────
    all_names = {**seed_names, **incr_names}
    _print_assignments(
        "Incremental assignments (novel = opened a new DocSet):",
        incr_result.get("assignments", {}),
        all_names,
    )
    n_new = incr_result.get("n_new_clusters", 0)
    n_existing = incr_result.get("n_assigned_existing", 0)
    failed = incr_result.get("failed_file_ids", [])
    print(
        f"\nSummary: {n_existing} assigned to existing DocSets, "
        f"{n_new} flagged novel (new DocSets)."
    )
    if failed:
        print(f"  unclusterable / failed: {len(failed)} file(s): {failed}")

    if args.json:
        print("\n--- raw JSON ---")
        print(json.dumps({"seed": seed_result, "incremental": incr_result}, indent=2))

    if made_temp and not args.keep:
        shutil.rmtree(ws_root, ignore_errors=True)
    else:
        print(f"\nWorkspace kept at: {ws_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
