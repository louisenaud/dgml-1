#!/usr/bin/env bash
# Licensed under the Apache License, Version 2.0 (the "License").
#
# End-to-end demo driven entirely through the `dgml` CLI:
#   1. Ingest a folder of documents            (dgml file add <dir>)
#   2. Cluster them into named DocSets          (dgml cluster --mode fresh)
#   3. Cluster again WITH consolidation enabled (dgml cluster, consolidation on)
#
# It then prints clustering metrics BEFORE and AFTER the consolidation pass and
# a side-by-side diff.
#
# Why two workspaces?  Consolidation is not a separate CLI command — it runs
# *inside* `dgml cluster` when `clustering.scenario.consolidation.enabled` is
# true in config.json.  Once files are assigned to DocSets, re-running `cluster`
# is a no-op, so there is no in-place "run consolidation now" step.  Instead we
# ingest ONCE, copy the ingested workspace, and cluster each copy: one with
# consolidation off (the "before"), one with it on (the "after").  Base
# clustering is seeded and deterministic, so the two partitions are identical
# except for what consolidation changes — a clean before/after.
#
# Requirements:
#   - `uv sync` (installs the clustering extra: torch, scikit-learn, …)
#   - Ghostscript on PATH (page-image rendering at ingest time)
#   - An API key for the naming/consolidation model (default GEMINI_API_KEY)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Config (override via env) ────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/Users/louisenaud/data/demo2026}"
# DATA_DIR="${DATA_DIR:-/Users/louisenaud/data/discovery_megalos/Discovery_Set_1}"
MODEL="${MODEL:-gemini/gemini-3.1-flash-lite}"
API_KEY_ENV="${API_KEY_ENV:-GEMINI_API_KEY}"     # env var the model reads its key from
ORG="${ORG:-demo2026}"                            # org slug embedded in DocSet URIs
RUN_ROOT="${RUN_ROOT:-$HOME/dgml-runs/demo2026}" # where workspaces + logs are written
CONSOLIDATE_QUANTILE="${CONSOLIDATE_QUANTILE:-0.2}"  # bottom fraction sent to adjudication
CONSOLIDATE_MODE="${CONSOLIDATE_MODE:-reassign}"     # reassign | repartition | auto
RECURSIVE="${RECURSIVE:-1}"                       # 1 = descend into subdirs (corpus is nested)
OCR_PROVIDER="${OCR_PROVIDER:-macos}"             # macos (Apple Vision, on-device) | azure | aws
                                                  # the corpus is scanned PDFs, so OCR is required;
                                                  # the seeded config ships an Azure placeholder that
                                                  # must be replaced before text extraction works.
TEXT_MODE="${TEXT_MODE:-digital}"                 # digital (embedded text) | ocr | hybrid.
                                                  # 'ocr' for scanned PDFs; 'hybrid' runs BOTH per
                                                  # page (2x cost) — prefer 'ocr' unless you need it.
DPI="${DPI:-300}"                                 # page-render resolution. 300 = archival default;
                                                  # set 150 to ~halve rasterization time + disk (fine
                                                  # for OCR + clustering). Changes the render, so the
                                                  # before/after clustering baseline shifts with it.
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)}"
                                                  # parallel file-add workers. `dgml file add <dir>`
                                                  # is serial internally and single-page PDFs never
                                                  # trigger its in-file OCR concurrency, so we fan
                                                  # out across files here. Set JOBS=1 for the plain
                                                  # serial bulk add.

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$RUN_ROOT/$STAMP"
WS_BEFORE="$RUN_DIR/ws-before"   # clustering + naming only
WS_AFTER="$RUN_DIR/ws-after"     # clustering + naming + consolidation
LOG="$RUN_DIR/run.log"
DGML=(uv run --project "$REPO_ROOT" dgml)
PYRUN=(uv run --project "$REPO_ROOT" python)   # same venv, for the config-patch helper

mkdir -p "$RUN_DIR"

# ── Logging ──────────────────────────────────────────────────────────────────
# Everything (stdout + stderr) is tee'd to $LOG with timestamps on section marks.
exec > >(tee -a "$LOG") 2>&1

log()     { printf '\n\033[1;36m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
substep() { printf '\033[0;90m    · %s\033[0m\n' "$*"; }
die()     { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ── Preflight ────────────────────────────────────────────────────────────────
log "Preflight checks"
[ -d "$DATA_DIR" ] || die "data dir not found: $DATA_DIR"
command -v gs >/dev/null 2>&1 || die "Ghostscript ('gs') not on PATH — needed for page rendering"
if [ -z "${!API_KEY_ENV:-}" ]; then
  die "\$$API_KEY_ENV is not set — needed to name DocSets and to run consolidation.
       export $API_KEY_ENV=... (or set MODEL / API_KEY_ENV to another provider)"
fi
substep "data dir  : $DATA_DIR"
substep "model     : $MODEL   (key from \$$API_KEY_ENV)"
substep "ocr       : $OCR_PROVIDER$([ "$OCR_PROVIDER" = macos ] && echo '   (Apple Vision, on-device — no key)')"
substep "text mode : $TEXT_MODE"
substep "dpi       : $DPI"
substep "ingest    : $JOBS parallel worker(s)"
substep "run dir   : $RUN_DIR"
substep "log file  : $LOG"

# Patch a workspace config.json in place: set the classification model and,
# when $2 == 1, enable the clustering consolidation pass.  Reads the seeded
# JSONC config through dgml_core so comments are handled, merges, rewrites JSON.
patch_config() {
  local cfg_path="$1" consolidation_enabled="$2"
  "${PYRUN[@]}" - "$cfg_path" "$consolidation_enabled" "$MODEL" \
      "$API_KEY_ENV" "$CONSOLIDATE_QUANTILE" "$CONSOLIDATE_MODE" "$OCR_PROVIDER" <<'PY'
import json, sys
from pathlib import Path
from dgml_core.storage import read_config

cfg_path, enabled, model, key_env, quantile, mode, ocr_provider = sys.argv[1:8]
cfg = read_config(Path(cfg_path))

# Naming (and, when enabled, consolidation) model.
cls = cfg.setdefault("classification", {})
cls["model"] = model
if key_env and key_env != "GEMINI_API_KEY":
    cls["api_key_env"] = key_env

# OCR: the seeded config ships an Azure placeholder endpoint
# (<your-di-resource>) that fails on any scanned PDF. Replace it with a
# working provider so text extraction actually produces page_text — without
# it every document is empty and clustering has nothing to encode.
if ocr_provider == "macos":
    cfg["ocr"] = {"provider": "macos"}  # Apple Vision, on-device, no key
elif ocr_provider == "azure":
    endpoint = cfg.get("ocr", {}).get("endpoint", "")
    if not endpoint or "<your-di-resource>" in endpoint:
        sys.exit(
            "OCR_PROVIDER=azure but config.json has no real Azure endpoint. "
            "Set ocr.endpoint (and credentials), or use OCR_PROVIDER=macos."
        )
    cfg.setdefault("ocr", {})["provider"] = "azure"
elif ocr_provider == "aws":
    cfg["ocr"] = {"provider": "aws", **{k: v for k, v in cfg.get("ocr", {}).items() if k == "region"}}
else:
    sys.exit(f"unknown OCR_PROVIDER {ocr_provider!r} (expected macos | azure | aws)")

# Only override scenario.consolidation; every other clustering default
# (encoder, algorithm, seed) is left untouched so both runs partition alike.
scenario = cfg.setdefault("clustering", {}).setdefault("scenario", {})
scenario["consolidation"] = {
    "enabled": enabled == "1",
    "apply": "auto",          # write the reassignments (so metrics actually move)
    "mode": mode,
    "candidates_k": 3,
    "model": model,
    "selector": {"strategy": "quantile", "quantile": float(quantile)},
}

with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
print(f"patched {cfg_path}: consolidation.enabled={enabled == '1'}")
PY
}

# ── Step 1: ingestion (once) ─────────────────────────────────────────────────
log "STEP 1/3 · Ingesting documents from $DATA_DIR"
# `workspace create` builds docsets/ + files/ + config.json (auto-creating the
# shared local_config.json if `dgml init` was never run).
"${DGML[@]}" workspace create "$WS_BEFORE" --organization "$ORG" >/dev/null
patch_config "$WS_BEFORE/config.json" 0     # consolidation OFF for the base workspace

add_json="$RUN_DIR/ingest.json"

if [ "$JOBS" -gt 1 ]; then
  # Parallel path: `dgml file add <dir>` is serial internally, so fan out
  # across files ourselves. Each worker adds ONE file and writes its own JSON
  # envelope into $adds_dir (per-file filenames ⇒ no concurrent-write races);
  # we then aggregate them into the same bulk-shaped envelope the serial path
  # emits, so everything downstream is unchanged. Concurrent adds are safe:
  # each file lands in its own files/<random-id>/ dir with no shared index.
  maxdepth=(); [ "$RECURSIVE" = "1" ] || maxdepth=(-maxdepth 1)
  adds_dir="$RUN_DIR/adds"
  mkdir -p "$adds_dir"
  substep "fanning out with $JOBS workers (text-mode=$TEXT_MODE)"
  # Resolve the CLI once so workers skip uv's per-call project resolution.
  dgml_bin="$REPO_ROOT/.venv/bin/dgml"
  [ -x "$dgml_bin" ] || dgml_bin=""
  find "$DATA_DIR" ${maxdepth[@]+"${maxdepth[@]}"} -type f -iname '*.pdf' -print0 \
    | xargs -0 -P "$JOBS" -I{} bash -c '
        f="$1"; ws="$2"; adds="$3"; tm="$4"; bin="$5"; repo="$6"; dpi="$7"
        out="$adds/$(printf "%s" "$f" | shasum | cut -c1-16).json"
        if [ -n "$bin" ]; then
          "$bin" --workspace "$ws" file add "$f" --on-conflict skip --text-mode "$tm" --dpi "$dpi" >"$out" 2>/dev/null || true
        else
          uv run --project "$repo" dgml --workspace "$ws" file add "$f" \
            --on-conflict skip --text-mode "$tm" --dpi "$dpi" >"$out" 2>/dev/null || true
        fi
      ' _ {} "$WS_BEFORE" "$adds_dir" "$TEXT_MODE" "$dgml_bin" "$REPO_ROOT" "$DPI" || true

  # Aggregate the per-file envelopes into one bulk-shaped ingest.json.
  "${PYRUN[@]}" - "$adds_dir" "$DATA_DIR" "$RECURSIVE" "$add_json" <<'PY'
import json, sys
from pathlib import Path

adds_dir, data_dir, recursive, out_path = sys.argv[1:5]
counts = {"added": 0, "skipped": 0, "soft_failed": 0, "hard_failed": 0}
results = []
for p in sorted(Path(adds_dir).glob("*.json")):
    try:
        payload = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        counts["hard_failed"] += 1
        continue
    if "error" in payload or "file" not in payload:  # error envelope / empty
        counts["hard_failed"] += 1
        status = "hard_failed"
    elif not payload.get("created"):
        counts["skipped"] += 1
        status = "skipped"
    elif any(
        payload.get(k)
        for k in ("page_render_error", "page_count_error",
                  "text_extraction_error", "conversion_error")
    ):
        counts["soft_failed"] += 1
        status = "soft_failed"
    else:
        counts["added"] += 1
        status = "added"
    results.append({"status": status, **payload})

envelope = {
    "directory": data_dir,
    "recursive": recursive == "1",
    "summary": {"total": len(results), **counts},
    "results": results,
}
Path(out_path).write_text(json.dumps(envelope, indent=2) + "\n")
PY
else
  # Serial path: the plain bulk add (JOBS=1).
  RECURSE_FLAG=()
  [ "$RECURSIVE" = "1" ] && RECURSE_FLAG=(--recursive)
  "${DGML[@]}" --workspace "$WS_BEFORE" --verbose \
      file add "$DATA_DIR" --on-conflict skip \
      ${RECURSE_FLAG[@]+"${RECURSE_FLAG[@]}"} --text-mode "$TEXT_MODE" --dpi "$DPI" > "$add_json"
fi

read -r n_added n_skipped n_total < <(python3 -c '
import json, sys
s = json.load(open(sys.argv[1])).get("summary", {})
print(s.get("added", "?"), s.get("skipped", "?"), s.get("total", "?"))
' "$add_json" 2>/dev/null || echo "? ? ?")
substep "ingest envelope: $add_json"
substep "files: $n_total ingestible, $n_added added, $n_skipped skipped (already present)"
if [ "$n_total" = "0" ]; then
  die "no ingestible files found in $DATA_DIR (RECURSIVE=$RECURSIVE) — nothing to cluster"
fi

# Clone the freshly-ingested workspace so the "after" run starts from the same
# corpus (no re-rendering) and enable consolidation on the clone.
log "Preparing consolidation workspace (copy of ingested corpus)"
cp -R "$WS_BEFORE" "$WS_AFTER"
patch_config "$WS_AFTER/config.json" 1      # consolidation ON

# ── Step 2: cluster + name (BEFORE consolidation) ────────────────────────────
log "STEP 2/3 · Clustering + naming (consolidation OFF)  → 'before'"
before_json="$RUN_DIR/cluster_before.json"
"${DGML[@]}" --workspace "$WS_BEFORE" --verbose cluster --mode fresh > "$before_json"
substep "cluster output: $before_json"

# ── Step 3: cluster + name + consolidate (AFTER consolidation) ───────────────
log "STEP 3/3 · Clustering + naming + CONSOLIDATION (consolidation ON) → 'after'"
after_json="$RUN_DIR/cluster_after.json"
"${DGML[@]}" --workspace "$WS_AFTER" --verbose cluster --mode fresh > "$after_json"
substep "cluster output: $after_json"

# ── Metrics + before/after diff ──────────────────────────────────────────────
log "Clustering metrics — BEFORE vs AFTER consolidation"
# Ground truth for the external metrics: file_id → source folder (one folder per
# class). `dgml file list` gives the id ↔ original-path mapping; ids are shared
# across both workspaces (ws-after is a copy of ws-before), so one dump serves
# both. labels.json, when present in the corpus, is used in preference.
files_json="$RUN_DIR/files.json"
"${DGML[@]}" --workspace "$WS_BEFORE" file list > "$files_json"

labels_flag=()
[ -f "$DATA_DIR/labels.json" ] && labels_flag=(--labels "$DATA_DIR/labels.json")

"${PYRUN[@]}" "$REPO_ROOT/scripts/clustering_metrics.py" \
    --before "$before_json" \
    --after "$after_json" \
    --files "$files_json" \
    ${labels_flag[@]+"${labels_flag[@]}"}

log "Done."
substep "before workspace : $WS_BEFORE   (clustering + naming only)"
substep "after  workspace : $WS_AFTER    (clustering + naming + consolidation)"
substep "full log         : $LOG"
echo
echo "Inspect DocSets with:  ${DGML[*]} --workspace \"$WS_AFTER\" docset list"
