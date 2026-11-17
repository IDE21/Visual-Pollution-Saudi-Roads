#!/usr/bin/env bash
# Plan v2, step 5: resolution screening, 640 vs 960 px. Decision on VAL only; the test split is not touched.
#
# Both runs: YOLO11s, the first half of the (sorted) train list (fraction 0.5), 40 epochs, seed 42,
# no early stopping. Then both best.pt are scored on val with incident-level bootstrap CIs, paired.
# Measured on an M5 Pro: 960 px ~ 1.2 s/batch, 17.9 GB -> about 11 h; 640 px about 4.5 h.
#
# Usage:  DATA="<dataset root>" RUNS="<output dir>" PY="<python>" bash scripts/screen_resolution.sh
# Re-running skips finished runs (marked with a DONE file).
# Keep the lid open and the charger connected: caffeinate stops idle sleep, not lid-close sleep.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA:?set DATA to the dataset root}" "${RUNS:?set RUNS to the output dir}"
PY="${PY:-python}"
MODEL="${MODEL:-yolo11s.pt}"
YAML="$DATA/data_v1.yaml"
OUT=reports/resolution_screen
mkdir -p "$RUNS" "$OUT"

run() {  # $1 = imgsz
  local dir="$RUNS/screen_$1"
  [ -f "$dir/DONE" ] && { echo "screen_$1 already done"; return; }
  [ -f "$dir/weights/last.pt" ] && { echo "screen_$1 exists but is unfinished: delete $dir or resume it by hand"; exit 1; }
  echo "$(date '+%F %T') training screen_$1"
  caffeinate -i -s "$PY" src/train.py --data "$YAML" --model "$MODEL" --imgsz "$1" --batch 16 \
    --epochs 40 --patience 100 --fraction 0.5 --no-test --project "$RUNS" --name "screen_$1" \
    > "$RUNS/screen_$1.log" 2>&1
  touch "$dir/DONE"
}

run 960
run 640

for s in 640 960; do
  echo "$(date '+%F %T') predicting val at $s px"
  caffeinate -i -s "$PY" src/eval_ci.py predict --weights "$RUNS/screen_$s/weights/best.pt" \
    --list "$DATA/val_v1.txt" --out "$RUNS/val_preds_$s.npz" --imgsz "$s" --batch 16 --device mps
  "$PY" src/eval_ci.py score "$RUNS/val_preds_$s.npz" --json "$OUT/val_$s.json"
done
# paired difference: 960 minus 640, same val images and the same incident resamples
"$PY" src/eval_ci.py score "$RUNS/val_preds_960.npz" --paired "$RUNS/val_preds_640.npz" --json "$OUT/val_960_minus_640.json"
for s in 640 960; do cp "$RUNS/screen_$s/results.csv" "$OUT/results_$s.csv"; done
echo "$(date '+%F %T') done -> $OUT"
