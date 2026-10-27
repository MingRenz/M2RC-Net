#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zzf/myeye/CAMS-MBRCNet
PYTHON=/home/zzf/miniconda3/envs/octa_repro/bin/python
MODEL=${1:?Usage: bash scripts/train_comparison_rossa.sh MODEL [SEED]}
SEED=${2:-0}

case "$MODEL" in
  unet|unetpp|swin_unet|h2former|vm_unet|h_vmunet|octamamba|hv_octamamba|dgnet|pmdc_net|s3_octa) ;;
  *) echo "Unknown comparison model: $MODEL" >&2; exit 2 ;;
esac
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be nonnegative integer" >&2; exit 2; }
# Match argparse integer seeds and the trainer seed_N directory contract.
SEED="${SEED#"${SEED%%[!0]*}"}"
SEED=${SEED:-0}

GROUP="$ROOT/log/ROSSA/comparison/$MODEL/seed_$SEED"
mkdir -p "$GROUP"
if pgrep -f "^$PYTHON -u -B -m comparison_models[.]train_(3m|multidataset) " >/dev/null; then
  echo "A comparison training process is still running; start the next run after it finishes." >&2
  exit 1
fi
number=1
while true; do
  RUN="$GROUP/run_$(printf '%03d' "$number")"
  if mkdir "$RUN" 2>/dev/null; then break; fi
  if [[ ! -e "$RUN" ]]; then
    echo "Cannot create run directory: $RUN" >&2
    exit 1
  fi
  number=$((number+1))
done

export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$ROOT"
nohup "$PYTHON" -u -B -m comparison_models.train_multidataset \
  --dataset "ROSSA" --model "$MODEL" --seed "$SEED" --result-dir "$RUN" \
  > "$RUN/console.log" 2>&1 &
echo "PID=$!"
echo "RUN=$RUN"
echo "LOG=$RUN/console.log"
