#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zzf/myeye/CAMS-MBRCNet
PYTHON=/home/zzf/miniconda3/envs/octa_repro/bin/python
# Dedicated launcher for the five additional comparison models.
if [[ "$#" -ne 2 ]]; then
  echo "Usage: bash $(basename "$0") MODEL SEED" >&2
  echo "Models: octamamba hv_octamamba dgnet pmdc_net s3_octa" >&2
  exit 2
fi
MODEL=$1
SEED=$2

case "$MODEL" in
  octamamba|hv_octamamba|dgnet|pmdc_net|s3_octa) ;;
  *) echo "Unsupported model: $MODEL; choose octamamba|hv_octamamba|dgnet|pmdc_net|s3_octa" >&2; exit 2 ;;
esac
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be nonnegative integer" >&2; exit 2; }
# Match argparse integer seeds and the trainer seed_N directory contract.
SEED="${SEED#"${SEED%%[!0]*}"}"
SEED=${SEED:-0}
if [[ ${#SEED} -gt 10 ]] || { [[ ${#SEED} -eq 10 ]] && [[ "$SEED" > "4294967295" ]]; }; then
  echo "SEED must be in [0, 4294967295] for NumPy/PyTorch seed compatibility" >&2
  exit 2
fi

GROUP="$ROOT/log/OCTA500_6M/comparison/$MODEL/seed_$SEED"
mkdir -p "$GROUP"
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
  --dataset "OCTA500_6M" --model "$MODEL" --seed "$SEED" --result-dir "$RUN" \
  > "$RUN/console.log" 2>&1 &
echo "PID=$!"
echo "RUN=$RUN"
echo "LOG=$RUN/console.log"
