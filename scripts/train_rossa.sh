#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zzf/myeye/CAMS-MBRCNet
variant="$1"
case "$variant" in
  base) group=base ;;
  cams) group=base_cams ;;
  mbrc) group=base_mbrc ;;
  cams_mbrcnet) group=CAMS-MBRCNet ;;
  *) echo "variant: base|cams|mbrc|cams_mbrcnet" >&2; exit 2 ;;
esac
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUNS="$ROOT/log/ROSSA/$group"
mkdir -p "$RUNS"
number=1
while true; do
  run="$RUNS/run_$(printf '%03d' "$number")"
  if mkdir "$run" 2>/dev/null; then break; fi
  number=$((number+1))
done
nohup /home/zzf/miniconda3/envs/octa_repro/bin/python -u -B "$ROOT/train.py"   --config "$ROOT/configs/rossa.yaml" --variant "$variant"   --result-dir "$run" > "$run/console.log" 2>&1 &
echo "PID=$!"
echo "RUN=$run"
echo "LOG=$run/console.log"
