#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zzf/myeye/CAMS-MBRCNet
PYTHON=/home/zzf/miniconda3/envs/octa_repro/bin/python
DATASET=${1:?Usage: train_proposed.sh DATASET VARIANT [SEED]}
VARIANT=${2:?Usage: train_proposed.sh DATASET VARIANT [SEED]}
SEED=${3:-0}
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a nonnegative integer" >&2; exit 2; }

case "$VARIANT" in
    base) GROUP=base ;;
    cams) GROUP=base_cams ;;
    mbrc) GROUP=base_mbrc ;;
    cams_mbrcnet) GROUP=CAMS-MBRCNet ;;
    *) echo "VARIANT must be base|cams|mbrc|cams_mbrcnet" >&2; exit 2 ;;
esac

case "$DATASET" in
    OCTA500_3M)
        case "$SEED" in
            0) CONFIG="$ROOT/configs/octa500_3m.yaml" ;;
            3|5) CONFIG="$ROOT/configs/octa500_3m_seed$SEED.yaml" ;;
            *) echo "No frozen 3M config for seed $SEED" >&2; exit 2 ;;
        esac
        ;;
    OCTA500_6M)
        [[ "$SEED" == 0 && "$VARIANT" == cams_mbrcnet ]] || { echo "6M is configured for Full seed 0 only" >&2; exit 2; }
        CONFIG="$ROOT/configs/octa500_6m.yaml"
        ;;
    ROSSA)
        [[ "$SEED" == 0 && "$VARIANT" == cams_mbrcnet ]] || { echo "ROSSA is configured for Full seed 0 only" >&2; exit 2; }
        CONFIG="$ROOT/configs/rossa.yaml"
        ;;
    *) echo "DATASET must be OCTA500_3M|OCTA500_6M|ROSSA" >&2; exit 2 ;;
esac

"$PYTHON" -B - "$CONFIG" "$DATASET" "$SEED" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
config = json.loads(path.read_text(encoding="utf-8"))
expected = {"dataset": sys.argv[2], "seed": int(sys.argv[3]), "max_epoch": 400, "patience": 40}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f"Config mismatch {key}: {config.get(key)!r} != {value!r}")
print(f"Validated config: {path}", flush=True)
PY

GROUP_DIR="$ROOT/log/$DATASET/proposed/$GROUP/seed_$SEED"
mkdir -p "$GROUP_DIR"
number=1
while true; do
    RUN="$GROUP_DIR/run_$(printf '%03d' "$number")"
    if mkdir "$RUN" 2>/dev/null; then break; fi
    if [[ ! -e "$RUN" ]]; then
        echo "Cannot create run directory: $RUN" >&2
        exit 1
    fi
    number=$((number + 1))
done

export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$ROOT"
nohup "$PYTHON" -u -B "$ROOT/train.py" \
    --config "$CONFIG" --variant "$VARIANT" --result-dir "$RUN" \
    > "$RUN/console.log" 2>&1 &
echo "PID=$!"
echo "RUN=$RUN"
echo "LOG=$RUN/console.log"
