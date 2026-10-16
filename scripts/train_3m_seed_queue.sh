#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zzf/myeye/CAMS-MBRCNet
PYTHON=/home/zzf/miniconda3/envs/octa_repro/bin/python

if [[ "$#" -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 SEED" >&2
    exit 2
fi

SEED="$1"
if [[ "$SEED" == 0 ]]; then
    CONFIG="$ROOT/configs/octa500_3m.yaml"
else
    CONFIG="$ROOT/configs/octa500_3m_seed$SEED.yaml"
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "Missing config: $CONFIG" >&2
    exit 1
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$ROOT"

"$PYTHON" -B - "$CONFIG" "$SEED" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as file:
    config = json.load(file)
seed = int(sys.argv[2])
expected = {
    "dataset": "OCTA500_3M",
    "seed": seed,
    "max_epoch": 400,
    "patience": 40,
    "split": {"train": 140, "val": 10, "test": 50},
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f"Config mismatch: {key}: {config.get(key)!r} != {value!r}")
print(f"Validated seed {seed} config: {sys.argv[1]}", flush=True)
PY

BASE="$ROOT/log/OCTA500_3M"
QUEUE_GROUP="$BASE/queues/seed_$SEED"
mkdir -p "$QUEUE_GROUP"
number=1
while true; do
    TAG="run_$(printf '%03d' "$number")"
    QUEUE="$QUEUE_GROUP/$TAG"
    occupied=0
    for PAIR in base:base cams:base_cams mbrc:base_mbrc cams_mbrcnet:CAMS-MBRCNet; do
        IFS=: read -r VARIANT GROUP <<< "$PAIR"
        if [[ -e "$BASE/proposed/$GROUP/seed_$SEED/$TAG" ]]; then
            occupied=1
            break
        fi
    done
    if (( occupied )); then
        number=$((number + 1))
        continue
    fi
    if mkdir "$QUEUE" 2>/dev/null; then
        break
    fi
    if [[ -e "$QUEUE" ]]; then
        number=$((number + 1))
        continue
    fi
    echo "Cannot create queue run directory: $QUEUE" >&2
    exit 1
done

exec > "$QUEUE/queue.log" 2>&1
echo "$(date -Is) QUEUE START seed=$SEED run=$TAG"

for PAIR in base:base cams:base_cams mbrc:base_mbrc cams_mbrcnet:CAMS-MBRCNet; do
    IFS=: read -r VARIANT GROUP <<< "$PAIR"
    RUN="$BASE/proposed/$GROUP/seed_$SEED/$TAG"
    mkdir -p "$(dirname "$RUN")"
    mkdir "$RUN"
    echo "$(date -Is) START $VARIANT $RUN"
    if "$PYTHON" -u -B "$ROOT/train.py" --config "$CONFIG" --variant "$VARIANT" --result-dir "$RUN" > "$RUN/console.log" 2>&1; then
        if [[ ! -f "$RUN/finished.flag" ]]; then
            echo "$(date -Is) FAILED $VARIANT: missing finished.flag" >&2
            exit 1
        fi
        echo "$(date -Is) FINISHED $VARIANT $RUN"
    else
        STATUS=$?
        echo "$(date -Is) FAILED $VARIANT exit=$STATUS; inspect $RUN/console.log" >&2
        exit "$STATUS"
    fi
done

echo "$(date -Is) ALL FOUR SEED $SEED RUNS FINISHED"
