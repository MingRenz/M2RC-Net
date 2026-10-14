#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zzf/myeye/CAMS-MBRCNet
PYTHON=/home/zzf/miniconda3/envs/octa_repro/bin/python

if [[ "$#" -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 SEED" >&2
    exit 2
fi

SEED="$1"
CONFIG="$ROOT/configs/octa500_3m_seed$SEED.yaml"
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

for PAIR in base:base cams:base_cams mbrc:base_mbrc cams_mbrcnet:CAMS-MBRCNet; do
    IFS=: read -r VARIANT GROUP <<< "$PAIR"
    RUN="$ROOT/log/OCTA500_3M/proposed/$GROUP/seed_$SEED"
    if [[ -e "$RUN" ]]; then
        echo "Refusing to overwrite existing run: $RUN" >&2
        exit 1
    fi
done

QUEUE="$ROOT/log/OCTA500_3M/queues/seed_$SEED"
mkdir -p "$(dirname "$QUEUE")"
if ! mkdir "$QUEUE"; then
    echo "Refusing to reuse existing queue: $QUEUE" >&2
    exit 1
fi
exec > "$QUEUE/queue.log" 2>&1
echo "$(date -Is) QUEUE START seed=$SEED"

for PAIR in base:base cams:base_cams mbrc:base_mbrc cams_mbrcnet:CAMS-MBRCNet; do
    IFS=: read -r VARIANT GROUP <<< "$PAIR"
    RUN="$ROOT/log/OCTA500_3M/proposed/$GROUP/seed_$SEED"
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
