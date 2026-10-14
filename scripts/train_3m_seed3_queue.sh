#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zzf/myeye/CAMS-MBRCNet
PYTHON=/home/zzf/miniconda3/envs/octa_repro/bin/python
CONFIG="$ROOT/configs/octa500_3m_seed3.yaml"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$ROOT"
"$PYTHON" -B -c 'import json,sys; c=json.load(open(sys.argv[1])); assert c["dataset"]=="OCTA500_3M" and c["seed"]==3 and c["max_epoch"]==400 and c["patience"]==40' "$CONFIG"
variants=(base cams mbrc cams_mbrcnet)
groups=(base base_cams base_mbrc CAMS-MBRCNet)
for group in "${groups[@]}"; do
    run="$ROOT/log/OCTA500_3M/proposed/$group/seed_3"
    if [[ -e "$run" ]]; then
        echo "Refusing to overwrite existing run: $run" >&2
        exit 1
    fi
done
for index in "${!variants[@]}"; do
    variant="${variants[$index]}"
    group="${groups[$index]}"
    run="$ROOT/log/OCTA500_3M/proposed/$group/seed_3"
    mkdir -p "$(dirname "$run")"
    mkdir "$run"
    echo "$(date -Is) START $variant $run"
    if "$PYTHON" -u -B "$ROOT/train.py" --config "$CONFIG" --variant "$variant" --result-dir "$run" > "$run/console.log" 2>&1; then
        if [[ ! -f "$run/finished.flag" ]]; then
            echo "$(date -Is) FAILED $variant: missing finished.flag" >&2
            exit 1
        fi
        echo "$(date -Is) FINISHED $variant $run"
    else
        status=$?
        echo "$(date -Is) FAILED $variant exit=$status; inspect $run/console.log" >&2
        exit "$status"
    fi
done
echo "$(date -Is) ALL FOUR SEED 3 RUNS FINISHED"
