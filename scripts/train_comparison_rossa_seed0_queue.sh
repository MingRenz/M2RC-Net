#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zzf/myeye/CAMS-MBRCNet
PYTHON=/home/zzf/miniconda3/envs/octa_repro/bin/python
MODELS=(swin_unet h2former vm_unet h_vmunet)
QUEUE_BASE="$ROOT/log/ROSSA/queues/comparison_seed_0"

allocate_run() {
    local parent=$1 number=1 candidate
    mkdir -p "$parent"
    while true; do
        printf -v candidate '%s/run_%03d' "$parent" "$number"
        if mkdir "$candidate" 2>/dev/null; then
            printf '%s\n' "$candidate"
            return 0
        fi
        if [[ ! -e "$candidate" ]]; then
            printf 'Cannot create run directory: %s\n' "$candidate" >&2
            return 1
        fi
        number=$((number + 1))
    done
}

[[ -x "$PYTHON" ]] || { echo "Python not found: $PYTHON" >&2; exit 1; }
[[ -f "$ROOT/comparison_models/train_multidataset.py" ]] || exit 1

if [[ ${1:-} != --worker ]]; then
    QUEUE_RUN=$(allocate_run "$QUEUE_BASE")
    SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")
    nohup bash "$SCRIPT" --worker "$QUEUE_RUN" \
        > "$QUEUE_RUN/queue.log" 2>&1 < /dev/null &
    QUEUE_PID=$!
    printf '%s\n' "$QUEUE_PID" > "$QUEUE_RUN/queue.pid"
    printf 'QUEUE_PID=%s\nQUEUE_RUN=%s\nQUEUE_LOG=%s/queue.log\n' \
        "$QUEUE_PID" "$QUEUE_RUN" "$QUEUE_RUN"
    exit 0
fi

QUEUE_RUN=${2:?Missing queue run directory}
mkdir -p "$QUEUE_BASE"
exec 9> "$QUEUE_BASE/.queue.lock"
flock -n 9 || { echo 'Another ROSSA seed-0 comparison queue is active.'; exit 1; }

cd "$ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CHILD_PID=
stop_queue() {
    trap - TERM INT
    if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
    printf 'interrupted\n' > "$QUEUE_RUN/status.txt"
    exit 130
}
trap stop_queue TERM INT

printf 'running\n' > "$QUEUE_RUN/status.txt"
printf 'model\tseed\trun\tstatus\n' > "$QUEUE_RUN/tasks.tsv"
printf '[%s] ROSSA seed=0 queue: %s\n' "$(date '+%F %T')" "${MODELS[*]}"

for MODEL in "${MODELS[@]}"; do
    WAIT_REPORTED=0
    while pgrep -f "^$PYTHON -u -B -m comparison_models[.]train_(3m|multidataset) " >/dev/null; do
        if (( WAIT_REPORTED == 0 )); then
            printf '[%s] Waiting for an existing comparison training process.\n' "$(date '+%F %T')"
            WAIT_REPORTED=1
        fi
        sleep 15
    done

    RUN=$(allocate_run "$ROOT/log/ROSSA/comparison/$MODEL/seed_0")
    printf '[%s] START %s seed=0 RUN=%s\n' "$(date '+%F %T')" "$MODEL" "$RUN"
    printf '%s\t0\t%s\trunning\n' "$MODEL" "$RUN" >> "$QUEUE_RUN/tasks.tsv"
    "$PYTHON" -u -B -m comparison_models.train_multidataset \
        --dataset ROSSA --model "$MODEL" --seed 0 --result-dir "$RUN" \
        > "$RUN/console.log" 2>&1 &
    CHILD_PID=$!
    printf '%s\n' "$CHILD_PID" > "$QUEUE_RUN/current.pid"
    printf '%s\n' "$RUN" > "$QUEUE_RUN/current_run.txt"
    if wait "$CHILD_PID"; then
        CHILD_PID=
        printf '[%s] DONE %s RUN=%s\n' "$(date '+%F %T')" "$MODEL" "$RUN"
        printf '%s\t0\t%s\tcompleted\n' "$MODEL" "$RUN" >> "$QUEUE_RUN/tasks.tsv"
    else
        STATUS=$?
        CHILD_PID=
        printf '[%s] FAILED %s exit=%s LOG=%s/console.log\n' \
            "$(date '+%F %T')" "$MODEL" "$STATUS" "$RUN"
        printf '%s\t0\t%s\tfailed:%s\n' "$MODEL" "$RUN" "$STATUS" >> "$QUEUE_RUN/tasks.tsv"
        printf 'failed:%s:%s\n' "$MODEL" "$STATUS" > "$QUEUE_RUN/status.txt"
        exit "$STATUS"
    fi
done

printf 'completed\n' > "$QUEUE_RUN/status.txt"
printf '[%s] All four ROSSA seed-0 experiments completed.\n' "$(date '+%F %T')"
