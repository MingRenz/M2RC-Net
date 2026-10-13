"""Run directory allocation and append-only experiment history."""
import json
from pathlib import Path


def next_run(root, dataset, display_name):
    group = Path(root) / "log" / dataset / display_name
    group.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        run = group / f"run_{index:03d}"
        try:
            run.mkdir()
            return run
        except FileExistsError:
            index += 1


def append_history(run, record):
    with (Path(run) / "history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_record(record, best_test):
    val_loss = record["val_loss"]
    if isinstance(val_loss, dict):
        val_loss = val_loss["total"]
    print(
        f"epoch={record['epoch']} train_loss={record['train_loss']:.6f} "
        f"val_loss={val_loss:.6f} "
        f"current_val_dice={record['current_val_dice']:.9f} "
        f"saved_best_epoch={record['saved_best_epoch']} "
        f"saved_best_val_dice={record['saved_best_val_dice']:.9f} "
        f"saved_best_test_dice={best_test:.9f} "
        f"saved_this_epoch={'YES' if record['saved_this_epoch'] else 'NO'} "
        f"epoch_seconds={record['epoch_seconds']:.1f}",
        flush=True,
    )
