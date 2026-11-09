import json
from pathlib import Path


def next_run(root, dataset, display_name, seed=0):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    group = Path(root) / "log" / dataset / "proposed" / display_name / f"seed_{seed}"
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


def capture_console(run):
    from contextlib import contextmanager
    import os
    import sys

    class Tee:
        def __init__(self, original, stream):
            self.original, self.stream = original, stream
        def write(self, text):
            result = self.original.write(text)
            self.stream.write(text)
            return result
        def flush(self):
            self.original.flush()
            self.stream.flush()
        def __getattr__(self, name):
            return getattr(self.original, name)

    def already_redirected(stream, path):
        try:
            return os.path.samestat(os.fstat(stream.fileno()), path.stat())
        except (AttributeError, OSError, ValueError):
            return False

    @contextmanager
    def capture():
        path = Path(run)/"console.log"
        original_out, original_err = sys.stdout, sys.stderr
        with path.open("a", encoding="utf-8", buffering=1) as log:
            sys.stdout = original_out if already_redirected(original_out, path) else Tee(original_out, log)
            sys.stderr = original_err if already_redirected(original_err, path) else Tee(original_err, log)
            try:
                yield
            except BaseException:
                import traceback
                traceback.print_exc()
                raise
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                sys.stdout, sys.stderr = original_out, original_err
    return capture()
