import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import importlib
from pathlib import Path
from utils.configuration import add_common_arguments, resolve_configuration, TRAINING_ENGINES


def build_parser():
    parser = argparse.ArgumentParser(description="Train an M2RC-Net configuration from scratch")
    add_common_arguments(parser)
    parser.add_argument("--result-dir", help="New run directory; existing training output is protected")
    return parser


def dispatch_training(config_path, model, run, config):
    module = importlib.import_module(TRAINING_ENGINES[model])
    if TRAINING_ENGINES[model] == "engine.runner":
        return module.train(config_path, model, run, config=config)
    return module.train(config_path, run, config=config)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config_path, model, config = resolve_configuration(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    from engine.runner import config_and_flags, DISPLAY_NAMES
    from utils.logger import next_run, capture_console
    config_and_flags(config, model)
    root = Path(__file__).resolve().parent
    if args.result_dir:
        run = Path(args.result_dir).expanduser().resolve()
        run.mkdir(parents=True, exist_ok=True)
    else:
        run = next_run(root, config["dataset"], DISPLAY_NAMES[model], seed=config["seed"])
    protected = ("metadata.json", "history.jsonl", "model_best.pth", "summary.json", "finished.flag")
    if any((run/name).exists() for name in protected):
        raise RuntimeError(f"Run already contains training output: {run}")
    with capture_console(run):
        print(f"MODEL={model} DATASET={config['dataset']} SEED={config['seed']} RUN={run}", flush=True)
        return dispatch_training(config_path, model, run, config)


if __name__ == "__main__":
    main()
