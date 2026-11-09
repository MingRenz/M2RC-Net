import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from pathlib import Path
from utils.configuration import add_common_arguments, resolve_configuration


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate an explicit Val-best checkpoint")
    add_common_arguments(parser)
    parser.add_argument("--checkpoint", required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config_path, variant, config = resolve_configuration(args)
        checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    import torch
    from torch.utils.data import DataLoader
    from datasets import build_dataset
    from engine.runner import config_and_flags, EXPECTED_PARAMS
    from engine.tester import evaluate_test
    from losses.segmentation import DiceLoss
    from losses.representation import LossWeights, LOSS_CONFIG
    from models import build_model
    from utils.checkpoint import load_checkpoint
    from utils.deterministic import seed_all

    config, (use_mscf, use_cve, use_mbrc) = config_and_flags(config, variant)
    seed_all(config["seed"])
    device = torch.device("cuda:0")
    model = build_model(use_mscf=use_mscf, use_cve=use_cve,
                        use_mbrc=use_mbrc, seed=config["seed"]).to(device)
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMS[variant]
    weights = LossWeights(LOSS_CONFIG).to(device) if use_cve else None
    load_checkpoint(checkpoint, model, weights)
    dataset = build_dataset(config["dataset"], config["data_root"], "test", use_cve,
                            config["map_root"] if use_cve else None, seed=config["seed"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    result = evaluate_test(variant, model, loader, DiceLoss(), device, weights)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
