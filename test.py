"""Unified test entry; loads a selected checkpoint and uses the frozen Otsu protocol."""
import argparse
import json
import torch
from torch.utils.data import DataLoader
from datasets import build_dataset
from engine.runner import config_and_flags, EXPECTED_PARAMS, resolve_variant
from engine.tester import evaluate_test
from losses.segmentation import DiceLoss
from losses.cams import LossWeights, LOSS_CONFIG
from models import build_model
from utils.checkpoint import load_legacy_checkpoint
from utils.deterministic import seed_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=("base","cams","mbrc","cams_mbrcnet"))
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    args.variant = resolve_variant(args.config, args.variant)
    config, (use_cams, use_mbrc) = config_and_flags(args.config, args.variant)
    seed_all(config["seed"])
    device = torch.device("cuda:0")
    model = build_model(use_cams, use_mbrc, seed=config["seed"]).to(device)
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMS[args.variant]
    weights = LossWeights(LOSS_CONFIG).to(device) if use_cams else None
    load_legacy_checkpoint(args.checkpoint, model, weights)
    dataset = build_dataset(
        config["dataset"], config["data_root"], "test", use_cams,
        config["map_root"] if use_cams else None, seed=config["seed"],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    result = evaluate_test(args.variant, model, loader, DiceLoss(), device, weights)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
