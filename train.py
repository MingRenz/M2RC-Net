"""Unified training entry for all four model variants and all three datasets."""
import argparse
from engine.runner import train, VARIANT_FLAGS, resolve_variant

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=tuple(VARIANT_FLAGS))
    parser.add_argument("--result-dir")
    args = parser.parse_args()
    train(args.config, resolve_variant(args.config, args.variant), args.result_dir)
