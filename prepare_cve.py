import argparse
import json
from pathlib import Path

from utils.configuration import DATASETS, dataset_alias


def build_parser():
    parser = argparse.ArgumentParser(description="Generate CVE caliber supervision on CPU from frozen GT splits")
    parser.add_argument("--dataset", type=dataset_alias, choices=tuple(DATASETS), required=True)
    parser.add_argument("--data-root", help="Dataset root; defaults to data/<dataset> under this project")
    parser.add_argument("--output-dir", required=True, help="New or empty cache directory; never overwrites existing files")
    parser.add_argument("--workers", type=int, default=1, help="Independent CPU workers; default: 1")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    dataset = DATASETS[args.dataset][0]
    data = args.data_root or str(Path(__file__).resolve().parent/"data"/dataset)
    from preprocessing.cve_cache import prepare_cache
    result = prepare_cache(dataset, data, args.output_dir, workers=args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
