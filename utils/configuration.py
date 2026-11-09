import copy
from pathlib import Path
import yaml

MODEL_FLAGS = {
    "base": (False, False, False),
    "mscf": (True, False, False),
    "mscf_cve": (True, True, False),
    "mbrc": (False, False, True),
    "m2rcnet_wo_cve": (True, False, True),
    "m2rcnet": (True, True, True),
}
DATASETS = {
    "3m": ("OCTA500_3M", "octa500_3m.yaml"),
    "6m": ("OCTA500_6M", "octa500_6m.yaml"),
    "rossa": ("ROSSA", "rossa.yaml"),
}
TRAINING_ENGINES = {
    model: ("engine.mscf" if model == "mscf" else
            "engine.m2rcnet_without_cve" if model == "m2rcnet_wo_cve" else
            "engine.runner")
    for model in MODEL_FLAGS
}


def dataset_alias(value):
    key = str(value).lower().replace("-", "_")
    aliases = {"octa500_3m": "3m", "octa500_6m": "6m"}
    key = aliases.get(key, key)
    if key not in DATASETS:
        raise ValueError("dataset must be 3m, 6m, or rossa")
    return key


def read_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Config must contain a mapping")
    return config


def add_common_arguments(parser):
    parser.add_argument("--model", choices=tuple(MODEL_FLAGS))
    parser.add_argument("--variant", choices=tuple(MODEL_FLAGS),
                        help="Alias for --model")
    parser.add_argument("--dataset", type=dataset_alias, choices=tuple(DATASETS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", help="Optional YAML config; CLI choices take precedence")
    parser.add_argument("--data-root", help="Existing dataset root; no data are copied")
    parser.add_argument("--map-root", help="Existing CVE supervision cache root")


def resolve_configuration(args, project_root=None):
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    if args.model and args.variant and args.model != args.variant:
        raise ValueError("--model and --variant disagree")
    if args.config:
        path = Path(args.config).expanduser().resolve(strict=True)
    elif args.dataset:
        path = root / "configs" / DATASETS[args.dataset][1]
    else:
        raise ValueError("Specify --dataset or --config")
    config = copy.deepcopy(read_config(path))
    config_dataset = dataset_alias(config["dataset"])
    if args.dataset and args.dataset != config_dataset:
        raise ValueError("--dataset disagrees with the supplied config")
    config["dataset"] = DATASETS[config_dataset][0]
    model = args.model or args.variant
    if model is None:
        if not args.config:
            raise ValueError("Specify --model when selecting a dataset without --config")
        flags = tuple(bool(config["model"][key]) for key in
                      ("use_mscf", "use_cve", "use_mbrc"))
        model = next((name for name, value in MODEL_FLAGS.items() if value == flags), None)
        if model is None:
            raise ValueError("Config contains an unsupported model combination")
    if args.seed is not None:
        config["seed"] = args.seed
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 4294967295]")
    config["model"] = dict(zip(("use_mscf", "use_cve", "use_mbrc"), MODEL_FLAGS[model]))
    for key, override in (("data_root", args.data_root), ("map_root", args.map_root)):
        value = override if override is not None else config.get(key)
        if value is None:
            if key == "map_root" and not MODEL_FLAGS[model][1]:
                continue
            raise ValueError(f"Missing config field: {key}")
        p = Path(value).expanduser()
        config[key] = str(p.resolve() if p.is_absolute() else (root/p).resolve())
    return path, model, config
