"""One real optimizer step per 3M variant; run in two fresh processes to compare."""
import hashlib
import json
from pathlib import Path

import torch

from engine.runner import VARIANT_FLAGS, build_training_objects, config_and_flags, epoch_train
from utils.deterministic import seed_all

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/octa500_3m_seed3.yaml"


class OneBatch:
    def __init__(self, batch):
        self.batch = batch

    def __len__(self):
        return 1

    def __iter__(self):
        yield self.batch


def state_hash(model, weights):
    digest = hashlib.sha256()
    for prefix, module in (("model", model), ("weights", weights)):
        if module is None:
            continue
        for key, tensor in module.state_dict().items():
            digest.update((prefix + "." + key).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    output = {}
    device = torch.device("cuda:0")
    for variant in VARIANT_FLAGS:
        config, _ = config_and_flags(CONFIG, variant)
        seed_all(config["seed"])
        _datasets, loaders, model, weights, criterion, optimizer = build_training_objects(
            config, variant, device
        )
        initial = state_hash(model, weights)
        iterator = iter(loaders["train"])
        batch = next(iterator)
        del iterator
        first_names = list(batch[0])
        loss = epoch_train(
            variant, model, weights, criterion, optimizer,
            OneBatch(batch), device, 1,
        )
        torch.cuda.synchronize()
        output[variant] = {
            "initial_state": initial,
            "first_names": first_names,
            "one_step_loss": loss,
            "after_step_state": state_hash(model, weights),
        }
        del loaders, model, weights, criterion, optimizer
        torch.cuda.empty_cache()
    print("REPLAY_JSON=" + json.dumps(output, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
