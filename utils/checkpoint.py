import json
import os
from pathlib import Path
import torch


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_weights(path, state):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, model, weights=None, map_location="cpu"):
    raw = torch.load(path, map_location=map_location)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
        if weights is not None and "loss_weights_state_dict" in raw:
            weights.load_state_dict(raw["loss_weights_state_dict"], strict=True)
    elif isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        state = raw["model"]
    else:
        state = raw
    model.load_state_dict(state, strict=True)
    return raw
