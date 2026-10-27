"""Bitwise source-vs-refactor probe on one fixed A40 synthetic batch."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/zfzf/final_final/code")
KINDS = ("base", "cams", "mbrc", "cams_mbrcnet")


def setup(origin, kind):
    if origin == "old":
        location = {
            "base": SOURCE / "clean_ssm_base",
            "cams": SOURCE / "clean_ssm_cams_3m",
            "mbrc": SOURCE / "clean_ssm_mbrc_3m",
            "cams_mbrcnet": SOURCE / "cams_mbrcnet_3m",
        }[kind]
        sys.path.insert(0, str(location))
        if kind == "mbrc":
            sys.path.insert(1, str(SOURCE / "clean_ssm_base"))
        from loss import DiceLoss
        if kind == "base":
            from our_model.clean_ssm import StableCleanSSMBase
            return StableCleanSSMBase, DiceLoss, None, None
        if kind == "cams":
            from our_model.clean_ssm_cams import StableCleanSSMBaseCAMS
            from cams_losses import LossWeights, LOSS_CONFIG, caliber_loss
            return StableCleanSSMBaseCAMS, DiceLoss, (LossWeights, LOSS_CONFIG, caliber_loss), None
        if kind == "mbrc":
            from joint_model import JointCleanSSMMBRC
            from variants import VARIANTS
            from objective import compute_parts
            return lambda: JointCleanSSMMBRC(VARIANTS["full"]), DiceLoss, None, compute_parts
        from joint_model import CAMS_MBRCNet
        from cams_losses import LossWeights, LOSS_CONFIG, caliber_loss
        from joint_objective import compute_parts
        return CAMS_MBRCNet, DiceLoss, (LossWeights, LOSS_CONFIG, caliber_loss), compute_parts
    sys.path.insert(0, str(ROOT))
    from models import build_model
    from losses.segmentation import DiceLoss
    if kind == "base":
        return lambda: build_model(False, False), DiceLoss, None, None
    if kind == "cams":
        from losses.cams import LossWeights, LOSS_CONFIG, caliber_loss
        return lambda: build_model(True, False), DiceLoss, (LossWeights, LOSS_CONFIG, caliber_loss), None
    if kind == "mbrc":
        from losses.mbrc import compute_parts
        return lambda: build_model(False, True), DiceLoss, None, compute_parts
    from losses.cams import LossWeights, LOSS_CONFIG, caliber_loss
    from losses.joint import compute_parts
    return lambda: build_model(True, True), DiceLoss, (LossWeights, LOSS_CONFIG, caliber_loss), compute_parts


def cpu(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu(v) for v in value)
    return value


def probe(origin, kind, output):
    import random
    import numpy as np
    import torch
    import torch.nn.functional as F
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG missing")
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    factory, dice_class, cams_pack, objective = setup(origin, kind)
    model = factory().cuda().train()
    weights = cams_pack[0](cams_pack[1]).cuda() if cams_pack else None
    criterion = dice_class()
    names = [name for name, _ in model.named_parameters()]
    initial = cpu(model.state_dict())
    initial_weights = cpu(weights.state_dict()) if weights else None
    rng = torch.get_rng_state().clone()
    input_generator = torch.Generator().manual_seed(777)
    image = torch.randn((1, 1, 224, 224), generator=input_generator).cuda()
    gt = (torch.rand((1, 1, 224, 224), generator=input_generator) > 0.65).float().cuda()
    caliber = (0.5 * gt).cuda()
    valid = torch.ones_like(gt)
    captures = {}
    backbone = model.base if kind in ("mbrc", "cams_mbrcnet") else model
    if kind in ("cams", "cams_mbrcnet"):
        for stage, module in backbone.guided().items():
            module.register_forward_hook(
                lambda _module, _inputs, value, stage=stage: captures.__setitem__(stage, value.detach().cpu().clone())
            )
    output_value = model(image)
    if kind == "base":
        outputs = {"logits": output_value}
        loss = criterion(torch.sigmoid(output_value), gt)
        parts = {"total": loss}
    elif kind == "cams":
        logits, r, vesselness_logits = output_value
        outputs = {"logits": logits, "R": r, "V": torch.sigmoid(vesselness_logits)}
        lseg = criterion(torch.sigmoid(logits), gt)
        lcal, _, _ = cams_pack[2](r, caliber, gt, valid, cams_pack[1])
        lves = F.binary_cross_entropy_with_logits(vesselness_logits, gt)
        loss = weights(lseg, lcal, lves)
        parts = {"total": loss, "seg": lseg, "cal": lcal, "ves": lves}
    elif kind == "mbrc":
        outputs = {key: value for key, value in output_value.items() if torch.is_tensor(value)}
        if origin == "old":
            from variants import VARIANTS
        else:
            from models.mbrc.variants import VARIANTS
        parts, _, _ = objective(output_value, gt, criterion, VARIANTS["full"])
        loss = parts["total"]
    else:
        outputs = {key: value for key, value in output_value.items() if torch.is_tensor(value)}
        parts = objective(output_value, gt, caliber, valid, weights, criterion)
        loss = parts["total"]
    parameters = list(model.parameters()) + (list(weights.parameters()) if weights else [])
    optimizer = torch.optim.Adam(parameters, lr=1e-4, weight_decay=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = {name: cpu(param.grad) if param.grad is not None else None
             for name, param in model.named_parameters()}
    weight_grads = {name: cpu(param.grad) if param.grad is not None else None
                    for name, param in weights.named_parameters()} if weights else None
    optimizer.step()
    if weights:
        weights.project()
    result = {
        "kind": kind, "origin": origin, "parameters": sum(p.numel() for p in model.parameters()),
        "names": names, "initial": initial, "initial_weights": initial_weights,
        "rng": rng, "outputs": cpu(outputs), "features": captures,
        "losses": cpu(parts), "grads": grads, "weight_grads": weight_grads,
        "updated": cpu(model.state_dict()),
        "updated_weights": cpu(weights.state_dict()) if weights else None,
        "optimizer": cpu(optimizer.state_dict()),
    }
    torch.save(result, output)
    print("PROBE", origin, kind, "OK", flush=True)


def compare(left, right, path="root"):
    import torch
    if type(left) is not type(right):
        raise AssertionError(f"{path}: type {type(left)} != {type(right)}")
    if torch.is_tensor(left):
        if not torch.equal(left, right):
            diff = (left.float()-right.float()).abs().max().item()
            raise AssertionError(f"{path}: tensor differs shape={tuple(left.shape)} maxabs={diff}")
    elif isinstance(left, dict):
        if list(left) != list(right):
            raise AssertionError(f"{path}: keys differ")
        for key in left:
            compare(left[key], right[key], path+"."+str(key))
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"{path}: length differs")
        for i, (a, b) in enumerate(zip(left, right)):
            compare(a, b, path+f"[{i}]")
    elif left != right:
        raise AssertionError(f"{path}: {left!r} != {right!r}")


def run_all():
    import torch
    directory = Path("/tmp/cams_mbrcnet_equivalence")
    directory.mkdir(exist_ok=True)
    report = {}
    for kind in KINDS:
        files = []
        for origin in ("old", "new"):
            output = directory / f"{kind}_{origin}.pt"
            subprocess.run([sys.executable, "-B", str(Path(__file__)), "probe",
                            "--origin", origin, "--variant", kind, "--output", str(output)],
                           env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, check=True)
            files.append(output)
        a, b = (torch.load(path, map_location="cpu") for path in files)
        compare({k:v for k,v in a.items() if k!="origin"},
                {k:v for k,v in b.items() if k!="origin"}, kind)
        report[kind] = {"parameters": a["parameters"], "initial": "bitwise",
                        "forward": "bitwise", "backward": "bitwise", "adam_step": "bitwise"}
        print("BITWISE", kind, "PASS", flush=True)
    (directory / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("probe","all"))
    parser.add_argument("--origin", choices=("old","new"))
    parser.add_argument("--variant", choices=KINDS)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.mode == "probe":
        probe(args.origin, args.variant, args.output)
    else:
        run_all()
