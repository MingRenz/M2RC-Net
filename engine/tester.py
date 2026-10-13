"""Test uses the same frozen evaluation math; only the selected split changes."""
from .validator import evaluate_base, evaluate_cams, evaluate_mbrc, evaluate_joint


def evaluate_test(variant, model, loader, criterion, device, weights=None):
    if variant == "base":
        loss, dice = evaluate_base(model, loader, criterion, device)
        return {"loss": loss, "dice": dice}
    if variant == "cams":
        losses, dice = evaluate_cams(model, weights, loader, criterion, device, "Test best")
        return {"loss": losses, "dice": dice}
    if variant == "mbrc":
        from models.mbrc.variants import VARIANTS
        return evaluate_mbrc(model, loader, criterion, VARIANTS["full"], device, "test")
    if variant == "cams_mbrcnet":
        return evaluate_joint(model, weights, criterion, loader, device, "TEST BEST")
    raise ValueError(variant)
