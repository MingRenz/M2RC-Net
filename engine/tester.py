from .validator import evaluate_base, evaluate_representation, evaluate_mbrc, evaluate_joint


def evaluate_test(variant, model, loader, criterion, device, weights=None):
    if variant in ("base", "mscf"):
        loss, dice = evaluate_base(model, loader, criterion, device)
        return {"loss": loss, "dice": dice}
    if variant == "mscf_cve":
        losses, dice = evaluate_representation(model, weights, loader, criterion, device, "Test best")
        return {"loss": losses, "dice": dice}
    if variant in ("mbrc", "m2rcnet_wo_cve"):
        from models.mbrc import FULL_MBRC
        return evaluate_mbrc(model, loader, criterion, FULL_MBRC, device, "test")
    if variant == "m2rcnet":
        return evaluate_joint(model, weights, criterion, loader, device, "TEST BEST")
    raise ValueError(variant)
