import torch.nn.functional as F

from models.mbrc.bfg import inner_boundary
from .representation import LOSS_CONFIG, caliber_loss
from .mbrc import balanced_action_loss, dynamic_actions


def compute_parts(output, gt, caliber_gt, valid, loss_weights, dice):
    base_dice = dice(output["base_prob"], gt)
    caliber, _, _ = caliber_loss(
        output["caliber"], caliber_gt, gt, valid, LOSS_CONFIG
    )
    vesselness = F.binary_cross_entropy_with_logits(
        output["vesselness_logits"], gt
    )
    representation = loss_weights(base_dice, caliber, vesselness)
    final = dice(output["final_prob"], gt)
    actions = dynamic_actions(output["base_prob"], gt)
    action = balanced_action_loss(output["action_logits"], actions, gt)
    boundary = F.binary_cross_entropy_with_logits(
        output["edge_logits"], inner_boundary(gt)
    )
    total = representation + final + 0.1 * action + 0.1 * boundary
    return {
        "total": total,
        "representation": representation,
        "base_dice": base_dice,
        "caliber": caliber,
        "vesselness": vesselness,
        "final": final,
        "action": action,
        "boundary": boundary,
    }
