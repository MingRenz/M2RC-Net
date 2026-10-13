"""Action/Magnitude or directly signed residual prediction."""

import torch
from torch import nn
import torch.nn.functional as F

from .mbrc_v2 import cna


def _shrink_first(head):
    old = head[0]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20261004)
        new = cna(32, 32, 3, 8)
    with torch.no_grad():
        new[0].weight.copy_(old[0].weight[:, :32])
        new[1].weight.copy_(old[1].weight)
        new[1].bias.copy_(old[1].bias)
    head[0] = new


def set_decision(mbrc, use_action, use_boundary):
    mbrc.use_action = use_action
    if use_action:
        if not use_boundary:
            _shrink_first(mbrc.action_head)
            _shrink_first(mbrc.magnitude_head)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261004)
            last = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        # Reuse the original Action feature extractor's initial weights.
        mbrc.signed_head = nn.Sequential(
            mbrc.action_head[0], mbrc.action_head[1], last
        )
        del mbrc.action_head
        del mbrc.magnitude_head
        if not use_boundary:
            _shrink_first(mbrc.signed_head)


def decide(mbrc, features):
    if mbrc.use_action:
        action_logits = mbrc.action_head(features)
        action_prob = torch.softmax(action_logits, dim=1)
        magnitude = F.softplus(mbrc.magnitude_head(features))
        delta = (
            action_prob[:, 0:1] * magnitude[:, 0:1]
            - action_prob[:, 1:2] * magnitude[:, 1:2]
        )
        return delta, action_logits
    return mbrc.signed_head(features), None
