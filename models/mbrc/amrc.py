import torch
import torch.nn.functional as F

def decide(mbrc, features):
    action_logits = mbrc.action_head(features)
    action_prob = torch.softmax(action_logits, dim=1)
    magnitude = F.softplus(mbrc.magnitude_head(features))
    delta = action_prob[:, 0:1] * magnitude[:, 0:1] - action_prob[:, 1:2] * magnitude[:, 1:2]
    return (delta, action_logits)
