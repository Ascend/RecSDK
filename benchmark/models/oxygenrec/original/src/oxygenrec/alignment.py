"""Public, auditable core of OxygenREC SA-GCPO post-training."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SAGCPOOutput:
    loss: Tensor
    objective: Tensor
    normalized_advantage: Tensor
    thresholded_advantage: Tensor
    importance_ratio: Tensor
    gradient_weight: Tensor


def normalized_group_advantage(rewards: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Equation 9, normalized independently within every generated group."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape [batch, group>=2]")
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    return centered / rewards.std(dim=1, keepdim=True, unbiased=False).clamp_min(epsilon)


def sa_gcpo_loss(
    current_log_probs: Tensor,
    old_log_probs: Tensor,
    rewards: Tensor,
    target_rewards: Tensor,
    *,
    token_mask: Tensor | None = None,
    tau_positive: float = 2.0,
    tau_negative: float = 5.0,
) -> SAGCPOOutput:
    """Implement paper Equations 4-9 as a maximization objective/loss pair."""
    if current_log_probs.shape != old_log_probs.shape or current_log_probs.ndim != 3:
        raise ValueError("log probabilities must share shape [batch, group, tokens]")
    batch, group, _ = current_log_probs.shape
    if rewards.shape != (batch, group) or target_rewards.shape != (batch,):
        raise ValueError("reward shapes must be [batch, group] and [batch]")
    if tau_positive <= 0 or tau_negative <= 0:
        raise ValueError("temperatures must be positive")
    if token_mask is None:
        token_mask = torch.ones_like(current_log_probs, dtype=torch.bool)
    if token_mask.shape != current_log_probs.shape or token_mask.dtype != torch.bool:
        raise ValueError("token_mask must be boolean and match log probabilities")
    if (~token_mask).all(dim=-1).any():
        raise ValueError("every sequence must contain at least one valid token")

    advantage = normalized_group_advantage(rewards)
    suppress = (advantage > 0) & (rewards < target_rewards.unsqueeze(1))
    thresholded = advantage.masked_fill(suppress, 0.0)
    ratio = torch.exp(current_log_probs - old_log_probs)
    tau = torch.where(
        thresholded.unsqueeze(-1) > 0,
        ratio.new_tensor(tau_positive),
        ratio.new_tensor(tau_negative),
    )
    probability = torch.sigmoid(tau * (ratio - 1.0))
    soft_gate = probability * 4.0 / tau
    valid = token_mask.to(ratio.dtype)
    per_sequence = (soft_gate * valid).sum(dim=-1) / valid.sum(dim=-1)
    objective = (per_sequence * thresholded).mean()
    gradient_weight = 4.0 * probability * (1.0 - probability)
    return SAGCPOOutput(
        loss=-objective,
        objective=objective,
        normalized_advantage=advantage,
        thresholded_advantage=thresholded,
        importance_ratio=ratio,
        gradient_weight=gradient_weight,
    )
