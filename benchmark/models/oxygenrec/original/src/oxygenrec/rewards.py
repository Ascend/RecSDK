"""Scenario-aware public reward mapping proxies for OxygenREC post-training."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RewardWeights:
    format: float = 1.0
    relative: float = 1.0
    ranking: float = 1.0
    diversity: float = 0.2


@dataclass(frozen=True)
class MappedRewards:
    total: Tensor
    format: Tensor
    relative: Tensor
    ranking: Tensor
    diversity: Tensor


def map_public_rewards(
    candidate_sids: Tensor,
    *,
    legal_mask: Tensor,
    relative_scores: Tensor,
    ranking_scores: Tensor,
    weights: RewardWeights = RewardWeights(),
) -> MappedRewards:
    """Map four paper reward families using externally auditable proxy scores."""
    if candidate_sids.ndim != 3:
        raise ValueError("candidate_sids must have shape [batch, group, levels]")
    shape = candidate_sids.shape[:2]
    if legal_mask.shape != shape or relative_scores.shape != shape or ranking_scores.shape != shape:
        raise ValueError("all reward components must have shape [batch, group]")
    format_reward = legal_mask.to(relative_scores.dtype)
    # Candidate-level diversity: mean normalized SID Hamming distance to peers.
    pairwise = (candidate_sids[:, :, None, :] != candidate_sids[:, None, :, :]).to(
        relative_scores.dtype
    ).mean(dim=-1)
    group = candidate_sids.shape[1]
    diversity = (
        (pairwise.sum(dim=-1) - pairwise.diagonal(dim1=1, dim2=2)) / max(group - 1, 1)
    )
    total = (
        weights.format * format_reward
        + weights.relative * relative_scores
        + weights.ranking * ranking_scores
        + weights.diversity * diversity
    )
    return MappedRewards(total, format_reward, relative_scores, ranking_scores, diversity)
