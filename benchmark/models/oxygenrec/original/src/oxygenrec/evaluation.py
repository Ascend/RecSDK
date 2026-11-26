"""对生成的 Semantic-ID 候选计算可审计排序指标。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

from .sid import SIDRegistry


@dataclass(frozen=True)
class RankingMetrics:
    """HR/Recall/MRR/NDCG 与合法 SID 比例的聚合结果。"""
    sample_count: int
    legal_sid_rate: float
    hit_rate: Mapping[int, float]
    recall: Mapping[int, float]
    mrr: float
    ndcg: float


@dataclass(frozen=True)
class SIDTokenMatch:
    """展开SID轨迹的逐token命中结果，与EA-TOSD可验reward口径一致。"""

    hits: tuple[bool, ...]
    accuracy: float
    geometric_reward: float


def evaluate_sid_token_match(
    prediction: Sequence[int],
    target: Sequence[int],
    *,
    geometric_decay: float = 0.9,
) -> SIDTokenMatch:
    """计算逐token准确率与归一化几何衰减命中reward。

    这是完整SID/item命中之外的敏感诊断；它不等价于商品命中，
    不能用来替代SID recall或工业推荐指标。
    """
    if not prediction or len(prediction) != len(target):
        raise ValueError("prediction and target must have the same positive length")
    if not 0.0 <= geometric_decay <= 1.0:
        raise ValueError("geometric_decay must be in [0, 1]")
    hits = tuple(left == right for left, right in zip(prediction, target, strict=True))
    weights = tuple(geometric_decay ** index for index in range(len(hits)))
    weight_sum = sum(weights)
    return SIDTokenMatch(
        hits=hits,
        accuracy=sum(hits) / len(hits),
        geometric_reward=sum(
            weight for hit, weight in zip(hits, weights, strict=True) if hit
        ) / weight_sum,
    )


def evaluate_sid_ranking(
    predictions: Sequence[Sequence[Sequence[int]]],
    target_item_ids: Sequence[str],
    registry: SIDRegistry,
    *,
    ks: Sequence[int] = (1, 10),
) -> RankingMetrics:
    """用每个样本的唯一下一商品 target 评测 SID 排序。

    若 SID 发生碰撞，只要 registry 中该 SID 的显式商品集合包含 target 就算命中。
    单一相关 target 协议下 Recall@K 等于 HR@K；两者都返回以保持表格口径明确。
    """

    if not predictions:
        raise ValueError("predictions must not be empty")
    if len(predictions) != len(target_item_ids):
        raise ValueError("predictions and target_item_ids must have equal length")
    normalized_ks = tuple(sorted(set(int(k) for k in ks)))
    if not normalized_ks or normalized_ks[0] < 1:
        raise ValueError("ks must contain positive cutoffs")
    if any(len(ranking) < normalized_ks[-1] for ranking in predictions):
        raise ValueError("every prediction ranking must cover the largest K")

    hits = {k: 0 for k in normalized_ks}
    reciprocal_rank = 0.0
    discounted_gain = 0.0
    legal = 0
    candidate_count = 0
    for ranking, raw_target in zip(predictions, target_item_ids, strict=True):
        target = str(raw_target)
        first_hit: int | None = None
        for rank, codes in enumerate(ranking, start=1):
            # SID 先通过 registry 还原成可能的商品集合，再判断 target 是否在其中。
            items = registry.items_for(codes)
            candidate_count += 1
            if items:
                legal += 1
            if first_hit is None and target in items:
                first_hit = rank
        if first_hit is not None:
            reciprocal_rank += 1.0 / first_hit
            discounted_gain += 1.0 / math.log2(first_hit + 1)
            for k in normalized_ks:
                if first_hit <= k:
                    hits[k] += 1

    count = len(predictions)
    rates = MappingProxyType({k: hits[k] / count for k in normalized_ks})
    return RankingMetrics(
        sample_count=count,
        legal_sid_rate=legal / candidate_count,
        hit_rate=rates,
        recall=MappingProxyType(dict(rates)),
        mrr=reciprocal_rank / count,
        ndcg=discounted_gain / count,
    )
