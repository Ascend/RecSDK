"""GPU/NPU固定验证集比较使用的纯Python指标与指纹工具。"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Callable, Sequence

from .evaluation import evaluate_sid_token_match


FlatSIDList = Sequence[int]
BeamSIDLists = Sequence[Sequence[int]]


def canonical_sha256(value: object) -> str:
    """对JSON可序列化对象生成与字段顺序无关的稳定指纹。"""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sid_items(path: FlatSIDList, levels: int) -> list[tuple[int, ...]]:
    if levels < 1 or not path or len(path) % levels:
        raise ValueError("a flattened SID list must contain complete positive SID items")
    return [
        tuple(int(code) for code in path[start : start + levels])
        for start in range(0, len(path), levels)
    ]


def summarize_listwise_validation(
    targets: Sequence[FlatSIDList],
    greedy: Sequence[FlatSIDList],
    beams: Sequence[BeamSIDLists],
    *,
    sid_levels: int,
    is_legal: Callable[[Sequence[int]], bool],
    geometric_decay: float = 0.9,
) -> dict[str, object]:
    """汇总v2列表任务指标；beam排序相关指标以完整目标列表命中为准。"""

    if not targets or len(targets) != len(greedy) or len(targets) != len(beams):
        raise ValueError("targets, greedy and beams must have the same positive size")
    beam_widths = {len(ranking) for ranking in beams}
    if len(beam_widths) != 1 or not beam_widths or next(iter(beam_widths)) < 1:
        raise ValueError("every sample must have the same positive beam width")
    beam_width = next(iter(beam_widths))

    target_count = overlap = position_hits = exact_lists = 0
    unique_targets = legal_greedy = legal_beam = 0
    greedy_items = beam_items = token_hits = token_count = 0
    geometric_reward = reciprocal_rank = discounted_gain = 0.0
    hit_at_one = hit_at_width = 0
    list_size: int | None = None

    for target_path, greedy_path, beam_ranking in zip(
        targets, greedy, beams, strict=True
    ):
        target = [int(code) for code in target_path]
        predicted = [int(code) for code in greedy_path]
        if len(target) != len(predicted):
            raise ValueError("target and greedy SID lists must have equal length")
        target_sids = _sid_items(target, sid_levels)
        predicted_sids = _sid_items(predicted, sid_levels)
        if list_size is None:
            list_size = len(target_sids)
        if len(target_sids) != list_size:
            raise ValueError("all target lists must use the same list size")

        target_set = set(target_sids)
        predicted_set = set(predicted_sids)
        target_count += len(target_sids)
        unique_targets += len(target_set)
        overlap += len(target_set & predicted_set)
        position_hits += sum(
            left == right
            for left, right in zip(predicted_sids, target_sids, strict=True)
        )
        exact_lists += int(predicted == target)
        legal_greedy += sum(is_legal(item) for item in predicted_sids)
        greedy_items += len(predicted_sids)

        token_match = evaluate_sid_token_match(
            predicted,
            target,
            geometric_decay=geometric_decay,
        )
        token_hits += sum(token_match.hits)
        token_count += len(token_match.hits)
        geometric_reward += token_match.geometric_reward

        first_hit: int | None = None
        for rank, beam_path in enumerate(beam_ranking, start=1):
            normalized_beam = [int(code) for code in beam_path]
            if len(normalized_beam) != len(target):
                raise ValueError("beam and target SID lists must have equal length")
            beam_sids = _sid_items(normalized_beam, sid_levels)
            legal_beam += sum(is_legal(item) for item in beam_sids)
            beam_items += len(beam_sids)
            if first_hit is None and normalized_beam == target:
                first_hit = rank
        if first_hit is not None:
            reciprocal_rank += 1.0 / first_hit
            discounted_gain += 1.0 / math.log2(first_hit + 1)
            hit_at_width += 1
            hit_at_one += int(first_hit == 1)

    sample_count = len(targets)
    assert list_size is not None
    return {
        "lists": sample_count,
        "list_size": list_size,
        "beam_width": beam_width,
        "target_sid_unique_rate": unique_targets / target_count,
        "sid_recall": overlap / target_count,
        "position_accuracy": position_hits / target_count,
        "exact_list_rate": exact_lists / sample_count,
        "beam_exact_list_hit_rate": {
            "1": hit_at_one / sample_count,
            str(beam_width): hit_at_width / sample_count,
        },
        "beam_exact_list_mrr": reciprocal_rank / sample_count,
        "beam_exact_list_ndcg": discounted_gain / sample_count,
        "greedy_legal_item_rate": legal_greedy / greedy_items,
        "beam_legal_item_rate": legal_beam / beam_items,
        "sid_token_accuracy": token_hits / token_count,
        "geometric_token_reward": geometric_reward / sample_count,
    }