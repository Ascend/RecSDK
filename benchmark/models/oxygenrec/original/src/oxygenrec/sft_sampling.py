"""为 Reasoning SFT 按历史证据分层抽样，避免随机样本被纯浏览行为淹没。"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from .data.temporal import NextItemSample


SFT_EVIDENCE_COHORTS = (
    "transaction_history", "cart_history", "repeat_view", "browse_only",
)


def evidence_cohort(sample: NextItemSample) -> str:
    """只根据 target 之前的 history 分类，不使用目标行为或目标商品。"""

    behaviors = {event.behavior.value for event in sample.history}
    if "transaction" in behaviors:
        return "transaction_history"
    if "addtocart" in behaviors:
        return "cart_history"
    item_counts = Counter(event.item_id for event in sample.history)
    if any(count > 1 for count in item_counts.values()):
        return "repeat_view"
    return "browse_only"


def select_stratified_sft_samples(
    samples: Sequence[NextItemSample], cases: int,
) -> tuple[list[NextItemSample], dict[str, int]]:
    """尽量等额选择四类历史证据；稀缺分层不足时再由剩余样本补齐。"""

    if cases < 1:
        raise ValueError("cases must be positive")
    buckets = {name: [] for name in SFT_EVIDENCE_COHORTS}
    for sample in samples:
        buckets[evidence_cohort(sample)].append(sample)
    base, remainder = divmod(cases, len(SFT_EVIDENCE_COHORTS))
    selected = []
    selected_ids: set[int] = set()
    for index, name in enumerate(SFT_EVIDENCE_COHORTS):
        quota = base + int(index < remainder)
        for sample in buckets[name][:quota]:
            selected.append(sample)
            selected_ids.add(id(sample))
    if len(selected) < cases:
        for sample in samples:
            if id(sample) in selected_ids:
                continue
            selected.append(sample)
            selected_ids.add(id(sample))
            if len(selected) == cases:
                break
    if len(selected) != cases:
        raise ValueError(f"requested {cases} cases but only {len(selected)} are available")
    counts = Counter(evidence_cohort(sample) for sample in selected)
    return selected, {name: counts[name] for name in SFT_EVIDENCE_COHORTS}
