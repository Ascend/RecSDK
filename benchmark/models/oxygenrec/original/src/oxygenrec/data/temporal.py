"""防止未来信息泄漏的时间切分与下一商品样本构造。"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from enum import Enum
import random
from typing import Iterable, Mapping, Sequence

from .events import Behavior, InteractionEvent


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class TemporalBoundaries:
    """所有用户和商品共享的全局时间边界，避免逐用户切分带来的口径漂移。"""

    train_end_ms: int
    validation_end_ms: int

    def __post_init__(self) -> None:
        if self.train_end_ms < 0:
            raise ValueError("train_end_ms must be non-negative")
        if self.validation_end_ms <= self.train_end_ms:
            raise ValueError("validation_end_ms must be greater than train_end_ms")

    def split_for(self, timestamp_ms: int) -> Split:
        """根据事件时间返回 train/validation/test。"""
        if timestamp_ms < self.train_end_ms:
            return Split.TRAIN
        if timestamp_ms < self.validation_end_ms:
            return Split.VALIDATION
        return Split.TEST


@dataclass(frozen=True)
class NextItemSample:
    """一个自回归下一商品样本；history 中每个事件都必须严格早于 target。"""

    split: Split
    user_id: str
    history: tuple[InteractionEvent, ...]
    target: InteractionEvent

    def __post_init__(self) -> None:
        if not self.history:
            raise ValueError("history must not be empty")
        if any(event.user_id != self.user_id for event in self.history):
            raise ValueError("all history events must belong to the sample user")
        if self.target.user_id != self.user_id:
            raise ValueError("target must belong to the sample user")
        if any(event.timestamp_ms >= self.target.timestamp_ms for event in self.history):
            raise ValueError("history must be strictly earlier than the target")


@dataclass(frozen=True)
class ListwiseTargetSample:
    """同一用户、日期和目标行为下的固定长度列表式训练样本。"""

    split: Split
    user_id: str
    history: tuple[InteractionEvent, ...]
    targets: tuple[InteractionEvent, ...]
    utc_day: int
    # EA-TOSD Teacher训练期可见；默认空元组保持普通listwise预训练接口。
    future_targets: tuple[InteractionEvent, ...] = ()

    def __post_init__(self) -> None:
        if not self.history:
            raise ValueError("history must not be empty")
        if not self.targets:
            raise ValueError("targets must not be empty")
        if any(event.user_id != self.user_id for event in self.history + self.targets):
            raise ValueError("all events must belong to the sample user")
        if len({event.item_id for event in self.targets}) != len(self.targets):
            raise ValueError("listwise targets must contain distinct items")
        behaviors = {event.behavior for event in self.targets}
        if len(behaviors) != 1:
            raise ValueError("one listwise sample must use one target behavior instruction")
        first_target_time = min(event.timestamp_ms for event in self.targets)
        if any(event.timestamp_ms >= first_target_time for event in self.history):
            raise ValueError("history must be strictly earlier than every list target")
        if any(event.timestamp_ms // 86_400_000 != self.utc_day for event in self.targets):
            raise ValueError("all targets must belong to the declared UTC day")
        last_target_time = max(event.timestamp_ms for event in self.targets)
        if any(event.user_id != self.user_id for event in self.future_targets):
            raise ValueError("all future targets must belong to the sample user")
        if any(event.timestamp_ms <= last_target_time for event in self.future_targets):
            raise ValueError("future targets must be strictly later than every list target")
        if tuple(sorted(self.future_targets)) != self.future_targets:
            raise ValueError("future targets must stay in chronological order")

    @property
    def target_behavior(self) -> Behavior:
        """返回该列表共同使用的目标行为标签。"""
        return self.targets[0].behavior

    def without_privileged_future(self) -> "ListwiseTargetSample":
        """返回部署Student可见的样本视图，显式移除Teacher未来信息。"""
        if not self.future_targets:
            return self
        return replace(self, future_targets=())


def training_item_ids(
    events: Iterable[InteractionEvent], boundaries: TemporalBoundaries
) -> frozenset[str]:
    """只用训练截止时间之前出现的商品建立词表。"""

    return frozenset(
        event.item_id
        for event in events
        if boundaries.split_for(event.timestamp_ms) is Split.TRAIN
    )

    """
    按照用户和时间遍历：
    NextItemSample(
        split,
        user_id,
        history=(event_1, ..., event_n),
        target=event_n+1,
    )
    """
def build_next_item_samples(
    events: Iterable[InteractionEvent],
    boundaries: TemporalBoundaries,
    *,
    target_behaviors: Sequence[Behavior] = (
        Behavior.VIEW,
        Behavior.ADD_TO_CART,
        Behavior.TRANSACTION,
    ),
    min_history: int = 1,
    max_history: int | None = None,
    require_target_in_training_items: bool = True,
    max_samples_per_split: Mapping[Split, int] | None = None,
    sample_seed: int = 0,
) -> list[NextItemSample]:
    """按时间构造下一商品样本，并阻断未来事件泄漏。

    validation/test 的目标可以使用所有严格早于自身的历史，即历史允许跨越 split
    边界；target 本身绝不会进入 history。同一毫秒的事件因顺序未知，也互不可见。
    """

    if min_history < 1:
        raise ValueError("min_history must be at least 1")
    if max_history is not None and max_history < min_history:
        raise ValueError("max_history must be at least min_history")
    limits = dict(max_samples_per_split or {})
    if any(limit < 1 for limit in limits.values()):
        raise ValueError("max_samples_per_split limits must be positive")

    # InteractionEvent 按 timestamp/source_row 排序，保证每次构造结果一致。
    ordered_events = sorted(events)
    train_items = training_item_ids(ordered_events, boundaries)
    targets = frozenset(target_behaviors)
    by_user: dict[str, list[InteractionEvent]] = defaultdict(list)
    for event in ordered_events:
        by_user[event.user_id].append(event)

    samples: list[NextItemSample] = []
    reservoirs: dict[Split, list[NextItemSample]] = defaultdict(list)
    seen_by_split: Counter[Split] = Counter()
    generators = {
        split: random.Random(sample_seed + index)
        for index, split in enumerate(Split)
    }
    for user_id, user_events in sorted(by_user.items()):
        history: list[InteractionEvent] = []
        cursor = 0
        while cursor < len(user_events):
            timestamp = user_events[cursor].timestamp_ms
            group_end = cursor + 1
            while (
                group_end < len(user_events)
                and user_events[group_end].timestamp_ms == timestamp
            ):
                group_end += 1

            # 先基于旧 history 构造这一时间组的所有 target，之后才整体加入 history。
            same_time_events = user_events[cursor:group_end]
            if len(history) >= min_history:
                selected_history = history[-max_history:] if max_history else history
                for target in same_time_events:
                    if target.behavior not in targets:
                        continue
                    if require_target_in_training_items and target.item_id not in train_items:
                        continue
                    sample = NextItemSample(
                        split=boundaries.split_for(target.timestamp_ms),
                        user_id=user_id,
                        history=tuple(selected_history),
                        target=target,
                    )
                    if sample.split not in limits:
                        samples.append(sample)
                    else:
                        seen_by_split[sample.split] += 1
                        bucket = reservoirs[sample.split]
                        limit = limits[sample.split]
                        if len(bucket) < limit:
                            bucket.append(sample)
                        else:
                            # 确定性 reservoir sampling：限制样本量但不偏向文件前部。
                            replacement = generators[sample.split].randrange(
                                seen_by_split[sample.split]
                            )
                            if replacement < limit:
                                bucket[replacement] = sample

            history.extend(same_time_events)
            cursor = group_end

    for split in Split:
        samples.extend(reservoirs[split])
    return samples


def build_daily_listwise_samples(
    events: Iterable[InteractionEvent],
    boundaries: TemporalBoundaries,
    *,
    list_size: int,
    target_behaviors: Sequence[Behavior] = (
        Behavior.VIEW,
        Behavior.ADD_TO_CART,
        Behavior.TRANSACTION,
    ),
    min_history: int = 1,
    max_history: int | None = None,
    require_target_in_training_items: bool = True,
    max_samples_per_split: Mapping[Split, int] | None = None,
    sample_seed: int = 0,
    max_future_targets: int = 0,
    minimum_future_behavior: Behavior = Behavior.VIEW,
    require_future_targets: bool = False,
) -> list[ListwiseTargetSample]:
    """构造公开数据代理的daily、行为同质列表式样本。

    同一用户、UTC日和split内，同一商品若出现多种行为，只保留意图最强的
    ``transaction > addtocart > view``。随后按行为分别排序，并把相邻目标组成
    固定长度N的列表。每个列表使用首个目标之前的历史快照，因此列表内任何目标
    都不会进入Encoder输入；不足N的尾部不会用padding伪造监督。

    ``max_future_targets=M``时，还会从列表最后一个真实目标之后按时间扫描同用户、
    同split事件，保留行为强度不低于``minimum_future_behavior``的至多M个目标，
    作为EA-TOSD Teacher训练期前缀F。严格限制在同split，避免train样本读取
    validation/test事件。这是对论文私有daily日志协议的可审计代理，不声称
    还原其未公开分组或near-future时间窗细节。
    """

    if list_size < 1:
        raise ValueError("list_size must be positive")
    if max_future_targets < 0:
        raise ValueError("max_future_targets cannot be negative")
    if require_future_targets and max_future_targets < 1:
        raise ValueError("require_future_targets needs max_future_targets >= 1")
    if min_history < 1:
        raise ValueError("min_history must be at least 1")
    if max_history is not None and max_history < min_history:
        raise ValueError("max_history must be at least min_history")
    limits = dict(max_samples_per_split or {})
    if any(limit < 1 for limit in limits.values()):
        raise ValueError("max_samples_per_split limits must be positive")

    ordered_events = sorted(events)
    train_items = training_item_ids(ordered_events, boundaries)
    targets = frozenset(target_behaviors)
    priority = {
        Behavior.VIEW: 0,
        Behavior.ADD_TO_CART: 1,
        Behavior.TRANSACTION: 2,
    }
    if minimum_future_behavior not in priority:
        raise ValueError("minimum_future_behavior is unsupported")
    by_user: dict[str, list[InteractionEvent]] = defaultdict(list)
    for event in ordered_events:
        by_user[event.user_id].append(event)

    samples: list[ListwiseTargetSample] = []
    reservoirs: dict[Split, list[ListwiseTargetSample]] = defaultdict(list)
    seen_by_split: Counter[Split] = Counter()
    generators = {
        split: random.Random(sample_seed + index)
        for index, split in enumerate(Split)
    }

    def keep(sample: ListwiseTargetSample) -> None:
        """按split做确定性reservoir采样，避免只保留时间靠前的用户。"""
        if sample.split not in limits:
            samples.append(sample)
            return
        seen_by_split[sample.split] += 1
        bucket = reservoirs[sample.split]
        limit = limits[sample.split]
        if len(bucket) < limit:
            bucket.append(sample)
            return
        replacement = generators[sample.split].randrange(seen_by_split[sample.split])
        if replacement < limit:
            bucket[replacement] = sample

    for user_id, user_events in sorted(by_user.items()):
        user_timestamps = [event.timestamp_ms for event in user_events]
        history: list[InteractionEvent] = []
        grouped: dict[
            tuple[int, Split],
            list[tuple[InteractionEvent, tuple[InteractionEvent, ...]]],
        ] = defaultdict(list)
        cursor = 0
        while cursor < len(user_events):
            timestamp = user_events[cursor].timestamp_ms
            group_end = cursor + 1
            while (
                group_end < len(user_events)
                and user_events[group_end].timestamp_ms == timestamp
            ):
                group_end += 1
            same_time_events = user_events[cursor:group_end]
            selected_history = history[-max_history:] if max_history else history
            if len(selected_history) >= min_history:
                snapshot = tuple(selected_history)
                for target in same_time_events:
                    if target.behavior not in targets:
                        continue
                    if require_target_in_training_items and target.item_id not in train_items:
                        continue
                    split = boundaries.split_for(target.timestamp_ms)
                    day = target.timestamp_ms // 86_400_000
                    grouped[(day, split)].append((target, snapshot))
            # 同毫秒事件全部完成候选判断后才进入历史，保持无并列时序泄漏。
            history.extend(same_time_events)
            cursor = group_end

        for (day, split), candidates in sorted(grouped.items()):
            # 每个item只留当天最高意图事件；同级时保留时间更早的稳定代表。
            best_by_item: dict[
                str, tuple[InteractionEvent, tuple[InteractionEvent, ...]]
            ] = {}
            for candidate in candidates:
                target, _ = candidate
                previous = best_by_item.get(target.item_id)
                if previous is None or priority[target.behavior] > priority[
                    previous[0].behavior
                ]:
                    best_by_item[target.item_id] = candidate
            by_behavior: dict[
                Behavior, list[tuple[InteractionEvent, tuple[InteractionEvent, ...]]]
            ] = defaultdict(list)
            for candidate in best_by_item.values():
                by_behavior[candidate[0].behavior].append(candidate)
            for behavior in sorted(by_behavior, key=lambda item: priority[item]):
                rows = sorted(by_behavior[behavior], key=lambda item: item[0])
                for start in range(0, len(rows) - list_size + 1, list_size):
                    chunk = rows[start : start + list_size]
                    first_history = chunk[0][1]
                    list_targets = tuple(target for target, _ in chunk)
                    future_targets: tuple[InteractionEvent, ...] = ()
                    if max_future_targets:
                        last_target_time = max(
                            target.timestamp_ms for target in list_targets
                        )
                        future_rows = []
                        future_start = bisect_right(user_timestamps, last_target_time)
                        for future in user_events[future_start:]:
                            future_split = boundaries.split_for(future.timestamp_ms)
                            if future_split is not split:
                                # 全局split按时间单调；越过本split后无需继续扫描。
                                break
                            if future.behavior not in targets:
                                continue
                            if priority[future.behavior] < priority[minimum_future_behavior]:
                                continue
                            if (
                                require_target_in_training_items
                                and future.item_id not in train_items
                            ):
                                continue
                            future_rows.append(future)
                            if len(future_rows) == max_future_targets:
                                break
                        future_targets = tuple(future_rows)
                    if require_future_targets and not future_targets:
                        continue
                    keep(ListwiseTargetSample(
                        split=split,
                        user_id=user_id,
                        history=first_history,
                        targets=list_targets,
                        utc_day=day,
                        future_targets=future_targets,
                    ))

    for split in Split:
        samples.extend(reservoirs[split])
    return samples