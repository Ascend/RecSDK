"""把时间样本转换为补齐后的 Semantic-ID 模型输入。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..sid import SIDRegistry
from .temporal import ListwiseTargetSample, NextItemSample


@dataclass(frozen=True)
class SIDModelBatch:
    """普通短历史批次；主要张量逻辑形状为 history=[B,T,L]、target=[B,L]。"""
    history_sids: tuple[tuple[tuple[int, ...], ...], ...]
    history_padding_mask: tuple[tuple[bool, ...], ...]
    history_behavior_ids: tuple[tuple[int, ...], ...]
    target_sids: tuple[tuple[int, ...], ...]
    target_behavior_ids: tuple[int, ...]


@dataclass(frozen=True)
class LongShortSIDModelBatch:
    """IGR 批次：近期短历史进入主干，较早长历史作为检索候选池。"""

    short_history_sids: tuple[tuple[tuple[int, ...], ...], ...]
    short_history_padding_mask: tuple[tuple[bool, ...], ...]
    short_history_behavior_ids: tuple[tuple[int, ...], ...]
    long_history_sids: tuple[tuple[tuple[int, ...], ...], ...]
    long_history_padding_mask: tuple[tuple[bool, ...], ...]
    long_history_behavior_ids: tuple[tuple[int, ...], ...]
    target_sids: tuple[tuple[int, ...], ...]
    target_behavior_ids: tuple[int, ...]
    scenario_ids: tuple[int, ...]


@dataclass(frozen=True)
class ListwiseSIDModelBatch:
    """v2列表批次，并可携带EA-TOSD Teacher未来前缀``[B,M,L]``。"""

    history_sids: tuple[tuple[tuple[int, ...], ...], ...]
    history_padding_mask: tuple[tuple[bool, ...], ...]
    history_behavior_ids: tuple[tuple[int, ...], ...]
    target_sids: tuple[tuple[tuple[int, ...], ...], ...]
    target_behavior_ids: tuple[int, ...]
    future_sids: tuple[tuple[tuple[int, ...], ...], ...]
    future_padding_mask: tuple[tuple[bool, ...], ...]


def build_sid_model_batch(
    samples: Sequence[NextItemSample],
    registry: SIDRegistry,
    *,
    max_history_items: int,
) -> SIDModelBatch:
    """通过固定 registry 把 item ID 映射为 SID，并在左侧补齐近期历史。

    registry 外的历史商品会被明确过滤；target 必须存在于训练期 registry。
    冷启动 target 应进入独立评测协议，不能在这里悄悄编码。
    """

    if not samples:
        raise ValueError("samples must not be empty")
    if max_history_items < 1:
        raise ValueError("max_history_items must be positive")

    histories: list[tuple[tuple[int, ...], ...]] = []
    history_behaviors: list[tuple[int, ...]] = []
    targets: list[tuple[int, ...]] = []
    target_behaviors: list[int] = []
    # RetailRocket没有曝光日志；view在本项目中作为论文click行为的公开代理。
    behavior_id = {"view": 0, "addtocart": 1, "transaction": 2}
    for sample in samples:
        if sample.target.item_id not in registry.item_to_sid:
            raise ValueError(
                f"target item {sample.target.item_id!r} is absent from SID registry"
            )
        # 只保留可编码商品，并截取最近 T 个事件。
        known_events = [
            event
            for event in sample.history
            if event.item_id in registry.item_to_sid
        ][-max_history_items:]
        known_history = [registry.sid_for(event.item_id).codes for event in known_events]
        if not known_history:
            raise ValueError(
                f"sample for user {sample.user_id!r} has no known history items"
            )
        histories.append(tuple(known_history))
        history_behaviors.append(tuple(behavior_id[event.behavior.value] for event in known_events))
        targets.append(registry.sid_for(sample.target.item_id).codes)
        target_behaviors.append(behavior_id[sample.target.behavior.value])

    padded_history: list[tuple[tuple[int, ...], ...]] = []
    padding_masks: list[tuple[bool, ...]] = []
    padded_behaviors: list[tuple[int, ...]] = []
    pad_sid = (0,) * registry.levels
    for history, behaviors in zip(histories, history_behaviors, strict=True):
        padding = max_history_items - len(history)
        # 左补齐保证真实的最近事件仍位于序列尾部。
        padded_history.append((pad_sid,) * padding + history)
        padding_masks.append((True,) * padding + (False,) * len(history))
        padded_behaviors.append((0,) * padding + behaviors)
    return SIDModelBatch(
        history_sids=tuple(padded_history),
        history_padding_mask=tuple(padding_masks),
        history_behavior_ids=tuple(padded_behaviors),
        target_sids=tuple(targets),
        target_behavior_ids=tuple(target_behaviors),
    )


def build_listwise_sid_model_batch(
    samples: Sequence[ListwiseTargetSample],
    registry: SIDRegistry,
    *,
    max_history_items: int,
    max_future_items: int = 0,
) -> ListwiseSIDModelBatch:
    """把行为同质列表映射为SID，并把历史在左侧补齐到T。"""

    if not samples:
        raise ValueError("samples must not be empty")
    if max_history_items < 1:
        raise ValueError("max_history_items must be positive")
    if max_future_items < 0:
        raise ValueError("max_future_items cannot be negative")
    list_sizes = {len(sample.targets) for sample in samples}
    if len(list_sizes) != 1:
        raise ValueError("all listwise samples in a batch must share list size N")

    behavior_id = {"view": 0, "addtocart": 1, "transaction": 2}
    histories: list[tuple[tuple[int, ...], ...]] = []
    history_behaviors: list[tuple[int, ...]] = []
    targets: list[tuple[tuple[int, ...], ...]] = []
    target_behaviors: list[int] = []
    future_rows: list[tuple[tuple[int, ...], ...]] = []
    future_masks: list[tuple[bool, ...]] = []
    for sample in samples:
        unknown_targets = [
            target.item_id
            for target in sample.targets
            if target.item_id not in registry.item_to_sid
        ]
        if unknown_targets:
            raise ValueError(
                f"target items are absent from SID registry: {unknown_targets[:3]}"
            )
        known_events = [
            event for event in sample.history if event.item_id in registry.item_to_sid
        ][-max_history_items:]
        if not known_events:
            raise ValueError(
                f"sample for user {sample.user_id!r} has no known history items"
            )
        histories.append(tuple(
            registry.sid_for(event.item_id).codes for event in known_events
        ))
        history_behaviors.append(tuple(
            behavior_id[event.behavior.value] for event in known_events
        ))
        targets.append(tuple(
            registry.sid_for(target.item_id).codes for target in sample.targets
        ))
        target_behaviors.append(behavior_id[sample.target_behavior.value])
        if len(sample.future_targets) > max_future_items:
            raise ValueError("sample future targets exceed max_future_items")
        unknown_future = [
            target.item_id
            for target in sample.future_targets
            if target.item_id not in registry.item_to_sid
        ]
        if unknown_future:
            raise ValueError(
                f"future target items are absent from SID registry: {unknown_future[:3]}"
            )
        encoded_future = tuple(
            registry.sid_for(target.item_id).codes for target in sample.future_targets
        )
        future_padding = max_future_items - len(encoded_future)
        pad_sid = (0,) * registry.levels
        # 未来前缀左补齐，使最后一个槽始终是最近的真实未来行为。
        # Decoder用未来块末端状态预测第一个gold token；若右补齐，该读出状态
        # 会落在padding query上，虽然key已被mask，语义上仍不是我们要的Teacher。
        future_rows.append((pad_sid,) * future_padding + encoded_future)
        future_masks.append(
            (True,) * future_padding + (False,) * len(encoded_future)
        )

    pad_sid = (0,) * registry.levels
    padded_history: list[tuple[tuple[int, ...], ...]] = []
    padding_masks: list[tuple[bool, ...]] = []
    padded_behaviors: list[tuple[int, ...]] = []
    for history, behaviors in zip(histories, history_behaviors, strict=True):
        padding = max_history_items - len(history)
        padded_history.append((pad_sid,) * padding + history)
        padding_masks.append((True,) * padding + (False,) * len(history))
        padded_behaviors.append((0,) * padding + behaviors)
    return ListwiseSIDModelBatch(
        history_sids=tuple(padded_history),
        history_padding_mask=tuple(padding_masks),
        history_behavior_ids=tuple(padded_behaviors),
        target_sids=tuple(targets),
        target_behavior_ids=tuple(target_behaviors),
        future_sids=tuple(future_rows),
        future_padding_mask=tuple(future_masks),
    )


def build_long_short_sid_model_batch(
    samples: Sequence[NextItemSample],
    registry: SIDRegistry,
    *,
    short_history_items: int,
    long_history_items: int,
    minimum_long_history_items: int = 1,
) -> LongShortSIDModelBatch:
    """把已知历史切成互不重叠的近期窗口和较早窗口。

    最近 ``short_history_items`` 条给 Encoder 主干；其前面的
    ``long_history_items`` 条作为 IGR 候选池。同一事件不会同时出现在两条分支；
    长历史有效候选不足时直接拒绝样本，避免 IGR 选中 padding。
    """

    if not samples:
        raise ValueError("samples must not be empty")
    if short_history_items < 1 or long_history_items < 1:
        raise ValueError("history window sizes must be positive")
    if not 1 <= minimum_long_history_items <= long_history_items:
        raise ValueError("minimum_long_history_items must fit the long window")

    pad_sid = (0,) * registry.levels
    short_rows = []
    short_masks = []
    short_behaviors = []
    long_rows = []
    long_masks = []
    long_behaviors = []
    targets = []
    target_behaviors = []
    scenarios = []
    scenario_by_behavior = {"view": 0, "addtocart": 1, "transaction": 2}
    for sample in samples:
        if sample.target.item_id not in registry.item_to_sid:
            raise ValueError(
                f"target item {sample.target.item_id!r} is absent from SID registry"
            )
        known_events = [
            event
            for event in sample.history
            if event.item_id in registry.item_to_sid
        ]
        known = [registry.sid_for(event.item_id).codes for event in known_events]
        known_behavior = [scenario_by_behavior[event.behavior.value] for event in known_events]
        # 时间顺序不变：尾部是短历史，紧邻其前的部分是长历史候选。
        short = known[-short_history_items:]
        long = known[: -len(short)][-long_history_items:] if short else []
        short_behavior = known_behavior[-short_history_items:]
        long_behavior = known_behavior[: -len(short_behavior)][-long_history_items:] if short_behavior else []
        if not short:
            raise ValueError(f"sample for user {sample.user_id!r} has no known short history")
        if len(long) < minimum_long_history_items:
            raise ValueError(
                f"sample for user {sample.user_id!r} has only {len(long)} known long-history items"
            )
        short_pad = short_history_items - len(short)
        long_pad = long_history_items - len(long)
        short_rows.append((pad_sid,) * short_pad + tuple(short))
        short_masks.append((True,) * short_pad + (False,) * len(short))
        short_behaviors.append((0,) * short_pad + tuple(short_behavior))
        long_rows.append((pad_sid,) * long_pad + tuple(long))
        long_masks.append((True,) * long_pad + (False,) * len(long))
        long_behaviors.append((0,) * long_pad + tuple(long_behavior))
        targets.append(registry.sid_for(sample.target.item_id).codes)
        target_behavior = scenario_by_behavior[sample.target.behavior.value]
        target_behaviors.append(target_behavior)
        scenarios.append(target_behavior)
    return LongShortSIDModelBatch(
        short_history_sids=tuple(short_rows),
        short_history_padding_mask=tuple(short_masks),
        short_history_behavior_ids=tuple(short_behaviors),
        long_history_sids=tuple(long_rows),
        long_history_padding_mask=tuple(long_masks),
        long_history_behavior_ids=tuple(long_behaviors),
        target_sids=tuple(targets),
        target_behavior_ids=tuple(target_behaviors),
        scenario_ids=tuple(scenarios),
    )