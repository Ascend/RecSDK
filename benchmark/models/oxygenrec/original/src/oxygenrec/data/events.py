"""统一交互事件定义，以及不同原始数据集到统一格式的读取器。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping


class Behavior(str, Enum):
    """当前 OxygenREC 公开数据实验支持的三种行为。"""

    VIEW = "view"
    ADD_TO_CART = "addtocart"
    TRANSACTION = "transaction"


@dataclass(frozen=True, order=True)
class InteractionEvent:
    """与具体数据集无关、可按时间排序的一条用户-商品交互。

    ``source_row`` 用作稳定的并列排序键，方便复现与排查。RetailRocket 没有
    给出同一毫秒内事件的因果顺序，因此构造样本时，同时间事件不能互相作为历史。
    """

    timestamp_ms: int
    source_row: int
    user_id: str
    item_id: str
    behavior: Behavior
    transaction_id: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        if self.source_row < 0:
            raise ValueError("source_row must be non-negative")
        if not self.user_id:
            raise ValueError("user_id must not be empty")
        if not self.item_id:
            raise ValueError("item_id must not be empty")


_RETAILROCKET_COLUMNS = {
    "timestamp",
    "visitorid",
    "event",
    "itemid",
    "transactionid",
}


def retailrocket_event_from_row(
    row: Mapping[str, str], *, source_row: int
) -> InteractionEvent:
    """把 RetailRocket ``events.csv`` 的一行转换为统一事件对象。"""

    try:
        behavior = Behavior(row["event"].strip().lower())
    except ValueError as error:
        raise ValueError(f"unsupported RetailRocket event {row.get('event')!r}") from error

    transaction_id = row.get("transactionid", "").strip() or None
    return InteractionEvent(
        timestamp_ms=int(row["timestamp"]),  # 3 位 Unix 时间戳，单位为毫秒，例如1433221275206 = 1433221275 s + 206 ms
        source_row=source_row,               # 原始events.csv中的物理行号，时间戳相同的时候，会按原始的CSV行号排序
        user_id=row["visitorid"].strip(),
        item_id=row["itemid"].strip(),
        behavior=behavior,
        transaction_id=transaction_id,
    )


def load_retailrocket_events(path: str | Path) -> Iterable[InteractionEvent]:
    """逐行读取 RetailRocket ``events.csv`` 并产出统一事件。

    使用生成器而不是一次性加载，从而控制约 276 万行原始文件的读取内存。
    """

    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = _RETAILROCKET_COLUMNS - columns
        if missing:
            raise ValueError(f"RetailRocket events CSV is missing columns: {sorted(missing)}")
        for source_row, row in enumerate(reader, start=2):
            yield retailrocket_event_from_row(row, source_row=source_row)
