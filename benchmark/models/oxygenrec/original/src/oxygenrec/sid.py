"""SID 量化、约束解码和评测共同使用的基础数据结构。

论文使用三层残差量化表示商品，但没有公开 SID 碰撞的处理细节。本模块不会
偷偷丢弃碰撞商品，而是保留同一 SID 对应的全部 item，让后续评测显式看到歧义。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping, Sequence


@dataclass(frozen=True, order=True)
class SemanticID:
    """经过层数与取值范围校验、不可修改的层次化 Semantic ID。"""

    codes: tuple[int, ...]

    def __init__(self, codes: Sequence[int], *, levels: int = 3, width: int = 8192):
        normalized = tuple(int(code) for code in codes)
        if len(normalized) != levels:
            raise ValueError(f"expected {levels} SID levels, got {len(normalized)}")
        for level, code in enumerate(normalized):
            if code < 0 or code >= width:
                raise ValueError(
                    f"SID code at level {level} must be in [0, {width}), got {code}"
                )
        object.__setattr__(self, "codes", normalized)

    def __iter__(self) -> Iterator[int]:
        return iter(self.codes)

    def __len__(self) -> int:
        return len(self.codes)


class SIDRegistry:
    """带版本号的 item↔SID 映射，同时保留并报告碰撞。"""

    def __init__(
        self,
        item_to_sid: Mapping[str, SemanticID | Sequence[int]],
        *,
        levels: int = 3,
        width: int = 8192,
        version: str = "unversioned",
    ) -> None:
        if not item_to_sid:
            raise ValueError("item_to_sid must not be empty")
        normalized: dict[str, SemanticID] = {}
        sid_to_items: dict[SemanticID, list[str]] = {}
        for raw_item_id, raw_sid in item_to_sid.items():
            item_id = str(raw_item_id)
            if not item_id:
                raise ValueError("item IDs must not be empty")
            sid = raw_sid if isinstance(raw_sid, SemanticID) else SemanticID(
                raw_sid, levels=levels, width=width
            )
            normalized[item_id] = sid
            sid_to_items.setdefault(sid, []).append(item_id)

        self.levels = levels
        self.width = width
        self.version = version
        self._item_to_sid = MappingProxyType(normalized)
        self._sid_to_items = MappingProxyType(
            {sid: tuple(sorted(items)) for sid, items in sid_to_items.items()}
        )

    @property
    def item_to_sid(self) -> Mapping[str, SemanticID]:
        return self._item_to_sid

    def sid_for(self, item_id: str) -> SemanticID:
        """查询一个商品的 SID；未知商品由映射本身抛出 KeyError。"""
        return self._item_to_sid[str(item_id)]

    def items_for(self, sid: SemanticID | Sequence[int]) -> tuple[str, ...]:
        """返回某个 SID 对应的全部商品，用于碰撞感知评测。"""
        key = sid if isinstance(sid, SemanticID) else SemanticID(
            sid, levels=self.levels, width=self.width
        )
        return self._sid_to_items.get(key, ())

    def collisions(self) -> Mapping[SemanticID, tuple[str, ...]]:
        """返回至少对应两个商品的 SID。"""
        return MappingProxyType(
            {sid: items for sid, items in self._sid_to_items.items() if len(items) > 1}
        )

    def collision_rate(self) -> float:
        """计算处在碰撞 SID 中的商品占全部已登记商品的比例。"""
        colliding_items = sum(len(items) for items in self.collisions().values())
        return colliding_items / len(self._item_to_sid)

    def to_json(self, path: str | Path) -> None:
        """把 registry 及其量化版本保存为可审计 JSON。"""
        payload = {
            "version": self.version,
            "levels": self.levels,
            "width": self.width,
            "items": {
                item_id: list(sid.codes)
                for item_id, sid in sorted(self._item_to_sid.items())
            },
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "SIDRegistry":
        """从 JSON 恢复完全相同的 item↔SID 映射。"""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            payload["items"],
            levels=int(payload["levels"]),
            width=int(payload["width"]),
            version=str(payload["version"]),
        )


class PrefixTrie:
    """合法 SID 的前缀树；生成每一层 code 时用于屏蔽非法续写。"""

    _END = object()

    def __init__(self, paths: Iterable[SemanticID | Sequence[int]]) -> None:
        self._root: dict[object, dict] = {}
        count = 0
        for raw_path in paths:
            path = raw_path.codes if isinstance(raw_path, SemanticID) else tuple(raw_path)
            if not path:
                raise ValueError("SID paths must not be empty")
            node = self._root
            for code in path:
                node = node.setdefault(int(code), {})
            node[self._END] = {}
            count += 1
        if count == 0:
            raise ValueError("at least one SID path is required")

    @classmethod
    def from_registry(cls, registry: SIDRegistry) -> "PrefixTrie":
        return cls(registry.item_to_sid.values())

    def allowed_next(self, prefix: Sequence[int]) -> tuple[int, ...]:
        """给定已生成前缀，返回下一层允许选择的 code。"""
        node = self._find(prefix)
        if node is None:
            return ()
        return tuple(sorted(key for key in node if key is not self._END))

    def is_valid_prefix(self, prefix: Sequence[int]) -> bool:
        return self._find(prefix) is not None

    def contains(self, path: Sequence[int]) -> bool:
        node = self._find(path)
        return node is not None and self._END in node

    def _find(self, prefix: Sequence[int]) -> dict | None:
        node = self._root
        for code in prefix:
            child = node.get(int(code))
            if child is None:
                return None
            node = child
        return node
