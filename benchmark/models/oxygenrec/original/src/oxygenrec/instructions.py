"""可审计的 Contextual Reasoning Instruction 公开代理。

论文的 Slow LLM 与私有 instruction store 未公开。本模块提供确定性的
文本→特征边界，让可读 instruction 能实际进入 Fast 模型；它是早期工程代理，
不是对语言模型本身的复现。当前真实 Qwen 路线见 ``llm_reasoning.py``。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence


_ASCII_WORD = re.compile(r"[A-Za-z0-9_]+")
_CJK = re.compile(r"[\u3400-\u9fff]")


@dataclass(frozen=True)
class ContextualInstruction:
    """一条可人工检查的 instruction，以及构造它所使用的证据和来源。"""

    sample_id: str
    text: str
    evidence: tuple[str, ...] = ()
    source: str = "deterministic_public_proxy"
    version: str = "v1"


def build_history_instruction(behaviors: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    """仅根据严格早于 target 的行为名称构造无泄漏的确定性代理文本。"""

    if not behaviors:
        raise ValueError("behaviors must not be empty")
    recent = tuple(behaviors[-5:])
    high_intent = sum(name in {"addtocart", "transaction"} for name in recent)
    repeated_views = sum(name == "view" for name in recent)
    evidence = (
        f"history_events={len(behaviors)}",
        f"recent_high_intent={high_intent}",
        f"recent_views={repeated_views}",
    )
    if high_intent:
        text = "近期历史包含加购或购买等高意图行为，优先检索相关历史商品。"
    elif repeated_views >= 3:
        text = "近期以连续浏览为主，结合长期历史检索稳定兴趣商品。"
    else:
        text = "近期行为证据较弱，扩大长期历史检索范围并保持候选多样性。"
    return text, evidence


def instruction_tokens(text: str) -> tuple[str, ...]:
    """不依赖外部模型，把英文词、中文单字和中文二元组切成简单 token。"""

    normalized = " ".join(text.strip().lower().split())
    words = _ASCII_WORD.findall(normalized)
    cjk = _CJK.findall(normalized)
    cjk_bigrams = ["".join(cjk[index:index + 2]) for index in range(len(cjk) - 1)]
    return tuple(words + cjk + cjk_bigrams)


def hash_instruction(text: str, dimension: int = 64) -> tuple[float, ...]:
    """使用带符号 feature hashing 编码文本，并做 L2 归一化。

    BLAKE2 不受 Python ``hash()`` 随机化影响，因此跨进程结果一致。
    该函数只用于早期公开代理与消融，不代表真实 LLM 语义表示。
    """

    if dimension < 1:
        raise ValueError("dimension must be positive")
    vector = [0.0] * dimension
    for token in instruction_tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=9).digest()
        index = int.from_bytes(digest[:8], "big") % dimension
        vector[index] += 1.0 if digest[8] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return tuple(vector)


def encode_instructions(
    instructions: Sequence[str], dimension: int = 64,
) -> list[tuple[float, ...]]:
    """批量调用 hash_instruction。"""
    return [hash_instruction(text, dimension) for text in instructions]


class InstructionStore:
    """保存代理文本、证据和来源的小型 JSONL 仓库。"""

    @staticmethod
    def save(path: str | Path, records: Iterable[ContextualInstruction]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for record in records:
                payload = asdict(record)
                payload["evidence"] = list(record.evidence)
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def load(path: str | Path) -> list[ContextualInstruction]:
        records = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                payload["evidence"] = tuple(payload.get("evidence", ()))
                records.append(ContextualInstruction(**payload))
        return records
