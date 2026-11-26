"""把 Slow-LLM 输出编译成有边界、可复现的 IGR 数值控制。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence


BEHAVIOR_IDS = {"view": 0, "addtocart": 1, "transaction": 2}

# paper_igr严格对应论文的query-item余弦Top-K；agentic_plan在相同语义分数上
# 额外执行结构化Plan。显式模式名可避免把工程扩展误记为论文原始方法。
RetrievalMode = Literal["paper_igr", "agentic_plan"]
PAPER_IGR: RetrievalMode = "paper_igr"
AGENTIC_PLAN: RetrievalMode = "agentic_plan"
RETRIEVAL_MODES = (PAPER_IGR, AGENTIC_PLAN)


@dataclass(frozen=True)
class ExecutableRetrievalPlan:
    """执行层真正消费的白名单计划；不包含任何自由文本。"""
    priority_behavior_ids: tuple[int, ...]
    recency: str
    prefer_repeated_items: bool
    diversity: str


def compile_retrieval_plan(
    plan: Mapping[str, object], behavior_counts: Mapping[str, int],
) -> ExecutableRetrievalPlan:
    """只编译检索实现明确支持的 schema 字段。

    Qwen 生成的优先行为必须确实出现在历史中；intent、evidence、strategy
    等自由文本绝不进入打分，避免语言模型无依据解释污染检索。
    """

    priorities = plan.get("priority_behaviors")
    if not isinstance(priorities, list) or not priorities:
        raise ValueError("priority_behaviors must be a non-empty list")
    observed = []
    for behavior in priorities:
        if behavior not in BEHAVIOR_IDS:
            raise ValueError(f"unsupported behavior: {behavior!r}")
        if int(behavior_counts.get(behavior, 0)) > 0 and behavior not in observed:
            observed.append(behavior)
    if not observed:
        raise ValueError("retrieval plan has no priority behavior observed in history")
    recency = plan.get("recency")
    diversity = plan.get("diversity")
    repeated = plan.get("prefer_repeated_items")
    if recency not in {"recent", "long_term", "balanced"}:
        raise ValueError("unsupported recency control")
    if diversity not in {"low", "medium", "high"}:
        raise ValueError("unsupported diversity control")
    if not isinstance(repeated, bool):
        raise ValueError("prefer_repeated_items must be boolean")
    return ExecutableRetrievalPlan(
        priority_behavior_ids=tuple(BEHAVIOR_IDS[item] for item in observed),
        recency=recency,
        prefer_repeated_items=repeated,
        diversity=diversity,
    )


def execute_retrieval_plan(
    semantic_scores, candidate_sids, candidate_behavior_ids, padding_mask,
    plans: Sequence[ExecutableRetrievalPlan], *, top_k: int,
):
    """在语义分数上施加有界计划偏置，再做考虑多样性的 top-k。

    输入形状：semantic_scores/behavior/mask=[B,T]，candidate_sids=[B,T,L]。
    语义相似度仍是主分数；Plan 只增加行为优先、时效、重复商品和重复 SID
    抑制等固定小偏置。本函数故意不接收生成的自然语言字段。
    """

    import torch

    if semantic_scores.ndim != 2 or candidate_sids.ndim != 3:
        raise ValueError("scores and candidate_sids must be [B,T] and [B,T,L]")
    if candidate_behavior_ids.shape != semantic_scores.shape:
        raise ValueError("candidate_behavior_ids must match scores")
    if padding_mask.shape != semantic_scores.shape or padding_mask.dtype != torch.bool:
        raise ValueError("padding_mask must be boolean and match scores")
    if len(plans) != semantic_scores.shape[0]:
        raise ValueError("one executable plan is required per batch row")
    if top_k < 1 or ((~padding_mask).sum(dim=1) < top_k).any():
        raise ValueError("top_k exceeds valid candidates")

    # clone 保留原始 semantic score，便于 paired baseline/plan 审计。
    adjusted = semantic_scores.clone()
    history = semantic_scores.shape[1]
    positions = torch.linspace(0.0, 1.0, history, device=adjusted.device)
    for batch, plan in enumerate(plans):
        for behavior_id in plan.priority_behavior_ids:
            match = candidate_behavior_ids[batch] == behavior_id
            # schema 定义的是优先行为“集合”而非排序列表；没有显式数值权重前必须等权。
            adjusted[batch] = adjusted[batch] + match.to(adjusted.dtype) * 0.15
        if plan.recency == "recent":
            adjusted[batch] = adjusted[batch] + 0.15 * positions
        elif plan.recency == "balanced":
            adjusted[batch] = adjusted[batch] + 0.05 * positions
        if plan.prefer_repeated_items:
            for index in range(history):
                if padding_mask[batch, index]:
                    continue
                repeated = (candidate_sids[batch] == candidate_sids[batch, index]).all(dim=-1)
                if int((repeated & ~padding_mask[batch]).sum()) > 1:
                    adjusted[batch, index] = adjusted[batch, index] + 0.10
    adjusted = adjusted.masked_fill(padding_mask, float("-inf"))

    # 逐项选择允许在已选 SID 的完全重复项上施加 medium/high 多样性约束。
    selected_rows = []
    selected_scores = []
    for batch, plan in enumerate(plans):
        working = adjusted[batch].clone()
        row_indices = []
        row_scores = []
        for _ in range(top_k):
            index = int(working.argmax())
            row_indices.append(index)
            row_scores.append(adjusted[batch, index])
            working[index] = float("-inf")
            duplicate = (candidate_sids[batch] == candidate_sids[batch, index]).all(dim=-1)
            if plan.diversity == "high":
                working = working.masked_fill(duplicate, float("-inf"))
            elif plan.diversity == "medium":
                working = working - duplicate.to(working.dtype) * 0.20
        selected_rows.append(row_indices)
        selected_scores.append(torch.stack(row_scores))
    return torch.tensor(selected_rows, device=adjusted.device), torch.stack(selected_scores), adjusted
