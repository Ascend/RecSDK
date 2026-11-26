"""把已生成并人工审核的 Retrieval Reasoning 转成 Qwen SFT 对话样本。"""

from __future__ import annotations

import json
from typing import Mapping

from .llm_reasoning import parse_reasoning_json, reasoning_system_prompt
from .retrieval_planning import compile_retrieval_plan


def reasoning_fidelity_issues(record: Mapping[str, object]) -> list[str]:
    """按可验证字段发现明显矛盾；只做自动拒绝依据，不能替代人工批准。"""

    evidence = record.get("input_evidence")
    reasoning = record.get("reasoning")
    if not isinstance(evidence, Mapping) or not isinstance(reasoning, Mapping):
        return ["missing_evidence_or_reasoning"]
    counts = evidence.get("behavior_counts")
    plan = reasoning.get("retrieval_plan")
    if not isinstance(counts, Mapping) or not isinstance(plan, Mapping):
        return ["missing_counts_or_plan"]
    priorities = set(plan.get("priority_behaviors", ()))
    issues = []
    if int(counts.get("transaction", 0)) > 0 and "transaction" not in priorities:
        issues.append("transaction_evidence_missing_from_priority")
    if int(counts.get("addtocart", 0)) > 0 and "addtocart" not in priorities:
        issues.append("cart_evidence_missing_from_priority")
    text = json.dumps(reasoning, ensure_ascii=False)
    if (
        int(counts.get("addtocart", 0)) <= int(counts.get("view", 0))
        and ("高于浏览占比" in text or "高于浏览行为的占比" in text)
    ):
        issues.append("unsupported_cart_vs_view_comparison")
    if plan.get("diversity") == "low" and ("多样性不足" in text or "跨品类" in text):
        issues.append("low_diversity_conflicts_with_reasoning")
    return issues


def build_reasoning_sft_example(
    record: Mapping[str, object], *, require_approved: bool = True,
) -> dict[str, object]:
    """校验一条 review 记录并生成标准 messages；默认拒绝未审核标签。"""

    case_id = record.get("case_id")
    prompt = record.get("prompt")
    evidence = record.get("input_evidence")
    reasoning = record.get("reasoning")
    review_status = record.get("review_status", "pending")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id must be a non-empty string")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"case {case_id}: prompt must be a non-empty string")
    if not isinstance(evidence, Mapping):
        raise ValueError(f"case {case_id}: input_evidence must be an object")
    if not isinstance(reasoning, Mapping):
        raise ValueError(f"case {case_id}: reasoning must be an object")
    if review_status not in {"pending", "approved", "rejected"}:
        raise ValueError(f"case {case_id}: unsupported review_status={review_status!r}")
    if require_approved and review_status != "approved":
        raise ValueError(f"case {case_id}: SFT label is not approved")

    # 复用线上生成时的严格 schema，保证训练标签和推理输出协议完全一致。
    assistant_text = json.dumps(reasoning, ensure_ascii=False, separators=(",", ":"))
    parsed = parse_reasoning_json(assistant_text)
    behavior_counts = evidence.get("behavior_counts")
    if not isinstance(behavior_counts, Mapping):
        raise ValueError(f"case {case_id}: behavior_counts must be an object")
    # Plan 还必须能被确定性 IGR 执行器编译，不能只满足 JSON 外形。
    compile_retrieval_plan(parsed["retrieval_plan"], behavior_counts)

    return {
        "messages": [
            {"role": "system", "content": reasoning_system_prompt()},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_text},
        ],
        "metadata": {
            "case_id": case_id,
            "review_status": review_status,
            "source": record.get("source", "oxygenrec_reasoning_review"),
        },
    }
