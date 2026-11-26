"""按预定义正反例规则选择可复现的代表性端到端案例。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


ROLE_LABELS = {
    "igr_hit": "IGR在repeat-eligible样本中取回目标SID",
    "igr_miss": "IGR在repeat-eligible样本中漏掉目标SID",
    "transaction_target": "稀有transaction目标行为",
    "addtocart_target": "稀有addtocart目标行为",
    "browse_only": "历史只有view的低意图样本",
    "q2i_best": "当前cohort中Q2I余弦最高",
    "q2i_worst": "当前cohort中Q2I余弦最低",
    "beam_hit": "约束beam命中目标商品",
}


def select_representative_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, str | None]]:
    """每条规则独立选最典型样本，并把重合角色合并到同一案例。

    输入只需要匿名sample_key、行为/检索布尔量和Q2I cosine。返回顺序固定，
    因而相同cohort与checkpoint会得到相同review集合。
    """

    if not rows:
        raise ValueError("representative selection rows must not be empty")
    normalized = [dict(row) for row in rows]
    if any(not isinstance(row.get("sample_key"), str) for row in normalized):
        raise ValueError("every review row requires a string sample_key")

    def ordered(candidates, *, reverse: bool = False, score="q2i_cosine"):
        return sorted(
            candidates,
            key=lambda row: (
                -float(row[score]) if reverse else float(row[score]),
                str(row["sample_key"]),
            ),
        )

    rules = (
        ("igr_hit", lambda row: row["repeat_eligible"] and row["igr_hit"], True, "q2i_cosine"),
        ("igr_miss", lambda row: row["repeat_eligible"] and not row["igr_hit"], False, "q2i_cosine"),
        ("transaction_target", lambda row: row["target_behavior"] == "transaction", True, "history_length"),
        ("addtocart_target", lambda row: row["target_behavior"] == "addtocart", True, "history_length"),
        ("browse_only", lambda row: bool(row["browse_only"]), True, "history_length"),
        ("q2i_best", lambda row: True, True, "q2i_cosine"),
        ("q2i_worst", lambda row: True, False, "q2i_cosine"),
        ("beam_hit", lambda row: row["beam_hit_rank"] is not None, False, "beam_hit_rank"),
    )
    selected: dict[str, dict[str, object]] = {}
    coverage: dict[str, str | None] = {}
    for role, predicate, reverse, score in rules:
        candidates = [row for row in normalized if predicate(row)]
        if not candidates:
            coverage[role] = None
            continue
        chosen = ordered(candidates, reverse=reverse, score=score)[0]
        key = str(chosen["sample_key"])
        coverage[role] = key
        if key not in selected:
            selected[key] = {**chosen, "selection_roles": []}
        selected[key]["selection_roles"].append(role)

    role_order = {role: index for index, (role, *_rest) in enumerate(rules)}
    result = sorted(
        selected.values(),
        key=lambda row: min(role_order[role] for role in row["selection_roles"]),
    )
    return result, coverage
