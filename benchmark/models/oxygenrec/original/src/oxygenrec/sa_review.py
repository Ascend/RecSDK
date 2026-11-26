"""为SA-GCPO后训练选择可复现的匿名代表轨迹。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


ROLE_LABELS = {
    "target_covered": "old-policy候选组包含真实目标SID",
    "largest_reward_spread": "候选组内部reward差异最大",
    "largest_policy_shift": "后训练前后策略概率变化最大",
    "most_threshold_suppression": "真实目标reward阈值抑制候选最多",
}


def select_sa_gcpo_trajectories(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int | None]]:
    """按固定规则选择正例和边界例，并合并落在同一行的角色。"""

    if not rows:
        raise ValueError("SA-GCPO trajectory rows must not be empty")
    normalized = [dict(row) for row in rows]
    if any(not isinstance(row.get("cohort_index"), int) for row in normalized):
        raise ValueError("every trajectory row requires an integer cohort_index")

    rules = (
        ("target_covered", lambda row: bool(row["target_covered"]), "reward_spread"),
        ("largest_reward_spread", lambda row: True, "reward_spread"),
        ("largest_policy_shift", lambda row: True, "policy_shift"),
        ("most_threshold_suppression", lambda row: True, "suppressed_count"),
    )
    selected: dict[int, dict[str, object]] = {}
    coverage: dict[str, int | None] = {}
    for role, predicate, score_name in rules:
        candidates = [row for row in normalized if predicate(row)]
        if not candidates:
            coverage[role] = None
            continue
        chosen = sorted(
            candidates,
            key=lambda row: (-float(row[score_name]), int(row["cohort_index"])),
        )[0]
        index = int(chosen["cohort_index"])
        coverage[role] = index
        if index not in selected:
            selected[index] = {**chosen, "selection_roles": []}
        selected[index]["selection_roles"].append(role)

    role_order = {role: index for index, (role, *_rest) in enumerate(rules)}
    result = sorted(
        selected.values(),
        key=lambda row: min(role_order[role] for role in row["selection_roles"]),
    )
    return result, coverage
