"""OxygenREC-v2 EA-TOSD的无外部Reward Model核心公式。

本模块只实现论文式可验证轨迹奖励、best-of-G选择以及低/高熵双路自蒸馏。
未来信息Teacher由``OxygenRECModel.forward(..., privileged_future_sids=...)``
提供；它与Student共享参数，区别仅是训练期多读取未来SID前缀。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class EATOSDConfig:
    """EA-TOSD超参数；默认值采用论文披露的三项权重与熵阈值。"""

    geometric_decay: float = 0.9
    verifiable_weight: float = 0.1
    self_distillation_weight: float = 0.01
    forward_kl_weight: float = 0.01
    # 论文算法同时运行behavior-weighted SFT anchor；正文未在同处披露其系数。
    supervised_weight: float = 1.0
    low_entropy_threshold: float = 0.75
    high_entropy_threshold: float = 2.6
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if not 0.0 <= self.geometric_decay <= 1.0:
            raise ValueError("geometric_decay must be in [0, 1]")
        for name in (
            "verifiable_weight",
            "self_distillation_weight",
            "forward_kl_weight",
            "supervised_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.low_entropy_threshold < 0 or self.high_entropy_threshold < 0:
            raise ValueError("entropy thresholds cannot be negative")
        if self.low_entropy_threshold >= self.high_entropy_threshold:
            raise ValueError("low entropy threshold must be smaller than high threshold")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


@dataclass(frozen=True)
class VerifiableTrajectorySelection:
    """几何token命中奖励与每个batch行的best-of-G选择结果。"""

    token_weights: Tensor  # [S]
    token_hits: Tensor  # [B,G,S]
    rewards: Tensor  # [B,G]
    best_indices: Tensor  # [B]
    selected_tokens: Tensor  # [B,S]
    selected_rewards: Tensor  # [B]


@dataclass(frozen=True)
class EATOSDOutput:
    """EA-TOSD总损失、三个后训练分量及熵门控诊断。"""

    loss: Tensor
    verifiable_loss: Tensor
    self_distillation_loss: Tensor
    forward_kl_loss: Tensor
    supervised_loss: Tensor
    selected_student_log_probs: Tensor  # [B,S]
    privilege_advantage: Tensor  # [B,S]，已stop-gradient
    teacher_entropy: Tensor  # [B,S]
    low_entropy_weights: Tensor  # [B,S]
    high_entropy_weights: Tensor  # [B,S]


@dataclass(frozen=True)
class PrivilegedCheckpointLoad:
    """预训练checkpoint迁入带未来前缀模型时的位置表扩展报告。"""

    copied_position_rows: int
    initialized_future_rows: int


def geometric_token_weights(
    token_count: int,
    decay: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """论文式``omega_t ∝ gamma^(t-1)``，返回和为1的``[S]``权重。"""
    if token_count < 1:
        raise ValueError("token_count must be positive")
    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must be in [0, 1]")
    exponents = torch.arange(token_count, device=device, dtype=dtype)
    weights = torch.pow(torch.as_tensor(decay, device=device, dtype=dtype), exponents)
    return weights / weights.sum()


def load_pretrained_with_future_positions(
    model: nn.Module,
    pretrained_state: Mapping[str, Tensor],
) -> PrivilegedCheckpointLoad:
    """严格加载listwise预训练权重，并只扩展Teacher需要的位置表。

    Student所用的旧位置行逐值保留。新增的``3M``行按固定offset从旧表复制：
    Teacher在物理位置``p+3M``预测时，初始位置向量等于Student位置``p``，
    避免未训练的随机位置embedding主导privilege advantage。除
    ``decoder_positions.weight``外出现任何缺失或shape差异都会立即报错。
    """
    current_state = model.state_dict()
    current_keys = set(current_state)
    pretrained_keys = set(pretrained_state)
    if current_keys != pretrained_keys:
        missing = sorted(current_keys - pretrained_keys)
        unexpected = sorted(pretrained_keys - current_keys)
        raise ValueError(
            f"checkpoint keys do not match model: missing={missing[:3]} "
            f"unexpected={unexpected[:3]}"
        )

    position_key = "decoder_positions.weight"
    merged: dict[str, Tensor] = {}
    initialized_rows = 0
    copied_rows = 0
    for key, current in current_state.items():
        previous = pretrained_state[key]
        if previous.shape == current.shape:
            merged[key] = previous
            if key == position_key:
                copied_rows = previous.shape[0]
            continue
        if key != position_key or previous.ndim != 2 or current.ndim != 2:
            raise ValueError(
                f"checkpoint tensor shape mismatch for {key}: "
                f"{tuple(previous.shape)} != {tuple(current.shape)}"
            )
        if previous.shape[1] != current.shape[1] or previous.shape[0] >= current.shape[0]:
            raise ValueError("decoder position table cannot be expanded from checkpoint")
        extra_rows = current.shape[0] - previous.shape[0]
        configured_extra = (
            model.config.sid_levels * model.config.max_future_items
        )
        if extra_rows != configured_extra or previous.shape[0] < extra_rows:
            raise ValueError(
                "decoder position difference must equal sid_levels * max_future_items"
            )
        adapted = current.clone()
        adapted[: previous.shape[0]].copy_(previous)
        for row in range(previous.shape[0], current.shape[0]):
            adapted[row].copy_(previous[row - extra_rows])
        merged[key] = adapted
        copied_rows = previous.shape[0]
        initialized_rows = extra_rows

    model.load_state_dict(merged, strict=True)
    return PrivilegedCheckpointLoad(
        copied_position_rows=copied_rows,
        initialized_future_rows=initialized_rows,
    )


def select_best_verifiable_trajectory(
    candidate_tokens: Tensor,
    gold_tokens: Tensor,
    *,
    geometric_decay: float = 0.9,
) -> VerifiableTrajectorySelection:
    """用逐token真值命中计算reward，并对每个上下文选reward最高轨迹。

    ``candidate_tokens``=[B,G,S]来自当前Student的on-policy采样，
    ``gold_tokens``=[B,S]是真实列表SID序列。并列时``argmax``稳定选择首条，
    不使用任何学习型ranker或外部Reward Model。
    """
    if candidate_tokens.ndim != 3:
        raise ValueError("candidate_tokens must have shape [batch, group, SID tokens]")
    batch_size, group_size, token_count = candidate_tokens.shape
    if group_size < 1:
        raise ValueError("candidate group cannot be empty")
    if gold_tokens.shape != (batch_size, token_count):
        raise ValueError("gold_tokens must have shape [batch, SID tokens]")
    if candidate_tokens.dtype != torch.long or gold_tokens.dtype != torch.long:
        raise ValueError("candidate_tokens and gold_tokens must use torch.long")

    weights = geometric_token_weights(
        token_count,
        geometric_decay,
        device=candidate_tokens.device,
        dtype=torch.float32,
    )
    hits = candidate_tokens.eq(gold_tokens.unsqueeze(1))
    rewards = (hits.to(weights.dtype) * weights.view(1, 1, -1)).sum(dim=-1)
    best_indices = rewards.argmax(dim=1)
    batch_indices = torch.arange(batch_size, device=candidate_tokens.device)
    return VerifiableTrajectorySelection(
        token_weights=weights,
        token_hits=hits,
        rewards=rewards,
        best_indices=best_indices,
        selected_tokens=candidate_tokens[batch_indices, best_indices],
        selected_rewards=rewards[batch_indices, best_indices],
    )


def _stack_logits(logits: Sequence[Tensor] | Tensor, name: str) -> Tensor:
    """将模型逐step tuple或现成``[B,S,V]``统一成三维tensor。"""
    if isinstance(logits, Tensor):
        stacked = logits
    else:
        if not logits:
            raise ValueError(f"{name} cannot be empty")
        stacked = torch.stack(tuple(logits), dim=1)
    if stacked.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, SID tokens, vocabulary]")
    return stacked


def ea_tosd_loss(
    student_logits: Sequence[Tensor] | Tensor,
    teacher_logits: Sequence[Tensor] | Tensor,
    selected_tokens: Tensor,
    selected_rewards: Tensor,
    supervised_loss: Tensor,
    *,
    config: EATOSDConfig | None = None,
) -> EATOSDOutput:
    """计算EA-TOSD的``L_VR + L_SD + L_FKL + L_SFT``。

    Student与Teacher logits都沿已选best-of-G轨迹teacher-forcing得到。
    Teacher分布和privilege advantage均stop-gradient，因此所有更新都落在
    部署Student这条共享backbone上，Teacher不是独立训练的外部模型。
    """
    config = EATOSDConfig() if config is None else config
    student = _stack_logits(student_logits, "student_logits")
    teacher = _stack_logits(teacher_logits, "teacher_logits")
    if student.shape != teacher.shape:
        raise ValueError("student_logits and teacher_logits must share shape")
    batch_size, token_count, _ = student.shape
    if selected_tokens.shape != (batch_size, token_count):
        raise ValueError("selected_tokens must have shape [batch, SID tokens]")
    if selected_tokens.dtype != torch.long:
        raise ValueError("selected_tokens must use torch.long")
    if selected_rewards.shape != (batch_size,):
        raise ValueError("selected_rewards must have shape [batch]")
    if supervised_loss.ndim != 0:
        raise ValueError("supervised_loss must be a scalar")
    if not torch.isfinite(selected_rewards).all():
        raise ValueError("selected_rewards must be finite")

    student_log_distribution = F.log_softmax(student, dim=-1)
    selected_student = student_log_distribution.gather(
        2, selected_tokens.unsqueeze(-1)
    ).squeeze(-1)

    # Teacher只提供目标分布：断开其反向图，避免共享参数的teacher分支追逐自身。
    teacher_log_distribution = F.log_softmax(teacher, dim=-1).detach()
    teacher_distribution = teacher_log_distribution.exp()
    selected_teacher = teacher_log_distribution.gather(
        2, selected_tokens.unsqueeze(-1)
    ).squeeze(-1)
    privilege_advantage = (selected_teacher - selected_student).detach()
    entropy = -(teacher_distribution * teacher_log_distribution).sum(dim=-1)

    low_gate = entropy.lt(config.low_entropy_threshold)
    high_gate = entropy.gt(config.high_entropy_threshold)
    low_weights = low_gate.to(student.dtype) / (
        low_gate.sum(dim=1, keepdim=True).to(student.dtype) + config.epsilon
    )
    high_weights = high_gate.to(student.dtype) / (
        high_gate.sum(dim=1, keepdim=True).to(student.dtype) + config.epsilon
    )

    verifiable_loss = -(
        selected_rewards.detach() * selected_student.sum(dim=1)
    ).mean()
    self_distillation_loss = -(
        low_weights * privilege_advantage * selected_student
    ).sum(dim=1).mean()
    token_forward_kl = (
        teacher_distribution
        * (teacher_log_distribution - student_log_distribution)
    ).sum(dim=-1)
    forward_kl_loss = (high_weights * token_forward_kl).sum(dim=1).mean()
    total = (
        config.verifiable_weight * verifiable_loss
        + config.self_distillation_weight * self_distillation_loss
        + config.forward_kl_weight * forward_kl_loss
        + config.supervised_weight * supervised_loss
    )
    return EATOSDOutput(
        loss=total,
        verifiable_loss=verifiable_loss,
        self_distillation_loss=self_distillation_loss,
        forward_kl_loss=forward_kl_loss,
        supervised_loss=supervised_loss,
        selected_student_log_probs=selected_student,
        privilege_advantage=privilege_advantage,
        teacher_entropy=entropy,
        low_entropy_weights=low_weights,
        high_entropy_weights=high_weights,
    )