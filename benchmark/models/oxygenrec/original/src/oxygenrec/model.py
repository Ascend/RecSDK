"""包含v1 instruction/Q2I/IGR与v2行为指令的Dense OxygenREC主干。

模型只消费已 token 化的商品 Semantic ID。数据层必须使用带版本的
``SIDRegistry`` 完成 item→SID 映射，避免 checkpoint 静默换用不同 codebook。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .sid import PrefixTrie
from .retrieval_planning import (
    AGENTIC_PLAN,
    PAPER_IGR,
    RETRIEVAL_MODES,
    ExecutableRetrievalPlan,
    RetrievalMode,
    execute_retrieval_plan,
)


@dataclass(frozen=True)
class OxygenRECConfig:
    """小型 Dense 复现模型的显式配置；默认值不是论文私有生产参数。"""

    sid_width: int
    sid_levels: int = 3
    instruction_vocab_size: int = 1
    scenario_vocab_size: int = 1
    instruction_feature_size: int = 0
    behavior_vocab_size: int = 0
    # v2 的目标行为指令词表；0 表示保持 v1 路径，不创建 Decoder 侧 I_b。
    behavior_instruction_vocab_size: int = 0
    behavior_time_decay: float = 0.0
    use_history_context_instruction: bool = False
    history_context_pooling: str = "mean"
    q2i_dimension: int = 128
    q2i_weight: float = 0.0
    q2i_variance_weight: float = 0.01
    q2i_decorrelation_weight: float = 0.01
    igr_top_k: int = 0
    hidden_size: int = 128
    attention_heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    feedforward_size: int = 512
    dropout: float = 0.1
    max_history_items: int = 256
    # v1默认为单目标；v2列表式训练按N个商品预留3N个SID解码位置。
    max_target_items: int = 1
    # EA-TOSD Teacher最多额外读取M个未来商品；0保持既有v1/v2预训练checkpoint形状。
    max_future_items: int = 0

    def __post_init__(self) -> None:
        positive = {
            "sid_width": self.sid_width,
            "sid_levels": self.sid_levels,
            "instruction_vocab_size": self.instruction_vocab_size,
            "scenario_vocab_size": self.scenario_vocab_size,
            "q2i_dimension": self.q2i_dimension,
            "hidden_size": self.hidden_size,
            "attention_heads": self.attention_heads,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "feedforward_size": self.feedforward_size,
            "max_history_items": self.max_history_items,
            "max_target_items": self.max_target_items,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if (
            self.instruction_feature_size < 0
            or self.behavior_vocab_size < 0
            or self.behavior_instruction_vocab_size < 0
            or self.igr_top_k < 0
            or self.max_future_items < 0
        ):
            raise ValueError(
                "instruction_feature_size, behavior vocabularies, igr_top_k and "
                "max_future_items "
                "cannot be negative"
            )
        if self.behavior_time_decay < 0:
            raise ValueError("behavior_time_decay cannot be negative")
        for name in ("q2i_weight", "q2i_variance_weight", "q2i_decorrelation_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.history_context_pooling not in {"mean", "attention"}:
            raise ValueError("history_context_pooling must be mean or attention")


@dataclass(frozen=True)
class OxygenRECOutput:
    """逐SID token logits、NTP/Q2I loss以及可选IGR诊断结果。"""

    logits: tuple[Tensor, ...]
    loss: Tensor | None = None
    level_losses: tuple[Tensor, ...] | None = None
    ntp_loss: Tensor | None = None
    q2i_loss: Tensor | None = None
    q2i_alignment_loss: Tensor | None = None
    q2i_cosine: Tensor | None = None
    igr_indices: Tensor | None = None
    igr_scores: Tensor | None = None


@dataclass(frozen=True)
class BeamSearchOutput:
    """按累计对数概率排序的合法 SID 路径。"""

    semantic_ids: Tensor  # [B, beam, 3N]
    scores: Tensor  # [batch, beam]


class OxygenRECModel(nn.Module):
    """小型 Transformer Encoder-Decoder；每一层 SID 有独立预测头。"""

    def __init__(self, config: OxygenRECConfig) -> None:
        """创建 SID、v1/v2 instruction、Q2I/IGR 与 Encoder-Decoder。"""
        super().__init__()
        self.config = config
        # 三层 SID 各有独立 embedding；一个商品向量由三层 embedding 相加得到。
        self.sid_embeddings = nn.ModuleList(
            nn.Embedding(
                # v2 按论文把 click/cart/order 放到 SID 词表上方的保留 token。
                config.sid_width + (
                    config.behavior_instruction_vocab_size if level == 0 else 0
                ),
                config.hidden_size,
            )
            for level in range(config.sid_levels)
        )
        self.history_positions = nn.Embedding(
            config.max_history_items + config.igr_top_k, config.hidden_size
        )
        self.instruction_embeddings = nn.Embedding(
            config.instruction_vocab_size, config.hidden_size
        )
        self.scenario_embeddings = nn.Embedding(config.scenario_vocab_size, config.hidden_size)
        self.behavior_embeddings = (
            nn.Embedding(config.behavior_vocab_size, config.hidden_size)
            if config.behavior_vocab_size else None
        )
        self.behavior_instruction_adapter = (
            nn.Sequential(
                nn.Linear(config.hidden_size, config.feedforward_size),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(config.dropout),
                nn.Linear(config.feedforward_size, config.hidden_size),
            )
            if config.behavior_instruction_vocab_size else None
        )
        self.instruction_feature_adapter = (
            nn.Linear(config.instruction_feature_size, config.hidden_size)
            if config.instruction_feature_size else None
        )
        self.history_context_adapter = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), nn.GELU()
        )
        self.history_context_query = nn.Linear(config.hidden_size, config.hidden_size)
        self.history_context_key = nn.Linear(config.hidden_size, config.hidden_size)
        self.history_context_value = nn.Linear(config.hidden_size, config.hidden_size)
        # [scenario; reasoning] -> Q2I/IGR 共用的归一化 query 向量。
        self.query_adapter = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size), nn.GELU(),
            nn.Linear(config.hidden_size, config.q2i_dimension),
        )
        self.item_adapter = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
            nn.Linear(config.hidden_size, config.q2i_dimension),
        )
        self.bos_embedding = nn.Parameter(torch.empty(config.hidden_size))
        self.decoder_positions = nn.Embedding(
            # 长度=P+3M+3N-1；M=0时保持旧checkpoint的位置表形状。
            config.sid_levels * config.max_target_items
            + config.sid_levels * config.max_future_items
            + 2
            + int(config.behavior_instruction_vocab_size > 0),
            config.hidden_size,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_size,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_size,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, config.encoder_layers, norm=nn.LayerNorm(config.hidden_size)
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, config.decoder_layers, norm=nn.LayerNorm(config.hidden_size)
        )
        # 三个预测头分别负责SID level 0/1/2；列表生成时按每个商品循环复用。
        self.prediction_heads = nn.ModuleList(
            nn.Linear(config.hidden_size, config.sid_width, bias=False)
            for _ in range(config.sid_levels)
        )
        self.dropout = nn.Dropout(config.dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """初始化代码中额外定义的位置和 BOS 参数。"""
        nn.init.normal_(self.bos_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.history_positions.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.decoder_positions.weight, mean=0.0, std=0.02)

    def forward(
        self,
        history_sids: Tensor,
        history_padding_mask: Tensor,
        *,
        target_sids: Tensor | None = None,
        history_behavior_ids: Tensor | None = None,
        sample_weights: Tensor | None = None,
        token_weights: Tensor | None = None,
        instruction_ids: Tensor | None = None,
        scenario_ids: Tensor | None = None,
        behavior_instruction_ids: Tensor | None = None,
        privileged_future_sids: Tensor | None = None,
        privileged_future_padding_mask: Tensor | None = None,
        instruction_features: Tensor | None = None,
        trigger_sids: Tensor | None = None,
        long_history_sids: Tensor | None = None,
        long_history_padding_mask: Tensor | None = None,
        long_history_behavior_ids: Tensor | None = None,
        retrieval_plans: Sequence[ExecutableRetrievalPlan] | None = None,
        retrieval_mode: RetrievalMode = PAPER_IGR,
        level_weights: Sequence[float] | Tensor | None = None,
        output_items: int = 1,
    ) -> OxygenRECOutput:
        """执行主前向：构造 query、可选 IGR、Encoder、Decoder 和联合损失。

        ``history_sids``=[B,T,L]，``history_padding_mask``=[B,T]，其中 True
        表示 padding。v1训练传入 ``target_sids``=[B,L]；v2列表式训练传入
        ``target_sids``=[B,N,L]，内部展平为3N个teacher-forcing token。
        EA-TOSD Teacher还可传``privileged_future_sids``=[B,M,L]与对应的
        ``privileged_future_padding_mask``=[B,M]；它只用于训练期teacher forcing，
        不进入部署时的``generate()``。
        不传target时生成``output_items``个商品。

        符号约定：B 为 batch size，T 为序列长度，H 为隐藏维度，
        V 为词表大小，L 为 SID 层数。``behavior_instruction_ids``=[B]
        是 v2 的目标行为（公开 RetailRocket 映射为 view/click=0、cart=1、order=2），
        与 Encoder 侧逐历史事件的 ``history_behavior_ids``=[B,T] 不同。
        """

        self._validate_inputs(history_sids, history_padding_mask, target_sids)
        future_codes, future_token_mask = self._prepare_privileged_future(
            privileged_future_sids,
            privileged_future_padding_mask,
            batch_size=history_sids.shape[0],
            device=history_sids.device,
        )
        if future_codes is not None and target_sids is None:
            raise ValueError("privileged future prefix requires target_sids")
        if future_codes is not None and self.config.behavior_instruction_vocab_size == 0:
            raise ValueError("EA-TOSD privileged teacher requires v2 behavior instruction")
        if not 1 <= output_items <= self.config.max_target_items:
            raise ValueError("output_items must be within configured max_target_items")
        batch_size = history_sids.shape[0]
        instruction_ids = self._default_ids(instruction_ids, batch_size, history_sids.device)
        scenario_ids = self._default_ids(scenario_ids, batch_size, history_sids.device)
        # 1) 从短历史得到可选上下文摘要，再与 scenario/reasoning 合成 query。
        history_context = self._history_context(
            history_sids, history_padding_mask, scenario_ids
        )
        scenario_prompt, reasoning_prompt, query = self._instruction_prompt(
            scenario_ids, instruction_ids, instruction_features, trigger_sids,
            history_context,
        )
        behavior_prompt = self._behavior_instruction_prompt(
            behavior_instruction_ids, batch_size, history_sids.device
        )
        # 2) IGR 从长历史选 K 个 SID，拼到短历史后形成 Encoder 输入。
        encoder_sids, encoder_mask, igr_indices, igr_scores = self._augment_history(
            history_sids, history_padding_mask, long_history_sids,
            long_history_padding_mask, query,
            long_history_behavior_ids=long_history_behavior_ids,
            retrieval_plans=retrieval_plans,
            retrieval_mode=retrieval_mode,
        )
        if long_history_sids is not None and history_behavior_ids is not None:
            raise ValueError("behavior-conditioned IGR requires long-history behavior IDs")
        # 3) Encoder: [B,T(+K),L] -> memory [B,T(+K),H]。
        memory = self._encode(encoder_sids, encoder_mask, history_behavior_ids)
        if target_sids is None:
            logits = self._autoregressive_logits(
                memory,
                encoder_mask,
                scenario_prompt,
                reasoning_prompt,
                behavior_prompt,
                output_items,
            )
            return OxygenRECOutput(logits=logits, igr_indices=igr_indices, igr_scores=igr_scores)
        # 4) teacher forcing：列表[B,N,L]按商品顺序展平为[B,3N]。
        flat_targets = self._flatten_target_sids(target_sids)
        prefix = flat_targets[:, :-1]
        hidden = self._decode(
            memory, encoder_mask, scenario_prompt, reasoning_prompt,
            behavior_prompt, prefix,
            privileged_future_codes=future_codes,
            privileged_future_token_mask=future_token_mask,
        )
        prediction_offset = (3 if behavior_prompt is not None else 2) + (
            0 if future_codes is None else future_codes.shape[1]
        )
        logits = tuple(
            self.prediction_heads[step % self.config.sid_levels](
                hidden[:, step + prediction_offset, :]
            )
            for step in range(flat_targets.shape[1])
        )
        ntp_loss, level_losses = self.weighted_ntp_loss(
            logits,
            flat_targets,
            level_weights,
            sample_weights=sample_weights,
            token_weights=token_weights,
        )
        loss = ntp_loss
        q2i_loss = alignment_loss = q2i_cosine = None
        # 5) Q2I 让 query 靠近目标商品向量；总损失=NTP+权重*Q2I。
        if self.config.q2i_weight > 0:
            if target_sids.ndim != 2:
                raise ValueError("Q2I listwise target alignment is not implemented")
            targets = F.normalize(self.item_adapter(self._item_embedding(target_sids)), dim=-1)
            q2i_cosine = (query * targets).sum(dim=-1)
            q2i_loss, alignment_loss = self.q2i_alignment_loss(query, targets)
            loss = ntp_loss + self.config.q2i_weight * q2i_loss
        return OxygenRECOutput(
            logits=logits, loss=loss, level_losses=level_losses, ntp_loss=ntp_loss,
            q2i_loss=q2i_loss, q2i_alignment_loss=alignment_loss,
            q2i_cosine=q2i_cosine,
            igr_indices=igr_indices, igr_scores=igr_scores,
        )

    def _autoregressive_logits(
        self,
        memory: Tensor,
        memory_padding_mask: Tensor,
        scenario_prompt: Tensor,
        reasoning_prompt: Tensor,
        behavior_prompt: Tensor | None,
        output_items: int,
    ) -> tuple[Tensor, ...]:
        """无target时按level循环，贪心产生N个商品的3N组logits。"""
        prefix = torch.empty(
            (memory.shape[0], 0), dtype=torch.long, device=memory.device
        )
        outputs = []
        total_steps = output_items * self.config.sid_levels
        for step in range(total_steps):
            level = step % self.config.sid_levels
            head = self.prediction_heads[level]
            hidden = self._decode(
                memory, memory_padding_mask, scenario_prompt, reasoning_prompt,
                behavior_prompt, prefix,
            )
            prediction_offset = 3 if behavior_prompt is not None else 2
            logits = head(hidden[:, step + prediction_offset, :])
            outputs.append(logits)
            prefix = torch.cat((prefix, logits.argmax(dim=-1, keepdim=True)), dim=1)
        return tuple(outputs)

    def _encode(
        self, history_sids: Tensor, padding_mask: Tensor,
        behavior_ids: Tensor | None = None,
    ) -> Tensor:
        """把历史 SID/位置/可选行为 embedding 相加后送入 Encoder。"""
        _, history_length, _ = history_sids.shape
        positions = torch.arange(history_length, device=history_sids.device)
        hidden = self.history_positions(positions).unsqueeze(0)
        # 每个历史商品的三层 SID embedding 求和，而不是沿层维拼接。
        hidden = hidden + sum(
            embedding(history_sids[:, :, level])
            for level, embedding in enumerate(self.sid_embeddings)
        )
        if behavior_ids is not None:
            if self.behavior_embeddings is None:
                raise ValueError("behavior_vocab_size must be configured")
            if behavior_ids.shape != history_sids.shape[:2] or behavior_ids.dtype != torch.long:
                raise ValueError("history_behavior_ids must be torch.long [batch, history]")
            behavior_hidden = self.behavior_embeddings(behavior_ids)
            if self.config.behavior_time_decay > 0:
                valid = ~padding_mask
                age = torch.flip(
                    torch.cumsum(torch.flip(valid.to(hidden.dtype), dims=(1,)), dim=1),
                    dims=(1,),
                ) - 1.0
                decay = torch.exp(-self.config.behavior_time_decay * age.clamp_min(0.0))
                behavior_hidden = behavior_hidden * decay.unsqueeze(-1)
            hidden = hidden + behavior_hidden
        return self.encoder(self.dropout(hidden), src_key_padding_mask=padding_mask)

    def _item_embedding(self, sids: Tensor) -> Tensor:
        """将任意前导形状的 SID [...,L] 转成商品向量 [...,H]。"""
        return sum(
            embedding(sids[..., level])
            for level, embedding in enumerate(self.sid_embeddings)
        )

    def _flatten_target_sids(self, target_sids: Tensor) -> Tensor:
        """把v1的[B,L]或v2的[B,N,L]目标统一转成[B,3N]。"""
        if target_sids.ndim == 2:
            return target_sids
        return target_sids.reshape(target_sids.shape[0], -1)

    def _prepare_privileged_future(
        self,
        future_sids: Tensor | None,
        future_padding_mask: Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[Tensor | None, Tensor | None]:
        """校验Teacher未来商品并展平为固定的SID token块``[B,3M]``。

        ``future_padding_mask=True``表示整个未来商品是padding。显式mask避免把
        合法SID code 0误当成padding；被mask商品会先规范化成固定占位SID，再被
        attention key mask隔离。固定占位很重要：Teacher用未来块末端位置预测首个
        目标token，不能让调用方填入的任意padding内容改变该读出状态。
        """
        if future_sids is None:
            if future_padding_mask is not None:
                raise ValueError(
                    "privileged_future_padding_mask requires privileged_future_sids"
                )
            return None, None
        if self.config.max_future_items == 0:
            raise ValueError("max_future_items must be configured for EA-TOSD teacher")
        if (
            future_sids.ndim != 3
            or future_sids.shape[0] != batch_size
            or future_sids.shape[2] != self.config.sid_levels
            or not 1 <= future_sids.shape[1] <= self.config.max_future_items
            or future_sids.dtype != torch.long
        ):
            raise ValueError(
                "privileged_future_sids must be torch.long [batch, future_items, levels] "
                "within max_future_items"
            )
        if (
            future_padding_mask is None
            or future_padding_mask.shape != future_sids.shape[:2]
            or future_padding_mask.dtype != torch.bool
        ):
            raise ValueError(
                "privileged_future_padding_mask must be boolean [batch, future_items]"
            )
        if (future_sids < 0).any() or (future_sids >= self.config.sid_width).any():
            raise ValueError("privileged_future_sids contains a code outside the SID vocabulary")
        future_sids = future_sids.to(device)
        future_padding_mask = future_padding_mask.to(device)
        # padding位置既参与固定长度布局，又可能充当首token预测的query位置。
        # 因此除了key mask，还必须把其输入embedding规范化为确定值。
        canonical_future_sids = future_sids.masked_fill(
            future_padding_mask.unsqueeze(-1), 0
        )
        return (
            canonical_future_sids.reshape(batch_size, -1),
            future_padding_mask.repeat_interleave(self.config.sid_levels, dim=1),
        )

    @staticmethod
    def _default_ids(ids: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        """缺省 instruction/scenario 使用 ID 0，并校验显式输入形状。"""
        if ids is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if ids.shape != (batch_size,) or ids.dtype != torch.long:
            raise ValueError("instruction/scenario IDs must be torch.long with shape [batch]")
        return ids

    def _instruction_prompt(
        self, scenario_ids: Tensor, instruction_ids: Tensor,
        instruction_features: Tensor | None, trigger_sids: Tensor | None,
        history_context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """融合 scenario、reasoning、trigger 和历史摘要，得到 Q2I/IGR query。"""
        batch_size = scenario_ids.shape[0]
        scenario = self.scenario_embeddings(scenario_ids)
        if trigger_sids is not None:
            if trigger_sids.shape != (batch_size, self.config.sid_levels):
                raise ValueError("trigger_sids must have shape [batch, levels]")
            scenario = scenario + self._item_embedding(trigger_sids)
        if instruction_features is None:
            reasoning = self.instruction_embeddings(instruction_ids)
        else:
            if self.instruction_feature_adapter is None:
                raise ValueError("instruction_feature_size must be configured")
            if instruction_features.shape != (batch_size, self.config.instruction_feature_size):
                raise ValueError("instruction_features has the wrong shape")
            reasoning = self.instruction_feature_adapter(instruction_features)
        if history_context is not None:
            reasoning = reasoning + history_context
        # query=[B,Q] 会同时用于长历史 IGR 相似度和目标商品 Q2I 对齐。
        query = F.normalize(self.query_adapter(torch.cat((scenario, reasoning), dim=-1)), dim=-1)
        return scenario, reasoning, query

    def _behavior_instruction_prompt(
        self,
        behavior_instruction_ids: Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> Tensor | None:
        """把目标行为 ID 映射为 v2 Decoder 前缀中的行为指令 ``I_b``。

        行为 token 使用第一层 Decoder/SID embedding 表中位于 ``V`` 之后的
        保留行，再经论文规定的两层 LeakyReLU 投影 ``psi`` 得到 [B,H]。
        这与 ``_encode()`` 中描述历史每一步行为的 embedding 完全分离。
        """
        configured = self.config.behavior_instruction_vocab_size
        if configured == 0:
            if behavior_instruction_ids is not None:
                raise ValueError(
                    "behavior_instruction_vocab_size must be configured for v2 I_b"
                )
            return None
        if behavior_instruction_ids is None:
            raise ValueError(
                "behavior_instruction_ids are required when v2 I_b is configured"
            )
        if (
            behavior_instruction_ids.shape != (batch_size,)
            or behavior_instruction_ids.dtype != torch.long
        ):
            raise ValueError(
                "behavior_instruction_ids must be torch.long with shape [batch]"
            )
        if (
            (behavior_instruction_ids < 0).any()
            or (behavior_instruction_ids >= configured).any()
        ):
            raise ValueError("behavior_instruction_ids contain an unknown behavior")
        if self.behavior_instruction_adapter is None:  # pragma: no cover - 构造期保证
            raise RuntimeError("behavior instruction adapter was not initialized")
        reserved_token_ids = behavior_instruction_ids.to(device) + self.config.sid_width
        behavior_embedding = self.sid_embeddings[0](reserved_token_ids)
        return self.behavior_instruction_adapter(behavior_embedding)

    def _history_context(
        self, history_sids: Tensor, padding_mask: Tensor, scenario_ids: Tensor
    ) -> Tensor | None:
        """对短历史做 masked mean 或 scenario-conditioned attention 摘要。"""
        if not self.config.use_history_context_instruction:
            return None
        items = self._item_embedding(history_sids)
        valid = (~padding_mask).unsqueeze(-1).to(items.dtype)
        if self.config.history_context_pooling == "mean":
            pooled = (items * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        else:
            scenario = self.scenario_embeddings(scenario_ids)
            query = self.history_context_query(scenario)
            keys = self.history_context_key(items)
            scores = torch.einsum("bd,bhd->bh", query, keys)
            scores = scores / self.config.hidden_size**0.5
            scores = scores.masked_fill(padding_mask, float("-inf"))
            weights = scores.softmax(dim=1)
            pooled = torch.einsum(
                "bh,bhd->bd", weights, self.history_context_value(items)
            )
        return self.history_context_adapter(pooled)

    def _augment_history(
        self, short_sids: Tensor, short_mask: Tensor, long_sids: Tensor | None,
        long_mask: Tensor | None, query: Tensor, *,
        long_history_behavior_ids: Tensor | None = None,
        retrieval_plans: Sequence[ExecutableRetrievalPlan] | None = None,
        retrieval_mode: RetrievalMode = PAPER_IGR,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """用 query 检索长历史并把 top-k SID 拼接到短历史。

        short_sids=[B,T_short,L]，long_sids=[B,T_long,L]，query=[B,Q]；
        返回的历史长度为 T_short+K。没有长历史时原样返回短历史。
        """
        if retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"unsupported retrieval_mode: {retrieval_mode!r}")
        if retrieval_mode == PAPER_IGR and retrieval_plans is not None:
            raise ValueError("paper_igr does not consume retrieval_plans")
        if retrieval_mode == AGENTIC_PLAN and retrieval_plans is None:
            raise ValueError("agentic_plan requires retrieval_plans")
        if long_sids is None:
            if long_mask is not None:
                raise ValueError("long_history_padding_mask requires long_history_sids")
            if retrieval_mode == AGENTIC_PLAN:
                raise ValueError("agentic_plan requires long_history_sids")
            return short_sids, short_mask, None, None
        if self.config.igr_top_k < 1:
            raise ValueError("igr_top_k must be positive when long history is provided")
        if long_sids.ndim != 3 or long_sids.shape[0] != short_sids.shape[0] or long_sids.shape[2] != self.config.sid_levels:
            raise ValueError("long_history_sids must have shape [batch, long_history, levels]")
        if long_mask is None or long_mask.shape != long_sids.shape[:2] or long_mask.dtype != torch.bool:
            raise ValueError("long_history_padding_mask must be boolean [batch, long_history]")
        if ((~long_mask).sum(dim=1) < self.config.igr_top_k).any():
            raise ValueError("each long history must contain at least igr_top_k valid items")
        # 论文中的长历史检索分支保持冻结：detach 阻止 IGR 反向更新 SID/item adapter。
        long_vectors = F.normalize(
            self.item_adapter(self._item_embedding(long_sids)).detach(), dim=-1
        )
        # 归一化 query 与 item 做点积即 cosine，padding 位置设为 -inf。
        scores = torch.einsum("bd,bhd->bh", query, long_vectors).masked_fill(long_mask, float("-inf"))
        if retrieval_mode == PAPER_IGR:
            top_scores, top_indices = scores.topk(self.config.igr_top_k, dim=1)
        else:
            if long_history_behavior_ids is None:
                raise ValueError("retrieval plans require long_history_behavior_ids")
            if long_history_behavior_ids.shape != scores.shape:
                raise ValueError("long_history_behavior_ids must match the long-history window")
            top_indices, top_scores, _ = execute_retrieval_plan(
                scores, long_sids, long_history_behavior_ids, long_mask,
                retrieval_plans, top_k=self.config.igr_top_k,
            )
        # 按 top-k 索引取回真实三层 SID，并追加到短历史尾部。
        gathered = long_sids.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, self.config.sid_levels))
        combined = torch.cat((short_sids, gathered), dim=1)
        combined_mask = torch.cat((short_mask, torch.zeros_like(top_indices, dtype=torch.bool)), dim=1)
        return combined, combined_mask, top_indices, top_scores

    def _decode(
        self,
        memory: Tensor,
        memory_padding_mask: Tensor,
        scenario_prompt: Tensor,
        reasoning_prompt: Tensor,
        behavior_prompt: Tensor | None,
        prefix_codes: Tensor | None,
        *,
        privileged_future_codes: Tensor | None = None,
        privileged_future_token_mask: Tensor | None = None,
    ) -> Tensor:
        """执行causal Decoder；EA-TOSD Teacher在I_b后插入训练期未来SID块。"""
        batch_size = memory.shape[0]
        bos = self.bos_embedding.unsqueeze(0).expand(batch_size, -1)
        if behavior_prompt is None:
            # 保留冻结 v1 checkpoint 的既有布局和位置语义。
            tokens = [scenario_prompt, reasoning_prompt, bos]
        else:
            # v2 论文式前缀：[BOS, I_s, I_r, I_b]；I_b 的位置预测首层 SID。
            tokens = [bos, scenario_prompt, reasoning_prompt, behavior_prompt]
        base_prefix_length = len(tokens)
        if privileged_future_codes is not None:
            if privileged_future_token_mask is None:
                raise ValueError("privileged future codes require a token padding mask")
            tokens.extend(
                self.sid_embeddings[step % self.config.sid_levels](
                    privileged_future_codes[:, step]
                )
                for step in range(privileged_future_codes.shape[1])
            )
        if prefix_codes is not None:
            tokens.extend(
                self.sid_embeddings[step % self.config.sid_levels](
                    prefix_codes[:, step]
                )
                for step in range(prefix_codes.shape[1])
            )
        hidden = torch.stack(tokens, dim=1)
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        hidden = hidden + self.decoder_positions(positions).unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones(
                hidden.shape[1], hidden.shape[1], dtype=torch.bool, device=hidden.device
            ),
            diagonal=1,
        )
        target_padding_mask = None
        if privileged_future_token_mask is not None:
            suffix_length = 0 if prefix_codes is None else prefix_codes.shape[1]
            target_padding_mask = torch.cat(
                (
                    torch.zeros(
                        batch_size,
                        base_prefix_length,
                        dtype=torch.bool,
                        device=hidden.device,
                    ),
                    privileged_future_token_mask,
                    torch.zeros(
                        batch_size,
                        suffix_length,
                        dtype=torch.bool,
                        device=hidden.device,
                    ),
                ),
                dim=1,
            )
        return self.decoder(
            self.dropout(hidden),
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )

    def q2i_alignment_loss(self, queries: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        """Q2I：余弦对齐，并加入方差保持与 batch 内去相关正则。"""
        if queries.shape != targets.shape or queries.ndim != 2:
            raise ValueError("queries and targets must share shape [batch, dimension]")
        batch_size = queries.shape[0]
        # 两侧都已 L2 normalize，点积就是 cosine；取负号后最小化即拉近。
        alignment = -(queries * targets).sum(dim=-1).mean()
        if batch_size < 2:
            return alignment, alignment
        variance_product = (
            queries.var(dim=0, unbiased=False).mean()
            * targets.var(dim=0, unbiased=False).mean()
        ).clamp_min(1e-8)
        variance = -torch.log(variance_product)
        gram = queries @ queries.transpose(0, 1)
        off_diagonal = gram.square().sum() - gram.diagonal().square().sum()
        decorrelation = off_diagonal / (batch_size * batch_size - batch_size)
        total = (
            alignment
            + self.config.q2i_variance_weight * variance
            + self.config.q2i_decorrelation_weight * decorrelation
        )
        return total, alignment

    @staticmethod
    def weighted_ntp_loss(
        logits: Sequence[Tensor],
        target_sids: Tensor,
        level_weights: Sequence[float] | Tensor | None = None,
        *,
        sample_weights: Tensor | None = None,
        token_weights: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        """计算SID token的加权交叉熵，同时兼容v1样本权重和v2行为权重。

        v1 ``sample_weights`` 会按权重和归一化；v2 ``token_weights``=[B,3N]
        严格按论文式 ``mean(w_b * CE)`` 计算，因此不会消除行为权重的绝对尺度。
        两种权重语义不同，不允许在同一次前向中混用。
        """

        if sample_weights is not None and token_weights is not None:
            raise ValueError("sample_weights and token_weights cannot be combined")
        if sample_weights is not None:
            if sample_weights.shape != (target_sids.shape[0],):
                raise ValueError("sample_weights must have shape [batch]")
            if not torch.isfinite(sample_weights).all() or (sample_weights < 0).any() or sample_weights.sum() <= 0:
                raise ValueError("sample_weights must be finite, non-negative, and sum positive")
        if token_weights is not None:
            expected_shape = (target_sids.shape[0], len(logits))
            if token_weights.shape != expected_shape:
                raise ValueError("token_weights must have shape [batch, SID tokens]")
            if (
                not torch.isfinite(token_weights).all()
                or (token_weights < 0).any()
                or token_weights.sum() <= 0
            ):
                raise ValueError(
                    "token_weights must be finite, non-negative, and sum positive"
                )
        per_level = tuple(
            F.cross_entropy(level_logits, target_sids[:, level], reduction="none")
            for level, level_logits in enumerate(logits)
        )
        level_losses = tuple(
            (
                losses.mean()
                if sample_weights is None and token_weights is None
                else (losses * sample_weights).sum() / sample_weights.sum()
                if sample_weights is not None
                else (losses * token_weights[:, level]).mean()
            )
            for level, losses in enumerate(per_level)
        )
        if level_weights is None:
            weights = target_sids.new_ones(len(level_losses), dtype=torch.float32)
        else:
            weights = torch.as_tensor(
                level_weights, dtype=level_losses[0].dtype, device=target_sids.device
            )
        if weights.shape != (len(level_losses),):
            raise ValueError("level_weights must contain one value per SID level")
        if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("level_weights must be finite, non-negative, and sum positive")
        loss = sum(weight * item for weight, item in zip(weights, level_losses))
        return loss / weights.sum(), level_losses

    def candidate_log_probs(
        self,
        history_sids: Tensor,
        history_padding_mask: Tensor,
        candidate_sids: Tensor,
        *,
        history_behavior_ids: Tensor | None = None,
        instruction_ids: Tensor | None = None,
        scenario_ids: Tensor | None = None,
        behavior_instruction_ids: Tensor | None = None,
        instruction_features: Tensor | None = None,
        trigger_sids: Tensor | None = None,
        long_history_sids: Tensor | None = None,
        long_history_padding_mask: Tensor | None = None,
        long_history_behavior_ids: Tensor | None = None,
        retrieval_plans: Sequence[ExecutableRetrievalPlan] | None = None,
        retrieval_mode: RetrievalMode = PAPER_IGR,
    ) -> Tensor:
        """对固定候选做teacher forcing，返回每个SID token的log-prob [B,G,3N]。"""
        if candidate_sids.ndim != 3:
            raise ValueError("candidate_sids must have shape [batch, group, SID tokens]")
        batch, group, sid_tokens = candidate_sids.shape
        if (
            batch != history_sids.shape[0]
            or sid_tokens % self.config.sid_levels
            or not 1 <= sid_tokens // self.config.sid_levels <= self.config.max_target_items
        ):
            raise ValueError("candidate batch/token count does not match model inputs")
        output_items = sid_tokens // self.config.sid_levels
        expanded_history = history_sids[:, None].expand(-1, group, -1, -1).reshape(
            batch * group, history_sids.shape[1], self.config.sid_levels
        )
        expanded_mask = history_padding_mask[:, None].expand(-1, group, -1).reshape(
            batch * group, history_padding_mask.shape[1]
        )
        flat_targets = candidate_sids.reshape(batch * group, sid_tokens)
        targets = (
            flat_targets
            if output_items == 1
            else flat_targets.reshape(batch * group, output_items, self.config.sid_levels)
        )
        expanded_plans = None
        if retrieval_plans is not None:
            if len(retrieval_plans) != batch:
                raise ValueError("one executable plan is required per batch row")
            expanded_plans = [plan for plan in retrieval_plans for _ in range(group)]
        def expand_vector(value: Tensor | None) -> Tensor | None:
            if value is None:
                return None
            return value[:, None].expand(-1, group).reshape(batch * group)

        def expand_features(value: Tensor | None) -> Tensor | None:
            if value is None:
                return None
            return value[:, None].expand(-1, group, *value.shape[1:]).reshape(
                batch * group, *value.shape[1:]
            )

        output = self(
            expanded_history, expanded_mask, target_sids=targets,
            history_behavior_ids=expand_features(history_behavior_ids),
            instruction_ids=expand_vector(instruction_ids),
            scenario_ids=expand_vector(scenario_ids),
            behavior_instruction_ids=expand_vector(behavior_instruction_ids),
            instruction_features=expand_features(instruction_features),
            trigger_sids=expand_features(trigger_sids),
            long_history_sids=expand_features(long_history_sids),
            long_history_padding_mask=expand_features(long_history_padding_mask),
            long_history_behavior_ids=expand_features(long_history_behavior_ids),
            retrieval_plans=expanded_plans,
            retrieval_mode=retrieval_mode,
        )
        selected = []
        for step, logits in enumerate(output.logits):
            selected.append(
                F.log_softmax(logits, dim=-1).gather(
                    1, flat_targets[:, step : step + 1]
                ).squeeze(1)
            )
        return torch.stack(selected, dim=-1).reshape(batch, group, sid_tokens)

    @torch.no_grad() # generate只负责推理生成SID，不进行参数更新，不需要计算图，所以使用这个装饰器来关闭Pytorch的梯度记录
    def generate(
        self,
        history_sids: Tensor,
        history_padding_mask: Tensor,
        trie: PrefixTrie,
        *,
        history_behavior_ids: Tensor | None = None,
        instruction_ids: Tensor | None = None,
        scenario_ids: Tensor | None = None,
        behavior_instruction_ids: Tensor | None = None,
        instruction_features: Tensor | None = None,
        trigger_sids: Tensor | None = None,
        long_history_sids: Tensor | None = None,
        long_history_padding_mask: Tensor | None = None,
        long_history_behavior_ids: Tensor | None = None,
        retrieval_plans: Sequence[ExecutableRetrievalPlan] | None = None,
        retrieval_mode: RetrievalMode = PAPER_IGR,
        output_items: int = 1,
    ) -> Tensor:
        """用PrefixTrie逐商品屏蔽非法code，贪心生成展平的[B,3N]。"""

        self._validate_inputs(history_sids, history_padding_mask, None)
        if not 1 <= output_items <= self.config.max_target_items:
            raise ValueError("output_items must be within configured max_target_items")
        batch_size = history_sids.shape[0]
        instruction_ids = self._default_ids(instruction_ids, batch_size, history_sids.device)
        scenario_ids = self._default_ids(scenario_ids, batch_size, history_sids.device)
        history_context = self._history_context(
            history_sids, history_padding_mask, scenario_ids
        )
        scenario_prompt, reasoning_prompt, query = self._instruction_prompt(
            scenario_ids, instruction_ids, instruction_features, trigger_sids,
            history_context,
        )
        behavior_prompt = self._behavior_instruction_prompt(
            behavior_instruction_ids, batch_size, history_sids.device
        )
        encoder_sids, encoder_mask, _, _ = self._augment_history(
            history_sids, history_padding_mask, long_history_sids,
            long_history_padding_mask, query,
            long_history_behavior_ids=long_history_behavior_ids,
            retrieval_plans=retrieval_plans,
            retrieval_mode=retrieval_mode,
        )
        if long_history_sids is not None and history_behavior_ids is not None:
            raise ValueError("behavior-conditioned IGR requires long-history behavior IDs")
        memory = self._encode(encoder_sids, encoder_mask, history_behavior_ids)
        generated = torch.empty(
            (batch_size, 0), dtype=torch.long, device=history_sids.device
        )
        total_steps = output_items * self.config.sid_levels
        for step in range(total_steps):
            level = step % self.config.sid_levels
            hidden = self._decode(
                memory, encoder_mask, scenario_prompt, reasoning_prompt,
                behavior_prompt, generated,
            )
            prediction_offset = 3 if behavior_prompt is not None else 2
            logits = self.prediction_heads[level](
                hidden[:, step + prediction_offset, :]
            )
            selected = []
            for row in range(batch_size):
                # 每三个token开始一个新商品；列表上下文仍保留在Decoder prefix中。
                item_start = step - level
                item_prefix = tuple(
                    int(code) for code in generated[row, item_start:].tolist()
                )
                allowed = trie.allowed_next(item_prefix)
                if not allowed:
                    raise ValueError(
                        f"trie has no legal continuation for prefix {item_prefix}"
                    )
                allowed_tensor = torch.tensor(
                    allowed, dtype=torch.long, device=logits.device
                )
                best = logits[row, allowed_tensor].argmax()
                selected.append(allowed_tensor[best])
            generated = torch.cat((generated, torch.stack(selected).unsqueeze(1)), dim=1)
        return generated

    @torch.no_grad()
    def sample_trajectories(
        self,
        history_sids: Tensor,
        history_padding_mask: Tensor,
        trie: PrefixTrie,
        *,
        group_size: int,
        history_behavior_ids: Tensor | None = None,
        instruction_ids: Tensor | None = None,
        scenario_ids: Tensor | None = None,
        behavior_instruction_ids: Tensor | None = None,
        instruction_features: Tensor | None = None,
        trigger_sids: Tensor | None = None,
        long_history_sids: Tensor | None = None,
        long_history_padding_mask: Tensor | None = None,
        long_history_behavior_ids: Tensor | None = None,
        retrieval_plans: Sequence[ExecutableRetrievalPlan] | None = None,
        retrieval_mode: RetrievalMode = PAPER_IGR,
        output_items: int = 1,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """从当前Student按PrefixTrie约束采样G条on-policy轨迹``[B,G,3N]``。

        每条轨迹独立采样，允许重复；论文只要求从当前策略采样候选组，并未规定
        组内去重。未来信息不会进入本函数，因而不会泄漏到部署Student。
        """
        if group_size < 1:
            raise ValueError("group_size must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self._validate_inputs(history_sids, history_padding_mask, None)
        if not 1 <= output_items <= self.config.max_target_items:
            raise ValueError("output_items must be within configured max_target_items")

        batch_size = history_sids.shape[0]
        instruction_ids = self._default_ids(
            instruction_ids, batch_size, history_sids.device
        )
        scenario_ids = self._default_ids(
            scenario_ids, batch_size, history_sids.device
        )
        history_context = self._history_context(
            history_sids, history_padding_mask, scenario_ids
        )
        scenario_prompt, reasoning_prompt, query = self._instruction_prompt(
            scenario_ids,
            instruction_ids,
            instruction_features,
            trigger_sids,
            history_context,
        )
        behavior_prompt = self._behavior_instruction_prompt(
            behavior_instruction_ids, batch_size, history_sids.device
        )
        encoder_sids, encoder_mask, _, _ = self._augment_history(
            history_sids,
            history_padding_mask,
            long_history_sids,
            long_history_padding_mask,
            query,
            long_history_behavior_ids=long_history_behavior_ids,
            retrieval_plans=retrieval_plans,
            retrieval_mode=retrieval_mode,
        )
        if long_history_sids is not None and history_behavior_ids is not None:
            raise ValueError("behavior-conditioned IGR requires long-history behavior IDs")
        memory = self._encode(encoder_sids, encoder_mask, history_behavior_ids)

        # 将每个上下文复制G份，只做一次batched采样解码。
        memory = memory.repeat_interleave(group_size, dim=0)
        encoder_mask = encoder_mask.repeat_interleave(group_size, dim=0)
        scenario_prompt = scenario_prompt.repeat_interleave(group_size, dim=0)
        reasoning_prompt = reasoning_prompt.repeat_interleave(group_size, dim=0)
        if behavior_prompt is not None:
            behavior_prompt = behavior_prompt.repeat_interleave(group_size, dim=0)
        rollout_batch = batch_size * group_size
        generated = torch.empty(
            rollout_batch, 0, dtype=torch.long, device=history_sids.device
        )
        total_steps = output_items * self.config.sid_levels
        for step in range(total_steps):
            level = step % self.config.sid_levels
            hidden = self._decode(
                memory,
                encoder_mask,
                scenario_prompt,
                reasoning_prompt,
                behavior_prompt,
                generated,
            )
            prediction_offset = 3 if behavior_prompt is not None else 2
            logits = self.prediction_heads[level](
                hidden[:, step + prediction_offset, :]
            )
            selected = []
            item_start = step - level
            for row in range(rollout_batch):
                item_prefix = tuple(
                    int(code) for code in generated[row, item_start:].tolist()
                )
                allowed = trie.allowed_next(item_prefix)
                if not allowed:
                    raise ValueError(
                        f"trie has no legal continuation for prefix {item_prefix}"
                    )
                allowed_tensor = torch.tensor(
                    allowed, dtype=torch.long, device=logits.device
                )
                probabilities = F.softmax(
                    logits[row, allowed_tensor] / temperature, dim=-1
                )
                choice = torch.multinomial(
                    probabilities, 1, generator=generator
                ).squeeze(0)
                selected.append(allowed_tensor[choice])
            generated = torch.cat(
                (generated, torch.stack(selected).unsqueeze(1)), dim=1
            )
        return generated.reshape(batch_size, group_size, total_steps)

    @torch.no_grad()
    def beam_search(
        self,
        history_sids: Tensor,
        history_padding_mask: Tensor,
        trie: PrefixTrie,
        *,
        beam_width: int,
        history_behavior_ids: Tensor | None = None,
        instruction_ids: Tensor | None = None,
        scenario_ids: Tensor | None = None,
        behavior_instruction_ids: Tensor | None = None,
        instruction_features: Tensor | None = None,
        trigger_sids: Tensor | None = None,
        long_history_sids: Tensor | None = None,
        long_history_padding_mask: Tensor | None = None,
        long_history_behavior_ids: Tensor | None = None,
        retrieval_plans: Sequence[ExecutableRetrievalPlan] | None = None,
        retrieval_mode: RetrievalMode = PAPER_IGR,
        output_items: int = 1,
    ) -> BeamSearchOutput:
        """对3N序列做约束beam search；每个商品边界重置Trie前缀。"""

        if beam_width < 1:
            raise ValueError("beam_width must be positive")
        self._validate_inputs(history_sids, history_padding_mask, None)
        if not 1 <= output_items <= self.config.max_target_items:
            raise ValueError("output_items must be within configured max_target_items")
        batch_size = history_sids.shape[0]
        instruction_ids = self._default_ids(instruction_ids, batch_size, history_sids.device)
        scenario_ids = self._default_ids(scenario_ids, batch_size, history_sids.device)
        history_context = self._history_context(
            history_sids, history_padding_mask, scenario_ids
        )
        scenario_prompt, reasoning_prompt, query = self._instruction_prompt(
            scenario_ids, instruction_ids, instruction_features, trigger_sids,
            history_context,
        )
        behavior_prompt = self._behavior_instruction_prompt(
            behavior_instruction_ids, batch_size, history_sids.device
        )
        encoder_sids, encoder_mask, _, _ = self._augment_history(
            history_sids, history_padding_mask, long_history_sids,
            long_history_padding_mask, query,
            long_history_behavior_ids=long_history_behavior_ids,
            retrieval_plans=retrieval_plans,
            retrieval_mode=retrieval_mode,
        )
        if long_history_sids is not None and history_behavior_ids is not None:
            raise ValueError("behavior-conditioned IGR requires long-history behavior IDs")
        memory = self._encode(encoder_sids, encoder_mask, history_behavior_ids)
        all_paths: list[list[tuple[int, ...]]] = []
        all_scores: list[list[float]] = []
        for row in range(batch_size):
            beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
            total_steps = output_items * self.config.sid_levels
            for step in range(total_steps):
                level = step % self.config.sid_levels
                item_start = step - level
                candidates: list[tuple[tuple[int, ...], float]] = []
                for prefix, score in beams:
                    prefix_tensor = torch.tensor(
                        [prefix], dtype=torch.long, device=history_sids.device
                    )
                    hidden = self._decode(
                        memory[row : row + 1],
                        encoder_mask[row : row + 1],
                        scenario_prompt[row : row + 1],
                        reasoning_prompt[row : row + 1],
                        behavior_prompt[row : row + 1] if behavior_prompt is not None else None,
                        prefix_tensor,
                    )
                    prediction_offset = 3 if behavior_prompt is not None else 2
                    logits = self.prediction_heads[level](
                        hidden[:, step + prediction_offset, :]
                    )
                    log_probabilities = F.log_softmax(logits[0], dim=-1)
                    allowed = trie.allowed_next(prefix[item_start:])
                    if not allowed:
                        continue
                    candidates.extend(
                        (
                            prefix + (code,),
                            score + float(log_probabilities[code]),
                        )
                        for code in allowed
                    )
                if not candidates:
                    raise ValueError("trie has no complete path for beam search")
                candidates.sort(key=lambda item: (-item[1], item[0]))
                beams = candidates[:beam_width]
            all_paths.append([path for path, _ in beams])
            all_scores.append([score for _, score in beams])

        returned_beams = min(len(paths) for paths in all_paths)
        paths_tensor = torch.tensor(
            [paths[:returned_beams] for paths in all_paths],
            dtype=torch.long,
            device=history_sids.device,
        )
        scores_tensor = torch.tensor(
            [scores[:returned_beams] for scores in all_scores],
            dtype=torch.float32,
            device=history_sids.device,
        )
        return BeamSearchOutput(paths_tensor, scores_tensor)

    def _validate_inputs(
        self,
        history_sids: Tensor,
        history_padding_mask: Tensor,
        target_sids: Tensor | None,
    ) -> None:
        """统一校验主干输入的维度、dtype、padding 和 SID 取值范围。"""
        if history_sids.ndim != 3:
            raise ValueError("history_sids must have shape [batch, history, levels]")
        batch_size, history_length, levels = history_sids.shape
        if levels != self.config.sid_levels:
            raise ValueError(
                f"expected {self.config.sid_levels} SID levels, got {levels}"
            )
        if history_length > self.config.max_history_items:
            raise ValueError("history exceeds max_history_items")
        if history_padding_mask.shape != (batch_size, history_length):
            raise ValueError("history_padding_mask must have shape [batch, history]")
        if history_padding_mask.dtype != torch.bool:
            raise ValueError("history_padding_mask must be boolean")
        if history_padding_mask.all(dim=1).any():
            raise ValueError("every sample must contain at least one history item")
        if target_sids is not None:
            single_target = target_sids.shape == (batch_size, self.config.sid_levels)
            listwise_target = (
                target_sids.ndim == 3
                and target_sids.shape[0] == batch_size
                and target_sids.shape[2] == self.config.sid_levels
                and 1 <= target_sids.shape[1] <= self.config.max_target_items
            )
            if not single_target and not listwise_target:
                raise ValueError(
                    "target_sids must have shape [batch, levels] or "
                    "[batch, items, levels] within max_target_items"
                )
        for name, tensor in (("history_sids", history_sids), ("target_sids", target_sids)):
            if tensor is None:
                continue
            if tensor.dtype != torch.long:
                raise ValueError(f"{name} must use torch.long")
            if (tensor < 0).any() or (tensor >= self.config.sid_width).any():
                raise ValueError(f"{name} contains a code outside the SID vocabulary")