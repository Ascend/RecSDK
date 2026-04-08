# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
import abc
import math
import time
import os
from typing import Dict, List, Tuple, Optional
import logging
import torchrec
import torch
from torch.nn.functional import scaled_dot_product_attention

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
from typing import Any
import functools

NPU_ENABLE = os.environ.get('NPU_FLAG', False)
HAS_ATTN_FUSION_OPS = True
ENABLE_JAGGED_OPS = True
if os.environ.get("NPU_FLAG", "True") == "False":
    NPU_ENABLE = False
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
else:
    import torch_npu
    from torch_npu.profiler import profile
    torch.npu.manual_seed(42)
    torch.npu.manual_seed_all(42)
    torch_npu.npu.manual_seed(42)
    torch_npu.npu.manual_seed_all(42)

class initializing:
    def __init__(self, enter_fn: Callable[[], None] = None, exit_fn: Callable[[], None] = None):
        self.enter_fn = enter_fn
        self.exit_fn = exit_fn

    def __enter__(self):
        if callable(self.enter_fn):
            self.enter_fn()

    def __exit__(self, *exc_info):
        if callable(self.exit_fn):
            self.exit_fn()

def truncated_normal(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Truncated normal to initialize tensor weights. Values exceeding +-2 are truncated.

    参数:
    - x: Tensor to be initialized
    - mean: mean of normal distribution for initialization
    - std: standard deviation of normal distribution for initialization

    返回:
    - initialized tensor
    """
    with torch.no_grad():
        size = x.shape
        tmp = x.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        x.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        x.data.mul_(std).add_(mean)
        return x
def weird_division(x, y):
    """
    :param x: 被除数
    :param y: 除数
    :return x/y: 商
    """
    if y < Const.EPS and y >= 0:
        y = Const.EPS
    elif y > -Const.EPS and y < 0:
        y = -Const.EPS
    return x / y


class Const:
    BEST_LOSS = 1e+7
    EPS = 1e-6
    TransformerCacheState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    LABEL_DICT = []
    MODULE_NAME = "name"
    IS_CUSTOMIZE = "is_customize"
    PATH = "path"
    CLS_NAME = "cls_name"
    HP = "hp"
    SUB_MODELS = "sub_models"
    IS_POST_INIT = "is_post_init"
    MODEL_CFG = "model_cfg"
    COMMON_HP = "common_hp"
    EXPECTED_NUM_UNIQUE_ITEMS = 499999

class Dense2Jagged(torch.autograd.Function):
    """For Dense_to_Jagged op"""

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        x, x_offsets, dense_length, jagged_length = args

        x_jagged = torch.ops.mxrec.dense_to_jagged(x, [x_offsets], jagged_length)[0]

        ctx.x_offsets = x_offsets
        ctx.dense_length = dense_length
        return x_jagged

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_dense = torch.ops.mxrec.jagged_to_padded_dense(grad_outputs[0], [ctx.x_offsets], ctx.dense_length, 0.0)

        return grad_dense, None, None, None


class Jagged2Dense(torch.autograd.Function):
    """For Jagged_to_Dense op"""

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        x, x_offsets, dense_length, jagged_length = args

        ctx.x_offsets = x_offsets
        ctx.jagged_length = jagged_length
        x_dense = torch.ops.mxrec.jagged_to_padded_dense(x, [x_offsets], dense_length, 0.0)

        return x_dense

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_jagged = torch.ops.mxrec.jagged_to_padded_dense_backward(grad_outputs[0],
                                                                      [ctx.x_offsets], ctx.jagged_length)

        return grad_jagged, None, None, None


def dense_to_jagged(dense: torch.Tensor, offsets: torch.Tensor, dense_length: int, jagged_length: int):
    return Dense2Jagged.apply(dense, offsets, dense_length, jagged_length)


def jagged_to_padded_dense(values: torch.Tensor, offsets: torch.Tensor, dense_length: int, jagged_length: int):
    return Jagged2Dense.apply(values, offsets, dense_length, jagged_length)


TransformerCacheState = Const.TransformerCacheState


class OneTrans(nn.Module):
    def __init__(self, model_cfg: Dict, common_hp: Dict, device):
        super().__init__()
        self.device = device
        self._verbose = model_cfg[Const.HP].get("verbose", True)
        self.num_nonseq_feat = model_cfg[Const.HP].get("num_nonseq_feat", None)

        model_conf = common_hp["model_conf"]

        feat_conf = common_hp["feature_conf"]

        self.input_propcessor_module = UserItemRatingInputFeaturePreprocessorLonger(model_cfg, common_hp, device)
        self.embedding_module = DistributeEmbeddingModuleWithSideInfoLonger(model_cfg, common_hp, device)
        self.embedding_module.to(self.device)

        self.sequence_model: SequentialModule = SequentialModule(model_cfg, common_hp)
        self.sequence_model.to(self.device)

        self.enable_compile = int(os.environ.get("ENABLE_COMPILE")) == 1
        self.enable_graph = int(os.environ.get("ENABLE_GRAPH")) == 1

        if self.enable_compile and self.enable_graph:
            print("即将开启compile+graph模式")
            self.sequence_model = torch.compile(self.sequence_model, backend="inductor", dynamic=False,
                                                mode="reduce-overhead")
        elif self.enable_compile:
            print("即将开启compile模式")
            self.sequence_model = torch.compile(self.sequence_model, backend="inductor", dynamic=False)

        self.attention_mask_module = AttentionMaskModule(model_cfg, common_hp)
        self.attention_mask_module.to(self.device)

        self.output_processor_module = LayerNormEmbeddingPostprocessor(model_cfg, common_hp)
        self.output_processor_module.to(self.device)

        self.infer_items_key = feat_conf.get("infer_items_key", "sequence_item_ids")
        self.hist_ts_key = feat_conf.get("raw_timestamps_column", 'history_timestamps')
        self.seq_feature_names = list(feat_conf.get("seq_feature_columns").keys())

        seq_feature_columns = feat_conf["seq_feature_columns"]
        self.group_size = common_hp["model_conf"].get("token_merge_group_size", 1)

        self._seq_len = [next(iter(seq_feature_columns.values()), {}).get("length", None)]
        self._seq_merged_len = [x["length"] for _, x in feat_conf["seq_feature_columns"].items()][0] // self.group_size
        self._seq_merged_sampled_len = model_conf.get("seq_sampled_len")

        self.balance_loss_coef = 1e-8
        self.reset_params()

    def reset_params(self):
        for name, params in self.named_parameters():
            if ("sequence_model" in name) or ("embedding_module" in name):
                if self._verbose:
                    logging.info("Skipping init for %s", name)
                continue
            try:
                torch.nn.init.xavier_normal_(params.data)
                if self._verbose:
                    logging.info("Initialize %s as xavier normal: %s params", name, params.data.shape[0])
            except Exception:
                if self._verbose:
                    logging.info("Failed to initialize %s: %s params", name, params.data.shape[0])

    def generate_input_seqence(
            self,
            model_inputs,
            past_lengths,
            num_rerank: int
    ) -> torch.Tensor:
        """
        综合序列信息，生成user, item1, action1, item2, action2...形式的输入序列。

        :return user_embeddings: 拼接后的输入给模型的token序列，形如user, item1, action1, item2, action2...
        :return past_embeddings: 原始的商品token序列，形如item1, item2, item3, ...
        :return all_timestamps: 时间戳序列
        """
        # item_feature_embs, user_feature_embs, seq_feature_embs = self.embedding_module.get_all_embeddings(model_inputs)
        # item_feature_embs = torch.rand((item_feature_embs.shape[0],self.num_nonseq_feat,item_feature_embs.shape[2]),device=seq_feature_embs.device)
        item_feature_embs, user_feature_embs, seq_feature_embs = self.embedding_module.get_all_embeddings(model_inputs)
        B = item_feature_embs.shape[0]
        H = item_feature_embs.shape[2]

        g_cpu = torch.Generator(device="cpu")
        g_cpu.manual_seed(1234)

        item_feature_embs = torch.rand(
            (B, self.num_nonseq_feat, H),
            generator=g_cpu,
            device="cpu",
            dtype=seq_feature_embs.dtype,
        ).to(seq_feature_embs.device)

        past_ids = [model_inputs[k] for k in self.seq_feature_names][0].squeeze()

        input_seq_embeddings, x_offsets, seq_offsets = self.input_propcessor_module(
            model_inputs=model_inputs,
            past_ids=[past_ids],
            num_rerank=num_rerank,
            past_lengths=past_lengths,
            user_feature_embs=user_feature_embs,
            item_feature_embs=item_feature_embs,
            seq_feature_embs=seq_feature_embs,
            deep_outputs=None,
        )

        device = past_lengths.device
        past_sum = torch.sum(past_lengths, dim=1)
        x_offsets = past_sum + num_rerank
        x_offsets = torch.cumsum(x_offsets, dim=0)
        x_offsets = torch.cat((torch.tensor([0], device=device), x_offsets))

        num_nonseq_feat = user_feature_embs.shape[1] + item_feature_embs.shape[1] if user_feature_embs is not None \
            else 0 + item_feature_embs.shape[1]

        return input_seq_embeddings, x_offsets, None, num_nonseq_feat

    def generate_user_embeddings(
            self,
            past_lengths: torch.Tensor,
            all_timestamps: torch.Tensor,
            seq_embeddings: torch.Tensor,
            cross_attn_mask: torch.Tensor,
            attn_mask: torch.Tensor,
            x_offsets: torch.Tensor,
            seq_offsets: torch.Tensor,
            cache: Optional[List[TransformerCacheState]] = None,
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            num_rerank: int = 1,
            num_nonseq_feat: int = 1,
    ) -> torch.Tensor:
        """
        综合序列信息，生成用户 embedding.
        [B, N] -> [B, N, D].
        """
        transformer_input = TransformerInput(
            x=seq_embeddings,
            all_timestamps=all_timestamps,
            invalid_attn_mask=attn_mask,
            invalid_cross_attn_mask=cross_attn_mask,
            past_lengths=past_lengths,
            num_rerank=num_rerank,
            offsets=x_offsets,
            return_cache_states=True,
        )
        item_embeddings, _ = self.sequence_model(
            transformer_input
        )
        item_embeddings = item_embeddings[:, - num_rerank * num_nonseq_feat:, :]

        return self.output_processor_module(item_embeddings)

    def set_hf32(self):
        if "npu" in self.device:
            import torch_npu
            torch_npu.npu.aclnn.allow_hf32 = True
            torch_npu.npu.conv.allow_hf32 = True
            torch_npu.npu.matmul.allow_hf32 = True

        elif "cuda" in self.device:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def forward(
            self,
            model_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        生成式推荐大模型前向传播过程

        :param model_input: 传入的字典，里面包括商品的特征id序列，用户的特征和其他序列信息。
        """
        past_ids = [model_inputs[k] for k in self.seq_feature_names][0].squeeze()
        past_lengths = []
        non_zero_count = (past_ids != 0).sum(dim=1)  # [B]
        merged_length = torch.ceil(non_zero_count.float() / self.group_size).long().unsqueeze(1)
        merged_length = merged_length.to(self.device)
        past_lengths.append(merged_length)
        past_lengths = torch.concat(past_lengths, dim=1)
        model_inputs.update(past_lengths=past_lengths)

        # num_rerank: target item的数量，取值大于或等于1， 当前约定训练时为1， 推理时大于等于1。
        num_rerank = model_inputs.get(self.infer_items_key).shape[-1]

        input_seq_embeddings, x_offsets, seq_offsets, num_nonseq_feat = self.generate_input_seqence(
            model_inputs=model_inputs,
            past_lengths=past_lengths,
            num_rerank=num_rerank)
        # TODO: fix cross attn mask and other components in longer
        cross_attn_mask = None

        # 使用 OneTransMask 模块作为掩码模块
        attn_mask = self.attention_mask_module(model_inputs, self._seq_len, num_rerank=num_rerank,
                                               num_user_tokens=num_nonseq_feat)

        attn_mask = attn_mask.unsqueeze(1)
        encoded_embeddings = self.generate_user_embeddings(past_lengths=past_lengths,
                                                           all_timestamps=model_inputs[self.hist_ts_key],
                                                           seq_embeddings=input_seq_embeddings,
                                                           cross_attn_mask=cross_attn_mask,
                                                           attn_mask=attn_mask,
                                                           x_offsets=x_offsets,
                                                           seq_offsets=seq_offsets,
                                                           num_rerank=num_rerank,
                                                           num_nonseq_feat=num_nonseq_feat)

        # results = self.feed_forward_module(encoded_embeddings)

        return encoded_embeddings
class SequentialModule(nn.Module):
    def __init__(self, model_cfg: Dict, common_hp: Dict):
        super().__init__()
        seq_module_conf = model_cfg[Const.HP]
        self.num_blocks = seq_module_conf.get("num_blocks", 8)

        self.num_heads = model_cfg[Const.HP].get("num_heads", 4)
        self.enable_fusion_ops = model_cfg[Const.HP].get("enable_fusion_ops", False)
        self.enable_jagged_ops = model_cfg[Const.HP].get("enable_jagged_ops", False)
        modules = [OneTransBlock(model_cfg, common_hp) for _ in range(self.num_blocks)]

        self._transformer = TransformerInner(
            modules=modules,
            num_head=self.num_heads,
            enable_fusion_ops=self.enable_fusion_ops,
            enable_jagged_ops=self.enable_jagged_ops,
            mask_type=3
        )

    def forward(
            self,
            ti: TransformerInput
    ) -> Tuple[torch.Tensor, List[TransformerCacheState]]:
        return self._transformer(
            ti
        )
class TransformerInner(torch.nn.Module):

    def __init__(
            self,
            modules: List[Transformer],
            num_head: int,
            enable_fusion_ops: bool,
            enable_jagged_ops: bool,
            mask_type: int

    ) -> None:
        super().__init__()
        self._attention_layers: torch.nn.ModuleList = torch.nn.ModuleList(modules=modules)
        self._use_rab = hasattr(modules[0], '_rel_attn_bias')
        self._num_heads: int = num_head

        self.enable_fusion_ops = enable_fusion_ops and HAS_ATTN_FUSION_OPS and self.training
        self.enable_jagged_ops = enable_jagged_ops and ENABLE_JAGGED_OPS and self.training
        self.mask_type = mask_type
        if not self.enable_fusion_ops and self.enable_jagged_ops:
            raise ValueError("It is impossible to set enable_jagged_ops to True while \
                                                        setting enable_fusion_ops to False")
        elif self.enable_fusion_ops and self.enable_jagged_ops:
            logging.info("Enable attn fusion ops with jagged mode.")
        elif self.enable_fusion_ops and not self.enable_jagged_ops:
            logging.info("Enable attn fusion ops with normal mode.")
        else:
            logging.info("Enable traditional einsum ops according to your configuration.")

    def prepare_transformer_input(self, ti: TransformerInput):

        bs = ti.bs
        invalid_attn_mask = ti.invalid_attn_mask_for_fused_operator
        if self._use_rab is not None:
            ti.ext_timestamps = (ti.all_timestamps.unsqueeze(2) - ti.all_timestamps.unsqueeze(1)).to(torch.float32)
        ti.mask_type = self.mask_type
        if self.enable_fusion_ops and self.mask_type == 3:
            if invalid_attn_mask.shape[0] == 1:
                invalid_attn_mask = invalid_attn_mask.repeat(bs, self._num_heads, 1, 1)
            else:
                invalid_attn_mask = invalid_attn_mask.repeat(1, self._num_heads, 1, 1)
            if invalid_attn_mask.dtype != torch.float32:
                invalid_attn_mask = invalid_attn_mask.to(torch.float32)
            ti.invalid_attn_mask_for_fused_operator = invalid_attn_mask
        if self.mask_type == 0:
            ti.invalid_attn_mask_for_fused_operator = None

        if ti.invalid_cross_attn_mask_for_fused_operator is not None:
            invalid_cross_attn_mask = ti.invalid_cross_attn_mask_for_fused_operator
            if self.enable_fusion_ops and self.mask_type == 3:
                if invalid_cross_attn_mask.shape[0] == 1:
                    invalid_cross_attn_mask = invalid_cross_attn_mask.repeat(bs, self._num_heads, 1, 1)
                else:
                    invalid_cross_attn_mask = invalid_cross_attn_mask.repeat(1, self._num_heads, 1, 1)
                if invalid_cross_attn_mask.dtype != torch.float32:
                    invalid_cross_attn_mask = invalid_cross_attn_mask.to(torch.float32)
                ti.invalid_cross_attn_mask_for_fused_operator = invalid_cross_attn_mask
            if self.mask_type == 0:
                ti.invalid_cross_attn_mask_for_fused_operator = None

        if self.enable_jagged_ops:
            ti.x = dense_to_jagged(ti.x, ti.x_offsets, ti.dense_length, ti.jagged_length)

        ti.fusion_enabled = self.enable_fusion_ops
        ti.jagged_enabled = self.enable_jagged_ops

    def update_transformer_input(self, ti: TransformerInput, x, layer_num, cache_states_i):
        ti.x = x
        ti.layer_num = layer_num
        if ti.return_cache_states:
            ti.cache_states.append(cache_states_i)

    def forward(
            self,
            ti: TransformerInput
    ) -> Tuple[torch.Tensor, List[TransformerCacheState]]:

        self.prepare_transformer_input(ti=ti)
        for i, layer in enumerate(self._attention_layers):
            x, cache_states_i, time_bias = layer(
                ti
            )
            self.update_transformer_input(ti, x, i + 1, cache_states_i)

        if self.enable_jagged_ops:
            x = jagged_to_padded_dense(
                ti.x,
                ti.x_offsets,
                ti.dense_length,
                ti.jagged_length
            )
        return x, ti.cache_states

class Transformer(nn.Module):
    def __init__(self, model_cfg: Dict, common_hp: Dict):
        super().__init__()
        model_conf = common_hp["model_conf"]
        feat_conf = common_hp["feature_conf"]
        train_conf = common_hp["train_conf"]
        seq_module_conf = model_cfg[Const.HP]

        # self._embedding_dim: int = model_conf.get("item_embedding_dim", 128)
        self._embedding_dim = int(os.environ.get("Dim"))

        self._linear_dim: int = seq_module_conf.get("dv", 32)
        self._attention_dim: int = seq_module_conf.get("dqk", 32)
        self._num_heads: int = seq_module_conf.get("num_heads", 4)
        self._linear_config: str = seq_module_conf.get("linear_config", "uvqk")
        self._linear_activation: str = seq_module_conf.get("linear_activation", "silu")

        self._max_sequence_length: int = model_conf.get("max_sequence_length", 512)

        self._norm_type: str = model_conf.get("norm_type", "RMS")

        self._eps: float = Const.EPS
        if self._linear_config == "uvqk":
            self._uvqk = torch.nn.Parameter(
                torch.empty((self._embedding_dim, self._linear_dim * 2 * self._num_heads +
                             self._attention_dim * self._num_heads * 2)).normal_(mean=0, std=0.02), )
            self._o = torch.nn.Linear(in_features=self._linear_dim * self._num_heads, out_features=self._embedding_dim)
            torch.nn.init.xavier_uniform_(self._o.weight)
        elif self._linear_config == "vqk":
            self.wq = torch.nn.Linear(self._embedding_dim, self._linear_dim * self._num_heads, bias=False)
            self.wk = torch.nn.Linear(self._embedding_dim, self._attention_dim * self._num_heads, bias=False)
            self.wv = torch.nn.Linear(self._embedding_dim, self._attention_dim * self._num_heads, bias=False)
            self._o = torch.nn.Linear(in_features=self._linear_dim * self._num_heads, out_features=self._embedding_dim)
            torch.nn.init.xavier_uniform_(self._o.weight)
        elif self._linear_config == "none":
            logging.info('No linear config, omitting linear layer')
        else:
            raise ValueError("Unknown linear_config %s", self._linear_config)

        if self._norm_type == "RMS":
            self.layer_norm_input = RMSNormNPU(self._embedding_dim, eps=self._eps)
            self.layer_norm_attn_output = RMSNormNPU(self._linear_dim * self._num_heads, eps=self._eps)

        self.qk_attn_denominator_value = self.set_qk_attn_denominator(
            seq_module_conf.get("qk_attn_denominator", "emb_dim"))
        logging.info(f'denominator value is {self.qk_attn_denominator_value}')
        self.fusion_rab_enable = train_conf.get("fusion_rab_enable", False)

    def set_qk_attn_denominator(self, qk_attn_denominator):
        if qk_attn_denominator == "emb_dim":
            return 1 / self._embedding_dim
        elif qk_attn_denominator == "sqrt_d":
            return 1 / math.sqrt(self._embedding_dim)
        elif qk_attn_denominator == "max_seq_len":
            return 1 / (self._max_sequence_length * 2 + 2)
        elif isinstance(qk_attn_denominator, int) or isinstance(qk_attn_denominator, float):
            return 1 / float(qk_attn_denominator)
        else:
            raise ValueError("Unknown string %s", qk_attn_denominator)
        return 1.0

    @abc.abstractmethod
    def forward(
            self,
            ti: TransformerInput
    ) -> Tuple[torch.Tensor, TransformerCacheState, torch.Tensor, torch.Tensor]:
        pass

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        if self._norm_type == "RMS":
            return self.layer_norm_input(x)
        elif self._norm_type == "layer_norm":
            return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)
        else:
            raise ValueError("Unknown norm_type %s", self._norm_type)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        if self._norm_type == "RMS":
            return self.layer_norm_attn_output(x)
        elif self._norm_type == "layer_norm":
            return F.layer_norm(
                x, normalized_shape=[self._linear_dim * self._num_heads], eps=self._eps
            )
        else:
            raise ValueError("Unknown norm_type %s", self._norm_type)

    def _linear_transform(self, normed_x: torch.Tensor) -> torch.Tensor:
        if self._linear_config == "uvqk":
            batched_mm_output = torch.matmul(normed_x, self._uvqk)
            if self._linear_activation == "silu":
                batched_mm_output = F.silu(batched_mm_output)
            elif self._linear_activation == "none":
                batched_mm_output = batched_mm_output
            u, v, q, k = torch.split(
                batched_mm_output,
                [self._linear_dim * self._num_heads, self._linear_dim * self._num_heads,
                 self._attention_dim * self._num_heads, self._attention_dim * self._num_heads],
                dim=-1,
            )
            return u, v, q, k
        else:
            raise ValueError("Unknown linear_config %s", self._linear_config)


class RMSNormNPU(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def cuda_rms_norm(self,
                      x: torch.Tensor,
                      weight: torch.Tensor,
                      epsilon=1e-6
                      ) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rsqrt = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + epsilon)
        output = (x * rsqrt * weight).to(dtype)
        return output, rsqrt

    def gpu_add_rms_norm(self, residual, x, weight, epsilon=1e-5):
        # 确保x和residual不共用内存（避免原地操作问题1）
        if x is residual:
            residual = residual.clone()

        # 执行Add操作
        sum_result = x + residual

        # 计算RMSNorm
        variance = sum_result.pow(2).mean(dim=-1, keepdim=True)
        normed = sum_result * torch.rsqrt(variance + epsilon)
        output = normed * weight
        return output

    def forward(self, x, residual: Optional[torch.Tensor] = None):
        if NPU_ENABLE:
            import torch_npu
            if residual is None:
                y = torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]
            else:
                y, _, x = torch_npu.npu_add_rms_norm(residual, x, self.weight, self.eps)
            return y
        else:
            if residual is None:
                y = self.cuda_rms_norm(x, self.weight, epsilon=self.eps)[0]
            else:
                # y, _, x = self.gpu_add_rms_norm(residual, x, self.weight, self.eps)
                y = self.gpu_add_rms_norm(residual, x, self.weight, self.eps)
            return y
        return y
class AttentionMaskModule(nn.Module):
    """
    "AttentionMaskModule": {
        "type": ["CausalAttentionMask", "TimeAttentionMask"],
        "cfg": {}
    }

    :param model_conf:
    :param model_factory:
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict):
        super().__init__()
        self.attention_mask_modules = nn.ModuleList([OneTransMask(model_cfg, common_hp)])
        self.use_user_embedding = common_hp["train_conf"].get("use_user_embedding", True)

    def forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len, num_rerank, num_user_tokens):
        init_mask = None
        if not self.use_user_embedding:
            # ck修改
            if num_user_tokens is not None and num_user_tokens > 0:
                pass
            else:
                num_user_tokens = 0
        for attn_module in self.attention_mask_modules:
            mask = attn_module(model_inputs=model_inputs, max_seq_len=max_seq_len, num_rerank=num_rerank,
                               num_user_tokens=num_user_tokens)
            if init_mask is None:
                init_mask = mask
            else:
                init_mask = init_mask * mask
        attn_mask = init_mask.detach().clone()
        return attn_mask
class OneTransMask(nn.Module):

    def __init__(self, model_cfg: Dict, common_hp: Dict):
        super().__init__()
        feat_conf = common_hp.get("feature_conf")
        self.hist_dates_key = feat_conf.get("history_date_column", 'history_timestamps')
        self.cand_dates_key = feat_conf.get("candidate_date_column", 'candidate_timestamps')

        train_conf = common_hp["train_conf"]
        self.use_user_embedding = train_conf.get('use_user_embedding', False)

    def forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len, num_rerank, num_user_tokens):
        """
            Returns a bs x (h+p+1) x (h+p+1)  attn mask
        """
        num_nonseq_feat = num_user_tokens
        hist_ts = model_inputs[self.hist_dates_key]
        cand_ts = model_inputs[self.cand_dates_key]
        bs, hist_len = hist_ts.shape[0], max_seq_len[0]

        # bs, hist_len = hist_ts.shape[0], 1024
        cand_len = cand_ts.shape[2]

        max_len = hist_len + num_nonseq_feat
        device = hist_ts.device

        # onetrans mask示意图
        #
        #       |  s1 | s2 | s3 | s4 | s5 |s6 |n1-1|n1-2|n2-1|n2-2|
        #       --------------------------------------------------
        #   s1  |  1 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
        #   s2  |  1 |  1 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
        #   s3  |  1 |  1 |  1 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
        #   s4  |  1 |  1 |  1 |  1 |  0 |  0 |  0 |  0 |  0 |  0 |
        #   s5  |  1 |  1 |  1 |  1 |  1 |  0 |  0 |  0 |  0 |  0 |
        #   s6  |  1 |  1 |  1 |  1 |  1 |  1 |  0 |  0 |  0 |  0 |
        #   n1-1|  1 |  1 |  1 |  1 |  1 |  1 |  1 |  1 |  0 |  0 |
        #   n1-2|  1 |  1 |  1 |  1 |  1 |  1 |  1 |  1 |  0 |  0 |
        #   n2-1|  1 |  1 |  1 |  1 |  1 |  1 |  0 |  0 |  1 |  1 |
        #   n2-2|  1 |  1 |  1 |  1 |  1 |  1 |  0 |  0 |  1 |  1 |

        mask = torch.zeros((max_len, max_len), device=device)
        # 第一区域
        hist_mask = torch.tril(torch.ones((hist_len, hist_len), device=device))
        mask[:hist_len, :hist_len] = hist_mask
        # 第二区域
        mask[hist_len:, :hist_len] = 1.0
        # 第三区域
        # TODO: fix logic here
        small_block = torch.ones((num_nonseq_feat, num_nonseq_feat), device=device)
        can_blocks = [small_block for _ in range(cand_len)]
        can_self_mask = torch.block_diag(*can_blocks)
        mask[hist_len:, hist_len:] = can_self_mask
        mask = mask.unsqueeze(0).expand(bs, -1, -1)

        return mask
class FeedForward(nn.Module):
    def __init__(self, hidden_dim, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-7),
            nn.Linear(hidden_dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class LayerNormEmbeddingPostprocessor(nn.Module):

    def __init__(self, model_cfg: Dict, common_hp: Dict):
        super().__init__()
        model_conf = common_hp["model_conf"]
        # self._embedding_dim: int = model_conf.get("item_embedding_dim", 64)

        self._embedding_dim = int(os.environ.get("Dim"))
        self._eps: float = Const.EPS
        self.layer_norm_input = torch.nn.LayerNorm((self._embedding_dim,), eps=self._eps)

    def forward(
            self,
            output_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        output_embeddings = output_embeddings[..., :self._embedding_dim]
        return self.layer_norm_input(output_embeddings)

@dataclass
class TransformerInput:
    """Wrapper class for transformer arguments"""

    def __init__(self, x, all_timestamps, invalid_attn_mask, past_lengths, num_rerank=0, layer_num=0,
                 delta_x_offsets=(torch.tensor([]), torch.tensor([])),
                 cache=(torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])),
                 return_cache_states=False, ext_timestamps=None, time_bias=None, offsets=None, offsets_list=None,
                 mask_type=3, invalid_cross_attn_mask=None
                 ):
        self.x = x
        self.all_timestamps = all_timestamps
        self.invalid_attn_mask = invalid_attn_mask
        self.invalid_cross_attn_mask = invalid_cross_attn_mask
        self.past_lengths = past_lengths
        self.num_rerank = num_rerank
        self.layer_num = layer_num
        self.delta_x_offsets = delta_x_offsets
        self.cache = cache
        self.return_cache_states = return_cache_states
        self.time_bias = time_bias
        self.ext_timestamps = ext_timestamps
        self.x_offsets: torch.Tensor = offsets
        self.seq_offsets: List = offsets_list
        self.bs = self.all_timestamps.shape[0]
        self.dense_length = invalid_attn_mask.shape[-1]
        self.jagged_length = 0
        if self.seq_offsets is not None:
            self.jagged_length = self.seq_offsets[-1]
        self.fusion_enabled = False
        self.jagged_enabled = False
        self.cache_states: List[TransformerCacheState] = []
        self.invalid_attn_mask_for_fused_operator = invalid_attn_mask
        self.invalid_cross_attn_mask_for_fused_operator = invalid_cross_attn_mask
        self.mask_type = mask_type
class EmbeddingModule(nn.Module):

    def __init__(self, model_cfg: Dict, common_hp):
        super().__init__()

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_all_item_id_only_embeddings(self, item_features) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_all_embeddings(self, all_features: Dict[str, torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None]:
        pass

    @abc.abstractmethod
    def get_candidate_item_embeddings(self, item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_user_embeddings(self, user_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass
