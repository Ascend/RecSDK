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

class DistributeEmbeddingModuleWithSideInfoLonger(EmbeddingModule):
    """
    带有sideinfo的Embedding模块, 用于生成物品和用户的表示。
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, device) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp)
        feat_conf = common_hp["feature_conf"]
        model_conf = common_hp["model_conf"]
        train_conf = common_hp["train_conf"]
        data_loader_conf = common_hp["data_loader_conf"]
        item_feature_columns: Dict = feat_conf.get('item_feature_columns', None)
        self.item_feature_columns = item_feature_columns
        user_feature_columns: Dict = feat_conf.get('user_feature_columns', None)
        self.user_feature_columns = user_feature_columns
        sequence_feature_columns: Dict = feat_conf.get('seq_feature_columns', None)
        self.sequence_feature_columns = sequence_feature_columns
        self.infer_item_id_name = feat_conf.get("infer_items_key")
        self.padding_index = feat_conf.get("padding_index", 0)
        self.num_rerank = data_loader_conf.get("num_rerank")
        self.use_nonseq_token = model_cfg[Const.HP].get("use_nonseq_token", False)
        self.device = device

        # if model_conf["item_user_mlp"]:
        self.item_mlp = True
        self.user_mlp = True

        # 构造辅助embedding字典
        self.aux_embedding_dict = torch.nn.ModuleDict()
        self.aux_dim = {}
        aux_dim_total = 0

        self.multi_value_prefix = feat_conf.get('multi_value_prefix', "pref_")
        if user_feature_columns is None or item_feature_columns is None or self.sequence_feature_columns is None:
            raise ValueError("user_feature_columns and item_feature_columns cannot be None")
        # 物品信息嵌入字典
        self._item_feature_names: List[str] = []
        self.item_dyte: Dict[str, str] = {}
        item_info_dims = {}
        # 用户信息嵌入字典
        self._user_feature_names: List[str] = []
        self.user_dyte: Dict[str, str] = {}
        user_info_dims = {}
        # 序列信息嵌入字典
        self._seq_feature_names: Dict[str, str] = {}
        self.seq_dyte: Dict[str, str] = {}
        seq_info_dims = {}
        # multi特征，emb切分时不能使用row_wise
        self.multi_feature_names: List[str] = []
        # 统计连续特征的个数
        con_item_feature_count = 0
        con_user_feature_count = 0

        # 初始化物品特征嵌入
        self.feature_tables: Dict[str, torchrec.EmbeddingConfig] = {}
        self.enabled_item_features: List[str] = []
        for feature_name, _feature_info in item_feature_columns.items():
            feature_count = int(item_feature_columns[feature_name].get('feature_count', 10))
            feature_enabled = item_feature_columns[feature_name].get("enabled", True)
            feature_dtype = item_feature_columns[feature_name].get("dtype", "int")
            self.item_dyte[feature_name] = feature_dtype
            if feature_enabled:
                self.enabled_item_features.append(feature_name)
                if feature_dtype == "con":
                    # 处理商品特征里的离散特征
                    feature_dim = 1
                    con_item_feature_count = con_item_feature_count + 1
                elif feature_dtype in ["int", "multi"]:
                    feature_dim = item_feature_columns[feature_name].get('dim', 32)
                    table = torchrec.EmbeddingConfig(
                        name=feature_name,
                        embedding_dim=feature_dim,
                        num_embeddings=feature_count + 1,
                        feature_names=[feature_name],
                    )
                    self.feature_tables[feature_name] = table
                    if feature_dtype == "multi":
                        self.multi_feature_names.append(feature_name)
                else:
                    logging.error("feature_dtype %s is undefined for %s.", feature_dtype, feature_name)
                self._item_feature_names.append(feature_name)
                if _feature_info.get("enable_aux_emb", False):
                    feature_dim = feature_dim + aux_dim_total
                item_info_dims[feature_name] = feature_dim
        # 初始化用户特征嵌入
        self.enabled_user_features: List[str] = []
        for feature_name, _feature_info in user_feature_columns.items():
            feature_count = int(user_feature_columns[feature_name].get('feature_count', 10))
            feature_enabled = user_feature_columns[feature_name].get("enabled", True)
            feature_dtype = user_feature_columns[feature_name].get("dtype", "int")
            associated_item_feature = user_feature_columns[feature_name].get("associated", None)
            self.user_dyte[feature_name] = feature_dtype
            if feature_enabled:
                self.enabled_user_features.append(feature_name)
                if feature_dtype == "pref":
                    # 默认的关联特征是去掉pref_前缀的商品特征名，例如pref_artist默认的关联特征是artist
                    associated = feature_name.replace(self.multi_value_prefix,"") if associated_item_feature is None \
                        else associated_item_feature
                    if associated in self.enabled_item_features:
                        # 若关联特征是item embedding 且启用
                        feature_dim = item_info_dims.get(associated)
                        user_info_dims[feature_name] = feature_dim
                        # 多个特征共享同一emb表
                        self.feature_tables[associated].feature_names.append(feature_name)

                    else:
                        logging.error(
                            "The assoicated feature %s for %s is not an item feature or not enabled in config.",
                            associated, feature_name)
                elif feature_dtype == "con":
                    # 处理商品特征里的离散特征
                    feature_dim = 1
                    con_user_feature_count = con_user_feature_count + 1
                    user_info_dims[feature_name] = feature_dim
                elif feature_dtype in ["int", "multi"]:
                    associated = user_feature_columns[feature_name].get("shared", None)
                    if associated is None or associated not in self.feature_tables:
                        # 若无关联特征，直接初始化
                        feature_dim = user_feature_columns[feature_name].get('dim', 32)
                        table = torchrec.EmbeddingConfig(
                            name=feature_name,
                            embedding_dim=feature_dim,
                            num_embeddings=feature_count + 1,
                            feature_names=[feature_name],
                        )
                        self.feature_tables[feature_name] = table
                        user_info_dims[feature_name] = feature_dim
                        if feature_dtype == "multi":
                            self.multi_feature_names.append(feature_name)
                    else:
                        feature_dim = user_info_dims.get(associated)
                        user_info_dims[feature_name] = feature_dim
                        self.feature_tables[associated].feature_names.append(feature_name)
                else:
                    logging.error("feature_dtype %s is undefined for the user.", feature_dtype)
                self._user_feature_names.append(feature_name)

        # 初始化序列特征嵌入
        user_item_feature_dim = {**item_info_dims, **user_info_dims}

        self.enabled_seq_features: List[str] = []
        for seq_name, seq_config in self.sequence_feature_columns.items():
            seq_info_dims[seq_name] = seq_config.get("length")
            feature_enabled = seq_config.get("enabled", True)
            if feature_enabled:
                feature_dtype = seq_config.get("dtype", "int")
                if feature_dtype == "pref":
                    # "pos_seq"属于所有序列都共享的特征，只处理第一遍就行，后面直接跳过
                    associated = seq_config.get("associated", None)
                    if associated in self.feature_tables:
                        if seq_name not in self.feature_tables[associated].feature_names:
                            self.feature_tables[associated].feature_names.append(seq_name)
                    else:
                        logging.error("assoiciated feature %s never appears before.", associated)

                    seq_info_dims[seq_name] = user_item_feature_dim.get(associated)
                elif feature_dtype in ["int"]:
                    feature_dim = seq_config.get("dim", 32)
                    feature_count = int(seq_config.get("feature_count", 10))
                    table = torchrec.EmbeddingConfig(
                        name=seq_name,
                        embedding_dim=feature_dim,
                        num_embeddings=feature_count + 1,
                        feature_names=[seq_name],
                    )
                    if seq_name not in self.feature_tables:
                        self.feature_tables[seq_name] = table
                    if _feature_info.get("enable_aux_emb", False):
                        feature_dim = feature_dim + aux_dim_total
                    seq_info_dims[seq_name] = feature_dim

        ecs: Dict[str, List[torchrec.EmbeddingConfig]] = {}
        self.ec_feature_names: Dict[str, List[str]] = {}
        for table in self.feature_tables.values():
            # table按dim分类：每个ec中的table.embedding_dim必须相同
            ec_attr = f"ec_dim{table.embedding_dim}"
            ecs.setdefault(ec_attr, []).append(table)
            self.ec_feature_names.setdefault(ec_attr, []).extend(table.feature_names)
        self.shared_ebc_attrs = self.ec_feature_names
        # 创建EmbeddingCollections
        for attr, tbls in ecs.items():
            setattr(self, attr,
                    torchrec.EmbeddingCollection(device=self.device, tables=tbls))
        self._con_item_info_embs = torch.nn.BatchNorm1d(con_item_feature_count)
        self._con_user_info_embs = torch.nn.BatchNorm1d(con_user_feature_count)
        # self._item_embedding_dim = common_hp["model_conf"].get("item_embedding_dim", 64)
        self._item_embedding_dim = int(os.environ.get("Dim"))

        # 计算物品和用户输入维度
        item_input_dim = sum(item_info_dims.values())
        user_input_dim = sum(user_info_dims.values())

        feat_conf["item_emb_dims"] = item_input_dim
        feat_conf["user_emb_dims"] = user_input_dim
        logging.info("item_emb_dims is %s", item_input_dim)
        logging.info("user_emb_dims is %s", user_input_dim)

        self.item_emb_mlp, self.user_emb_mlp, self.seq_emb_mlp = torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity()
        self.item_emb_mlp = torch.nn.Sequential(
            torch.nn.Linear(item_input_dim, self._item_embedding_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(self._item_embedding_dim * 4, self._item_embedding_dim), )
        logging.info('Set item_emb_mlp to Linear: %s -> %s', item_input_dim, self._item_embedding_dim)

        self.user_emb_mlp = torch.nn.Sequential(
            torch.nn.Linear(user_input_dim, self._item_embedding_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(self._item_embedding_dim * 4, self._item_embedding_dim), )
        logging.info('Set user_emb_mlp to Linear: %s -> %s', user_input_dim, self._item_embedding_dim)

        seq_input_dim = sum(seq_info_dims.values())
        if self._item_embedding_dim != 0:
            self.seq_emb_mlp = torch.nn.Sequential(
                torch.nn.Linear(seq_input_dim, self._item_embedding_dim * 4),
                torch.nn.ReLU(),
                torch.nn.Linear(self._item_embedding_dim * 4, self._item_embedding_dim),
                torch.nn.ReLU()
            )
            logging.info('Set seq_emb_mlp to Linear: %s -> %s' % (seq_input_dim, self._item_embedding_dim))
        else:
            if item_input_dim != user_input_dim:
                raise RuntimeError('item_input_dim and user_input_dim mismatch! user_dim : %s, item_dim : %s' %
                                   (user_input_dim, item_input_dim))
            self._item_embedding_dim = item_input_dim

        self.feature_values_cache = None

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim

    def debug_str(self) -> str:
        return self.__class__.__name__

    def reset_params(self):
        seen = set()
        for name, params in self.named_parameters():
            if 'emb' in name:
                ptr = params.data_ptr()
                if ptr in seen:
                    continue  # 已初始化过底层 tensor
                seen.add(ptr)
                logging.info("Initialize %s as truncated normal: %s params", name, params.data.size())
                truncated_normal(params, mean=0.0, std=0.02)
            elif 'weight' in name and 'mlp' in name:
                torch.nn.init.normal_(params, mean=0.0, std=0.01)
                logging.info("Initialize %s with normal(0, 0.01)", name)
            elif 'bias' in name and 'mlp' in name:
                torch.nn.init.constant_(params, 0.0)
                logging.info("Initialize %s with constant(0.0)", name)
            else:
                logging.info("Skipping initializing params %s - not configured", name)

    def _get_feature_values(self, all_features: Dict[str, torch.Tensor]) -> ChainMap[str, torch.Tensor]:
        # 由于torch.fx中Proxy object cannot be iterated，EC查表返回的字典不能通过update等方式合并，
        # 所以该处使用collections.ChainMap将多个map映射连接起来，在避免fx报错的情况下达到字典合并的效果。
        from torchrec import JaggedTensor
        jt_dicts: List[Dict[str, JaggedTensor]] = []
        for ec_attr in self.shared_ebc_attrs.keys():
            jt_dicts.append(getattr(self, ec_attr)(all_features[ec_attr]))
        embs_dict = ChainMap(*jt_dicts)

        return embs_dict

    def _init_embs(self, features: Dict[str, torch.Tensor]):
        self.feature_values_cache = self._get_feature_values(features)

    def _reset_embs(self):
        if self.feature_values_cache is not None:
            del self.feature_values_cache
            self.feature_values_cache = None

    def prepare_embeddings_if_necessary(self, features: Dict[str, torch.Tensor]):
        return initializing(functools.partial(self._init_embs, features), self._reset_embs)

    def _get_item_embs(self, item_features: Dict[str, torch.Tensor], all_embs) -> torch.Tensor:
        """
        根据物品特征获取物品嵌入。

        :param item_features: 物品特征字典。
        :return: 物品嵌入张量。
        """
        eps = 1e-6
        feature_emb_list = []
        con_inputs = []
        con_names = []
        for feature_name in self._item_feature_names:
            if feature_name in self.enabled_item_features:
                item_feature_id = item_features[feature_name]
                if self.item_dyte[feature_name] == "con":
                    con_inputs.append(item_feature_id)
                    con_names.append(feature_name)
                    continue
                elif self.item_dyte[feature_name] == "multi":
                    if self.training:
                        # item_feature_id： (B, N, M) M 是多值特征padding后的长度
                        item_feature_id = item_feature_id.unsqueeze(1)
                    B, N, M = item_feature_id.shape
                    # feature_values：（B, N, M, D）
                    feature_values = all_embs[feature_name].values().reshape(B, N, M, -1)
                    feature_dim = feature_values.size(-1)
                    feat_nonzero = item_feature_id != self.padding_index
                    feat_mask = feat_nonzero.unsqueeze(-1).repeat(1, 1, 1, feature_dim)
                    # 对pref多值特征做mean pooling，并忽略掉值为self.padding_index的填充index
                    # 形状为(B, N, D)
                    feature_value = (feature_values * feat_mask).sum(dim=2) / (feat_mask.sum(dim=2) + eps)
                else:
                    if not self.training:
                        # item_feature_id: (B, N) N是候选商品数量
                        item_feature_id = item_feature_id.squeeze(1)
                    B, N = item_feature_id.shape[0], 1
                    # feature_value：（B, N, D）
                    feature_value = all_embs[feature_name].values().reshape(B, N, -1)

                if self.item_feature_columns[feature_name].get("enable_aux_emb", False):
                    # 如果此特征启用辅助embedding，则将辅助embedding和该特征拼接
                    if self.item_dyte[feature_name] in ["multi"]:
                        aux_embeddings = []
                        for k, v in self.aux_embedding_dict.items():
                            aux_dim = self.aux_dim.get(k)
                            feat_mask = feat_nonzero.unsqueeze(-1).repeat(1, 1, aux_dim)
                            aux_value = v(item_feature_id)
                            aux_value = (aux_value * feat_mask).sum(dim=1) / (feat_mask.sum(dim=1) + eps)
                            aux_embeddings.append(aux_value)
                    else:
                        aux_embeddings = [v(item_feature_id)
                                          for _, v in self.aux_embedding_dict.items()]
                    aux_embeddings.append(feature_value)
                    feature_value = torch.cat(aux_embeddings, dim=-1)

            feature_emb_list.append(feature_value)
        if con_inputs:
            con_input_tensor = torch.cat(con_inputs, dim=1)
            normed_con = self._con_item_info_embs(con_input_tensor)
            normed_con = normed_con.transpose(1, 2)
            # 一般 shape[1] 就是 con 数量
            num_chunks = len(con_names)
            chunks = normed_con.chunk(num_chunks, dim=-1)
            for i in range(num_chunks):
                feature_emb_list.append(chunks[i])

        if len(self.enabled_item_features) > 0:

            feature_embs_original = torch.cat(feature_emb_list, dim=-1)
            if self.item_mlp and self._item_embedding_dim != 0:
                # (B, S, _item_embedding_dim)
                feature_embs = self.item_emb_mlp(feature_embs_original)
            else:
                feature_embs = feature_embs_original
        else:
            feature_embs = None
        return feature_embs

    def _get_user_embs(self, user_features: Dict[str, torch.Tensor], all_embs) -> torch.Tensor:
        """
        根据用户特征获取用户嵌入。

        :param user_features: 用户特征字典。
        :return: 用户嵌入张量。
        """
        eps = 1e-6
        feature_emb_list = []
        con_inputs = []
        con_names = []
        for feature_name in self._user_feature_names:
            if feature_name in self.enabled_user_features:
                user_feature_id = user_features[feature_name]
                feature_value = all_embs[feature_name].values()
                if self.user_dyte[feature_name] == "con":
                    con_inputs.append(user_feature_id)
                    con_names.append(feature_name)
                    continue
                elif self.user_dyte[feature_name] in ["pref", "multi"]:
                    B, M = user_feature_id.shape
                    # feature_values: (B, M, D) M 是多值特征padding后的长度
                    feature_value = feature_value.reshape(B, M, -1)
                    feature_dim = feature_value.size(-1)
                    feat_nonzero = user_feature_id != self.padding_index
                    feat_mask = feat_nonzero.unsqueeze(-1).repeat(1, 1, feature_dim)
                    # 对pref多值特征做mean pooling，并忽略掉值为self.padding_index的填充index
                    feature_value = (feature_value * feat_mask).sum(dim=1) / (feat_mask.sum(dim=1) + eps)

            feature_emb_list.append(feature_value)

        if con_inputs:
            con_input_tensor = torch.cat(con_inputs, dim=1)
            normed_con = self._con_user_info_embs(con_input_tensor)
            normed_con = normed_con.transpose(1, 2)
            # 一般 shape[1] 就是 con 数量
            num_chunks = len(con_names)
            chunks = normed_con.chunk(num_chunks, dim=-1)
            for i in range(num_chunks):
                feature_emb_list.append(chunks[i])

        if len(self.enabled_user_features) > 0:
            feature_embs_original = torch.cat(feature_emb_list, dim=-1)
            if self.user_mlp and self._item_embedding_dim != 0:
                # (B, S, _item_embedding_dim)
                feature_embs = self.user_emb_mlp(feature_embs_original)
            else:
                feature_embs = feature_embs_original
            feature_embs = feature_embs.unsqueeze(1)
        else:
            feature_embs = None
        return feature_embs

    def _get_seq_embs(self, seq_features: Dict[str, torch.Tensor], all_embs) -> torch.Tensor:
        feature_emb_list = []
        for seq_name, seq_config in self.sequence_feature_columns.items():
            # for feature_name, feature_info in seq_config.items():
            seq_feature_id = seq_features[seq_name]
            S = seq_config.get("length")
            B = seq_feature_id.shape[0]
            L = seq_feature_id.shape[-1]

            feature_value = all_embs[seq_name].values().reshape(B, L, -1)[:, :S, :]

            if seq_config.get("enable_aux_emb", False):
                aux_embeddings = [v(seq_feature_id) for _, v in self.aux_embedding_dict.items()]
                aux_embeddings.append(feature_value)
                feature_value = torch.cat(aux_embeddings, dim=-1)
            feature_emb_list.append(feature_value)
            seq_emb = torch.cat(feature_emb_list, dim=-1)
        if self._item_embedding_dim != 0:
            # (B, S, _item_embedding_dim)
            feature_embs = self.seq_emb_mlp(seq_emb)
        return feature_embs

    def get_all_embeddings(self, all_features: Dict[str, torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        # 由于torch.fx中Proxy object cannot be iterated，EC查表返回的字典不能通过update等方式合并，
        # 所以该处使用collections.ChainMap将多个map映射连接起来，在避免fx报错的情况下达到字典合并的效果。
        from torchrec import JaggedTensor, KeyedJaggedTensor
        from typing import Dict, List, Tuple, ChainMap, Callable
        device = self.device

        new_all_features = {}
        for k, v in all_features.items():
            if isinstance(v, KeyedJaggedTensor):
                new_all_features[k] = KeyedJaggedTensor(
                    keys=v.keys(),
                    values=v.values(),
                    lengths=v.lengths() if v.lengths() is not None else None,
                    offsets=v.offsets() if v.offsets() is not None else None,
                    weights=v.weights_or_none() if v.weights_or_none() is not None else None,
                    stride=v.stride(),
                    length_per_key=v.length_per_key(),
                    offset_per_key=v.offset_per_key(),
                    index_per_key=v.index_per_key(),
                )
            elif isinstance(v, JaggedTensor):
                new_all_features[k] = JaggedTensor(
                    values=v.values(),
                    lengths=v.lengths() if v.lengths() is not None else None,
                    offsets=v.offsets() if v.offsets() is not None else None,
                    weights=v.weights() if v.weights() is not None else None,
                )
            elif hasattr(v, "to"):
                new_all_features[k] = v
            else:
                new_all_features[k] = v

        all_features = new_all_features

        jt_dicts: List[Dict[str, JaggedTensor]] = []

        for ec_attr in self.shared_ebc_attrs.keys():
            jt_dicts.append(getattr(self, ec_attr)(all_features[ec_attr]))
        embs_dict = ChainMap(*jt_dicts)

        item_feature_embs = self._get_item_embs(all_features, embs_dict)

        user_feature_embs = self._get_user_embs(all_features, embs_dict)
        seq_feature_embs = self._get_seq_embs(all_features, embs_dict)

        if self.use_nonseq_token:
            user_feature_embs = self._get_user_embs(all_features, embs_dict)
            return item_feature_embs, user_feature_embs, seq_feature_embs
        else:
            return item_feature_embs, user_feature_embs, seq_feature_embs

    def get_item_embeddings(self, item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise RuntimeError(
            "Please call 'get_all_embeddings', return a tuple contains item_embeddings, user_embeddings, seq_feature_embs")

    def get_user_embeddings(self, user_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise RuntimeError(
            "Please call 'get_all_embeddings', return a tuple contains item_embeddings, user_embeddings, seq_feature_embs")

    def get_candidate_item_embeddings(self, item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise RuntimeError(
            "Please call 'get_all_embeddings', return a tuple contains item_embeddings, user_embeddings, seq_feature_embs")


class OneTransBlock(Transformer):

    def __init__(self, model_cfg: Dict, common_hp: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp)
        self.num_nonseq_feat = model_cfg[Const.HP].get("num_nonseq_feat", None)
        # self.d_model = model_cfg[Const.HP].get("d_model", None)

        # self.num_nonseq_feat = int(os.environ.get("Non_Seq_Len"))
        self.d_model = int(os.environ.get("Dim"))

        self.num_head = model_cfg[Const.HP].get("nhead", None)
        self.head_dim = self.d_model // self.num_head
        self.use_heterogeneous = model_cfg[Const.HP].get("use_heterogeneous", None)
        self.use_pyramid = model_cfg[Const.HP].get("use_pyramid", None)
        self.num_blocks = model_cfg[Const.HP].get("num_blocks", None)
        self.num_pyramid = self.num_nonseq_feat // (self.num_blocks + 2)

        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)

        # 共享参数
        self.Wq_share = nn.Linear(self.d_model, self.d_model, bias=False)
        self.Wk_share = nn.Linear(self.d_model, self.d_model, bias=False)
        self.Wv_share = nn.Linear(self.d_model, self.d_model, bias=False)

        self.attn = nn.MultiheadAttention(self.d_model, self.num_head, batch_first=True)

        self.out_share = nn.Linear(self.d_model, self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.GELU(),
            nn.Linear(self.d_model * 4, self.d_model),
        )

        # 非共享参数
        if self.use_heterogeneous:
            self.Wq_ns = nn.Parameter(
                torch.empty(self.num_nonseq_feat, self.d_model, self.d_model).normal_(mean=0, std=0.02))
            self.Wk_ns = nn.Parameter(
                torch.empty(self.num_nonseq_feat, self.d_model, self.d_model).normal_(mean=0, std=0.02))
            self.Wv_ns = nn.Parameter(
                torch.empty(self.num_nonseq_feat, self.d_model, self.d_model).normal_(mean=0, std=0.02))
            self.out_w_ns = nn.Parameter(
                torch.empty(self.num_nonseq_feat, self.d_model, self.d_model).normal_(mean=0, std=0.02))
            self.out_b_ns = nn.Parameter(torch.empty(self.num_nonseq_feat, self.d_model).normal_(mean=0, std=0.02))
            self.w1_ns = nn.Parameter(
                torch.empty(self.num_nonseq_feat, self.d_model, self.d_model * 4).normal_(mean=0, std=0.02))
            self.w2_ns = nn.Parameter(
                torch.empty(self.num_nonseq_feat, self.d_model * 4, self.d_model).normal_(mean=0, std=0.02))
            self.b1_ns = nn.Parameter(torch.empty(self.num_nonseq_feat, self.d_model * 4).normal_(mean=0, std=0.02))
            self.b2_ns = nn.Parameter(torch.empty(self.num_nonseq_feat, self.d_model).normal_(mean=0, std=0.02))

    def forward(self, ti: TransformerInput) -> Tuple[torch.Tensor, TransformerCacheState, torch.Tensor]:
        X = ti.x
        attention_mask = ti.invalid_attn_mask_for_fused_operator
        num_rerank = ti.num_rerank
        layer_num = ti.layer_num

        B, N, D = X.shape
        nonseq_len = num_rerank * self.num_nonseq_feat

        residual = X
        Xn = self.ln1(X)
        X_s = Xn[:, :-nonseq_len, :]
        X_ns = Xn[:, -nonseq_len:, :]

        Q_s, K_s, V_s = self.Wq_share(X_s), self.Wk_share(X_s), self.Wv_share(X_s)

        if self.use_heterogeneous:
            Q_ns = torch.einsum("bnd,ndk->bnk", X_ns, self.Wq_ns)
            K_ns = torch.einsum("bnd,ndk->bnk", X_ns, self.Wk_ns)
            V_ns = torch.einsum("bnd,ndk->bnk", X_ns, self.Wv_ns)
        else:
            Q_ns, K_ns, V_ns = self.Wq_share(X_ns), self.Wk_share(X_ns), self.Wv_share(X_ns)

        Q = torch.cat([Q_s, Q_ns], dim=1)
        K = torch.cat([K_s, K_ns], dim=1)
        V = torch.cat([V_s, V_ns], dim=1)

        if self.use_pyramid:
            attention_new_mask = attention_mask[:, :, self.num_pyramid * layer_num:, self.num_pyramid * layer_num:]
        else:
            attention_new_mask = attention_mask

        attn_mask_2d = attention_new_mask.squeeze(1)[0]

        # out, _ = self.attn(Q, K, V, attn_mask=attn_mask_2d)

        out = scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask_2d,  # 可选的注意力掩码
            dropout_p=0,  # Dropout 概率
            is_causal=False  # 启用因果掩码（适用于自回归任务）
        )

        if self.use_heterogeneous:
            out_s = self.out_share(out[:, :-nonseq_len, :])
            out_ns = torch.einsum("bnd,ndf->bnf", out[:, -nonseq_len:, :], self.out_w_ns) + self.out_b_ns
            out = torch.cat([out_s, out_ns], dim=1)
        else:
            out = self.out_share(out)
        X = residual + out  # residual connection

        # FFN
        if self.use_heterogeneous:
            X_s = self.ffn(
                self.ln2(
                    X[:, :-nonseq_len, :]
                ))
            X_ns = torch.einsum(
                "bnd,ndf->bnf",
                self.ln2(
                    X[:, -nonseq_len:, :]
                ), self.w1_ns) + self.b1_ns
            X_ns = F.gelu(X_ns)
            X_ns = torch.einsum("bnd,ndf->bnf", X_ns, self.w2_ns) + self.b2_ns
            X = X + torch.cat([X_s, X_ns], dim=1)
        else:
            X = + self.ffn(self.ln2(X))

        if self.use_pyramid:
            X = X[:, self.num_pyramid:, :]
        # print(X.shape)
        return X, (None, None, None, None), ti.time_bias
class InputFeaturesPreprocessorModule(nn.Module):

    def __init__(self, model_cfg: Dict, common_hp: Dict):
        super().__init__()

    @abc.abstractmethod
    def forward(
            self,
            user_feature_embs: torch.Tensor,
            past_embeddings: torch.Tensor,
            past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pass
import numpy as np
import random
def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class UserItemRatingInputFeaturePreprocessorLonger(InputFeaturesPreprocessorModule):

    def __init__(self, model_cfg: Dict, common_hp: Dict, device) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp)
        model_conf = common_hp["model_conf"]
        feat_conf = common_hp["feature_conf"]
        seq_feature_conf = feat_conf.get("seq_feature_columns")
        self.device = device
        seq_lens = [max(subcfg["length"] for subcfg in seq_feature_conf.values())]

        self.seq_lens = seq_lens
        self._embedding_dim = int(os.environ.get("Dim"))

        set_seed(1234)
        self._pos_emb = torch.nn.Embedding(sum(seq_lens) + 3, self._embedding_dim)
        # 先在 CPU 上初始化
        self._pos_emb = self._pos_emb.cpu()
        # 再搬到对应设备
        self._pos_emb = self._pos_emb.to(self.device)

        self._seq_type_emb = torch.nn.Embedding(len(seq_lens) + 1, self._embedding_dim)
        self._seq_type_emb = self._seq_type_emb.to(self.device)
        self.use_nonseq_token = model_cfg[Const.HP].get("use_nonseq_token", False)
        self._use_auto_split = model_cfg[Const.HP].get("use_auto_split", False)

        if self.use_nonseq_token and self._use_auto_split:
            total_dim = 0
            for name, feat_info in {**feat_conf["item_feature_columns"], **feat_conf["user_feature_columns"]}.items():
                if feat_info["enabled"]:
                    total_dim += feat_info["dim"]
            self.auto_layer = torch.nn.Linear(total_dim, total_dim)

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb, mean=0.0, std=math.sqrt(weird_division(1.0, self._embedding_dim)),
        )

    def forward(
            self,
            model_inputs,
            past_ids: List[torch.Tensor],
            num_rerank: int,
            past_lengths,
            user_feature_embs: torch.Tensor,
            item_feature_embs: torch.Tensor,
            seq_feature_embs: torch.Tensor,
            deep_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        B, N, D = item_feature_embs.size()
        device = past_lengths.device
        seq_feature_embs = seq_feature_embs

        if self.use_nonseq_token:
            if self._use_auto_split:
                num_nonseq_feat = item_feature_embs.shape[1] + user_feature_embs.shape[1]
                deep_outputs = torch.cat(
                    [item_feature_embs.reshape(B, -1),
                     user_feature_embs.reshape(B, -1)
                     ], dim=-1)
                deep_outputs = self.auto_layer(deep_outputs)
                deep_outputs = deep_outputs.reshape(B, num_nonseq_feat, -1)
            else:
                if user_feature_embs is not None:
                    deep_outputs = torch.cat(
                        [item_feature_embs,
                         user_feature_embs
                         ], dim=1)
                else:
                    deep_outputs = item_feature_embs
            # D = deep_outputs.shape[1]
            # deep_pos_ids = torch.ones((1, num_rerank), dtype=torch.int, device=device)
            # deep_pos_embs = self._pos_emb(deep_pos_ids).repeat(B, D, 1)
            # deep_outputs = deep_outputs + deep_pos_embs
            deep_outputs = deep_outputs
        else:
            deep_pos_ids = torch.ones((1, num_rerank), dtype=torch.int, device=device)
            deep_pos_embs = self._pos_emb(deep_pos_ids).repeat(B, N, 1)
            deep_outputs = deep_pos_embs

        x_offsets = None
        seq_offsets = None

        past_sum = torch.sum(past_lengths, dim=1)
        x_offsets = past_sum + num_rerank
        x_offsets = torch.cumsum(x_offsets, dim=0)
        x_offsets = torch.cat((torch.tensor([0], device=device), x_offsets))

        max_seq_len = seq_feature_embs.shape[1]
        indices = torch.arange(max_seq_len, device=device)

        mask_valid = torch.concat(past_ids, dim=-1)
        mask_valid = (mask_valid != 0)

        past_sum = past_sum.unsqueeze(-1).unsqueeze(-1)
        mask_sort = (indices < past_sum).squeeze(1)

        deep_outputs = deep_outputs

        whole_seq_embeddings = torch.cat(
            [deep_outputs,
             seq_feature_embs
             ], dim=1)

        return whole_seq_embeddings, x_offsets, seq_offsets