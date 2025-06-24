#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Callable

import torch
from torch import nn
import torch.distributed as dist

from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    EmbeddingLocation,
    ComputeDevice,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_common import PoolingMode
from hybrid_torchrec.distributed.sharding.post_input_dist import UniqueHashFeatureProcess
from hybrid_torchrec.distributed.batched_embedding_kernel import (
    HybridSplitTableBatchedEmbeddingBagsCodegen,
)
from hybrid_torchrec.modules.ids_process import IdsMapper
from hybrid_torchrec.sparse.jagged_tensor_with_looup_helper import KeyedJaggedTensorWithLookHelper
from torchrec import KeyedJaggedTensor, JaggedTensor


class Awaitable:
    def __init__(self, a_function=None, *args):
        if a_function is not None:
            self.future = executor.submit(a_function, *args)
            self.result = None

    def wait(self):
        if self.result is None:
            self.result = self.future.result()
        return self.result


executor = ThreadPoolExecutor(6)


class LookupAndOutputDistAwaitable(Awaitable):
    def __init__(self, post_awaitable, lookup_and_out_dist_function, *args):
        super().__init__()
        self.post_awaitable = post_awaitable
        self.lookup_and_out_dist_function = lookup_and_out_dist_function
        self.args = args

    def wait(self):
        post_result = self.post_awaitable.wait()
        return self.lookup_and_out_dist_function(post_result, *self.args)


# 不支持一表多查
@dataclass
class EmbeddingConfig:
    table_name: str
    num_embedding: int = 0
    embedding_dim: int = 0
    optimizer: OptimType = OptimType.EXACT_ADAGRAD
    world_size: int = 0
    rank: int = 0


# 示例
class AllGatherEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fwd_pg, bwd_pg, embedding: torch.Tensor):
        ctx.fwd_gp = fwd_pg
        ctx.bwd_pg = bwd_pg
        ctx.embedding = embedding.data
        result_list = [torch.zeros_like(embedding) for i in range(2)]
        fwd_pg.allgather(result_list, embedding.data).wait()
        return tuple(result_list)

    @staticmethod
    def backward(ctx, *grad_output: torch.Tensor):
        grad_output = [g.data for g in grad_output]
        result = torch.zeros_like(grad_output[0])
        if grad_output[0].device != torch.device("cpu"):
            ctx.bwd_pg.reduce_scatter(result, grad_output).wait()
        else:
            # 模拟reduce scatter
            rank = ctx.bwd_pg.rank()
            grad_output_concat = torch.concat(grad_output)
            ctx.bwd_pg.allreduce(grad_output_concat, op=dist.reduce_op.SUM).wait()
            result = grad_output_concat.split(6)[rank]
        return None, None, result


@dataclass
class LookupContext:
    rank: int
    fwd_pg: dist.ProcessGroup
    bwd_pg: dist.ProcessGroup
    communication_metrix: Dict[str, List[torch.Size]] = field(default_factory=dict)


class AllGatherEmbeddings(torch.autograd.Function):
    @staticmethod
    def forward(ctx, embedding: torch.Tensor, feat_name: str, context: LookupContext) -> Tuple[torch.Tensor]:
        ctx.context = context
        ctx.feat_name = feat_name
        result_list = [torch.empty(size) for size in context.communication_metrix[feat_name]]
        context.fwd_pg.allgather(result_list, embedding).wait()
        return tuple(result_list)

    @staticmethod
    def backward(ctx, *grad_output: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, None, None]:
        grad_output = [g.data for g in grad_output]
        result = torch.empty(grad_output[ctx.context.rank].size())
        ctx.context.bwd_pg.reduce_scatter(result, grad_output).wait()

        return result, None, None


class HashEmbeddingModuleCollection(nn.Module):
    def __init__(self, configs: List[EmbeddingConfig], pipe_n_batch=6):
        super().__init__()
        self.fwd_pg = dist.new_group(backend="gloo")
        self.bwd_pg = dist.new_group(backend="gloo")
        self.rank = configs[0].rank
        self.configs = configs

        self.post_input_dist_module_dict: Dict[str, UniqueHashFeatureProcess] = {}
        self.create_post_input_dist()

        self.lookup_module_dict: Dict[str, nn.Module] = self.create_lookups()
        self._weight_init_mins = 0
        self._weight_init_maxs = 1
        self.init_parameters()
        
    def init_parameters(self):
        for module in self.lookup_module_dict.values():
            embedding_weight = module.split_embedding_weights()[0]
            embedding_weight.data.uniform_(
                self._weight_init_mins, self._weight_init_maxs
            )

    def compute_context(self, kjt_list_each_rank: List[KeyedJaggedTensor]) -> LookupContext:
        communication_metrix = defaultdict(list)
        for kjt in kjt_list_each_rank:
            jt_dict: Dict[str, JaggedTensor] = kjt.to_dict()
            for feat_name, jt in jt_dict.items():
                communication_metrix[feat_name].append(jt.values().size())

        return LookupContext(self.rank, self.fwd_pg, self.bwd_pg, communication_metrix)

    def create_post_input_dist(self):
        for config_ in self.configs:
            table_name = config_.table_name
            self.post_input_dist_module_dict[table_name] = UniqueHashFeatureProcess(
                [table_name],
                [1],
                [IdsMapper(config_.num_embedding)],
                False
            )

    def create_lookups(self) -> Dict[str, nn.Module]:
        lookup_module_dict = {}
        for config in self.configs:
            name = config.table_name
            num_embeddings = config.num_embedding // config.world_size
            if self.rank < config.num_embedding % config.world_size:
                num_embeddings += 1
            embedding_spec = (num_embeddings, config.embedding_dim, EmbeddingLocation.DEVICE, ComputeDevice.NPU)
            feature_table_map = [0]
            output_dtype = SparseType.FP32
            optimizer = config.optimizer
            learning_rate = 0.01
            pooling_mode = PoolingMode.NONE
            device = torch.device("npu")
            table_names = [name]
            lookup_module = HybridSplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[embedding_spec],
                feature_table_map=feature_table_map,
                output_dtype=output_dtype,
                optimizer=optimizer,
                learning_rate=learning_rate,
                pooling_mode=pooling_mode,
                device=device,
                table_names=table_names,
            )
            lookup_module_dict[name] = lookup_module
        return lookup_module_dict

    def post_input_dist(
        self, features: JaggedTensor, feat_name: str
    ):
        with torch.no_grad():
            kjt = KeyedJaggedTensor.from_jt_dict({feat_name: features})
            return self.post_input_dist_module_dict[feat_name](kjt)

    def do_lookup_and_post_dist(
        self, jt: JaggedTensor, feat_name: str, context: LookupContext
    ):
        return

    # 示例代码
    def lookup(self, kjt: KeyedJaggedTensorWithLookHelper, feat_name: str):
        kjt = kjt.to(device=torch.device(f"npu:{self.rank}"))
        return self.lookup_module_dict[feat_name](
            indices=kjt.values().long(),
            offsets=kjt.offsets().long(),
            hash_indices=kjt.hash_indices,
            unique_indices=kjt.unique_indices,
            unique_offset=kjt.unique_offset,
            unique_inverse=kjt.unique_inverse
        )

    def lookup_and_post_dist(self, kjt: KeyedJaggedTensorWithLookHelper, feat_name: str, context: LookupContext):
        embedding = self.lookup(kjt, feat_name)
        result = AllGatherEmbeddings().apply(embedding, feat_name, context)
        return result

    def forward(self, kjt_list_each_rank: List[KeyedJaggedTensor]):
        context = self.compute_context(kjt_list_each_rank)
        jt_dict: Dict[str, JaggedTensor] = kjt_list_each_rank[self.rank].to_dict()
        awaitable_dict: Dict[str, LookupAndOutputDistAwaitable] = {}
        for feat_name in jt_dict.keys():
            post_awaitable = self.post_input_dist(jt_dict[feat_name], feat_name)
            lookup_and_output_dist_awaitable = LookupAndOutputDistAwaitable(
                post_awaitable, self.lookup_and_post_dist, feat_name, context
            )
            awaitable_dict[feat_name] = lookup_and_output_dist_awaitable
        return awaitable_dict
