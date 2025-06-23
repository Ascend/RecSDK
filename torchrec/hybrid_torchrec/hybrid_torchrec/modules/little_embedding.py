#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
from typing import List, Tuple, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import logging

import torch
from torch import nn
import torch.distributed as dist
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


class PostInpuDistAwaitable(Awaitable):
    pass


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
    optimizer: torch.optim.Optimizer = 0
    world_sie: int = 0
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
    communication_metrix: Dict[str, List[torch.Size]] = {}


class AllGatherEmbeddings(torch.autograd.Function):
    @staticmethod
    def forward(ctx, embedding: torch.Tensor, feat_name: str, context: LookupContext) -> Tuple[torch.Tensor]:
        ctx.context = context
        ctx.feat_name = feat_name
        result_list = [torch.zeros_like(size) for size in context.communication_metrix[feat_name]]
        context.fwd_pg.allgather(result_list, embedding.data).wait()
        return tuple(result_list)

    @staticmethod
    def backward(ctx, *grad_output: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, None]:
        grad_output = [g.data for g in grad_output]
        result = torch.zeros_like(grad_output[ctx.context.rank])
        ctx.context.bwd_pg.reduce_scatter(result, grad_output).wait()

        return result, None, None


class HashEmbeddingModuleCollection(nn.Module):
    def __init__(self, configs=List[EmbeddingConfig], pipe_n_batch=6):
        super().__init__()
        self.fwd_pg = dist.new_group(backend="gloo")
        self.bwd_pg = dist.new_group(backend="gloo")
        self.rank = configs[0].rank
        self.post_input_dist_module_dict: Dict[str, nn.Module] = (
            self.create_post_input_dist()
        )
        self.lookup_module_dict: Dict[str, nn.Module] = self.create_lookups()

    def compute_context(self, kjt_list_each_rank: List[KeyedJaggedTensor]) -> LookupContext:
        communication_metrix = defaultdict(list)
        for kjt in kjt_list_each_rank:
            jt_dict: Dict[str, JaggedTensor] = kjt.to_dict()
            for feat_name, jt in jt_dict.items():
                communication_metrix[feat_name].append(jt.values().size())

        return LookupContext(self.rank, self.fwd_pg, self.bwd_pg, communication_metrix)

    def create_post_input_dist(self) -> Dict[str, nn.Module]:
        return

    def create_lookups(self) -> Dict[str, nn.Module]:
        return

    def do_post_input_dist(
        self, jt: JaggedTensor, feat_name: str, context: LookupContext
    ):
        return

    def do_lookup_and_post_dist(
        self, jt: JaggedTensor, feat_name: str, context: LookupContext
    ):
        return

    # 示例代码
    def fids2indices(self, fids: KeyedJaggedTensor):
        return fids

    # 示例代码
    def lookup(self, fids: JaggedTensor, feat_name: str):
        result = []
        test_num = 10

        for a_id in fids.values():
            result.append([a_id / test_num, a_id / test_num])
        return nn.Parameter(torch.Tensor(result), requires_grad=True)

    def post_input_dist(self, jt: JaggedTensor, feat_name: str):
        indices = self.fids2indices(jt)
        return indices

    def lookup_and_post_dist(self, jt: JaggedTensor, feat_name: str, context: LookupContext):
        embedding = self.lookup(jt, feat_name)
        result = AllGatherEmbeddings().apply(embedding, feat_name,context)
        return result

    def forward(self, kjt_list_each_rank: List[KeyedJaggedTensor]):
        context = self.compute_context(kjt_list_each_rank)
        jt_dict: Dict[str, JaggedTensor] = kjt_list_each_rank[self.rank].to_dict()
        awaitable_dict: Dict[str, LookupAndOutputDistAwaitable] = {}
        for feat_name in jt_dict.keys():
            post_awaitable = PostInpuDistAwaitable(
                self.post_input_dist, jt_dict[feat_name], feat_name
            )
            lookup_and_outdist_awaitable = LookupAndOutputDistAwaitable(
                post_awaitable, self.lookup_and_post_dist, feat_name, context
            )
            awaitable_dict[feat_name] = lookup_and_outdist_awaitable
        return awaitable_dict
