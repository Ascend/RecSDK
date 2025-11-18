#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
import itertools
import logging
import random
import sysconfig
from collections import defaultdict
from dataclasses import dataclass

import pytest
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import SplitTableBatchedEmbeddingBagsCodegen
from hybrid_torchrec.distributed.batched_embedding_kernel import HybridSplitTableBatchedEmbeddingBagsCodegen
from torch.optim import Adam, Adagrad, SGD, SparseAdam

import torchrec
from torchrec import JaggedTensor, KeyedJaggedTensor, PoolingType, ComputeDevice


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

logging.getLogger().setLevel(logging.INFO)
DEVICEID = "npu"
EPOCH = 4
ACCUMULATE_STEP = 5

TORCH_POOLING_MODE_TO_FBGEMM = {
    PoolingType.SUM: PoolingMode.SUM,
    PoolingType.MEAN: PoolingMode.MEAN,
    PoolingType.NONE: PoolingMode.NONE,
}
TORCH_OPTIMIZER_TO_FBGEMM = {
    SparseAdam: EmbOptimType.ADAM,
    Adagrad: EmbOptimType.EXACT_ADAGRAD,
    SGD: EmbOptimType.EXACT_SGD
}
OPTIMIZER_PARAM = {
    SparseAdam: dict(lr=0.01),
    Adagrad: dict(lr=0.01, eps=1.0e-8),
    SGD: dict(lr=0.01),
}


@dataclass
class LookupParams:
    tables: list[list[int]]
    mutile_hots: list[int]
    batch_size: int
    pooling_mode: PoolingMode
    unique: bool
    optim: torch.optim
    feature_map: list[int] = None


def create_data(data_params):
    indices_test = []
    offsets_test = []
    jt_lst = []

    for ind, tid in enumerate(data_params.feature_map):
        table = data_params.tables[tid]  # seq,dim (200, 32)

        indices = torch.randint(0, table[0], (data_params.batch_size * data_params.mutile_hots[ind],)).to(torch.int64)
        indices_test.append(indices)

        offsets = torch.Tensor([data_params.mutile_hots[ind] for _ in range(data_params.batch_size)]).to(torch.int64)
        offsets_test.append(offsets)

        jt_lst.append(JaggedTensor(values=indices, lengths=offsets))

    return indices_test, offsets_test, jt_lst


def generate_unique(jt_lst, feature_map):
    unique_indices = []
    unique_inverse = []
    unique_offset = []
    start = 0

    jt_values = defaultdict(list)
    for ind, tid in enumerate(feature_map):
        jt_values[tid].append(jt_lst[ind].values())

    for key in jt_values:
        jt = torch.cat(jt_values[key])
        unique_indice, inverse = torch.unique(jt, return_inverse=True)
        unique_indices.append(unique_indice)
        unique_inverse.append(inverse)
        unique_offset.extend(len(jt_values[key]) * [start])
        start += unique_indice.shape[0]

    unique_offset.extend([start])
    return unique_indices, unique_inverse, unique_offset


def look_table(indices, offsets, jt_lst, tbe, lookup_params):
    if lookup_params.unique:
        unique_indices_var, unique_inverse_var, unique_offset_var = generate_unique(jt_lst, lookup_params.feature_map)
        unique_indices = torch.cat(unique_indices_var).to(DEVICEID).to(torch.int64)
        unique_inverse = torch.cat(unique_inverse_var).to(DEVICEID).to(torch.int64)
        unique_offset = torch.Tensor(unique_offset_var).to(DEVICEID).to(torch.int64)
        kwargs = dict(unique_indices=unique_indices, unique_offset=unique_offset, unique_inverse=unique_inverse)
    else:
        kwargs = dict()

    output = tbe(indices, offsets, **kwargs)  # bs,dim
    loss = torch.sum(output ** 2 / 2)
    loss.backward()
    return tbe.weights_dev


def concat_tensors_by_category(lookup_tensor):
    num_categories = len(lookup_tensor[0])
    result = []
    for category_idx in range(num_categories):
        tensors = [sublist[category_idx] for sublist in lookup_tensor]
        result.append(torch.cat(tensors))

    return result


def verify_grad_aggregation(lookup_params):
    torch.npu.set_device(DEVICEID)

    embedding_specs = [
        (num_embeddings, embedding_dim, EmbeddingLocation.DEVICE, ComputeDevice.NPU)
        for (num_embeddings, embedding_dim) in lookup_params.tables]

    if lookup_params.unique:
        ebc_class = HybridSplitTableBatchedEmbeddingBagsCodegen
    else:
        ebc_class = SplitTableBatchedEmbeddingBagsCodegen

    tbe_grad_aggregation = ebc_class(
        embedding_specs=embedding_specs,
        optimizer=TORCH_OPTIMIZER_TO_FBGEMM[lookup_params.optim],
        device=torch.device(DEVICEID),
        pooling_mode=TORCH_POOLING_MODE_TO_FBGEMM[lookup_params.pooling_mode],
        feature_table_map=lookup_params.feature_map,
        use_accumulate=True,
        accumulate_step=ACCUMULATE_STEP
    )

    total_size = sum([num_embeddings * embedding_dim for (num_embeddings, embedding_dim) in lookup_params.tables])
    weights_test = torch.randn(total_size).to(torch.float32)
    weights_test = weights_test.to(DEVICEID)

    tbe_grad_aggregation.weights_dev = torch.nn.Parameter(weights_test.clone()).to(DEVICEID)

    tbe_no_grad_aggregation = ebc_class(
        embedding_specs=embedding_specs,
        optimizer=TORCH_OPTIMIZER_TO_FBGEMM[lookup_params.optim],
        device=torch.device(DEVICEID),
        pooling_mode=TORCH_POOLING_MODE_TO_FBGEMM[lookup_params.pooling_mode],
        feature_table_map=lookup_params.feature_map,
        use_accumulate=False)

    tbe_no_grad_aggregation.weights_dev = torch.nn.Parameter(weights_test.clone()).to(DEVICEID)

    weights_grad_aggregation_ls = []
    weights_no_grad_aggregation_ls = []
    for i in range(EPOCH):
        all_idx = []
        all_offsets = []
        # with grad aggregation
        for step in range(ACCUMULATE_STEP):
            indices_test, offsets_test, jt_lst = create_data(lookup_params)
            all_idx.append(indices_test)
            all_offsets.append(offsets_test)

            indices_test = torch.cat(indices_test).to(torch.int64)

            offsets_test = torch.cat(offsets_test).to(torch.int64)
            offsets_test = torch.cat([torch.Tensor([0]), offsets_test]).to(torch.int64)
            offsets_test = torch.cumsum(offsets_test, dim=0)  # [0, 1, 2, 3, 4, 5, 6, 7, 8]

            indices_test = indices_test.to(DEVICEID)
            offsets_test = offsets_test.to(DEVICEID)
            weights_grad_aggregation = look_table(indices_test,
                                                  offsets_test,
                                                  jt_lst,
                                                  tbe_grad_aggregation,
                                                  lookup_params)
        weights_grad_aggregation_ls.append(weights_grad_aggregation)

        # with no grad aggregation
        all_jt_lst = []
        all_idx = concat_tensors_by_category(all_idx)

        all_offsets = concat_tensors_by_category(all_offsets)
        for index in range(len(lookup_params.tables)):
            all_jt_lst.append(JaggedTensor(values=all_idx[index], lengths=all_offsets[index]))

        all_idx = torch.cat(all_idx)

        all_offsets = torch.cat(all_offsets)
        all_offsets = torch.cat([torch.Tensor([0]), all_offsets]).to(torch.int64)
        all_offsets = torch.cumsum(all_offsets, dim=0)

        all_idx = all_idx.to(DEVICEID)
        all_offsets = all_offsets.to(DEVICEID)

        weights_no_grad_aggregation = look_table(all_idx,
                                                 all_offsets,
                                                 all_jt_lst,
                                                 tbe_no_grad_aggregation,
                                                 lookup_params)
        weights_no_grad_aggregation_ls.append(weights_no_grad_aggregation)

    for weights_aggregation, weights_no_aggregation in zip(weights_grad_aggregation_ls, weights_no_grad_aggregation_ls):
        assert torch.allclose(weights_aggregation, weights_no_aggregation, rtol=1e-04, atol=1e-04
                              ), "gloden and result is not closed"


params = {
    "tables": [[(1000, 32), (20000, 32), (3000, 32)]], # the same dim
    "mutile_hots": [[1, 1, 1], [2, 4, 8]],
    "batch_size": [4],
    "pooling_model": [PoolingType.NONE],               # ec
    "unique": [True],                                  # must True
    "optim": [Adagrad, SparseAdam, SGD],
    "feature_map": [[0, 1, 2]]
}


@pytest.mark.parametrize("config", [
    LookupParams(*v) for v in itertools.product(*params.values())
])
def test_verify_grad_aggregation(config: LookupParams):
    verify_grad_aggregation(config)