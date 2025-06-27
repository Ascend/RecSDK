#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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
from torch.optim import Adam, Adagrad, SGD

import torchrec
from torchrec import JaggedTensor, KeyedJaggedTensor, PoolingType, ComputeDevice

logging.getLogger().setLevel(logging.INFO)
DEVICEID = "npu:0"
EPOCH = 1
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

TORCH_POOLING_MODE_TO_FBGEMM = {
    PoolingType.SUM: PoolingMode.SUM,
    PoolingType.MEAN: PoolingMode.MEAN,
    PoolingType.NONE: PoolingMode.NONE,
}
TORCH_OPTIMIZER_TO_FBGEMM = {
    Adam: EmbOptimType.ADAM,
    Adagrad: EmbOptimType.EXACT_ADAGRAD,
    SGD: EmbOptimType.EXACT_SGD
}
OPTIMIZER_PARAM = {
    Adam: dict(lr=0.01),
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


class TestModel(torch.nn.Module):
    def __init__(self, weights, tables, mode):
        super().__init__()
        if mode == PoolingType.NONE:
            collection = torchrec.EmbeddingCollection
            param_name = "embeddings"
        else:
            collection = torchrec.EmbeddingBagCollection
            param_name = "embedding_bags"

        self.param_name = param_name
        self.table_names = list(map(lambda x: x.name, tables))

        self.ec = collection(device="cpu", tables=tables)
        for (table, weight) in zip(tables, weights):
            self.get_table_weights(table.name).copy_(weight)

    def forward(self, kjt):
        return self.ec(kjt)

    def get_table_weights(self, table_name):
        table_dict = getattr(self.ec, self.param_name)
        return table_dict[table_name].weight.data

    def get_all_tables_weights(self):
        return list(map(self.get_table_weights, self.table_names))


def construct_collection_configs(weights, params):
    if params.pooling_mode == PoolingType.NONE:
        table_config_type = torchrec.EmbeddingConfig
        pooling_mode_dict = dict()
    else:
        table_config_type = torchrec.EmbeddingBagConfig
        pooling_mode_dict = dict(pooling=params.pooling_mode)

    features = defaultdict(list)
    for ind, tid in enumerate(params.feature_map):
        features[f"t_{tid}"].append(f"f_{ind}")

    table_configs_list, table_weights_list = [], []
    weights_offset = 0
    for table_id, (num_embeddings, embedding_dim) in enumerate(params.tables):
        table_name = f"t_{table_id}"
        table_configs = table_config_type(
            name=table_name,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=features[table_name],
            **pooling_mode_dict
        )

        # 将一维的weights整理成多张二位的embedding表
        table_size = num_embeddings * embedding_dim
        table_weights = weights[weights_offset:weights_offset + table_size]
        table_weights = table_weights.reshape(num_embeddings, embedding_dim)

        table_configs_list.append(table_configs)
        table_weights_list.append(table_weights)

        weights_offset += table_size
    return table_weights_list, table_configs_list


def lookup_cpu(kjt, weights, params):
    collection_configs = construct_collection_configs(weights, params)
    model = TestModel(*collection_configs, params.pooling_mode)
    model.zero_grad()
    optimizer = params.optim(model.parameters(), **OPTIMIZER_PARAM[params.optim])

    output = None
    for _ in range(EPOCH):
        # forward
        output = model(kjt)

        # 将多个表的查询结果合并
        reshaped_output = [jt.values() if isinstance(jt, JaggedTensor) else jt.view(1, -1) for jt in output.values()]
        output = torch.cat(reshaped_output, dim=0)

        loss = torch.sum(output ** 2 / 2)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    updated_weights = list(map(lambda x: x.reshape(-1), model.get_all_tables_weights()))
    updated_weights = torch.cat(updated_weights, dim=0)
    return output, updated_weights


def lookup_npu(indices, offsets, weights, jt_lst, params):
    torch.npu.set_device(DEVICEID)

    indices = indices.to(DEVICEID)
    offsets = offsets.to(DEVICEID)
    weights = weights.to(DEVICEID)

    embedding_specs = [
        (num_embeddings, embedding_dim, EmbeddingLocation.DEVICE, ComputeDevice.NPU)
        for (num_embeddings, embedding_dim) in params.tables
    ]
    if params.unique:
        ebc_class = HybridSplitTableBatchedEmbeddingBagsCodegen
        unique_indices, unique_inverse, unique_offset = generate_unique(jt_lst, params.feature_map)
        unique_indices = torch.cat(unique_indices).to(DEVICEID).to(torch.int64)
        unique_inverse = torch.cat(unique_inverse).to(DEVICEID).to(torch.int64)
        unique_offset = torch.Tensor(unique_offset).to(DEVICEID).to(torch.int64)
        kwargs = dict(unique_indices=unique_indices, unique_offset=unique_offset, unique_inverse=unique_inverse)
    else:
        ebc_class = SplitTableBatchedEmbeddingBagsCodegen
        kwargs = dict()

    tbe = ebc_class(
        embedding_specs,
        optimizer=TORCH_OPTIMIZER_TO_FBGEMM[params.optim],
        device=torch.device(DEVICEID),
        pooling_mode=TORCH_POOLING_MODE_TO_FBGEMM[params.pooling_mode],
        feature_table_map=params.feature_map,
    )

    tbe.weights_dev = torch.nn.Parameter(weights.clone()).to(DEVICEID)

    output = tbe(indices, offsets, **kwargs)
    loss = torch.sum(output ** 2 / 2)
    loss.backward()
    return output, tbe.weights_dev


def create_data(params):
    total_size = sum([num_embeddings * embedding_dim for (num_embeddings, embedding_dim) in params.tables])

    indices_test = []
    offsets_test = []
    jt_lst = []
    for ind, tid in enumerate(params.feature_map):
        table = params.tables[tid]
        indices = torch.randint(0, table[0], (params.batch_size * params.mutile_hots[ind],)).to(torch.int64)
        indices_test.append(indices)
        offsets = torch.Tensor([params.mutile_hots[ind] for _ in range(params.batch_size)]).to(torch.int64)
        offsets_test.append(offsets)

        jt_lst.append(JaggedTensor(values=indices, lengths=offsets))

    indices_test = torch.cat(indices_test).to(torch.int64)
    offsets_test = torch.cat(offsets_test).to(torch.int64)
    offsets_test = torch.cat([torch.Tensor([0]), offsets_test]).to(torch.int64)
    offsets_test = torch.cumsum(offsets_test, dim=0)

    weights_test = torch.randn(total_size).to(torch.float32)

    jt_dict = {f"f_{i}": jt for i, jt in enumerate(jt_lst)}
    kjt = KeyedJaggedTensor.from_jt_dict(jt_dict)

    return indices_test, offsets_test, weights_test, kjt, jt_lst


def generate_tables(pooling_model):
    tables = []
    mutile_hots = []
    max_batch = 100
    max_tables = 10
    max_rows = 20000
    max_dims = 100
    max_offset = 100
    batches = random.randint(1, max_batch)
    table_num = random.randint(1, max_tables)
    embed_dim = random.randint(1, max_dims) * 8
    for _ in range(table_num):
        row = random.randint(1, max_rows)
        if pooling_model == PoolingType.NONE:
            col = embed_dim
        else:
            col = random.randint(1, max_dims) * 8
        tables.append((row, col))
        mutile_hots.append(random.randint(1, max_offset))
    return tables, mutile_hots, batches


def generate_unique(jt_lst, feature_map):
    unique_indices = []
    unique_inverse = []
    unique_offset = []
    start = 0
    # 合并同一个表的不同feature
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


def execute(params):
    if params.unique and (params.optim in [SGD, Adam]):
        return  # 暂未适配adam unique算子
    if params.feature_map is None:
        params.feature_map = list(range(len(params.tables)))
    indices_test, offsets_test, weights_test, kjt, jt_lst = create_data(params)

    lookup_golden, weights_golden = lookup_cpu(kjt, weights_test, params)
    lookup_npu_result, weights_npu_result = lookup_npu(indices_test, offsets_test, weights_test, jt_lst,
                                                       params)

    total_size = sum([num_embeddings * embedding_dim for (num_embeddings, embedding_dim) in params.tables])
    lookup_npu_result = lookup_npu_result.detach().cpu()
    weights_npu_result = weights_npu_result.detach().cpu()

    logging.info("====== forward ===========")
    lookup_compare = torch.isclose(lookup_golden, lookup_npu_result, 1e-4, 1e-4)
    logging.info((~lookup_compare).sum())
    logging.info(lookup_npu_result[~lookup_compare])
    logging.info(lookup_golden[~lookup_compare])

    logging.info("====== backward ===========")
    weights_compare = torch.isclose(weights_golden, weights_npu_result, 1e-4, 1e-4)
    logging.info((~weights_compare).sum())
    logging.info(torch.arange(total_size)[~weights_compare])
    logging.info(weights_npu_result[~weights_compare])
    logging.info(weights_golden[~weights_compare])

    assert (~lookup_compare).sum() == 0
    assert (~weights_compare).sum() / total_size < 1e-4


@pytest.mark.parametrize("tables", [[(20000, 32), (40000, 32)], [(40000, 128), (80000, 128)]])
@pytest.mark.parametrize("mutile_hots", [[8, 16, 100], [2, 64, 200]])
@pytest.mark.parametrize("batch_size", [8, 16, 64])
@pytest.mark.parametrize("unique", [False])
@pytest.mark.parametrize("feature_map", [[0, 0, 1], [0, 1, 1]])
@pytest.mark.parametrize("pooling_model", [PoolingType.SUM, PoolingType.MEAN, PoolingType.NONE])
@pytest.mark.parametrize("optim", [SGD, Adagrad, Adam])
def test_lookup_two_tables(tables, mutile_hots, batch_size, pooling_model, unique, optim, feature_map):
    params = LookupParams(tables, mutile_hots, batch_size, pooling_model, unique, optim, feature_map)
    execute(params)


@pytest.mark.parametrize("tables", [[(10240, 1024)], [(1234, 1536)], [(1, 8)]])
@pytest.mark.parametrize("mutile_hots", [[1], [4], [11], [69]])
@pytest.mark.parametrize("batch_size", [2341, 1])
@pytest.mark.parametrize("unique", [False])
@pytest.mark.parametrize("pooling_model", [PoolingType.SUM, PoolingType.MEAN, PoolingType.NONE])
@pytest.mark.parametrize("optim", [SGD, Adagrad, Adam])
def test_lookup_backward_one_table(tables, mutile_hots, batch_size, pooling_model, unique, optim):
    params = LookupParams(tables, mutile_hots, batch_size, pooling_model, unique, optim, None)
    execute(params)


@pytest.mark.parametrize("unique", [False])
@pytest.mark.parametrize("pooling_model", [PoolingType.SUM, PoolingType.MEAN, PoolingType.NONE])
@pytest.mark.parametrize("optim", [SGD, Adagrad, Adam])
def test_lookup_multi_tables(pooling_model, unique, optim):
    tables, mutile_hots, batch_size = generate_tables(pooling_model)
    params = LookupParams(tables, mutile_hots, batch_size, pooling_model, unique, optim, None)
    execute(params)
