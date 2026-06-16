# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Tuple, cast

import pytest
import torch

from dynamic_emb.distributed.batched_dynamicemb_table import BatchedDynamicEmbeddingTablesV2
from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
)
from dynamic_emb.distributed.key_value_table import KeyValueTable
from dynamic_emb.distributed.optimizers.base_dynamicemb_optimizer import (
    BaseDynamicEmbeddingOptimizerV2,
    EmbOptimType,
)
from dynamic_emb.distributed.types import Storage, StorageOptionalFilePaths
from dynamic_emb_extensions import EvictStrategy


class PyDictStorage(Storage[DynamicEmbTableOptions, BaseDynamicEmbeddingOptimizerV2]):
    def __init__(
        self,
        options: DynamicEmbTableOptions,
        optimizer: BaseDynamicEmbeddingOptimizerV2,
    ):
        self.options = options
        self.dict: Dict[int, torch.Tensor] = {}
        self.capacity = options.max_capacity
        self.optimizer = optimizer

        self._emb_dim = self.options.dim
        self._emb_dtype = self.options.embedding_dtype
        self._value_dim = self._emb_dim + optimizer.get_state_dim(self._emb_dim)
        self._initial_optim_state = optimizer.get_initial_optim_states()

        device_idx = torch.npu.current_device()
        self.device = torch.device(f"npu:{device_idx}")

    def find_impl(
        self,
        unique_keys: torch.Tensor,
        unique_embs: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_unique_keys = unique_keys.cpu()
        lookup_dim = unique_embs.size(1)
        results = []
        missing_keys = []
        missing_indices = []
        founds_ = []
        for i in range(h_unique_keys.size(0)):
            key = h_unique_keys[i].item()
            if key in self.dict:
                results.append(self.dict[key][0:lookup_dim])
                founds_.append(True)
            else:
                missing_keys.append(key)
                missing_indices.append(i)
                founds_.append(False)
        founds_ = torch.tensor(founds_, dtype=torch.bool, device=self.device)
        if len(results) > 0:
            unique_embs[founds_, :] = torch.cat([t.unsqueeze(0) for t in results], dim=0)
        if founds is not None:
            founds[:] = founds_

        missing_keys = torch.tensor(missing_keys, dtype=unique_keys.dtype, device=self.device)
        missing_indices = torch.tensor(missing_indices, dtype=torch.long, device=self.device)
        return len(missing_keys), missing_keys, missing_indices

    def find_embeddings(
        self,
        unique_keys: torch.Tensor,
        unique_embs: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.find_impl(unique_keys, unique_embs, founds)

    def find(
        self,
        unique_keys: torch.Tensor,
        unique_vals: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.find_impl(unique_keys, unique_vals, founds)

    def insert(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> None:
        h_keys = keys.cpu()
        for i in range(h_keys.size(0)):
            key = h_keys[i].item()
            self.dict[key] = values[i, :].clone()

    def update(self, keys: torch.Tensor, grads: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise ValueError("Can't call update of PyDictStorage")

    def enable_update(self) -> bool:
        return False

    def dump(
        self,
        meta_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        optional_files: Optional[StorageOptionalFilePaths] = None,
    ) -> None:
        raise NotImplementedError

    def load(
        self,
        meta_file_path: str,
        emb_file_path: str,
        embedding_file_path: str,
        include_optim: bool,
        optional_files: Optional[StorageOptionalFilePaths] = None,
    ) -> None:
        raise NotImplementedError

    def embedding_dtype(
        self,
    ) -> torch.dtype:
        return self._emb_dtype

    def embedding_dim(
        self,
    ) -> int:
        return self._emb_dim

    def value_dim(
        self,
    ) -> int:
        return self._value_dim

    def init_optimizer_state(
        self,
    ) -> float:
        return self._initial_optim_state


def _insert_table_weights(
    table: Storage,
    indices: torch.Tensor,
    split: torch.Tensor,
) -> None:
    emb_dim = split.size(1)
    if isinstance(table, KeyValueTable):
        table = cast(KeyValueTable, table)
        val_dim = table.value_dim()
        assert emb_dim == table.embedding_dim()
        values = torch.empty(split.size(0), val_dim, dtype=split.dtype, device=split.device)
        values[:, :emb_dim] = split
        values[:, emb_dim:val_dim] = table.init_optimizer_state()
        if table.evict_strategy() != EvictStrategy.kLru:
            scores = torch.empty(split.size(0), device=indices.device, dtype=torch.int64)
            scores.fill_(1)
        else:
            scores = None
        table.set_score(1)
        table.insert(indices, values, scores)
    elif isinstance(table, PyDictStorage):
        pydict = cast(PyDictStorage, table)
        val_dim = pydict.value_dim()
        assert emb_dim == pydict.embedding_dim()
        values = torch.empty(split.size(0), val_dim, dtype=split.dtype, device=split.device)
        values[:, :emb_dim] = split
        values[:, emb_dim:val_dim] = pydict.init_optimizer_state()
        pydict.insert(indices, values)
    else:
        raise ValueError("Not support table type")


def init_embedding_tables(
    bdet: BatchedDynamicEmbeddingTablesV2,
    num_embs: List[int],
    dims: List[int],
    device: torch.device,
    weights: Optional[List[torch.Tensor]] = None,
) -> List[torch.Tensor]:
    initialized_weights: List[torch.Tensor] = []
    for table_idx, (table, num_emb, emb_dim) in enumerate(zip(bdet.tables, num_embs, dims)):
        indices = torch.arange(num_emb, device=device, dtype=torch.long)
        if weights is not None:
            split = weights[table_idx]
        else:
            split = torch.empty(num_emb, emb_dim, dtype=torch.float32, device=device)
            split.uniform_(0, 1)
        initialized_weights.append(split)
        _insert_table_weights(table, indices, split)
    return initialized_weights


def create_reference_model(
    table_names,
    dyn_emb_table_options_list,
    feature_table_map,
    pooling_mode,
    opt_type,
    opt_params,
    device,
    prefetch_pipeline=False,
):
    return BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=feature_table_map,
        pooling_mode=pooling_mode,
        optimizer=opt_type,
        device=device,
        prefetch_pipeline=prefetch_pipeline,
        **opt_params,
    )


@pytest.fixture(autouse=True)
def setup_npu_device():
    assert torch.npu.is_available()
    torch.npu.set_device(0)


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
        (
            EmbOptimType.ADAM,
            {
                "learning_rate": 0.3,
                "weight_decay": 0.06,
                "eps": 3e-5,
                "beta1": 0.8,
                "beta2": 0.888,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "caching, PS",
    [
        (True, None),
        (False, None),
        (True, PyDictStorage),
    ],
)
def test_forward_train_eval(opt_type, opt_params, caching, PS):
    print(f"step in test_forward_train_eval , opt_type = {opt_type} opt_params = {opt_params}")
    device_id = 0
    device = torch.device(f"npu:{device_id}")

    dims = [8, 8, 8]
    table_names = ["table0", "table1", "table2"]
    key_type = torch.int64
    value_type = torch.float32

    init_capacity = 1024
    max_capacity = 2048

    dyn_emb_table_options_list = []
    for dim in dims:
        dyn_emb_table_options = DynamicEmbTableOptions(
            dim=dim,
            init_capacity=init_capacity,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=caching,
            local_hbm_for_values=1024**3,
            external_storage=PS,
        )
        dyn_emb_table_options_list.append(dyn_emb_table_options)

    bdebt = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=[0, 0, 1, 2],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=opt_type,
        device=device,
        use_index_dedup=True,
        **opt_params,
    )
    # feature number = 4, batch size = 2
    #
    # f0  [0,1],      [12],
    # f1  [64,8],     [12],
    # f2  [15, 2],    [7,105],
    # f3  [],         [0]
    indices = torch.tensor([0, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], dtype=key_type, device=device)
    offsets = torch.tensor([0, 2, 3, 5, 6, 8, 10, 10, 11], dtype=key_type, device=device)

    embs_train = bdebt(indices, offsets)
    torch.npu.synchronize()

    with torch.no_grad():
        bdebt.eval()
        embs_eval = bdebt(indices, offsets)
    torch.npu.synchronize()

    # non-exist key
    indices = torch.tensor([777, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], device=device).to(key_type)
    offsets = torch.tensor([0, 2, 3, 5, 6, 8, 10, 10, 11], device=device).to(key_type)
    embs_non_exist = bdebt(indices, offsets)
    torch.npu.synchronize()

    # train
    bdebt.train()
    embs_train_non_exist = bdebt(indices, offsets)
    torch.npu.synchronize()

    torch.testing.assert_close(embs_train, embs_eval)
    torch.testing.assert_close(embs_train[1:, :], embs_non_exist[1:, :])
    assert torch.all(embs_non_exist[0, :] == 0)
    assert torch.all(embs_train_non_exist[0, :] != 0)
    torch.testing.assert_close(embs_train_non_exist[1:, :], embs_non_exist[1:, :])

    print("all check passed")


# For torchrec's adam optimizer, it will increment the optimizer_step in every forward,
# which will affect the weights update, pay attention to it or try to use `set_optimizer_step()`
# to control(not verified) it.


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
        (
            EmbOptimType.ADAM,
            {
                "learning_rate": 0.3,
                "weight_decay": 0.06,
                "eps": 3e-5,
                "beta1": 0.8,
                "beta2": 0.888,
            },
        ),
        (
            EmbOptimType.EXACT_ADAGRAD,
            {
                "learning_rate": 0.3,
                "eps": 3e-5,
            },
        ),
        (
            EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            {
                "learning_rate": 0.3,
                "eps": 3e-5,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "caching, pooling_mode, dims",
    [
        (True, DynamicEmbPoolingMode.NONE, [8, 8, 8]),
        (False, DynamicEmbPoolingMode.NONE, [16, 16, 16]),
        (False, DynamicEmbPoolingMode.SUM, [128, 32, 16]),
        (False, DynamicEmbPoolingMode.MEAN, [4, 8, 16]),
    ],
)
@pytest.mark.parametrize(
    "PS",
    [
        None,
        PyDictStorage,
    ],
)
def test_backward(opt_type, opt_params, caching, pooling_mode, dims, PS):
    if PS is PyDictStorage and not caching:
        pytest.skip("external_storage requires caching=True")

    print(f"step in test_backward , opt_type = {opt_type} opt_params = {opt_params}")
    device_id = 0
    device = torch.device(f"npu:{device_id}")

    table_names = ["table0", "table1", "table2"]
    key_type = torch.int64
    value_type = torch.float32

    max_capacity = 2048

    dyn_emb_table_options_list = []
    for dim in dims:
        dyn_emb_table_options = DynamicEmbTableOptions(
            dim=dim,
            init_capacity=max_capacity,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=caching,
            local_hbm_for_values=1024**3,
            external_storage=PS,
        )
        dyn_emb_table_options_list.append(dyn_emb_table_options)

    feature_table_map = [0, 0, 1, 2]
    bdeb = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=feature_table_map,
        pooling_mode=pooling_mode,
        optimizer=opt_type,
        device=device,
        **opt_params,
    )
    bdeb_ref = create_reference_model(
        table_names,
        dyn_emb_table_options_list,
        feature_table_map,
        pooling_mode,
        opt_type,
        opt_params,
        device,
    )
    num_embs = [max_capacity // 2 for _ in dims]
    weights = init_embedding_tables(bdeb, num_embs, dims, device)
    init_embedding_tables(bdeb_ref, num_embs, dims, device, weights)
    # feature number = 4, batch size = 2
    #
    # f0  [0,1],      [12],
    # f1  [64,8],     [12],
    # f2  [15, 2, 7], [105],
    # f3  [],         [0]
    for i in range(10):
        indices = torch.tensor([0, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], device=device).to(key_type)
        offsets = torch.tensor([0, 2, 3, 5, 6, 9, 10, 10, 11], device=device).to(key_type)

        embs_bdeb = bdeb(indices, offsets)
        embs_ref = bdeb_ref(indices, offsets)

        torch.npu.synchronize()
        with torch.no_grad():
            torch.testing.assert_close(embs_bdeb, embs_ref, rtol=1e-06, atol=1e-06)

        loss = embs_bdeb.mean()
        loss.backward()
        loss_ref = embs_ref.mean()
        loss_ref.backward()

        torch.npu.synchronize()
        torch.testing.assert_close(loss, loss_ref)

        print(f"Passed iteration {i}")


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
        (
            EmbOptimType.ADAM,
            {
                "learning_rate": 0.3,
                "weight_decay": 0.06,
                "eps": 3e-5,
                "beta1": 0.8,
                "beta2": 0.888,
            },
        ),
        (
            EmbOptimType.EXACT_ADAGRAD,
            {
                "learning_rate": 0.3,
                "eps": 3e-5,
            },
        ),
        (
            EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            {
                "learning_rate": 0.3,
                "eps": 3e-5,
            },
        ),
    ],
)
@pytest.mark.parametrize("PS", [None, PyDictStorage])
def test_prefetch_flush_in_cache(opt_type, opt_params, PS):
    print(f"step in test_prefetch_flush , opt_type = {opt_type} opt_params = {opt_params}")
    device_id = 0
    device = torch.device(f"npu:{device_id}")

    table_names = ["table0", "table1", "table2"]
    key_type = torch.int64
    value_type = torch.float32

    max_capacity = 2048
    dims = [8, 8, 8]

    dyn_emb_table_options_list = []
    for dim in dims:
        dyn_emb_table_options = DynamicEmbTableOptions(
            dim=dim,
            init_capacity=max_capacity,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.STEP,
            caching=True,
            local_hbm_for_values=1024**3,
            external_storage=PS,
        )
        dyn_emb_table_options_list.append(dyn_emb_table_options)

    feature_table_map = [0, 0, 1, 2]
    bdeb = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=feature_table_map,
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=opt_type,
        device=device,
        prefetch_pipeline=False,
        **opt_params,
    )
    bdeb.enable_prefetch = True
    bdeb.set_record_cache_metrics(True)

    bdeb_ref = create_reference_model(
        table_names,
        dyn_emb_table_options_list,
        feature_table_map,
        DynamicEmbPoolingMode.NONE,
        opt_type,
        opt_params,
        device,
    )

    num_embs = [max_capacity // 2 for _ in dims]
    weights = init_embedding_tables(bdeb, num_embs, dims, device)
    init_embedding_tables(bdeb_ref, num_embs, dims, device, weights)

    forward_stream = torch.npu.Stream()
    prefetch_stream = torch.npu.Stream()

    # 1. Prepare input
    # Input A
    # feature number = 4, batch size = 2
    #
    # f0  [0, 1],      [12],
    # f1  [64,8],     [12],
    # f2  [15, 2],    [7,105],
    # f3  [],         [0]
    indicesA = torch.tensor([0, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], device=device).to(key_type)
    offsetsA = torch.tensor([0, 2, 3, 5, 6, 8, 10, 10, 11], device=device).to(key_type)

    # Input B
    # A intersection B is not none
    # feature number = 4, batch size = 2
    #
    # f0  [4, 12],        [55],
    # f1  [2, 17],        [1],
    # f2  [],             [5, 13, 105],
    # f3  [0, 23],        [42]
    indicesB = torch.tensor([4, 12, 55, 2, 17, 1, 5, 13, 105, 0, 23, 42], device=device).to(key_type)
    offsetsB = torch.tensor([0, 2, 3, 5, 6, 6, 9, 11, 12], device=device).to(key_type)

    # stream capture will bring a npuMalloc.
    with torch.npu.stream(forward_stream):
        _ = indicesB + 1
    with torch.npu.stream(prefetch_stream):
        _ = indicesB + 1

    # 2. Test prefetch works when Cache empty
    with torch.npu.stream(prefetch_stream):
        bdeb.prefetch(indicesA, offsetsA, forward_stream)
        assert bdeb.num_prefetch_ahead == 1
        assert list(bdeb.get_score().values()) == [1] * len(dims)

    with torch.npu.stream(forward_stream):
        torch.npu.current_stream().wait_stream(prefetch_stream)
        embs_bdeb_A = bdeb(indicesA, offsetsA)
        loss_bdet_A = embs_bdeb_A.mean()
        loss_bdet_A.backward()

    embs_ref_A = bdeb_ref(indicesA, offsetsA)
    loss_ref_A = embs_ref_A.mean()
    loss_ref_A.backward()

    with torch.no_grad():
        torch.npu.synchronize()
        torch.testing.assert_close(embs_bdeb_A, embs_ref_A, rtol=1e-06, atol=1e-06)
        torch.testing.assert_close(loss_bdet_A, loss_ref_A, rtol=1e-06, atol=1e-06)

        for cache in bdeb.caches:
            metrics = cache.cache_metrics()
            # cache hit_rate = 100% as we do prefetch.
            assert metrics[0].item() == metrics[1].item()

    with torch.no_grad():
        bdeb.flush()
        bdeb.reset_cache_states()

    # 3. Test prefetch works when Cache not empty
    with torch.npu.stream(prefetch_stream):
        bdeb.prefetch(indicesA, offsetsA, forward_stream)
        bdeb.prefetch(indicesB, offsetsB, forward_stream)
        assert bdeb.num_prefetch_ahead == 2
        assert list(bdeb.get_score().values()) == [2] * len(dims)

    with torch.npu.stream(forward_stream):
        torch.npu.current_stream().wait_stream(prefetch_stream)
        embs_bdeb_A = bdeb(indicesA, offsetsA)
        loss_bdet_A = embs_bdeb_A.mean()
        loss_bdet_A.backward()
        embs_bdeb_B = bdeb(indicesB, offsetsB)
        loss_bdet_B = embs_bdeb_B.mean()
        loss_bdet_B.backward()

    embs_ref_A = bdeb_ref(indicesA, offsetsA)
    loss_ref_A = embs_ref_A.mean()
    loss_ref_A.backward()
    embs_ref_B = bdeb_ref(indicesB, offsetsB)
    loss_ref_B = embs_ref_B.mean()
    loss_ref_B.backward()

    with torch.no_grad():
        torch.npu.synchronize()
        torch.testing.assert_close(embs_bdeb_A, embs_ref_A, rtol=1e-06, atol=1e-06)
        torch.testing.assert_close(loss_bdet_A, loss_ref_A, rtol=1e-06, atol=1e-06)
        torch.testing.assert_close(embs_bdeb_B, embs_ref_B, rtol=1e-06, atol=1e-06)
        torch.testing.assert_close(loss_bdet_B, loss_ref_B, rtol=1e-06, atol=1e-06)

        for cache in bdeb.caches:
            metrics = cache.cache_metrics()
            # cache hit_rate = 100% as we do prefetch.
            assert metrics[0].item() == metrics[1].item()
