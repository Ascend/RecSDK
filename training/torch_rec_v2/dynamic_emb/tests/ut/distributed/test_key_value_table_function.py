#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import pytest
from typing import Optional, Tuple, Dict, Callable

from dynamic_emb.distributed.key_value_table import KeyValueTableFunction

npu_device = torch.device("npu:0")

class Storage:
    pass

class BaseDynamicEmbeddingOptimizerV2:
    """优化器基类"""
    pass


class MockStorage(Storage):
    def __init__(self, emb_dim: int = 16):
        self.emb_dim = emb_dim
        self.device = npu_device  
        self.emb_dtype = torch.float32
        self.storage_data: Dict[int, torch.Tensor] = {}  # 存储插入的数据

    def embedding_dim(self) -> int:
        return self.emb_dim

    def embedding_dtype(self) -> torch.dtype:
        return self.emb_dtype

    def value_dim(self) -> int:
        """ embedding+优化器状态 """
        return self.emb_dim * 2

    def find_embeddings(self, unique_keys: torch.Tensor, unique_embs: torch.Tensor, founds: torch.Tensor) \
          -> Tuple[int, torch.Tensor, torch.Tensor]:
        """ 查找嵌入，模拟一半key缺失"""
        total_keys = unique_keys.numel()
        device = unique_keys.device
        
        # 模拟：前一半key存在，后一半key缺失
        num_missing = total_keys // 2
        founds[:] = True
        founds[-num_missing:] = False if num_missing > 0 else True
        
        # 存在的key直接赋值随机数，缺失的返回索引
        existing_num = total_keys - num_missing
        if existing_num > 0:
            tmp = torch.empty_like(unique_embs[:existing_num])
            tmp.normal_()
            unique_embs[:existing_num] = tmp
        
        # 缺失的key和索引
        missing_keys = unique_keys[-num_missing:] \
            if num_missing > 0 else torch.tensor([], device=device, dtype=torch.long)
        missing_indices = torch.arange(existing_num, total_keys, device=device, dtype=torch.long) \
              if num_missing > 0 else torch.tensor([], device=device, dtype=torch.long)
        
        return num_missing, missing_keys, missing_indices

    def find(self, keys: torch.Tensor, values: torch.Tensor, founds: Optional[torch.Tensor] = None) \
        -> Tuple[int, torch.Tensor, torch.Tensor]:
        if founds is not None:
            founds[:] = True  # 模拟所有key都找到

        tmp = torch.empty_like(values[:, :self.emb_dim])
        tmp.normal_()
        values[:, :self.emb_dim] = tmp
        return 0, torch.tensor([], device=self.device, dtype=torch.long), torch.tensor([], device=self.device, dtype=torch.long)

    def init_optimizer_state(self) -> torch.Tensor:
        return torch.zeros(self.emb_dim, dtype=self.emb_dtype, device=npu_device)

    def enable_update(self) -> bool:
        return False

    def update(self, keys: torch.Tensor, grads: torch.Tensor, return_missing: bool = False):
        pass

    def insert(self, keys: torch.Tensor, values: torch.Tensor):
        for k, v in zip(keys.tolist(), values):
            self.storage_data[k] = v.clone()

class MockOptimizer(BaseDynamicEmbeddingOptimizerV2):
    def fused_update(self, grads: torch.Tensor, values: torch.Tensor):
        # 模拟更新：值 = 值 - 梯度
        values[:, :values.shape[1]//2] -= grads

class MockInitializer:
    def __call__(self, embs: torch.Tensor, indices: torch.Tensor, keys: torch.Tensor):
        """初始化缺失的嵌入向量"""
        tmp = torch.empty_like(embs[indices])
        tmp.normal_()
        embs[indices] = tmp


@pytest.mark.parametrize("training", [True, False])
def test_key_value_table_lookup(training: bool):
    storage = MockStorage(emb_dim=16)
    initializer = MockInitializer()

    # 测试数据
    unique_keys = torch.tensor([10, 20, 30, 40], dtype=torch.long, device=npu_device)
    unique_embs = torch.zeros(4, 16, dtype=torch.float32, device=npu_device)

    # lookup
    KeyValueTableFunction.lookup(
        storage=storage,
        unique_keys=unique_keys,
        unique_embs=unique_embs,
        initializer=initializer,
        training=training
    )

    assert not torch.all(unique_embs == 0)
    assert unique_embs.shape == (4, 16)
    if training:
        assert len(storage.storage_data) == 2
    else:
        assert len(storage.storage_data) == 0
    print("√ lookup 测试通过")

def test_key_value_table_update():
    storage = MockStorage(emb_dim=16)
    optimizer = MockOptimizer()

    # 测试数据
    unique_keys = torch.tensor([5, 6, 7], dtype=torch.long, device=npu_device)
    unique_grads = torch.randn(3, 16, dtype=torch.float32, device=npu_device)

    # 执行update
    KeyValueTableFunction.update(
        storage=storage,
        unique_keys=unique_keys,
        unique_grads=unique_grads,
        optimizer=optimizer
    )

    assert len(storage.storage_data) == 3
    print("√ update 测试通过")

def test_key_value_table_lookup_invalid_dim():
    """ 非1维key触发异常 """
    storage = MockStorage()
    initializer = MockInitializer()
    
    # 构造2维key（非法输入）
    invalid_keys = torch.tensor([[1,2], [3,4]], dtype=torch.long, device=npu_device)
    unique_embs = torch.randn(2, 16, device=npu_device)

    with pytest.raises(RuntimeError) as excinfo:
        KeyValueTableFunction.lookup(storage, invalid_keys, unique_embs, initializer, training=True)
    
    assert "unique_keys dim not equal 1" in str(excinfo.value)
    print("√ lookup 异常测试通过")

def test_key_value_table_lookup_empty_keys():
    """ 空key边界情况 """
    storage = MockStorage()
    initializer = MockInitializer()
    
    empty_keys = torch.tensor([], dtype=torch.long, device=npu_device)
    unique_embs = torch.empty(0, 16, device=npu_device)

    KeyValueTableFunction.lookup(storage, empty_keys, unique_embs, initializer, training=True)
    print("√ lookup 空key测试通过")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])