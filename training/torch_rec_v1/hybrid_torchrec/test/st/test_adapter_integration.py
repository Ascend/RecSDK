#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""适配器模块系统测试。

测试覆盖：
1. 适配器在实际训练流程中的集成
2. 版本适配对训练结果的正确性影响
3. 多版本API兼容性验证
"""

import logging
import pytest

import torch
from torch.utils.data import DataLoader

from dataset import RandomRecDataset
from hybrid_torchrec import (
    HashEmbeddingBagCollection,
    HashEmbeddingBagConfig,
    IS_TORCH_REC_120,
    IS_TORCH_REC_150,
)
from hybrid_torchrec._adapters import adapter
from model import Model
from util import setup_logging

import torchrec


LOOP_TIMES = 4
BATCH_NUM = 16


def weight_init(param: torch.nn.Parameter):
    if len(param.shape) != 2:
        return
    torch.manual_seed(param.shape[1])
    result = torch.randn((1, param.shape[1])).repeat(param.shape[0], 1)
    param.data.copy_(result)


def generate_hash_config(embedding_dims, num_embeddings, pool_type):
    test_table_configs = []
    for i, (table_dim, num_embedding) in enumerate(zip(embedding_dims, num_embeddings)):
        config = HashEmbeddingBagConfig(
            name=f"table{i}",
            embedding_dim=table_dim,
            num_embeddings=num_embedding,
            feature_names=[f"feat{i}"],
            pooling=pool_type,
            init_fn=weight_init,
        )
        test_table_configs.append(config)
    return test_table_configs


@pytest.mark.parametrize("embedding_dims", [[32, 64]])
@pytest.mark.parametrize("num_embeddings", [[400, 4000]])
@pytest.mark.parametrize("pool_type", [torchrec.PoolingType.MEAN])
@pytest.mark.parametrize("lookup_len", [1024])
@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_adapter_integration_train(embedding_dims, num_embeddings, pool_type, lookup_len, device):
    """测试适配器在训练流程中的集成。"""
    setup_logging(rank=0)
    logging.info(f"Testing adapter integration with torchrec {adapter.version}")

    # 验证版本标志与适配器版本一致
    assert IS_TORCH_REC_120 == (adapter.version == (1, 2, 0))
    assert IS_TORCH_REC_150 == (adapter.version == (1, 5, 0))

    # 使用兼容性工具方法创建配置
    test_table_configs = generate_hash_config(embedding_dims, num_embeddings, pool_type)

    num_features = sum([c.num_features() for c in test_table_configs])
    dataset = RandomRecDataset(BATCH_NUM, lookup_len, num_embeddings, len(test_table_configs))

    # 创建模型并训练
    ebc_hash = HashEmbeddingBagCollection(tables=test_table_configs, device=device)
    test_model = Model(ebc_hash, num_features).to(device)
    test_dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
    )

    # 训练循环
    opt = torch.optim.Adagrad(test_model.parameters(), lr=0.02, eps=1e-8)
    results = []
    iter_ = iter(test_dataloader)

    for _ in range(LOOP_TIMES):
        batch = next(iter_).to(device)
        opt.zero_grad()
        loss, output = test_model(batch)
        results.append(loss.detach().cpu())
        results.append(output.detach().cpu())
        loss.backward()
        opt.step()

    # 验证结果
    assert len(results) == LOOP_TIMES * 2
    for result in results:
        assert not torch.isnan(result).any(), "NaN detected in training results"

    logging.info(f"Adapter integration test passed for torchrec {adapter.version}")


@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_adapter_version_compatibility(device):
    """测试不同版本的API兼容性。"""
    setup_logging(rank=0)
    logging.info(f"Testing API compatibility for torchrec {adapter.version}")

    # 测试兼容性工具方法
    from torchrec.modules.embedding_configs import EmbeddingTableConfig

    # 使用兼容性方法创建配置（包含可能不存在的字段）
    cfg = adapter.make_embedding_table_config(
        num_embeddings=100,
        embedding_dim=8,
        name="test_table",
        # 这些字段在某些版本可能不存在
        total_num_buckets=1000,
        use_virtual_table=False,
        virtual_table_eviction_policy="lru",
        enable_embedding_update=True,
    )

    assert isinstance(cfg, EmbeddingTableConfig)
    assert cfg.num_embeddings == 100

    # 测试枚举值获取
    kernel_values = adapter.embedding_compute_kernel_values(
        "DENSE", "SPARSE", "SSD_VIRTUAL_TABLE", "DRAM_VIRTUAL_TABLE"
    )
    assert isinstance(kernel_values, set)
    assert len(kernel_values) > 0

    logging.info(f"API compatibility test passed for torchrec {adapter.version}")


@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_adapter_methods_in_training(device):
    """测试适配器方法在训练中的实际调用。"""
    setup_logging(rank=0)
    logging.info(f"Testing adapter methods in training context for torchrec {adapter.version}")

    # 测试 get_learning_rate 方法
    from dataclasses import dataclass

    @dataclass
    class MockCommonArgs:
        learning_rate: float = 0.5

    @dataclass
    class MockOptimizerArgs:
        learning_rate: float = 0.1

    lr = adapter.get_learning_rate(MockCommonArgs(), MockOptimizerArgs())

    # 根据版本验证结果
    if adapter.version >= (1, 2, 0):
        assert lr == 0.5, f"Expected 0.5 for v1.2.0+, got {lr}"
    else:
        assert lr == 0.1, f"Expected 0.1 for v1.1.0, got {lr}"

    # 测试 get_kernel_learning_rate
    class MockKernel:
        def get_learning_rate(self):
            return 0.3

    kernel_lr = adapter.get_kernel_learning_rate(MockKernel())

    if adapter.version >= (1, 2, 0):
        assert kernel_lr == 0.3, f"Expected 0.3 for v1.2.0+, got {kernel_lr}"
    else:
        assert kernel_lr == 0.0, f"Expected 0.0 for v1.1.0, got {kernel_lr}"

    logging.info(f"Adapter methods test passed for torchrec {adapter.version}")


@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_adapter_singleton_consistency(device):
    """测试适配器单例的一致性。"""
    setup_logging(rank=0)
    logging.info(f"Testing adapter singleton consistency for torchrec {adapter.version}")

    from hybrid_torchrec._adapters import adapter as adapter1
    from hybrid_torchrec._adapters import _create_adapter

    adapter2 = _create_adapter()

    # 验证是同一个实例
    assert adapter1 is adapter2, "Adapter instances should be the same"
    assert adapter1.version == adapter2.version, "Adapter versions should match"

    # 验证多次导入的适配器相同
    from hybrid_torchrec._adapters import adapter as adapter3

    assert adapter1 is adapter3, "Adapter should be a singleton"

    logging.info("Adapter singleton test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
