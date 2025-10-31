#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import pytest

import torch
from torchrec.modules.embedding_modules import reorder_inverse_indices, process_pooled_embeddings

from hybrid_torchrec.constants import (
    MAX_NUM_TABLES,
    MAX_WORLD_SIZE,
    MAX_BATCH_SIZE
)
from torchrec_embcache.distributed.embedding_bag import (
    EmbCacheEmbeddingBagConfig,
    EmbCacheEmbeddingBagCollection
)
from torchrec import KeyedJaggedTensor, KeyedTensor
from torchrec.modules.embedding_configs import (
    DataType,
    PoolingType,
)


def _create_emb_cache_ebc_config(embedding_dims, num_embeddings, pooling_type):
    configs = []
    for i, emb_dim in enumerate(embedding_dims):
        configs.append(
            EmbCacheEmbeddingBagConfig(
                name=f"table{i}",
                embedding_dim=emb_dim,
                num_embeddings=num_embeddings[i],
                feature_names=[f"feature{i}"],
                data_type=DataType.FP32,
                pooling=pooling_type
            )
        )
    return configs


def _forward_impl(self, features: KeyedJaggedTensor) -> KeyedTensor:
    flat_feature_names: List[str] = []
    for names in self._feature_names:
        flat_feature_names.extend(names)
    inverse_indices = reorder_inverse_indices(
        inverse_indices=features.inverse_indices_or_none(),
        feature_names=flat_feature_names,
    )
    pooled_embeddings: List[torch.Tensor] = []
    feature_dict = features.to_dict()
    for i, embedding_bag in enumerate(self.embedding_bags.values()):
        for feature_name in self._feature_names[i]:
            f = feature_dict[feature_name]
            res = embedding_bag(
                input_feat=f.values(),
                offsets=f.offsets(),
                per_sample_weights=f.weights() if self._is_weighted else None,
            ).float()
            pooled_embeddings.append(res)
    return KeyedTensor(
        keys=self._embedding_names,
        values=process_pooled_embeddings(
            pooled_embeddings=pooled_embeddings,
            inverse_indices=inverse_indices,
        ),
        length_per_key=self._lengths_per_embedding,
    )


class TestEmbCacheEmbeddingBagCollection:
    # 测试用例
    @pytest.mark.parametrize("embedding_dims", [[32, 8], [16, 32]])
    @pytest.mark.parametrize("num_embeddings", [[1024, 512]])
    @pytest.mark.parametrize("pooling_type", [PoolingType.SUM, PoolingType.MEAN])
    def test_hash_embedding_bag_collection(self,
                                           embedding_dims,
                                           num_embeddings,
                                           pooling_type
                                           ):
        # 重写下forward。 EmbCacheEmbeddingBagCollection未实现直调forward，只能调用到父类。
        # 但父类forward中，往每个表调用前向时，入参和EmbCacheEmbeddingBagCollection中子表前向入参有差异
        EmbCacheEmbeddingBagCollection.forward = _forward_impl

        ebc_configs = _create_emb_cache_ebc_config(embedding_dims, num_embeddings, pooling_type)
        # 初始化模型
        model = EmbCacheEmbeddingBagCollection(
            tables=ebc_configs,
            world_size=1,
            batch_size=1,
            multi_hot_sizes=[1] * len(ebc_configs),
            is_weighted=False,
        )

        # 构造输入
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature0", "feature1"],
            values=torch.tensor([1, 2, 3, 4, 5, 6]),
            lengths=torch.tensor([2, 0, 1, 1, 2, 0])
        )

        # 前向传播
        result = model(features)

        # 验证输出
        assert isinstance(result, KeyedTensor)
        assert result.keys() == ["feature0", "feature1"]
        assert result.values().shape == (3, sum(embedding_dims))

    @staticmethod
    def test_invalid_emb_config_params():
        """测试HashEmbeddingBagCollection参数检查"""
        def _create_table_configs(table_num: int):
            return [EmbCacheEmbeddingBagConfig(
                name=f"table{i}",
                embedding_dim=8,
                num_embeddings=400,
                feature_names=["feature1"],
                data_type=DataType.FP32,
                pooling=PoolingType.SUM,
            ) for i in range(table_num)]

        with pytest.raises(ValueError, match=f"{MAX_NUM_TABLES}"):
            invalid_config_num = MAX_NUM_TABLES + 1
            # tables列表长度超过上限
            _ = EmbCacheEmbeddingBagCollection(
                tables=_create_table_configs(invalid_config_num),
                world_size=1,
                batch_size=1,
                multi_hot_sizes=[1] * invalid_config_num,
                device=torch.device("cpu"),
            )
        with pytest.raises(ValueError, match="must be False"):
            _ = EmbCacheEmbeddingBagCollection(
                tables=_create_table_configs(1),
                world_size=1,
                batch_size=1,
                multi_hot_sizes=[1],
                is_weighted=True,  # 不支持True
                device=torch.device("cpu"),
            )
        with pytest.raises(ValueError, match="should be a list"):
            _ = EmbCacheEmbeddingBagCollection(
                tables="param is not list object",  # 参数类型错误
                world_size=1,
                batch_size=1,
                multi_hot_sizes=[1],
                device=torch.device("cpu"),
            )
        with pytest.raises(ValueError, match="EmbCacheEmbeddingBagConfig"):
            tables = _create_table_configs(1)
            tables.append("str")
            _ = EmbCacheEmbeddingBagCollection(
                tables=tables,  # 列表中不支持的元素类型
                world_size=1,
                batch_size=1,
                multi_hot_sizes=[1] * len(tables),
                device=torch.device("cpu"),
            )
        with pytest.raises(ValueError, match="should be equal"):
            _ = EmbCacheEmbeddingBagCollection(
                tables=_create_table_configs(1),
                world_size=1,
                batch_size=1,
                multi_hot_sizes=[1] * 2,  # multi_hot_sizes列表长度和tables长度不相等
                device=torch.device("cpu"),
            )
        with pytest.raises(ValueError, match="should be a list of int"):
            _ = EmbCacheEmbeddingBagCollection(
                tables=_create_table_configs(1),
                world_size=1,
                batch_size=1,
                multi_hot_sizes=["str type"],  # multi_hot_sizes类型不支持
                is_weighted=False,
            )
        with pytest.raises(ValueError, match=f"{MAX_WORLD_SIZE}"):
            _ = EmbCacheEmbeddingBagCollection(
                tables=_create_table_configs(1),
                world_size=MAX_WORLD_SIZE + 1,   # 超出阈值
                batch_size=1,
                multi_hot_sizes=[1],
                is_weighted=False,
            )
        with pytest.raises(ValueError, match=f"{MAX_WORLD_SIZE}"):
            _ = EmbCacheEmbeddingBagCollection(
                tables=_create_table_configs(1),
                world_size="sss",   # 类型错误
                batch_size=1,
                multi_hot_sizes=[1],
                is_weighted=False,
            )
        with pytest.raises(ValueError, match=f"{MAX_BATCH_SIZE}"):
            _ = EmbCacheEmbeddingBagCollection(
                tables=_create_table_configs(1),
                world_size=1,
                batch_size=MAX_BATCH_SIZE + 1,  # 超出阈值
                multi_hot_sizes=[1],
                is_weighted=False,
            )

