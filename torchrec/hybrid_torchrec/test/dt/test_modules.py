#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import pytest

import torch

from hybrid_torchrec.constants import (
    MAX_EMBEDDINGS_DIM,
    MAX_NUM_EMBEDDINGS,
    EMBEDDINGS_DIM_ALIGNMENT,
)
from hybrid_torchrec.modules.hash_embeddingbag import (
    reorder_inverse_indices,
    process_pooled_embeddings,
    is_valid_feat_name,
    check_embedding_config_valid,
    HybridHashTable,
    HashEmbeddingBag,
    HashEmbeddingBagConfig,
    HashEmbeddingBagCollection
)
from torchrec import KeyedJaggedTensor, KeyedTensor
from torchrec.modules.embedding_configs import (
    DataType,
    PoolingType,
)


class TestReorderInverseIndices:
    @staticmethod
    def test_basic_reordering():
        # 输入特征名与索引张量
        inverse_indices = (["featA", "featB", "featC"], torch.tensor([10, 20, 30]))
        # 目标特征名（带@后缀）
        feature_names = ["featB@v1", "featA@v2"]
        result = reorder_inverse_indices(inverse_indices, feature_names)
        expected = torch.tensor([20, 10])  # 按featB, featA顺序提取

        assert torch.equal(result, expected)

    @staticmethod
    def test_empty_input():
        assert torch.equal(reorder_inverse_indices(None, ["featX"]), torch.empty(0))


class TestProcessPooledEmbeddings:
    @staticmethod
    def test_empty_inverse_indices():
        # 空索引测试
        embeddings = [torch.randn(3, 4), torch.randn(3, 5)]
        empty_indices = torch.tensor([])
        result = process_pooled_embeddings(embeddings, empty_indices)
        assert result.shape == (3, 9)  # 直接拼接4+5维

    @staticmethod
    def test_index_selection():
        # 索引选择功能测试
        emb1 = torch.tensor([[1, 1], [2, 2]])
        emb2 = torch.tensor([[3, 3], [4, 4]])
        indices = torch.tensor([1, 0])  # 倒序索引

        # 模拟fbgemm操作结果
        expected = torch.cat([
            torch.index_select(emb1, 0, indices),
            torch.index_select(emb2, 0, indices)
        ], dim=1)

        with unittest.mock.patch(
                'torch.ops.fbgemm.group_index_select_dim0',
                side_effect=lambda x, y: [torch.index_select(t, 0, indices) for t in x]
        ):
            result = process_pooled_embeddings([emb1, emb2], indices)
            assert torch.equal(result, expected)


class TestIsValidFeatname:
    @staticmethod
    def test_valid_feature_names():
        """测试合法特征名"""
        assert is_valid_feat_name("feature_1") == True
        assert is_valid_feat_name("FEATURE2") == True
        assert is_valid_feat_name("_private_feat") == True
        assert is_valid_feat_name("a" * 100) == True  # 长字符串测试

    @staticmethod
    def test_invalid_feature_names():
        """测试非法字符"""
        assert is_valid_feat_name("feature@") == False  # 特殊字符
        assert is_valid_feat_name("space in") == False  # 空格
        assert is_valid_feat_name("dash-ed") == False  # 连字符

    @staticmethod
    def test_edge_cases():
        """边界条件测试"""
        assert is_valid_feat_name("") == True  # 空字符串
        assert is_valid_feat_name("_") == True  # 单下划线
        assert is_valid_feat_name("1") == True  # 纯数字


class TestEmbeddingConfigValid:
    @staticmethod
    def test_valid_config():
        """测试完全合法的配置"""
        config = HashEmbeddingBagConfig(
            embedding_dim=16,
            num_embeddings=100,
            data_type=DataType.FP32,
            feature_names=["valid_feat1", "valid_feat2"],
            pooling=PoolingType.SUM
        )
        # 不应该抛出异常
        check_embedding_config_valid(config)

    @staticmethod
    def test_embedding_dim_alignment():
        """测试embedding_dim对齐检查"""
        with pytest.raises(ValueError, match="multiple of 8"):
            config = HashEmbeddingBagConfig(
                embedding_dim=100,  # 不是8的倍数
                num_embeddings=100
            )
            check_embedding_config_valid(config)

    @staticmethod
    def test_embedding_dim_range():
        """测试embedding_dim范围检查"""
        # 测试下边界
        with pytest.raises(ValueError, match=f"should be in"):
            config = HashEmbeddingBagConfig(
                embedding_dim=0,  # 小于8
                num_embeddings=100
            )
            check_embedding_config_valid(config)
        # 测试上边界
        with pytest.raises(ValueError, match=f"should be in"):
            config = HashEmbeddingBagConfig(
                embedding_dim=MAX_EMBEDDINGS_DIM + 8,
                num_embeddings=100
            )
            check_embedding_config_valid(config)

    @staticmethod
    def test_num_embeddings_range():
        """测试num_embeddings范围检查"""
        # 测试下边界
        with pytest.raises(ValueError, match=f"should be in"):
            config = HashEmbeddingBagConfig(
                embedding_dim=8,
                num_embeddings=0.5,  # 小于1
            )
            check_embedding_config_valid(config)
        # 测试上边界
        with pytest.raises(ValueError, match=f"should be in"):
            config = HashEmbeddingBagConfig(
                embedding_dim=8,
                num_embeddings=MAX_NUM_EMBEDDINGS + 8,
            )
            check_embedding_config_valid(config)

    @staticmethod
    def test_data_type_validation():
        """测试数据类型检查"""
        with pytest.raises(ValueError, match="should be FP32"):
            config = HashEmbeddingBagConfig(
                embedding_dim=8,
                num_embeddings=100,
                data_type=DataType.FP16,  # 不是FP32
            )
            check_embedding_config_valid(config)

    @staticmethod
    def test_feature_names_validation():
        """测试特征名检查"""
        # 测试空特征名
        with pytest.raises(ValueError, match="should not be empty"):
            config = HashEmbeddingBagConfig(
                embedding_dim=8,
                num_embeddings=100,
                feature_names=[] # should not be empty
            )
            check_embedding_config_valid(config)

        # 测试非法特征名
        with pytest.raises(ValueError, match="should contain a-Z, 0-9, _"):
            config = HashEmbeddingBagConfig(
                embedding_dim=8,
                num_embeddings=100,
                feature_names=["valid_feat", "invalid@feat"]
            )
            check_embedding_config_valid(config)

    @staticmethod
    def test_weight_init_validation():
        """测试权重初始化参数检查"""
        with pytest.raises(ValueError, match="should be None"):
            config = HashEmbeddingBagConfig(
                embedding_dim=8,
                num_embeddings=100,
                feature_names=["feat"],
                weight_init_max=1.0 # should be None
            )
            check_embedding_config_valid(config)

        with pytest.raises(ValueError, match="should be None"):
            config = HashEmbeddingBagConfig(
                embedding_dim=8,
                num_embeddings=100,
                feature_names=["feat"],
                weight_init_min=1.0 # should be None
            )
            check_embedding_config_valid(config)

    @staticmethod
    def test_num_embeddings_post_pruning_validation():
        with pytest.raises(ValueError, match="should be None"):
            config = HashEmbeddingBagConfig(
                embedding_dim=8,
                num_embeddings=100,
                feature_names=["feat"],
                num_embeddings_post_pruning="Not None"  # should be None
            )
            check_embedding_config_valid(config)

    @staticmethod
    def test_pooling_type_validation():
        """测试池化类型检查"""
        with pytest.raises(ValueError, match="should be in"):
            config = HashEmbeddingBagConfig(
                embedding_dim=8,
                num_embeddings=100,
                feature_names=["feat"],
                pooling="INVALID"  # 不是枚举值
            )
            check_embedding_config_valid(config)

    @staticmethod
    def test_init_fn_validation():
        """测试初始化函数检查"""
        with pytest.raises(ValueError, match="should be callable"):
            config = HashEmbeddingBagConfig(
                embedding_dim=8,
                num_embeddings=100,
                feature_names=["feat"],
                init_fn="not_callable"  # 不是可调用对象
            )
            check_embedding_config_valid(config)


class TestHashEmbeddingBagCollection:
    # 测试用例
    @pytest.mark.parametrize("embedding_dims", [[32, 64], [128, 256]])
    @pytest.mark.parametrize("num_embeddings", [[1024, 512]])
    @pytest.mark.parametrize("pooling_type", [PoolingType.SUM, PoolingType.MEAN])
    def test_hash_embedding_bag_collection(self,
                                           embedding_dims,
                                           num_embeddings,
                                           pooling_type
                                           ):
        # 创建测试配置
        config1 = HashEmbeddingBagConfig(
            name="table1",
            embedding_dim=embedding_dims[0],
            num_embeddings=num_embeddings[0],
            feature_names=["feature1"],
            data_type=DataType.FP32,
            pooling=pooling_type
        )
        config2 = HashEmbeddingBagConfig(
            name="table2",
            embedding_dim=embedding_dims[1],
            num_embeddings=num_embeddings[1],
            feature_names=["feature2"],
            data_type=DataType.FP32,
            pooling=pooling_type
        )
        config3 = HashEmbeddingBagConfig(
            name="table1",
            embedding_dim=embedding_dims[1],
            num_embeddings=num_embeddings[1],
            feature_names=["feature2"],
            data_type=DataType.FP32,
            pooling=pooling_type
        )

        # 初始化模型
        model = HashEmbeddingBagCollection(
            tables=[config1, config2],
            is_weighted=False,
            device="cpu"
        )

        # 创建测试输入
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature1", "feature2"],
            values=torch.tensor([1, 2, 3, 4, 5, 6]),
            lengths=torch.tensor([2, 0, 1, 1, 2, 0])
        )

        # 前向传播
        output = model(features)

        # 验证输出
        assert isinstance(output, KeyedTensor)
        assert output.keys() == ["feature1", "feature2"]
        assert output.values().shape == (3, sum(embedding_dims))  # 64+32=96维

        # 表名相同
        with pytest.raises(ValueError, match="Duplicate table name"):
            model = HashEmbeddingBagCollection(
                tables=[config1, config3],
                is_weighted=False,
                device="cpu"
            )
        # device不支持
        with pytest.raises(NotImplementedError, match="not implemented"):
            model = HashEmbeddingBagCollection(
                tables=[config1, config2],
                is_weighted=False,
                device="dpu"
            )

    @staticmethod
    def test_reset_parameters():
        # 创建测试配置
        config = HashEmbeddingBagConfig(
            name="test_table",
            embedding_dim=16,
            num_embeddings=100,
            feature_names=["test_feature"],
            data_type=DataType.FP32
        )
        # 初始化模型
        model = HashEmbeddingBagCollection(
            tables=[config],
            device="cpu"
        )
        # 获取初始权重
        original_weight = model.embedding_bags["test_table"].weight.clone()
        # 重置参数
        model.reset_parameters()
        # 获取重置后的权重
        reset_weight = model.embedding_bags["test_table"].weight
        # 验证权重已改变
        assert not torch.allclose(original_weight, reset_weight), "权重未重置"
        # 验证权重形状
        assert reset_weight.shape == (100, 16), "权重形状错误"


class TestHashEmbeddingBag:
    @staticmethod
    def test_initialization():
        config = HashEmbeddingBagConfig(
            name="test_table",
            embedding_dim=16,
            num_embeddings=100
        )
        device = "cpu"
        model = HashEmbeddingBag(config=config, device=device)
        assert isinstance(model, torch.nn.Module)

    @staticmethod
    def test_find_and_insert():
        config = HashEmbeddingBagConfig(
            name="test_table",
            embedding_dim=8,
            num_embeddings=50
        )
        device = torch.device("cpu")
        model = HashEmbeddingBag(config=config, device=device)

        keys = torch.tensor([1, 2, 3])
        values = torch.randn(3, 8)
        scores = torch.tensor([0.9, 0.8, 0.7])
        founds = torch.tensor([0, 0, 0], dtype=torch.bool)

        assert model.find_and_insert(keys, values, scores, founds) == NotImplemented

    @staticmethod
    def test_forward():
        config = HashEmbeddingBagConfig(
            name="test_table",
            embedding_dim=4,
            num_embeddings=10
        )
        device = torch.device("cpu")
        model = HashEmbeddingBag(config=config, device=device)
        input_tensor = torch.tensor([1, 2, 3, 4])
        offsets = torch.tensor([0, 2, 4])

        assert model.forward(input_tensor, offsets) == NotImplemented
