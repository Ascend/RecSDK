#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import unittest.mock
from types import SimpleNamespace

import pytest

import torch

from hybrid_torchrec.constants import MAX_EMBEDDINGS_DIM, MAX_NUM_EMBEDDINGS, MAX_NUM_TABLES
from hybrid_torchrec._adapters import _build_methods
from hybrid_torchrec._adapters._adapter_base import TorchRecVersionAdapter
from hybrid_torchrec._adapters._version_diff import (
    check_config_new_item_v120,
    check_config_new_item_v150,
)
from hybrid_torchrec.modules.hash_embeddingbag import (
    reorder_inverse_indices,
    process_pooled_embeddings,
    _check_name_format,
    check_embedding_config_valid,
    HashEmbeddingBag,
    HashEmbeddingBagConfig,
    HashEmbeddingBagCollection,
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
        expected = torch.cat([torch.index_select(emb1, 0, indices), torch.index_select(emb2, 0, indices)], dim=1)

        with unittest.mock.patch(
            'torch.ops.fbgemm.group_index_select_dim0',
            side_effect=lambda x, y: [torch.index_select(t, 0, indices) for t in x],
        ):
            result = process_pooled_embeddings([emb1, emb2], indices)
            assert torch.equal(result, expected)


class TestCheckNameFormat:
    @staticmethod
    def test_valid_feature_names():
        """测试合法特征名"""
        # 这些名称应该不会抛出异常
        _check_name_format("feature_1")
        _check_name_format("FEATURE2")
        _check_name_format("_private_feat")
        _check_name_format("a" * 100)  # 长字符串测试
        _check_name_format(".feat")

    @staticmethod
    def test_invalid_feature_names():
        """测试非法字符"""
        # 这些名称应该抛出 ValueError
        with pytest.raises(ValueError):
            _check_name_format("feature@")  # 特殊字符
        with pytest.raises(ValueError):
            _check_name_format("space in")  # 空格
        with pytest.raises(ValueError):
            _check_name_format("dash-ed")  # 连字符

    @staticmethod
    def test_edge_cases():
        """边界条件测试"""
        # 空字符串应该抛出异常
        with pytest.raises(ValueError):
            _check_name_format("")
        # 这些名称应该不会抛出异常
        _check_name_format("_")  # 单下划线
        _check_name_format("1")  # 纯数字


class TestEmbeddingConfigValid:
    @staticmethod
    def test_valid_config():
        """测试完全合法的配置"""
        config = HashEmbeddingBagConfig(
            embedding_dim=16,
            num_embeddings=100,
            data_type=DataType.FP32,
            feature_names=["valid_feat1", "valid_feat2"],
            pooling=PoolingType.SUM,
        )
        # 不应该抛出异常
        check_embedding_config_valid(config)

    @staticmethod
    def test_embedding_dim_alignment():
        """测试embedding_dim对齐检查"""
        with pytest.raises(ValueError, match="multiple of 8"):
            config = HashEmbeddingBagConfig(
                embedding_dim=100,  # 不是8的倍数
                num_embeddings=100,
            )
            check_embedding_config_valid(config)

    @staticmethod
    def test_embedding_dim_range():
        """测试embedding_dim范围检查"""
        # 测试下边界
        with pytest.raises(ValueError, match="should be in"):
            config = HashEmbeddingBagConfig(
                embedding_dim=0,  # 小于8
                num_embeddings=100,
            )
            check_embedding_config_valid(config)
        # 测试上边界
        with pytest.raises(ValueError, match="should be in"):
            config = HashEmbeddingBagConfig(embedding_dim=MAX_EMBEDDINGS_DIM + 8, num_embeddings=100)
            check_embedding_config_valid(config)

    @staticmethod
    def test_num_embeddings_range():
        """测试num_embeddings范围检查"""
        # 测试下边界
        with pytest.raises(ValueError, match="should be in"):
            config = HashEmbeddingBagConfig(
                embedding_dim=8,
                num_embeddings=0.5,  # 小于1
            )
            check_embedding_config_valid(config)
        # 测试上边界
        with pytest.raises(ValueError, match="should be in"):
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
                feature_names=[],  # should not be empty
            )
            check_embedding_config_valid(config)

        # 测试非法特征名
        with pytest.raises(ValueError, match="should only contain alphanumeric characters"):
            config = HashEmbeddingBagConfig(
                embedding_dim=8, num_embeddings=100, feature_names=["valid_feat", "invalid@feat"]
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
                weight_init_min=1.0,  # should be None
            )
            check_embedding_config_valid(config)

    @staticmethod
    def test_num_embeddings_post_pruning_validation():
        with pytest.raises(ValueError, match="should be None"):
            config = HashEmbeddingBagConfig(
                embedding_dim=8,
                num_embeddings=100,
                feature_names=["feat"],
                num_embeddings_post_pruning="Not None",  # should be None
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
                pooling="INVALID",  # 不是枚举值
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
                init_fn="not_callable",  # 不是可调用对象
            )
            check_embedding_config_valid(config)

    @staticmethod
    def test_need_pos_validation():
        """测试need_pos参数检查"""
        with pytest.raises(ValueError, match="only support False"):
            config = HashEmbeddingBagConfig(
                embedding_dim=8,
                num_embeddings=100,
                feature_names=["feat"],
                need_pos=True,  # 不支持False参数场景
            )
            check_embedding_config_valid(config)


class TestHashEmbeddingBagCollection:
    # 测试用例
    @pytest.mark.parametrize("embedding_dims", [[32, 64], [128, 256]])
    @pytest.mark.parametrize("num_embeddings", [[1024, 512]])
    @pytest.mark.parametrize("pooling_type", [PoolingType.SUM, PoolingType.MEAN])
    def test_hash_embedding_bag_collection(self, embedding_dims, num_embeddings, pooling_type):
        # 创建测试配置
        config1 = HashEmbeddingBagConfig(
            name="table1",
            embedding_dim=embedding_dims[0],
            num_embeddings=num_embeddings[0],
            feature_names=["feature1"],
            data_type=DataType.FP32,
            pooling=pooling_type,
        )
        config2 = HashEmbeddingBagConfig(
            name="table2",
            embedding_dim=embedding_dims[1],
            num_embeddings=num_embeddings[1],
            feature_names=["feature2"],
            data_type=DataType.FP32,
            pooling=pooling_type,
        )
        config3 = HashEmbeddingBagConfig(
            name="table1",
            embedding_dim=embedding_dims[1],
            num_embeddings=num_embeddings[1],
            feature_names=["feature2"],
            data_type=DataType.FP32,
            pooling=pooling_type,
        )

        # 初始化模型
        model = HashEmbeddingBagCollection(tables=[config1, config2], is_weighted=False, device="cpu")

        # 创建测试输入
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature1", "feature2"],
            values=torch.tensor([1, 2, 3, 4, 5, 6]),
            lengths=torch.tensor([2, 0, 1, 1, 2, 0]),
        )

        # 前向传播
        output = model(features)

        # 验证输出
        assert isinstance(output, KeyedTensor)
        assert output.keys() == ["feature1", "feature2"]
        assert output.values().shape == (3, sum(embedding_dims))  # 64+32=96维

        # 表名相同
        with pytest.raises(ValueError, match="Duplicate table name"):
            _ = HashEmbeddingBagCollection(tables=[config1, config3], is_weighted=False, device="cpu")

    @staticmethod
    def test_invalid_emb_config_params():
        """测试HashEmbeddingBagCollection参数检查"""

        def _create_table_configs(table_num: int):
            return [
                HashEmbeddingBagConfig(
                    name=f"table{i}",
                    embedding_dim=8,
                    num_embeddings=400,
                    feature_names=["feature1"],
                    data_type=DataType.FP32,
                    pooling=PoolingType.SUM,
                )
                for i in range(table_num)
            ]

        with pytest.raises(ValueError, match=f"{MAX_NUM_TABLES}"):
            _ = HashEmbeddingBagCollection(
                tables=[],  # tables列表长度为0
                is_weighted=False,
                device="cpu",
            )
        with pytest.raises(ValueError, match=f"{MAX_NUM_TABLES}"):
            invalid_config_num = MAX_NUM_TABLES + 1
            # tables列表长度超过上限
            _ = HashEmbeddingBagCollection(
                tables=_create_table_configs(invalid_config_num), is_weighted=False, device="cpu"
            )
        with pytest.raises(ValueError, match="must be False"):
            _ = HashEmbeddingBagCollection(
                tables=_create_table_configs(1),
                is_weighted=True,  # 不支持True
                device="cpu",
            )
        with pytest.raises(ValueError, match="must be a list"):
            _ = HashEmbeddingBagCollection(
                tables="param is not list object",  # 参数类型错误
                is_weighted=False,
                device="cpu",
            )
        with pytest.raises(ValueError, match="HashEmbeddingBagConfig"):
            tables = _create_table_configs(1)
            tables.append("str")
            _ = HashEmbeddingBagCollection(
                tables=tables,  # 列表中不支持的元素类型
                is_weighted=False,
                device="cpu",
            )
        with pytest.raises(ValueError, match="device type or value is invalid"):
            _ = HashEmbeddingBagCollection(
                tables=_create_table_configs(1),
                is_weighted=False,
                device="cpu2",  # 不支持的值
            )
        with pytest.raises(ValueError, match="device type or value is invalid"):
            _ = HashEmbeddingBagCollection(
                tables=_create_table_configs(1),
                is_weighted=False,
                device=0,  # 不支持的参数类型
            )

    @staticmethod
    def test_reset_parameters():
        # 创建测试配置
        config = HashEmbeddingBagConfig(
            name="test_table",
            embedding_dim=16,
            num_embeddings=100,
            feature_names=["test_feature"],
            data_type=DataType.FP32,
        )
        # 初始化模型
        model = HashEmbeddingBagCollection(tables=[config], device="cpu")
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
        config = HashEmbeddingBagConfig(name="test_table", embedding_dim=16, num_embeddings=100)
        device = "cpu"
        model = HashEmbeddingBag(config=config, device=device)
        assert isinstance(model, torch.nn.Module)

    @staticmethod
    def test_find_and_insert():
        config = HashEmbeddingBagConfig(name="test_table", embedding_dim=8, num_embeddings=50)
        device = torch.device("cpu")
        model = HashEmbeddingBag(config=config, device=device)

        keys = torch.tensor([1, 2, 3])
        values = torch.randn(3, 8)
        scores = torch.tensor([0.9, 0.8, 0.7])
        founds = torch.tensor([0, 0, 0], dtype=torch.bool)

        assert model.find_and_insert(keys, values, scores, founds) == NotImplemented

    @staticmethod
    def test_forward():
        config = HashEmbeddingBagConfig(name="test_table", embedding_dim=4, num_embeddings=10)
        device = torch.device("cpu")
        model = HashEmbeddingBag(config=config, device=device)
        input_tensor = torch.tensor([1, 2, 3, 4])
        offsets = torch.tensor([0, 2, 4])

        assert model.forward(input_tensor, offsets) == NotImplemented


class _FakeAdapterSelf:
    """模拟适配器 self，提供 check_config_new_item_v120 方法供 v150 委托调用。"""

    def check_config_new_item_v120(self, config) -> None:
        check_config_new_item_v120(self, config)


class TestCheckConfigNewItemV120:
    """DT测试: v1.2.0 新增配置项检查函数 check_config_new_item_v120。"""

    @staticmethod
    def test_no_input_dim_attribute():
        """config 不含 input_dim 属性时不抛异常。"""
        config = SimpleNamespace()
        check_config_new_item_v120(None, config)

    @staticmethod
    def test_input_dim_is_none():
        """config.input_dim 为 None 时不抛异常。"""
        config = SimpleNamespace(input_dim=None)
        check_config_new_item_v120(None, config)

    @staticmethod
    def test_input_dim_not_none_raises():
        """config.input_dim 非 None 时抛 ValueError。"""
        config = SimpleNamespace(input_dim=64)
        with pytest.raises(ValueError, match="input_dim"):
            check_config_new_item_v120(None, config)

    @staticmethod
    def test_input_dim_zero_raises():
        """config.input_dim 为 0 时仍抛异常（0 is not None）。"""
        config = SimpleNamespace(input_dim=0)
        with pytest.raises(ValueError, match="input_dim"):
            check_config_new_item_v120(None, config)

    @staticmethod
    def test_error_message_contains_actual_value():
        """错误消息中包含实际的 input_dim 值。"""
        config = SimpleNamespace(input_dim=128)
        with pytest.raises(ValueError, match="128"):
            check_config_new_item_v120(None, config)

    @staticmethod
    def test_other_attributes_ignored():
        """input_dim 合法时，其它属性不影响校验。"""
        config = SimpleNamespace(
            input_dim=None,
            total_num_buckets=100,
            use_virtual_table=True,
        )
        check_config_new_item_v120(None, config)


class TestCheckConfigNewItemV150:
    """DT测试: v1.5.0 新增配置项检查函数 check_config_new_item_v150。"""

    @staticmethod
    def _make_self():
        """构造带 check_config_new_item_v120 方法的 mock self。"""
        return _FakeAdapterSelf()

    @staticmethod
    def _make_valid_config():
        """构造所有 v1.5.0 新增字段均为默认值的合法 config。"""
        return SimpleNamespace(
            input_dim=None,
            total_num_buckets=None,
            use_virtual_table=False,
            virtual_table_eviction_policy=None,
            enable_embedding_update=False,
        )

    @staticmethod
    def test_all_valid_config():
        """所有 v1.5.0 新增字段均为默认值时不抛异常。"""
        check_config_new_item_v150(
            TestCheckConfigNewItemV150._make_self(), TestCheckConfigNewItemV150._make_valid_config()
        )

    @staticmethod
    def test_no_new_attributes():
        """config 不含任何 v1.5.0 新增属性时不抛异常。"""
        config = SimpleNamespace()
        check_config_new_item_v150(TestCheckConfigNewItemV150._make_self(), config)

    @staticmethod
    def test_total_num_buckets_not_none_raises():
        """total_num_buckets 非 None 时抛 ValueError。"""
        config = TestCheckConfigNewItemV150._make_valid_config()
        config.total_num_buckets = 100
        with pytest.raises(ValueError, match="total_num_buckets"):
            check_config_new_item_v150(TestCheckConfigNewItemV150._make_self(), config)

    @staticmethod
    def test_use_virtual_table_true_raises():
        """use_virtual_table 为 True 时抛 ValueError。"""
        config = TestCheckConfigNewItemV150._make_valid_config()
        config.use_virtual_table = True
        with pytest.raises(ValueError, match="use_virtual_table"):
            check_config_new_item_v150(TestCheckConfigNewItemV150._make_self(), config)

    @staticmethod
    def test_use_virtual_table_false_ok():
        """use_virtual_table 为 False 时不抛异常。"""
        config = TestCheckConfigNewItemV150._make_valid_config()
        config.use_virtual_table = False
        check_config_new_item_v150(TestCheckConfigNewItemV150._make_self(), config)

    @staticmethod
    def test_virtual_table_eviction_policy_not_none_raises():
        """virtual_table_eviction_policy 非 None 时抛 ValueError。"""
        config = TestCheckConfigNewItemV150._make_valid_config()
        config.virtual_table_eviction_policy = "LRU"
        with pytest.raises(ValueError, match="virtual_table_eviction_policy"):
            check_config_new_item_v150(TestCheckConfigNewItemV150._make_self(), config)

    @staticmethod
    def test_enable_embedding_update_true_raises():
        """enable_embedding_update 为 True 时抛 ValueError。"""
        config = TestCheckConfigNewItemV150._make_valid_config()
        config.enable_embedding_update = True
        with pytest.raises(ValueError, match="enable_embedding_update"):
            check_config_new_item_v150(TestCheckConfigNewItemV150._make_self(), config)

    @staticmethod
    def test_enable_embedding_update_false_ok():
        """enable_embedding_update 为 False 时不抛异常。"""
        config = TestCheckConfigNewItemV150._make_valid_config()
        config.enable_embedding_update = False
        check_config_new_item_v150(TestCheckConfigNewItemV150._make_self(), config)

    @staticmethod
    def test_delegates_to_v120_input_dim_check():
        """v150 应委托 v120 检查 input_dim 字段。"""
        config = TestCheckConfigNewItemV150._make_valid_config()
        config.input_dim = 64
        with pytest.raises(ValueError, match="input_dim"):
            check_config_new_item_v150(TestCheckConfigNewItemV150._make_self(), config)

    @staticmethod
    def test_v120_check_runs_before_v150_checks():
        """v120 检查应先于 v150 检查执行（input_dim 优先报错）。"""
        config = TestCheckConfigNewItemV150._make_valid_config()
        config.input_dim = 64
        config.total_num_buckets = 100
        with pytest.raises(ValueError, match="input_dim"):
            check_config_new_item_v150(TestCheckConfigNewItemV150._make_self(), config)


class TestCheckConfigNewItemAdapter:
    """DT测试: 适配器层 check_embedding_config_new_item 的版本差异注册与调用。"""

    @staticmethod
    def test_baseline_110_no_op():
        """1.1.0 基类默认实现不做任何检查。"""
        methods = _build_methods((1, 1, 0))
        assert "check_embedding_config_new_item" not in methods

        class Adapter110(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 1, 0)

        a = Adapter110()
        a.check_embedding_config_new_item(SimpleNamespace(input_dim=64))

    @staticmethod
    def test_version_120_registers_v120():
        """1.2.0 差异表注册 check_config_new_item_v120。"""
        methods = _build_methods((1, 2, 0))
        assert "check_embedding_config_new_item" in methods
        assert methods["check_embedding_config_new_item"] is check_config_new_item_v120

    @staticmethod
    def test_version_120_via_adapter_raises():
        """1.2.0 适配器对 input_dim 非 None 抛 ValueError。"""
        methods = _build_methods((1, 2, 0))

        class Adapter120(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 2, 0)

        a = type("Adapter120_cfg", (Adapter120,), methods)()
        with pytest.raises(ValueError, match="input_dim"):
            a.check_embedding_config_new_item(SimpleNamespace(input_dim=64))

    @staticmethod
    def test_version_120_via_adapter_passes():
        """1.2.0 适配器对 input_dim=None 不抛异常。"""
        methods = _build_methods((1, 2, 0))

        class Adapter120(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 2, 0)

        a = type("Adapter120_cfg_ok", (Adapter120,), methods)()
        a.check_embedding_config_new_item(SimpleNamespace(input_dim=None))

    @staticmethod
    def test_version_150_registers_v150():
        """1.5.0 差异表注册 check_config_new_item_v150（覆盖 v120）。"""
        methods = _build_methods((1, 5, 0))
        assert "check_embedding_config_new_item" in methods
        assert methods["check_embedding_config_new_item"] is check_config_new_item_v150

    @staticmethod
    def test_version_150_registers_v120_helper():
        """1.5.0 差异表同时注册 check_config_new_item_v120，供 v150 委托调用。"""
        methods = _build_methods((1, 5, 0))
        assert "check_config_new_item_v120" in methods
        assert methods["check_config_new_item_v120"] is check_config_new_item_v120

    @staticmethod
    def test_version_150_via_adapter_raises_for_total_num_buckets():
        """1.5.0 适配器对 total_num_buckets 非 None 抛 ValueError。

        v150 内部通过 self.check_config_new_item_v120 委托 v120 检查，
        该方法已由 1.5.0 差异表注册，适配器无需额外提供。
        """
        methods = _build_methods((1, 5, 0))

        class Adapter150(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 5, 0)

        a = type("Adapter150_cfg", (Adapter150,), methods)()
        config = SimpleNamespace(
            input_dim=None,
            total_num_buckets=100,
            use_virtual_table=False,
            virtual_table_eviction_policy=None,
            enable_embedding_update=False,
        )
        with pytest.raises(ValueError, match="total_num_buckets"):
            a.check_embedding_config_new_item(config)

    @staticmethod
    def test_version_150_via_adapter_raises_for_use_virtual_table():
        """1.5.0 适配器对 use_virtual_table=True 抛 ValueError。"""
        methods = _build_methods((1, 5, 0))

        class Adapter150(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 5, 0)

        a = type("Adapter150_cfg_vt", (Adapter150,), methods)()
        config = SimpleNamespace(
            input_dim=None,
            total_num_buckets=None,
            use_virtual_table=True,
            virtual_table_eviction_policy=None,
            enable_embedding_update=False,
        )
        with pytest.raises(ValueError, match="use_virtual_table"):
            a.check_embedding_config_new_item(config)

    @staticmethod
    def test_version_150_via_adapter_all_valid():
        """1.5.0 适配器对所有字段均为默认值时不抛异常。"""
        methods = _build_methods((1, 5, 0))

        class Adapter150(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 5, 0)

        a = type("Adapter150_cfg_ok", (Adapter150,), methods)()
        config = SimpleNamespace(
            input_dim=None,
            total_num_buckets=None,
            use_virtual_table=False,
            virtual_table_eviction_policy=None,
            enable_embedding_update=False,
        )
        a.check_embedding_config_new_item(config)

    @staticmethod
    def test_higher_version_inherits_v150():
        """1.6.0 应继承 1.5.0 的 check_config_new_item_v150 及其委托的 v120 助手。"""
        methods = _build_methods((1, 6, 0))
        assert "check_embedding_config_new_item" in methods
        assert methods["check_embedding_config_new_item"] is check_config_new_item_v150
        assert "check_config_new_item_v120" in methods
        assert methods["check_config_new_item_v120"] is check_config_new_item_v120
