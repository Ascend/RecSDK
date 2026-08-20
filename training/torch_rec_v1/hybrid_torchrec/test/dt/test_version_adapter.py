#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pylint: disable=duplicate-code
"""适配器层单元测试：直接 import 即可运行。

测试覆盖：
1. 版本检测：_version 模块（含解析边界、优先级、fallback）
2. 差异表：_version_diff 模块
3. 适配器工厂：根据 _version_diff 自动选择正确实现
4. 兼容性辅助：_adapter_base 的兼容性工具方法
5. 版本标志与端到端适配器行为
"""

import pytest
import unittest
from dataclasses import dataclass
from importlib import metadata
from unittest.mock import patch
from types import SimpleNamespace

import torchrec

from hybrid_torchrec._adapters import _build_methods, adapter
from hybrid_torchrec._adapters._adapter_base import TorchRecVersionAdapter
from hybrid_torchrec._adapters._version import _torchrec_version_tuple
from hybrid_torchrec._adapters._version_diff import (
    check_config_new_item_v120,
    check_config_new_item_v150,
)


@dataclass
class _FakeCommonArgs:
    learning_rate: float = 0.5


@dataclass
class _FakeOptimizerArgs:
    learning_rate: float = 0.1


@dataclass
class _FakeKernel:
    """模拟 torchrec 的 SplitTableBatchedEmbeddingBagsCodegen。"""

    _optimizer_args: _FakeOptimizerArgs = None
    _method_lr: float = None

    def get_learning_rate(self) -> float:
        if self._method_lr is None:
            raise AttributeError("1.1.0 kernel has no get_learning_rate()")
        return self._method_lr


class TestVersionDetection(unittest.TestCase):
    def test_torchrec_version_tuple_shape(self):
        v = _torchrec_version_tuple()
        self.assertIsInstance(v, tuple)
        self.assertEqual(len(v), 3)
        for part in v:
            self.assertIsInstance(part, int)

    def test_current_adapter_version_matches_detection(self):
        self.assertEqual(adapter.version, _torchrec_version_tuple())

    def test_version_tuple_boundary_cases(self):
        """测试版本解析的边界情况。"""
        test_cases = [
            ("1.2.0", (1, 2, 0)),
            ("1.5.0+cpu", (1, 5, 0)),
            ("1.10.0", (1, 10, 0)),
            ("2.0", (2, 0, 0)),
            ("1", (1, 0, 0)),
            ("1.2.3.4", (1, 2, 3)),
            ("1.x.0", (1, 0, 0)),
        ]
        for version_str, expected in test_cases:
            with patch("hybrid_torchrec._adapters._version._torchrec_version", return_value=version_str):
                result = _torchrec_version_tuple()
                self.assertEqual(result, expected, f"Failed for {version_str}")

    def test_version_priority_order(self):
        """验证版本获取的优先级顺序。"""
        from hybrid_torchrec._adapters._version import _torchrec_version

        # 优先级: __version__ > version module > version.txt > metadata
        with patch.object(torchrec, '__version__', '1.5.0'):
            self.assertEqual(_torchrec_version(), '1.5.0')

        # 测试 fallback 到 metadata（当 __version__ 不存在时）
        def mock_import(name, *args, **kwargs):
            if name == 'torchrec.version':
                raise ImportError()
            return __import__(name, *args, **kwargs)

        with patch.object(torchrec, '__version__', None):
            with patch('hybrid_torchrec._adapters._version.metadata.version', return_value='1.2.0'):
                with patch('builtins.__import__', side_effect=mock_import):
                    with patch.object(torchrec, '__path__', []):
                        self.assertEqual(_torchrec_version(), '1.2.0')

    def test_version_txt_fallback(self):
        """测试通过 version.txt 文件获取版本（优先级第3级）。"""
        import os
        import tempfile
        from pathlib import Path

        from hybrid_torchrec._adapters._version import _torchrec_version

        # 创建临时 version.txt
        tmpdir = tempfile.mkdtemp()
        version_txt_path = Path(tmpdir) / "version.txt"
        version_txt_path.write_text("1.3.0+cpu\n")

        def mock_import(name, *args, **kwargs):
            if name == 'torchrec.version':
                raise ImportError()
            return __import__(name, *args, **kwargs)

        with patch.object(torchrec, '__version__', None):
            with patch.object(torchrec, '__path__', [tmpdir]):
                with patch('builtins.__import__', side_effect=mock_import):
                    result = _torchrec_version()
                    self.assertEqual(result, "1.3.0+cpu")

        # 清理
        os.remove(version_txt_path)
        os.rmdir(tmpdir)

    def test_version_fallback_to_zero(self):
        """测试所有来源都失败时抛出 RuntimeError。"""
        from hybrid_torchrec._adapters._version import _torchrec_version

        def mock_import(name, *args, **kwargs):
            if name == 'torchrec.version':
                raise ImportError()
            return __import__(name, *args, **kwargs)

        with patch.object(torchrec, '__version__', None):
            with patch('hybrid_torchrec._adapters._version.metadata.version') as mock_metadata:
                mock_metadata.side_effect = metadata.PackageNotFoundError()
                with patch('builtins.__import__', side_effect=mock_import):
                    with patch.object(torchrec, '__path__', []):
                        with self.assertRaises(RuntimeError) as context:
                            _torchrec_version()
                        self.assertIn("Can not get torchrec version", str(context.exception))


class TestVersionDiffTable(unittest.TestCase):
    """测试 _version_diff 模块的差异表功能。"""

    def test_version_diffs_format(self):
        """验证 VERSION_DIFFS 格式正确。"""
        from hybrid_torchrec._adapters._version_diff import VERSION_DIFFS

        for ver, overrides in VERSION_DIFFS:
            self.assertIsInstance(ver, tuple)
            self.assertEqual(len(ver), 3)
            self.assertIsInstance(overrides, dict)
            for key, func in overrides.items():
                self.assertIsInstance(key, str)
                self.assertTrue(callable(func), f"{key} is not callable")

    def test_version_diffs_sorted(self):
        """验证 VERSION_DIFFS 按版本号升序排列。"""
        from hybrid_torchrec._adapters._version_diff import VERSION_DIFFS

        versions = [ver for ver, _ in VERSION_DIFFS]
        for i in range(len(versions) - 1):
            self.assertLessEqual(versions[i], versions[i + 1], "VERSION_DIFFS should be sorted by version")


class TestBuildMethods(unittest.TestCase):
    """直接测试 _build_methods 的纯函数逻辑。"""

    def test_baseline_110(self):
        """1.1.0 版本没有任何覆盖，返回空字典，使用基类默认实现"""
        methods = _build_methods((1, 1, 0))
        # 1.1.0 没有版本差异，所以 methods 为空
        self.assertEqual(methods, {})

        # 通过基类验证默认行为
        class TestAdapter(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 1, 0)

        adapter_110 = TestAdapter()
        self.assertEqual(
            adapter_110.get_learning_rate(_FakeCommonArgs(), _FakeOptimizerArgs()),
            0.1,  # 来自 optimizer_args
        )

    def test_version_120(self):
        """1.2.0 版本有覆盖"""
        methods = _build_methods((1, 2, 0))
        # 1.2.0 应该有 get_learning_rate 的覆盖
        self.assertIn("get_learning_rate", methods)

        class TestAdapter(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 2, 0)

        # 创建带有覆盖的适配器实例
        adapter_120 = type("TestAdapter120", (TestAdapter,), methods)()
        self.assertEqual(
            adapter_120.get_learning_rate(_FakeCommonArgs(), _FakeOptimizerArgs()),
            0.5,  # 来自 common_args
        )

    def test_version_150(self):
        """1.5.0 版本有更多覆盖"""
        methods = _build_methods((1, 5, 0))
        # 1.5.0 应该包含 1.2.0 和 1.5.0 的所有覆盖
        self.assertIn("get_learning_rate", methods)
        self.assertIn("build_args_kwargs", methods)

        class TestAdapter(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 5, 0)

        adapter_150 = type("TestAdapter150", (TestAdapter,), methods)()
        self.assertEqual(
            adapter_150.get_learning_rate(_FakeCommonArgs(), _FakeOptimizerArgs()),
            0.5,  # 继承自 1.2.0
        )

    def test_higher_version_keeps_latest(self):
        """1.6.0 应继承 1.5.0 的所有行为。"""
        methods = _build_methods((1, 6, 0))
        # 1.6.0 应该继承所有之前的覆盖
        self.assertIn("get_learning_rate", methods)

        class TestAdapter(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 6, 0)

        adapter_160 = type("TestAdapter160", (TestAdapter,), methods)()
        self.assertEqual(
            adapter_160.get_learning_rate(_FakeCommonArgs(), _FakeOptimizerArgs()),
            0.5,  # 继承自 1.2.0
        )


class TestAdapterIsCallable(unittest.TestCase):
    """验证 adapter 暴露了所有声明的版本方法。"""

    def test_required_methods_exist(self):
        for name in (
            "version",
            "get_learning_rate",
            "create_sharding_infos",
            "build_args_kwargs",
            "get_kernel_learning_rate",
        ):
            self.assertTrue(hasattr(adapter, name), f"adapter missing method: {name}")

    def test_adapter_has_all_methods(self):
        """验证适配器暴露所有声明的方法。"""
        required_methods = [
            'version',
            'get_learning_rate',
            'create_sharding_infos',
            'build_args_kwargs',
            'get_output_dtensor',
            'get_kernel_learning_rate',
            'get_virtual_table_feature_num_buckets',
            'make_embedding_table_config',
            'make_awaitable',
            'make_kjt_list_splits_awaitable',
            'filter_rw_sparse_features_dist_kwargs',
            'embedding_compute_kernel_values',
        ]
        for method_name in required_methods:
            self.assertTrue(hasattr(adapter, method_name), f"adapter missing method: {method_name}")

    def test_adapter_singleton(self):
        """验证适配器是单例。"""
        from hybrid_torchrec._adapters import _create_adapter

        adapter1 = adapter
        adapter2 = _create_adapter()
        self.assertIs(adapter1, adapter2)
        self.assertEqual(adapter1.version, adapter2.version)

    def test_get_kernel_learning_rate_per_version(self):
        """验证 get_kernel_learning_rate 在 1.1.0/1.2.0+ 上的差异已被差异表正确接管。"""
        kernel_110 = _FakeKernel(_optimizer_args=_FakeOptimizerArgs(learning_rate=0.1))
        kernel_120 = _FakeKernel(_method_lr=0.5)

        # 1.1.0 使用基类默认实现
        methods_110 = _build_methods((1, 1, 0))
        self.assertEqual(methods_110, {})  # 没有覆盖

        class Adapter110(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 1, 0)

        adapter_110 = Adapter110()
        self.assertEqual(adapter_110.get_kernel_learning_rate(kernel_110), 0.0)

        # 1.2.0 有覆盖
        methods_120 = _build_methods((1, 2, 0))
        self.assertIn("get_kernel_learning_rate", methods_120)

        class Adapter120(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 2, 0)

        adapter_120 = type("Adapter120", (Adapter120,), methods_120)()
        self.assertEqual(adapter_120.get_kernel_learning_rate(kernel_120), 0.5)

        # 1.5.0 继承 1.2.0 的覆盖
        methods_150 = _build_methods((1, 5, 0))
        self.assertIn("get_kernel_learning_rate", methods_150)

        class Adapter150(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 5, 0)

        adapter_150 = type("Adapter150", (Adapter150,), methods_150)()
        self.assertEqual(adapter_150.get_kernel_learning_rate(kernel_120), 0.5)


class TestCompatMakeEmbeddingTableConfig(unittest.TestCase):
    def test_ignores_unknown_fields(self):
        from torchrec.modules.embedding_configs import EmbeddingTableConfig

        cfg = adapter.make_embedding_table_config(
            num_embeddings=100,
            embedding_dim=8,
            name="t",
            # 1.6.0 假设字段，当前版本不存在，应被自动过滤
            some_future_field_xyz=42,
        )
        self.assertIsInstance(cfg, EmbeddingTableConfig)
        self.assertEqual(cfg.num_embeddings, 100)


class TestCompatMakeAwaitable(unittest.TestCase):
    def test_filters_unknown_kwargs(self):
        from torchrec.distributed.embedding_sharding import KJTListSplitsAwaitable

        obj = adapter.make_awaitable(
            KJTListSplitsAwaitable,
            awaitables=[],
            ctx=None,
            module_fqn="x",
            sharding_types=[],
            future_only_kwarg=True,
        )
        self.assertIsInstance(obj, KJTListSplitsAwaitable)


class TestCompatFilterRwSparseFeaturesDist(unittest.TestCase):
    def test_filters_unknown_kwargs(self):
        result = adapter.filter_rw_sparse_features_dist_kwargs(
            pg=None,
            num_features=1,
            feature_hash_sizes=[1],
            device=None,
            is_sequence=False,
            has_feature_processor=False,
            need_pos=False,
            future_field_abc=1,
        )
        self.assertIn("pg", result)
        self.assertIn("num_features", result)
        self.assertNotIn("future_field_abc", result)


class TestGetOutputDtensor(unittest.TestCase):
    """测试 get_output_dtensor 静态方法。"""

    def test_from_env_true(self):
        env = type("FakeEnv", (), {"output_dtensor": True})()
        self.assertTrue(adapter.get_output_dtensor(env, None))

    def test_from_env_false(self):
        env = type("FakeEnv", (), {"output_dtensor": False})()
        self.assertFalse(adapter.get_output_dtensor(env, None))

    def test_fallback_to_fused_params(self):
        env = object()
        self.assertTrue(adapter.get_output_dtensor(env, {"output_dtensor": True}))
        self.assertFalse(adapter.get_output_dtensor(env, {"output_dtensor": False}))

    def test_default_false(self):
        env = object()
        self.assertFalse(adapter.get_output_dtensor(env, None))
        self.assertFalse(adapter.get_output_dtensor(env, {}))
        self.assertFalse(adapter.get_output_dtensor(env, {"other": 1}))


class TestGetVirtualTableFeatureNumBuckets(unittest.TestCase):
    """测试 get_virtual_table_feature_num_buckets 版本行为差异。"""

    def test_baseline_110_returns_default(self):
        """1.1.0/1.2.0 返回 (None, False)。"""
        methods = _build_methods((1, 1, 0))
        self.assertNotIn("get_virtual_table_feature_num_buckets", methods)

        class Adapter110(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 1, 0)

        a = Adapter110()
        result = a.get_virtual_table_feature_num_buckets(object())
        self.assertEqual(result, (None, False))

    def test_version_150_calls_instance_method(self):
        """1.5.0 调用实例的 _get_virtual_table_feature_num_buckets()。"""
        methods = _build_methods((1, 5, 0))
        self.assertIn("get_virtual_table_feature_num_buckets", methods)

        class FakeInstance:
            def _get_virtual_table_feature_num_buckets(self):
                return ([100, 200], False)

        class Adapter150(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 5, 0)

        a = type("Adapter150_vt", (Adapter150,), methods)()
        result = a.get_virtual_table_feature_num_buckets(FakeInstance())
        self.assertEqual(result, ([100, 200], False))

    def test_higher_version_inherits_150(self):
        """更高版本应继承 1.5.0 的行为。"""
        methods = _build_methods((1, 6, 0))
        self.assertIn("get_virtual_table_feature_num_buckets", methods)

        class FakeInstance:
            def _get_virtual_table_feature_num_buckets(self):
                return ([5], True)

        class Adapter160(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 6, 0)

        a = type("Adapter160_vt", (Adapter160,), methods)()
        result = a.get_virtual_table_feature_num_buckets(FakeInstance())
        self.assertEqual(result, ([5], True))


class TestBuildArgsKwargs(unittest.TestCase):
    """测试 build_args_kwargs 的版本行为差异。"""

    def test_version_150_uses_instance_method(self):
        """1.5.0 使用 forward_args.build_args_kwargs 实例方法。"""
        methods = _build_methods((1, 5, 0))
        self.assertIn("build_args_kwargs", methods)

        class FakeForwardArgs:
            def build_args_kwargs(self, batch):
                return (batch, {"result": "ok"})

        class Adapter150(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 5, 0)

        a = type("Adapter150_bak", (Adapter150,), methods)()
        result = a.build_args_kwargs("data", FakeForwardArgs())
        self.assertEqual(result, ("data", {"result": "ok"}))

    def test_version_150_raises_when_no_method(self):
        """1.5.0 下 forward_args 缺少 build_args_kwargs 时抛 RuntimeError。"""
        methods = _build_methods((1, 5, 0))

        class Adapter150(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 5, 0)

        a = type("Adapter150_err", (Adapter150,), methods)()
        with self.assertRaises(RuntimeError) as ctx:
            a.build_args_kwargs("data", object())
        self.assertIn("build_args_kwargs not available", str(ctx.exception))

    def test_version_120_uses_module_function(self):
        """1.2.0 使用模块级 _build_args_kwargs 函数。"""
        methods = _build_methods((1, 2, 0))
        # 1.2.0 没有 build_args_kwargs 覆盖
        self.assertNotIn("build_args_kwargs", methods)

        class Adapter120(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 2, 0)

        a = Adapter120()
        # 基类默认实现调用 torchrec 模块级 _build_args_kwargs
        self.assertTrue(callable(a.build_args_kwargs))


class TestCreateShardingInfos(unittest.TestCase):
    """测试 create_sharding_infos 的版本行为差异。"""

    def test_version_120_uses_instance_method(self):
        """1.2.0 使用 instance.create_grouped_sharding_infos。"""
        methods = _build_methods((1, 2, 0))
        self.assertIn("create_sharding_infos", methods)

        class FakeInstance:
            def create_grouped_sharding_infos(self, mod, shard, prefix, fp):
                return {"method": "instance", "prefix": prefix}

        class Adapter120(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 2, 0)

        a = type("Adapter120_si", (Adapter120,), methods)()
        result = a.create_sharding_infos(FakeInstance(), "module", "shard", "myprefix", {})
        self.assertEqual(result["method"], "instance")
        self.assertEqual(result["prefix"], "myprefix")

    def test_version_110_uses_module_function(self):
        """1.1.0 使用模块级 create_sharding_infos_by_sharding。"""
        methods = _build_methods((1, 1, 0))
        self.assertNotIn("create_sharding_infos", methods)

        class Adapter110(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 1, 0)

        a = Adapter110()
        self.assertTrue(callable(a.create_sharding_infos))

    def test_version_150_inherits_120(self):
        """1.5.0 应继承 1.2.0 的 create_sharding_infos 行为。"""
        methods = _build_methods((1, 5, 0))
        self.assertIn("create_sharding_infos", methods)

        fake_instance = type(
            "FakeInstance", (), {"create_grouped_sharding_infos": lambda self, mod, shard, prefix, fp: {"ver": 1.5}}
        )()

        class Adapter150(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 5, 0)

        a = type("Adapter150_si", (Adapter150,), methods)()
        result = a.create_sharding_infos(fake_instance, "module", "sharding", "p", None)
        self.assertEqual(result, {"ver": 1.5})


class TestEmbeddingComputeKernelValues(unittest.TestCase):
    """测试 embedding_compute_kernel_values 安全枚举获取。"""

    def test_known_enum(self):
        from torchrec.distributed.embedding_types import EmbeddingComputeKernel

        result = adapter.embedding_compute_kernel_values("DENSE")
        self.assertIn(EmbeddingComputeKernel.DENSE.value, result)

    def test_unknown_enum_silently_ignored(self):
        """不存在的枚举值应被静默忽略，不应抛异常。"""
        result = adapter.embedding_compute_kernel_values("DENSE", "NON_EXISTENT_ENUM_VALUE")
        self.assertIsInstance(result, set)
        # DENSE 仍应包含
        from torchrec.distributed.embedding_types import EmbeddingComputeKernel

        self.assertIn(EmbeddingComputeKernel.DENSE.value, result)

    def test_multiple_enum_values(self):
        """同时查询多个枚举值。"""
        from torchrec.distributed.embedding_types import EmbeddingComputeKernel

        result = adapter.embedding_compute_kernel_values("DENSE", "SPARSE", "SSD_VIRTUAL_TABLE", "DRAM_VIRTUAL_TABLE")
        self.assertIsInstance(result, set)
        self.assertGreater(len(result), 0)
        self.assertIn(EmbeddingComputeKernel.DENSE.value, result)


class TestMakeKjtListSplitsAwaitable(unittest.TestCase):
    """测试 make_kjt_list_splits_awaitable 构造方法。"""

    def test_construct_with_all_params(self):
        from torchrec.distributed.embedding_sharding import KJTListSplitsAwaitable

        obj = adapter.make_kjt_list_splits_awaitable(
            awaitables=[],
            ctx=None,
            module_fqn="test.mod",
            sharding_types=["rowwise", "colwise"],
        )
        self.assertIsInstance(obj, KJTListSplitsAwaitable)

    def test_construct_minimal_params(self):
        from torchrec.distributed.embedding_sharding import KJTListSplitsAwaitable

        obj = adapter.make_kjt_list_splits_awaitable(
            awaitables=[],
            ctx=None,
            module_fqn=None,
            sharding_types=[],
        )
        self.assertIsInstance(obj, KJTListSplitsAwaitable)


class TestVersionFlags(unittest.TestCase):
    """测试向后兼容的版本标志。"""

    def test_version_flags_consistent(self):
        """验证版本标志与适配器版本一致。"""
        from hybrid_torchrec import IS_TORCH_REC_120, IS_TORCH_REC_150

        self.assertEqual(IS_TORCH_REC_120, adapter.version == (1, 2, 0))
        self.assertEqual(IS_TORCH_REC_150, adapter.version == (1, 5, 0))


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


if __name__ == "__main__":
    unittest.main()
