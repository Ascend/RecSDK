#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""版本适配模块完整单元测试。

测试覆盖：
1. _version.py: 版本检测逻辑
2. _version_diff.py: 声明式差异表
3. _adapter_base.py: 适配器基类与兼容性工具
4. _adapters/__init__.py: 适配器工厂
"""

import unittest
from dataclasses import dataclass
from unittest.mock import patch
from importlib import metadata

import torchrec


# ==================== 版本检测模块测试 ====================


class TestVersionDetection(unittest.TestCase):
    """测试 _version 模块的版本检测功能。"""

    def test_torchrec_version_tuple_shape(self):
        """验证版本元组格式正确。"""
        from hybrid_torchrec._adapters._version import _torchrec_version_tuple

        v = _torchrec_version_tuple()
        self.assertIsInstance(v, tuple)
        self.assertEqual(len(v), 3)
        for part in v:
            self.assertIsInstance(part, int)

    def test_version_tuple_boundary_cases(self):
        """测试版本解析的边界情况。"""
        from hybrid_torchrec._adapters._version import _torchrec_version_tuple

        # 测试各种版本格式
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


# ==================== 差异表模块测试 ====================


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


# ==================== 适配器工厂测试 ====================


@dataclass
class _FakeCommonArgs:
    learning_rate: float = 0.5


@dataclass
class _FakeOptimizerArgs:
    learning_rate: float = 0.1


@dataclass
class _FakeKernel:
    _method_lr: float = None

    def get_learning_rate(self) -> float:
        if self._method_lr is None:
            raise AttributeError("1.1.0 kernel has no get_learning_rate()")
        return self._method_lr


class TestAdapterFactory(unittest.TestCase):
    """测试适配器工厂的动态构建逻辑。"""

    def test_build_methods_baseline_110(self):
        """测试 1.1.0 基线版本的方法选择。"""
        from hybrid_torchrec._adapters import _build_methods
        from hybrid_torchrec._adapters._adapter_base import TorchRecVersionAdapter

        methods = _build_methods((1, 1, 0))

        # 1.1.0 没有版本差异，所以 methods 为空
        self.assertEqual(methods, {})

        # 通过基类验证默认行为
        class Adapter110(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 1, 0)

        adapter_110 = Adapter110()

        # get_learning_rate 应从 optimizer_args 获取 (1.1.0 行为)
        result = adapter_110.get_learning_rate(_FakeCommonArgs(), _FakeOptimizerArgs())
        self.assertEqual(result, 0.1)

        # get_kernel_learning_rate 应返回 0.0 (1.1.0 行为)
        result = adapter_110.get_kernel_learning_rate(_FakeKernel())
        self.assertEqual(result, 0.0)

    def test_build_methods_120_overrides(self):
        """测试 1.2.0 的方法覆盖。"""
        from hybrid_torchrec._adapters import _build_methods
        from hybrid_torchrec._adapters._adapter_base import TorchRecVersionAdapter

        methods = _build_methods((1, 2, 0))

        # 1.2.0 应该有覆盖
        self.assertIn("get_learning_rate", methods)
        self.assertIn("get_kernel_learning_rate", methods)

        class Adapter120(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 2, 0)

        adapter_120 = type("Adapter120", (Adapter120,), methods)()

        # get_learning_rate 应从 common_args 获取 (1.2.0 行为)
        result = adapter_120.get_learning_rate(_FakeCommonArgs(), _FakeOptimizerArgs())
        self.assertEqual(result, 0.5)

        # get_kernel_learning_rate 应调用 kernel.get_learning_rate()
        result = adapter_120.get_kernel_learning_rate(_FakeKernel(_method_lr=0.5))
        self.assertEqual(result, 0.5)

    def test_build_methods_150_overrides(self):
        """测试 1.5.0 的方法覆盖。"""
        from hybrid_torchrec._adapters import _build_methods
        from hybrid_torchrec._adapters._adapter_base import TorchRecVersionAdapter

        methods = _build_methods((1, 5, 0))

        # 1.5.0 应该包含之前所有版本的覆盖
        self.assertIn("get_learning_rate", methods)

        class Adapter150(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 5, 0)

        adapter_150 = type("Adapter150", (Adapter150,), methods)()

        # 继承 1.2.0 的 get_learning_rate 行为
        result = adapter_150.get_learning_rate(_FakeCommonArgs(), _FakeOptimizerArgs())
        self.assertEqual(result, 0.5)

    def test_build_methods_higher_version_inherits(self):
        """测试更高版本继承所有先前行为。"""
        from hybrid_torchrec._adapters import _build_methods
        from hybrid_torchrec._adapters._adapter_base import TorchRecVersionAdapter

        methods = _build_methods((1, 10, 0))

        class Adapter100(TorchRecVersionAdapter):
            @property
            def version(self):
                return (1, 10, 0)

        adapter_100 = type("Adapter100", (Adapter100,), methods)()

        # 应继承 1.5.0 的行为
        result = adapter_100.get_learning_rate(_FakeCommonArgs(), _FakeOptimizerArgs())
        self.assertEqual(result, 0.5)

        # 应继承 1.2.0 的 kernel 行为
        result = adapter_100.get_kernel_learning_rate(_FakeKernel(_method_lr=0.7))
        self.assertEqual(result, 0.7)

    def test_adapter_singleton(self):
        """验证适配器是单例。"""
        from hybrid_torchrec._adapters import adapter, _create_adapter

        adapter1 = adapter
        adapter2 = _create_adapter()

        self.assertIs(adapter1, adapter2)
        self.assertEqual(adapter1.version, adapter2.version)

    def test_adapter_has_all_methods(self):
        """验证适配器暴露所有声明的方法。"""
        from hybrid_torchrec._adapters import adapter

        required_methods = [
            'version',
            'get_learning_rate',
            'create_sharding_infos',
            'build_args_kwargs',
            'get_output_dtensor',
            'get_kernel_learning_rate',
            'make_embedding_table_config',
            'make_awaitable',
            'filter_rw_sparse_features_dist_kwargs',
            'embedding_compute_kernel_values',
        ]

        for method_name in required_methods:
            self.assertTrue(hasattr(adapter, method_name), f"adapter missing method: {method_name}")


# ==================== 兼容性工具测试 ====================


class TestCompatibilityUtils(unittest.TestCase):
    """测试适配器的兼容性工具方法。"""

    def test_make_embedding_table_config_filters_unknown_fields(self):
        """测试自动过滤不支持的字段。"""
        from hybrid_torchrec._adapters import adapter
        from torchrec.modules.embedding_configs import EmbeddingTableConfig

        cfg = adapter.make_embedding_table_config(
            num_embeddings=100,
            embedding_dim=8,
            name="test_table",
            # 未来版本字段，应被过滤
            future_field_1=42,
            virtual_table_eviction_policy="lru",
            enable_embedding_update=True,
        )

        self.assertIsInstance(cfg, EmbeddingTableConfig)
        self.assertEqual(cfg.num_embeddings, 100)
        self.assertEqual(cfg.embedding_dim, 8)
        self.assertEqual(cfg.name, "test_table")

    def test_make_awaitable_filters_kwargs(self):
        """测试自动适配构造函数参数。"""
        from hybrid_torchrec._adapters import adapter
        from torchrec.distributed.embedding_sharding import KJTListSplitsAwaitable

        obj = adapter.make_awaitable(
            KJTListSplitsAwaitable,
            awaitables=[],
            ctx=None,
            module_fqn="test.module",
            sharding_types=["rowwise", "colwise"],
            # 未来版本参数
            future_param="value",
        )

        self.assertIsInstance(obj, KJTListSplitsAwaitable)

    def test_filter_rw_sparse_features_dist_kwargs(self):
        """测试过滤 RwSparseFeaturesDist 参数。"""
        from hybrid_torchrec._adapters import adapter

        kwargs = adapter.filter_rw_sparse_features_dist_kwargs(
            pg=None,
            num_features=3,
            feature_hash_sizes=[1000, 2000, 3000],
            device=None,
            is_sequence=False,
            has_feature_processor=False,
            need_pos=False,
            # 未来版本参数
            virtual_table_feature_num_buckets=[100, 200, 300],
            has_uneven_virtual_tables=True,
            future_field_abc=123,
        )

        self.assertIn("pg", kwargs)
        self.assertIn("num_features", kwargs)
        self.assertIn("feature_hash_sizes", kwargs)
        # 不应该包含完全不存在的字段
        self.assertNotIn("future_field_abc", kwargs)

    def test_embedding_compute_kernel_values(self):
        """测试安全获取枚举值。"""
        from hybrid_torchrec._adapters import adapter
        from torchrec.distributed.embedding_types import EmbeddingComputeKernel

        # 测试已知枚举值 - DENSE 在所有版本中都存在
        result = adapter.embedding_compute_kernel_values("DENSE")
        self.assertIn(EmbeddingComputeKernel.DENSE.value, result)

        # 测试可能存在的枚举值 - SPARSE 可能不存在于某些版本
        result = adapter.embedding_compute_kernel_values("DENSE", "SPARSE")
        self.assertIn(EmbeddingComputeKernel.DENSE.value, result)
        # SPARSE 如果存在则应包含，不存在则被忽略
        sparse_value = getattr(EmbeddingComputeKernel, "SPARSE", None)
        if sparse_value is not None:
            self.assertIn(sparse_value.value, result)

        # 测试不存在的枚举值（不应抛出异常）
        result = adapter.embedding_compute_kernel_values("DENSE", "VIRTUAL_TABLE", "NON_EXISTENT")
        self.assertIn(EmbeddingComputeKernel.DENSE.value, result)
        # NON_EXISTENT 应被忽略

    def test_make_kjt_list_splits_awaitable(self):
        """测试构造 KJTListSplitsAwaitable。"""
        from hybrid_torchrec._adapters import adapter
        from torchrec.distributed.embedding_sharding import KJTListSplitsAwaitable

        obj = adapter.make_kjt_list_splits_awaitable(
            awaitables=[],
            ctx=None,
            module_fqn="test.module",
            sharding_types=["rowwise"],
        )

        self.assertIsInstance(obj, KJTListSplitsAwaitable)


# ==================== 版本标志测试 ====================


class TestVersionFlags(unittest.TestCase):
    """测试向后兼容的版本标志。"""

    def test_version_flags_consistent(self):
        """验证版本标志与适配器版本一致。"""
        from hybrid_torchrec import IS_TORCH_REC_120, IS_TORCH_REC_150
        from hybrid_torchrec._adapters import adapter

        self.assertEqual(IS_TORCH_REC_120, adapter.version == (1, 2, 0))
        self.assertEqual(IS_TORCH_REC_150, adapter.version == (1, 5, 0))


# ==================== 综合测试 ====================


class TestEndToEndAdapter(unittest.TestCase):
    """端到端测试适配器在实际场景中的行为。"""

    def test_adapter_version_matches_current_torchrec(self):
        """验证适配器版本与当前 torchrec 版本一致。"""
        from hybrid_torchrec._adapters import adapter
        from hybrid_torchrec._adapters._version import _torchrec_version_tuple

        self.assertEqual(adapter.version, _torchrec_version_tuple())

    def test_adapter_methods_callable(self):
        """验证适配器所有方法可调用。"""
        from hybrid_torchrec._adapters import adapter

        # 测试 get_learning_rate
        lr = adapter.get_learning_rate(_FakeCommonArgs(), _FakeOptimizerArgs())
        self.assertIsInstance(lr, float)

        # 测试 get_kernel_learning_rate
        kernel_lr = adapter.get_kernel_learning_rate(_FakeKernel(_method_lr=0.1))
        self.assertIsInstance(kernel_lr, float)


if __name__ == "__main__":
    unittest.main()
