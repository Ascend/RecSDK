#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""适配器层单元测试：直接 import 即可运行。

测试覆盖：
1. 版本检测：_version 模块
2. 适配器工厂：根据 _version_diff 自动选择正确实现
3. 兼容性辅助：_compat 模块的过滤函数
"""

import unittest
from dataclasses import dataclass

from hybrid_torchrec._adapters._adapter_base import TorchRecVersionAdapter
from hybrid_torchrec._adapters import _build_methods, adapter
from hybrid_torchrec._adapters._version import _torchrec_version_tuple


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


if __name__ == "__main__":
    unittest.main()
