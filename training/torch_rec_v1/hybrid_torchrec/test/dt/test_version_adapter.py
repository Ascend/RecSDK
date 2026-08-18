#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pylint: disable=duplicate-code
"""适配器层单元测试：直接 import 即可运行。

测试覆盖：
1. 版本检测：_version 模块
2. 适配器工厂：根据 _version_diff 自动选择正确实现
3. 兼容性辅助：_compat 模块的过滤函数
"""

import unittest
from dataclasses import dataclass

from hybrid_torchrec._adapters import _build_methods, adapter
from hybrid_torchrec._adapters._adapter_base import TorchRecVersionAdapter
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


if __name__ == "__main__":
    unittest.main()
