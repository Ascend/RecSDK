# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NGO可选依赖模块单元测试."""

import unittest
from unittest.mock import patch, Mock

from ngo.utils.optional_deps import _try_import, torch_npu, has_torch_npu


class TestTryImportFunction(unittest.TestCase):
    """测试_try_import函数."""

    @patch("ngo.utils.optional_deps.importlib.import_module")
    def test_try_import_success(self, mock_import_module):
        """测试成功导入模块."""
        mock_module = Mock()
        mock_import_module.return_value = mock_module

        result = _try_import("test_module")

        mock_import_module.assert_called_once_with("test_module")
        self.assertEqual(result, mock_module)

    @patch("ngo.utils.optional_deps.importlib.import_module")
    def test_try_import_failure(self, mock_import_module):
        """测试导入模块失败."""
        mock_import_module.side_effect = ImportError("Module not found")

        result = _try_import("nonexistent_module")

        mock_import_module.assert_called_once_with("nonexistent_module")
        self.assertIsNone(result)

    @patch("ngo.utils.optional_deps.importlib.import_module")
    def test_try_import_torch_npu_main_failure(self, mock_import_module):
        """测试导入torch_npu主模块失败."""
        mock_import_module.side_effect = ImportError("torch_npu not found")

        result = _try_import("torch_npu")

        mock_import_module.assert_called_once_with("torch_npu")
        self.assertIsNone(result)

    @patch("ngo.utils.optional_deps.importlib.import_module")
    def test_try_import_none_result(self, mock_import_module):
        """测试导入返回None的情况."""
        mock_import_module.return_value = None

        result = _try_import("test_module")

        mock_import_module.assert_called_once_with("test_module")
        self.assertIsNone(result)


class TestTorchNpuVariable(unittest.TestCase):
    """测试torch_npu变量."""

    def test_torch_npu_is_defined(self):
        """测试torch_npu变量已定义."""
        # torch_npu变量应该在模块级别定义
        from ngo.utils import optional_deps

        self.assertTrue(hasattr(optional_deps, "torch_npu"))

    def test_torch_npu_type(self):
        """测试torch_npu变量的类型."""
        # torch_npu应该是模块对象或None
        self.assertTrue(torch_npu is None or hasattr(torch_npu, "__name__"))


class TestHasTorchNpuFunction(unittest.TestCase):
    """测试has_torch_npu函数."""

    def test_has_torch_npu_returns_boolean(self):
        """测试has_torch_npu返回布尔值."""
        result = has_torch_npu()
        self.assertIsInstance(result, bool)

    @patch("ngo.utils.optional_deps.torch_npu", None)
    def test_has_torch_npu_when_none(self):
        """测试torch_npu为None时has_torch_npu返回False."""
        result = has_torch_npu()
        self.assertFalse(result)

    @patch("ngo.utils.optional_deps.torch_npu", Mock())
    def test_has_torch_npu_when_available(self):
        """测试torch_npu可用时has_torch_npu返回True."""
        result = has_torch_npu()
        self.assertTrue(result)

    def test_has_torch_npu_consistency(self):
        """测试has_torch_npu与torch_npu变量的一致性."""
        expected = torch_npu is not None
        actual = has_torch_npu()
        self.assertEqual(expected, actual)


class TestOptionalDepsIntegration(unittest.TestCase):
    """测试可选依赖模块的集成功能."""

    def test_module_imports(self):
        """测试模块导入."""
        # 测试所有公共函数和变量都可以导入
        from ngo.utils.optional_deps import _try_import, torch_npu, has_torch_npu

        self.assertTrue(callable(_try_import))
        self.assertTrue(callable(has_torch_npu))
        # torch_npu可能是None或模块对象

    def test_has_torch_npu_usage_example(self):
        """测试has_torch_npu的使用示例."""
        if has_torch_npu():
            # 如果torch_npu可用，应该可以访问它
            self.assertIsNotNone(torch_npu)
            # 这里可以添加更多torch_npu特定的测试
        else:
            # 如果torch_npu不可用，应该为None
            self.assertIsNone(torch_npu)

    @patch("ngo.utils.optional_deps.torch_npu", Mock())
    def test_conditional_import_usage(self):
        """测试条件导入的使用模式."""
        # 模拟torch_npu可用的情况
        if has_torch_npu():
            # 应该能够使用torch_npu
            from ngo.utils.optional_deps import torch_npu

            self.assertIsNotNone(torch_npu)
            # 这里可以添加使用torch_npu的代码

    @patch("ngo.utils.optional_deps.torch_npu", None)
    def test_conditional_import_usage_none(self):
        """测试条件导入在torch_npu不可用时的情况."""
        # 模拟torch_npu不可用的情况
        if not has_torch_npu():
            # 应该使用替代方案
            self.assertIsNone(torch_npu)
            # 这里可以添加替代方案的代码


class TestOptionalDepsEdgeCases(unittest.TestCase):
    """测试可选依赖模块的边缘情况."""

    def test_try_import_invalid_module_name(self):
        """测试导入无效的模块名."""
        result = _try_import("invalid.module.name.with.dots")
        # 结果可能是None或抛出异常，取决于模块是否存在
        self.assertTrue(result is None or hasattr(result, "__name__"))


if __name__ == "__main__":
    unittest.main()
