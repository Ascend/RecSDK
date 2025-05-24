#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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

import os
import importlib.util
import traceback

from pattern.constant import PATTERN_DIR_NAME, TORCHINDUCTOR_CACHE_ENV, CACHE_DIR_NAME
from pattern.util import clean_up_inductor_cache
from utils.path_util import get_specified_file_dir
from utils.logger import default_logger

file_dir = get_specified_file_dir(__file__)

# 设置Inductor缓存目录
os.environ[TORCHINDUCTOR_CACHE_ENV] = os.path.join(file_dir, CACHE_DIR_NAME)

# 获取 pattern 目录路径
pattern_dir = os.path.join(file_dir, PATTERN_DIR_NAME)
default_logger.info("Pattern dir path: %s", pattern_dir)


# 新增统计变量
total_perform_test_calls = 0
patterns_without_perform_test = []


def run_all_patterns():
    total_count = 0
    success_count = 0
    fail_count = 0
    failed_files = []

    # 遍历目录下的所有 pattern 文件
    for filename in os.listdir(pattern_dir):
        if not (filename.endswith(".py") and filename.startswith("pattern_")):
            continue

        module_name = filename[:-3]  # 去掉 .py 后缀
        file_path = os.path.join(pattern_dir, filename)
        total_count += 1

        try:
            # 动态加载模块
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            def wrapped_perform_test(*args, **kwargs):
                global total_perform_test_calls
                total_perform_test_calls += 1
                from pattern.util import perform_test

                return perform_test(*args, **kwargs)

            # 替换 perform_test 引用
            if hasattr(module, "perform_test"):
                module.perform_test = wrapped_perform_test
            else:
                patterns_without_perform_test.append(module_name)
                continue

            # 检查是否存在 main 函数并调用
            if hasattr(module, "main") and callable(module.main):
                default_logger.info("正在执行 %s.main()...", module_name)
                try:
                    module.main()
                    success_count += 1
                except Exception as e:
                    error_msg = f"执行 {module_name}.main() 时出错: {str(e)}"
                    default_logger.error("%s\n%s", error_msg, traceback.format_exc())
                    fail_count += 1
                    failed_files.append(filename)
            else:
                warning_msg = f"模块 {module_name} 中没有可调用的 main() 函数"
                default_logger.warning(warning_msg)
                fail_count += 1
                failed_files.append(filename)

        except Exception as e:
            error_msg = f"加载模块 {module_name} 时出错: {str(e)}"
            default_logger.error("%s\n%s", error_msg, traceback.format_exc())
            fail_count += 1
            failed_files.append(filename)

    # 输出最终统计结果
    default_logger.info(
        "总用例数: %d, 成功用例数: %d, 失败用例数: %d",
        total_count,
        success_count,
        fail_count,
    )
    default_logger.info(
        "没有适配 perform_test函数 的pattern数量: %d， 对应的文件 %s: ",
        len(patterns_without_perform_test),
        patterns_without_perform_test,
    )
    if failed_files:
        default_logger.info("以下是失败的用例文件: %s", failed_files)


if __name__ == "__main__":
    run_all_patterns()
    clean_up_inductor_cache()
