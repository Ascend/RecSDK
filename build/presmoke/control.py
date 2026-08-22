#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
import subprocess

from pathlib import Path

ALLOWED_EXTENSIONS = {'.py', '.h', '.cpp', '.hpp', '.sh', '.cmake'}
PATH_PREFIX_MAPS = {
    'cust_op/ascendc_op/ai_core_op/hstu_dense_forward': ['hstu'],
    'cust_op/ascendc_op/ai_core_op/hstu_dense_backward': ['hstu'],
    'cust_op/framework/torch_plugin/torch_library/hstu': ['hstu'],
    'cust_op/framework/torch_plugin/torch_library/common': ['hstu'],
    'training/torch_rec_v1/hybrid_torchrec': ['torchrec'],
    'training/torch_rec_v1/torchrec_npu': ['torchrec'],
    'training/torch_rec_v1/torchrec_embcache': ['torchrec'],
}
# torchrec 冒烟依赖 libfbgemm_npu_api.so（A2 cust_op torch_plugin），它由
# PTA_DIR（torch_library/common）的 build_ops.sh 编译并安装到 site-packages。
# hybrid_torchrec 已不再 import fbgemm_ascend（避免 A5/A2 算子冲突），只能依赖
# 该 .so 注册 A2 的 split_embedding_codegen_lookup_adagrad_function，故 torchrec
# 模块必须与 hstu 一样先构建 PTA，否则 NPU 路径算子缺失会 AttributeError。
PTA_REQUIRED_MODULES = {'hstu', 'torchrec'}
_ALREADY_BUILT_PTA = False
_PRESMOKE_DIR = Path(os.environ.get("PRESMOKE_DIR", "")).absolute()
_PTA_DIR = Path(os.environ.get("PTA_DIR", "")).absolute()


def is_source_code_file(file: str) -> bool:
    return Path(file).suffix in ALLOWED_EXTENSIONS


def get_changed_files() -> list[str]:
    changes_file = _PRESMOKE_DIR / "changes.txt"
    if not changes_file.exists():
        raise RuntimeError(f"changes.txt file does not exist in {_PRESMOKE_DIR}")
    changes = changes_file.read_text(encoding="utf-8").splitlines()
    return [f.strip() for f in changes if is_source_code_file(f.strip())]


def parse_module(file: str) -> list[str]:
    for prefix, modules in PATH_PREFIX_MAPS.items():
        if file.startswith(prefix):
            return modules
    return []


def build_pta(module) -> None:
    global _ALREADY_BUILT_PTA
    if module not in PTA_REQUIRED_MODULES or _ALREADY_BUILT_PTA:
        return
    subprocess.run(["dos2unix", "build_ops.sh"], check=True, shell=False, cwd=_PTA_DIR)
    subprocess.run(["bash", "build_ops.sh"], check=True, shell=False, cwd=_PTA_DIR)
    _ALREADY_BUILT_PTA = True


def main():
    modules = set()
    changes = get_changed_files()
    if not changes:
        return
    for file in changes:
        for module in parse_module(file):
            modules.add(module)

    for module in modules:
        build_pta(module)
        result = subprocess.run(
            ["bash", f"{module}/run.sh"],
            check=False,
            shell=False,
            cwd=_PRESMOKE_DIR,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Module {module} run.sh failed with return code {result.returncode}")


if __name__ == '__main__':
    main()
