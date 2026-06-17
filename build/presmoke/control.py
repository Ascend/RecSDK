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
import subprocess

from pathlib import Path

ALLOWED_EXTENSIONS = {'.py', '.h', '.cpp', '.hpp', '.sh', '.cmake'}
PATH_PREFIX_MAPS = {
    'cust_op/ascendc_op/ai_core_op/hstu_dense_forward': 'hstu',
    'cust_op/ascendc_op/ai_core_op/hstu_dense_backward': 'hstu',
    'cust_op/framework/torch_plugin/torch_library/hstu': 'hstu',
}


def is_source_code_file(file: str) -> bool:
    return Path(file).suffix in ALLOWED_EXTENSIONS


def get_changed_files() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "origin/develop...HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    if result.returncode != 0:
        return []
    changes = result.stdout.splitlines()
    return [f.strip() for f in changes if is_source_code_file(f.strip())]


def parse_module(file: str) -> str:
    for prefix, module in PATH_PREFIX_MAPS.items():
        if file.startswith(prefix):
            return module
    return ""


def main():
    modules = set()
    changes = get_changed_files()
    if not changes:
        return
    for file in changes:
        module = parse_module(file)
        if module:
            modules.add(module)

    for module in modules:
        path = Path(f"{module}/run.sh").absolute()
        subprocess.run(["bash", path], check=True, shell=False)


if __name__ == '__main__':
    main()
