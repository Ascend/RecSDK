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
import os.path
import subprocess

from pathlib import Path

CHANGES = "changes.txt"
ALLOWED_EXTENSIONS = {'.py', '.h', '.cpp', '.hpp', '.sh', '.cmake'}
MODULE_MAPS = {
    'common': 'hstu',
    'cmake': 'hstu',
    'hstu_dense_forward': 'hstu'
}


def is_source_code_file(file: str) -> bool:
    return Path(file).suffix in ALLOWED_EXTENSIONS


def read_changes(src: str) -> list[str]:
    if not os.path.exists(src):
        return []
    with open(src, "r") as f:
        changes = map(lambda line: line.strip(), f)
        changes = list(filter(is_source_code_file, changes))
    return changes


def parse_modules(file: str) -> str:
    if file.startswith("cust_op"):
        if file.startswith("cust_op/ascendc_op/ai_core_op"):
            relative_path = Path(file).relative_to("cust_op/ascendc_op/ai_core_op")
            return relative_path.parts[0]
        elif file.startswith("cust_op/framework/torch_plugin/torch_library"):
            relative_path = Path(file).relative_to("cust_op/framework/torch_plugin/torch_library")
            return relative_path.parts[0]
    if file.startswith("training"):
        return "torchrec"
    return ""


def main(src):
    modules = set()
    changes = read_changes(src)
    if not changes:
        return
    for file in changes:
        module = parse_modules(file)
        clean_module = MODULE_MAPS.get(module)
        if clean_module:
            modules.add(clean_module)

    for module in modules:
        path = Path(f"{module}/run.sh").absolute()
        subprocess.run(["bash", path], check=True, shell=False)


if __name__ == '__main__':
    main(CHANGES)
