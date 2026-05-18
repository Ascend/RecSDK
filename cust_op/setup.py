#!/usr/bin/env python3
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
import platform
import setuptools
import torch

try:
    from skbuild import setup as skbuild_setup
except ImportError:
    print("scikit-build is required to build from source.")
    raise

arch = platform.machine().lower()
if arch in ["aarch64", "arm64"]:
    arch_suffix = "aarch64"
elif arch in ["x86_64", "amd64"]:
    arch_suffix = "x86_64"
else:
    arch_suffix = f"{arch}"

package_version = f"1.0.0+{arch_suffix}"


def _get_torch_prefix():
    return os.path.dirname(torch.__file__)


def _build_variants():
    """AscendC 算子编译范围（受 CANN 平台信息限制）。
    torch_plugin 适配层 .so 始终编译全部变体，不受此影响。
    """
    value = os.environ.get("RECSDK_BUILD_VERS")
    if value:
        return value
    return "A2,A3,A5"


def _ascend_serial_build():
    value = os.environ.get("RECSDK_ASCEND_SERIAL_BUILD", "ON").strip().upper()
    return "OFF" if value in {"0", "OFF", "FALSE", "NO"} else "ON"


def cmake_args():
    torch_root = _get_torch_prefix()

    os.environ.setdefault(
        "CMAKE_BUILD_PARALLEL_LEVEL",
        str((os.cpu_count() or 4) // 2),
    )

    return [
        f"-DCMAKE_PREFIX_PATH={torch_root}",
        f"-DRECSDK_BUILD_VERS={_build_variants()}",
        f"-DRECSDK_ASCEND_SERIAL_BUILD={_ascend_serial_build()}",
    ]


# 始终构建所有芯片版本 (A5/A2/A3)，运行时根据 SOC 检测选择加载
skbuild_setup(
    name="rec_sdk_ops",
    version=package_version,
    description="RecSDK Custom Operations for Multiple Chips",
    packages=setuptools.find_packages(where=".", include=["rec_sdk_ops", "rec_sdk_ops.*"]),
    cmake_args=cmake_args(),
    cmake_install_dir="rec_sdk_ops",
    include_package_data=True,
    zip_safe=False,
)
