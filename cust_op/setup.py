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
import sys
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

_torch_ver = torch.__version__.split("+")[0]
package_version = f"{_torch_ver}+{arch_suffix}"


def _get_torch_prefix():
    return os.path.dirname(torch.__file__)


def _build_variants():
    """AscendC 算子编译范围（受 CANN 平台信息限制）。
    torch_plugin 适配层 .so 按芯片拆分编译，仅包含对应芯片的算子。
    """
    value = os.environ.get("RECSDK_BUILD_VERS")
    if value:
        return value
    return "A2,A3,A5"


def _ascend_serial_build():
    value = os.environ.get("RECSDK_ASCEND_SERIAL_BUILD", "ON").strip().upper()
    return "OFF" if value in {"0", "OFF", "FALSE", "NO"} else "ON"


def _get_cxx11_abi():
    v = getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", 0)
    return 1 if v else 0


def cmake_args():
    torch_root = _get_torch_prefix()

    os.environ.setdefault(
        "CMAKE_BUILD_PARALLEL_LEVEL",
        str((os.cpu_count() or 4) // 2),
    )

    # 安全编译选项
    # 编译选项: 栈保护
    compiler_flags = "-fstack-protector-strong"
    # 链接选项: RELRO(GOT表保护)、BIND_NOW(立即绑定)、Strip(删除符号表)
    link_flags = "-Wl,-z,relro -Wl,-z,now -Wl,-s"

    return [
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DCMAKE_PREFIX_PATH={torch_root}",
        f"-DRECSDK_BUILD_VERS={_build_variants()}",
        f"-DRECSDK_ASCEND_SERIAL_BUILD={_ascend_serial_build()}",
        f"-D_GLIBCXX_USE_CXX11_ABI={_get_cxx11_abi()}",
        # 编译选项
        f"-DCMAKE_C_FLAGS={compiler_flags}",
        f"-DCMAKE_CXX_FLAGS={compiler_flags}",
        # 链接选项
        f"-DCMAKE_EXE_LINKER_FLAGS={link_flags}",
        f"-DCMAKE_SHARED_LINKER_FLAGS={link_flags}",
        # 禁用 RPATH
        "-DCMAKE_SKIP_BUILD_RPATH=TRUE",
        "-DCMAKE_SKIP_INSTALL_RPATH=TRUE",
    ]


# 始终构建所有芯片版本 (A5/A3/A2)，运行时根据 SOC 检测选择加载
skbuild_setup(
    name="rec_ops",
    version=package_version,
    description="RecSDK Custom Operations for Multiple Chips",
    packages=setuptools.find_packages(where=".", include=["rec_ops", "rec_ops.*"]),
    cmake_args=cmake_args(),
    cmake_install_dir="rec_ops",
    include_package_data=True,
    zip_safe=False,
)
