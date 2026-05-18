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

import logging
import os
import shutil
import subprocess
import torch

_EXT_SUFFIX = ".pyd" if os.name == "nt" else ".so"

# Get the absolute path to this python package
package_dir = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))


# ========================================================================
# 1. SOC 检测与 OPP 路径设置
# ========================================================================
def _detect_soc_from_npu_smi():
    npu_smi = shutil.which("npu-smi")
    if not npu_smi:
        logging.warning("rec_sdk_ops: npu-smi not found.")
        return None
    try:
        out = subprocess.check_output(
            [npu_smi, "info", "-m"],
            stderr=subprocess.STDOUT,
        ).decode(errors="ignore")
    except subprocess.CalledProcessError:
        logging.warning("rec_sdk_ops: npu-smi failed.")
        return None

    for line in out.splitlines():
        if "Ascend" in line and "Mcu" not in line:
            suffix = line.split("Ascend", 1)[1].strip().split()[0]
            return f"Ascend{suffix}"
    return None


def _detect_soc_version():
    soc = os.environ.get("SOC_VERSION")
    if soc:
        return soc
    return _detect_soc_from_npu_smi()


def _map_soc_to_variant(soc):
    if not soc:
        logging.warning("rec_sdk_ops: SOC not detected, defaulting to A5 variants")
        return "A5"
    if soc.startswith("Ascend95"):
        return "A5"
    if soc.startswith("Ascend910B"):
        return "A2"
    if soc.startswith("Ascend910_93"):
        return "A3"
    logging.warning("rec_sdk_ops: unknown SOC '%s', defaulting to A5", soc)
    return "A5"


def _select_opp_variant():
    override = os.environ.get("RECSDK_FORCE_BUILD_VER", "").strip().upper()
    if override in {"A5", "A2", "A3"}:
        return override
    return _map_soc_to_variant(_detect_soc_version())


def _setup_custom_opp_path():
    """在当前进程内刷新 ASCEND_CUSTOM_OPP_PATH。

    设计原则：
    - 幂等：多次 import 不会无限追加同一前缀
    - 不覆盖用户自定义路径：保留原有非本包前缀
    """
    custom_opp_path = os.path.join(package_dir, "custom_opp")
    if not os.path.isdir(custom_opp_path):
        return

    variant = _select_opp_variant()
    variant_root = os.path.join(custom_opp_path, variant)

    new_paths = []
    # 添加 variant 级目录
    if os.path.isdir(variant_root):
        vendors_dir = os.path.join(variant_root, "vendors")
        if os.path.isdir(vendors_dir):
            try:
                for name in os.listdir(vendors_dir):
                    vendor_path = os.path.join(vendors_dir, name)
                    if os.path.isdir(vendor_path):
                        new_paths.append(vendor_path)
            except Exception as e:
                logging.error("rec_sdk_ops: failed to scan vendors dir: %s", e)
        new_paths.append(variant_root)

    # 同时保留所有芯片目录（兼容老方式）
    if not new_paths:
        for chip_dir_name in os.listdir(custom_opp_path):
            chip_dir = os.path.join(custom_opp_path, chip_dir_name)
            if os.path.isdir(chip_dir):
                new_paths.append(chip_dir)

    existing = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    old_parts = [p for p in existing.split(os.pathsep) if p and not p.startswith(custom_opp_path)]
    merged = new_paths + old_parts

    # 去重同时保持顺序
    seen = set()
    deduped = []
    for p in merged:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(deduped)
    logging.info(
        "rec_sdk_ops: ASCEND_CUSTOM_OPP_PATH=%s (variant=%s)",
        os.environ["ASCEND_CUSTOM_OPP_PATH"],
        variant,
    )


# ========================================================================
# 2. torch_plugin 适配层 .so 加载
# ========================================================================
_HOST_LIB_CANDIDATES = {
    "A5": [
        "rec_sdk_ops_py_a5" + _EXT_SUFFIX,
        "librec_sdk_ops_py_a5.so",
    ],
    "A2A3": [
        "rec_sdk_ops_py_a2a3" + _EXT_SUFFIX,
        "librec_sdk_ops_py_a2a3.so",
    ],
}


def _candidate_lib_names(variant):
    key = "A5" if variant == "A5" else "A2A3"
    names = list(_HOST_LIB_CANDIDATES.get(key, []))
    for other_key, values in _HOST_LIB_CANDIDATES.items():
        if other_key != key:
            names.extend(values)
    return names


def _load_library(no_throw=False):

    search_dirs = [package_dir]
    # 可编辑安装时 .so 可能在 _skbuild 中
    pkg_root = os.path.dirname(package_dir)
    skbuild_dir = os.path.join(pkg_root, "_skbuild")
    if os.path.isdir(skbuild_dir):
        for name in os.listdir(skbuild_dir):
            install_base = os.path.join(skbuild_dir, name, "cmake-install", "rec_sdk_ops")
            if os.path.isdir(install_base):
                search_dirs.append(install_base)

    variant = _select_opp_variant()
    _setup_custom_opp_path()

    lib_names = _candidate_lib_names(variant)

    for search_dir in search_dirs:
        for filename in lib_names:
            lib_path = os.path.join(search_dir, filename)
            if os.path.isfile(lib_path):
                try:
                    torch.ops.load_library(lib_path)
                    logging.info("rec_sdk_ops: loaded '%s'", lib_path)
                    return
                except Exception as e:
                    logging.error("rec_sdk_ops: could not load '%s': %s", lib_path, e)
                    if not no_throw:
                        raise

    # 如果找不到 host library，仅设置 OPP 路径（兼容无 torch_plugin 的情况）
    msg = "rec_sdk_ops: host library not found (variant=%s, dirs=%s)" % (variant, search_dirs)
    if no_throw:
        logging.warning(msg)
        return
    logging.warning(msg)


# ========================================================================
# 初始化
# ========================================================================
_load_library(no_throw=True)
