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
# pylint: disable=duplicate-code

import os
import sys
import zipfile
import glob
from setuptools import setup, find_packages
from setuptools.command.install import install

import _setup_common

try:
    from _version import SDK_VERSION
except ImportError:
    SDK_VERSION = os.environ.get("RECSDK_VERSION", "26.2.0")

custom_build_py = _setup_common.make_custom_build_py_new("_torchrec_merged_src", chmod_so=True)


class CustomInstallCommand(install):
    pass


def detect_pt_version():
    return _setup_common.detect_version_via_pip("torch")


current_dir = os.path.dirname(os.path.abspath(__file__))
packages = find_packages()
package_dir = {}
data_files = []
merged_tmp_dir = os.path.join(current_dir, "_torchrec_merged_src")


def _max_mtime_under(root):
    """返回 root 下所有文件的最大 mtime（用于 O3 缓存失效判断）。"""
    m = 0.0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            p = os.path.join(dirpath, name)
            try:
                mt = os.path.getmtime(p)
                m = max(m, mt)
            except OSError:
                pass
    return m


is_building = any(arg in sys.argv for arg in ['install', 'bdist_wheel', 'build', 'develop', 'egg_info'])

if is_building:
    # 检测PyTorch版本
    torch_version = detect_pt_version()
    if torch_version.startswith("2.7."):
        whl_dir = "mindxsdk-torchrec/pt2.7_whl"
    elif torch_version.startswith("2.10."):
        whl_dir = "mindxsdk-torchrec/pt2.10_whl"
    else:
        sys.exit(f"Error: Unsupported PyTorch version: {torch_version}.")

    _setup_common.install_requirements_new()
    full_whl_dir = os.path.join(current_dir, whl_dir)

    if not os.path.exists(full_whl_dir):
        # build_wrapper 在无 venv 场景下可能只构建单一 pt 版本的 whl（按构建机当前
        # pytorch 环境产包），此时发布包内不存在其他 pt 版本目录。安装时明确列出本包
        # 实际包含的 pt 版本，提示用户切换到匹配的 pytorch 环境安装。
        whl_root = os.path.join(current_dir, "mindxsdk-torchrec")
        available = []
        if os.path.isdir(whl_root):
            available = sorted(
                d for d in os.listdir(whl_root) if d.endswith("_whl") and os.path.isdir(os.path.join(whl_root, d))
            )
        if available:
            sys.exit(
                f"Error: PyTorch {torch_version} whl directory '{whl_dir}' not found in this package. "
                f"This release only contains: {', '.join(available)}. "
                f"Please install under a matching PyTorch environment."
            )
        sys.exit(f"Error: Cannot find directory of whl_file: {full_whl_dir}.")

    whl_paths = glob.glob(os.path.join(full_whl_dir, "*.whl"))
    if not whl_paths:
        sys.exit(f"Error: Cannot find any .whl file in {full_whl_dir}")

    hybrid_whls = [p for p in whl_paths if 'hybrid_torchrec' in os.path.basename(p)]
    embcache_whls = [p for p in whl_paths if 'torchrec_embcache' in os.path.basename(p)]
    other_whls = [p for p in whl_paths if p not in hybrid_whls and p not in embcache_whls]

    # 解包顺序：hybrid_torchrec -> torchrec_embcache -> 其他（后解包者覆盖同路径文件）
    whl_paths_sorted = []
    whl_paths_sorted.extend(hybrid_whls)
    whl_paths_sorted.extend(embcache_whls)
    whl_paths_sorted.extend(other_whls)

    print(f"Debugging full_whl_dir = {full_whl_dir}")
    print(f"Debugging whl_paths = {whl_paths_sorted}")

    if not os.path.exists(merged_tmp_dir):
        os.makedirs(merged_tmp_dir)

    for idx, whl_path in enumerate(whl_paths_sorted):
        print(f"  -> [{idx + 1}/{len(whl_paths_sorted)}] unzipping: {os.path.basename(whl_path)}")
        with zipfile.ZipFile(whl_path, 'r') as zip_ref:
            zip_ref.extractall(merged_tmp_dir)

    # O3 优化：缓存 find_packages 结果到 merged_tmp_dir/.pkglist（基于目录 mtime
    # 失效检测——解压后 mtime 必更新，自动重扫）。省去大目录（torchrec 几千个 .py）
    # 的 AST 扫描开销。
    pkg_cache = os.path.join(merged_tmp_dir, ".pkglist")
    if os.path.exists(pkg_cache) and os.path.getmtime(pkg_cache) >= _max_mtime_under(merged_tmp_dir):
        with open(pkg_cache, "r", encoding="utf-8") as f:
            extracted_pkgs = [line.strip() for line in f if line.strip()]
        print(f"  -> [CACHE] Reused find_packages ({len(extracted_pkgs)} packages)")
    else:
        extracted_pkgs = find_packages(where=merged_tmp_dir)
        with open(pkg_cache, "w", encoding="utf-8") as f:
            f.write("\n".join(extracted_pkgs))
        print(f"  -> [CACHE] Cached find_packages ({len(extracted_pkgs)} packages)")

    for pkg in extracted_pkgs:
        if pkg not in packages:
            packages.append(pkg)
            top_level = pkg.split('.')[0]
            if top_level not in package_dir:
                package_dir[top_level] = os.path.join("_torchrec_merged_src", top_level)
                print(f"  -> Linked Python package: {top_level} (and its subpackages)")


setup(
    name='torch_rec_v1',
    version=SDK_VERSION,
    description='torchrec wrapper package with dynamic PyTorch version detection',
    install_requires=[],
    cmdclass={
        'install': CustomInstallCommand,
        'build_py': custom_build_py,
    },
    packages=packages,
    package_dir=package_dir,
    data_files=data_files,
    include_package_data=True,
    package_data={'': ['*', '*/*', '*/*/*', '*/*/*/*', '*/*/*/*/*', '*/*/*/*/*/*']},
)
