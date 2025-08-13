#!/usr/bin/env bash
# Copyright (c) huawei Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

SCRIPT_PATH=$(cd $(dirname $0); pwd)
version_file="${SCRIPT_PATH}/version.txt"

if [ ! -f "${version_file}" ]; then
  VERSION="1.0.0"
else
  VERSION=$(head -n 1 "${version_file}")
fi
ARCH=$(uname -m)

package_name="Ascend-mindxsdk-torchrec-embcache-"${VERSION}"-linux-"${ARCH}".tar.gz"
if [ -f "${package_name}" ]; then
  rm "${package_name}"
fi


function prepare_deps()
{
    python3 -m pip install pybind11
    cd "${SCRIPT_PATH}/src/3rdparty"
    git clone -b master https://gitee.com/Janisa/huawei_secure_c.git securec
    cd -
}

function check_ret_fn()
{
    if [ $? -ne 0 ]; then
        echo "[FAIL] $@ failed" 1>&2
        exit 1
    else
        echo "[SUCCESS] $@ successful" 
    fi 
}

function build_with_cmake_func()
{
    mkdir -p ${SCRIPT_PATH}/src/cmake_build
    cd ${SCRIPT_PATH}/src/cmake_build
    torch_path=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
    cmake ../ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${torch_path}"


    make -j8
    make install

    check_ret_fn "build torchrec_embcache"
    cd -
}

function build_whl_pkg_with_setup_func()
{
    rm -rf build
    rm -rf dist
    rm -rf *.egg-info

    rm -f src/torchrec_embcache/*.so*
    cp src/cmake_build/torchrec_embcache/csrc/embcache_pybind.so src/torchrec_embcache/
    # cp src/3rdparty/securec/lib/libsecurec.so src/torchrec_embcache/

    python3 setup.py bdist_wheel --plat-name linux_"${ARCH}"
    check_ret_fn "python3 setup.py bdist_wheel"
}

function build_tar_pkg_func()
{
    cp "${SCRIPT_PATH}/requirements.txt" dist/
    tar -czvf "${package_name}" -C dist .
    check_ret_fn "tar gz ${package_name}"
}

prepare_deps
build_with_cmake_func
build_whl_pkg_with_setup_func
build_tar_pkg_func
