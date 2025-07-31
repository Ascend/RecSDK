#!/usr/bin/env bash
/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

set -e

SCRIPT_PATH=$(cd $(dirname $0); pwd)

function check_ret_fn()
{
    if [ $? -ne 0 ]; then
        echo "[FAIL] $@ failed"
        exit 1
    else
        echo "[SUCCESS] $@ successful"
    fi 
}

function build_with_cmake_func()
{
    mkdir -p ${SCRIPT_PATH}/src/cmake_build
    cd ${SCRIPT_PATH}/src/cmake_build
    cmake ../ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${SCRIPT_PATH}/cmake_build/install

    make -j8
    make install

    check_ret_fn "build torchrec_embcache"
}

build_with_cmake_func
