#!/bin/bash
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

function check_ret_fn()
{
    if [ $? -ne 0 ]; then
        echo "[FAIL] $@ failed"
        exit -1
    else
        echo "[SUCCESS] $@ successful"
    fi 
}

function build_with_cmake_func()
{
  bash zbuild_cmake.sh
  check_ret_fn "bash zbuild_cmake.sh"
}

function build_whl_pkg_with_setup_func()
{
  rm -rf build
  rm -rf dist
  rm -rf *.egg-info

  rm -f src/torchrec_embcache/*.so*
  cp 3rdparty/securec/lib/libsecurec.so src/torchrec_embcache/
  cp cmake_build/install/embcache_pybind.so src/torchrec_embcache/
  cp -a cmake_build/install/lib64/*.so* src/torchrec_embcache/   # for glog/gtest

  python3 zsetup.py bdist_wheel
  check_ret_fn "python3 zsetup.py bdist_wheel"
}

build_with_cmake_func
build_whl_pkg_with_setup_func