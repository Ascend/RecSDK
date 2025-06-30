#!/bin/bash
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e
warn() { echo >&2 -e "\033[1;31m[WARN ][Depend  ] $1\033[1;37m" ; }

script_path=$(cd $(dirname $0); pwd)

#---------------------------------------
# build
#---------------------------------------
mkdir -p src/cmake_build
cd src/cmake_build
cmake ../ \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${script_path}/cmake_build/install

make -j8
make install