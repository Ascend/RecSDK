#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
# Description: build script.
# Author: MindX SDK
# Create: 2022
# History: NA
set -e
[ -d build ] && rm -rf build;
mkdir build && cd build || exit 1
# HDF5_PATH, EASY_PROFILER_PATH is optional
python_path="$(dirname "$(dirname "$(realpath "$(which python3.7)")")")"
if [ -d /usr/local/Ascend/ascend-toolkit/latest ]; then
    ascend_path=/usr/local/Ascend/ascend-toolkit/latest
elif [ -d /usr/local/Ascend/latest ]; then
    ascend_path=/usr/local/Ascend/latest
else
    echo "ERROR: can not find toolkit and tfplugin"
    exit 1
fi

cmake -DCMAKE_BUILD_TYPE=Release \
    -DTF_PATH="$1" \
    -DOMPI_PATH="$(whereis openmpi)" \
    -DPYTHON_PATH="$python_path" \
    -DEASY_PROFILER_PATH=/ \
    -DASCEND_PATH="$ascend_path" \
    -DABSEIL_PATH="$python_path"/lib/python3.7/site-packages/tensorflow_core/ \
    -DSECUREC_PATH="$2"/platform/securec \
    -DCMAKE_INSTALL_PREFIX="$2"/output \
    -DBUILD_CUST="$3" ..
make -j
make install
