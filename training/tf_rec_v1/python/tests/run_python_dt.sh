#!/bin/bash

# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

set -e

CUR_PATH=$(cd "$(dirname "$0")" || { warn "Failed to check path/to/run_python_dt.sh" ; exit ; } ; pwd)
TOP_PATH="${CUR_PATH}"/../../../../

ARCH="$(uname -m)"
if [ $ARCH == "aarch64" ]; then
  gomp_lib="$(gcc -print-file-name=libgomp.so.1)"
  if [ ! -f "$gomp_lib" ]; then
    for candidate in /usr/local/gcc11.2.0/lib64/libgomp.so.1 /usr/local/gcc7.3.0/lib64/libgomp.so.1; do
      if [ -f "$candidate" ]; then
        gomp_lib="$candidate"
        break
      fi
    done
  fi
  [ -f "$gomp_lib" ] && export LD_PRELOAD="$gomp_lib"
fi

# build Rec SDK and get output directory
bash "$TOP_PATH"/build/build_tf1.sh

# so 目录与模型运行时（site-packages/mx_rec/libasc、rec_sdk_common/lib）同构：
# build_tf1.sh 已将全部交付 so（tf1 各模块 install 产物、AccCTR、tf_plugin、securec）
# 收集至包内 libasc / lib；禁用 RPATH 后，so 依赖统一由 LD_LIBRARY_PATH 定位
so_path="${TOP_PATH}"/training/tf_rec_v1/python/libasc
common_so_path="${TOP_PATH}"/training/common/python/lib

# Make the pure-Python packages importable without a pre-installed wheel.
# mx_rec ships as tf_rec_v1/python and rec_sdk_common as common/python (see their
# setup.py which copies "python" to the package name), so alias them in-tree.
ln -sfn python "$TOP_PATH"/training/tf_rec_v1/mx_rec
ln -sfn python "$TOP_PATH"/training/common/rec_sdk_common

# set environment variable
export PYTHONPATH="${TOP_PATH}"/training/tf_rec_v1:"${TOP_PATH}"/training/common:${so_path}:${common_so_path}:"${TOP_PATH}":$PYTHONPATH
export LD_LIBRARY_PATH=${so_path}:${common_so_path}:/usr/local/lib:$LD_LIBRARY_PATH

rm -rf result
mkdir -p result

function run_test_cases() {
    echo "Get testcases final result."
    pytest --cov="${CUR_PATH}"/../ --cov-report=html --cov-report=xml --junit-xml=./final.xml --html=./final.html --self-contained-html --durations=5 -vv --cov-branch
    coverage xml -i --omit="build/*,cust_op/*,src/*"
    cp coverage.xml final.xml final.html ./result
    cp -r htmlcov ./result
    rm -rf coverage.xml final.xml final.html htmlcov
}

echo "************************************* Start Rec SDK LLT Test *************************************"
start=$(date +%s)
run_test_cases
ret=$?
end=$(date +%s)
echo "*************************************  End  Rec SDK LLT Test *************************************"
echo "LLT running take: $(expr "${end}" - "${start}") seconds"

rm -rf "$TOP_PATH"/training/tf_rec_v1/mx_rec
rm -rf "$TOP_PATH"/training/common/rec_sdk_common

exit "${ret}"
