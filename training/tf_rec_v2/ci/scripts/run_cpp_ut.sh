#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

if [ -z "$1" ]; then
    echo "Usage: $0 <tf_version>"
    exit 1
fi

rec_package_path=$(dirname "$(dirname "$(which python3.7)")")/lib/python3.7/site-packages/mxrec
so_path=${rec_package_path}/librec
common_package_path=$(dirname "$(dirname "$(which python3.7)")")/lib/python3.7/site-packages/rec_sdk_common
common_so_path=${common_package_path}/lib
export PYTHONPATH=${so_path}:${common_so_path}:$PYTHONPATH
export LD_LIBRARY_PATH=${so_path}:${common_so_path}:/usr/local/lib:$LD_LIBRARY_PATH

python_home="$(dirname "$(dirname "$(readlink -f "$(which python3.7)")")")"
echo "python_home=$python_home"
echo "ASCEND_TOOLKIT_HOME=$ASCEND_TOOLKIT_HOME"
tf_version=$1

scripts_dir=$(dirname "$(readlink -f "$0")")
project_dir=$(dirname "$(dirname "${scripts_dir}")")
build_dir=${scripts_dir}/build
emock_src_dir=${project_dir}/third_party/emock
emock_build_dir=${emock_src_dir}/build
num_cores=$(($(nproc)-1))

echo "scripts_dir=${scripts_dir}"
echo "project_dir=${project_dir}"

build_emock(){
    cmake -S "${emock_src_dir}" -B "${emock_build_dir}" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 -Wno-unused-function -Wno-sign-compare"

    cmake --build "${emock_build_dir}" --config Debug --parallel ${num_cores}
}
clean_emock(){
    if [ -d ${emock_build_dir} ]; then
        rm -rf ${emock_build_dir}
    fi
}
build_emock

if [ -d /usr/local/Ascend/ascend-toolkit/latest ]; then
    ascend_toolkit_home=/usr/local/Ascend/ascend-toolkit/latest
elif [ -d /usr/local/Ascend/latest ]; then
    ascend_toolkit_home=/usr/local/Ascend/latest
else
    echo "ERROR: can not find toolkit and tfplugin"
    exit 1
fi
cmake -S "$project_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DPYTHON_HOME="$python_home" \
    -DASCEND_TOOLKIT_HOME="$ascend_toolkit_home" \
    -DTF_VERSION="$tf_version"

cmake --build "$build_dir" --config Debug --parallel $num_cores

cd "$build_dir"/mxrec/core/tests
./runTests --gtest_output=xml:test_detail.xml
cd -

COVERAGE_FILE=coverage.info
REPORT_FOLDER=coverage_report
lcov --rc lcov_branch_coverage=1 -c -d ./build -o "${COVERAGE_FILE}" \
    --exclude "${project_dir}/third_party/*" \
    --exclude "${project_dir}/mxrec/core/tests/*" \
    --exclude "/usr/*" \
    --exclude "/opt/*"

genhtml --rc genhtml_branch_coverage=1 "${COVERAGE_FILE}" -o "${REPORT_FOLDER}"

[ -e "${COVERAGE_FILE}" ] && rm -rf "${COVERAGE_FILE}"
clean_emock