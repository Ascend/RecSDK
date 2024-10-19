#!/bin/bash
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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

TF_VERSION=$1
if [ "${TF_VERSION}" == "tf1" ]; then
    TF_DIR=tensorflow_core
elif [ "${TF_VERSION}" == "tf2" ];then
    TF_DIR=tensorflow
else
    echo "TF_VERSION should be tf1 or tf2"
fi

# add mpirun env
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

source /etc/profile
source /opt/rh/devtoolset-7/enable

CUR_DIR=$(dirname "$(readlink -f "$0")")
ROOT_DIR=$(dirname "${CUR_DIR}")
opensource_path="${ROOT_DIR}"/../opensource
acc_ctr_path="${ROOT_DIR}"/src/AccCTR
export LD_LIBRARY_PATH="${acc_ctr_path}"/output/ock_ctr_common/lib:$LD_LIBRARY_PATH


function prepare_googletest(){
  cd ${opensource_path}
  if [ ! -d googletest-release-1.8.1 ]; then
    unzip googletest-release-1.8.1.zip
  fi
  cd googletest-release-1.8.1
  if [ ! -d build ]; then
    mkdir build
  fi
  cd build
  rm -f CMakeCache.txt
  cmake -DBUILD_SHARED_LIBS=ON ..
  make -j8
  make install
}

function prepare_emock(){
  cd ${opensource_path}
  if [ ! -d emock-0.9.0 ]; then
    unzip emock-0.9.0.zip
  fi
  cd emock-0.9.0
  if [ ! -d build ]; then
    mkdir build
  fi
  cd build
  rm -f CMakeCache.txt
  cmake ..
  make -j8
  make install
}

function prepare_securec(){
  cd "${opensource_path}"
  if [ ! -d securec ]; then
    unzip huaweicloud-sdk-c-obs-3.23.9.zip
    mv huaweicloud-sdk-c-obs-3.23.9/platform/huaweisecurec securec
    rm -rf huaweicloud-sdk-c-obs-3.23.9
    rm -rf securec/lib/*
  fi
}

function compile_securec(){
  cd ${opensource_path}
    if [[ ! -d "${opensource_path}/securec" ]]; then
        echo "securec is not exist"
        exit 1
    fi

    if [[ ! -f "${opensource_path}/securec/lib/libsecurec.so" ]]; then
        cd "${opensource_path}/securec/src"
        make -j4
    fi
}

function prepare_pybind(){
  cd "${opensource_path}"
  if [ ! -d pybind11 ]; then
    unzip pybind11-2.10.3.zip
    mv pybind11-2.10.3 pybind11
  fi
}

prepare_pybind
echo "opensource path:${opensource_path}"
prepare_googletest
prepare_emock
prepare_securec
compile_securec

compile_acc_ctr_so_file()
{
  cd "${acc_ctr_path}"
  chmod u+x build.sh
  ./build.sh "release"
}

echo "-----Build AccCTR -----"
compile_acc_ctr_so_file

cd "${ROOT_DIR}"/src

find ./ -name "*.sh" -exec dos2unix {} \;
find ./ -name "*.sh" -exec chmod +x {} \;

[ -d build ] && rm -rf build

mkdir build
cd build

python_path="$(dirname "$(dirname "$(which python3.7)")")"
# config asan environment variable
export ASAN_OPTIONS=halt_on_error=1:detect_leaks=1:fast_unwind_on_malloc=0
export LSAN_OPTIONS=suppressions=../tests/leaks.supp

cmake -DCMAKE_BUILD_TYPE=Debug \
    -DTF_PATH="${python_path}"/lib/python3.7/site-packages/"${TF_DIR}" \
    -DOMPI_PATH=/usr/local/openmpi/ \
    -DPYTHON_PATH="${python_path}" \
    -DEASY_PROFILER_PATH=/opt/buildtools/ \
    -DASCEND_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    -DABSEIL_PATH="${python_path}"/lib/python3.7/site-packages/"${TF_DIR}" \
    -DSECUREC_PATH="${ROOT_DIR}"/../opensource/securec \
    -DBUILD_TESTS=on -DCOVERAGE=on "$(dirname "${PWD}")"

make -j8
make install

# Run Test
DATE=$(date +%Y-%m-%d-%H-%M-%S)
if [[ "$1" == "--with-memcheck" ]]; then
  echo "we are going to run test_main with memcheck via valgrind"
  valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --log-file=../"memcheck_${DATE}.log" \
    ./tests/test_main 2>&1 |tee ../"test_main_${DATE}.log"
else
  mpirun -np 4 ./tests/test_main
fi

cd "$(dirname "${PWD}")"

COVERAGE_FILE=coverage.info
REPORT_FOLDER=coverage_report
lcov --rc lcov_branch_coverage=1 -c -d build -o "${COVERAGE_FILE}"_tmp
lcov -r "${COVERAGE_FILE}"_tmp 'ut/*' '/usr1/mxRec/src/core/hybrid_mgmt*' '/usr1/mxRec/src/core/emb_table*' '/usr1/mxRec/src/core/host_emb*' '7/ext*' '*7/bits*' 'platform/*' '/usr/local/*' '/usr/include/*' '/opt/buildtools/python-3.7.5/lib/python3.7/site-packages/tensorflow*' 'tests/*' '/usr1/mxRec/src/core/ock_ctr_common/include*' --rc lcov_branch_coverage=1 -o "${COVERAGE_FILE}"
genhtml --rc genhtml_branch_coverage=1 "${COVERAGE_FILE}" -o "${REPORT_FOLDER}"
[ -d "${COVERAGE_FILE}"_tmp ] && rm -rf "${COVERAGE_FILE}"_tmp
[ -d "${COVERAGE_FILE}" ] && rm -rf "${COVERAGE_FILE}"

if [[ "$OSTYPE" == "darwin"* ]]; then
    open ./"${REPORT_FOLDER}"/index.html
fi
