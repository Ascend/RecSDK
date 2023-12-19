#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
# Description: NA
# Author: MindX SDK
# Create: 2022
# History: NA
set -e

# add mpirun env
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

source /etc/profile
source /opt/rh/devtoolset-7/enable

CUR_DIR=$(dirname "$(readlink -f "$0")")
ROOT_DIR=$(dirname "${CUR_DIR}")
acc_ctr_path="${ROOT_DIR}"/src/platform/AccCTR
cp -rf "${ROOT_DIR}"/platform/securec/* "${acc_ctr_path}"/3rdparty/huawei_secure_c
export LD_LIBRARY_PATH="${acc_ctr_path}"/output/ock_ctr_common/lib:$LD_LIBRARY_PATH

compile_securec()
{
    if [[ ! -d "${ROOT_DIR}"/platform/securec ]]; then
        echo "securec is not exist"
        exit 1
    fi

    if [[ ! -f "${ROOT_DIR}"/platform/securec/lib/libsecurec.so ]]; then
        cd "${ROOT_DIR}"/platform/securec/src
        make -j
    fi
}
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

cmake -DCMAKE_BUILD_TYPE=Debug \
    -DTF_PATH="$(dirname "$(dirname "$(which python3.7)")")"/lib/python3.7/site-packages/tensorflow_core \
    -DOMPI_PATH=/usr/local/openmpi/ \
    -DPYTHON_PATH="$(dirname "$(dirname "$(which python3.7)")")" \
    -DEASY_PROFILER_PATH=/opt/buildtools/ \
    -DASCEND_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    -DABSEIL_PATH="$python_path"/lib/python3.7/site-packages/tensorflow_core/ \
    -DSECUREC_PATH="${ROOT_DIR}"/platform/securec \
    -DBUILD_TESTS=on -DCOVERAGE=on "$(dirname "${PWD}")"

make -j
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
lcov -r "${COVERAGE_FILE}"_tmp 'ut/*' '/usr1/mxRec/src/core/hybrid_mgmt*' '/usr1/mxRec/src/core/host_emb*' '7/ext*' '*7/bits*' 'platform/*' '/usr/local/*' '/usr/include/*' '/opt/buildtools/python-3.7.5/lib/python3.7/site-packages/tensorflow*' 'tests/*' '/usr1/mxRec/src/core/ock_ctr_common/include*' --rc lcov_branch_coverage=1 -o "${COVERAGE_FILE}"
genhtml --rc genhtml_branch_coverage=1 "${COVERAGE_FILE}" -o "${REPORT_FOLDER}"
[ -d "${COVERAGE_FILE}"_tmp ] && rm -rf "${COVERAGE_FILE}"_tmp
[ -d "${COVERAGE_FILE}" ] && rm -rf "${COVERAGE_FILE}"

if [[ "$OSTYPE" == "darwin"* ]]; then
    open ./"${REPORT_FOLDER}"/index.html
fi
