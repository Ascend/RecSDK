#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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

compile_securec()
{
    if [[ ! -d ${CUR_DIR}/../platform/securec ]]; then
        echo "securec is not exist"
        exit 1
    fi

    if [[ ! -f ${CUR_DIR}/../platform/securec/lib/libsecurec.so ]]; then
        cd ${CUR_DIR}/../platform/securec/src
        make -j
    fi
}
compile_securec

cd ${CUR_DIR}
cd ../src/

find ./ -name "*.sh" -exec dos2unix {} \;
find ./ -name "*.sh" -exec chmod +x {} \;

rm -rf build

mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=Debug \
    -DTF_PATH=/opt/buildtools/python-3.7.5/lib/python3.7/site-packages/tensorflow_core \
    -DOMPI_PATH=/usr/local/openmpi/ \
    -DPYTHON_PATH=/opt/buildtools/python-3.7.5/ \
    -DEASY_PROFILER_PATH=/opt/buildtools/ \
    -DASCEND_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    -DABSEIL_PATH=${PWD}/../../install/abseil/ \
    -DSECUREC_PATH=${CUR_DIR}/../platform/securec \
    -DBUILD_TESTS=on -DCOVERAGE=on ../

make -j
make install

# Run Test
mpirun -np 4 ./tests/test_main

cd ../

COVERAGE_FILE=coverage.info
REPORT_FOLDER=coverage_report
lcov --rc lcov_branch_coverage=1 -c -d build -o ${COVERAGE_FILE}_tmp
lcov --rc lcov_branch_coverage=1  -e ${COVERAGE_FILE}_tmp "*src*" -o ${COVERAGE_FILE}
genhtml --rc genhtml_branch_coverage=1 ${COVERAGE_FILE} -o ${REPORT_FOLDER}
rm -rf ${COVERAGE_FILE}_tmp
rm -rf ${COVERAGE_FILE}

if [[ "$OSTYPE" == "darwin"* ]]; then
    open ./${REPORT_FOLDER}/index.html
fi