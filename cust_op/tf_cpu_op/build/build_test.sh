#!/bin/bash
# build test
# Copyright © Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

set -e
readonly CURRENT_PATH=$(cd "$(dirname "$0")"; pwd)
DATA_PATH=${CURRENT_PATH}/../data
BUILD_PATH=${CURRENT_PATH}/tests
UT_PATH=${CURRENT_PATH}/tests/ut
CPU_TYPE=$(arch)
BUILD_MODE=$1

ut_cover()
{
    cd ${UT_PATH}/src
    ./test_rec_base --gtest_output=xml:./
    GENERATE_DIR=${BUILD_PATH}/cov/gen
    rm -rf ${BUILD_PATH}/cov/; mkdir -p ${GENERATE_DIR}

    echo "================"${CURRENT_PATH}
    find ${CURRENT_PATH}/.. -name "*.gcda" | xargs  -i mv {} ${GENERATE_DIR}
    find ${CURRENT_PATH}/.. -name "*.gcno" | xargs  -i mv {} ${GENERATE_DIR}

    lcov --d ${GENERATE_DIR} --c --output-file ${GENERATE_DIR}/coverage.info --rc lcov_branch_coverage=1
    if [ 0 != $? ];then
      echo "Failed to generate all coverage info"
      exit 1
    fi

    lcov -r ${GENERATE_DIR}/coverage.info "*7.3.0*" -o ${GENERATE_DIR}/coverage.info --rc lcov_branch_coverage=1
    if [ 0 != $? ];then
      echo "Failed to remove *7.3.0* from coverage info"
      exit 1
    fi

    lcov -r ${GENERATE_DIR}/coverage.info "*tests/ut*" -o ${GENERATE_DIR}/coverage.info --rc lcov_branch_coverage=1
    if [ 0 != $? ];then
      echo "Failed to remove *tests/ut* from coverage info"
      exit 1
    fi

    lcov -r ${GENERATE_DIR}/coverage.info "*install*" -o ${GENERATE_DIR}/coverage.info --rc lcov_branch_coverage=1
    if [ 0 != $? ];then
      echo "Failed to remove *install* from coverage info"
      exit 1
    fi

    genhtml -o ${GENERATE_DIR}/result ${GENERATE_DIR}/coverage.info --show-details --legend --rc lcov_branch_coverage=1
    if [ 0 != $? ];then
      echo "Failed to generate all coverage info with html format"
      exit 1
    fi
}

if [ "${BUILD_MODE}" == "ut" ]; then
    ut_cover
    if [ 0 != $? ];then
          echo "Failed to ut_cover"
          exit 1
    fi
elif [ "${BUILD_MODE}" == "debug" ];then
    echo "BUILD_MODE ${BUILD_MODE} skip"
elif [ "${BUILD_MODE}" == "release" ];then
    echo "BUILD_MODE "${BUILD_MODE}" skip"
else
    echo "BUILD_MODE "${BUILD_MODE}" not exists"
fi