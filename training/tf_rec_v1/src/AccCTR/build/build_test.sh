#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# build test
set -e
readonly CURRENT_PATH=$(cd "$(dirname "$0")"; pwd)
DATA_PATH=${CURRENT_PATH}/../data
BUILD_PATH=${CURRENT_PATH}/tests
TOOL_PATH=${CURRENT_PATH}/../tests/tools
UT_PATH=${CURRENT_PATH}/tests/ut
TOOL_FILE="create_fake_id.py"
CPU_TYPE=$(arch)
BUILD_MODE=$1

# config asan environment variable
export ASAN_OPTIONS=halt_on_error=1:detect_leaks=1

create_data()
{
    cd ${TOOL_PATH}
    python3 $TOOL_FILE
}

ut_cover()
{
    cd ${UT_PATH}/src
    scp -r ${TOOL_PATH}/*.txt .
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CURRENT_PATH}/src
    ./test_unique_files --gtest_output=xml:./
    GENERATE_DIR=${BUILD_PATH}/cov/gen
    rm -rf ${BUILD_PATH}/cov/; mkdir -p ${GENERATE_DIR}

    echo "================"${CURRENT_PATH}
    find ${CURRENT_PATH}/.. -name "*.gcda" | xargs  -i mv {} ${GENERATE_DIR}
    find ${CURRENT_PATH}/.. -name "*.gcno" | xargs  -i mv {} ${GENERATE_DIR}

    lcov --d ${GENERATE_DIR} --c --output-file ${GENERATE_DIR}/coverage.info --rc lcov_branch_coverage=1 --filter branch
    if [ 0 != $? ];then
      echo "Failed to generate all coverage info"
      exit 1
    fi

    lcov -r ${GENERATE_DIR}/coverage.info "*7.3.0*" -o ${GENERATE_DIR}/coverage.info --rc lcov_branch_coverage=1 --filter branch
    if [ 0 != $? ];then
      echo "Failed to remove *7.3.0* from coverage info"
      exit 1
    fi

    lcov -r ${GENERATE_DIR}/coverage.info "*tests/ut*" -o ${GENERATE_DIR}/coverage.info --rc lcov_branch_coverage=1 --filter branch
    if [ 0 != $? ];then
      echo "Failed to remove *tests/ut* from coverage info"
      exit 1
    fi

    lcov -r ${GENERATE_DIR}/coverage.info "*install*" -o ${GENERATE_DIR}/coverage.info --rc lcov_branch_coverage=1 --filter branch
    if [ 0 != $? ];then
      echo "Failed to remove *install* from coverage info"
      exit 1
    fi

    genhtml -o ${GENERATE_DIR}/result ${GENERATE_DIR}/coverage.info --show-details --legend --rc lcov_branch_coverage=1 --filter branch
    if [ 0 != $? ];then
      echo "Failed to generate all coverage info with html format"
      exit 1
    fi
}

if [ "${BUILD_MODE}" == "ut" ]; then
    create_data
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