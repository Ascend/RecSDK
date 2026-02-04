#!/usr/bin/env bash
# -*- coding: utf-8 -*-
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

SCRIPT_DIR=$(cd $(dirname $0) && pwd)
PROJECT_ROOT=$(cd ${SCRIPT_DIR}/../../../../ && pwd)

TARGET_OP_BUILD_DIR="${PROJECT_ROOT}/dynamic_emb/build"
TARGET_OP_SO_DEST="${PROJECT_ROOT}/dynamic_emb/so"

COV_OUTPUT_DIR="${SCRIPT_DIR}/coverage_report"
TEST_FILE_PATH="${SCRIPT_DIR}"

RUN_MODE="${RUN_MODE:-npu}"
SOC_VERSION="${SOC_VERSION:-Ascend910_9579}"
ASCEND_CANN_PATH="${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"

PYBIND11_DIR=$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')

EXCLUDE_PATTERNS=(
    "/usr/*"               
    "${PYBIND11_DIR%/cmake}/*" 
    "${ASCEND_CANN_PATH}/*"
    "${COV_OUTPUT_DIR}/*"
)

init_workspace(){
    if [ -d "${COV_OUTPUT_DIR}" ]; then
        rm -rf "${COV_OUTPUT_DIR}"
    fi
    mkdir -p "${COV_OUTPUT_DIR}"

    if [ -d "${TARGET_OP_BUILD_DIR}" ]; then
        rm -rf "${TARGET_OP_BUILD_DIR}"
    fi
    mkdir -p "${TARGET_OP_BUILD_DIR}"

    if [ -d "${TARGET_OP_SO_DEST}" ]; then
        rm -rf "${TARGET_OP_SO_DEST}"
    fi
    mkdir -p "${TARGET_OP_SO_DEST}" 

    if [ -f "${TARGET_OP_SO_DEST}/dynamic_emb_extensions*.so" ]; then
        rm -f "${TARGET_OP_SO_DEST}/dynamic_emb_extensions*.so"
    fi
}


build_with_coverage(){
    cd "${TARGET_OP_BUILD_DIR}"

    cmake "${PROJECT_ROOT}/dynamic_emb/" \
        -DCMAKE_CXX_FLAGS="-O0 -g -fPIC" \
        -DCMAKE_C_FLAGS="-O0 -g -fPIC" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${TARGET_OP_BUILD_DIR}" \
        -DPYTHON_EXECUTABLE="$(which python3)" \
        -Dpybind11_DIR="${PYBIND11_DIR}" \
        -DCMAKE_PREFIX_PATH="${PYBIND11_DIR}" \
        -DRUN_MODE="${RUN_MODE}" \
        -DSOC_VERSION="${SOC_VERSION}" \
        -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PATH}" \
        -DENABLE_COVERAGE=ON
    
    make clean
    make dynamic_emb_extensions -j$(nproc)
    cp -f "${TARGET_OP_BUILD_DIR}/dynamic_emb_extensions"*.so "${TARGET_OP_SO_DEST}/"
    cd "${SCRIPT_DIR}"
}

init_workspace
build_with_coverage

export PYTHONPATH="${TARGET_OP_SO_DEST}:${PYTHONPATH}"

pytest "${TEST_FILE_PATH}" \
    -v \
    -s \
    --tb=short \
    --disable-warnings


collect_coverage(){
    lcov --capture \
        --directory "${TARGET_OP_BUILD_DIR}" \
        --output-file "${COV_OUTPUT_DIR}/coverage_raw.info" \
        --rc lcov_branch_coverage=1 \
        --gcov-tool gcov
}

filter_coverage(){
    EXCLUDE_ARGS=()
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
        EXCLUDE_ARGS+=("--remove" "${COV_OUTPUT_DIR}/coverage_raw.info" "${pattern}")
    done
    lcov "${EXCLUDE_ARGS[@]}" \
        --output-file "${COV_OUTPUT_DIR}/coverage_filtered.info" \
        --rc lcov_branch_coverage=1
}

collect_coverage
filter_coverage

generate_report(){
    genhtml "${COV_OUTPUT_DIR}/coverage_filtered.info" \
        --output-directory "${COV_OUTPUT_DIR}/html" \
        --title "Dynamic Emb Op Coverage" \
        --show-details \
        --legend \
        --rc lcov_branch_coverage=1
}

generate_report

rm -rf "${TARGET_OP_SO_DEST}"
rm -rf "${TARGET_OP_BUILD_DIR}"
