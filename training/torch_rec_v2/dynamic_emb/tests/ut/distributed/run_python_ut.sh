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

dynamic_package_path=$(dirname "$(dirname "$(which python3.11)")")/lib/python3.11/site-packages/dynamic_emb
so_path=$(dirname ${dynamic_package_path})
common_package_path=$(dirname "$(dirname "$(which python3.11)")")/lib/python3.11/site-packages/rec_sdk_common
common_so_path=${common_package_path}/lib
export PYTHONPATH=${so_path}:${common_so_path}:$PYTHONPATH
export LD_LIBRARY_PATH=${so_path}:${common_so_path}:/usr/local/lib:$LD_LIBRARY_PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
INCREMENTAL_DUMP_DIR="${SCRIPT_DIR}/incremental_dump"
RESULT_DIR="${SCRIPT_DIR}/result"
COVERAGERC="${SCRIPT_DIR}/.coveragerc"

if [ -d "${RESULT_DIR}" ]; then
    rm -rf "${RESULT_DIR}"
fi
mkdir -p "${RESULT_DIR}"

pytest "${SCRIPT_DIR}" \
    --ignore="${INCREMENTAL_DUMP_DIR}" \
    --cov="${dynamic_package_path}" \
    --cov-config "${COVERAGERC}" \
    --cov-report=term \
    --junit-xml="${SCRIPT_DIR}/final.xml" \
    --html="${SCRIPT_DIR}/final.html" --self-contained-html --durations=5 -vv --cov-branch

# 执行incremental_dump目录下的测试
INCREMENTAL_DUMP_RUNNER="${INCREMENTAL_DUMP_DIR}/test_incremental_dump.sh"
export COVERAGE_RCFILE="${COVERAGERC}"
export COVERAGE_FILE="${SCRIPT_DIR}/.coverage"
export DYNAMIC_EMB_COV_TARGET="${dynamic_package_path}"
export RUN_INCREMENTAL_WITH_COVERAGE=1

if [ ! -f "${INCREMENTAL_DUMP_RUNNER}" ]; then
    echo "error: missing ${INCREMENTAL_DUMP_RUNNER}" >&2
    exit 1
fi
(cd "${INCREMENTAL_DUMP_DIR}" && bash ./test_incremental_dump.sh)

# 合并coverage数据
coverage combine "${SCRIPT_DIR}"
coverage html -d "${SCRIPT_DIR}/htmlcov"
coverage xml -i --omit="*/tests/*" -o "${SCRIPT_DIR}/coverage.xml"

cp "${SCRIPT_DIR}/coverage.xml" "${SCRIPT_DIR}/final.xml" "${SCRIPT_DIR}/final.html" "${RESULT_DIR}/"
cp -r "${SCRIPT_DIR}/htmlcov" "${RESULT_DIR}/"
rm -rf "${SCRIPT_DIR}/coverage.xml" "${SCRIPT_DIR}/final.xml" "${SCRIPT_DIR}/final.html" "${SCRIPT_DIR}/htmlcov" \
    "${SCRIPT_DIR}/.coverage"
find "${SCRIPT_DIR}" -maxdepth 1 -name '.coverage.*' -exec rm -f {} +
