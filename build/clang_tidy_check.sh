#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
check_version=$1

if [ "${check_version}" != "tf_v1" ] && [ "${check_version}" != "tf_v2" ]; then
    echo "Error: The argument must be 'tf_v1' or 'tf_v2', but received '${check_version}'"
    exit 1
fi

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")   # .../RecSDK/build/
MxRec_DIR=$(dirname "${SCRIPT_DIR}")

PROJECT_COMMON_DIR="${MxRec_DIR}"/training/common
PROJECT_TFV1_DIR="${MxRec_DIR}"/training/tf_rec_v1
PROJECT_TFV2_DIR="${MxRec_DIR}"/training/tf_rec_v2
acc_ctr_path="${PROJECT_TFV1_DIR}"/src/AccCTR

function marking_rule() {
    # 禁用规则
    sed -i '/^  google-build-using-namespace/s/^  /  -/' .clang-tidy
    sed -i '/^  cppcoreguidelines-pro-type-reinterpret-cast/s/^  /  -/' .clang-tidy
    sed -i '/^  cppcoreguidelines-special-member-functions/s/^  /  -/' .clang-tidy
    sed -i '/^  cppcoreguidelines-owning-memory/s/^  /  -/' .clang-tidy
}

function clean_log() {
    rm -rf /tmp/clang_tidy*.log
}

#################### clang-tidy 检查 ####################
clean_log

# 检查 common
echo "==================  clang-tidy check common dir  ======================"
cd "${PROJECT_COMMON_DIR}"/src/
cp "${MxRec_DIR}"/.clang-tidy ./
marking_rule
rm -rf build
bash ./build.sh "${MxRec_DIR}" "Yes"

find ./core ./pybind \( -name "*.cpp" -o -name "*.h" \) -print | \
    xargs -P4 -I{} sh -c '
        echo "Check file: '"${PROJECT_COMMON_DIR}"'/src/$1"
        clang-tidy -p=build \
            --extra-arg=-Wno-ignored-optimization-argument \
            --extra-arg=-Wno-unused-command-line-argument \
            "$1" 2>&1
    ' _ {} | tee /tmp/clang_tidy_common.log

echo "==================  common dir check completed ======================"

if [ "${check_version}" == "tf_v1" ]; then
    # 检查 AccCTR
    echo "==================  clang-tidy check AccCTR dir  ======================"
    cd "${acc_ctr_path}"
    cp "${MxRec_DIR}"/.clang-tidy ./
    marking_rule
    cd build && find . -maxdepth 1 ! -name "*.sh" ! -name "." -exec rm -rf {} + && cd -
    bash ./build.sh "release"

    find ./src \( -name "*.cpp" -o -name "*.h" \) -print | \
        xargs -P4 -I{} sh -c '
            echo "Check file: '"${acc_ctr_path}"'/src/$1"
            clang-tidy -p=build \
                --extra-arg=-Wno-ignored-optimization-argument \
                --extra-arg=-Wno-unused-command-line-argument \
                "$1" 2>&1
        ' _ {} | tee tee /tmp/clang_tidy_AccCTR.log
    echo "==================  AccCTR dir check completed ======================"

    # 检查tf_rec_v1
    echo "==================  clang-tidy check tf_rec_v1 dir  ======================"
    cd "${PROJECT_TFV1_DIR}"/src/
    cp "${MxRec_DIR}"/.clang-tidy ./
    marking_rule
    rm -rf build
    tf1_path=$(dirname "$(dirname "$(which python3.7)")")/lib/python3.7/site-packages/tensorflow_core
    bash build.sh "${tf1_path}" "${MxRec_DIR}" "YES"

    find ./ \
        \( -path "*/build" -o -path "*/AccCTR" -o -path "*/tests" \) -prune -o \
        \( -name "*.cpp" -o -name "*.h" \) -print | \
        xargs -P4 -I{} sh -c '
            echo "Check file: '"${PROJECT_TFV1_DIR}"'/src/$1"
            clang-tidy -p=build \
                --extra-arg=-Wno-ignored-optimization-argument \
                --extra-arg=-Wno-unused-command-line-argument \
                "$1" 2>&1
        ' _ {} | tee tee /tmp/clang_tidy_tfv1.log
    echo "==================  tf_rec_v1 dir check completed ======================"
else
    # 检查tf_rec_v2
    echo "==================  clang-tidy check tf_rec_v2 dir  ======================"
    echo ""
    cd "${PROJECT_TFV2_DIR}"
    cp "${MxRec_DIR}"/.clang-tidy ./
    marking_rule
    rm -rf build
    bash ci/scripts/build_tf1.sh

    find ./mxrec/core/host/ \( -name "*.cpp" -o -name "*.h" \) -print | \
        xargs -P4 -I{} sh -c '
            echo "Check file: '"${PROJECT_TFV2_DIR}"'/$1"
            clang-tidy -p=build \
                --extra-arg=-Wno-ignored-optimization-argument \
                --extra-arg=-Wno-unused-command-line-argument \
                "$1" 2>&1
        ' _ {} | tee /tmp/clang_tidy_tfv2.log
    echo "==================  tf_rec_v2 dir check completed ======================"
fi

# 统计不同类型的错误
compiler_errors=$(grep "error:.*\[clang-diagnostic-" /tmp/clang_tidy*.log 2>/dev/null | wc -l)
warning_errors=$(grep "error:.*\[-warnings-as-errors\]" /tmp/clang_tidy*.log 2>/dev/null | wc -l)

echo ""
echo "========================================"
echo "Error Statistics:"
echo "  - compiler_errors: ${compiler_errors}"
echo "  - warnings-as-errors: ${warning_errors}"

if [ "${compiler_errors}" -gt 0 ] || [ "${warning_errors}" -gt 0 ]; then
    echo "RESULT: FAIL"
    echo ""
    echo "Specific error:"
    grep "error:" /tmp/clang_tidy*.log 2>/dev/null | head -30
    echo "========================================"
    exit 1
else
    echo "RESULT: PASS"
    echo "========================================"
    exit 0
fi
