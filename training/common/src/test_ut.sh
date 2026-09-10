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

# add mpirun env
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Remove CC env, and subsequent compilation uses CC in devtoolset.
unset CC
source /etc/profile
source /opt/rh/devtoolset-7/enable

CUR_DIR=$(dirname "$(readlink -f "$0")")
ROOT_DIR=$(dirname "${CUR_DIR}")
opensource_path="${ROOT_DIR}"/../../../opensource
# add asan lib path
export LIBRARY_PATH=${LIBRARY_PATH}:/usr/local/gcc7.3.0/lib64/

# Detect available CPUs once and reuse everywhere.
NUM_CPUS=$(nproc) || NUM_CPUS=32
if [ "${NUM_CPUS}" -lt 1 ]; then NUM_CPUS=1; fi
echo "Detected ${NUM_CPUS} CPUs"

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
  make -j${NUM_CPUS}
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
  make -j${NUM_CPUS}
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
        make -j${NUM_CPUS}
    fi
}

function prepare_pybind(){
  cd "${opensource_path}"
  if [ ! -d pybind11 ]; then
    rm -rf pybind11-*/
    zip_file=$(find . -maxdepth 1 -name "pybind11-*.zip" -type f -printf "%T@ %p\n" | sort -nr \
                  | head -n 1 | cut -d' ' -f2-)
    unzip "${zip_file}"
    mv pybind11-*/ pybind11
  fi
}

# Allow parent (tf1/test_ut.sh) to skip redundant dep preparation.
if [ -z "$DEPS_ALREADY_PREPARED" ]; then
  prepare_pybind
  echo "opensource path:${opensource_path}"
  prepare_googletest
  prepare_emock
  prepare_securec
  compile_securec
else
  echo "Dependencies already prepared by parent script, skipping"
  compile_securec
fi

cd "${ROOT_DIR}"/src

[ -d build ] && rm -rf build

mkdir build
cd build

python_path="$(dirname "$(dirname "$(which python3.7)")")"
# config asan environment variable
export ASAN_OPTIONS=halt_on_error=1:detect_leaks=1:fast_unwind_on_malloc=0
export LSAN_OPTIONS=suppressions=../tests/leaks.supp

cmake -DCMAKE_BUILD_TYPE=Debug \
    -DPYTHON_PATH="${python_path}" \
    -DASCEND_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    -DSECUREC_PATH="${ROOT_DIR}"/../../../opensource/securec \
    -DBUILD_TESTS=on -DCOVERAGE=on "$(dirname "${PWD}")"

make -j${NUM_CPUS}
make install

# Each mpirun -np 4 only actually runs 1 test at a time inside MPI, so
# running more parallel groups than NUM_CPUS/4 still keeps CPU within budget.
#
# (NUM_CPUS + 1) / 2 is an empirical heuristic: it assumes all NUM_CPUS logical
# CPUs are uniform (no hyper-threading, no mixed performance/efficiency cores).
# On non-uniform CPUs this may be sub-optimal - override via PARALLEL_JOBS.
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-8}
DEFAULT_JOBS=$(( (NUM_CPUS + 1) / 2 ))
if [ $DEFAULT_JOBS -lt 1 ]; then
  DEFAULT_JOBS=1
fi
if [ $DEFAULT_JOBS -gt $MAX_PARALLEL_JOBS ]; then
  DEFAULT_JOBS=$MAX_PARALLEL_JOBS
fi
PARALLEL_JOBS=${PARALLEL_JOBS:-$DEFAULT_JOBS}
echo "Detected ${NUM_CPUS} CPUs, running with ${PARALLEL_JOBS} parallel jobs"

# Google Test filter uses : to separate multiple patterns.
TEST_GROUPS=(
  "TestStringFormat.*"
  "TESTToString.*:InitializerTest.*"
  "LcalSockExchangeTest.*:LcalSockExchange.*:LcalCommTest.*"
  "Log.*"
  "TestGetShmAddr.*"
)

# PARALLEL_SCHEME: direct (no mpirun) or mpirun (-np 4 per group).
# direct removes the 4x redundant execution that mpirun introduces when
# tests don't actually use MPI_Comm_* (only MPI_Init/Finalize).
PARALLEL_SCHEME=${PARALLEL_SCHEME:-direct}

if [ "${PARALLEL_SCHEME}" = "direct" ]; then
  echo "=== direct parallel scheme (no mpirun) ==="
  TEST_LIST=/tmp/gtest_list_common_$$.txt
  ./tests/test_main --gtest_list_tests > "${TEST_LIST}" 2>/dev/null || true

  if [ -s "${TEST_LIST}" ]; then
    DIRECT_GROUPS_FILE=/tmp/gtest_groups_common_$$.txt
    awk '
      /^[A-Za-z_]/ {
        current=$0
        sub(/^[^A-Za-z0-9_]+/, "", current)
        sub(/\.$/, "", current)
        next
      }
      /^  / {
        test=$1
        if (current != "" && test != "") {
          print current "." test
        }
      }
    ' "${TEST_LIST}" > /tmp/all_cases_common_$$.txt

    TOTAL_CASES=$(wc -l < /tmp/all_cases_common_$$.txt)
    if [ "${TOTAL_CASES}" -gt 0 ]; then
      CASES_PER_GROUP=$(( (TOTAL_CASES + NUM_CPUS - 1) / NUM_CPUS ))
      if [ "${CASES_PER_GROUP}" -lt 1 ]; then CASES_PER_GROUP=1; fi
      split -l "${CASES_PER_GROUP}" -d -a 3 /tmp/all_cases_common_$$.txt "${DIRECT_GROUPS_FILE}_"
      > "${DIRECT_GROUPS_FILE}"
      for f in "${DIRECT_GROUPS_FILE}_"*; do
        tr '\n' ':' < "${f}" | sed 's/:$//' >> "${DIRECT_GROUPS_FILE}"
        echo "" >> "${DIRECT_GROUPS_FILE}"
      done
      rm -f "${DIRECT_GROUPS_FILE}_"*
    fi
    rm -f "${TEST_LIST}" /tmp/all_cases_common_$$.txt

    if [ -s "${DIRECT_GROUPS_FILE}" ]; then
      TEST_GROUPS_FILE="${DIRECT_GROUPS_FILE}"
      PARALLEL_JOBS=${NUM_CPUS}
      RUN_SCHEME_TAG="DIRECT-${NUM_CPUS}w"
    else
      echo "WARNING: failed to build direct groups, falling back to mpirun"
      PARALLEL_SCHEME=mpirun
    fi
  else
    echo "WARNING: gtest_list_tests failed, falling back to mpirun"
    PARALLEL_SCHEME=mpirun
  fi
fi

if [ "${PARALLEL_SCHEME}" = "mpirun" ]; then
  echo "=== mpirun scheme: ${PARALLEL_JOBS} groups x mpirun -np 4 ==="
  TEST_GROUPS_FILE=""
  RUN_SCHEME_TAG="MPIRUN-${PARALLEL_JOBS}x4"
fi

echo "=== Run scheme: ${RUN_SCHEME_TAG} ==="

# Run Test
DATE=$(date +%Y-%m-%d-%H-%M-%S)
if [[ "$1" == "--with-memcheck" ]]; then
  echo "we are going to run all tests with memcheck via valgrind"
  for test_group in "${TEST_GROUPS[@]}"; do
    valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --log-file="../memcheck_${test_group// /_}_${DATE}.log" \
      ./tests/test_main --gtest_break_on_failure --gtest_filter="${test_group}" 2>&1 | tee "../test_${test_group// /_}_${DATE}.log"
  done
else
  if [ "${PARALLEL_SCHEME}" = "direct" ] && [ -n "${TEST_GROUPS_FILE}" ]; then
    echo "Starting ${PARALLEL_JOBS} direct gtest processes (no mpirun)..."
    SCHEME_START=$(date +%s)
    cat "${TEST_GROUPS_FILE}" | xargs -P ${PARALLEL_JOBS} -I {} bash -c \
      './tests/test_main --gtest_break_on_failure --gtest_filter="$1" 2>>"../group_${RUN_SCHEME_TAG}_${$}.log" || true' _ {} 2>&1 | tail -20
    SCHEME_END=$(date +%s)
    echo "=== ${RUN_SCHEME_TAG} took $((SCHEME_END - SCHEME_START)) seconds ==="
    rm -f "${TEST_GROUPS_FILE}"
  else
    # || true keeps the script going if any group fails so coverage still runs.
    SCHEME_START=$(date +%s)
    printf "%s\n" "${TEST_GROUPS[@]}" | xargs -P ${PARALLEL_JOBS} -I {} bash -c \
      'mpirun -np 4 ./tests/test_main --gtest_break_on_failure --gtest_filter="$1" 2>>"../group_${RUN_SCHEME_TAG}_${$}.log" || true' _ {}
    SCHEME_END=$(date +%s)
    echo "=== ${RUN_SCHEME_TAG} took $((SCHEME_END - SCHEME_START)) seconds ==="
  fi
fi

cd "$(dirname "${PWD}")"

COVERAGE_FILE=coverage.info
REPORT_FOLDER=coverage_report
mkdir -p -m 750 "${REPORT_FOLDER}"

# Scope lcov capture to where .gcda actually live (build/tests/CMakeFiles/...)
# to avoid recursive stat() on the full build/ tree.
LCOV_CAPTURE_DIR="build"
if [ -d "build/tests/CMakeFiles/test_main.dir" ]; then
  LCOV_CAPTURE_DIR="build/tests"
fi
LCOV_PARALLEL_RC="--rc geninfo_unexecuted_blocks=1 --rc geninfo_filter_threads=$((NUM_CPUS > 4 ? NUM_CPUS / 2 : 2))"

lcov --rc lcov_branch_coverage=1 -c -d "${LCOV_CAPTURE_DIR}" -o "${COVERAGE_FILE}"_tmp \
  ${LCOV_PARALLEL_RC} --ignore-errors gcda,unused,empty,corrupt
lcov -r "${COVERAGE_FILE}"_tmp 'ut/*' '7/ext*' '*7/bits*' 'platform/*' '/usr/local/*' '/usr/include/*' '/opt/buildtools/python-3.7.5/lib/python3.7/site-packages/tensorflow*' '/opt/rh/devtoolset-7/root/usr/lib/gcc/x86_64-redhat-linux/7/include/*' 'tests/*' --rc lcov_branch_coverage=1 --ignore-errors unused,unused -o "${COVERAGE_FILE}"
genhtml "${COVERAGE_FILE}" --output-directory "${REPORT_FOLDER}" --branch-coverage --filter branch --ignore-errors source,category
