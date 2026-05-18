#!/usr/bin/env bash
# -*- coding: utf-8 -*-
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

# Resolve test files from this script's directory so callers need not cd here.
DUMP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

COV_ARGS=()
if [ "${RUN_INCREMENTAL_WITH_COVERAGE:-}" = "1" ]; then
    COV_ARGS=(
        "--cov=${DYNAMIC_EMB_COV_TARGET:?set by run_python_ut.sh}"
        "--cov-config=${COVERAGE_RCFILE:?set by run_python_ut.sh}"
        --cov-append
        --cov-branch
    )
fi

echo "[incremental_dump] running: test_dynamicemb_extensions.py"
pytest "${DUMP_DIR}/test_dynamicemb_extensions.py" -s "${COV_ARGS[@]}" --cov-report=term --durations=5 -vv

echo "[incremental_dump] running: test_batched_dynamicemb_tables.py"
pytest "${DUMP_DIR}/test_batched_dynamicemb_tables.py" -s "${COV_ARGS[@]}" --cov-report=term --durations=5 -vv

echo "[incremental_dump] running: test_distributed_dynamicemb.py (torchrun --nproc_per_node=2)"
torchrun --nproc_per_node=2 -m pytest "${DUMP_DIR}/test_distributed_dynamicemb.py" -s
