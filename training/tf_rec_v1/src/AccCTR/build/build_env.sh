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

# set env for building project
set -e
CURRENT_PATH=$(cd "$(dirname "$0")"; pwd)
TOP_DIR=${CURRENT_PATH}/..
OUTPUT_PATH=${TOP_DIR}/output

OCK_CTR_PATH=${TOP_DIR}/src
OCK_CTR_DIAGNOSE_PATH=${TOP_DIR}/tests
OCK_CTR_OPENSOURCE_PATH=${TOP_DIR}/../../../opensource

CPU_TYPE=$(arch)
OCK_VERSION=22.0.0
