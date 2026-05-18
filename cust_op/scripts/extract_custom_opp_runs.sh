#!/usr/bin/env bash
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
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <op_name> <stage_dir>" >&2
    exit 2
fi

op_name="$1"
stage_dir="$2"
build_out_dir="${op_name}/build_out"

mkdir -p "${stage_dir}"

if [ ! -d "${build_out_dir}" ]; then
    exit 0
fi

shopt -s nullglob
for pkg in "${build_out_dir}"/custom_opp*.run; do
    [ -f "${pkg}" ] || continue
    bash "${pkg}" --quiet --install-path="${stage_dir}"
done
