#!/bin/bash
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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
warn() { echo >&2 -e "\033[1;31m[WARN ][Depend  ] $1\033[1;37m" ; }
ARCH="$(uname -m)"
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
MxRec_DIR=$(dirname "${SCRIPT_DIR}")
pkg_dir=mindxsdk-mxrec
tf_version=$1

function move_whl_file_2_pkg_dir() {
    mkdir -p "$SCRIPT_DIR"/"${pkg_dir}"/"${tf_version}"_whl
    mv ${MxRec_DIR}/dist/mx_rec*.whl "$SCRIPT_DIR"/"${pkg_dir}"/"${tf_version}"_whl
    cd "$SCRIPT_DIR"/"${pkg_dir}"/"${tf_version}"_whl
    whl_file=$(ls .)
    mv "$whl_file" "${whl_file/any/linux_${ARCH}}"
    cd -
}

move_whl_file_2_pkg_dir