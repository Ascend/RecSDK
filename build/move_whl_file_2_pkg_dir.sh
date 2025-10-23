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
warn() { echo >&2 -e "\033[1;31m[WARN ][Depend  ] $1\033[1;37m" ; }
ARCH="$(uname -m)"
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
MxRec_DIR=$(dirname "${SCRIPT_DIR}")
Tf_Rec_V1_DIR=${MxRec_DIR}/training/tf_rec_v1/
pkg_dir=mindxsdk-mxrec
tf_version=$1

function move_whl_file_2_pkg_dir() {
    mkdir -p "$SCRIPT_DIR"/"${pkg_dir}"/"${tf_version}"_whl
    rm -rf "$SCRIPT_DIR"/"${pkg_dir}"/"${tf_version}"_whl/*
    cp ${Tf_Rec_V1_DIR}/dist/*.whl "$SCRIPT_DIR"/"${pkg_dir}"/"${tf_version}"_whl
    cd "$SCRIPT_DIR"/"${pkg_dir}"/"${tf_version}"_whl
    for whl in *.whl; do
      new_name="${whl/any/linux_${ARCH}}"
      echo "Renaming $whl to $new_name"
      mv "$whl" "$new_name"
    done
    cd -
}

move_whl_file_2_pkg_dir