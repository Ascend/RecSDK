#!/bin/bash
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
warn() { echo >&2 -e "\033[1;31m[WARN ][Depend  ] $1\033[1;37m" ; }
ARCH="$(uname -m)"
tf_version=$1
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
MxRec_DIR=$(dirname "$(dirname "${SCRIPT_DIR}")")

pkg_dir=mxrec-for-lingqu2.0
whl_dir=${MxRec_DIR}/build/${pkg_dir}/${tf_version}_whl

function move_whl_file_2_pkg_dir() {
    rm -rf ${whl_dir}
    mkdir -p ${whl_dir}

    cp ${MxRec_DIR}/dist/mxrec_for_lingqu*.whl ${whl_dir}
    cd ${whl_dir}
    whl_file=$(ls .)
    mv "$whl_file" "${whl_file/any/linux_${ARCH}}"
    cd -
    rm -rf ${MxRec_DIR}/dist
}

move_whl_file_2_pkg_dir

# Set permissions
chmod 550 ${whl_dir}
chmod 550 ${whl_dir}/mxrec_for_lingqu*.whl

echo "Wheel file moved and renamed for ${tf_version}"