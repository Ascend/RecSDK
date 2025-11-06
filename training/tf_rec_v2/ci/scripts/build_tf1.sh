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

##################################################################
# requirements：
#   Python3.7.5
#   GCC 7.3.0
#   CMake 3.20.6
# compile mxrec
##################################################################

set -e
warn() { echo >&2 -e "\033[1;31m[WARN ][Depend  ] $1\033[1;37m" ; }
ARCH="$(uname -m)"
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT_DIR=$(dirname "$(dirname "${SCRIPT_DIR}")")

# 配置tf1路径
[ -e /usr/local/python3.7.5/tf1_env/bin/activate ] && source /usr/local/python3.7.5/tf1_env/bin/activate
python_home=$(dirname "$(dirname "$(which python3.7)")")
tf1_path=$python_home/lib/python3.7/site-packages/tensorflow_core
[ -e /usr/local/python3.7.5/tf1_env/bin/activate ] && deactivate tf1_env
if [ ! -d "${tf1_path}" ]; then
    echo "TensorFlow path not found: ${tf1_path}"
    exit 1
fi

cd "${PROJECT_ROOT_DIR}"
rm -rf "${PROJECT_ROOT_DIR}"/build/lib/*

code_path="${PROJECT_ROOT_DIR}"/mxrec/
function collect_files() {
  echo "collecting mxrec files"
  rm -rf "${code_path}"/librec
  mkdir -p "${code_path}"/librec
  cd "${code_path}"
  chmod u+x librec

  touch "${code_path}"/librec/__init__.py
  chmod 640 "${code_path}"/librec/__init__.py
  cp "${PROJECT_ROOT_DIR}"/build/mxrec/core/host/*.so "${code_path}"/librec || echo "Warning: No files in mxrec directory"
  cp "${PROJECT_ROOT_DIR}"/build/mxrec/core/device/framework/*.so "${code_path}"/librec || echo "Warning: No files in dir"
}


echo "----------------        moving files to new structure   ----------------"
collect_files

echo "----------------        compile MxRec success!!!!        ----------------"
echo "New package structure created at: ${PROJECT_ROOT_DIR}/build/mxrec-for-lingqu2.0"