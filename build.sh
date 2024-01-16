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
ROOT_DIR=$(dirname "$(readlink -f "$0")")

remove()
{
  if [ -d "$1" ]; then
    rm -rf "$1"
  elif [ -f "$1" ]; then
    rm -f "$1"
  fi
}

clean()
{
  remove "${ROOT_DIR}"/dist
  remove "${ROOT_DIR}"/install
  remove "${ROOT_DIR}"/mx_rec.egg-info
  remove "${ROOT_DIR}"/src/build
  remove "${ROOT_DIR}"/build/bdist.linux-"$(arch)"
  remove "${ROOT_DIR}"/build/tf1_env
  remove "${ROOT_DIR}"/build/tf2_env
  remove "${ROOT_DIR}"/build/lib
  remove "${ROOT_DIR}"/build/mindxsdk-mxrec
}

clean
