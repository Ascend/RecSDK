#!/bin/bash
# -*- coding: utf-8 -*-
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
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
MxRec_DIR=$(dirname "$(dirname "${SCRIPT_DIR}")")

VERSION_FILE="${MxRec_DIR}/dynamic_emb/dynamic_emb/__init__.py"

get_version() {
    if [ -f "${VERSION_FILE}" ]; then
        VERSION=$(sed -nE "/__version__/s/.*__version__[[:space:]]*=[[:space:]]*['\"]([^'\"]+)['\"].*/\1/p" "${VERSION_FILE}")
        VERSION=$(echo "${VERSION//+/.}" | tr 'a-z' 'A-Z')
        if [[ "${VERSION}" == *.[b/B]* ]] && [[ "${VERSION}" != *.[RC/rc]* ]]; then
            VERSION=${VERSION%.*}
        fi
    else
        VERSION="7.3.T50"
    fi
}

get_version

pkg_dir="dynamic-emb"
release_tar="Ascend-mindxsdk-${pkg_dir}-${VERSION}-pytorch2.7.1-linux-${ARCH}.tar.gz"

function gen_tar_file()
{
  # change dirs and files' permission
  cd "${MxRec_DIR}"
  cd ./build
  chmod 550 ./dynamic_emb*.whl

  tar -zvcf ${release_tar} dynamic_emb*.whl || {
      warn "compression failed, packages might be broken"
  }
  chmod 640 ${release_tar}
}

function clean()
{
  rm -rf ${MxRec_DIR}/dynamic_emb/build
  rm -rf ${MxRec_DIR}/dynamic_emb/dynamic_emb.egg-info
  chmod -R 750 ${MxRec_DIR}/build  # for ci permission
  rm -rf ${MxRec_DIR}/build
}

function move_to_output_dir()
{
  echo "for ci requirement, move ${release_tar} to output dir"
  OUTPUT_DIR=${MxRec_DIR}/output
  if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir ${OUTPUT_DIR}
  fi
  mv ${MxRec_DIR}/build/*tar.gz ${OUTPUT_DIR}
}

gen_tar_file

move_to_output_dir

clean