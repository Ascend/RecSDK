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

VERSION_FILE="${MxRec_DIR}"/../mindxsdk/build/conf/config.yaml
get_version() {
  if [ -f "$VERSION_FILE" ]; then
    VERSION=$(sed '/.*mindxsdk:/!d;s/.*: //' "$VERSION_FILE")
    if [[ "$VERSION" == *.[b/B]* ]] && [[ "$VERSION" != *.[RC/rc]* ]]; then
      VERSION=${VERSION%.*}
    fi
  else
    VERSION="6.0.RC2"
  fi
}

get_version
echo "MindX SDK mxrec: ${VERSION}" >> ./version.info

pkg_dir=mindxsdk-mxrec
release_tar=Ascend-"${pkg_dir}"_"${VERSION}"_linux-"${ARCH}".tar.gz
mv version.info "${SCRIPT_DIR}"/"${pkg_dir}"

function gen_tar_file()
{
  cd "${MxRec_DIR}"
  cp -r ./cust_op ./build/"${pkg_dir}"
  cp -r ./examples  ./build/"${pkg_dir}"
  # change dirs and files 's permission
  chmod 550 ./build/"${pkg_dir}"/tf1_whl
  chmod 550 ./build/"${pkg_dir}"/tf1_whl/mx_rec*.whl
  chmod 550 ./build/"${pkg_dir}"/tf2_whl
  chmod 550 ./build/"${pkg_dir}"/tf2_whl/mx_rec*.whl
  chmod 550 ./build/"${pkg_dir}"/cust_op/
  chmod 550 ./build/"${pkg_dir}"/cust_op/cust_op_by_addr
  cd ./build/"${pkg_dir}"/cust_op/cust_op_by_addr
  chmod 550 *.sh
  chmod 640 *.json
  chmod 550 op_host op_kernel op_host/* op_kernel/*
  cd -
  cd ./build/"${pkg_dir}"/cust_op/
  chmod 550 -R fused_lazy_adam
  chmod 640 fused_lazy_adam/*.json
  cd -
  cd ./build
  tar -zvcf "${release_tar}" "${pkg_dir}" || {
      warn "compression failed, packages might be broken"
  }

  mv "${release_tar}" ../output/

}

function clean()
{
  rm -rf "${MxRec_DIR}"/dist
  rm -rf "${MxRec_DIR}"/mx_rec.egg-info
  rm -rf "${MxRec_DIR}"/src/build
  rm -rf "${MxRec_DIR}"/mx_rec/libasc
  rm -rf "${MxRec_DIR}"/build/lib
  rm -rf "${MxRec_DIR}"/build/bdist.linux-${ARCH}
}

gen_tar_file

clean

# compile cust op
echo "----------------        start to compile cust op        ----------------"
cd "${MxRec_DIR}"/cust_op/cust_op_by_addr
chmod u+x run.sh
./run.sh
echo "----------------      compile cust op success!!!!       ----------------"