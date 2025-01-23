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
ROOT_DIR=$(dirname "${SCRIPT_DIR}")
cd "$SCRIPT_DIR"


VERSION_FILE="${ROOT_DIR}"/../mindxsdk/build/conf/config.yaml
get_version() {
  if [ -f "$VERSION_FILE" ]; then
    VERSION=$(sed '/.*mindxsdk:/!d;s/.*: //' "$VERSION_FILE")
    if [[ "$VERSION" == *.[b/B]* ]] && [[ "$VERSION" != *.[RC/rc]* ]]; then
      VERSION=${VERSION%.*}
    fi
  else
    VERSION="5.0.0"
  fi
}

remove()
{
  if [ -d "$1" ]; then
    rm -rf "$1"
  elif [ -f "$1" ]; then
    rm -f "$1"
  fi
}

project_output_path="${ROOT_DIR}"/output/
remove "${project_output_path}"
remove "${SCRIPT_DIR}/lib"
get_version
export VERSION
echo "Rec SDK: ${VERSION}" >> ./version.info

pkg_dir=mindxsdk-mxrec
remove "${pkg_dir}"
mkdir "${pkg_dir}"
mv version.info "${pkg_dir}"

src_path="${ROOT_DIR}"/src
cd "${ROOT_DIR}"

release_tar=Ascend-"${pkg_dir}"_"${VERSION}"_linux-"${ARCH}".tar.gz

gen_tar_file()
{
  cd "${src_path}"
  cp -r "${src_path}"/../cust_op ../build/"${pkg_dir}"
  cp -r "${src_path}"/../examples  ../build/"${pkg_dir}"
  # change dirs and files 's permission
  chmod 550 ../build/"${pkg_dir}"/tf1_whl
  chmod 550 ../build/"${pkg_dir}"/tf1_whl/mx_rec*.whl
  chmod 550 ../build/"${pkg_dir}"/tf2_whl
  chmod 550 ../build/"${pkg_dir}"/tf2_whl/mx_rec*.whl
  chmod 550 ../build/"${pkg_dir}"/cust_op/
  chmod 550 ../build/"${pkg_dir}"/cust_op/cust_op_by_addr
  cd ../build/"${pkg_dir}"/cust_op/cust_op_by_addr
  chmod 550 *.sh
  chmod 640 *.json
  chmod 550 op_host op_kernel op_host/* op_kernel/*
  cd -
  cd ../build
  tar -zvcf "${release_tar}" "${pkg_dir}" || {
      warn "compression failed, packages might be broken"
  }

  mv "${release_tar}" "${SCRIPT_DIR}"/../output/

}

clean()
{
  remove "${ROOT_DIR}"/dist
  remove "${ROOT_DIR}"/install
  remove "${ROOT_DIR}"/mx_rec.egg-info
  remove "${ROOT_DIR}"/src/build
  remove "${ROOT_DIR}"/build/bdist.linux-"$(arch)"
  remove "${ROOT_DIR}"/build/tf2_env
  remove "${ROOT_DIR}"/build/tf1_env
  remove "${ROOT_DIR}"/build/lib
  remove "${ROOT_DIR}"/build/mindxsdk-mxrec
}


if [ "$(uname -m)" = "x86_64" ]
then
  echo "-----Build gen tar -----"
  bash ${ROOT_DIR}/build/build_tf1_with_opensource.sh
  bash ${ROOT_DIR}/build/build_tf2_with_opensource.sh
  gen_tar_file
  echo "-----Build gen tar finished-----"

  # clean
  echo "-----Done-----"
fi

if [ "$(uname -m)" = "aarch64" ]
then
  echo "-----Build gen tar -----"
  bash ${ROOT_DIR}/build/build_tf1_with_opensource.sh
  bash ${ROOT_DIR}/build/build_tf2_with_opensource.sh
  gen_tar_file
  echo "-----Build gen tar finished-----"

  # clean
  echo "-----Done-----"
fi