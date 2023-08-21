#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2023. All rights reserved.
# Description: build script.
# Author: MindX SDK
# Create: 2021
# History: NA

export GLOG_CUSTOM_PREFIX_SUPPORT=1

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
    VERSION="5.0.rc3"
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
echo "MindX SDK mxrec: ${VERSION}" >> ./version.info

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
  mv  "${ROOT_DIR}"/tf1_whl ../build/"${pkg_dir}"
  mv  "${ROOT_DIR}"/tf2_whl ../build/"${pkg_dir}"
  cp -r  "${src_path}"/../cust_op ../build/"${pkg_dir}"
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
  bash ${ROOT_DIR}/build/build_tf1.sh
  bash ${ROOT_DIR}/build/build_tf2.sh
  gen_tar_file
  echo "-----Build gen tar finished-----"

  clean
  echo "-----Done-----"
fi

if [ "$(uname -m)" = "aarch64" ]
then
  echo "-----Build gen tar -----"
  bash ${ROOT_DIR}/build/build_tf1.sh
  bash ${ROOT_DIR}/build/build_tf2.sh
  gen_tar_file
  echo "-----Build gen tar finished-----"

  clean
  echo "-----Done-----"
fi