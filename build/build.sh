#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2023. All rights reserved.
# Description: build script.
# Author: MindX SDK
# Create: 2021
# History: NA

set -e
warn() { echo >&2 -e "\033[1;31m[WARN ][Depend  ] $1\033[1;37m" ; }
ARCH="$(uname -m)"
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
ROOT_DIR=$(dirname "${SCRIPT_DIR}")
cd "$SCRIPT_DIR"
if [ "$(uname -m)" = "aarch64" ]
then
  pip3 install virtualenv --force-reinstall
  virtualenv -p "$(which python3.7)" tf2_env
  source tf2_env/bin/activate
  tf265="tensorflow-2.6.5-cp37-cp37m-manylinux2014_aarch64.whl"
  [ ! -f "${tf265}" ] && artget pull "mindx_img_tools 1.0.0" -ru software -rp "${tf265}" -ap ./
  pip3  install "${tf265}" --no-deps
  pip3 install setuptools==49.2.1
  tf2_path=$(dirname "$(dirname "$(which python3)")")/lib/python3.7/site-packages/tensorflow
  deactivate tf2_env
  virtualenv -p "$(which python3.7)" tf1_env
  source tf1_env/bin/activate
  tf115="tensorflow-1.15.0-cp37-cp37m-manylinux2014_aarch64.whl"
  [ ! -f "${tf115}" ] && artget pull "mindx_img_tools 1.0.0" -ru software -rp "${tf115}" -ap ./
  pip3  install "${tf115}" --no-deps
  pip3 install setuptools==49.2.1
  tf1_path=$(dirname "$(dirname "$(which python3)")")/lib/python3.7/site-packages/tensorflow_core
  deactivate tf1_env
fi

if [ "$(uname -m)" = "x86_64" ]
then
  pip3 install virtualenv --force-reinstall
  virtualenv -p "$(which python3.7)" tf2_env
  source tf2_env/bin/activate
  tf265="tensorflow_cpu-2.6.5-cp37-cp37m-manylinux2010_x86_64.whl"
  [ ! -f "${tf265}" ] && artget pull "mindx_img_tools 1.0.0" -ru software -rp "${tf265}" -ap ./
  pip3  install "${tf265}" --no-deps
  pip3 install setuptools==49.2.1
  tf2_path=$(dirname "$(dirname "$(which python3)")")/lib/python3.7/site-packages/tensorflow
  deactivate tf2_env
  virtualenv -p "$(which python3.7)" tf1_env
  source tf1_env/bin/activate
  tf115="tensorflow-1.15.0-cp37-cp37m-manylinux2010_x86_64.whl"
  [ ! -f "${tf115}" ] && artget pull "mindx_img_tools 1.0.0" -ru software -rp "${tf115}" -ap ./
  pip3  install "${tf115}" --no-deps
  pip3 install setuptools==49.2.1
  tf1_path=$(dirname "$(dirname "$(which python3)")")/lib/python3.7/site-packages/tensorflow_core
  deactivate tf1_env
fi

VERSION_FILE="${ROOT_DIR}"/../mindxsdk/build/conf/config.yaml
get_version() {
  if [ -f "$VERSION_FILE" ]; then
    VERSION=$(sed '/.*mindxsdk:/!d;s/.*: //' "$VERSION_FILE")
    if [[ "$VERSION" == *.[b/B]* ]] && [[ "$VERSION" != *.[RC/rc]* ]]; then
      VERSION=${VERSION%.*}
    fi
  else
    VERSION="5.0.T104"
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

opensource_path="${ROOT_DIR}"/../opensource/opensource
abseil_src_path=${opensource_path}/abseil
echo "${abseil_src_path}"
abseil_install_path="${ROOT_DIR}"/install/abseil

src_path="${ROOT_DIR}"/src
acc_ctr_path="${ROOT_DIR}"/src/platform/AccCTR
cp -rf ../platform/securec/* /usr1/mxRec/src/platform/AccCTR/3rdparty/huawei_secure_c
cd "${ROOT_DIR}"

release_tar=Ascend-"${pkg_dir}"_"${VERSION}"_linux-"${ARCH}".tar.gz

install_abseil()
{
    remove "${abseil_install_path}"
    echo "${abseil_install_path}"
    if [[ ! -d "${abseil_install_path}" ]]
    then mkdir -p "${abseil_install_path}"
    fi

    cd "${abseil_src_path}"
    echo "${abseil_src_path}"
    remove CMakeCache.txt
    cmake -DCMAKE_INSTALL_PREFIX="${abseil_install_path}" . && make -j8 && make install

    echo "${project_output_path}"/abseil
    mkdir -p "${project_output_path}"/abseil
    if [ -d "${abseil_install_path}"/lib64/ ]; then
        cp -rf "${abseil_install_path}"/lib64/libabsl* "${project_output_path}"/abseil
    elif [ -d "${abseil_install_path}"/lib/ ]; then
        cp -rf "${abseil_install_path}"/lib/libabsl* "${project_output_path}"/abseil
    else
        echo "${abseil_install_path}"/lib64/ not exist
        exit 1
    fi
}

compile_securec()
{
    if [[ ! -d "${ROOT_DIR}"/platform/securec ]]; then
        echo "securec is not exist"
        exit 1
    fi

    if [[ ! -f "${ROOT_DIR}"/platform/securec/lib/libsecurec.so ]]; then
        cd "${ROOT_DIR}"/platform/securec/src
        make -j
    fi
}

compile_so_file()
{
  cd "${src_path}"
  chmod u+x build.sh
  ./build.sh "$1" "${ROOT_DIR}"
  cd ..
}

compile_acc_ctr_so_file()
{
  cd "${acc_ctr_path}"
  chmod u+x build.sh
  ./build.sh "release"
}

collect_so_file()
{
  cd "${src_path}"
  remove "${src_path}"/libasc
  mkdir -p "${src_path}"/libasc
  chmod u+x libasc

  cp ${acc_ctr_path}/output/ock_ctr_common/lib/* libasc
  cp -df "${ROOT_DIR}"/output/*.so* libasc
  cp "${ROOT_DIR}"/platform/securec/lib/libsecurec.so libasc
}

gen_wheel_file()
{
  cd "${ROOT_DIR}"
  touch "${src_path}"/libasc/__init__.py
  remove "${ROOT_DIR}"/mx_rec/libasc
  mv "${src_path}"/libasc "${ROOT_DIR}"/mx_rec
  cp -rf "${ROOT_DIR}"/tools "${ROOT_DIR}"/mx_rec
  python3 setup.py bdist_wheel --plat-name=linux_$(arch)
  mkdir -p "$1"
  mv dist/mx_rec*.whl "$1"
  remove "${ROOT_DIR}"/mx_rec/libasc
}

gen_tar_file()
{
  cd "${src_path}"
  mv  "${ROOT_DIR}"/tf1_whl ../build/"${pkg_dir}"
  mv  "${ROOT_DIR}"/tf2_whl ../build/"${pkg_dir}"
  cp -r  "${src_path}"/../example ../build/"${pkg_dir}"
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
  remove "${ROOT_DIR}"/build/tf1_env
  remove "${ROOT_DIR}"/build/tf2_env
  remove "${ROOT_DIR}"/build/lib
  remove "${ROOT_DIR}"/build/mindxsdk-mxrec
}

install_abseil
compile_securec

echo "-----Build AccCTR -----"
compile_acc_ctr_so_file

echo "-----Build Start tf1 -----"
source "${SCRIPT_DIR}"/tf1_env/bin/activate
compile_so_file "${tf1_path}"
collect_so_file
gen_wheel_file  "${ROOT_DIR}"/tf1_whl
deactivate tf1_env

echo "-----Build Start tf2 -----"
source "${SCRIPT_DIR}"/tf2_env/bin/activate
compile_so_file "${tf2_path}"
collect_so_file
gen_wheel_file  "${ROOT_DIR}"/tf2_whl
deactivate tf2_env

echo "-----Build gen tar -----"
gen_tar_file

clean
echo "-----Done-----"
