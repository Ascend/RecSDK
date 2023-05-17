#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
# Description: build script.
# Author: MindX SDK
# Create: 2022
# History: NA

set -e
warn() { echo >&2 -e "\033[1;31m[WARN ][Depend  ] $1\033[1;37m" ; }
ARCH="$(uname -m)"
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"
if [ "$(uname -m)" = "aarch64" ]
then
  pip3 install virtualenv --force-reinstall
  virtualenv -p "$(which python3.7)" tf2_env
  source tf2_env/bin/activate
  [ ! -f tensorflow-2.6.5-cp37-cp37m-manylinux2014_aarch64.whl ] && wget  --no-check-certificate https://cmc-szver-artifactory.cmc.tools.huawei.com/artifactory/cmc-software-release/MindX/mindx_img_tools/1.0.0/tensorflow-2.6.5-cp37-cp37m-manylinux2014_aarch64.whl
  pip3  install tensorflow-2.6.5-cp37-cp37m-manylinux2014_aarch64.whl --no-deps
  pip3 install setuptools==49.2.1
  tf2_path=$(dirname "$(dirname "$(which python3)")")/lib/python3.7/site-packages/tensorflow
  deactivate tf2_env
  virtualenv -p "$(which python3.7)" tf1_env
  source tf1_env/bin/activate
  [ ! -f tensorflow-1.15.0-cp37-cp37m-manylinux2014_aarch64.whl ] && wget  --no-check-certificate  https://cmc-szver-artifactory.cmc.tools.huawei.com/artifactory/cmc-software-release/MindX/mindx_img_tools/1.0.0/tensorflow-1.15.0-cp37-cp37m-manylinux2014_aarch64.whl
  pip3  install tensorflow-1.15.0-cp37-cp37m-manylinux2014_aarch64.whl --no-deps
  pip3 install setuptools==49.2.1
  tf1_path=$(dirname "$(dirname "$(which python3)")")/lib/python3.7/site-packages/tensorflow_core
  deactivate tf1_env
fi

if [ "$(uname -m)" = "x86_64" ]
then
  pip3 install virtualenv --force-reinstall
  virtualenv -p "$(which python3.7)" tf2_env
  source tf2_env/bin/activate
  [ ! -f tensorflow-2.6.5-cp37-cp37m-manylinux2010_x86_64.whl ] && wget  --no-check-certificate https://cmc-hgh-artifactory.cmc.tools.huawei.com/artifactory/opensource_general/Tensorflow/2.6.5/package/tensorflow-2.6.5-cp37-cp37m-manylinux2010_x86_64.whl
  pip3  install tensorflow-2.6.5-cp37-cp37m-manylinux2010_x86_64.whl --no-deps
  pip3 install setuptools==49.2.1
  tf2_path=$(dirname "$(dirname "$(which python3)")")/lib/python3.7/site-packages/tensorflow
  deactivate tf2_env
  virtualenv -p "$(which python3.7)" tf1_env
  source tf1_env/bin/activate
  [ ! -f tensorflow-1.15.0-cp37-cp37m-manylinux2010_x86_64.whl ] && wget  --no-check-certificate  https://cmc-szver-artifactory.cmc.tools.huawei.com/artifactory/cmc-software-release/MindX/mindx_img_tools/1.0.0/tensorflow-1.15.0-cp37-cp37m-manylinux2010_x86_64.whl
  pip3  install tensorflow-1.15.0-cp37-cp37m-manylinux2010_x86_64.whl --no-deps
  pip3 install setuptools==49.2.1
  tf1_path=$(dirname "$(dirname "$(which python3)")")/lib/python3.7/site-packages/tensorflow_core
  deactivate tf1_env
fi

VERSION_FILE=${SCRIPT_DIR}/../../mindxsdk/build/conf/config.yaml
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

project_root_folder=${SCRIPT_DIR}/..
project_output_path=${project_root_folder}/output/
rm -rf "${project_output_path}"
rm -rf "${SCRIPT_DIR}/lib"
get_version
export VERSION
echo "MindX SDK mxrec: ${VERSION}" >> ./version.info

pkg_dir=mindxsdk-mxrec
[ -d ${pkg_dir} ] && rm -rf ${pkg_dir}
mkdir ${pkg_dir}
mv version.info ${pkg_dir}

opensource_path=${project_root_folder}/../opensource/opensource
abseil_src_path=${opensource_path}/abseil
echo "${abseil_src_path}"
abseil_install_path=${project_root_folder}/install/abseil

src_path=${project_root_folder}/src

cd "${project_root_folder}"

release_tar=Ascend-${pkg_dir}_${VERSION}_linux-${ARCH}.tar.gz

install_abseil()
{
    rm -rf "${abseil_install_path}"
    echo "${abseil_install_path}"
    if [[ ! -d ${abseil_install_path} ]]
    then mkdir -p "${abseil_install_path}"
    fi

    cd "${abseil_src_path}"
    echo "${abseil_src_path}"
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
    if [[ ! -d ${project_root_folder}/platform/securec ]]; then
        echo "securec is not exist"
        exit 1
    fi

    if [[ ! -f ${project_root_folder}/platform/securec/lib/libsecurec.so ]]; then
        cd ${project_root_folder}/platform/securec/src
        make -j
    fi
}

compile_so_file()
{
  cd "${src_path}"
  chmod u+x build.sh
  ./build.sh "$1" "${project_root_folder}"
  cd ..
}

collect_so_file()
{
  cd "${src_path}"
  rm -rf "${src_path}"/libasc
  mkdir -p "${src_path}"/libasc
  chmod u+x libasc

  cp -df "${project_root_folder}"/output/*.so* libasc
  cp ${project_root_folder}/platform/securec/lib/libsecurec.so libasc
}

gen_wheel_file()
{
  cd "${project_root_folder}"
  touch "${src_path}"/libasc/__init__.py
  [ -d "${project_root_folder}"/mx_rec/libasc ] && rm -rf "${project_root_folder}"/mx_rec/libasc
  mv "${src_path}"/libasc "${project_root_folder}"/mx_rec
  python3 setup.py bdist_wheel
  mkdir -p "$1"
  mv dist/mx_rec*.whl "$1"
  rm -rf "${project_root_folder}"/mx_rec/libasc
}

gen_tar_file()
{
  cd "${src_path}"
  mv  "${project_root_folder}"/tf1_whl ../build/${pkg_dir}
  mv  "${project_root_folder}"/tf2_whl ../build/${pkg_dir}
  cp -r  "${src_path}"/../example ../build/${pkg_dir}
  cd ../build
  tar -zvcf "${release_tar}" "${pkg_dir}" || {
      warn "compression failed, packages might be broken"
  }

  mv "${release_tar}" "${SCRIPT_DIR}"/../output/

}

install_abseil
compile_securec

echo "-----Build Start tf1 -----"
source "${SCRIPT_DIR}"/tf1_env/bin/activate
compile_so_file "${tf1_path}"
collect_so_file
gen_wheel_file  "${project_root_folder}"/tf1_whl
deactivate tf1_env

echo "-----Build Start tf2 -----"
source "${SCRIPT_DIR}"/tf2_env/bin/activate
compile_so_file "${tf2_path}"
collect_so_file
gen_wheel_file  "${project_root_folder}"/tf2_whl
deactivate tf2_env

echo "-----Build gen tar -----"
gen_tar_file

echo "-----Done-----"
