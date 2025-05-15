#!/bin/bash
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

set -e

version_file="version.txt"

if [ ! -f "${version_file}" ]; then
  VERSION="1.0.0"
else
  VERSION=$(head -n 1 "${version_file}")
fi
ARCH=$(uname -m)

# compile so
cur_path=${PWD}
cd ${PWD}/src/
bash run.sh
cp ./build/*.so ${cur_path}/hybrid_torchrec/modules/
cd -

package_name="Ascend-mindxsdk-hybrid-torchrec-"${VERSION}"-linux-"${ARCH}".tar.gz"
if [ -f "${package_name}" ]; then
  rm "${package_name}"
fi
python3 setup.py bdist_wheel --plat-name linux_"${ARCH}"
cp requirements.txt dist/
tar -czvf "${package_name}" -C dist .