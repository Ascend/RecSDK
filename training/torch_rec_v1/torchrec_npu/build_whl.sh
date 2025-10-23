#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

set -e

VERSION="1.1.0-npu"
ARCH=$(uname -m)

current_dir="$(pwd)"
package_name="Ascend-mindxsdk-torchrec-"${VERSION}"-linux-"${ARCH}".tar.gz"

# 依赖torchrec源码,版本固定为1.1.0，提交hash固定为2c5f6ee，避免网络不稳定问题，流水线下载好。
# git clone -b release/v1.1.0 https://github.com/pytorch/torchrec.git
cd torchrec
# patch
cp ../torchrec_npu.patch ./ && dos2unix torchrec_npu.patch
git init
git apply torchrec_npu.patch

# 编译安装包
if [ -f "${package_name}" ]; then
  rm "${package_name}"
fi
python3 setup.py bdist_wheel --plat-name linux_"${ARCH}"
cp requirements.txt dist/
tar -czvf "${package_name}" -C dist .