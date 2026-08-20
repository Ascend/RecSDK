#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

set -e

VERSION="1.5.0-npu"
ARCH=$(uname -m)

current_dir="$(pwd)"
package_name="Ascend-mindxsdk-torchrec-"${VERSION}"-linux-"${ARCH}".tar.gz"

# 依赖torchrec源码,版本固定为1.5.0，为避免网络不稳定问题，请提前下载好并切换到指定commit_id。
# git clone -b release/v1.5.0 https://github.com/pytorch/torchrec.git
# cd torchrec && git checkout 4002d8bc326f2864982a3d11dd4f01c59dce6457
cd torchrec
# patch
cp ../torchrec1.5.0_npu.patch ./ && dos2unix torchrec1.5.0_npu.patch
git init
git apply torchrec1.5.0_npu.patch

# 编译安装包
if [ -f "${package_name}" ]; then
  rm "${package_name}"
fi
python3 setup.py bdist_wheel --plat-name linux_"${ARCH}"
cp requirements.txt dist/
tar -czvf "${package_name}" -C dist .
