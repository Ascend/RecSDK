#!/bin/bash
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

set -e

VERSION="1.1.0-npu"
ARCH=$(uname -m)

current_dir="$(pwd)"
torchrec_path="$current_dir/torchrec"
package_name="Ascend-mindxsdk-torchrec-"${VERSION}"-linux-"${ARCH}".tar.gz"

# 检查目录是否存在
if [ -d "$torchrec_path" ]; then
  rm -rf "$torchrec_path"
fi

# 依赖torchrec源码,版本固定为1.1.0，提交hash固定为2c5f6ee。
git clone -b release/v1.1.0 https://github.com/pytorch/torchrec.git
cd torchrec && git checkout 2c5f6ee
# patch
cp ../torchrec_npu.patch ./ && dos2unix torchrec_npu.patch
git apply torchrec_npu.patch

# 编译安装包
if [ -f "${package_name}"]; then
  rm "${package_name}"
fi
python3 setup.py bdist_wheel --plat-name linux_"${ARCH}"
cp requirements.txt dist/
tar -czvf "${package_name}" -C dist .