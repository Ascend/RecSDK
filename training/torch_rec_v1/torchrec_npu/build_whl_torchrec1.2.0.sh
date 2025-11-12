#!/bin/bash
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

set -e

# 依赖torchrec v1.2.0源码固定版本，参考README下载源码至当前目录。
VERSION="1.2.0"
cd torchrec
# apply patch
cp ../torchrec${VERSION}_npu.patch ./ && dos2unix torchrec${VERSION}_npu.patch
git init
git apply torchrec${VERSION}_npu.patch

ARCH=$(uname -m)
package_name="Ascend-mindxsdk-torchrec-"${VERSION}-npu"-linux-"${ARCH}".tar.gz"

# 编译安装包
if [ -f "${package_name}" ]; then
  rm "${package_name}"
fi
python3 setup.py bdist_wheel --plat-name linux_"${ARCH}"
cp requirements.txt dist/
tar -czvf "${package_name}" -C dist .