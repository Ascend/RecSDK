#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

# 依赖torchrec v1.6.0源码固定版本，参考README下载源码至当前目录。
VERSION="1.6.0"
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

echo "Done."
