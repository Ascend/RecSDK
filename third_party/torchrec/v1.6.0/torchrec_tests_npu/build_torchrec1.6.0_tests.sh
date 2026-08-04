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

# 基于torchrec v1.6.0源码固定版本
VERSION="1.6.0"
cd torchrec
# copy tests shell
cp ../run_tests.sh ./ && dos2unix run_tests.sh
# apply patch
cp ../torchrec${VERSION}_tests_npu.patch ./ && dos2unix torchrec${VERSION}_tests_npu.patch
git init
git apply --ignore-whitespace torchrec${VERSION}_tests_npu.patch

echo "Done."
