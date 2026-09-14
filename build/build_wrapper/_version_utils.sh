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
# RecSDK 版本工具：集中管理 SDK 基础版本号，所有组件共用同一来源。

# 防止被直接执行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: This file is a library. Please source it." >&2
    exit 1
fi

# RecSDK 基础版本号（所有组件统一）。版本更新时修改此处定义即可。
readonly REC_SDK_BASE_VERSION="26.2.0"

# 获取 SDK 版本号
# 用法: get_sdk_version
# 返回: 回显版本号字符串
get_sdk_version() {
    echo "${REC_SDK_BASE_VERSION}"
}
