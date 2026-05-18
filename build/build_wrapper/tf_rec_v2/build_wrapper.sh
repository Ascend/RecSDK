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

SDK_VERSION="26.1.0"
export RECSDK_VERSION="$SDK_VERSION"
# 保存脚本所在目录的绝对路径
SCRIPT_BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# 初始化 tf_rec_v2 自身的子模块（不使用外部的 catlass 和 hkv）
TF_REC_V2_DIR="$SCRIPT_BASE_DIR/../../../training/tf_rec_v2"
GITMODULES="$TF_REC_V2_DIR/.gitmodules"

# 跳过 Git SSL 证书验证（解决内网环境证书问题）
export GIT_SSL_NO_VERIFY=1

# 读取 .gitmodules 文件，直接 clone 每个子模块
while IFS= read -r line; do
    if [[ "$line" =~ ^\[submodule\ \"([^\"]+)\" ]]; then
        submodule_path=""
        submodule_url=""
        submodule_branch=""
    elif [[ "$line" =~ ^[[:space:]]+path[[:space:]]*=[[:space:]]*(.+)$ ]]; then
        submodule_path="$TF_REC_V2_DIR/${BASH_REMATCH[1]}"
    elif [[ "$line" =~ ^[[:space:]]+url[[:space:]]*=[[:space:]]*(.+)$ ]]; then
        submodule_url="${BASH_REMATCH[1]}"
    elif [[ "$line" =~ ^[[:space:]]+branch[[:space:]]*=[[:space:]]*(.+)$ ]]; then
        submodule_branch="${BASH_REMATCH[1]}"
    fi

    # 当读取完 url 和 branch 后，执行 clone
    if [[ -n "$submodule_path" && -n "$submodule_url" && -n "$submodule_branch" ]]; then
        if [[ ! -d "$submodule_path" ]]; then
            echo "Cloning submodule: $submodule_url (branch: $submodule_branch)"
            git clone --depth=1 -b "$submodule_branch" "$submodule_url" "$submodule_path"
        fi
        submodule_path=""
        submodule_url=""
        submodule_branch=""
    fi
done < "$GITMODULES"

cd "$SCRIPT_BASE_DIR/resources" || { echo "Error: Cannot cd to resources"; exit 1; }

BUILD_DIR="mx_rec_builder"
rm -rf $BUILD_DIR && mkdir -p $BUILD_DIR

for file in {"setup.py","MANIFEST.in","requirements.txt"}; do
    if [[ -f "$file" ]]; then
        cp "$file" "$BUILD_DIR"
        echo "Copied: $file"
    else
        echo "Error: Cannot find file $file."
        exit 1
    fi
done

# 1. 准备开源依赖 (opensource)
OPENSOURCE_DIR="../../../../../opensource"
if [ ! -d "$OPENSOURCE_DIR" ]; then
    echo "Downloading opensource dependencies to $OPENSOURCE_DIR ..."
    mkdir -p "$OPENSOURCE_DIR"
    wget -q https://github.com/pybind/pybind11/archive/refs/tags/v2.10.3.zip -O "$OPENSOURCE_DIR/pybind11-2.10.3.zip"
    wget -q https://github.com/huaweicloud/huaweicloud-sdk-c-obs/archive/refs/tags/v3.23.9.zip -O "$OPENSOURCE_DIR/huaweicloud-sdk-c-obs-3.23.9.zip"
    echo "Successfully downloaded opensource dependencies."
else
    echo "Opensource dependencies found in $OPENSOURCE_DIR."
fi

# 2. 编译 SDK Python 包
# tf_rec_v2 目前仅适配 TF1 分支
echo "Set CANN environment to a5 for software package compilation..."
source /usr/local/set_cann_env.sh a5

echo "Start compiling TF_REC_V2 SDK using setup_tf1.py inside training/tf_rec_v2..."
(cd ../../../../training/tf_rec_v2 && python3 setup_tf1.py bdist_wheel)

# 3. 准备打包目录
echo "Start assembling distribution package..."

mkdir -p $BUILD_DIR/mindxsdk-mxrec/tf1_whl
mkdir -p $BUILD_DIR/npu-ops/A5/recsdk-npu-ops/recsdk_ops

# 收集生成的 .whl 包
TF1_WHL=$(find ../../../../training/tf_rec_v2/build/mxrec-for-lingqu2.0/tf1_whl -type f -name "*mxrec*.whl" | head -n 1)

if [ -z "$TF1_WHL" ]; then
    echo "Error: Cannot find generated tf1_whl file."
    exit 1
fi

cp "$TF1_WHL" "$BUILD_DIR/mindxsdk-mxrec/tf1_whl"

# 4. 按需编译框架算子 (仅限 A5)
OPS_REQ="op_requirements.txt"
if [ ! -f "$OPS_REQ" ]; then
    echo "Error: Cannot find file $OPS_REQ."
    exit 1
fi

echo "Start compiling specific operators for A5-TF based on $OPS_REQ..."

while IFS= read -r op || [[ -n "$op" ]]; do
    op=$(echo "$op" | tr -d '\r' | xargs)
    if [[ -n "$op" ]] && [[ "$op" != "#"* ]]; then
        op_dir_name=$op
        
        # 对应的 A5 (c310) 算子源码路径
        op_src_path_a5="../../../../cust_op/ascendc_op/ai_core_op/${op_dir_name}/c310"
        
        if [ -d "$op_src_path_a5" ]; then
            echo "Compiling $op_dir_name for Ascend950 (A5)..."
            source /usr/local/set_cann_env.sh a5
            (cd "$op_src_path_a5" && bash ./run.sh --ai-core ai_core-Ascend950 || echo "Warning: Compilation failed for A5")
            
            gen_run_file_a5=$(find "$op_src_path_a5/${op_dir_name}/build_out" -type f -name "custom_opp*.run" 2>/dev/null | head -n 1)
            if [ -n "$gen_run_file_a5" ]; then
                cp "$gen_run_file_a5" "$BUILD_DIR/npu-ops/A5/recsdk-npu-ops/recsdk_ops/mxrec_opp_${op}.run"
                echo "Successfully generated and collected mxrec_opp_${op}.run (A5)"
            else
                echo "Warning: Failed to generate .run file for A5 for $op_dir_name"
            fi
        else
            echo "Warning: c310 directory not found for $op_dir_name, skipping A5 compilation"
        fi
    fi
done < "$OPS_REQ"

# 5. 生成统一的 tar.gz 发布包
cd $BUILD_DIR
python3 setup.py sdist

if [ -d "dist" ]; then
    TAR_FILE=$(ls dist/*.tar.gz | head -n 1)
    if [ ! -z "$TAR_FILE" ]; then
        ARCH=$(uname -m)
        if [ "$ARCH" = "aarch64" ]; then
            SUFFIX="aarch64"
        elif [ "$ARCH" = "x86_64" ]; then
            SUFFIX="x86_64"
        else
            SUFFIX=$ARCH
        fi
        mkdir -p ../../../../output
        mv "$TAR_FILE" "../../../../output/tf_rec_v2-${SDK_VERSION}-linux_${SUFFIX}.tar.gz"
    fi
fi

cd ..
rm -rf $BUILD_DIR

unset RECSDK_VERSION
echo "Done!"