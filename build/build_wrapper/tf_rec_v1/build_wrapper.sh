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
cd "$(dirname "$0")/resources" || { echo "Error: Cannot cd to resources"; exit 1; }

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

# 2. 编译 SDK C++ 及 Python 包
echo "Set CANN environment to a2 for software package compilation..."
source /usr/local/set_cann_env.sh a2

echo "Start compiling TF1 SDK using setup_tf1.py..."
(cd ../../../../ && python3 setup_tf1.py bdist_wheel)

echo "Start compiling TF2 SDK using setup_tf2.py..."
(cd ../../../../ && python3 setup_tf2.py bdist_wheel)

# 3. 准备打包目录
echo "Start assembling distribution package..."

mkdir -p $BUILD_DIR/mindxsdk-mxrec/tf1_whl
mkdir -p $BUILD_DIR/mindxsdk-mxrec/tf2_whl
mkdir -p $BUILD_DIR/npu-ops/A2/recsdk-npu-ops/recsdk_ops
mkdir -p $BUILD_DIR/npu-ops/A3/recsdk-npu-ops/recsdk_ops
mkdir -p $BUILD_DIR/npu-ops/A5/recsdk-npu-ops/recsdk_ops

# 取出刚刚经由 setup_tf_x.py 挪动并改名好的 .whl 包
TF1_WHL=$(find ../../../../build/mindxsdk-mxrec/tf1_whl -type f -name "*mx_rec*.whl" | head -n 1)
TF2_WHL=$(find ../../../../build/mindxsdk-mxrec/tf2_whl -type f -name "*mx_rec*.whl" | head -n 1)

if [ -z "$TF1_WHL" ] || [ -z "$TF2_WHL" ]; then
    echo "Error: Cannot find generated tf1_whl file or tf2_whl file."
    exit 1
fi

cp "$TF1_WHL" "$BUILD_DIR/mindxsdk-mxrec/tf1_whl"
cp "$TF2_WHL" "$BUILD_DIR/mindxsdk-mxrec/tf2_whl"

# 4. 按需编译框架算子
OPS_REQ="op_requirements.txt"
if [ ! -f "$OPS_REQ" ]; then
    echo "Error: Cannot find file $OPS_REQ."
    exit 1
fi

echo "Start compiling specific operators for A2-TF based on $OPS_REQ..."

while IFS= read -r op || [[ -n "$op" ]]; do
    op=$(echo "$op" | tr -d '\r' | xargs)
    if [[ -n "$op" ]] && [[ "$op" != "#"* ]]; then
        # 算子目录名（对应 op_requirements.txt 中的行）
        op_dir_name=$op
        
        # 对应的算子源码路径 (基于 run.sh 逻辑，构建产物在 v220/算子名/build_out 下)
        op_src_path="../../../../cust_op/ascendc_op/ai_core_op/${op_dir_name}/v220"
        
        if [ ! -d "$op_src_path" ]; then
            echo "Warning: Operator source $op_src_path not found, skipping $op"
            continue
        fi
        
        echo "Compiling $op_dir_name for Ascend910B1 (A2-TF)..."
        source /usr/local/set_cann_env.sh a2
        (cd "$op_src_path" && bash ./run.sh --ai-core ai_core-Ascend910B1 || echo "Warning: Compilation failed for A2")
        
        # 收集编译出的 .run 产物
        gen_run_file=$(find "$op_src_path/${op_dir_name}/build_out" -type f -name "custom_opp*.run" 2>/dev/null | head -n 1)
        if [ -n "$gen_run_file" ]; then
            cp "$gen_run_file" "$BUILD_DIR/npu-ops/A2/recsdk-npu-ops/recsdk_ops/mxrec_opp_${op}.run"
            echo "Successfully generated and collected mxrec_opp_${op}.run (A2)"
        else
            echo "Warning: Failed to generate .run file for A2 for $op_dir_name"
        fi

        echo "Compiling $op_dir_name for Ascend910_93 (A3)..."
        source /usr/local/set_cann_env.sh a3
        (cd "$op_src_path" && bash ./run.sh --ai-core ai_core-Ascend910_93 || echo "Warning: Compilation failed for A3")
        
        gen_run_file_a3=$(find "$op_src_path/${op_dir_name}/build_out" -type f -name "custom_opp*.run" 2>/dev/null | head -n 1)
        if [ -n "$gen_run_file_a3" ]; then
            cp "$gen_run_file_a3" "$BUILD_DIR/npu-ops/A3/recsdk-npu-ops/recsdk_ops/mxrec_opp_${op}.run"
            echo "Successfully generated and collected mxrec_opp_${op}.run (A3)"
        else
            echo "Warning: Failed to generate .run file for A3 for $op_dir_name"
        fi

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
        mv "$TAR_FILE" "../../../../output/tf_rec_v1-${SDK_VERSION}-linux_${SUFFIX}.tar.gz"
    fi
fi

cd ..
rm -rf $BUILD_DIR

unset RECSDK_VERSION
echo "Done!"