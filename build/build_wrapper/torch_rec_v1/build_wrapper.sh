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
# 保存脚本所在目录的绝对路径（相对于 torchrec_npu 的位置）
SCRIPT_BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_BASE_DIR/resources" || { echo "Error: Cannot cd to resources"; exit 1; }

# 计算各目录的绝对路径（4个..回到dockerfiles目录）
TORCHREC_NPU_DIR="$SCRIPT_BASE_DIR/../../../training/torch_rec_v1/torchrec_npu"
HYBRID_TORCHREC_DIR="$SCRIPT_BASE_DIR/../../../training/torch_rec_v1/hybrid_torchrec"
TORCHREC_EMBCACHE_DIR="$SCRIPT_BASE_DIR/../../../training/torch_rec_v1/torchrec_embcache"
TORCHREC_BUILD_DIR="$SCRIPT_BASE_DIR/../../../training/torch_rec_v1/torchrec_npu/torchrec_1.1.0_build"
TORCHREC_BUILD_DIR_120="$SCRIPT_BASE_DIR/../../../training/torch_rec_v1/torchrec_npu/torchrec_1.2.0_build"

BUILD_DIR="$SCRIPT_BASE_DIR/resources/torch_rec_builder"
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
cp ../../_setup_common.py "$BUILD_DIR/" || { echo "Error: Cannot find _setup_common.py"; exit 1; }
echo "Copied: _setup_common.py"

# 1. 准备开源依赖 (opensource)
OPENSOURCE_DIR="../../../../../opensource"
if [ ! -d "$OPENSOURCE_DIR" ]; then
    echo "Downloading opensource dependencies to $OPENSOURCE_DIR ..."
    mkdir -p "$OPENSOURCE_DIR"
    wget -q https://github.com/pybind/pybind11/archive/refs/tags/v2.10.3.zip -O "$OPENSOURCE_DIR/pybind11-2.10.3.zip" || { echo "Error: Failed to download pybind11-2.10.3.zip"; exit 1; }
    wget -q https://github.com/huaweicloud/huaweicloud-sdk-c-obs/archive/refs/tags/v3.23.9.zip -O "$OPENSOURCE_DIR/huaweicloud-sdk-c-obs-3.23.9.zip" || { echo "Error: Failed to download huaweicloud-sdk-c-obs-3.23.9.zip"; exit 1; }
    echo "Successfully downloaded opensource dependencies."
else
    echo "Opensource dependencies found in $OPENSOURCE_DIR."
fi

# 2. 编译 SDK Python 包
echo "Start compiling TorchRec SDK..."

# ========== Part 1: torchrec 1.1.0 + pt2.6.0 虚拟环境 ==========
echo "===== Part 1: Building torchrec 1.1.0 for pt2.6.0 ====="

# 删除旧目录，确保干净环境
rm -rf $TORCHREC_NPU_DIR/torchrec
echo "Cloning torchrec v1.1.0..."
cd $TORCHREC_NPU_DIR
git clone -b release/v1.1.0 https://github.com/pytorch/torchrec.git
(cd $TORCHREC_NPU_DIR/torchrec && git checkout 2c5f6ee)
echo "Set CANN environment to a2 for torchrec 1.1.0 compilation..."
source /usr/local/set_cann_env.sh a2
(cd $TORCHREC_NPU_DIR && bash build_whl_torchrec1.1.0.sh)

# 找到 torchrec whl 并安装到 pt2.6.0 虚拟环境
TORCHREC_110_WHL=$(find $TORCHREC_NPU_DIR/torchrec/dist -name "torchrec-1.1.0*.whl" | head -n 1)
if [ -z "$TORCHREC_110_WHL" ]; then
    echo "Error: torchrec-1.1.0 whl not found."
    exit 1
fi
echo "Installing torchrec 1.1.0 to pt2.6.0..."
source /opt/buildtools/torch_v1_pt2.6.0/bin/activate
pip install $TORCHREC_110_WHL
deactivate

# 编译 hybrid/embcache (会自动检测 torchrec 版本为 1.1.0)
echo "Compiling hybrid/embcache with pt2.6.0 (torchrec 1.1.0)..."
echo "Set CANN environment to a2 for hybrid/embcache compilation..."
source /usr/local/set_cann_env.sh a2
source /opt/buildtools/torch_v1_pt2.6.0/bin/activate
(cd $HYBRID_TORCHREC_DIR && bash build_whl.sh)
deactivate

# 移动产物到临时目录 pt2.6_whl
mkdir -p $BUILD_DIR/mindxsdk-torchrec/pt2.6_whl
mv $HYBRID_TORCHREC_DIR/dist/*.whl $BUILD_DIR/mindxsdk-torchrec/pt2.6_whl/ 2>/dev/null || true
mv $TORCHREC_NPU_DIR/torchrec/dist/*.whl $BUILD_DIR/mindxsdk-torchrec/pt2.6_whl/ 2>/dev/null || true

# 清理源码
rm -rf $TORCHREC_NPU_DIR/torchrec

# ========== Part 2: torchrec 1.2.0 + pt2.7.1 虚拟环境 ==========
echo "===== Part 2: Building torchrec 1.2.0 for pt2.7.1 ====="

rm -rf $TORCHREC_NPU_DIR/torchrec_v1.2.0
echo "Cloning torchrec v1.2.0..."
cd $TORCHREC_NPU_DIR
git clone -b release/v1.2.0 https://github.com/pytorch/torchrec.git torchrec_v1.2.0
(cd $TORCHREC_NPU_DIR/torchrec_v1.2.0 && git checkout 5db1a21)

# 创建软链接让脚本能找到目录
ln -sf torchrec_v1.2.0 torchrec
echo "Set CANN environment to a2 for torchrec 1.2.0 compilation..."
source /usr/local/set_cann_env.sh a2
(cd $TORCHREC_NPU_DIR && bash build_whl_torchrec1.2.0.sh)

# 找到 torchrec whl 并安装到 pt2.7.1 虚拟环境
TORCHREC_120_WHL=$(find $TORCHREC_NPU_DIR/torchrec/dist -name "torchrec-1.2.0*.whl" | head -n 1)
if [ -z "$TORCHREC_120_WHL" ]; then
    echo "Error: torchrec-1.2.0 whl not found."
    exit 1
fi
echo "Installing torchrec 1.2.0 to pt2.7.1..."
source /opt/buildtools/torch_v1_pt2.7.1/bin/activate
pip install $TORCHREC_120_WHL
deactivate

# 编译 hybrid/embcache (会自动检测 torchrec 版本为 1.2.0)
echo "Compiling hybrid/embcache with pt2.7.1 (torchrec 1.2.0)..."
echo "Set CANN environment to a2 for hybrid/embcache compilation..."
source /usr/local/set_cann_env.sh a2
source /opt/buildtools/torch_v1_pt2.7.1/bin/activate
(cd $HYBRID_TORCHREC_DIR && bash build_whl.sh)
deactivate

# 移动产物到临时目录 pt2.7_whl
mkdir -p $BUILD_DIR/mindxsdk-torchrec/pt2.7_whl
mv $HYBRID_TORCHREC_DIR/dist/*.whl $BUILD_DIR/mindxsdk-torchrec/pt2.7_whl/ 2>/dev/null || true
mv $TORCHREC_NPU_DIR/torchrec/dist/*.whl $BUILD_DIR/mindxsdk-torchrec/pt2.7_whl/ 2>/dev/null || true

# 清理
rm -rf $TORCHREC_NPU_DIR/torchrec
rm -rf $TORCHREC_NPU_DIR/torchrec_v1.2.0

# 3. 生成统一的 tar.gz 发布包
echo "Start building..."

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
        mv "$TAR_FILE" "../../../../output/torch_rec_v1-${SDK_VERSION}-linux_${SUFFIX}.tar.gz"
    fi
fi

cd ..
rm -rf $BUILD_DIR

unset RECSDK_VERSION
echo "Done!"
