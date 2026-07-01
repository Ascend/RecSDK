#!/bin/bash

set -e
rm -rf build
mkdir -p build

# 默认A2 (v220)，支持传入c310参数编译A5版本
BUILD_VER=${1:-v220}

# 验证版本参数
if [ "${BUILD_VER}" != "v220" ] && [ "${BUILD_VER}" != "c310" ]; then
    echo "ERROR: Unknown BUILD_VER:${BUILD_VER}"
    echo "Supported versions: v220, c310"
    echo "Usage: bash $0 [v220|c310]"
    exit 1
fi

echo "Building version: ${BUILD_VER} (A5: c310, A2/A3: v220)"

cmake -B build -DBUILD_VER="${BUILD_VER}"
cmake --build build -j
chmod 550 ./build/*.so
