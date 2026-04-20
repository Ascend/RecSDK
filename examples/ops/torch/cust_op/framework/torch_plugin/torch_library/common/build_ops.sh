#!/bin/bash

set -e
rm -rf build
mkdir -p build

# 默认A2
if [ "$#" -eq 0 ]; then
    BUILD_VER="v220"
    echo "No version specified, using default: ${BUILD_VER}"
elif [ "$#" -eq 1 ]; then
    BUILD_VER=${1}
else
    echo "ERROR: Too many arguments. Usage: 'bash $0 [v220|c310]'"
    echo "If no argument is provided, default is c310"
    exit 1
fi

# 验证版本参数
if [ "${BUILD_VER}" == "v220" ] || [ "${BUILD_VER}" == "c310" ]; then
    echo "BUILD_VER: ${BUILD_VER}"
else
    echo "ERROR: Unknown BUILD_VER:${BUILD_VER}"
    echo "Supported versions: v220, c310"
    exit 1
fi

cmake -B build -DBUILD_VER="${BUILD_VER}"

cmake --build build -j
chmod 550 ./build/*.so
# 默认放在python3,site-package目录下
PACKAGE_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")
if [ -d "$PACKAGE_PATH" ]; then
  echo "build to: $PACKAGE_PATH"
  cp -a ./build/*.so ${PACKAGE_PATH}/
else
  echo "build to: ${PWD}/build"
fi
