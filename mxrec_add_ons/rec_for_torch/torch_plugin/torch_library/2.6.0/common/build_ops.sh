#!/bin/bash

set -e
rm -rf build
mkdir -p build
cmake -B build
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
