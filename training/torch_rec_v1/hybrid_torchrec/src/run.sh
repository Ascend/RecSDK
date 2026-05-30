#!/usr/bin/env bash

rm -rf build
mkdir build
cd build
export CMAKE_BUILD_PARALLEL_LEVEL=24
torch_path=$(python3 -m pip show torch 2>/dev/null | grep -E "^Location:" | awk '{print $2}')/torch/share/cmake
echo "======== current torch_path is: ${torch_path} ========="
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${torch_path}" ..
cmake  --build . --config Release
