#!/usr/bin/env bash

rm -rf build
mkdir build
cd build
export CMAKE_BUILD_PARALLEL_LEVEL=24
torch_path=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${torch_path}" ..
cmake  --build . --config Release