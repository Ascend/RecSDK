#!/bin/bash

set -e
rm -rf build
mkdir -p build
cmake -B build
cmake --build build -j
chmod 550 ./build/*.so
export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/expand_into_jagged_permute/op_api/lib:$LD_LIBRARY_PATH
