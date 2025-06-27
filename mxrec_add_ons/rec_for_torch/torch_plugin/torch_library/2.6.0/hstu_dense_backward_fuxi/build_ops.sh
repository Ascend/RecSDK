#!/bin/bash

set -e
rm -rf build
mkdir -p build
cmake -B build
cmake --build build -j
export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/hstu_dense_backward_fuxi/op_api/lib:$LD_LIBRARY_PATH