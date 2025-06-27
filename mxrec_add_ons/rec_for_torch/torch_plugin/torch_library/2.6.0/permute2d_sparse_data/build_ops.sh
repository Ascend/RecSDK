#!/bin/bash

set -e
rm -rf build
mkdir -p build
cmake -B build
cmake --build build -j
export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/permute2d_sparse_data/op_api/lib:$LD_LIBRARY_PATH