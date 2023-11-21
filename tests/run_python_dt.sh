#!/bin/bash

# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
# Description: start script.
# Author: MindX SDK
# Create: 2023
# History: NA

set -e

CUR_PATH=$(cd "$(dirname "$0")" || { warn "Failed to check path/to/run_python_dt.sh" ; exit ; } ; pwd)
TOP_PATH="${CUR_PATH}"/../

# build mxRec and get output directory
bash "$TOP_PATH"/build/build_tf1.sh

# create libasc directory and copy so files into it
cd "$TOP_PATH"/mx_rec
mkdir -p libasc
cp -f "$TOP_PATH"/output/*.so ./libasc
cd -

# set environment variable
export PYTHONPATH="${TOP_PATH}"/output:$PYTHONPATH
export LD_LIBRARY_PATH="${TOP_PATH}"/output:/usr/local/lib:$LD_LIBRARY_PATH

rm -rf result
mkdir -p result

function run_test_cases() {
    echo "Get testcases final result."
    pytest --cov="${CUR_PATH}"/../mx_rec --cov-report=html --cov-report=xml --junit-xml=./final.xml --html=./final.html --self-contained-html --durations=5 -vv
    coverage xml -i --omit="build/*,cust_op/*,src/*"
    cp coverage.xml final.xml final.html ./result
    cp -r htmlcov ./result
    rm -rf coverage.xml final.xml final.html htmlcov
}

echo "************************************* Start MxRec LLT Test *************************************"
start=$(date +%s)
run_test_cases
ret=$?
end=$(date +%s)
echo "*************************************  End  MxRec LLT Test *************************************"
echo "LLT running take: $(expr "${end}" - "${start}") seconds"

rm -rf "$TOP_PATH"/mx_rec/libasc

exit "${ret}"
