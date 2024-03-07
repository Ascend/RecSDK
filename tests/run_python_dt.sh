#!/bin/bash

# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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

CUR_PATH=$(cd "$(dirname "$0")" || { warn "Failed to check path/to/run_python_dt.sh" ; exit ; } ; pwd)
TOP_PATH="${CUR_PATH}"/../

ARCH="$(uname -m)"
if [ $ARCH == "aarch64" ]; then
  export LD_PRELOAD=/usr/local/gcc7.3.0/lib64/libgomp.so.1
fi

# build mxRec and get output directory
pip3 install setuptools==65.6.3
bash "$TOP_PATH"/build/build_tf1_with_opensource.sh

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
    pytest --cov="${CUR_PATH}"/../mx_rec --cov-report=html --cov-report=xml --junit-xml=./final.xml --html=./final.html --self-contained-html --durations=5 -vv --cov-branch
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
