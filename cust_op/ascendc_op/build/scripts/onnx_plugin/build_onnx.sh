#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
JSON_FILE=$SCRIPT_DIR/json.hpp

function get_nlohmann()
{
    cd $SCRIPT_DIR
    if [ ! -e "v3.9.1.tar.gz" ]; then
        echo "The required component 'v3.9.1.tar.gz' for the ONNX plugin does not exist."
    else
        tar -xvf v3.9.1.tar.gz
        cp json-3.9.1/single_include/nlohmann/json.hpp .
    fi
}

if [ ! -e "$JSON_FILE" ]; then
    get_nlohmann
fi