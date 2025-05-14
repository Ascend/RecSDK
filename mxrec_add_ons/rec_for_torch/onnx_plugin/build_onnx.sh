#!/bin/bash
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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
    wget -t 3 -nc --timeout=10 https://github.com/nlohmann/json/archive/v3.9.1.tar.gz --no-check-certificate
    tar -xvf v3.9.1.tar.gz
    cp json-3.9.1/single_include/nlohmann/json.hpp .
}

if [ ! -e "$JSON_FILE" ]; then
    get_nlohmann
fi