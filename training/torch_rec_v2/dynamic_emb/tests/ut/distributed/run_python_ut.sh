#!/usr/bin/env bash
# -*- coding: utf-8 -*-
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

dynamic_package_path=$(dirname "$(dirname "$(which python3.11)")")/lib/python3.11/site-packages/dynamic_emb
so_path=$(dirname ${dynamic_package_path})
common_package_path=$(dirname "$(dirname "$(which python3.11)")")/lib/python3.11/site-packages/rec_sdk_common
common_so_path=${common_package_path}/lib
export PYTHONPATH=${so_path}:${common_so_path}:$PYTHONPATH
export LD_LIBRARY_PATH=${so_path}:${common_so_path}:/usr/local/lib:$LD_LIBRARY_PATH

if [ -d "result" ]; then
    rm -rf result
fi

mkdir result

pytest --cov=$dynamic_package_path \
--cov-config ./.coveragerc --cov-report=html --cov-report=xml --junit-xml=./final.xml \
--html=./final.html --self-contained-html --durations=5 -vv --cov-branch

coverage xml -i --omit="*/tests/*"
cp coverage.xml final.xml final.html ./result
cp -r htmlcov ./result
rm -rf coverage.xml final.xml final.html htmlcov
