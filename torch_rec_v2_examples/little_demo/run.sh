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

python_path=$(dirname "$(dirname "$(which python3.11)")")/lib/python3.11/site-packages
rec_package_path=${python_path}/dynamic_emb
so_path=${python_path}
common_package_path=${python_path}/rec_sdk_common
common_so_path=${common_package_path}/lib
export PYTHONPATH=${rec_package_path}:${so_path}:${common_so_path}:$PYTHONPATH
export LD_LIBRARY_PATH=${so_path}:${common_so_path}:/usr/local/lib:$LD_LIBRARY_PATH

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:6000 --nnodes=1 --nproc-per-node=1 main.py --train "$@"
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:6000 --nnodes=1 --nproc-per-node=1 main.py --load --dump "$@"