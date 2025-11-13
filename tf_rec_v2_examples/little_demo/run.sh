#!/usr/bin/env python3
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

rec_package_path=$(dirname "$(dirname "$(which python3.7)")")/lib/python3.7/site-packages/mxrec
so_path=${rec_package_path}/librec
common_package_path=$(dirname "$(dirname "$(which python3.7)")")/lib/python3.7/site-packages/rec_sdk_common
common_so_path=${common_package_path}/lib
export PYTHONPATH=${so_path}:${common_so_path}:$PYTHONPATH
export LD_LIBRARY_PATH=${so_path}:${common_so_path}:/usr/local/lib:$LD_LIBRARY_PATH

toml_path=$1

# Distributed config.
interface=$(grep -oP 'interface = "\K[^"]+' ${toml_path})
local_rank_size=$(grep -oP 'local_rank_size = \K\d+' ${toml_path})
num_server=$(grep -oP 'num_server = \K\d+' ${toml_path})
num_process=$((num_server * local_rank_size))
mpi_args='-x BIND_INFO="0:12 12:48 60:48" -bind-to none -x NCCL_SOCKET_IFNAME=docker0 -mca btl_tcp_if_exclude docker0'

# CANN config.
export JOB_ID=10086
export HCCL_ALGO="level0:NA;level1:pipeline"
# Please configure according to the actual situation.
use_ranktable=$(grep -oP '^use_ranktable\s*=\s*\Ktrue' ${toml_path})
if [ "$use_ranktable" == "true" ]; then
  ranktable_path=$(pwd)/../ranktable_samples/ranktable_${num_process}p.json
  topo_path=$(pwd)/../ranktable_samples/topo_${num_process}p.json
  export RANK_TABLE_FILE=${ranktable_path}
  export HCCL_TOPO_FILE_PATH=${topo_path}
  echo "RANK_TABLE_FILE=$RANK_TABLE_FILE"
  echo "HCCL_TOPO_FILE_PATH=$HCCL_TOPO_FILE_PATH"
fi

horovodrun --network-interface ${interface} -np ${num_process} --mpi-args "${mpi_args}" --mpi -H localhost:${local_rank_size} \
python3.7 main.py -p ${toml_path} 2>&1 | tee temp_${num_process}p_$(date +%Y%m%d_%H%M%S).log
