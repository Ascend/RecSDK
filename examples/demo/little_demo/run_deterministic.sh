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

export USE_DETERMINISTIC=1

sh run.sh main.py | tee log

grep -rn "loss" log | grep "1,0" | awk '{print $NF}'> loss

rm -f log

soc_name=`python3 -c 'import acl;print(acl.get_soc_name())'`
echo "soc_name: $soc_name"

loss_file=deterministic_loss/loss${soc_name:10:1}

if [ ! -e $loss_file ];then
    echo "$loss_file file does not exist"
    rm -f loss
    exit
fi


diff $loss_file loss

if [ $? -eq 0 ]; then
  echo "deterministic loss check passed"
else
  echo "deterministic loss check failed"
fi

rm -f loss
