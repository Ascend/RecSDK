# coding=utf-8
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

import os
import numpy as np
from glob import glob


def split_auc(log_input):
    with open(log_input, 'r') as log:
        all_auc = []
        for line in log.readlines():
            if 'Test' in line:
                all_auc.append(float(line.split(';')[0].split(':')[-1].strip()))
    all_auc_len = len(all_auc)
    all_auc_arr = np.array(all_auc)[:all_auc_len - all_auc_len%8]
    test_auc = np.mean(all_auc_arr.reshape(-1, 8), axis=-1)
    return test_auc


log_path_all = 'latest_*.log'
log_path_list = glob(log_path_all)

for log_path in log_path_list:
    print(os.path.basename(log_path))
    print(split_auc(log_path))
    print('*'*20)