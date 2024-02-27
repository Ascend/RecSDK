# coding: UTF-8
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

import numpy as np

from mx_rec.util.communication.hccl_ops import get_rank_id
from mx_rec.util.log import logger


def get_data_generator(config, batch_number):
    rank_id = get_rank_id()

    def data_generator():
        i = 0
        while i < batch_number:
            item_ids = np.random.randint(0, config.item_range, (config.batch_size, config.item_feat_cnt))
            user_ids = np.random.randint(0, config.user_range, (config.batch_size, config.user_feat_cnt))
            category_ids = np.random.randint(0, config.category_range, (config.batch_size, config.category_feat_cnt))
            label_0 = np.random.randint(0, 2, (config.batch_size,))
            label_1 = np.random.randint(0, 2, (config.batch_size,))

            yield {"item_ids": item_ids,
                   "user_ids": user_ids,
                   "category_ids": category_ids,
                   "label_0": label_0,
                   "label_1": label_1}
            i += 1

        logger.debug(f"================ end of data generator for {config.task_name} task | rank id {rank_id} "
                     f"================")

    return data_generator


def get_large_scale_data_generator(config):
    def data_generator():
        i = 0
        while True:
            id_list = [np.random.randint(0, config.vocabulary_size, (config.batch_size,))
                       for _ in range(config.lookup_count)]

            data_block = dict(zip(config.tensor_name_list, id_list))

            label_0 = np.random.randint(0, 2, (config.batch_size,))
            label_1 = np.random.randint(0, 2, (config.batch_size,))
            data_block["label_0"] = label_0
            data_block["label_1"] = label_1

            logger.debug(f"================ generate NO.{i} step ================")
            yield data_block
            i += 1

    return data_generator
