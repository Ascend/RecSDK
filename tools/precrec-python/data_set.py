#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import logging
import sys
import os
import re
from typing import Dict, List

import numpy as np

from utils import validate_path


DATASET_REGEX_STR = r"^data_rank_\d+_batch_\d+_.+\.npy$"
DATA_SET_NUMPY_ATOL = "1e-10"


class BatchDataSet:
    def __init__(self, data_dir:str, data_step:int, rank_id:int):
        self.batch_data_path = os.path.join(data_dir, "01dump_dataset")

        data_pattern_str = (
            r"data_rank_"
            f"{re.escape(str(rank_id))}/"  # rankid
            r"_batch_"
            f"{re.escape(str(data_step))}"  # step
            r"_+\.npy$"
        )
        self.data_pattern = re.compile(data_pattern_str)

        self.batch_data_name_list = self.get_dump_data_names()
        self.bathc_data = self.parse_batch_data()

    def __eq__(self, other) -> bool:
        logging.info(f"[BatchDataSet] comparison start......")
        if not isinstance(other, BatchDataSet):
            logging.error(
                f"[BatchDataSet] comparison must between BatchDataSet, but {other.__class__} is given"
            )
            return False

        if self.batch_data_name_list != other.batch_data_name_list:
            logging.error(
                f"[BatchDataSet] comparison must content same batch data files, "
                f"Test: {self.batch_data_name_list} Gold: {other.batch_data_name_list}"
            )
            return False

        for batch_data_name in self.batch_data_name_list:
            test_data = self.bathc_data[batch_data_name]
            golden_data = other.bathc_data[batch_data_name]
            if not np.allclose(test_data, golden_data, rtol=float(DATA_SET_NUMPY_ATOL)):
                logging.error(
                    f"[BatchDataSet] data different, batch data name {batch_data_name}"
                    f"Test data: {test_data} Gold data: {golden_data}"
                )
                return False

        return True

    def get_dump_data_names(self) -> List[str]:
        dataset_file_names = []
        for dirpath, _, files in os.walk(self.batch_data_path):
            for filename in files:
                file_path = os.path.join(dirpath, filename)
                validate_path(file_path, "dump_dataset", DATASET_REGEX_STR)
                if self.data_pattern.match(filename):
                    logging.debug(f"Find matched DataSet op path: {file_path}")
                dataset_file_names.append(filename)
        dataset_file_names = sorted(dataset_file_names)
        return dataset_file_names

    def parse_batch_data(self) -> Dict[str, np.array]:
        batch_data_dict = {}
        for batch_data_name in self.batch_data_name_list:
            batch_data_path = os.path.join(self.batch_data_path, batch_data_name)
            batch_data = np.load(batch_data_path)
            batch_data_dict[batch_data_name] = batch_data
        return batch_data_dict
