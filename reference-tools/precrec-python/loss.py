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
import os

from utils import parse_json_to_dict


DUMP_LOSS_STR = "03dump_loss"


class Loss:
    """
    This class is used to represent parsed loss for choosen step and rank.
    """

    def __init__(self, data_dir, data_step, rank_id):
        self.path = os.path.join(data_dir, DUMP_LOSS_STR, f"{rank_id}_rank_loss.json")
        self.loss_dict = parse_json_to_dict(self.path)
        self.loss_value = self.loss_dict[str(data_step)]

    def __eq__(self, other) -> bool:
        logging.info("[Loss] comparison start......")
        target_class = other.__class__
        if not isinstance(other, Loss):
            logging.error(
                "[Loss]Comparison must between Loss, but %s is given", target_class
            )
            return False

        if self.loss_value != other.loss_value:
            logging.error(
                "[Loss]Loss value not equal, Test: %s, Golden: %s",
                self.loss_value,
                other.loss_value,
            )
            return False
        return True
