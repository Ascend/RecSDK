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


class Loss:
    def __init__(self, data_dir, data_step, rank_id):
        self.path = os.path.join(data_dir, "03dump_loss", f"{rank_id}_rank_loss.json")
        self.loss_dict = parse_json_to_dict(self.path)
        self.loss_value = self.loss_dict[str(data_step)]

    def __eq__(self, other) -> bool:
        logging.info(f"[Loss] comparison start......")
        if not isinstance(other, Loss):
            logging.error(
                f"[Loss] comparison must between Loss, but {other.__class__} is given"
            )
            return False

        if self.loss_value != other.loss_value:
            return False
        return True
