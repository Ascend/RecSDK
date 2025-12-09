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

import sys
import os

root_path = os.path.abspath(__file__)
root_path = os.path.sep.join(root_path.split(os.path.sep)[:-2])
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from easydict import EasyDict as edict
from datasets.fake_data import TestFakeDataHandler
from datasets.squad import load_data
from utils.handler import ModelHandler, get_params, get_opts, set_all_seed
from utils.logger import logger

# 禁用flash-attention大算子，会影响gpu融合性能
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)  # 强制使用数学模式

class BertHandler(nn.Module):
    def __init__(self, params: edict):
        super().__init__()
        self.params = params
        self.reuse_hash = None
        self.model = BertModel.from_pretrained(params.model_dir)


    def forward(self, features, mode="train"):
        feeds = []
        for _, val in features.items():
            feeds.append(val.to(self.params.device))
        res = self.model(*feeds)
        return {'res': res}

    def loss(self, pred, labels):
        loss = 0
        return loss

if __name__ == "__main__":
    params = get_params()
    params.update(
        edict(
            {
                "inputs": 'input_ids.1,attention_mask.1,token_type_ids.1',
                "input_shapes": {
                    'input_ids.1': [1, 384],
                    'attention_mask.1': [1, 384],
                    'token_type_ids.1': [1, 384],
                },
                "input_type": 'LONG',
                "model": "bert",
            }
        )
    )
    params = get_opts(sys.argv, params)
    set_all_seed(params)

    model = BertHandler(params).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    handler = ModelHandler(
        params, model, optimizer, load_data, TestFakeDataHandler(params)
    )
    handler.run()
