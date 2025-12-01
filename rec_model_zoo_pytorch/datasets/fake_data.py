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

import numpy as np
import torch
from utils.handler import TestHandler


INPUT_TYPE = {
    'LONG': int,
    'FLOAT32': np.float32,
}


class TestFakeDataHandler(TestHandler):
    def __init__(self, params):
        super().__init__(params)

    def generate_data(self, batch_size):
        features = {}
        device = self.params.device
        for key, val in self.params.input_shapes.items():
            val = [val[0] * batch_size] + val[1:]
            features[key] = np.random.random(size=val).astype(INPUT_TYPE[self.params.input_type])
            features[key] = torch.tensor(features[key]).to(device)
        return features