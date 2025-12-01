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
from torch.utils.data import Dataset, DataLoader
from datasets.open_imagenet.data_loader import ImagenetLoader

INPUT_TYPE = {
    'LONG': int,
    'FLOAT32': np.float32
}


class ImagenetDataset(Dataset):
    def __init__(self, params):
        self.params = params
        self.data_loader = ImagenetLoader({
            'framework': 'Pytorch',
            'model': params.model,
            'inputs': params.inputs,
            'dataset_path': params.data_dir,
        })
        self.data_loader.preprocess()
        self.length = self.data_loader.items

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_features, batch_labels = self.data_loader.get_samples(idx)
        for key, value in batch_features.items():
            batch_features[key] = torch.tensor(value).to(self.params.device)
        return batch_features, torch.tensor(batch_labels).to(self.params.device)


def load_data(params):
    dataset = ImagenetDataset(params)
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
    return None, loader, loader