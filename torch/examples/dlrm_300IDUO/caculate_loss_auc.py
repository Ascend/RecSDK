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

import logging
from absl import app, flags

import torch_npu


import dlrm.scripts.utils as utils
import torch

FLAGS = flags.FLAGS

flags.DEFINE_string("path", None, "Path to the model")


loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def caculate(argv):
    path = FLAGS.path
    output_click_lst = torch.load(path)
    for i, output_click in enumerate(zip(*output_click_lst)):
        y_true, y_score = output_click
        y_score = torch.sigmoid(y_score).float()
        auc = utils.roc_auc_score(y_true, y_score)
        loss = loss_fn(y_score, y_true).item()
        logging.info(f"step {i}, loss: {loss}, auc: {auc}")


if __name__ == "__main__":
    app.run(caculate)