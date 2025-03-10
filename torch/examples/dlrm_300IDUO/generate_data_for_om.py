# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
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

import os
from absl import app, flags, logging

import numpy as np
import torch_npu

from dlrm.data.feature_spec import FeatureSpec
from dlrm.data.data_loader import get_data_loaders
from dlrm.data.utils import prefetcher, get_embedding_sizes
from dlrm.scripts.main import FLAGS
from dlrm.utils import distributed as dist
from dlrm.utils.distributed import get_device_mapping
import torch


flags.DEFINE_string("output_dir", "./", "Output directory for test mode")


def load_feature_spec():
    fspec_path = os.path.join(FLAGS.dataset, FLAGS .feature_spec)
    return FeatureSpec.from_yaml(fspec_path)


def batch_dataloader():
    feature_spec = load_feature_spec(FLAGS)
    use_gpu = "cpu" not in FLAGS.base_device.lower()
    _, world_size, _ = dist.init_distributed(backend=FLAGS.backend, use_gpu=use_gpu)
    world_embedding_sizes = get_embedding_sizes(feature_spec, FLAGS.max_table_size)
    device_mapping = get_device_mapping(world_embedding_sizes, num_gpus=world_size)
    _, data_loader_test = get_data_loaders(FLAGS, device_mapping=device_mapping,
                                                    feature_spec=feature_spec)
    return data_loader_test, world_size


def main(argv):
    torch.npu.set_compile_mode(jit_compile=False)

    dataloader, world_size = batch_dataloader()

    batch_size_per_gpu = [FLAGS.test_batch_size // world_size for _ in range(world_size)]
    test_batch_sizes = sum(batch_size_per_gpu)

    if FLAGS.test_batch_size != test_batch_sizes:
        logging.error("Batch size must be divisible by the number of GPUs")
        return
    
    data_stream = torch.cuda.Stream()
    batch_iter = prefetcher(iter(dataloader), data_stream)

    for count, step in enumerate(dataloader):
        numerical_features, categorical_features, click = next(batch_iter)
        torch.cuda.synchronize()

        if click.shape[0] != test_batch_sizes:
            last_batch_size = click.shape[0]
            padding_size = test_batch_sizes - last_batch_size

            if numerical_features is not None:
                padding_numerical = torch.empty(
                    padding_size, 
                    numerical_features.shape[1], 
                    device=numerical_features.device, 
                    dtype=numerical_features.dtype)
                numerical_features = torch.cat([numerical_features, padding_numerical], dim=0)

            if categorical_features is not None:
                padding_categorical = torch.ones(
                    padding_size, categorical_features.shape[1], 
                    device=categorical_features.device, dtype=categorical_features.dtype)
                categorical_features = torch.cat([categorical_features, padding_categorical], dim=0)

        output_dir = FLAGS.output_dir
        if not os.path.exists(os.path.join(output_dir, "numerical_features")):
            os.makedirs(os.path.join(output_dir, "numerical_features"))
        if not os.path.exists(os.path.join(output_dir, "categorical_features")):
            os.makedirs(os.path.join(output_dir, "categorical_features"))
        if not os.path.exists(os.path.join(output_dir, "click")):
            os.makedirs(os.path.join(output_dir, "click"))
        numerical_features_array = numerical_features.numpy()
        categorical_features_array = categorical_features.numpy()
        click_array = click.numpy()
        np.save(os.path.join(output_dir, 
                             "numerical_features", 
                             f"numerical_features_{count}.npy"), 
                             numerical_features_array)
        np.save(os.path.join(output_dir, 
                             "categorical_features", 
                             f"categorical_features_{count}.npy"), 
                             categorical_features_array)
        np.save(os.path.join(output_dir, 
                             "click", 
                             f"click_{count}.npy"), 
                             click_array)


if __name__ == '__main__':
    app.run(main)