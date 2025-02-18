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

from absl import app, flags

import numpy as np
import torch_npu


from dlrm.data.utils import get_embedding_sizes
from dlrm.model.distributed import DistributedDlrm
from dlrm.scripts.main import FLAGS
from dlrm.scripts.main import load_feature_spec, validate_flags
from dlrm.utils import distributed as dist
from dlrm.utils.checkpointing.distributed import make_distributed_checkpoint_loader
from dlrm.utils.distributed import get_device_mapping, is_main_process
import torch


def main(argv):
    torch.manual_seed(FLAGS.seed)
    torch.npu.set_compile_mode(jit_mode=False)

    use_gpu = "cpu" not in FLAGS.base_device.lower()
    rank, world_size, gpu = dist.init_distributed(backed=FLAGS.backend, use_gpu=use_gpu)
    device = FLAGS.base_device

    feature_spec = load_feature_spec(FLAGS)

    cat_feature_count = len(get_embedding_sizes(feature_spec, None))
    validate_flags(cat_feature_count)

    world_embedding_size = get_embedding_sizes(feature_spec, max_table_size=FLAGS.max_table_size)
    world_categorical_feature_sizes = np.asarray(world_embedding_size)
    device_mapping = get_device_mapping(world_embedding_size, num_gpus=world_size)

    categorical_feature_sizes = world_categorical_feature_sizes[device_mapping["embedding"][rank]].tolist()
    num_numerical_features = feature_spec.get_number_of_numerical_features()

    bottom_mlp_sizes = FLAGS.bottom_mlp_sizes if rank == device_mapping["bottom_mlp"] else None

    model = DistributedDlrm(
        vectors_per_gpu=device_mapping['vectors_per_gpu'],
        embedding_device_mapping=device_mapping['embedding'],
        embedding_type=FLAGS.embedding_type,
        embedding_dim=FLAGS.embedding_dim,
        world_num_categorical_features=len(world_categorical_feature_sizes),
        categorical_feature_sizes=categorical_feature_sizes,
        num_numerical_features=num_numerical_features,
        hash_indices=FLAGS.hash_indices,
        bottom_mlp_sizes=bottom_mlp_sizes,
        top_mlp_sizes=FLAGS.top_mlp_sizes,
        interaction_op=FLAGS.interaction_op,
        fp16=FLAGS.amp,
        use_cpp_mlp=FLAGS.optimized_mlp,
        bottom_features_ordered=FLAGS.bottom_features_ordered,
        device=device
    )

    dist.setup_distributed_print(is_main_process())

    checkpoint_loader = make_distributed_checkpoint_loader(device_mapping=device_mapping, rank=rank)

    if FLAGS.load_checkpoint_path:
        checkpoint_loader.load_checkpoint(model, FLAGS.load_checkpoint_path)
        model.to(device)

    def parallelize(model):
        if world_size <= 1:
            return model
        
        model.top_model = torch.nn.parallel.DistributedDataParallel(model.top_model)
        return model
    
    model = parallelize(model)
    model = model.eval()

    numerical_features_example = torch.zeros(1, 13, dtype=torch.float32)
    numerical_features_example = numerical_features_example.to(device)

    categorical_features_example = torch.zeros(1, 26, dtype=torch.int64)
    categorical_features_example = categorical_features_example.to(device)

    input_names = ["numerical_features", "categorical_features"]
    output_names = ["output"]

    dynamic_axes = {
        "numerical_features": {0: "batch_size"},
        "categorical_features": {0: "batch_size"},
        "output": {0: "batch_size"}
    }

    x = (numerical_features_example, categorical_features_example)

    torch.onnx.export(model, 
                      x, 
                      "output/dlrm.onnx", 
                      input_names=input_names, 
                      output_names=output_names, 
                      dynamic_axes=dynamic_axes,
                      opset_version=11
                    )
    

if __name__ == "__main__":
    app.run(main)