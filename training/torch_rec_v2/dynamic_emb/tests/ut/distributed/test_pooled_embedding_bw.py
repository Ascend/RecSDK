# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import random
import sys
from typing import Dict, List

import torch
import torch.distributed as dist
import torchrec
from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbTableOptions,
)
from dynamic_emb.distributed.optimizers.base_dynamicemb_optimizer import EmbOptimType
from dynamic_emb.distributed.batched_dynamicemb_table import BatchedDynamicEmbeddingTablesV2
from torch.distributed.elastic.multiprocessing.errors import record


def str2poolingmode(v):
    if v.lower() in ("sum"):
        return DynamicEmbPoolingMode.SUM
    elif v.lower() in ("mean"):
        return DynamicEmbPoolingMode.MEAN
    else:
        raise argparse.ArgumentTypeError("Only sum and mean is supported for pooled embedding")


def table_idx_to_name(i):
    return f"t_{i}"


def feature_idx_to_name(i):
    return f"cate_{i}"


def generate_sparse_feature(
    feature_num, num_embeddings_list, multi_hot_sizes, local_batch_size=50, device=torch.device(f"npu:{0}")
):
    feature_batch = feature_num * local_batch_size

    indices = []
    lengths = []

    for i in range(feature_batch):
        f = i // local_batch_size
        cur_bag_size = random.randint(0, multi_hot_sizes[f])
        cur_bag = set({})
        while len(cur_bag) < cur_bag_size:
            cur_bag.add(random.randint(0, num_embeddings_list[f] - 1))

        indices.extend(list(cur_bag))
        lengths.append(cur_bag_size)

    return torchrec.KeyedJaggedTensor(
        keys=[feature_idx_to_name(feature_idx) for feature_idx in range(feature_num)],
        values=torch.tensor(indices, dtype=torch.int64).to(device),  # key [0,1] on rank0, [2] on rank 1
        lengths=torch.tensor(lengths, dtype=torch.int64).to(device),
    )


def count_tensor_to_dict(x, d):
    x = x.to("cpu")
    for i in x:
        key = i.item()
        if key not in d:
            d[key] = 1
        else:
            d[key] += 1


def test(args):
    backend = "hccl"
    dist.init_process_group(backend=backend)

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")

    table_num = args.num_embedding_table
    dim = args.embedding_dim
    embedding_dtype = torch.float32
    total_hbm_for_values = 1023 * 3
    table_options = [
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=embedding_dtype,
            dim=dim,
            max_capacity=num_emb,
            # Align with C++ DynamicEmbTable (next_power_of_2 on max_capacity).
            local_hbm_for_values=total_hbm_for_values,
            bucket_capacity=128,
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.DEBUG,
            ),
        )
        for num_emb in args.num_embeddings_per_feature
    ]

    var = BatchedDynamicEmbeddingTablesV2(
        table_options=table_options,
        output_dtype=torch.float32,
        table_names=[table_idx_to_name(i) for i in range(table_num)],
        pooling_mode=args.pooling_mode,
        optimizer=EmbOptimType.SGD,
        learning_rate=1.0,
        device=device,
    )

    num_iterations = args.num_iterations

    local_batch_size = args.batch_size // world_size
    for i in range(num_iterations):
        sparse_feature = generate_sparse_feature(
            feature_num=table_num,
            num_embeddings_list=args.num_embeddings_per_feature,
            multi_hot_sizes=args.multi_hot_sizes,
            local_batch_size=local_batch_size,
            device=device,
        )
        indices = sparse_feature.values()
        offsets = sparse_feature.offsets()
        # Cache on CPU before forward/backward. torchrec KeyedJaggedTensor.split()
        # on NPU after custom ops can trigger ACL stream synchronize failures.
        values_cpu = indices.detach().cpu()
        offsets_cpu = offsets.detach().cpu()

        res = var(indices, offsets)
        torch.npu.synchronize()
        grad = torch.ones_like(res)
        res.backward(grad)
        torch.npu.synchronize()

        with torch.no_grad():
            ref_dicts: List[Dict[int, int]] = [{} for _ in range(table_num)]
            lengths = []
            for f, dict_ in enumerate(ref_dicts):
                start = offsets_cpu[f * local_batch_size].item()
                end = offsets_cpu[(f + 1) * local_batch_size].item()
                indices_per_table = values_cpu[start:end]
                count_tensor_to_dict(indices_per_table, dict_)
                lengths.append(indices_per_table.numel())
                assert indices_per_table.dim() == 1

            assert res.dim() == 2
            assert res.size()[0] == local_batch_size
            assert res.size()[1] == table_num * dim
    dist.barrier()
    dist.destroy_process_group()


@record
def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(description="Dynamic sequence embedding's backward")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=32,
        help="number of iterations",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default="608,1072,7184,1968",
        help="Comma separated max index per sparse feature (embedding table capacity). "
        "Defaults are sized for single-NPU UT; use larger values only when device HBM allows.",
    )
    parser.add_argument(
        "--multi_hot_sizes",
        type=str,
        default="20,17,101,49",
        help="Comma separated multihot size per sparse feature. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--emb_precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16", "fp8"],
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adagrad",
        choices=["sgd", "adagrad", "rowwise_adagrad"],
        help="optimizer type.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--pooling_mode",
        type=str2poolingmode,
        default=DynamicEmbPoolingMode.SUM,
        help="Pooling mode of dynamic embedding bag.",
    )

    args = parser.parse_args()
    args.num_embeddings_per_feature = [int(v) for v in args.num_embeddings_per_feature.split(",")]
    args.multi_hot_sizes = [int(v) for v in args.multi_hot_sizes.split(",")]
    args.num_embedding_table = len(args.num_embeddings_per_feature)

    # Print all arguments
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    test(args)


if __name__ == "__main__":
    main(sys.argv[1:])
