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
import math
import os
import random
import sys
from typing import Dict, List, Optional

import pytest
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def table_idx_to_name(i):
    return f"t_{i}"


def feature_idx_to_name(i):
    return f"cate_{i}"


def generate_sparse_feature(
    feature_num,
    num_embeddings_list,
    multi_hot_sizes,
    local_batch_size=50,
    device=torch.device("npu:0"),
):
    random.seed(42)
    feature_batch = feature_num * local_batch_size

    indices = []
    lengths = []

    for i in range(feature_batch):
        f = i // local_batch_size
        cur_bag_size = random.randint(0, multi_hot_sizes[f])
        cur_bag = set()
        while len(cur_bag) < cur_bag_size:
            cur_bag.add(random.randint(0, num_embeddings_list[f] - 1))

        indices.extend(list(cur_bag))
        lengths.append(cur_bag_size)

    return torchrec.KeyedJaggedTensor(
        keys=[feature_idx_to_name(feature_idx) for feature_idx in range(feature_num)],
        values=torch.tensor(indices, dtype=torch.int64).to(device),
        lengths=torch.tensor(lengths, dtype=torch.int64).to(device),
    )


def count_tensor_to_dict(x, d):
    x = x.to("cpu")
    for i in x:
        key = i.item()
        d[key] = d.get(key, 0) + 1


def indices_per_feature(values_cpu, offsets_cpu, feature_idx, local_batch_size):
    start = offsets_cpu[feature_idx * local_batch_size].item()
    end = offsets_cpu[(feature_idx + 1) * local_batch_size].item()
    return values_cpu[start:end]


def _setup_single_npu_dist_env() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")


def _run_sequence_embedding_bw(args) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")

    table_num = args.num_embedding_table
    dim = args.embedding_dim
    total_hbm_for_values = 1023**3
    table_options = [
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            dim=dim,
            max_capacity=num_emb,
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
        pooling_mode=DynamicEmbPoolingMode.NONE,
        use_index_dedup=args.use_index_dedup,
        optimizer=EmbOptimType.SGD,
        learning_rate=1.0,
        device=device,
    )

    num_iterations = args.num_iterations
    local_batch_size = args.batch_size

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
                indices_per_table = indices_per_feature(values_cpu, offsets_cpu, f, local_batch_size)
                count_tensor_to_dict(indices_per_table, dict_)
                lengths.append(indices_per_table.numel())
                assert indices_per_table.dim() == 1

            assert res.dim() == 2
            sample_res = res[:, 0].to("cpu")

            updated_res = var(indices, offsets)
            torch.npu.synchronize()
            sample_updated_res = updated_res[:, 0].to("cpu")

            diffs = sample_res - sample_updated_res
            offset = 0
            total_indices = diffs.numel()
            hit_counter = 0
            for length, f in zip(lengths, range(table_num)):
                diffs_cur_table = diffs[offset : offset + length]
                offset += length
                indices_per_table = indices_per_feature(values_cpu, offsets_cpu, f, local_batch_size)

                for j, index in enumerate(indices_per_table):
                    index = index.item()
                    res_diff = diffs_cur_table[j]
                    expected_diff = float(ref_dicts[f][index] % 100000)
                    if math.isclose(res_diff.item(), expected_diff):
                        hit_counter += 1
                    else:
                        print(
                            "diff:",
                            res_diff,
                            expected_diff,
                            f" j = {j} index = {index}",
                        )
            print(f"Iteration {i} hit rate = ", float(hit_counter) / total_indices)


def _init_and_run_sequence_embedding_bw(args) -> None:
    dist.init_process_group(backend="hccl")
    try:
        _run_sequence_embedding_bw(args)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dynamic sequence embedding backward")
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
        help="Comma separated multihot size per sparse feature.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--use_index_dedup",
        type=str2bool,
        default=True,
        help="Use index deduplication (default: True).",
    )
    return parser


def _parse_args(argv: Optional[List[str]] = None):
    args = _build_parser().parse_args([] if argv is None else argv)
    args.num_embeddings_per_feature = [int(v) for v in args.num_embeddings_per_feature.split(",")]
    args.multi_hot_sizes = [int(v) for v in args.multi_hot_sizes.split(",")]
    args.num_embedding_table = len(args.num_embeddings_per_feature)
    return args


@pytest.fixture(scope="module")
def backend_session():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    was_initialized = dist.is_initialized()
    if not was_initialized:
        _setup_single_npu_dist_env()
        dist.init_process_group(backend="hccl")
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    yield
    if not was_initialized and dist.is_initialized():
        dist.destroy_process_group()


def test_sequence_embedding_backward(request):
    request.getfixturevalue("backend_session")
    _run_sequence_embedding_bw(_parse_args())


@record
def main(argv: List[str]) -> None:
    args = _parse_args(argv)

    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    _init_and_run_sequence_embedding_bw(args)


if __name__ == "__main__":
    main(sys.argv[1:])
