# pylint: disable=duplicate-code
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
from typing import List, Optional

import pytest
import torch
import torch.distributed as dist
import torchrec
from debug import Debugger
from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbCheckMode,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbTableOptions,
)
from dynamic_emb.distributed.planner.enumerators import DynamicEmbeddingEnumerator
from dynamic_emb.distributed.planner.planners import DynamicEmbeddingShardingPlanner
from dynamic_emb.distributed.planner.types import DynamicEmbParameterConstraints
from dynamic_emb.distributed.embedding import DynamicEmbeddingCollectionSharder
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from torch.distributed.elastic.multiprocessing.errors import record
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import ParameterConstraints, Topology
from torchrec.distributed.types import BoundsCheckMode, ShardingType


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


def get_planner(args, device, eb_configs):
    dict_const = {}
    for i in range(args.num_embedding_table):
        if args.data_parallel_embeddings is not None and i in args.data_parallel_embeddings:
            const = ParameterConstraints(
                sharding_types=[ShardingType.DATA_PARALLEL.value],
                pooling_factors=[args.multi_hot_sizes[i]],
                num_poolings=[1],
                enforce_hbm=True,
                bounds_check_mode=BoundsCheckMode.NONE,
            )
        else:
            # RecSDK DynamicEmbParameterConstraints only allows use_dynamicemb=True.
            const = DynamicEmbParameterConstraints(  # pylint: disable=unexpected-keyword-arg
                sharding_types=[ShardingType.ROW_WISE.value],
                compute_kernels=["fused"],
                use_dynamicemb=True,
                dynamicemb_options=DynamicEmbTableOptions(
                    global_hbm_for_values=1024**3,
                    local_hbm_for_values=1024**3,
                    initializer_args=DynamicEmbInitializerArgs(mode=DynamicEmbInitializerMode.DEBUG),
                    safe_check_mode=DynamicEmbCheckMode.WARNING,
                ),
            )

        dict_const[table_idx_to_name(i)] = const

    topology = Topology(
        world_size=dist.get_world_size(),
        compute_device=device.type,
    )

    enumerator = DynamicEmbeddingEnumerator(
        topology=topology,
        constraints=dict_const,
    )

    return DynamicEmbeddingShardingPlanner(
        eb_configs=eb_configs,
        topology=topology,
        constraints=dict_const,
        batch_size=args.batch_size,
        enumerator=enumerator,
        debug=True,
    )


def generate_sparse_feature(
    feature_num,
    num_embeddings_list,
    multi_hot_sizes,
    local_batch_size=50,
    device=torch.device("npu:0"),
):
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


def _setup_single_npu_dist_env() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")


def _run_sequence_embedding_fw(args) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")

    eb_configs = [
        torchrec.EmbeddingConfig(
            name=table_idx_to_name(feature_idx),
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_embeddings_per_feature[feature_idx],
            feature_names=[feature_idx_to_name(feature_idx)],
        )
        for feature_idx in range(args.num_embedding_table)
    ]
    ebc = torchrec.EmbeddingCollection(
        device=torch.device("meta"),
        tables=eb_configs,
    )

    optimizer_kwargs = {
        "learning_rate": args.learning_rate,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "weight_decay": args.weight_decay,
        "eps": args.eps,
    }
    if args.optimizer_type == "sgd":
        optimizer_kwargs["optimizer"] = EmbOptimType.EXACT_SGD
    elif args.optimizer_type == "exact_sgd":
        optimizer_kwargs["optimizer"] = EmbOptimType.EXACT_SGD
    elif args.optimizer_type == "adam":
        optimizer_kwargs["optimizer"] = EmbOptimType.ADAM
    elif args.optimizer_type == "exact_adagrad":
        optimizer_kwargs["optimizer"] = EmbOptimType.EXACT_ADAGRAD
    elif args.optimizer_type == "exact_row_wise_adagrad":
        optimizer_kwargs["optimizer"] = EmbOptimType.EXACT_ROWWISE_ADAGRAD
    else:
        raise ValueError("unknown optimizer type")

    planner = get_planner(args, device, eb_configs)

    fused_params = {"output_dtype": SparseType.FP32}
    fused_params.update(optimizer_kwargs)
    sharder = DynamicEmbeddingCollectionSharder(
        fused_params=fused_params,
        use_index_dedup=args.use_index_dedup,
    )
    plan = planner.collective_plan(ebc, [sharder], dist.GroupMember.WORLD)

    model = DistributedModelParallel(
        module=ebc,
        device=device,
        sharders=[sharder],
        plan=plan,
    )

    if local_rank == 0 and args.print_sharding_plan:
        for collectionkey, plans in model._plan.plan.items():
            print(collectionkey)
            for table_name, plan_item in plans.items():
                print(table_name, "\n", plan_item, "\n")

    debugger = Debugger()

    for i in range(args.num_iterations):
        sparse_feature = generate_sparse_feature(
            feature_num=args.num_embedding_table,
            num_embeddings_list=args.num_embeddings_per_feature,
            multi_hot_sizes=args.multi_hot_sizes,
            local_batch_size=args.batch_size // world_size,
            device=device,
        )
        ret = model(sparse_feature)

        feature_names = []
        jagged_tensors = []
        for k, v in ret.items():
            feature_names.append(k)
            jagged_tensors.append(v)

        feature_num = len(ret.keys())
        dims = [args.embedding_dim for _ in range(feature_num)]

        dyn_emb_features = [feature_idx_to_name(i) for i in range(args.dynamicemb_num)]

        debugger.feature_before_all2all(sparse_feature)
        debugger.sequence_embds_after_all2all(jagged_tensors, feature_names, dyn_emb_features, dims[0])
        print(f"DynamicEmb iteration {i + 1} Passed")


def _init_and_run_sequence_embedding_fw(args) -> None:
    dist.init_process_group(backend="hccl")
    try:
        _run_sequence_embedding_fw(args)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TorchRec EmbeddingCollection forward with DynamicEmb")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="number of iterations",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default="65536,32768,4096,8192",
        help="Comma separated max index per sparse feature.",
    )
    parser.add_argument(
        "--multi_hot_sizes",
        type=str,
        default="16,8,20,1",
        help="Comma separated multihot size per sparse feature.",
    )
    parser.add_argument(
        "--print_sharding_plan",
        action="store_true",
        help="Print the sharding plan used for each embedding table.",
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
        default="adam",
        choices=["sgd", "exact_sgd", "adam", "exact_adagrad", "exact_row_wise_adagrad"],
        help="optimizer type.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate.",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="beta1.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="beta2.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.001,
        help="eps.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="weight_decay.",
    )
    parser.add_argument(
        "--data_parallel_embeddings",
        type=str,
        default=None,
        help="Comma separated data parallel embedding table ids.",
    )
    parser.add_argument(
        "--dynamicemb_num",
        type=int,
        default=4,
        help="Number of dynamic embedding features to verify in debugger "
        "(all tables use DynamicEmb in RecSDK planner).",
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
    args.data_parallel_embeddings = (
        None if args.data_parallel_embeddings is None else [int(v) for v in args.data_parallel_embeddings.split(",")]
    )
    args.num_embedding_table = len(args.num_embeddings_per_feature)
    args.dynamicemb_num = min(args.dynamicemb_num, args.num_embedding_table)

    if args.embedding_dim % 4 != 0:
        print(
            f"INFO: args.embedding_dim = {args.embedding_dim} is not aligned with 4, "
            "all tables use DynamicEmb in RecSDK."
        )
        args.dynamicemb_num = args.num_embedding_table
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


def test_sequence_embedding_forward(request):
    request.getfixturevalue("backend_session")
    _run_sequence_embedding_fw(_parse_args())


@record
def main(argv: List[str]) -> None:
    args = _parse_args(argv)

    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    _init_and_run_sequence_embedding_fw(args)


if __name__ == "__main__":
    main(sys.argv[1:])
