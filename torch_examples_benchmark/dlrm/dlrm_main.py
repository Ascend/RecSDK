#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import argparse
import itertools
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, List, Optional
from tqdm import tqdm

import torch
import torchmetrics as metrics
from pyre_extensions import none_throws
from torch import distributed as dist
from torch.utils.data import DataLoader
from ec_dcnv2 import DlrmDcnEc as DLRM_DCN_EC

import torch_npu
from torchrec import EmbeddingBagCollection, EmbeddingCollection, EmbeddingConfig
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    get_default_sharders,
)
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology, ParameterConstraints
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.models.dlrm import DLRM, DLRM_DCN, DLRM_Projection, DLRMTrain
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.distributed.types import ShardingEnv


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(message)s')

# OSS import
try:
    # pyre-ignore[21]
    # @manual=//ai_codesign/benchmarks/dlrm/torchrec_dlrm/data:dlrm_dataloader
    from data.dlrm_dataloader import get_dataloader

    # pyre-ignore[21]
    # @manual=//ai_codesign/benchmarks/dlrm/torchrec_dlrm:lr_scheduler
    from lr_scheduler import LRPolicyScheduler

    # pyre-ignore[21]
    # @manual=//ai_codesign/benchmarks/dlrm/torchrec_dlrm:multi_hot
    from multi_hot import Multihot, RestartableMap
except ImportError:
    pass

# internal import
try:
    from .data.dlrm_dataloader import get_dataloader  # noqa F811
    from .lr_scheduler import LRPolicyScheduler  # noqa F811
    from .multi_hot import Multihot, RestartableMap  # noqa F811
except ImportError:
    pass

lib_fbgemm_npu_api_so_path = os.getenv('LIB_FBGEMM_NPU_API_SO_PATH')
torch.ops.load_library(lib_fbgemm_npu_api_so_path)

with_torchrec = os.getenv('WITH_TORCHREC', 'False').strip().lower() in {'true', '1', 'yes'}
with_hybrid_torchrec = os.getenv('WITH_HYBRID_TORCHREC', 'False').strip().lower() in {'true', '1', 'yes'}
with_embcache = os.getenv('WITH_EMBCACHE', 'False').strip().lower() in {'true', '1', 'yes'}

use_ec = os.getenv('USE_EC', 'False').strip().lower() in {'true', '1', 'yes'}

g_sparse_optim_num = int(os.getenv('SPARSE_OPTIM_NUM', '1'))


class InteractionType(Enum):
    ORIGINAL = "original"
    DCN = "dcn"
    PROJECTION = "projection"

    def __str__(self):
        return self.value


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--drop_last_training_batch",
        dest="drop_last_training_batch",
        action="store_true",
        help="Drop the last non-full training batch",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="batch size to use for validation and testing",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=int,
        default=None,
        help="number of train batches",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=int,
        default=None,
        help="number of validation batches",
    )
    parser.add_argument(
        "--limit_test_batches",
        type=int,
        default=None,
        help="number of test batches",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["criteo_1t", "criteo_kaggle"],
        default="criteo_1t",
        help="dataset for experiment, current support criteo_1tb, criteo_kaggle",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=100_000,
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to 100_000 if num_embeddings_per_feature is not supplied.",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default=None,
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,64",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="512,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--interaction_branch1_layer_sizes",
        type=str,
        default="2048,2048",
        help="Comma separated layer sizes for interaction branch1 (only on dlrm with projection).",
    )
    parser.add_argument(
        "--interaction_branch2_layer_sizes",
        type=str,
        default="2048,2048",
        help="Comma separated layer sizes for interaction branch2 (only on dlrm with projection).",
    )
    parser.add_argument(
        "--dcn_num_layers",
        type=int,
        default=3,
        help="Number of DCN layers in interaction layer (only on dlrm with DCN).",
    )
    parser.add_argument(
        "--dcn_low_rank_dim",
        type=int,
        default=512,
        help="Low rank dimension for DCN in interaction layer (only on dlrm with DCN).",
    )
    parser.add_argument(
        "--undersampling_rate",
        type=float,
        help="Desired proportion of zero-labeled samples to retain (i.e. undersampling zero-labeled rows)."
        " Ex. 0.3 indicates only 30pct of the rows with label 0 will be kept."
        " All rows with label 1 will be kept. Value should be between 0 and 1."
        " When not supplied, no undersampling occurs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        help="Use pinned memory when loading data.",
    )
    parser.add_argument(
        "--mmap_mode",
        dest="mmap_mode",
        action="store_true",
        help="--mmap_mode mmaps the dataset."
        " That is, the dataset is kept on disk but is accessed as if it were in memory."
        " --mmap_mode is intended mostly for faster debugging. Use --mmap_mode to bypass"
        " preloading the dataset when preloading takes too long or when there is "
        " insufficient memory available to load the full dataset.",
    )
    parser.add_argument(
        "--in_memory_binary_criteo_path",
        type=str,
        default=None,
        help="Directory path containing the Criteo dataset npy files.",
    )
    parser.add_argument(
        "--synthetic_multi_hot_criteo_path",
        type=str,
        default=None,
        help="Directory path containing the MLPerf v2 synthetic multi-hot dataset npz files.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=15.0,
        help="Learning rate.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Epsilon for Adagrad optimizer.",
    )
    parser.add_argument(
        "--shuffle_batches",
        dest="shuffle_batches",
        action="store_true",
        help="Shuffle each batch during training.",
    )
    parser.add_argument(
        "--shuffle_training_set",
        dest="shuffle_training_set",
        action="store_true",
        help="Shuffle the training set in memory. This will override mmap_mode",
    )
    parser.add_argument(
        "--validation_freq_within_epoch",
        type=int,
        default=None,
        help="Frequency at which validation will be run within an epoch.",
    )
    parser.set_defaults(
        pin_memory=None,
        mmap_mode=None,
        drop_last=None,
        shuffle_batches=None,
        shuffle_training_set=None,
    )
    parser.add_argument(
        "--adagrad",
        dest="adagrad",
        action="store_true",
        help="Flag to determine if adagrad optimizer should be used.",
    )
    parser.add_argument(
        "--interaction_type",
        type=InteractionType,
        choices=list(InteractionType),
        default=InteractionType.ORIGINAL,
        help="Determine the interaction type to be used (original, dcn, or projection)"
        " default is original DLRM with pairwise dot product",
    )
    parser.add_argument(
        "--collect_multi_hot_freqs_stats",
        dest="collect_multi_hot_freqs_stats",
        action="store_true",
        help="Flag to determine whether to collect stats on freq of embedding access.",
    )
    parser.add_argument(
        "--multi_hot_sizes",
        type=str,
        default=None,
        help="Comma separated multihot size per sparse feature. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--multi_hot_distribution_type",
        type=str,
        choices=["uniform", "pareto"],
        default=None,
        help="Multi-hot distribution options.",
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_decay_start", type=int, default=0)
    parser.add_argument("--lr_decay_steps", type=int, default=0)
    parser.add_argument(
        "--print_lr",
        action="store_true",
        help="Print learning rate every iteration.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TensorFloat-32 mode for matrix multiplications on A100 (or newer) GPUs.",
    )
    parser.add_argument(
        "--print_sharding_plan",
        action="store_true",
        help="Print the sharding plan used for each embedding table.",
    )
    return parser.parse_args(argv)


def _evaluate(
    limit_batches: Optional[int],
    pipeline: TrainPipelineSparseDist,
    eval_dataloader: DataLoader,
    stage: str,
) -> float:
    """
    Evaluates model. Computes and prints AUROC. Helper function for train_val_test.

    Args:
        limit_batches (Optional[int]): Limits the dataloader to the first `limit_batches` batches.
        pipeline (TrainPipelineSparseDist): data pipeline.
        eval_dataloader (DataLoader): Dataloader for either the validation set or test set.
        stage (str): "val" or "test".

    Returns:
        float: auroc result
    """
    pipeline._model.eval()
    device = pipeline._device

    iterator = itertools.islice(iter(eval_dataloader), limit_batches)

    auroc = metrics.AUROC("binary").to(device)

    is_rank_zero = dist.get_rank() == 0
    if is_rank_zero:
        pbar = tqdm(
            iter(int, 1),
            desc=f"Evaluating {stage} set",
            total=len(eval_dataloader),
            disable=False,
        )
    with torch.no_grad():
        while True:
            try:
                _loss, logits, labels = pipeline.progress(iterator)
                preds = torch.sigmoid(logits)
                auroc(preds, labels)
                if is_rank_zero:
                    pbar.update(1)
            except StopIteration:
                break

    auroc_result = auroc.compute().item()
    num_samples = torch.tensor(sum(map(len, auroc.target)), device=device)
    dist.reduce(num_samples, 0, op=dist.ReduceOp.SUM)

    if is_rank_zero:
        logging.info("AUROC over %s set: %s.", stage, auroc_result)
        logging.info(f"Number of %s samples: %s.", stage, num_samples)
    return auroc_result


def batched(it: Iterator, n: int):
    if n < 1:
        raise ValueError("Error: n must be gather than 1")
    for x in it:
        yield itertools.chain((x,), itertools.islice(it, n - 1))


@dataclass
class TrainParam:
    pipeline: TrainPipelineSparseDist
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    lr_scheduler: LRPolicyScheduler
    print_lr: bool
    validation_freq: Optional[int]
    limit_train_batches: Optional[int]
    limit_val_batches: Optional[int]


@dataclass
class TrainValTestParam:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    test_dataloader: DataLoader
    lr_scheduler: LRPolicyScheduler


def _train(epoch: int,
        params: TrainParam
) -> float:
    """
    Trains model for 1 epoch. Helper function for train_val_test.

    Args:
        pipeline (TrainPipelineSparseDist): data pipeline.
        train_dataloader (DataLoader): Training set's dataloader.
        val_dataloader (DataLoader): Validation set's dataloader.
        epoch (int): The number of complete passes through the training set so far.
        lr_scheduler (LRPolicyScheduler): Learning rate scheduler.
        print_lr (bool): Whether to print the learning rate every training step.
        validation_freq (Optional[int]): The number of training steps between validation runs within an epoch.
        limit_train_batches (Optional[int]): Limits the training set to the first `limit_train_batches` batches.
        limit_val_batches (Optional[int]): Limits the validation set to the first `limit_val_batches` batches.

    Returns:
        None.
    """
    pipeline = params.pipeline
    train_dataloader = params.train_dataloader
    val_dataloader = params.val_dataloader
    lr_scheduler = params.lr_scheduler
    print_lr = params.print_lr
    validation_freq = params.validation_freq
    limit_train_batches = params.limit_train_batches
    limit_val_batches = params.limit_val_batches

    pipeline._model.train()

    iterator = itertools.islice(iter(train_dataloader), limit_train_batches)

    is_rank_zero = dist.get_rank() == 0
    if is_rank_zero:
        pbar = tqdm(
            iter(int, 1),
            desc=f"Epoch {epoch}",
            total=len(train_dataloader),
            disable=False,
        )

    start_it = 0
    n = (
        validation_freq
        if validation_freq
        else limit_train_batches if limit_train_batches else len(train_dataloader)
    )
    # time start
    time0 = time.time()
    for batched_iterator in batched(iterator, n):
        for it in itertools.count(start_it):
            try:
                if is_rank_zero and print_lr:
                    for i, g in enumerate(pipeline._optimizer.param_groups):
                        logging.info("lr: %s %s %.6f", it, i, g['lr'])
                pipeline.progress(batched_iterator)
                lr_scheduler.step()
                if is_rank_zero:
                    pbar.update(1)
            except StopIteration:
                if is_rank_zero:
                    logging.info("Total number of iterations: %s.", it)
                start_it = it
                break

        if validation_freq and start_it % validation_freq == 0:
            _evaluate(limit_val_batches, pipeline, val_dataloader, "val")
            pipeline._model.train()

    all_its = (limit_train_batches if limit_train_batches else len(train_dataloader))
    time1 = time.time()
    avg_speed = all_its / (time1 - time0)
    return avg_speed


@dataclass
class TrainValTestResults:
    val_aurocs: List[float] = field(default_factory=list)
    test_auroc: Optional[float] = None
    train_speed: Optional[float] = None


def train_val_test(
    args: argparse.Namespace,
    params: TrainValTestParam
) -> TrainValTestResults:
    """
    Train/validation/test loop.

    Args:
        args (argparse.Namespace): parsed command line args.
        model (torch.nn.Module): model to train.
        optimizer (torch.optim.Optimizer): optimizer to use.
        device (torch.device): device to use.
        train_dataloader (DataLoader): Training set's dataloader.
        val_dataloader (DataLoader): Validation set's dataloader.
        test_dataloader (DataLoader): Test set's dataloader.
        lr_scheduler (LRPolicyScheduler): Learning rate scheduler.

    Returns:
        TrainValTestResults.
    """
    model = params.model
    optimizer = params.optimizer
    device = params.device
    test_dataloader = params.test_dataloader

    results = TrainValTestResults()

    if with_hybrid_torchrec:
        from hybrid_torchrec.distributed.hybrid_train_pipeline import HybridTrainPipelineSparseDist
        pipeline = HybridTrainPipelineSparseDist(
            model, optimizer, device, execute_all_batches=True
        )
    elif with_embcache:
        from torchrec_embcache_embedding.distributed.train_pipeline import EmbCacheTrainPipelineSparseDist
        cpu_device: torch.device = torch.device("cpu")
        pipeline = EmbCacheTrainPipelineSparseDist(
            model, optimizer, cpu_device=cpu_device, npu_device=device, execute_all_batches=True
        )
    else:
        # 原始torchrec
        pipeline = TrainPipelineSparseDist(
            model, optimizer, device, execute_all_batches=True
        )

    train_params = TrainParam(
        pipeline=pipeline,
        train_dataloader=params.train_dataloader,
        val_dataloader=params.val_dataloader,
        lr_scheduler=params.lr_scheduler,
        print_lr=args.print_lr,
        validation_freq=args.validation_freq_within_epoch,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )

    for epoch in range(args.epochs):
        train_speed = _train(
            epoch,
            train_params
        )

    test_auroc = _evaluate(args.limit_test_batches, pipeline, test_dataloader, "test")
    results.test_auroc = test_auroc
    results.train_speed = train_speed
    return results


def main(argv: List[str]) -> None:
    """
    Trains, validates, and tests a Deep Learning Recommendation Model (DLRM)
    (https://arxiv.org/abs/1906.00091). The DLRM model contains both data parallel
    components (e.g. multi-layer perceptrons & interaction arch) and model parallel
    components (e.g. embedding tables). The DLRM model is pipelined so that dataloading,
    data-parallel to model-parallel comms, and forward/backward are overlapped. Can be
    run with either a random dataloader or an in-memory Criteo 1 TB click logs dataset
    (https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/).

    Args:
        argv (List[str]): command line args.

    Returns:
        None.
    """
    args = parse_args(argv)
    for name, val in vars(args).items():
        try:
            vars(args)[name] = list(map(int, val.split(",")))
        except (ValueError, AttributeError):
            pass

    torch_npu.npu.matmul.allow_hf32 = args.allow_tf32

    if args.multi_hot_sizes is not None:
        if (
                args.num_embeddings_per_feature is not None
                and len(args.multi_hot_sizes) != len(args.num_embeddings_per_feature)
        ):
            raise ValueError(
                "--multi_hot_sizes must be a comma delimited list the same size as the number of embedding tables."
            )
        elif (
                args.num_embeddings_per_feature is None
                and len(args.multi_hot_sizes) != len(DEFAULT_CAT_NAMES)
        ):
            raise ValueError(
                "--multi_hot_sizes must be a comma delimited list the same size as the number of embedding tables."
            )
    if (
            args.in_memory_binary_criteo_path is None
            and args.synthetic_multi_hot_criteo_path is not None
    ):
        raise ValueError(
            "--in_memory_binary_criteo_path and --synthetic_multi_hot_criteo_path are mutually exclusive CLI arguments."
        )
    if (
            args.multi_hot_sizes is not None
            and args.synthetic_multi_hot_criteo_path is not None
    ):
        raise ValueError(
            "--multi_hot_sizes is used to convert 1-hot to multi-hot. "
            "It's inapplicable with --synthetic_multi_hot_criteo_path."
        )
    if (
            args.multi_hot_distribution_type is not None
            and args.synthetic_multi_hot_criteo_path is not None
    ):
        raise ValueError(
            "--multi_hot_distribution_type is used to convert 1-hot to multi-hot. "
            "It's inapplicable with --synthetic_multi_hot_criteo_path."
        )

    rank_str = os.environ["LOCAL_RANK"]
    if rank_str is None:
        raise KeyError("Environment variable LOCAL_RANK is not set.")
    try:
        rank = int(rank_str)
    except ValueError as e:
        raise Exception("Environment variable LOCAL_RANK is not a valid integer.") from e

    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    elif torch_npu.npu.is_available():
        device: torch.device = torch.device(f"npu:{rank}")
        backend = "hccl"
        torch_npu.npu.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"

    if rank == 0:
        logging.info(
            "PARAMS: (%s, %s, %s, %s, %s): ",
            args.learning_rate, args.batch_size, args.lr_warmup_steps, args.lr_decay_start, args.lr_decay_steps)
    dist.init_process_group(backend=backend)

    if args.num_embeddings_per_feature is not None:
        args.num_embeddings = None

    # Sets default limits for random dataloader iterations when left unspecified.
    if (
        args.in_memory_binary_criteo_path
        is args.synthetic_multi_hot_criteo_path
        is None
    ):
        for split in ["train", "val", "test"]:
            attr = f"limit_{split}_batches"
            if getattr(args, attr) is None:
                setattr(args, attr, 10)

    train_dataloader = get_dataloader(args, backend, "train")
    val_dataloader = get_dataloader(args, backend, "val")
    test_dataloader = get_dataloader(args, backend, "test")

    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=args.embedding_dim,
            num_embeddings=(
                none_throws(args.num_embeddings_per_feature)[feature_idx]
                if args.num_embeddings is None
                else args.num_embeddings
            ),
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    ]

    ec_configs = [
        EmbeddingConfig(
            name=f"t_{feature_name}",
            embedding_dim=args.embedding_dim,
            num_embeddings=(
                none_throws(args.num_embeddings_per_feature)[feature_idx]
                if args.num_embeddings is None
                else args.num_embeddings
            ),
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    ]

    sharded_module_kwargs = {}
    if args.over_arch_layer_sizes is not None:
        sharded_module_kwargs["over_arch_layer_sizes"] = args.over_arch_layer_sizes

    if args.interaction_type == InteractionType.ORIGINAL:
        from hybrid_torchrec import HashEmbeddingBagCollection
        dlrm_model = DLRM(
            embedding_bag_collection=HashEmbeddingBagCollection(
                tables=eb_configs, device=torch.device("meta")
            ),
            dense_in_features=len(DEFAULT_INT_NAMES),
            dense_arch_layer_sizes=args.dense_arch_layer_sizes,
            over_arch_layer_sizes=args.over_arch_layer_sizes,
            dense_device=device,
        )
    elif args.interaction_type == InteractionType.DCN:
        if with_hybrid_torchrec:
            if use_ec:
                from hybrid_torchrec import HashEmbeddingCollection
                dlrm_model = DLRM_DCN_EC(
                    embedding_collection=HashEmbeddingCollection(
                        tables=ec_configs, device=torch.device("meta")
                    ),
                    multi_hot_sizes=args.multi_hot_sizes,
                    dense_in_features=len(DEFAULT_INT_NAMES),
                    dense_arch_layer_sizes=args.dense_arch_layer_sizes,
                    over_arch_layer_sizes=args.over_arch_layer_sizes,
                    dcn_num_layers=args.dcn_num_layers,
                    dcn_low_rank_dim=args.dcn_low_rank_dim,
                    dense_device=device,
                )
            else:
                from hybrid_torchrec import HashEmbeddingBagCollection
                dlrm_model = DLRM_DCN(
                    embedding_bag_collection=HashEmbeddingBagCollection(
                        tables=eb_configs, device=torch.device("meta")
                    ),
                    dense_in_features=len(DEFAULT_INT_NAMES),
                    dense_arch_layer_sizes=args.dense_arch_layer_sizes,
                    over_arch_layer_sizes=args.over_arch_layer_sizes,
                    dcn_num_layers=args.dcn_num_layers,
                    dcn_low_rank_dim=args.dcn_low_rank_dim,
                    dense_device=device,
                )
        elif with_embcache:
            if use_ec:
                from torchrec_embcache_embedding.distributed.embedding import EmbCacheEmbeddingCollection
                dlrm_model = DLRM_DCN_EC(
                    embedding_collection=EmbCacheEmbeddingCollection(
                        tables=ec_configs,
                        batch_size=args.batch_size,
                        multi_hot_sizes=args.multi_hot_sizes,
                        need_indices=False,
                        world_size=dist.get_world_size(),
                        device=torch.device("meta"),
                        sparse_optim_num=g_sparse_optim_num,
                    ),
                    multi_hot_sizes=args.multi_hot_sizes,
                    dense_in_features=len(DEFAULT_INT_NAMES),
                    dense_arch_layer_sizes=args.dense_arch_layer_sizes,
                    over_arch_layer_sizes=args.over_arch_layer_sizes,
                    dcn_num_layers=args.dcn_num_layers,
                    dcn_low_rank_dim=args.dcn_low_rank_dim,
                    dense_device=device,
                )
            else:
                from torchrec_embcache_embedding.distributed.embedding_bag import EmbCacheEmbeddingBagCollection
                dlrm_model = DLRM_DCN(
                    embedding_bag_collection=EmbCacheEmbeddingBagCollection(
                        tables=eb_configs,
                        batch_size=args.batch_size,
                        multi_hot_sizes=args.multi_hot_sizes,
                        world_size=dist.get_world_size(),
                        device=torch.device("meta"),
                        sparse_optim_num=g_sparse_optim_num,
                    ),
                    dense_in_features=len(DEFAULT_INT_NAMES),
                    dense_arch_layer_sizes=args.dense_arch_layer_sizes,
                    over_arch_layer_sizes=args.over_arch_layer_sizes,
                    dcn_num_layers=args.dcn_num_layers,
                    dcn_low_rank_dim=args.dcn_low_rank_dim,
                    dense_device=device,
                )
        else:
            if use_ec:
                dlrm_model = DLRM_DCN_EC(
                    embedding_collection=EmbeddingCollection(
                        tables=ec_configs, device=torch.device("meta")
                    ),
                    multi_hot_sizes=args.multi_hot_sizes,
                    dense_in_features=len(DEFAULT_INT_NAMES),
                    dense_arch_layer_sizes=args.dense_arch_layer_sizes,
                    over_arch_layer_sizes=args.over_arch_layer_sizes,
                    dcn_num_layers=args.dcn_num_layers,
                    dcn_low_rank_dim=args.dcn_low_rank_dim,
                    dense_device=device,
                )
            else:
                dlrm_model = DLRM_DCN(
                    embedding_bag_collection=EmbeddingBagCollection(
                        tables=eb_configs, device=torch.device("meta")
                    ),
                    dense_in_features=len(DEFAULT_INT_NAMES),
                    dense_arch_layer_sizes=args.dense_arch_layer_sizes,
                    over_arch_layer_sizes=args.over_arch_layer_sizes,
                    dcn_num_layers=args.dcn_num_layers,
                    dcn_low_rank_dim=args.dcn_low_rank_dim,
                    dense_device=device,
                )
    elif args.interaction_type == InteractionType.PROJECTION:
        from hybrid_torchrec import HashEmbeddingBagCollection
        dlrm_model = DLRM_Projection(
            embedding_bag_collection=HashEmbeddingBagCollection(
                tables=eb_configs, device=torch.device("meta")
            ),
            dense_in_features=len(DEFAULT_INT_NAMES),
            dense_arch_layer_sizes=args.dense_arch_layer_sizes,
            over_arch_layer_sizes=args.over_arch_layer_sizes,
            interaction_branch1_layer_sizes=args.interaction_branch1_layer_sizes,
            interaction_branch2_layer_sizes=args.interaction_branch2_layer_sizes,
            dense_device=device,
        )
    else:
        raise ValueError(
            "Unknown interaction option set. Should be original, dcn, or projection."
        )

    host_gp = dist.new_group(backend='gloo')
    host_env = ShardingEnv(world_size=dist.get_world_size(), rank=rank, pg=host_gp)

    train_model = DLRMTrain(dlrm_model)
    embedding_optimizer = torch.optim.Adagrad if args.adagrad else torch.optim.SGD

    optimizer_kwargs = {"lr": args.learning_rate}
    if args.adagrad:
        optimizer_kwargs["eps"] = args.eps
    apply_optimizer_in_backward(
        embedding_optimizer,
        train_model.model.sparse_arch.parameters(),
        optimizer_kwargs,
    )

    constraints = {
        f"t_{feature_name}": ParameterConstraints(
            sharding_types=["row_wise"],
        )
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    }
    if with_torchrec:
        # 不指定sharding_type
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                local_world_size=get_local_size(),
                world_size=dist.get_world_size(),
                compute_device=device.type,
            ),
            batch_size=args.batch_size,
            # If experience OOM, increase the percentage. see
            storage_reservation=HeuristicalStorageReservation(percentage=0.05),
        )
    else:
        # row_wise
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                local_world_size=get_local_size(),
                world_size=dist.get_world_size(),
                compute_device=device.type,
            ),
            batch_size=args.batch_size,
            # If experience OOM, increase the percentage. see
            storage_reservation=HeuristicalStorageReservation(percentage=0.05),
            constraints=constraints,
        )

    sharders = []
    if with_embcache:
        cpu_device = torch.device("cpu")
        cpu_pg = dist.new_group(backend="gloo")
        cpu_env = ShardingEnv.from_process_group(cpu_pg)
        if use_ec:
            from torchrec_embcache_embedding.distributed.sharding.embedding_sharder import (
                EmbCacheEmbeddingCollectionSharder
            )
            embcache_sharder = EmbCacheEmbeddingCollectionSharder(
                cpu_device=cpu_device,
                cpu_env=cpu_env,
                npu_device=device,
                npu_env=ShardingEnv.from_process_group(dist.GroupMember.WORLD),
            )
        else:
            from torchrec_embcache_embedding.distributed.sharding.embedding_sharder import (
                EmbCacheEmbeddingBagCollectionSharder
            )
            embcache_sharder = EmbCacheEmbeddingBagCollectionSharder(
                cpu_device=cpu_device,
                cpu_env=cpu_env,
                npu_device=device,
                npu_env=ShardingEnv.from_process_group(dist.GroupMember.WORLD),
            )
        sharders = [embcache_sharder]
    elif with_hybrid_torchrec:
        from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders
        sharders = get_default_hybrid_sharders(host_env)
    else:
        sharders = get_default_sharders()

    plan = planner.collective_plan(
        train_model, sharders, dist.GroupMember.WORLD
    )
    if rank == 0:
        logging.info("plan:%s", plan)

    model = DistributedModelParallel(
        module=train_model,
        sharders=sharders,
        device=device,
        plan=plan,
    )
    if rank == 0 and args.print_sharding_plan:
        for collectionkey, plans in model._plan.plan.items():
            logging.info("collectionkey: %s", collectionkey)
            for table_name, plan in plans.items():
                logging.info("%s\n%s\n", table_name, plan)

    def optimizer_with_params():
        if args.adagrad:
            return lambda params: torch.optim.Adagrad(
                params, lr=args.learning_rate, eps=args.eps
            )
        return lambda params: torch.optim.SGD(params, lr=args.learning_rate)

    dense_optimizer = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(model.named_parameters())),
        optimizer_with_params(),
    )
    optimizer = CombinedOptimizer([model.fused_optimizer, dense_optimizer])
    lr_scheduler = LRPolicyScheduler(
        optimizer, args.lr_warmup_steps, args.lr_decay_start, args.lr_decay_steps
    )

    if args.multi_hot_sizes is not None:
        multihot = Multihot(
            args.multi_hot_sizes,
            args.num_embeddings_per_feature,
            args.batch_size,
            collect_freqs_stats=args.collect_multi_hot_freqs_stats,
            dist_type=args.multi_hot_distribution_type,
        )
        multihot.pause_stats_collection_during_val_and_test(model)
        train_dataloader = RestartableMap(
            multihot.convert_to_multi_hot, train_dataloader
        )
        val_dataloader = RestartableMap(multihot.convert_to_multi_hot, val_dataloader)
        test_dataloader = RestartableMap(multihot.convert_to_multi_hot, test_dataloader)

    train_val_test_params = TrainValTestParam(
        model=model,
        optimizer=optimizer,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        lr_scheduler=lr_scheduler,
    )

    results = train_val_test(
        args,
        train_val_test_params,
    )
    if args.collect_multi_hot_freqs_stats:
        multihot.save_freqs_stats()

    if rank == 0:
        logging.info("Train avg speed: %.4f its/s", results.train_speed)
        logging.info("Test AUROC: %.6f", results.test_auroc)
    return results


def invoke_main() -> None:
    main(sys.argv[1:])


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
