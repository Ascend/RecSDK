#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import (
    Any,
    Deque,
    cast,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Callable,
)
from collections import defaultdict, deque
import logging
import time
import os

import torch_npu
import torch
from torch.autograd.profiler import record_function

from hybrid_torchrec.distributed.sharding.sequence_sharding import (
    HybridSequenceShardingContext,
)

import embcache_pybind
from torchrec_embcache.distributed.sharding.rw_sharding import (
    EmbCacheRwSparseFeaturesDistAwaitable,
)

from torchrec.distributed.train_pipeline import In, Out, _wait_for_batch
from torchrec.distributed.train_pipeline.utils import (
    In,
    Out,
    PrefetchPipelinedForward,
    _build_args_kwargs,
    _wait_for_batch,
    TrainPipelineContext,
    PipelinedForward,
    _rewrite_model,
    _override_input_dist_forwards,
)

from torchrec.distributed.types import Awaitable, ShardedModule
from torchrec.streamable import Pipelineable
from torchrec.distributed.embedding_sharding import (
    FusedKJTListSplitsAwaitable,
    KJTListSplitsAwaitable,
    KJTSplitsAllToAllMeta,
)
from torchrec.distributed.embedding_types import KJTList
from torchrec.distributed.train_pipeline.train_pipelines import TrainPipelineSparseDist

MIN_EVICT_STEP_INTERVAL = 10
logger: logging.Logger = logging.getLogger(__name__)


class EmbCacheAwaitableAdapterThreadPoolExecutorSingleton:
    _instance: "EmbCacheAwaitableAdapterThreadPoolExecutorSingleton" = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(
                EmbCacheAwaitableAdapterThreadPoolExecutorSingleton, cls
            ).__new__(cls, *args, **kwargs)
            cls.executor = ThreadPoolExecutor(1)
        return cls._instance


class AwaitableAdapter(Awaitable):
    def __init__(self, awaitable) -> None:
        super().__init__()

        def get_awaitable_result(awaitable: Awaitable):
            return awaitable.wait()

        self.future = (
            EmbCacheAwaitableAdapterThreadPoolExecutorSingleton().executor.submit(
                get_awaitable_result, awaitable
            )
        )

    def _wait_impl(self) -> Any:
        return self.future.result()


@dataclass
class EmbCacheTrainPipelineContext(TrainPipelineContext):
    sparse_features_after_dist: Dict[str, KJTList] = field(default_factory=dict)
    sparse_features_after_post_dist: Dict[str, KJTList] = field(default_factory=dict)
    sparse_features_after_restore_future: Dict[str, KJTList] = field(
        default_factory=dict
    )
    swap_info_future: Dict[str, embcache_pybind.AsyncSwapInfo] = field(
        default_factory=dict
    )
    swap_info: Dict[str, embcache_pybind.SwapInfo] = field(default_factory=dict)
    swapout_embs: Dict[str, torch.Tensor] = field(default_factory=dict)
    swapout_optims: Dict[str, torch.Tensor] = field(default_factory=dict)
    swapin_tensor_future: Dict[str, embcache_pybind.AsyncSwapinTensor] = field(
        default_factory=dict
    )
    swapin_embs: Dict[str, torch.Tensor] = field(default_factory=dict)
    swapin_optims: Dict[str, torch.Tensor] = field(default_factory=dict)
    update_future: Dict[str, embcache_pybind.AsyncUpdate] = field(default_factory=dict)
    event_can_swapout: Optional[torch_npu.npu.Event] = None
    event_gather_swapouted: Optional[torch_npu.npu.Event] = None
    event_swapin_scattered: Optional[torch_npu.npu.Event] = None
    post_input_dist_awaitable: Dict[str, Awaitable] = field(default_factory=dict)
    memcpy_stream: Optional[torch.Stream] = None


class EmbCachePipelinedForward(PipelinedForward):
    # pyre-ignore [2, 24]
    def __call__(self, *input_feature, **kwargs) -> Awaitable:
        self._context.sparse_features_after_restore_future.pop(self._name).get()
        data = self._context.sparse_features_after_post_dist.pop(self._name)
        # 由于把global_unique的结果unique_ids作为输入给到get_swap_info, 因此get_swap_info的结果batch_offs即为unique_indices
        data[0].unique_indices = self._context.swap_info[self._name].batch_offs

        ctx = self._context.module_contexts.pop(self._name)
        cur_stream = torch.get_device_module(self._device).current_stream()
        with torch_npu.npu.stream(self._context.memcpy_stream):
            for index, data_item in enumerate(data):
                data[index] = data_item.to(self._device, non_blocking=True)
                data[index].record_stream(cur_stream)

            for sharding_ctx in ctx.sharding_contexts:
                if not isinstance(sharding_ctx, HybridSequenceShardingContext):
                    continue
                sharding_ctx.sparse_features_recat = None  # 不考虑vbe场景
                # 因查表前卸载到cpu,查表时要to device.
                if sharding_ctx.unbucketize_permute_tensor is not None:
                    sharding_ctx.unbucketize_permute_tensor = (
                        sharding_ctx.unbucketize_permute_tensor.to(
                            self._device, non_blocking=True
                        )
                    )
                    sharding_ctx.unbucketize_permute_tensor.record_stream(cur_stream)

            copy_done_event = torch_npu.npu.Event()
            copy_done_event.record(self._context.memcpy_stream)

        cur_stream.wait_event(copy_done_event)
        return self._module.compute_and_output_dist(ctx, data)


def _start_data_dist(
    pipelined_modules: List[ShardedModule],
    batch: Pipelineable,
    context: TrainPipelineContext,
) -> None:
    if context.version == 0:
        context.input_dist_splits_requests.clear()
        context.module_contexts_next_batch.clear()
        context.fused_splits_awaitables.clear()

    for module in pipelined_modules:
        forward = module.forward
        if not (
            isinstance(forward, PipelinedForward)
            or isinstance(forward, PrefetchPipelinedForward)
            or isinstance(forward, EmbCachePipelinedForward)
        ):
            raise RuntimeError("forward should be in [PipelinedForward," \
            " PrefetchPipelinedForward, EmbCachePipelinedForward]")

        # Retrieve argument for the input_dist of EBC
        # is_getitem True means this argument could be retrieved by a list
        # False means this argument is getting while getattr
        # and this info was done in the _rewrite_model by tracing the
        # entire model to get the arg_info_list
        args, kwargs = _build_args_kwargs(batch, forward.args)

        # Start input distribution.
        module_ctx = module.create_context()
        if context.version == 0:
            context.module_contexts_next_batch[forward.name] = module_ctx
        else:
            context.module_contexts[forward.name] = module_ctx
        context.input_dist_splits_requests[forward.name] = module.input_dist(
            module_ctx, *args, **kwargs
        )


def _fuse_input_dist_splits(context: TrainPipelineContext) -> None:
    with record_function("## _fuse_input_dist_splits ##"):
        names_per_pg = defaultdict(list)
        for name, request in context.input_dist_splits_requests.items():
            pg = None
            if isinstance(request, KJTListSplitsAwaitable):
                for awaitable in request.awaitables:
                    if isinstance(awaitable, KJTSplitsAllToAllMeta) or isinstance(
                        awaitable, EmbCacheRwSparseFeaturesDistAwaitable
                    ):
                        pg = awaitable.pg
                        break
            names_per_pg[pg].append(name)

        for name, request in context.input_dist_splits_requests.items():
            for ind, awaitable in enumerate(
                context.input_dist_splits_requests[name].awaitables
            ):
                if isinstance(awaitable, EmbCacheRwSparseFeaturesDistAwaitable):
                    context.input_dist_splits_requests[name].awaitables[ind] = (
                        context.input_dist_splits_requests[name].awaitables[ind].wait()
                    )

        for pg, names in names_per_pg.items():
            context.fused_splits_awaitables.append(
                (
                    names,
                    FusedKJTListSplitsAwaitable(
                        # pyre-ignore[6]
                        requests=[
                            context.input_dist_splits_requests[name] for name in names
                        ],
                        contexts=[
                            (
                                context.module_contexts_next_batch[name]
                                if context.version == 0
                                else context.module_contexts[name]
                            )
                            for name in names
                        ],
                        pg=pg,
                    ),
                )
            )


class EmbCacheTrainPipelineSparseDist(TrainPipelineSparseDist[In, Out]):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cpu_device: torch.device,
        npu_device: torch.device,
        return_loss: bool = False,
        evict_step_interval: Optional[int] = None,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        context_type: Type[EmbCacheTrainPipelineContext] = EmbCacheTrainPipelineContext,
        # keep for backward compatibility
        pipeline_postproc: bool = False,
        custom_model_fwd: Optional[
            Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
        ] = None,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=npu_device,
            execute_all_batches=execute_all_batches,
            apply_jit=apply_jit,
            context_type=context_type,
            pipeline_postproc=pipeline_postproc,
            custom_model_fwd=custom_model_fwd,
        )
        self.contexts: Deque[EmbCacheTrainPipelineContext] = deque()
        self._cpu_device = cpu_device
        self._npu_device = npu_device
        self._return_loss = return_loss
        self._global_steps = 0
        if (
            evict_step_interval is not None
            and evict_step_interval < MIN_EVICT_STEP_INTERVAL
        ):
            raise ValueError(
                f"Param error, evict_step_interval must greater or equal than {MIN_EVICT_STEP_INTERVAL},"
                f" but got {evict_step_interval}."
            )
        self._evict_step_interval = evict_step_interval or 0
        self._default_stream = torch.get_device_module(
            self._npu_device
        ).current_stream()
        self.local_unique_parallel_batch_num = int(
            os.environ.get("LOCAL_UNIQUE_PARALLEL_BATCH_NUM", 2)
        )

    def _init_pipelined_modules(
        self,
        batch: In,
        context: TrainPipelineContext,
        pipelined_forward: Type[PipelinedForward] = PipelinedForward,
    ) -> None:
        """
        Retrieves the pipelined modules after overriding their forwards, initializes the
        modules' input dists, and overrides the input dist forwards to support fusing
        the splits collective in the input dist.
        """
        if self._pipelined_modules:
            self._set_module_context(context)
            self.start_sparse_data_dist(batch, context)
            _fuse_input_dist_splits(context)
            return

        (
            self._pipelined_modules,
            self._model,
            self._original_forwards,
            self._pipelined_postprocs,
            _,
        ) = _rewrite_model(
            model=self._model,
            context=context,
            dist_stream=self._data_dist_stream,
            default_stream=torch.get_device_module(self._device).current_stream(),
            batch=batch,
            apply_jit=self._apply_jit,
            pipelined_forward=pipelined_forward,
            pipeline_postproc=self._pipeline_postproc,
        )
        # initializes input dist, so we can override input dist forwards
        self.start_sparse_data_dist(batch, context)
        _fuse_input_dist_splits(context)
        self._original_kjt_dist_forwards = _override_input_dist_forwards(
            self._pipelined_modules
        )

    def start_sparse_data_dist(
        self, batch: Optional[In], context: TrainPipelineContext
    ) -> None:
        """
        Waits for batch to finish getting copied to GPU, then starts the input dist.
        """
        if batch is None:
            return
        with record_function(f"## start_sparse_data_dist {context.index} ##"):
            with self._stream_context(self._data_dist_stream):
                _wait_for_batch(batch, self._memcpy_stream)

                original_contexts = [p.get_context() for p in self._pipelined_postprocs]

                # Temporarily set context for next iter to populate cache
                for postproc_mod in self._pipelined_postprocs:
                    postproc_mod.set_context(context)

                _start_data_dist(self._pipelined_modules, batch, context)

                # Restore context for model fwd
                for module, context in zip(
                    self._pipelined_postprocs, original_contexts
                ):
                    module.set_context(context)

    def wait_sparse_data_dist(self, context: TrainPipelineContext) -> None:
        """
        Waits on the input dist splits requests to get the input dist tensors requests,
        and populates the context with them.
        """
        with record_function(f"## wait_sparse_data_dist {context.index} ##"):
            with self._stream_context(self._data_dist_stream):
                for names, awaitable in context.fused_splits_awaitables:
                    for name, request in zip(names, awaitable.wait()):
                        context.input_dist_tensors_requests[name] = AwaitableAdapter(
                            request
                        )
        context.input_dist_splits_requests.clear()
        context.fused_splits_awaitables.clear()

    def _create_context(self) -> EmbCacheTrainPipelineContext:
        context = self._context_type(index=self._next_index, version=1)
        context.event_gather_swapouted = torch.get_device_module(
            self._npu_device
        ).Event()
        context.event_swapin_scattered = torch.get_device_module(
            self._npu_device
        ).Event()
        context.event_can_swapout = torch.get_device_module(self._npu_device).Event()
        context.memcpy_stream = self._memcpy_stream
        self._next_index += 1
        return context

    def _compute_swap_info_async(self, context: EmbCacheTrainPipelineContext) -> None:
        with record_function("## _compute_swap_info_async ##"):
            for module in self._pipelined_modules:
                module_name = module.forward.name
                sparse_features = context.sparse_features_after_post_dist[module_name]
                context.swap_info_future[module_name] = module.compute_swap_info_async(
                    sparse_features
                )

    def do_post_input_dist(self, context: EmbCacheTrainPipelineContext):
        with record_function("## _post_input_dist ##"):
            for name, module in zip(
                context.input_dist_tensors_requests.keys(),
                self._pipelined_modules,
            ):
                awaitable = context.input_dist_tensors_requests[name]

                with record_function("## wait input_dist_tensors_requests ##"):
                    kjt_list = awaitable.wait()

                if hasattr(module, "post_input_dist"):
                    post_waitable = module.post_input_dist(
                        context.module_contexts[name], kjt_list
                    )
                    context.post_input_dist_awaitable[name] = post_waitable
                else:
                    raise RuntimeError(
                        "EmbCacheTrainPipelineSparseDist can't be used for module with no post_input method"
                    )

    def do_restore_async(self, context: EmbCacheTrainPipelineContext):
        with record_function("## restore ##"):
            for module in self._pipelined_modules:
                module_name = module.forward.name
                data = context.sparse_features_after_post_dist[module_name]
                context.sparse_features_after_restore_future[module_name] = (
                    embcache_pybind.restore_async(
                        context.swap_info[module_name].batch_offs,
                        data[0].unique_inverse,
                        data[0].unique_offset,
                        data[0].offset_per_key(),
                        data[0].hash_indices,
                    )
                )

    def wait_and_get_swap_info(self, context: EmbCacheTrainPipelineContext) -> None:
        with record_function("## wait_and_get_swap_info ##"):
            with torch_npu.npu.stream(self._memcpy_stream):
                for module in self._pipelined_modules:
                    module_name = module.forward.name
                    swap_info_future = context.swap_info_future.pop(module_name)
                    if swap_info_future is None:
                        continue

                    swap_info = swap_info_future.get()
                    swap_info.swapout_offs = swap_info.swapout_offs.to(
                        self._npu_device, non_blocking=True
                    )
                    swap_info.swapin_offs = swap_info.swapin_offs.to(
                        self._npu_device, non_blocking=True
                    )
                    context.swap_info[module_name] = swap_info

    def swap_out(self, context: EmbCacheTrainPipelineContext) -> None:
        with record_function("## swap_out ##"):
            with torch_npu.npu.stream(self._memcpy_stream):
                for module in self._pipelined_modules:
                    module_name = module.forward.name
                    swapout_offs = context.swap_info[module_name].swapout_offs
                    if swapout_offs is None or swapout_offs.numel() == 0:
                        context.swapout_embs[module_name] = None
                        context.swapout_optims[module_name] = None
                        continue
                    _stb_eb_codegen = module.get_batched_embedding_kernels()[0][0]
                    self._memcpy_stream.wait_event(context.event_can_swapout)
                    context.swapout_embs[module_name] = _stb_eb_codegen.gather_embs(
                        swapout_offs
                    ).to(self._cpu_device, non_blocking=True)

                    context.swapout_optims[module_name] = []
                    for momentum in _stb_eb_codegen.gather_momentum(swapout_offs):
                        context.swapout_optims[module_name].append(
                            momentum.to(self._cpu_device, non_blocking=True)
                        )
                context.event_gather_swapouted.record(self._memcpy_stream)

    def host_embedding_update_async(
        self, context: EmbCacheTrainPipelineContext
    ) -> None:
        with record_function("## _host_embedding_update ##"):
            context.event_gather_swapouted.synchronize()
            for module in self._pipelined_modules:
                module_name = module.forward.name
                if context.swapout_embs[module_name] is None:
                    # 手动计数host侧embedding update执行次数，用于淘汰功能判断删除emb table中数据的时机
                    module.record_host_emb_update_times()
                    continue
                swap_info = context.swap_info[module_name]
                swapout_embs = context.swapout_embs.get(module_name)
                swapout_optims = context.swapout_optims.get(module_name)
                context.update_future[module_name] = module.host_embedding_update_async(
                    swap_info, swapout_embs, swapout_optims
                )

    def wait_host_update(self, context: EmbCacheTrainPipelineContext) -> None:
        with record_function("## wait wait_host_update ##"):
            for module in self._pipelined_modules:
                module_name = module.forward.name
                update_future = context.update_future.get(module_name)
                if update_future is not None:
                    update_future.get()

    def host_embedding_lookup_async(
        self, context: EmbCacheTrainPipelineContext
    ) -> None:
        with record_function("## host_embedding_lookup_async ##"):
            for module in self._pipelined_modules:
                module_name = module.forward.name
                swap_info = context.swap_info[module_name]
                context.swapin_tensor_future[module_name] = (
                    module.host_embedding_lookup_async(swap_info)
                )

    def swapin_tensors_to_npu(self, context: EmbCacheTrainPipelineContext) -> None:
        with record_function("## swapin_embs_optims_to_npu ##"):
            # 在_memcpy_stream将swapin tensor to npu
            with torch_npu.npu.stream(self._memcpy_stream):
                for module in self._pipelined_modules:
                    module_name = module.forward.name
                    with record_function(
                        "## embedding_lookup_async get in initiate_swap_in ##"
                    ):
                        swapin_tensors = context.swapin_tensor_future.pop(
                            module_name
                        ).get()
                    swapin_offs = context.swap_info[module_name].swapin_offs
                    swapin_offs.record_stream(self._default_stream)
                    if swapin_offs.numel() == 0:
                        context.swapin_embs[module_name] = None
                        context.swapin_optims[module_name] = None
                        continue

                    # 非阻塞拷贝到NPU设备上，在_memcpy_stream上执行
                    swapin_embs = swapin_tensors.swapin_embs.to(
                        self._npu_device, non_blocking=True
                    )  
                    swapin_optims = []
                    for optim in swapin_tensors.swapin_optims:
                        swapin_optims.append(
                            optim.to(self._npu_device, non_blocking=True)
                        )  

                    swapin_embs.record_stream(self._default_stream)
                    for swapin_optim in swapin_optims:
                        swapin_optim.record_stream(self._default_stream)

                    context.swapin_embs[module_name] = swapin_embs
                    context.swapin_optims[module_name] = swapin_optims

                context.event_swapin_scattered.record(self._memcpy_stream)

    def wait_and_swapin(self, context: EmbCacheTrainPipelineContext):
        with record_function("## wait_and_swapin ##"):
            # 在主流等待拷贝流上的event，确保数据已拷贝到NPU
            self._default_stream.wait_event(context.event_swapin_scattered)

            # 在主流scatter_update
            for module in self._pipelined_modules:
                module_name = module.forward.name
                if context.swapin_embs[module_name] is None:
                    continue

                _stb_eb_codegen = module.get_batched_embedding_kernels()[0][0]
                swapin_offs = context.swap_info[module_name].swapin_offs
                _stb_eb_codegen.scatter_update_embs(
                    swapin_offs, context.swapin_embs[module_name]
                )
                _stb_eb_codegen.scatter_update_momentum(
                    swapin_offs, context.swapin_optims[module_name]
                )

    def start_compute_swap_info(self, context: EmbCacheTrainPipelineContext):
        with record_function("## start_compute_swap_info ##"):
            for name, module in zip(
                context.input_dist_tensors_requests.keys(),
                self._pipelined_modules,
            ):
                context.sparse_features_after_post_dist[name] = (
                    context.post_input_dist_awaitable[name].wait()
                )
                self._compute_swap_info_async(context)

    def attach(self, model: Optional[torch.nn.Module] = None) -> None:
        if model:
            self._model = model

        self._model_attached = True
        if self.contexts:
            self._pipeline_model(
                batch=self.batches[0],
                context=self.contexts[0],
                pipelined_forward=EmbCachePipelinedForward,
            )
        else:
            # attaching the model after end of train pipeline
            # model rewrite for SDD needs context but self.contexts is empty
            # reset _pipelined_modules so _fill_pipeline will rewrite model on progress()
            self._pipelined_modules = []

    def fill_pipeline(self, dataloader_iter: Iterator[In]) -> None:
        # pipeline is already filled
        if len(self.batches) >= 4:
            return
        # executes last batch in pipeline
        if self.batches and self._dataloader_exhausted and self._execute_all_batches:
            return

        # batch i
        if not self.enqueue_batch(dataloader_iter):
            return

        self._init_pipelined_modules(
            # pyre-ignore [6]
            self.batches[0],
            self.contexts[0],
            EmbCachePipelinedForward,
        )
        self.wait_sparse_data_dist(self.contexts[0])
        self.do_post_input_dist(self.contexts[0])
        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cast(In, self.batches[0]), self._data_dist_stream)
        self.start_compute_swap_info(self.contexts[0])
        self.wait_and_get_swap_info(self.contexts[0])
        self.host_embedding_lookup_async(self.contexts[0])
        self.do_restore_async(self.contexts[0])
        self.contexts[0].event_can_swapout.record(self._default_stream)
        self.swap_out(self.contexts[0])

        # batch i+1
        if not self.enqueue_batch(dataloader_iter):
            return
        self.start_sparse_data_dist(self.batches[1], self.contexts[1])
        _fuse_input_dist_splits(self.contexts[1])
        self.wait_sparse_data_dist(self.contexts[1])
        self.do_post_input_dist(self.contexts[1])
        with record_function("## wait_for_batch##"):
            _wait_for_batch(cast(In, self.batches[1]), self._data_dist_stream)
        self.start_compute_swap_info(self.contexts[1])

        # batch i+2
        if not self.enqueue_batch(dataloader_iter):
            return
        self.start_sparse_data_dist(self.batches[2], self.contexts[2])
        _fuse_input_dist_splits(self.contexts[2])
        self.wait_sparse_data_dist(self.contexts[2])
        self.do_post_input_dist(self.contexts[2])
        with record_function("## wait_for_batch##"):
            _wait_for_batch(cast(In, self.batches[2]), self._data_dist_stream)

        # batch i+3
        if not self.enqueue_batch(dataloader_iter):
            return
        self.start_sparse_data_dist(self.batches[3], self.contexts[3])
        _fuse_input_dist_splits(self.contexts[3])

        # batch i+4 ~ batchi+4+self.local_unique_parallel_batch_num-1
        for i in range(self.local_unique_parallel_batch_num):
            if not self.enqueue_batch(dataloader_iter):
                return
            self.start_sparse_data_dist(self.batches[4 + i], self.contexts[4 + i])

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        self._global_steps += 1
        if not self._model_attached:
            self.attach(self._model)

        self.fill_pipeline(dataloader_iter)
        if not self.batches:
            raise StopIteration

        self._set_module_context(self.contexts[0])

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        # wait batch_ip2
        if len(self.batches) >= 4:
            self.wait_sparse_data_dist(self.contexts[3])
            with record_function("## wait_for_batch ##"):
                _wait_for_batch(cast(In, self.batches[3]), self._data_dist_stream)

        if len(self.batches) >= 5:
            _fuse_input_dist_splits(self.contexts[4])

        self.enqueue_batch(dataloader_iter)
        if len(self.batches) >= 5 + self.local_unique_parallel_batch_num:
            self.start_sparse_data_dist(
                self.batches[4 + self.local_unique_parallel_batch_num],
                self.contexts[4 + self.local_unique_parallel_batch_num],
            )

        # 更新 host 侧的 Embedding 和 优化器参数
        self.contexts[0].event_gather_swapouted.synchronize()
        self.host_embedding_update_async(self.contexts[0])

        # 等待 swapin tensor 并 swapin
        self.swapin_tensors_to_npu(self.contexts[0])
        self.wait_and_swapin(self.contexts[0])

        # forward
        with record_function("## forward ##"):
            losses, output = self._model_fwd(self.batches[0])

        # 等待 i+1 swap 参数计算完成
        if len(self.batches) >= 2:
            self.wait_and_get_swap_info(self.contexts[1])

        # 处理淘汰信息
        # 因fill_pipeline 内已经执行了2次 get_swap_info, 因此对_evict_step_interval取余时要对global_step+1
        if (
            self._evict_step_interval
            and (self._global_steps + 1) % self._evict_step_interval == 0
        ):
            with record_function("## feature_evict ##"):
                self._start_feature_evict()

        # 计算 i+2 batch 的 swap pairs 和 key2offset
        if len(self.batches) >= 3:
            self.start_compute_swap_info(self.contexts[2])

        # batches[i+1] 在host侧预查
        if len(self.batches) >= 2:
            # 保险起见，i+1轮的swapout等i轮的swapin做完之后做
            self.contexts[1].event_can_swapout.record(self._default_stream)
            self.host_embedding_lookup_async(self.contexts[1])
            self.do_restore_async(self.contexts[1])

        if self._model.training:
            # backward
            with record_function("## backward ##"):
                torch.sum(losses, dim=0).backward()
            # update
            with record_function("## optimizer ##"):
                self._optimizer.step()

        if len(self.batches) >= 4:
            self.do_post_input_dist(self.contexts[3])
        # swapout
        if len(self.batches) >= 2:
            self.swap_out(self.contexts[1])
        self.wait_host_update(self.contexts[0])

        self.dequeue_batch()
        if self._return_loss:
            return output, losses
        else:
            return output

    def _start_feature_evict(self):
        logging.info("Start invoke embcache_mgr.evict_features()")
        for module in self._pipelined_modules:
            module.host_embedding_evict()


class SimpleEmbCacheTrainPipelineSparseDist(EmbCacheTrainPipelineSparseDist):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cpu_device: torch.device,
        npu_device: torch.device,
        return_loss: bool = False,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        context_type: Type[EmbCacheTrainPipelineContext] = EmbCacheTrainPipelineContext,
        # keep for backward compatibility
        pipeline_postproc: bool = False,
        custom_model_fwd: Optional[
            Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
        ] = None,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            cpu_device=cpu_device,
            npu_device=npu_device,
            return_loss=return_loss,
            execute_all_batches=execute_all_batches,
            apply_jit=apply_jit,
            context_type=context_type,
            pipeline_postproc=pipeline_postproc,
            custom_model_fwd=custom_model_fwd,
        )
        self.contexts: Deque[EmbCacheTrainPipelineContext] = deque()
        self._cpu_device = cpu_device
        self._npu_device = npu_device
        self._return_loss = return_loss

    def fill_pipeline(self, dataloader_iter: Iterator[In]) -> None:
        # pipeline is already filled
        if len(self.batches) >= 2:
            return
        # executes last batch in pipeline
        if self.batches and self._dataloader_exhausted and self._execute_all_batches:
            return

        # batch i
        if not self.enqueue_batch(dataloader_iter):
            return

        self._init_pipelined_modules(
            # pyre-ignore [6]
            self.batches[0],
            self.contexts[0],
            EmbCachePipelinedForward,
        )
        _fuse_input_dist_splits(self.contexts[0])
        self.wait_sparse_data_dist(self.contexts[0])
        self.do_post_input_dist(self.contexts[0])

        # batch i+1
        if not self.enqueue_batch(dataloader_iter):
            return

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        time0 = time.time()
        if not self._model_attached:
            self.attach(self._model)

        time1 = time.time()
        self.fill_pipeline(dataloader_iter)
        if not self.batches:
            raise StopIteration
        time2 = time.time()

        self._set_module_context(self.contexts[0])
        time3 = time.time()
        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()
        time4 = time.time()

        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cast(In, self.batches[0]), self._data_dist_stream)
        time5 = time.time()
        with record_function("## start_sparse_data_dist ##"):
            if len(self.batches) >= 2:
                self.start_sparse_data_dist(self.batches[1], self.contexts[1])
        time6 = time.time()

        with record_function("## enqueue_batch ##"):
            # batch i+2
            self.enqueue_batch(dataloader_iter)
        time7 = time.time()

        with record_function("## get_swap_info ##"):
            # get swap_info
            self.start_compute_swap_info(self.contexts[0])
            self.wait_and_get_swap_info(self.contexts[0])
        time8 = time.time()
        self.do_restore_async(self.contexts[0])

        with record_function("## swapout ##"):
            # swapout
            self.contexts[0].event_can_swapout.record(self._default_stream)
            self.swap_out(self.contexts[0])
        time9 = time.time()

        with record_function("## host_embedding_update ##"):
            self.host_embedding_update_async(self.contexts[0])
            self.wait_host_update(self.contexts[0])
        time10 = time.time()

        with record_function("## host_embedding_lookup_async ##"):
            self.host_embedding_lookup_async(self.contexts[0])
        time11 = time.time()

        with record_function("## swapin ##"):
            # swapin
            self.swapin_tensors_to_npu(self.contexts[0])
            self.wait_and_swapin(self.contexts[0])
        time12 = time.time()

        # forward
        with record_function("## forward ##"):
            losses, output = self._model_fwd(self.batches[0])

        if len(self.batches) >= 2:
            _fuse_input_dist_splits(self.contexts[1])
            self.wait_sparse_data_dist(self.contexts[1])
            self.do_post_input_dist(self.contexts[1])

        if self._model.training:
            # backward
            with record_function("## backward ##"):
                torch.sum(losses, dim=0).backward()

            # update
            with record_function("## optimizer ##"):
                self._optimizer.step()
        time13 = time.time()

        self.dequeue_batch()
        time14 = time.time()
        if self._return_loss:
            return output, losses
        else:
            return output
