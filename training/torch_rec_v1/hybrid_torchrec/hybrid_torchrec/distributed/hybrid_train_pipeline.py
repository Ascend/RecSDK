#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# ==============================================================================
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    cast,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
)
import torch
from torch.autograd.profiler import record_function
import torch_npu
from hybrid_torchrec.distributed.sharding.hybrid_rw_sharding import (
    HashRwSparseFeaturesDistAwaitable,
    InputDistThreadPoolExecutorSingleton,
)
from torchrec.distributed.embedding_types import KJTList
from torchrec.distributed.types import Awaitable
from torchrec.distributed.embedding_sharding import KJTListAwaitable
from torchrec.streamable import Multistreamable, Pipelineable
from torchrec.distributed.train_pipeline.utils import (
    FusedKJTListSplitsAwaitable,
    _to_device,
    _rewrite_model,
    _override_input_dist_forwards,
    ShardedModule,
    TrainPipelineContext,
    PipelinedForward,
    PrefetchPipelinedForward,
    EmbeddingPipelinedForward,
    _build_args_kwargs,
    defaultdict,
    KJTListSplitsAwaitable,
    KJTSplitsAllToAllMeta,
)
from torchrec.distributed.train_pipeline.train_pipelines import TrainPipelineSparseDist

logger: logging.Logger = logging.getLogger(__name__)

In = TypeVar("In", bound=Pipelineable)
Out = TypeVar("Out")

FRONT_BATCH = 0
SECOND_BATCH = 1
THIRD_BATCH = 2

MAX_PIPE_N_BATCH = 12


class TaskType(Enum):
    SPLIT = 0
    FIRST_ALL2ALL = 1
    SENCOND_ALL2ALL = 2
    POST_INPUT = 3
    COPY2NPU = 4


FIRST_CONTEXT_TASK_POS = TaskType.COPY2NPU.value - TaskType.COPY2NPU.value
SECOND_CONTEXT_TASK_POS = TaskType.COPY2NPU.value - TaskType.POST_INPUT.value
THIRD_CONTEXT_TASK_POS = TaskType.COPY2NPU.value - TaskType.SENCOND_ALL2ALL.value
FOURTH_CONTEXT_TASK_POS = TaskType.COPY2NPU.value - TaskType.FIRST_ALL2ALL.value
FIFTH_CONTEXT_TASK_POS = TaskType.COPY2NPU.value - TaskType.SPLIT.value


class AwaitableApapter(Awaitable):
    def __init__(self, awaitable) -> None:
        super().__init__()

        def get_awaitable_result(awaitable: Awaitable):
            return awaitable.wait()

        self.future = InputDistThreadPoolExecutorSingleton().executor.submit(
            get_awaitable_result, awaitable
        )

    def _wait_impl(self) -> Any:
        return self.future.result()


@dataclass
class HybridTrainPipelineContext:
    """
    Context information for a `TrainPipelineSparseDist` instance.

    Attributes:
        batch (In): Stores input dist
            requests in the splits awaitable stage, which occurs after starting the
            input dist.
        awaitables (List): List of fused splits input dist awaitable.
        module_contexts (Dict[str, Multistreamable]): Stores module contexts from the
            input dist for the current batch.
    """

    batch: In = None
    awaitables: List = field(default_factory=list)
    module_contexts: Dict[str, Multistreamable] = field(default_factory=dict)
    preproc_fwd_results: Dict[str, Any] = field(default_factory=dict)
    index: Optional[int] = None
    version: int = 0


def _fuse_input_dist_splits(context: HybridTrainPipelineContext) -> None:
    if context.batch is None:
        return
    names_per_pg = defaultdict(list)
    for name, request in context.awaitables[TaskType.SPLIT.value].items():
        pg = None
        if isinstance(request, KJTListSplitsAwaitable):
            for awaitable in request.awaitables:
                if isinstance(awaitable, KJTSplitsAllToAllMeta) or isinstance(
                    awaitable, HashRwSparseFeaturesDistAwaitable
                ):
                    pg = awaitable.pg
                    break
        names_per_pg[pg].append(name)

    for pg, names in names_per_pg.items():
        for ind, awaitable in enumerate(
            context.awaitables[TaskType.SPLIT.value][name].awaitables
        ):
            if isinstance(awaitable, HashRwSparseFeaturesDistAwaitable):
                context.awaitables[TaskType.SPLIT.value][name].awaitables[ind] = (
                    context.awaitables[TaskType.SPLIT.value][name]
                    .awaitables[ind]
                    .wait()
                )

        context.awaitables[TaskType.FIRST_ALL2ALL.value]["".join(names)] = (
            names,
            FusedKJTListSplitsAwaitable(
                requests=[
                    context.awaitables[TaskType.SPLIT.value][name] for name in names
                ],
                contexts=[(context.module_contexts[name]) for name in names],
                pg=pg,
            ),
        )


def _start_data_dist(
    pipelined_modules: List[ShardedModule],
    batch: Pipelineable,
    context: HybridTrainPipelineContext,
) -> None:

    context.awaitables = [None for _ in range(len(TaskType))]
    for taks_type in TaskType:
        context.awaitables[taks_type.value] = {}

    for module in pipelined_modules:
        forward = module.forward
        args, kwargs = _build_args_kwargs(batch, forward.args)

        # Start input distribution.
        module_ctx = module.create_context()

        context.module_contexts[forward.name] = module_ctx
        context.awaitables[TaskType.SPLIT.value][forward.name] = module.input_dist(
            module_ctx, *args, **kwargs
        )


class HybridPipelinedForward(PipelinedForward):
    def __call__(self, *inputs, **kwargs) -> Awaitable:
        self._context: HybridTrainPipelineContext
        if self._name not in self._context.awaitables[TaskType.COPY2NPU.value]:
            raise ValueError(f"{self._name} is not in TaskType.COPY2NPU.value")
        data = self._context.awaitables[TaskType.COPY2NPU.value][self._name]

        if self._stream is not None:
            torch_npu.npu.current_stream().wait_stream(self._stream)
            cur_stream = torch_npu.npu.current_stream()

            if not isinstance(data, (torch.Tensor, Multistreamable)):
                raise ValueError(
                    f"{type(data)} must implement Multistreamable interface"
                )

            data.record_stream(cur_stream)

            ctx = self._context.module_contexts[self._name]
            ctx.record_stream(cur_stream)

        return self._module.compute_and_output_dist(
            self._context.module_contexts[self._name], data
        )

    def set_current_context(self, context: HybridTrainPipelineContext):
        self._context = context


def kjt_list_to_device(batch: KJTList, device: torch.device, non_blocking: bool) -> In:
    if not isinstance(batch, (KJTList)):
        raise ValueError(f"{type(batch)} must be KJTList")
    result = KJTList(
        [
            f.pin_memory().to(device=device, non_blocking=non_blocking)
            for f in batch.features
        ]
    )
    return result


class HybridTrainPipelineSparseDist(TrainPipelineSparseDist[In, Out]):
    """
    This pipeline overlaps device transfer, and `ShardedModule.input_dist()` with
    forward and backward. This helps hide the all2all latency while preserving the
    training forward / backward ordering.

    `ShardedModule.input_dist()` is only done for top-level modules in the call graph.
    To be considered a top-level module, a module can only depend on 'getattr' calls on
    input.

    Input model must be symbolically traceable with the exception of `ShardedModule` and
    `DistributedDataParallel` modules.

    Args:
        model (torch.nn.Module): model to pipeline.
        optimizer (torch.optim.Optimizer): optimizer to use.
        device (torch.device): device where device transfer, sparse data dist, and
            forward/backward pass will happen.
        execute_all_batches (bool): executes remaining batches in pipeline after
            exhausting dataloader iterator.
        apply_jit (bool): apply torch.jit.script to non-pipelined (unsharded) modules.
        return_loss (bool): return loss or not.
        pipe_n_batch (int): pipe_n_batch pipelines to progress.  
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        return_loss: bool = False,
        pipe_n_batch: int = 6,
    ) -> None:
        self.param_check(model, device, pipe_n_batch, apply_jit, execute_all_batches)
        super().__init__(model, optimizer, device, execute_all_batches, apply_jit)
        self._return_loss = return_loss
        self._contexts = [[] for _ in range(pipe_n_batch)]
        self._current_line_id = 0
        self._batch_forward = None
        self._pipe_n_batch = pipe_n_batch
        torch.set_num_threads(1)

    def param_check(
        self,
        model: torch.nn.Module,
        device: torch.device,
        pipe_n_batch,
        apply_jit, 
        execute_all_batches
    ):
        if not isinstance(pipe_n_batch, int):
            raise ValueError(f"pipe_n_batch must be an int but got type: {type(pipe_n_batch)}")

        if pipe_n_batch <= 0 or pipe_n_batch > MAX_PIPE_N_BATCH:
            raise ValueError(f"pipe_n_batch must be in range in [1, {MAX_PIPE_N_BATCH}], "
                             f"but pipe_n_batch is {pipe_n_batch}")

        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"model expected to be an instance of torch.nn.Module, "
                            f"but got {type(model)} instead.")

        if not isinstance(device, torch.device):
            raise TypeError(f"device expected to be an instance of torch.device, "
                            f"but got {type(device)} instead.")

        if device.type != "npu":
            raise ValueError(f"device type only support npu, but got {device.type}.")

        if model.device is None or not isinstance(model.device, torch.device):
            raise ValueError(f"model.device must be not None and type is torch.device, but got {type(model.device)}.")

        if model.device != device:
            raise ValueError(f"model device is {model.device}, but input device is {device}.")
        
        if apply_jit:
            raise ValueError(f"apply_jit is not support")
        
        if not execute_all_batches:
            raise ValueError(f"execute_all_batches cant not be false")
    
    def enque_context(self, line_id, context: HybridTrainPipelineContext):
        self._contexts[line_id].append(context)

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        self._fill_pipeline(dataloader_iter)

        if self._model.training:
            if not isinstance(self._optimizer, torch.optim.Optimizer):
                raise TypeError(f"self._optimizer expected to be an instance of torch.optim.Optimizer, \
                                but got {type(self._optimizer)} instead.")
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        cur_line_id = self._current_line_id % self._pipe_n_batch
        next_line_id = (self._current_line_id + 1) % self._pipe_n_batch

        cur_context_forward = self._contexts[cur_line_id].pop(0)
        with record_function("## wait_for_batch ##"):
            self._wait_for_batch(
                cast(In, cur_context_forward.batch),
                self._memcpy_stream,
                cur_context_forward,
            )

        self._do_post_input_dist(self._contexts[cur_line_id][FIRST_CONTEXT_TASK_POS])

        # 此条线的顶端batch已经结束，下一条线还是拷贝到npu上
        self._contexts[next_line_id][FIRST_CONTEXT_TASK_POS].batch = self._copy_to_npu(
            self._contexts[next_line_id][FIRST_CONTEXT_TASK_POS].batch,
            self._contexts[next_line_id][FIRST_CONTEXT_TASK_POS],
        )

        with record_function("## forward ##"):
            losses, output = cast(
                Tuple[torch.Tensor, Out], self._model(cur_context_forward.batch)
            )

        # 每一个batch向前移动一个位置
        self._wait_sparse_data_dist(
            self._contexts[cur_line_id][SECOND_CONTEXT_TASK_POS]
        )

        if self._model.training:
            # backward
            with record_function("## backward ##"):
                torch.sum(losses, dim=0).backward()

        _fuse_input_dist_splits(self._contexts[cur_line_id][THIRD_CONTEXT_TASK_POS])

        if self._model.training:
            # update
            with record_function("## optimizer ##"):
                self._optimizer.step()

        # 每一个batch向前移动一个位置
        cur_context_forward.batch = self._get_next_batch(dataloader_iter)
        self._start_sparse_data_dist(
            cur_context_forward.batch,
            cur_context_forward,
        )
        self.enque_context(cur_line_id, cur_context_forward)

        self._current_line_id += 1

        if self._return_loss:
            return output, losses
        else:
            return output

    def _get_next_batch(self, dataloader_iter):
        with record_function("## get_next_batch ##"):
            batch = next(dataloader_iter, None)
            if batch is None and not self._execute_all_batches:
                raise StopIteration
            return batch

    def _fill_pipeline(self, dataloader_iter: Iterator[In]) -> None:
        # executes last batch in pipeline
        start_id = self._current_line_id % self._pipe_n_batch
        if (
            len(self._contexts[start_id]) > 0
            and self._contexts[start_id][0].batch is not None
        ):
            return
        logging.info("fill pipe")

        # 重新创建lines的context
        self._contexts = [[] for _ in range(self._pipe_n_batch)]
        self._current_line_id = 0

        # batch 1
        batch_start = self._get_next_batch(dataloader_iter)

        if batch_start is None:
            raise StopIteration
        init_context = HybridTrainPipelineContext()
        self._init_pipelined_modules(batch_start, context=init_context)

        # 1级流水
        for line_i in range(self._pipe_n_batch):

            if line_i == start_id:
                context = init_context
                context.batch = batch_start
            else:
                context = HybridTrainPipelineContext()
                context.batch = self._get_next_batch(dataloader_iter)
            self._start_sparse_data_dist(context.batch, context=context)
            _fuse_input_dist_splits(context)
            self._wait_sparse_data_dist(context=context)
            self._do_post_input_dist(context=context)
            self.enque_context(line_i, context)

        context: HybridTrainPipelineContext = self._contexts[start_id][0]
        context.batch = self._copy_to_npu(context.batch, context)

        # 2级流水
        for line_i in range(self._pipe_n_batch):
            context = HybridTrainPipelineContext()
            context.batch = self._get_next_batch(dataloader_iter)

            self._start_sparse_data_dist(context.batch, context)
            _fuse_input_dist_splits(context)
            self._wait_sparse_data_dist(context=context)
            self.enque_context(line_i, context)

        # 3级流水
        for line_i in range(self._pipe_n_batch):
            context = HybridTrainPipelineContext()
            context.batch = self._get_next_batch(dataloader_iter)

            self._start_sparse_data_dist(context.batch, context)
            _fuse_input_dist_splits(context)
            self.enque_context(line_i, context)

        # 4级流水
        for line_i in range(self._pipe_n_batch):
            context = HybridTrainPipelineContext()
            context.batch = self._get_next_batch(dataloader_iter)

            self._start_sparse_data_dist(context.batch, context)
            self.enque_context(line_i, context)

    def _wait_for_batch(
        self,
        batch: In,
        stream: Optional[torch.cuda.streams.Stream],
        context: HybridTrainPipelineContext,
    ) -> None:
        if batch is None:
            return
        if stream is not None:
            torch_npu.npu.current_stream().wait_stream(stream)

        for module in self._pipelined_modules:
            module.forward.set_current_context(context)

        if stream is None:
            return
        cur_stream = torch_npu.npu.current_stream()
        if not isinstance(batch, (torch.Tensor, Multistreamable)):
            raise ValueError(f"{type(batch)} must implement Multistreamable interface")

        batch.record_stream(cur_stream)
        for forward_name in context.awaitables[TaskType.COPY2NPU.value].keys():
            context.awaitables[TaskType.COPY2NPU.value][forward_name].record_stream(
                cur_stream
            )

    def _do_post_input_dist(self, context: HybridTrainPipelineContext):
        if context.batch is None:
            return
        with record_function("## _post_input_dist ##"):
            for name, module in zip(
                context.awaitables[TaskType.SENCOND_ALL2ALL.value].keys(),
                self._pipelined_modules,
            ):
                awaitable = context.awaitables[TaskType.SENCOND_ALL2ALL.value][name]
                kjt_list = awaitable.wait()

                if hasattr(module, "post_input_dist"):
                    post_waitable = module.post_input_dist(
                        context.module_contexts[name], kjt_list
                    )
                    context.awaitables[TaskType.POST_INPUT.value][name] = post_waitable
                else:
                    raise RuntimeError(
                        "HybridTrainPipelineSparseDist can't be used for module with no post_input method"
                    )

    def _copy_to_npu(
        self, batch: In, context: HybridTrainPipelineContext
    ) -> Optional[In]:
        """
        Retrieves batch from dataloader and moves it to the provided device.

        Raises:
            StopIteration: if the dataloader iterator is exhausted; unless
                `self._execute_all_batches=True`, then returns None.
        """
        if batch is None:
            return None
        for preproc_mod in self._pipelined_preprocs:
            preproc_mod.set_context(context)
        with record_function("## _copy_to_npu ##"):
            with torch_npu.npu.stream(self._memcpy_stream):
                for name in context.awaitables[TaskType.POST_INPUT.value].keys():
                    kjt_list = context.awaitables[TaskType.POST_INPUT.value][
                        name
                    ].wait()
                    context.awaitables[TaskType.COPY2NPU.value][name] = (
                        kjt_list_to_device(kjt_list, self._device, non_blocking=True)
                    )
                if batch is not None:
                    batch = _to_device(batch, self._device, non_blocking=True)
                elif not self._execute_all_batches:
                    raise StopIteration
                return batch

    def _init_pipelined_modules(
        self, batch: In, context: HybridTrainPipelineContext
    ) -> None:
        """
        Retrieves the pipelined modules after overriding their forwards, initializes the
        modules' input dists, and overrides the input dist forwards to support fusing
        the splits collective in the input dist.
        """
        if self._pipelined_modules:
            return
        (
            self._pipelined_modules,
            self._model,
            self._original_forwards,
            self._pipelined_preprocs,
            _,
        ) = _rewrite_model(
            model=self._model,
            context=context,
            dist_stream=None,
            batch=batch,
            apply_jit=self._apply_jit,
            pipelined_forward=HybridPipelinedForward,
        )
        # initializes input dist, so we can override input dist forwards
        self._start_sparse_data_dist(batch, context)
        _fuse_input_dist_splits(context)
        _override_input_dist_forwards(self._pipelined_modules)

    def _start_sparse_data_dist(
        self, batch: Optional[In], context: HybridTrainPipelineContext
    ) -> None:
        """
        Waits for batch to finish getting copied to GPU, then starts the input dist.
        """
        if batch is None:
            return
        with record_function("## start_sparse_data_dist ##"):
            # Temporarily set context for next iter to populate cache
            for preproc_mod in self._pipelined_preprocs:
                preproc_mod.set_context(context)
            _start_data_dist(self._pipelined_modules, batch, context)

    def _wait_sparse_data_dist(self, context: HybridTrainPipelineContext) -> None:
        """
        Waits on the input dist splits requests to get the input dist tensors requests,
        and populates the context with them.
        """
        if context.batch is None:
            return
        with record_function("## wait_sparse_data_dist ##"):
            for names, awaitable in context.awaitables[
                TaskType.FIRST_ALL2ALL.value
            ].values():
                for name, request in zip(names, awaitable.wait()):
                    context.awaitables[TaskType.SENCOND_ALL2ALL.value][name] = (
                        AwaitableApapter(request)
                    )
