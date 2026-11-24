#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.
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
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator, Type, Optional, Dict, Any, Tuple, List, cast, TypeVar

import torch
from torch.autograd.profiler import record_function
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.embedding import EmbeddingCollectionContext
from torchrec.distributed.embedding_types import KJTList
from torchrec.distributed.train_pipeline import _wait_for_batch
from torchrec.distributed.train_pipeline.utils import TrainPipelineContext, PipelinedForward
from torchrec.distributed.types import Awaitable
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
from torchrec.streamable import Multistreamable, Pipelineable
import torch_npu


In = TypeVar("In", bound=Pipelineable)

FIRST_CONTEXT_TASK_POS = 0
SECOND_CONTEXT_TASK_POS = 1
THIRD_CONTEXT_TASK_POS = 2
FOURTH_CONTEXT_TASK_POS = 3
FIFTH_CONTEXT_TASK_POS = 4
PIPELINE_FILLED_LEN = 3

class AsyncEmbeddingPipelinedForward(PipelinedForward):
    
    def __call__(self, *args, **kwargs) -> Awaitable:

        data = self._context.detach_embedding[self._name]
        self._context.module_contexts.pop(self._name)

        return data
    
    def output(self, stream: Optional[torch.cuda.streams.Stream]):
        with torch_npu.npu.stream(self._stream):
            # Make sure that both result of input_dist and context
            # are properly transferred to the current stream
            request = self._context.input_dist_tensors_requests.pop(self._name)
            assert isinstance(request, Awaitable)
            with record_function("## runtime_forward_assemble_KJT ##"):
                # Finish waiting on the dist_stream,
                # in case some delayed stream scheduling happens during the wait() call.
                with torch.get_device_module(self._device).stream(self._stream):
                    data = request.wait()

                # Make sure that both result of input_dist and context
                # are properly transferred to the current stream.
                ctx = self._context.module_contexts[self._name]

            if self._stream is not None:
                torch.get_device_module(self._device).current_stream().wait_stream(
                    self._stream
                )

                assert isinstance(
                    data, (torch.Tensor, Multistreamable)
                ), f"{type(data)} must implement Multistreamable interface"
                data.record_stream(stream)
                ctx.record_stream(stream)

            return self._module.compute_and_output_dist(ctx, data)


@dataclass
class AsyncEmbeddingTrainPipelineContext(TrainPipelineContext):
    compute_and_output_result: Dict[str, Multistreamable] = field(default_factory=dict)
    detach_embedding: Dict[str, Awaitable[Any]] = field(default_factory=dict)
    detach_embedding_grad: Dict[str, Awaitable[Any]] = field(default_factory=dict)
    sharding_plan: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    context_batch_id: int = 0


class TrainPipelineSparseDistAsyncEmbedding(TrainPipelineSparseDist):
    def __init__(self, 
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 execute_all_batches: bool = True,
                 apply_jit: bool = False,
                 context_type: Type[TrainPipelineContext] = AsyncEmbeddingTrainPipelineContext,
                 pipeline_postproc: bool = False,
                 sharding_plan: dict = None
                ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            execute_all_batches=execute_all_batches,
            apply_jit=apply_jit,
            context_type=context_type,
            pipeline_postproc=pipeline_postproc
        )
        self.forward_fn = AsyncEmbeddingPipelinedForward
        self._embedding_stream = (
            (torch.get_device_module(device).Stream(priority=-1))
            if device.type in ["cuda", "mtia", "npu"]
            else None
        )
        self._dense_stream = (
            (torch.get_device_module(device).Stream(priority=-1))
            if device.type in ["cuda", "mtia", "npu"]
            else None
        )
        self._data_dist_stream = (
            (torch.get_device_module(device).Stream(priority=-1))
            if device.type in ["cuda", "mtia", "npu"]
            else None
        )
        self.backward_sync_event = torch.npu.Event(enable_timing=False)
        self.forward_sync_event = torch.npu.Event(enable_timing=False)
        self.compute_unique_sync_event = torch.npu.Event(enable_timing=False)
        self._i_batch = 0
        self.batches = deque()
        self.contexts = deque()
        self.sharding_plan = sharding_plan

    def fill_first_batch(self):
        self.contexts[FIRST_CONTEXT_TASK_POS].context_batch_id = self._i_batch
        self._init_pipelined_modules(
            # pyre-ignore [6]
            self.batches[FIRST_CONTEXT_TASK_POS],
            self.contexts[FIRST_CONTEXT_TASK_POS],
            self.forward_fn,
        )
        self.pipelined_modules = self._pipelined_modules
        self.pipeline_module_names = [module.forward.name for module in self.pipelined_modules]
        self.wait_sparse_data_dist(self.contexts[FIRST_CONTEXT_TASK_POS])
        if self.pipelined_modules:
            self._set_module_context(self.contexts[FIRST_CONTEXT_TASK_POS])
            self.set_sharding_plan(self.contexts[FIRST_CONTEXT_TASK_POS])

        with torch_npu.npu.stream(self._embedding_stream):
            self.get_embedding(
                self.contexts[FIRST_CONTEXT_TASK_POS], self.batches[FIRST_CONTEXT_TASK_POS], self._embedding_stream
                )

        with torch_npu.npu.stream(self._dense_stream):
            torch.get_device_module(self._device).current_stream().wait_stream(self._embedding_stream)
            self.get_detach_embedding(self.contexts[FIRST_CONTEXT_TASK_POS])
            dense_loss, _ = self._model(self.batches[FIRST_CONTEXT_TASK_POS])
            self._zero_grad()
            if self._model.training:
                dense_loss.sum().backward()
            self.store_embedding_grads(self.contexts[FIRST_CONTEXT_TASK_POS])
            self.backward_sync_event.record(self._dense_stream)
            self._optimizer.step()

    def _zero_grad(self):
        self._optimizer.zero_grad()

    def fill_pipeline(self, data_iter: Iterator):
        # pipeline is already filled
        if len(self.batches) >= PIPELINE_FILLED_LEN:
            return
        
        # executes last batch in pipeline
        if self.batches and self._execute_all_batches:
            self.clear_prefetched_batches()
            return
        
        # batch i
        if not self.enqueue_batch(data_iter):
            return
        self.fill_first_batch()

        # batch i+1
        if not self.enqueue_batch(data_iter):
            return
        self.contexts[SECOND_CONTEXT_TASK_POS].context_batch_id = self._i_batch + SECOND_CONTEXT_TASK_POS
        self.start_sparse_data_dist(self.batches[SECOND_CONTEXT_TASK_POS], self.contexts[SECOND_CONTEXT_TASK_POS])
        self.wait_sparse_data_dist(self.contexts[SECOND_CONTEXT_TASK_POS])

        with torch_npu.npu.stream(self._embedding_stream):
            if self.pipelined_modules:
                self._set_module_context(self.contexts[SECOND_CONTEXT_TASK_POS])
                self.set_sharding_plan(self.contexts[SECOND_CONTEXT_TASK_POS])
            self.get_embedding(
                self.contexts[SECOND_CONTEXT_TASK_POS], self.batches[SECOND_CONTEXT_TASK_POS], self._embedding_stream
                )
        with torch_npu.npu.stream(self._dense_stream):
            torch.get_device_module(self._device).current_stream().wait_stream(self._embedding_stream)
            self.get_detach_embedding(self.contexts[SECOND_CONTEXT_TASK_POS])

        # batch i+2
        if not self.enqueue_batch(data_iter):
            return
        self.contexts[THIRD_CONTEXT_TASK_POS].context_batch_id = self._i_batch + THIRD_CONTEXT_TASK_POS
        self.start_sparse_data_dist(self.batches[THIRD_CONTEXT_TASK_POS], self.contexts[THIRD_CONTEXT_TASK_POS])
        self.wait_sparse_data_dist(self.contexts[THIRD_CONTEXT_TASK_POS])

        # batch i+3
        if not self.enqueue_batch(data_iter):
            return
        self.contexts[FOURTH_CONTEXT_TASK_POS].context_batch_id = self._i_batch + FOURTH_CONTEXT_TASK_POS

    def progress(self, data_iter: Iterator):
        if self._model.training:
            return self.train_process(data_iter)
        else:
            return self.eval_process(data_iter)
        
    def train_process(self, data_iter: Iterator):
        self._i_batch += 1
        self.model_attach()
        self.fill_pipeline(data_iter)
        self.check_pipeline_stop()

        self._set_module_context(self.contexts[FIRST_CONTEXT_TASK_POS])

        # batch i sparse backward
        with torch_npu.npu.stream(self._embedding_stream):
            self.sparse_backward(self.contexts[FIRST_CONTEXT_TASK_POS], self._embedding_stream)
        # batch i+3 feature all2all
        if len(self.batches) >= FOURTH_CONTEXT_TASK_POS + 1:
            with torch_npu.npu.stream(self._data_dist_stream):
                self.feature_all2all(self.batches[FOURTH_CONTEXT_TASK_POS], self.contexts[FOURTH_CONTEXT_TASK_POS])

        # batch i+1 dense forward
        with torch_npu.npu.stream(self._dense_stream):
            dense_loss, _ = self.dense_forward(
                self.contexts[SECOND_CONTEXT_TASK_POS], self.batches[SECOND_CONTEXT_TASK_POS]
                )

        # batch i+2 sparse forward
        if len(self.batches) >= THIRD_CONTEXT_TASK_POS + 1:
            if self.pipelined_modules:
                self._set_module_context(self.contexts[THIRD_CONTEXT_TASK_POS])
                self.set_sharding_plan(self.contexts[THIRD_CONTEXT_TASK_POS])
            with torch_npu.npu.stream(self._embedding_stream):
                self.wait_sparse_data_dist(self.contexts[THIRD_CONTEXT_TASK_POS])
                self.get_embedding(
                    self.contexts[THIRD_CONTEXT_TASK_POS], self.batches[THIRD_CONTEXT_TASK_POS], self._embedding_stream
                    )
                self.forward_sync_event.record(self._embedding_stream)
        
        # batch i+2 get_detach_embedding
        if len(self.batches) >= THIRD_CONTEXT_TASK_POS + 1:
            with torch_npu.npu.stream(self._dense_stream):
                _wait_for_batch(cast(In, self.batches[THIRD_CONTEXT_TASK_POS]), self._embedding_stream)
                self.forward_sync_event.wait(self._dense_stream)
                self.get_detach_embedding(self.contexts[THIRD_CONTEXT_TASK_POS])

        # batch i+1 dense backward
        with torch_npu.npu.stream(self._dense_stream):
            self._zero_grad()
            output = self.dense_backward(dense_loss)

        # batch i+1 dense update
        with record_function("## batch i+1 optimizer step ##"):
            with torch_npu.npu.stream(self._dense_stream):
                self._optimizer.step()

        # batch i+4 dataloader
        self.get_new_batch(data_iter)
        self.contexts[FIFTH_CONTEXT_TASK_POS].context_batch_id = self._i_batch + FIFTH_CONTEXT_TASK_POS
        with record_function("## dequeue_batch ##"):
            self.dequeue_batch()
            loss = output.detach()
        return loss, 0

    def eval_process(self, data_iter: Iterator):
        raise NotImplementedError("Eval process is not implemented in this demo.")

    def model_attach(self):
        if not self._model_attached:
            self.attach(self._model)

    def feature_all2all(self, batch: Any, context: TrainPipelineContext):
        with record_function("## batch i+3 start_sparse_data_dist ##"):
            self.start_sparse_data_dist(batch, context)

    def get_new_batch(self, data_iter: Iterator):
        with record_function("## batch i+4 dataloader ##"):
            self.enqueue_batch(data_iter)

    def feature_unique(self):
        with record_function("## batch i+4 feature unique ##"):
            self.wait_sparse_data_dist(self.contexts[FIFTH_CONTEXT_TASK_POS])
            if self.pipelined_modules:
                self._set_module_context(self.contexts[FIFTH_CONTEXT_TASK_POS])
                self.set_sharding_plan(self.contexts[FIFTH_CONTEXT_TASK_POS])
            for module in self.pipelined_modules:
                module.forward.compute_feature_unique(self._data_dist_stream)
                self.compute_unique_sync_event.record(self._data_dist_stream)

    def compute_sparse_loss(self, context: TrainPipelineContext):
        sparse_loss = 0
        for module_name in self.pipeline_module_names:
            keyed_tensor = context.compute_and_output_result[module_name].values()
            detach_embedding_grad = context.detach_embedding_grad[module_name]
            sparse_loss += torch.dot(
                keyed_tensor.flatten(),
                detach_embedding_grad.flatten()
            )
        return sparse_loss
    
    def free_embedding(self, context: TrainPipelineContext):
        for module_name in self.pipeline_module_names:
            context.detach_embedding_grad[module_name] = None
        for i, _ in enumerate(self.pipelined_modules):
            context.detach_embedding[self.pipeline_module_names[i]] = None

    def sparse_backward(self, context: TrainPipelineContext, stream: torch.cuda.Stream):
        with record_function("## batch i sparse_backward ##"):
            self.backward_sync_event.wait(stream)
            sparse_loss = self.compute_sparse_loss(context)
            sparse_loss.record_stream(stream)
            sparse_loss.backward(retain_graph=True)
            self.free_embedding(context)

    def dense_forward(self, context: TrainPipelineContext, batch: Any):
        with record_function("## batch i+1 dense_forward ##"):
            self._set_module_context(context)
            dense_loss = self._model(batch)
            return dense_loss
        
    def dense_backward(self, dense_loss: torch.Tensor):
        with record_function("## batch i+1 dense_backward ##"):
            output = torch.sum(dense_loss)
            output.backward()
            self.store_embedding_grads(self.contexts[SECOND_CONTEXT_TASK_POS])
            self.backward_sync_event.record(self._dense_stream)
            return output

    def check_pipeline_stop(self):
        if len(self.batches) < SECOND_CONTEXT_TASK_POS + 1:
            self.clear_prefetched_batches()
            raise StopIteration

    def set_sharding_plan(self, context: TrainPipelineContext):
        context.sharding_plan = self.sharding_plan

    def get_detach_embedding(self, context: TrainPipelineContext):
        with record_function("## get_detach_embedding ##"):
            for module_name in self.pipeline_module_names:
                original_keyed_tensor = context.compute_and_output_result[module_name]
                detached_tensor = original_keyed_tensor.values().detach().requires_grad_(True)
                detached_keyed_tensor = KeyedTensor(
                    keys=original_keyed_tensor.keys(),
                    values=detached_tensor,
                    length_per_key=original_keyed_tensor.length_per_key()
                )
    
                context.detach_embedding[module_name] = detached_keyed_tensor

    def store_embedding_grads(self, context: TrainPipelineContext):
        for module_name in self.pipeline_module_names:
            if context.detach_embedding[module_name].values().grad is not None:
                detach_embedding_grad = context.detach_embedding[module_name].values().grad
                detach_embedding_grad.record_stream(self._dense_stream)
            context.detach_embedding_grad[module_name] = detach_embedding_grad

    def check_multistreamable(self, cur_tensor: torch.Tensor):
        if not isinstance(cur_tensor, (torch.Tensor, Multistreamable)):
            raise ValueError("Output of pipelined module must be Tensor or Multistreamable")

    def get_embedding(self, context: TrainPipelineContext, batch: Any, stream: torch.cuda.Stream):
        with record_function("## get_embedding ##"):
            # self.compute_unique_sync_event.wait(stream)
            if self.pipelined_modules:
                self._set_module_context(context)
            for i, module in enumerate(self.pipelined_modules):
                context.compute_and_output_result[self.pipeline_module_names[i]] = module.forward.output(stream)
                tensors = context.compute_and_output_result[self.pipeline_module_names[i]].values()
                self.check_multistreamable(tensors)
                tensors.record_stream(stream)

    def clear_prefetched_batches(self):
        self._i_batch = 0
        self.batches.clear()
        self.contexts.clear()
        self._optimizer.zero_grad()
        self._embedding_stream = (
            (torch.get_device_module(self._device).Stream(priority=-1))
            if self._device.type in ["cuda", "mtia", "npu"]
            else None
        )
        self._dense_stream = (
            (torch.get_device_module(self._device).Stream(priority=-1))
            if self._device.type in ["cuda", "mtia", "npu"]
            else None
        )
        self._data_dist_stream = (
            (torch.get_device_module(self._device).Stream(priority=-1))
            if self._device.type in ["cuda", "mtia", "npu"]
            else None
        )
        self.backward_sync_event = torch.npu.Event(enable_timing=False)
        self.forward_sync_event = torch.npu.Event(enable_timing=False)
        self.compute_unique_sync_event = torch.npu.Event(enable_timing=False)
