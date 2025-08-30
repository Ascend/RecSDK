#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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
import abc
from typing import List, Dict, Set, Tuple, Union

import pandas as pd
import tensorflow as tf
from tensorflow import Operation, Tensor, SparseTensor, Graph, variant, resource
from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter

from rec_sdk_common.log import logger
from rec_sdk_common.validator.validator import ClassValidator, para_checker_decorator
from mx_rec.graph import utils
from mx_rec.constants.constants import (
    ASCAnchorAttr,
    ASCEND_TIMESTAMP,
    MAX_WHILE_SIZE,
    ASCEND_SPARSE_LOOKUP_ENTRANCE,
    ORPHAN_LOOKUP_KEY_PREFIX
)
from mx_rec.graph.constants import DeprecatedOp, AnchorDatasetOp, AnchorIteratorOp
from mx_rec.core.emb.base_sparse_embedding import BaseSparseEmbedding


class NoGradSubgraphSlicer(metaclass=abc.ABCMeta):
    _SLICED_OP_NAME_PREFIX = "sliced"

    _SLICING_SUMMARY_NAME = "slicing_summary.csv"
    _UNSLICED_FULL_GRAPH_NAME = "unsliced_full_graph.pbtxt"
    _SLICED_SUB_GRAPH_NAME = "sliced_sub_graph.pbtxt"
    _SLICED_FULL_GRAPH_NAME = "sliced_full_graph.pbtxt"

    _INVALID_STR_IN_OP_TYPE = ("Dataset", "Summary")
    _INVALID_STR_IN_OP_NAME = ("save", "report_", "loss")
    _INVALID_CONSUMER_OP_TYPE = ("Assign", "SaveV2")

    _VALID_TENSOR_CLASS = (Tensor, SparseTensor)
    _INVALID_TENSOR_DTYPE = (variant, resource)

    def __init__(self, full_graph: Graph = None, info_dir: str = "slicing") -> None:
        if not full_graph:
            full_graph = tf.compat.v1.get_default_graph()
        self._full_graph = full_graph

        if not os.path.exists(info_dir):
            os.makedirs(info_dir)
        self._info_dir = info_dir

    @staticmethod
    def _find_min_dep_ops(
        tgt_ops: Set[Operation],
    ) -> Set[Operation]:
        logger.debug("Search from base nodes: %s.", tgt_ops)
        base_ops = tgt_ops.copy()
        visited_ops = base_ops

        loop_cnt = 0
        while base_ops:
            loop_cnt += 1
            if loop_cnt > MAX_WHILE_SIZE:
                raise RuntimeError(f"maximum loop times exceed limit: {MAX_WHILE_SIZE}.")

            parent_ops = set()
            for base_node in base_ops:
                if len(base_node.control_inputs) != 0:
                    raise ValueError("control dependencies are not supported.")

                parent_ops.update(
                    tensor_in.op
                    for tensor_in in base_node.inputs
                    if tensor_in.op.type != AnchorIteratorOp.ITERATOR_GET_NEXT.value
                )

            new_ops = parent_ops - visited_ops
            base_ops = parent_ops
            visited_ops.update(new_ops)

        logger.debug("Found minimum dependency graph nodes: %s.", visited_ops)
        return visited_ops

    @staticmethod
    def _validate_op(op: Operation) -> bool:
        op_type = op.type
        op_name = op.name
        op_inputs = op.inputs
        op_outputs = op.outputs

        for s in NoGradSubgraphSlicer._INVALID_STR_IN_OP_TYPE:
            if s in op_type:
                logger.warning("Invalid operation type: %s which contains str: %s.", op_type, s)
                return False
        for s in NoGradSubgraphSlicer._INVALID_STR_IN_OP_NAME:
            if s in op_name:
                logger.warning("Invalid operation name: %s which contains str: %s.", op_name, s)
                return False
        for t in op_inputs:
            if t.dtype in NoGradSubgraphSlicer._INVALID_TENSOR_DTYPE:
                logger.warning("Invalid operation input tensor of operation: %s whose type is %s.", t, t.dtype)
                return False
        for t in op_outputs:
            if t.dtype in NoGradSubgraphSlicer._INVALID_TENSOR_DTYPE:
                logger.warning("Invalid operation output tensor of operation: %s whose type is %s.", t, t.dtype)
                return False

        return True

    @staticmethod
    def _update_subgraph_in(
        base_ops: Operation,
        input_to_edge_ops: Dict[Operation, Set[Operation]],
        sub_graph_ops: Set[Operation],
    ) -> None:
        for input_tensor in base_ops.inputs:
            input_node = input_tensor.op
            if input_node not in sub_graph_ops:
                res = input_to_edge_ops.get(input_node, set())
                res.add(base_ops)
                input_to_edge_ops[input_node] = res

    @staticmethod
    def _update_subgraph_out(
        base_ops: Operation,
        out_op_to_edge_ops: Dict[Operation, Set[Operation]],
        sub_graph_ops: Set[Operation],
    ) -> None:
        for output_tensor in base_ops.outputs:
            for output_consumer in output_tensor.consumers():
                if output_consumer not in sub_graph_ops:
                    res = out_op_to_edge_ops.get(output_consumer, set())
                    res.add(base_ops)
                    out_op_to_edge_ops[output_consumer] = res


    @staticmethod
    def _topo_sort_sliced_ops(sliced_ops: Set[Operation]) -> List[Operation]:
        topo_subgraph_list = []
        topo_subgraph_set = set()
        start_nodes = set()
        [start_nodes.add(x) for x in sliced_ops]
        logger.info("Got topo_subgraph start nodes: %s", start_nodes)

        def topo_sort_helper(curr_op, output_list, output_set):
            if not isinstance(curr_op, Operation):
                raise RuntimeError(f"topo_subgraph_dfs input should be node(aka. tf.Operator). {curr_op}")
            curr_inputs = curr_op.inputs
            logger.debug("Got topo_dfs: %s <- %s", curr_op.name, [x.name for x in curr_inputs])
            current_control_inputs = curr_op.control_inputs
            if len(current_control_inputs) > 0:
                raise RuntimeError(
                    f"control input are not supported: {curr_op.name}, control_inputs: {current_control_inputs}"
                )
            if curr_op in output_set:
                return
            output_set.add(curr_op)
            for tensor in curr_inputs:
                node = tensor.op
                if node.type != AnchorIteratorOp.ITERATOR_GET_NEXT.value and node not in output_set:
                    topo_sort_helper(node, output_list, output_set)
            output_list.append(curr_op)

        [topo_sort_helper(x, topo_subgraph_list, topo_subgraph_set) for x in start_nodes]
        if len(topo_subgraph_list) != len(topo_subgraph_set):
            raise RuntimeError(f"got duplicated topo node: {sorted(topo_subgraph_list, key=lambda x: x.name)}.")
        logger.info("Got topo_subgraph: %s", topo_subgraph_list)
        return topo_subgraph_list

    @staticmethod
    def _get_mapping_for_subgraph_in(
        from_op: Operation,
        to_ops: Set[Operation],
        tensor_mapping: Union[Dict[Tensor, Tensor], Dict[SparseTensor, SparseTensor]],
    ) -> None:
        if from_op.type != AnchorIteratorOp.ITERATOR_GET_NEXT.value:
            raise RuntimeError(f"expect IteratorGetNext for input tensor of subgraph, but got {from_op}")
        for node in to_ops:
            for each_tensor in node.inputs:
                if each_tensor.op.type != AnchorIteratorOp.ITERATOR_GET_NEXT.value:
                    continue
                old_tensor_name = each_tensor.name
                x_index = int(old_tensor_name.split(":")[-1])
                g = tf.compat.v1.get_default_graph()
                arg_tensor = g.get_tensor_by_name("args_%d:0" % x_index)
                tensor_mapping[each_tensor] = arg_tensor

    @staticmethod
    def _get_mapping_for_subgraph(
        old_op: Operation,
        node_mapping: Dict[Operation, Operation],
        tensor_mapping: Dict[Tensor, Tensor],
    ) -> None:
        logger.debug("old operation name: %s\nold operation inputs: %s\n", old_op.name, [x for x in old_op.inputs])

        for each_tensor in old_op.inputs:
            if each_tensor not in tensor_mapping:
                raise RuntimeError(
                    f"each_tensor(input) {each_tensor} need by {old_op.name} not in tensor_mapping.{tensor_mapping}"
                )
        new_inputs = NoGradSubgraphSlicer._get_mapped_tensor(tensor_mapping, old_op.inputs)

        node_def = old_op.node_def
        node_def.name = "{}/{}".format(NoGradSubgraphSlicer._SLICED_OP_NAME_PREFIX, node_def.name)
        new_node = tf.Operation(node_def=node_def, g=tf.compat.v1.get_default_graph(), inputs=new_inputs)

        node_mapping[old_op] = new_node
        for old_out_tensor, new_out_tensor in zip(old_op.outputs, new_node.outputs):
            tensor_mapping[old_out_tensor] = new_out_tensor

    @staticmethod
    def _get_mapped_tensor(tensor2tensor: Dict[Tensor, Tensor], keys: List[Tensor]) -> List[Tensor]:
        tensors = []
        for k in keys:
            if k not in tensor2tensor:
                raise KeyError(f"failed to find key tensor: {k} from tensor map: {tensor2tensor}.")
            tensors.append(tensor2tensor[k])
        return tensors

    @staticmethod
    def _sort_sliced_graph_outputs(subgraph_out: Dict[Operation, Set[Operation]]) -> List[Tensor]:
        extra_outputs = []
        sorted_outputs = sorted(subgraph_out.items(), key=lambda x: x[0].name)
        for outside_op, edge_ops in sorted_outputs:
            outside_op_inputs = set(outside_op.inputs)
            for edge_op in edge_ops:
                NoGradSubgraphSlicer._add_sorted_additional_tensors(extra_outputs, outside_op_inputs, edge_op)
        return extra_outputs

    @staticmethod
    def _add_sorted_additional_tensors(extra_outputs, outside_op_inputs, edge_op) -> None:
        for each_tensor in sorted(edge_op.outputs, key=lambda x: x.name):
            if each_tensor not in outside_op_inputs:
                continue
            if each_tensor in extra_outputs:
                continue
            extra_outputs.append(each_tensor)

    @staticmethod
    def _get_tensor_consumers(tensor: Tensor) -> List[Operation]:
        if not isinstance(tensor, NoGradSubgraphSlicer._VALID_TENSOR_CLASS):
            raise RuntimeError(f"expected 'tf.Tensor' or 'tf.SparseTensor', but got: {tensor}")

        graph = tensor.graph
        consumers = []
        consumer_names = [op.name for op in tensor.consumers()]

        with graph._lock:
            for name in consumer_names:
                if name not in graph._nodes_by_name:  # ignore deleted node
                    continue
                consumers.append(graph._nodes_by_name[name])

        return consumers

    @abc.abstractmethod
    def summarize(self) -> None:
        pass

    @abc.abstractmethod
    def slice(self) -> None:
        pass

    def _slice_ops(self, sliceable_ops: Set[Operation], is_training: bool) -> None:
        """Slice the minimum dependency graph of given operation set.

        Args:
            sliceable_ops (Set[Operation]): The operation set that can be sliced.
            is_training (bool): Whether the slicing is for training graph or not.
        """

        sliced_ops = self._find_min_dep_ops(sliceable_ops)
        in_op_to_edge_ops, out_op_to_edge_ops = self._find_subgraph_in_and_out(sliced_ops)

        old_get_next = self._find_old_get_next(sliceable_ops)
        old_dataset = self._find_old_dataset(old_get_next, is_training)

        new_dataset = self._make_new_dataset(old_dataset, sliced_ops, in_op_to_edge_ops, out_op_to_edge_ops)
        new_dataset = new_dataset.prefetch(0)

        new_get_next = self._make_new_get_next(old_get_next, new_dataset)
        self._replace_get_next(old_get_next, new_get_next, out_op_to_edge_ops, sliced_ops)

    def _make_new_dataset(
        self,
        old_dataset: DatasetV1Adapter,
        sliced_ops: Set[Operation],
        in_op_to_edge_ops: Dict[Operation, Set[Operation]],
        out_op_to_edge_ops: Dict[Operation, Set[Operation]],
    ) -> DatasetV1Adapter:
        """Make a new dataset which clones the sliced subgraph by mapfunc.

        Args:
            old_dataset: The old dataset that needs to be mapped.
            sliced_ops: The operation set that has been sliced.
            in_op_to_edge_ops: The input relationship of sliced subgraph.
            out_op_to_edge_ops: The output relationship of sliced subgraph.

        Returns:
            DatasetV1Adapter: The new dataset that has cloned the sliced subgraph.
        """

        def slice_map_func(*batch):  # pragma: no cover
            logger.debug("The layout of old batch: %s.", batch)

            funcgraph = tf.compat.v1.get_default_graph()
            flatten_batch = tf.nest.flatten(batch)

            for t in flatten_batch:
                if isinstance(t, NoGradSubgraphSlicer._VALID_TENSOR_CLASS):
                    continue
                raise RuntimeError(f"expected 'tf.Tensor' or 'tf.SparseTensor' in batch, but got %s.", t)

            new_batch = self._clone_subgraph_into_funcgraph(sliced_ops, in_op_to_edge_ops, out_op_to_edge_ops, batch)
            utils.export_pb_graph(
                file_name=NoGradSubgraphSlicer._SLICED_SUB_GRAPH_NAME,
                dump_graph=True,
                graph_def=funcgraph.as_graph_def(),
                export_path=self._info_dir,
            )

            return new_batch

        return old_dataset.map(slice_map_func)

    def _find_subgraph_in_and_out(
        self,
        sub_graph_ops: Set[Operation],
    ) -> Tuple[Dict[Operation, Set[Operation]], Dict[Operation, Set[Operation]]]:
        """Find the input and output relationship of sliced subgraph.

        Args:
            sub_graph_ops: The operation set that has been sliced.

        Returns:
            in_op_to_edge_ops: The input relationship of sliced subgraph.
            out_op_to_edge_ops: The output relationship of sliced subgraph.
        """

        in_op_to_edge_ops = dict()
        out_op_to_edge_ops = dict()

        for base_node in sub_graph_ops:
            self._update_subgraph_in(base_node, in_op_to_edge_ops, sub_graph_ops)
            self._update_subgraph_out(base_node, out_op_to_edge_ops, sub_graph_ops)

        logger.info("Got input relationship of extracted subgraph: %s", in_op_to_edge_ops)
        logger.info("Got output relationship of extracted subgraph: %s", out_op_to_edge_ops)
        return in_op_to_edge_ops, out_op_to_edge_ops

    def _find_old_get_next(self, sliceable_ops: Set[Operation]) -> Operation:
        """Find the old 'IteratorGetNext' operation.

        Args:
            sliceable_ops: The operation set that can be sliced.

        Returns:
            old_get_next: The old 'IteratorGetNext' operation.
        """

        old_get_next = utils.upward_bfs_op(sliceable_ops, AnchorIteratorOp.ITERATOR_GET_NEXT.value)

        self._full_graph.add_to_collection(DeprecatedOp.DEPRECATED_ITERATOR_GET_NEXT, old_get_next)
        logger.info("Old 'IteratorGetNext' operation has been deprecated now.")

        return old_get_next

    def _find_old_dataset(self, get_next: Operation, is_training: bool) -> DatasetV1Adapter:
        """Find the old dataset that needs to be mapped.

        Due to the different iterator types, the search method is different.
        1. If the iterator type is 'MakeIterator', this func will exec upward bfs search through get_next.
        2. If the iterator type is 'OneShotIterator', this func will fetch all operation in 'self._full_graph', then
        filter out the 'PrefetchDataset' operation. This diff is caused by the isolation of 'OneShotIterator' and the
        'PrefetchDataset'.

        Args:
            get_next: The old 'IteratorGetNext' operation.
            is_training: Whether the slicing is for training graph or not.

        Returns:
            old_dataset: The old dataset that needs to be mapped.
        """

        tgt_trans_dataset = None
        try:
            tgt_trans_dataset = utils.find_trans_dataset(self._full_graph, get_next)
        except (ValueError, TypeError, RuntimeError) as err:
            trans_datasets = [
                op for op in self._full_graph.get_operations() if AnchorDatasetOp.PREFETCH_DATASET.value in op.name
            ]
            trans_datasets = list(
                filter(
                    lambda op: op not in tf.compat.v1.get_collection(DeprecatedOp.DEPRECATED_PREFETCH_DATASET),
                    trans_datasets,
                )
            )
            sorted_datasets = sorted(trans_datasets, key=lambda op: op.name)

            if len(trans_datasets) == 1:
                tgt_trans_dataset = sorted_datasets[0]
            elif is_training and len(sorted_datasets) == 2:
                tgt_trans_dataset = sorted_datasets[0]
            elif not is_training and len(sorted_datasets) == 2:
                tgt_trans_dataset = sorted_datasets[0]
            else:
                raise RuntimeError(f"target transformation dataset not found, got datasets: {trans_datasets}.") from err
        except Exception as err:
            raise RuntimeError(f"the dataset was not found, the error is `{err}`.") from err

        if not tgt_trans_dataset.outputs:
            raise ValueError(f"the length of the outputs of target op `{tgt_trans_dataset}` is 0.")
        logger.info("Find target op `%s`, and output is `%s`.", tgt_trans_dataset.name, tgt_trans_dataset.outputs)

        # WARN: Couple with modifier module, global collection used for filtering deprecated prefetch dataset.
        self._full_graph.add_to_collection(DeprecatedOp.DEPRECATED_PREFETCH_DATASET, tgt_trans_dataset)
        old_dataset = utils.find_target_instance_dataset(self._full_graph, tgt_trans_dataset.outputs[0])

        return old_dataset

    def _clone_subgraph_into_funcgraph(
        self,
        sliced_ops: Set[Operation],
        in_op_to_edge_ops: Set[Operation],
        out_op_to_edge_ops: Set[Operation],
        batch: Tuple[Dict[str, Union[Tensor, SparseTensor, Dict]]],
    ) -> Dict[str, Union[Tensor, SparseTensor, Dict]]:
        """Clone the sliced subgraph into a new funcgraph.

        Args:
            sliced_ops: The operation set that has been sliced.
            in_op_to_edge_ops: The input relationship of sliced subgraph.
            out_op_to_edge_ops: The output relationship of sliced subgraph.
            batch: The original batch layout of old dataset.

        Returns:
            new_batch: The new batch layout of new dataset.
        """

        topo_subgraph_list = self._topo_sort_sliced_ops(sliced_ops)

        node_mapping = {}  # subgraph-node -> funcgraph-node
        tensor_mapping = {}  # subgraph-tensor -> funcgraph-tensor
        for in_op, edge_ops in in_op_to_edge_ops.items():
            self._get_mapping_for_subgraph_in(in_op, edge_ops, tensor_mapping)
        for old_op in topo_subgraph_list:
            self._get_mapping_for_subgraph(old_op, node_mapping, tensor_mapping)

        logger.info("Got node_mapping: %s", node_mapping)
        logger.info("Got tensor_mapping: %s", tensor_mapping)

        ordered_output_tensors = self._sort_sliced_graph_outputs(out_op_to_edge_ops)
        extra_output_tensor = self._get_mapped_tensor(tensor_mapping, ordered_output_tensors)

        if not isinstance(batch, tuple):
            batch = (batch,)

        new_batch = batch[0]
        for tensor in extra_output_tensor:
            next_last_key = f"{sorted(new_batch)[-1]}_"
            new_batch[next_last_key] = tensor

        logger.debug("Got new batch layout: %s.", new_batch)
        return new_batch

    def _make_new_get_next(
        self,
        old_get_next: Operation,
        new_dataset: DatasetV1Adapter,
    ) -> Operation:
        """Make new 'IteratorGetNext' operation.

        1. This func will automatically detect the iterator type of the old dataset, and then make 'IteratorGetNext'
        from the corresponding iterator.
        2. Only 'MakeIterator' and 'OneShotIterator' are available now.

        Args:
            old_get_next: The old 'IteratorGetNext' operation.
            new_dataset: The new dataset which contains sliced subgraph and corresponding additional outputs.

        Returns:
            new_get_next: The new 'IteratorGetNext' operation.
        """

        if not old_get_next.outputs:
            raise RuntimeError("no available tensor in the dataset. Please check the dataset and data processing.")

        iter_type = None
        if old_get_next.inputs:
            iter_type = old_get_next.inputs[0].op.type
        if iter_type == AnchorIteratorOp.ITERATOR_V2.value:
            iter_type = utils.find_make_iterator_op(self._full_graph, old_get_next.outputs[0]).type
        if iter_type not in (AnchorIteratorOp.MAKE_ITERATOR.value, AnchorIteratorOp.ONE_SHOT_ITERATOR.value):
            raise RuntimeError(
                f"only iterators `MakeIterator` and `OneShotIterator` are supported in `graph modify` mode, "
                f"but the current iterator is `{iter_type}`."
            )
        logger.info("The iterator type of old dataset is %s.", iter_type)

        if iter_type == AnchorIteratorOp.MAKE_ITERATOR.value:
            new_iterator = tf.compat.v1.data.make_initializable_iterator(new_dataset)
        else:
            new_iterator = tf.compat.v1.data.make_one_shot_iterator(new_dataset)
        logger.info("Got new iterator: %s from dataset %s.", new_iterator, new_dataset)

        new_batch_name = "{}/{}".format(
            NoGradSubgraphSlicer._SLICED_OP_NAME_PREFIX, AnchorIteratorOp.ITERATOR_GET_NEXT.value
        )
        new_batch = new_iterator.get_next(name=new_batch_name)

        # WARN: Couple with user model, this collection has been addded manually.
        if "timestamp" in new_batch.keys():
            tf.compat.v1.add_to_collection(ASCEND_TIMESTAMP, new_batch["timestamp"])

        try:
            new_batch_tensor = new_batch
            while not isinstance(new_batch_tensor, NoGradSubgraphSlicer._VALID_TENSOR_CLASS):
                if isinstance(new_batch_tensor, tuple):
                    new_batch_tensor = new_batch_tensor[0]
                elif isinstance(new_batch_tensor, dict):
                    new_batch_tensor = list(new_batch_tensor.values())
                elif isinstance(new_batch_tensor, list):
                    new_batch_tensor = new_batch_tensor[0]
                elif isinstance(new_batch_tensor, NoGradSubgraphSlicer._VALID_TENSOR_CLASS):
                    break
                else:
                    raise RuntimeError(f"batch value {new_batch_tensor} of {type(new_batch_tensor)} is not supported.")
        except IndexError as err:
            raise IndexError("cannot find a tensor from given batch.") from err

        new_get_next = utils.upward_bfs_op(new_batch_tensor.op, AnchorIteratorOp.ITERATOR_GET_NEXT.value)

        logger.info("Got old_new_get_next: %s.", new_get_next)
        return new_get_next

    def _replace_get_next(
        self,
        old_get_next: Operation,
        new_get_next: Operation,
        out_op_to_edge_ops: Dict[Operation, Set[Operation]],
        sliced_ops: Set[Operation],
    ) -> None:
        """Replace the old 'IteratorGetNext' operation with the new one.

        1. This func will update the consumer of the old 'IteratorGetNext' operation to the new one.
        2. This func will update the consumer of the output tensors of the sliced subgraph to the new one.

        Args:
            old_get_next: The old 'IteratorGetNext' operation.
            new_get_next: The new 'IteratorGetNext' operation.
            out_op_to_edge_ops: The output relationship of sliced subgraph.
            sliced_ops: The operation set that has been sliced.
        """

        for t in old_get_next.outputs:
            self._update_old_get_next_consumer(t, new_get_next, sliced_ops)

        next_offset = len(old_get_next.outputs) - 1
        sorted_outputs = self._sort_sliced_graph_outputs(out_op_to_edge_ops)

        for t in sorted_outputs:
            next_offset += 1
            self._update_sliced_graph_consumer(t, new_get_next, next_offset)

    def _update_old_get_next_consumer(
        self, old_get_next_output: Tensor, new_get_next: Operation, sliced_ops: Set[Operation]
    ) -> None:
        """Update the consumer of the old 'IteratorGetNext' operation to the new one.

        Args:
            old_get_next_output: The output tensor of the old 'IteratorGetNext' operation.
            new_get_next: The new 'IteratorGetNext' operation.
            sliced_ops: The operation set that has been sliced.
        """

        old_tensor_name = old_get_next_output.name
        output_index = old_tensor_name.split(":")[-1]
        new_tensor_name = f"{new_get_next.name}:{output_index}"
        new_tensor = self._full_graph.get_tensor_by_name(new_tensor_name)

        old_tensor_consumers = self._get_tensor_consumers(old_get_next_output)
        for consumer in old_tensor_consumers:
            if consumer in sliced_ops:
                logger.debug("Ignore consumer: %s in sliced operations.", consumer.name)
                continue
            for i, t in enumerate(consumer.inputs):
                if t != old_get_next_output:
                    logger.debug(
                        "Ignore input %s of consumer %s, cause it not output of 'IteratorGetNext'.",
                        t.name,
                        consumer.name,
                    )
                    continue
                consumer._update_input(i, new_tensor)
                logger.debug(
                    "Succeed replace old input %s of consumer %s to new input %s.",
                    old_tensor_name,
                    consumer.name,
                    new_tensor,
                )

    def _update_sliced_graph_consumer(
        self, sliced_graph_output: Tensor, new_get_next: Operation, next_offset: int
    ) -> None:
        """Update the consumer of the output tensors of the sliced subgraph to the new one.

        The outputs of the sliced subgraph are not the original outputs of 'IteratorGetNext'. Thus, next offset should
        trace the last index of outputs of new 'IteratorGetNext'.

        Args:
            sliced_graph_output: The output tensor of the sliced subgraph.
            new_get_next: The new 'IteratorGetNext' operation.
            next_offset: The last offset of the new 'IteratorGetNext' operation.
        """

        new_tensor_name = f"{new_get_next.name}:{next_offset}"
        new_tensor = self._full_graph.get_tensor_by_name(new_tensor_name)

        old_tensor_consumers = self._get_tensor_consumers(sliced_graph_output)
        for consumer in old_tensor_consumers:
            if consumer.type in NoGradSubgraphSlicer._INVALID_CONSUMER_OP_TYPE:
                logger.debug("Ignore invalid consumer: %s.", consumer.name)
                continue
            for i, t in enumerate(consumer.inputs):
                if t != sliced_graph_output:
                    logger.debug(
                        "Ignore input %s of consumer %s, cause it not output of sliced graph.",
                        t.name,
                        consumer.name,
                    )
                    continue
                consumer._update_input(i, new_tensor)
                logger.debug(
                    "Succeed replace old input %s of consumer %s to new input %s.",
                    sliced_graph_output,
                    consumer.name,
                    new_tensor,
                )


@para_checker_decorator(
    check_option_list=[
        ("op_types", ClassValidator, {"classes": (list,)}),
        ("full_graph", ClassValidator, {"classes": (Graph, type(None))}),
        ("info_dir", ClassValidator, {"classes": (str,)}),
    ]
)
class LookupSubgraphSlicer(NoGradSubgraphSlicer):
    def __init__(self, op_types: List[str], full_graph: Graph = None, info_dir: str = "lookup_slicing") -> None:
        """Initialize LookupSubgraphSlicer.
        Args:
            op_types: The list of operation types to be sliced in lookup subgraph.
            full_graph: The full graph to be sliced. If None, the default graph will be used.
            info_dir: The directory to save the slicing information. Defaults to "lookup_slicing".
        """
        super().__init__(full_graph, info_dir)
        if not op_types:
            raise ValueError("no slicing operation types specified!")
        self._op_types = set(op_types)

    def summarize(self) -> None:  # pragma: no cover
        all_tgt_ops = self._find_all_tgt_ops()
        (train_sliceable_tgt_ops, eval_sliceable_tgt_ops) = self._find_sliceable_tgt_ops()
        all_sliceable_tgt_ops = train_sliceable_tgt_ops | eval_sliceable_tgt_ops

        result = {"Operation Type": [], "Total Num": [], "Sliceable Num": [], "Sliceable Ratio": []}

        for op_type in self._op_types:
            tgt_ops = set(filter(lambda op: op.type == op_type, all_tgt_ops))
            sliceable_tgt_ops = set(filter(lambda op: op.type == op_type, all_sliceable_tgt_ops))

            total_num = len(tgt_ops)
            sliceable_num = len(sliceable_tgt_ops)

            try:
                sliceable_ratio = sliceable_num / total_num
            except ZeroDivisionError:
                logger.warning("No target operaiton types '%s' found in given graph.", self._op_types)

            result["Operation Type"].append(op_type)
            result["Total Num"].append(total_num)
            result["Sliceable Num"].append(sliceable_num)
            result["Sliceable Ratio"].append(sliceable_ratio)

        result_df = pd.DataFrame(data=result)
        file = "{}/{}".format(self._info_dir, NoGradSubgraphSlicer._SLICING_SUMMARY_NAME)
        result_df.to_csv(file, sep=",")

        logger.info("Summary of slicing:\n%s", result_df)

    def slice(self) -> None:
        utils.export_pb_graph(
            file_name=NoGradSubgraphSlicer._UNSLICED_FULL_GRAPH_NAME,
            dump_graph=True,
            graph_def=self._full_graph.as_graph_def(),
            export_path=self._info_dir,
        )

        (train_sliceable_ops, eval_sliceable_ops) = self._find_sliceable_tgt_ops()

        if train_sliceable_ops:
            logger.info("Start to slice training lookup subgraph.")
            self._slice_ops(train_sliceable_ops, is_training=True)

        if eval_sliceable_ops:
            logger.info("Start to slice evaluation lookup subgraph.")
            self._slice_ops(eval_sliceable_ops, is_training=False)

        utils.export_pb_graph(
            file_name=NoGradSubgraphSlicer._SLICED_FULL_GRAPH_NAME,
            dump_graph=True,
            graph_def=self._full_graph.as_graph_def(),
            export_path=self._info_dir,
        )

    def _find_all_tgt_ops(self) -> Set[Operation]:
        """Found all operations of specific types in full graph."""
        all_tgt_ops = set()
        all_ops = self._full_graph.get_operations()

        for op in all_ops:
            if op.type not in self._op_types:
                continue
            all_tgt_ops.add(op)

        return all_tgt_ops

    def _find_sliceable_tgt_ops(self) -> Tuple[Set[Operation], Set[Operation]]:
        """Found sliceable operations of given types in lookup subgraph."""

        # WARN: Couple with mx_rec::core::embedding module.
        lookup_keys = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_ENTRANCE)

        train_base_ops = set()
        eval_base_ops = set()
        for t in lookup_keys:
            if BaseSparseEmbedding.get_anchor_attribute(t, ASCAnchorAttr.IS_TRAINING):
                train_base_ops.add(t.op)
            else:
                eval_base_ops.add(t.op)

        def find_sliceable_ops(base_ops):
            min_dep_ops = self._find_min_dep_ops(base_ops)

            sliceable_ops = set()
            for op in min_dep_ops:
                if not self._validate_op(op):
                    continue
                if op.type not in self._op_types:
                    continue
                sliceable_ops.add(op)

            return sliceable_ops

        train_sliceable_ops = find_sliceable_ops(train_base_ops)
        eval_sliceable_ops = find_sliceable_ops(eval_base_ops)

        logger.debug("Found sliceable operations in training lookup subgraph: %s.", train_sliceable_ops)
        logger.debug("Found sliceable operations in evaluation lookup subgraph: %s.", eval_sliceable_ops)
        return (train_sliceable_ops, eval_sliceable_ops)


@para_checker_decorator(
    check_option_list=[
        ("full_graph", ClassValidator, {"classes": (Graph, type(None))}),
        ("info_dir", ClassValidator, {"classes": (str,)}),
    ]
)
class OrphanLookupKeySlicer(NoGradSubgraphSlicer):
    def __init__(self, full_graph: Graph = None, info_dir: str = "orphan_slicing") -> None:
        """Initialize OrphanLookupKeySlicer.
        Args:
            full_graph: The full graph to be sliced. If None, the default graph will be used.
            info_dir: The directory to save the slicing information. Defaults to "orphan_slicing".
        """
        super().__init__(full_graph, info_dir)

    def summarize(self) -> None:  # pragma: no cover
        (train_sliceable_ops, _) = self._find_sliceable_tgt_ops()

        if len(train_sliceable_ops) == 0:
            return

        result = {"Operation Type": [], "Operation Name": []}
        for op in train_sliceable_ops:
            result["Operation Type"].append(op.type)
            result["Operation Name"].append(op.name)

        result_df = pd.DataFrame(data=result)
        file = "{}/{}".format(self._info_dir, NoGradSubgraphSlicer._SLICING_SUMMARY_NAME)
        result_df.to_csv(file, sep=",")

        logger.info("Summary of slicing:\n%s", result_df)

    def slice(self) -> None:
        utils.export_pb_graph(
            file_name=NoGradSubgraphSlicer._UNSLICED_FULL_GRAPH_NAME,
            dump_graph=True,
            graph_def=self._full_graph.as_graph_def(),
            export_path=self._info_dir,
        )

        (train_sliceable_ops, eval_sliceable_ops) = self._find_sliceable_tgt_ops()

        if train_sliceable_ops:
            logger.info("Start to slice training lookup subgraph.")
            self._slice_ops(train_sliceable_ops, is_training=True)

        if eval_sliceable_ops:
            logger.info("Start to slice evaluation lookup subgraph.")
            self._slice_ops(eval_sliceable_ops, is_training=False)

        utils.export_pb_graph(
            file_name=NoGradSubgraphSlicer._SLICED_FULL_GRAPH_NAME,
            dump_graph=True,
            graph_def=self._full_graph.as_graph_def(),
            export_path=self._info_dir,
        )

    def _slice_ops(self, sliceable_ops: Set[Operation], is_training: bool) -> None:
        """Override the '_slice_ops' protected method of super class."""

        sliced_ops = self._find_min_dep_ops(sliceable_ops)
        in_op_to_edge_ops, out_op_to_edge_ops = self._find_subgraph_in_and_out(sliced_ops)

        all_get_nexts = [
            op for op in self._full_graph.get_operations() if op.type == AnchorIteratorOp.ITERATOR_GET_NEXT.value
        ]
        alive_get_nexts = list(
            filter(
                lambda op: op not in self._full_graph.get_collection(DeprecatedOp.DEPRECATED_ITERATOR_GET_NEXT),
                all_get_nexts,
            )
        )
        alive_get_nexts = sorted(alive_get_nexts, key=lambda op: op.name)

        old_get_next = None
        if len(alive_get_nexts) == 1:
            old_get_next = alive_get_nexts[0]
        else:
            old_get_next = alive_get_nexts[0] if is_training else alive_get_nexts[1]

        old_dataset = self._find_old_dataset(old_get_next, is_training)

        new_dataset = self._make_new_dataset(old_dataset, sliced_ops, in_op_to_edge_ops, out_op_to_edge_ops)
        new_dataset = new_dataset.prefetch(0)

        new_get_next = self._make_new_get_next(old_get_next, new_dataset)
        self._replace_get_next(old_get_next, new_get_next, out_op_to_edge_ops, sliced_ops)

    def _find_sliceable_tgt_ops(self) -> Tuple[Set[Operation], Set[Operation]]:
        """Found orhpan keys' additional identity operation in lookup subgraph."""

        # WARN: Couple with mx_rec::core::embedding module.
        lookup_keys = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_ENTRANCE)

        train_base_ops = set()
        eval_base_ops = set()
        for t in lookup_keys:
            if BaseSparseEmbedding.get_anchor_attribute(t, ASCAnchorAttr.IS_TRAINING):
                train_base_ops.add(t.op)
            else:
                eval_base_ops.add(t.op)

        def find_sliceable_ops(base_ops):
            min_dep_ops = self._find_min_dep_ops(base_ops)

            sliceable_ops = set()
            for op in min_dep_ops:
                if not self._validate_op(op):
                    continue
                if ORPHAN_LOOKUP_KEY_PREFIX not in op.name:
                    continue
                sliceable_ops.add(op)

            return sliceable_ops

        train_sliceable_ops = find_sliceable_ops(train_base_ops)
        eval_sliceable_ops = find_sliceable_ops(eval_base_ops)

        logger.debug("Found sliceable operations in training lookup subgraph: %s.", train_sliceable_ops)
        logger.debug("Found sliceable operations in evaluation lookup subgraph: %s.", eval_sliceable_ops)
        return (train_sliceable_ops, eval_sliceable_ops)
