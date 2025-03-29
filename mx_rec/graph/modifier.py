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

import dataclasses
import mxrec_pybind
import tensorflow as tf

from collections import defaultdict
from collections.abc import Callable
from tensorflow import Operation, Tensor, Graph
from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from typing import List, Dict, Tuple, DefaultDict, Union

from mx_rec.core.embedding_proxy import MergeableEmbeddingTableProxy
from mx_rec.graph import utils
from mx_rec.constants.constants import (
    ASCEND_CUTTING_POINT_INITIALIZER,
    ASCEND_SPARSE_LOOKUP_ENTRANCE,
    ASCAnchorAttr,
    ASCEND_TIMESTAMP,
    MAX_WHILE_SIZE,
    LIBREC_EOS_OPS_SO,
    TRAIN_CHANNEL_ID,
    EVAL_CHANNEL_ID,
)
from mx_rec.core.asc.feature_spec import FeatureSpec
from mx_rec.core.asc.helper import get_asc_insert_func
from mx_rec.core.asc.manager import start_asc_pipeline
from mx_rec.core.asc.swap_args import SwapArgs, SwapDataType
from mx_rec.core.asc.build_graph import SwapInfo
from mx_rec.core.emb.base_sparse_embedding import BaseSparseEmbedding
from mx_rec.graph.merge_lookup import do_merge_lookup
from mx_rec.graph.utils import check_and_force_list, export_pb_graph
from mx_rec.graph.constants import DeprecatedOp, AnchorDatasetOp, AnchorIteratorOp
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.util.log import logger
from mx_rec.util.ops import import_host_pipeline_ops
from mx_rec.util.perf import performance
from mx_rec.util.tf_version_adapter import npu_ops
from mx_rec.validator.validator import para_checker_decorator, ClassValidator
from mx_rec.util.communication.hccl_ops import get_rank_id, get_device_id
host_pipeline_ops = import_host_pipeline_ops()


class GraphModifierHook(tf.estimator.SessionRunHook):
    @para_checker_decorator(
        check_option_list=[
            ("dump_graph", ClassValidator, {"classes": (bool,)}),
            ("modify_graph", ClassValidator, {"classes": (bool,)}),
        ]
    )
    def __init__(self, dump_graph: bool = False, modify_graph: bool = True):
        self._dump_graph = dump_graph
        self._modify_graph = modify_graph
        self._iterator_type = None

        ConfigInitializer.get_instance().train_params_config.is_graph_modify_hook_running = True

    def begin(self):
        if self._modify_graph:
            modify_graph_and_start_emb_cache(dump_graph=self._dump_graph)
        else:
            start_asc_pipeline()

        self._iterator_type = ConfigInitializer.get_instance().train_params_config.iterator_type
        if self._modify_graph and self._iterator_type not in (
            AnchorIteratorOp.MAKE_ITERATOR.value,
            AnchorIteratorOp.ONE_SHOT_ITERATOR.value,
        ):
            raise ValueError("the value of iterator type should be like `MakeIterator` or `OneShotIterator`.")
        logger.debug("In GraphModifierHook, iterator type is `%s`.", self._iterator_type)

    def after_create_session(self, session, coord):
        if self._modify_graph and self._iterator_type == AnchorIteratorOp.MAKE_ITERATOR.value:
            session.run(tf.compat.v1.get_collection(ASCEND_CUTTING_POINT_INITIALIZER))


@dataclasses.dataclass
class _AnchorRecord:
    replacement_spec: DefaultDict[Tensor, List[Tuple[int, Operation]]]
    passing_tensors: List[Tensor]
    batch_tensor_indexs: List[int]
    sub_cutting_points: List[Tensor]
    sub_graph_def: GraphDef
    input_names: List[str]
    output_names: List[str]
    is_training: bool
    input_indexs: List[int] = None


class _GraphModifier:
    @para_checker_decorator(
        check_option_list=[
            ("dump_graph", ClassValidator, {"classes": (bool,)}),
            ("modify_graph", ClassValidator, {"classes": (bool,)}),
        ]
    )
    def __init__(self, full_graph: Graph = None, dump_graph: bool = False):
        if not full_graph:
            full_graph = tf.compat.v1.get_default_graph()
        self._full_graph = full_graph
        self._dump_graph = dump_graph

    @staticmethod
    def _get_preprocessing_map_func(
        graph_def: GraphDef,
        input_names: List[str],
        output_names: List[str],
        pipeline_input_indexes: List[int] = None,
    ) -> Callable:  # pragma: no cover
        input_names = check_and_force_list(input_names, str)
        output_names = check_and_force_list(output_names, str)
        pipeline_input_indexes = check_and_force_list(pipeline_input_indexes, int)

        def map_func(*args) -> tuple:
            batch = args
            if not isinstance(batch, tuple) or len(batch) == 0:
                raise ValueError(f"The dataset batch is invalid, and the batch is: {batch}.")
            logger.debug("In get_preprocessing_map_func, the parse batch is: %s.", batch)

            input_tensors = []
            graph = tf.compat.v1.get_default_graph()
            for index in pipeline_input_indexes:
                tensor = graph.get_tensor_by_name("args_%d:0" % index)
                input_tensors.append(tensor)

            # 以tf.import_graph_def()作为read emb key的输入，保证数据读取到传入lookup的ids过程中的特征处理关系能够保留在子图中。
            output_list = tf.import_graph_def(
                graph_def, input_map=dict(zip(input_names, input_tensors)), return_elements=output_names
            )

            output_batch = list(batch)
            output_batch.append(tuple(output_list))
            logger.debug("In get_preprocessing_map_func, the output batch is: %s.", output_batch)
            return tuple(output_batch)

        return map_func

    @performance("graph_modifier")
    def modify_graph_for_asc(self, prefetch: int = 10):
        cutting_point_list = self._full_graph.get_collection(ASCEND_SPARSE_LOOKUP_ENTRANCE)
        utils.check_cutting_points(cutting_point_list)
        if not cutting_point_list:
            logger.warning("Nothing to revise.")
            return

        export_pb_graph("old_graph.pbtxt", self._dump_graph, graph_def=self._full_graph.as_graph_def())
        get_next_op_map = self._generate_get_next_op_specs(cutting_point_list)
        logger.debug(
            "In modify_graph_for_asc function, get_next_op_map.len: %d, get_next_op_map.key: %s.",
            len(get_next_op_map),
            get_next_op_map.keys(),
        )

        for get_next_op, record in get_next_op_map.items():
            is_training = record.is_training

            # get source dataset
            src_dataset = self._get_src_dataset(get_next_op, is_training)

            # generate target dataset
            timestamp_index = _get_timestamp_index(self._full_graph, get_next_op, is_training)
            original_batch_tensor_count = _get_dataset_tensor_count(src_dataset)
            sub_cutting_points = record.sub_cutting_points
            input_index_list = _get_input_index_list(
                sub_cutting_points,
                record.replacement_spec,
                record.output_names,
                original_batch_tensor_count,
                timestamp_index=timestamp_index,
            )
            record.input_indexs = input_index_list

            with self._full_graph.as_default():
                tgt_dataset = self._get_tgt_dataset(src_dataset, sub_cutting_points, record, prefetch=prefetch)
                self._update_iterator_getnext(get_next_op, tgt_dataset, is_training, record)

            # In eval mode, backward is not required. In addition, compute gradients is not executed when
            # only eval is used. Therefore, `do_merge_lookup` needs to be invoked during modify graph.
            if not is_training:
                with self._full_graph.as_default():
                    do_merge_lookup(is_train=False)
                if "evaluate" in ConfigInitializer.get_instance().train_params_config.bool_gauge_set:
                    logger.debug("In estimator mode, eval re-creates graph each time, so the flag needs to be cleared.")
                    ConfigInitializer.get_instance().train_params_config.insert_merged_multi_lookup(is_training, False)
            # In training mode, `do_merge_lookup` should have been executed in compute gradients phase.
            if is_training and not ConfigInitializer.get_instance().train_params_config.get_merged_multi_lookup(True):
                raise RuntimeError(
                    "In training mode, `do_merge_lookup` should have been executed in compute gradients "
                    "phase. Please check whether compute gradients is performed."
                )

        self._modify_graph_for_ddr(get_next_op_map)

        logger.info("Graph has been revised.")
        export_pb_graph("new_graph.pbtxt", self._dump_graph, graph_def=self._full_graph.as_graph_def())

    def _modify_graph_for_ddr(self, get_next_op_map: Dict[Tensor, _AnchorRecord]):
        # 通过create_hash_optimizer创建optimizer_instance
        optimizer_instance = ConfigInitializer.get_instance().optimizer_config.optimizer_instance
        # Predict mode
        if optimizer_instance is None:
            slot_num = 0
        else:
            # DDR和扩容需要在获取优化器后重置ext
            change_ext_emb_size_by_opt(optimizer_instance)
            slot_num = optimizer_instance.slot_num

        for _, record in get_next_op_map.items():
            is_training = record.is_training
            channel_id = 0 if is_training else 1
            replace_anchor_for_ddr_ssd(self._full_graph, slot_num, channel_id)

    def _generate_get_next_op_specs(self, cutting_point_list: List[Tensor]) -> Dict[Tensor, _AnchorRecord]:
        get_next_op_map = defaultdict(dict)

        for input_tensor in cutting_point_list:
            get_next_op = utils.upward_bfs_op(input_tensor.op, AnchorIteratorOp.ITERATOR_GET_NEXT.value)
            if get_next_op not in get_next_op_map:
                logger.debug("find a new get_next_op named '%s'", get_next_op.name)

                replacement_specs = utils.record_ops_to_replace(self._full_graph, get_next_op)
                passing_tensors, batch_tensor_indexs, sub_cutting_points = _get_passing_tensor_list(
                    cutting_point_list, get_next_op
                )
                sub_graph_def, input_names, output_names = self._get_sub_graph(passing_tensors, sub_cutting_points)
                is_training = BaseSparseEmbedding.get_anchor_attribute(input_tensor, ASCAnchorAttr.IS_TRAINING)

                record = _AnchorRecord(
                    replacement_specs,
                    passing_tensors,
                    batch_tensor_indexs,
                    sub_cutting_points,
                    sub_graph_def,
                    input_names,
                    output_names,
                    is_training,
                )
                get_next_op_map[get_next_op] = record

                export_pb_graph(f"cut_graph_{get_next_op.name}.pbtxt", self._dump_graph, graph_def=sub_graph_def)

        return get_next_op_map

    def _get_sub_graph(
        self, input_tensors: List[Tensor], output_tensors: List[Tensor]
    ) -> Tuple[GraphDef, List[str], List[str]]:
        input_tensors = check_and_force_list(input_tensors, tf.Tensor)
        output_tensors = check_and_force_list(output_tensors, tf.Tensor)
        input_op_name_list = [tensor.op.name for tensor in input_tensors]
        output_op_name_list = [tensor.op.name for tensor in output_tensors]

        graph_def = self._full_graph.as_graph_def()
        cut_graph_input = tf.compat.v1.graph_util.extract_sub_graph(graph_def, input_op_name_list)
        cut_graph_output = tf.compat.v1.graph_util.extract_sub_graph(graph_def, output_op_name_list)

        node_list = []
        node_list_input = cut_graph_input.node
        node_list_output = cut_graph_output.node
        for node in node_list_output:
            if node not in node_list_input:
                node_list.append(node)

        sub_graph_def = tf.compat.v1.GraphDef()
        sub_graph_def.node.extend(node_list)

        input_name_list = [tensor.name for tensor in input_tensors]
        output_name_list = [tensor.name for tensor in output_tensors]

        return sub_graph_def, input_name_list, output_name_list

    def _get_src_dataset(self, get_next_op: Operation, is_training: bool) -> DatasetV1Adapter:
        """
        根据`IteratorGetNext`算子在计算图中找出原始dataset.

        Args:
            get_next_op: `IteratorGetNext`算子
            is_training: 当前是否为训练模式，训练模式为True，否则为False

        Returns: 原始数据集

        """

        try:
            target_op = utils.find_trans_dataset(self._full_graph, get_next_op)
        except (ValueError, TypeError, RuntimeError) as err:
            logger.warning("The dataset op was not found, the error is `%s`. Start to traverse the operations.", err)
            graph = self._full_graph
            dataset_op_list = [op for op in graph.get_operations() if AnchorDatasetOp.PREFETCH_DATASET.value in op.name]

            # WARN: Couple with NoGradSubGraphSlicer::_find_old_dataset.
            dataset_op_list = list(
                filter(
                    lambda op: op not in self._full_graph.get_collection(DeprecatedOp.DEPRECATED_PREFETCH_DATASET),
                    dataset_op_list,
                )
            )
            dataset_op_list = sorted(dataset_op_list, key=lambda op: op.name)

            logger.debug(
                "In get_src_dataset function, current mode(train: True, eval: False): %s, dataset_op_list: %s.",
                is_training,
                dataset_op_list,
            )

            if len(dataset_op_list) == 1:
                target_op = dataset_op_list[0]
            elif is_training and len(dataset_op_list) == 2:
                prefetch_dataset_op_list = sorted(dataset_op_list, key=lambda op: op.name)
                target_op = prefetch_dataset_op_list[0]
            elif not is_training and len(dataset_op_list) == 3:
                prefetch_dataset_op_list = sorted(dataset_op_list, key=lambda op: op.name)
                target_op = prefetch_dataset_op_list[1]
            else:
                raise RuntimeError(
                    f"'{AnchorDatasetOp.PREFETCH_DATASET.value}' not found, got transformation datasets: "
                    f"{dataset_op_list}."
                ) from err
        except Exception as err:
            raise RuntimeError(f"The dataset was not found, the error is `{err}`.") from err

        if not target_op.outputs:
            raise ValueError(f"The length of the outputs of target op `{target_op}` is 0.")
        logger.debug("Find target op `%s`, and output is `%s`.", target_op.name, target_op.outputs)
        src_dataset = utils.find_target_instance_dataset(self._full_graph, target_op.outputs[0])
        # The element spec of the dataset is used to restore the original batch.
        ConfigInitializer.get_instance().train_params_config.dataset_element_spec = src_dataset.element_spec
        return src_dataset

    def _get_tgt_dataset(
        self,
        src_dataset: DatasetV1Adapter,
        sub_cutting_point_list: List[Tensor],
        record: _AnchorRecord,
        prefetch: int = 10,
    ) -> DatasetV1Adapter:
        """
        根据原始数据集生成新的数据集实例.

        Args:
            src_dataset: 原始数据集实例
            sub_cutting_point_list: 打桩的lookup ids列表
            records: 记录被打桩ids对应输入/输出算子、子图关系等信息的字典
            dump_graph: 是否dump计算图，默认为False
            prefetch: dataset预取数据量，默认为10

        Returns: 新数据集实例

        """

        librec = import_host_pipeline_ops(LIBREC_EOS_OPS_SO)
        channel_id = ConfigInitializer.get_instance().train_params_config.get_training_mode_channel_id(
            record.is_training
        )
        # 在数据读取完时，通过EosDataset向acl数据通道发送end_of_sequence
        max_train_steps = ConfigInitializer.get_instance().max_steps
        max_eval_steps = ConfigInitializer.get_instance().eval_steps
        src_dataset = src_dataset.eos_map(librec, channel_id, max_train_steps, max_eval_steps)

        tgt_dataset = src_dataset.map(
            self._get_preprocessing_map_func(
                record.sub_graph_def,
                record.input_names,
                record.output_names,
                pipeline_input_indexes=record.batch_tensor_indexs,
            )
        )

        feature_numbers = [
            BaseSparseEmbedding.get_anchor_attribute(cutting_point, ASCAnchorAttr.FEATURE_SPEC).feat_cnt
            for cutting_point in sub_cutting_point_list
        ]
        table_names = [
            BaseSparseEmbedding.get_anchor_attribute(cutting_point, ASCAnchorAttr.FEATURE_SPEC).table_name
            for cutting_point in sub_cutting_point_list
        ]
        tgt_dataset = tgt_dataset.map(
            get_asc_insert_func(
                feature_numbers=feature_numbers,
                table_names=table_names,
                args_index_list=record.input_indexs,
                is_training=record.is_training,
                dump_graph=self._dump_graph,
            )
        )

        tgt_dataset = tgt_dataset.prefetch(prefetch)
        return tgt_dataset

    def _update_iterator_getnext(
        self, get_next_op: Operation, tgt_dataset: DatasetV1Adapter, is_training: bool, record: _AnchorRecord
    ) -> None:
        """
        用新数据集中的`IteratorGetNext`算子替换计算图中原始数据集的`IteratorGetNext`算子，即用新数据集的batch替换原始数据集的batch.

        Args:
            get_next_op: `IteratorGetNext`算子
            tgt_dataset: 新数据集
            is_training: 当前是否为训练模式，训练模式为True，否则为False
            records: 记录被打桩ids对应输入/输出算子、子图关系等信息的字典

        Returns: None

        """
        if not get_next_op.outputs:
            raise RuntimeError("there is no tensor in the dataset. Please check the dataset and data processing.")
        iterator_type = ""
        if get_next_op.outputs[0].op.inputs:
            iterator_type = get_next_op.outputs[0].op.inputs[0].op.type
        if iterator_type == "IteratorV2":
            iterator_type = utils.find_make_iterator_op(self._full_graph, get_next_op.outputs[0]).type
        if iterator_type not in (AnchorIteratorOp.MAKE_ITERATOR.value, AnchorIteratorOp.ONE_SHOT_ITERATOR.value):
            raise RuntimeError(
                f"Only iterators `MakeIterator` and `OneShotIterator` are supported in `graph modify` mode, "
                f"but the current iterator is `{iterator_type}`."
            )
        ConfigInitializer.get_instance().train_params_config.iterator_type = iterator_type
        logger.info("The iterator type of dataset is `%s`.", iterator_type)

        if iterator_type == AnchorIteratorOp.MAKE_ITERATOR.value:
            new_iterator = tgt_dataset.make_initializable_iterator()
            tf.compat.v1.add_to_collection(ASCEND_CUTTING_POINT_INITIALIZER, new_iterator.initializer)
            ConfigInitializer.get_instance().train_params_config.set_initializer(is_training, new_iterator.initializer)
        else:
            new_iterator = tgt_dataset.make_one_shot_iterator()
        new_batch = new_iterator.get_next()
        ConfigInitializer.get_instance().train_params_config.set_target_batch(is_training, new_batch)

        new_batch_tensor = _get_new_batch_tensor(new_batch)
        logger.debug("New dataset batch tensor is : %s.", new_batch_tensor)
        new_get_next_op_name = utils.upward_bfs_op(new_batch_tensor.op, AnchorIteratorOp.ITERATOR_GET_NEXT.value).name
        self._update_input_tensor_with_new_batch(record.replacement_spec, new_get_next_op_name, new_batch)

    def _update_input_tensor_with_new_batch(
        self,
        replacement_specs: DefaultDict[Tensor, List[Tuple[int, Operation]]],
        new_get_next_op_name: str,
        new_batch: Dict[str, Tensor],
    ) -> None:
        """
        用新batch中的IteratorGetNext替换计算图中老batch的IteratorGetNext.

        Args:
            replacement_specs: 记录待替换算子的dict，key为老batch的IteratorGetNext，value为以老batch作为输入的算子
            new_get_next_op_name: 新数据集的get_next算子名称
            new_batch: 新数据集的batch

        Returns: None

        """

        for old_tensor, item in replacement_specs.items():
            for idx, operator in item:
                old_tensor_name = old_tensor.name
                output_index = old_tensor_name.split(":")[-1]
                new_tensor_name = f"{new_get_next_op_name}:{output_index}"
                new_tensor = self._full_graph.get_tensor_by_name(new_tensor_name)
                try:
                    operator._update_input(idx, new_tensor)
                except InvalidArgumentError as err:
                    logger.info(
                        "The replacement specs keys (old batch) is: %s. \n\t\t The new batch is: %s.",
                        replacement_specs.keys(),
                        new_batch,
                    )
                    raise RuntimeError(
                        f"Cannot update edge, old tensor: {old_tensor}, new tensor: {new_tensor}."
                    ) from err


@para_checker_decorator(
    check_option_list=[
        ("full_graph", ClassValidator, {"classes": (Graph, type(None))}),
        ("dump_graph", ClassValidator, {"classes": (bool,)}),
    ]
)
def modify_graph_and_start_emb_cache(full_graph: Graph = None, dump_graph: bool = False):
    modifier = _GraphModifier(full_graph=full_graph, dump_graph=dump_graph)
    modifier.modify_graph_for_asc()
    MergeableEmbeddingTableProxy().reset()
    start_asc_pipeline()


def _get_input_index_list(
    cutting_point_list: List[Tensor],
    replacement_specs: DefaultDict[Tensor, List[Tuple[int, Operation]]],
    mapping_name_list: List[str],
    base_count: int,
    timestamp_index: int = None,
) -> List[int]:
    input_index_list = []
    for cutting_point in cutting_point_list:
        if cutting_point in replacement_specs:
            index = int(cutting_point.name.split(":")[1])

        elif cutting_point.name in mapping_name_list:
            index = base_count + mapping_name_list.index(cutting_point.name)

        else:
            raise ValueError(f"Cannot find a matching output for cutting point tensor named '{cutting_point.name}'.")
        input_index_list.append(index)
    if timestamp_index is not None:
        input_index_list = [timestamp_index] + input_index_list

    return input_index_list


def _get_passing_tensor_list(
    src_tensors: List[Tensor], target_op: Operation
) -> Tuple[List[Tensor], List[int], List[Tensor]]:
    def get_passing_tensors(src_tensor):
        passing_tensors = []
        tensor_list = [src_tensor]
        while_num = 0
        while tensor_list:
            while_num += 1
            if while_num > MAX_WHILE_SIZE:
                raise RuntimeError(
                    f"In get_passing_tensors function, the maximum cycle depth is greater " f"than {MAX_WHILE_SIZE}."
                )
            last_tensor = tensor_list.pop()
            if last_tensor.op is target_op:
                passing_tensors.append(last_tensor)
            else:
                tensor_list.extend(list(last_tensor.op.inputs))

        return passing_tensors

    src_tensors = check_and_force_list(src_tensors, Tensor)
    passing_tensor_list = []
    sub_src_tensors = []
    for tensor in src_tensors:
        passing_tensors = get_passing_tensors(tensor)
        for passing_tensor in passing_tensors:
            if passing_tensor not in passing_tensor_list:
                passing_tensor_list.append(passing_tensor)
        if len(passing_tensors) != 0:
            logger.info("passing_tensors: %s", passing_tensors)
            sub_src_tensors.append(tensor)
        else:
            logger.info("Cannot find passing tensor for given tensor '%s'.", tensor)

    output_index_list = [int(tensor.name.split(":")[1]) for tensor in passing_tensor_list]

    return passing_tensor_list, output_index_list, sub_src_tensors


def _get_dataset_tensor_count(dataset: DatasetV1Adapter) -> int:
    """
    获取数据集中batch的tensor数量.

    Args:
        dataset: 数据集实例

    Returns: 数据集batch中的tensor数量

    """

    src_element_spec = dataset.element_spec
    if not isinstance(src_element_spec, (list, tuple)):
        src_element_spec = [src_element_spec]
    src_sorted_keys = utils.make_sorted_key_to_tensor_list(src_element_spec, [])

    return len(src_sorted_keys)


def _get_timestamp_index(graph: Graph, get_next_op: Operation, is_training: bool) -> int:
    timestamp_tensor_list = graph.get_collection(ASCEND_TIMESTAMP)
    timestamp_index = None
    for timestamp in timestamp_tensor_list:
        if timestamp in get_next_op.outputs:
            timestamp_index = int(timestamp.name.split(":")[1])
            timestamp_feature_spec = ConfigInitializer.get_instance().feature_spec_config.get_feature_spec("timestamp")
            if timestamp_feature_spec is None:
                timestamp_feature_spec = FeatureSpec("timestamp", index_key=timestamp_index, is_timestamp=True)
                timestamp_feature_spec.include_timestamp(is_training)
                ConfigInitializer.get_instance().feature_spec_config.insert_feature_spec(
                    timestamp_feature_spec, is_training
                )
                break

            if timestamp_feature_spec.index_key != timestamp_index:
                raise ValueError(
                    f"Given timestamp_index, which is {timestamp_index}, does not match index "
                    f"key. Please double check."
                )
            timestamp_feature_spec.include_timestamp(is_training)
            break
    return timestamp_index


def change_ext_emb_size_by_opt(optimizer: tf.compat.v1.train.Optimizer):
    for _, table_instance in ConfigInitializer.get_instance().sparse_embed_config.table_instance_dict.items():
        # When dynamic expansion mode, ext_emb_size is set by optimizer
        if ConfigInitializer.get_instance().use_dynamic_expansion or not table_instance.is_hbm:
            table_instance.ext_emb_size = table_instance.emb_size * (1 + optimizer.slot_num)
            logger.info("ext_emb_size is reset to be %s in change_ext_emb_size_by_opt", table_instance.ext_emb_size)


def _get_variable_and_slot_list(each_var, slot_num, table_name, channel_id):
    variable_and_slot_list = [each_var]
    if slot_num == 0:
        return variable_and_slot_list

    # 通过apply_gradients创建optimizer
    is_training = True if channel_id == TRAIN_CHANNEL_ID else False
    optimizer = ConfigInitializer.get_instance().optimizer_config.get_optimizer_by_table_name(
        table_name, is_training=is_training
    )
    if optimizer is None and channel_id == TRAIN_CHANNEL_ID:
        raise RuntimeError(
            "In training mode, table_instance should have been set_optimizer_for_table "
            "before modify_graph, please check whether apply_gradients is performed"
        )

    # For eval or predict, there is no need to pass an optimizer. However, if the customer has created an optimizer,
    # in DDR/SSD mode, the loaded embedding with dimension ext_size is used for swapping in and out. Therefore,
    # placeholders need to be provided for the slots.
    if optimizer is None and channel_id == EVAL_CHANNEL_ID:
        # This is an interim solution to fix ssd precision problem.
        if not ConfigInitializer.get_instance().train_params_config.bool_gauge_set:
            slot_place_holder = tf.zeros_like(each_var)
        else:
            slot_place_holder = tf.ones_like(each_var)

        for _ in range(slot_num):
            variable_and_slot_list.append(slot_place_holder)
    else:
        # opt name to slot dict
        for slot_dict in optimizer.values():
            for slot_val in slot_dict.values():
                variable_and_slot_list.append(slot_val)

    return variable_and_slot_list


def shm_swap(tables, swap_in_index, swap_out_index, h2d_name, d2h_name) -> tf.Operation:
    #var and clot num for table，set max num
    max_table_nun = 6
    table_list = []
    table_num = len(tables)
    for i in range(max_table_nun):
        if i < table_num:
            table_list.append(tables[i])
        else:
            table_list.append(tables[0])
    swap_in_index = tf.cast(swap_in_index, dtype=tf.int64)
    swap_out_index = tf.cast(swap_out_index, dtype=tf.int64)

    device_id = get_device_id()
    h2d_name_id = f'{h2d_name}_{device_id}'
    d2h_name_id = f'{d2h_name}_{device_id}'

    capacity = 50
    rma_shm_host_swap_in = mxrec_pybind.get_shm_mem(h2d_name_id, device_id, capacity)
    shm_swap_in = str(rma_shm_host_swap_in)

    rma_shm_host_swap_out = mxrec_pybind.get_shm_mem(d2h_name_id, device_id, capacity)
    shm_swap_out = str(rma_shm_host_swap_out)

    shm_swap_op = host_pipeline_ops.rma_swap_multi_tables(swap_in_index=swap_in_index,
                                                          swap_out_index=swap_out_index,
                                                          table_a=table_list[0],
                                                          table_b=table_list[1],
                                                          table_c=table_list[2],
                                                          table_d=table_list[3],
                                                          table_e=table_list[4],
                                                          table_f=table_list[5],
                                                          table_num=table_num,
                                                          shm_swap_in=shm_swap_in,
                                                          shm_swap_out=shm_swap_out)
    return shm_swap_op


def _get_swap_info(
    table_instance: BaseSparseEmbedding,
    variable_and_slot_list: List[tf.Variable],
    swap_info: SwapInfo,
    channel_id: int,
) -> List[tf.Operation]:  # pragma: no cover
    """
    Get swap op.
    :param table_instance: BaseSparseEmbedding
    :param variable_and_slot_list: [var + slots]
    :param swap_info: swap in/out length and position
    :param channel_id: train or predict
    :return: swap op
    """
    if table_instance.is_hbm:
        return [tf.no_op()]
    if len(variable_and_slot_list) == 0:
        raise RuntimeError("When enable emb_transfer, optimizer should have slots")
    use_static = ConfigInitializer.get_instance().use_static
    max_lookup_vec_size = None
    if use_static:
        max_lookup_vec_size = (
            table_instance.send_count * table_instance.rank_size
            if not table_instance.is_dp else table_instance.send_count
        )
    swap_out_pos = swap_info.swap_out_pos
    swap_in_pos = swap_info.swap_in_pos
    if use_static:
        swap_out_pos = swap_out_pos[: swap_info.swap_out_len]
        swap_in_pos = swap_in_pos[: swap_info.swap_in_len]
    use_shm_swap = ConfigInitializer.get_instance().use_shm_swap
    if use_shm_swap:
        optimizer = ConfigInitializer.get_instance().optimizer_config.get_optimizer_by_table_name(
            table_instance.table_name)
        h2d_name = f'{table_instance.table_name}_h2d_{channel_id}'
        d2h_name = f'{table_instance.table_name}_d2h_{channel_id}'
        if optimizer is None and channel_id == EVAL_CHANNEL_ID:
            swap_op = [shm_swap([variable_and_slot_list[0]], swap_in_index=swap_in_pos,
                                swap_out_index=swap_out_pos, h2d_name=h2d_name, d2h_name=d2h_name)]
        else:
            swap_op = [shm_swap(variable_and_slot_list, swap_in_index=swap_in_pos, swap_out_index=swap_out_pos,
                                h2d_name=h2d_name, d2h_name=d2h_name)]
        return swap_op

    with tf.compat.v1.variable_scope("h2d_emb"):
        logger.debug("Channel %s_h2d_%s was built for getnext.", table_instance.table_name, channel_id)
        h2d_emb = npu_ops.gen_npu_ops.get_next(
            output_types=[tf.float32],
            output_shapes=[[max_lookup_vec_size, table_instance.ext_emb_size]],
            channel_name=f"{table_instance.table_name}_h2d_{channel_id}",
        )[0]

    logger.debug("h2d_emb shape: %s", h2d_emb)
    if use_static:
        h2d_emb = h2d_emb[:swap_info.swap_in_len, :]
    swap_outs = [tf.gather(one_table, swap_out_pos) for one_table in variable_and_slot_list]
    swap_out = tf.concat(swap_outs, axis=1)
    logger.debug("Channel %s_d2h_%s was built for op outfeed.", table_instance.table_name, channel_id)
    swap_out_op = npu_ops.outfeed_enqueue_op(
        channel_name=f"{table_instance.table_name}_d2h_{channel_id}", inputs=[swap_out]
    )
    with tf.control_dependencies([swap_out_op]):
        nd_swap_pos = tf.expand_dims(swap_in_pos, 1)
        var_num = len(variable_and_slot_list)
        h2d_emb_split = tf.split(h2d_emb, var_num, axis=1)

        is_training = True if channel_id == TRAIN_CHANNEL_ID else False
        optimizer = ConfigInitializer.get_instance().optimizer_config.get_optimizer_by_table_name(
            table_instance.table_name, is_training=is_training
        )
        if optimizer is None and channel_id == EVAL_CHANNEL_ID:
            swap_in_op = [tf.compat.v1.scatter_nd_update(variable_and_slot_list[0], nd_swap_pos, h2d_emb_split[0])]
        else:
            swap_in_op = [
                tf.compat.v1.scatter_nd_update(variable_and_slot_list[i], nd_swap_pos, h2d_emb_split[i])
                for i in range(var_num)]
    return swap_in_op


def _get_new_batch_tensor(new_batch: Union[List, Tuple, Dict, tf.Tensor]) -> tf.Tensor:
    """
    Get a tensor from the new batch.

    Args:
        new_batch: New dataset batch.

    Returns: A tensor in the batch.

    """

    if isinstance(new_batch, list):
        batch_tensor = new_batch.pop()
        return _get_new_batch_tensor(batch_tensor)
    elif isinstance(new_batch, tuple):
        new_batch = list(new_batch)
        batch_tensor = new_batch.pop()
        return _get_new_batch_tensor(batch_tensor)
    elif isinstance(new_batch, dict):
        for _, value in new_batch.items():
            return _get_new_batch_tensor(value)

    if not isinstance(new_batch, tf.Tensor):
        raise TypeError(f"Cannot find a tensor from give batch: {new_batch}.")
    if AnchorIteratorOp.ITERATOR_GET_NEXT.value not in new_batch.name:
        raise ValueError(f"{new_batch} is not {AnchorIteratorOp.ITERATOR_GET_NEXT.value} tensor.")

    return new_batch


def replace_anchor_for_ddr_ssd(graph: tf.Graph, slot_num: int, channel_id: int):
    swap_args = SwapArgs()
    sparse_variables = graph.get_collection(
        ConfigInitializer.get_instance().train_params_config.ascend_global_hashtable_collection
    )

    for each_var in sparse_variables:
        table_instance = ConfigInitializer.get_instance().sparse_embed_config.get_table_instance(each_var)
        if table_instance.is_hbm:
            continue

        variable_and_slot_list = _get_variable_and_slot_list(each_var, slot_num, table_instance.table_name, channel_id)
        swap_args_dict = swap_args.swap_config_dict[table_instance.table_name][channel_id]
        swap_op = _get_swap_info(table_instance, variable_and_slot_list, swap_args_dict["swap_info"], channel_id)
        # Gather for id_offset need to be executed after swap_op.
        swap_control_dict = swap_args.swap_control_dict[table_instance.table_name][channel_id]
        if SwapDataType.CONTROL_OPS.value not in swap_control_dict:
            raise ValueError("swap control missing key [control_ops] in modify_graph_for_asc")
        control_ops = swap_control_dict[SwapDataType.CONTROL_OPS.value]
        utils.replace_anchor_control(graph, control_ops, swap_op)

        if channel_id == TRAIN_CHANNEL_ID and slot_num > 1:
            # Gather for slot need to be executed after swap_op.
            slot_control_dict = swap_args.slot_control_dict[table_instance.variable]
            if SwapDataType.CONTROL_OPS.value not in slot_control_dict:
                raise ValueError("slot control missing key [control_ops] in modify_graph_for_asc")
            slot_control_ops = slot_control_dict[SwapDataType.CONTROL_OPS.value]
            utils.replace_anchor_control(graph, slot_control_ops, swap_op)
