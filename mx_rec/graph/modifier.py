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
from collections import defaultdict
from collections.abc import Callable
from typing import Any, List, Dict, Tuple, DefaultDict

import tensorflow as tf
from tensorflow import Operation, Tensor
from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from mx_rec.constants.constants import ASCEND_CUTTING_POINT_INITIALIZER, ASCEND_SPARSE_LOOKUP_ENTRANCE, \
    ASCAnchorAttr, ASCEND_TIMESTAMP, MAX_WHILE_SIZE, LIBREC_EOS_OPS_SO
from mx_rec.core.asc.feature_spec import FeatureSpec
from mx_rec.core.asc.helper import get_asc_insert_func
from mx_rec.core.asc.manager import start_asc_pipeline
from mx_rec.core.asc.swap_args import SwapArgs
from mx_rec.core.emb.base_sparse_embedding import BaseSparseEmbedding
from mx_rec.graph.merge_lookup import do_merge_lookup
from mx_rec.graph.utils import check_input_list, find_parent_op, check_cutting_points, record_ops_to_replace, \
    export_pb_graph, make_sorted_key_to_tensor_list, replace_anchor_control
from mx_rec.graph.constants import DeprecatedOp, AnchorDatasetOp, AnchorIteratorOp
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.util.log import logger
from mx_rec.util.ops import import_host_pipeline_ops
from mx_rec.util.perf import performance
from mx_rec.util.tf_version_adapter import hccl_ops, npu_ops
from mx_rec.validator.validator import para_checker_decorator, ClassValidator


@dataclasses.dataclass
class AnchorRecord:
    replacement_spec: DefaultDict[Tensor, List[Tuple[int, Operation]]]
    passing_tensors: List[Tensor]
    batch_tensor_indexs: List[int]
    sub_cutting_points: List[Tensor]
    sub_graph_def: GraphDef
    input_names: List[str]
    output_names: List[str]
    is_training: bool
    input_indexs: List[int] = None


def get_preprocessing_map_func(
        graph_def: GraphDef,
        input_names: List[str],
        output_names: List[str],
        batch_tensor_names: List[str] = None,
        pipeline_input_indexes: List[int] = None
) -> Callable:
    input_names = check_input_list(input_names, str)
    output_names = check_input_list(output_names, str)
    batch_tensor_names = check_input_list(batch_tensor_names, str)
    pipeline_input_indexes = check_input_list(pipeline_input_indexes, int)
    both_is_none = batch_tensor_names is None and pipeline_input_indexes is None
    both_not_none = batch_tensor_names is not None and pipeline_input_indexes is not None
    if both_is_none or both_not_none:
        raise ValueError("It is legal when and only when one of the parameters 'batch_tensor_names' and "
                         "'pipeline_input_indexes' was given.")

    def map_func(*args):
        logger.debug("In get_preprocessing_map_func, the old batch is: %s.", args)
        batch = dict()
        parse_batch(args, batch, key=None)
        logger.debug("In get_preprocessing_map_func, the parse batch is: %s.", batch)

        input_tensors = []
        if batch_tensor_names is not None:
            for tensor_name in batch_tensor_names:
                tensor = batch.get(tensor_name)
                if tensor is None:
                    raise ValueError(f"Given input_tensor_name '{tensor_name}' is invalid.")

                input_tensors.append(tensor)

        else:
            graph = tf.compat.v1.get_default_graph()
            for index in pipeline_input_indexes:
                tensor = graph.get_tensor_by_name("args_%d:0" % index)
                input_tensors.append(tensor)

        # 以tf.import_graph_def()作为read emb key的输入，保证数据读取到传入lookup的ids过程中的特征处理关系能够保留在子图中。
        output_list = tf.import_graph_def(graph_def, input_map=dict(zip(input_names, input_tensors)),
                                          return_elements=output_names)

        output_batch = [batch, tuple(output_list)]
        logger.debug("In get_preprocessing_map_func, the output batch is: %s.", output_batch)
        return tuple(output_batch)

    return map_func


def parse_batch(data_args: Any, data_batch: dict, key: str = None):
    """
    解析原始数据集中的batch，并将非dict格式的batch转为dict格式.
    Args:
        data_args: 待解析的batch
        data_batch: 解析后的batch
        key: batch中的key

    Returns: None

    """

    def parse_tensor(data_tensor: Tensor, data_batch: dict, key: str = None):
        """
        将待解析batch中的tensor写入解析后的batch中，如果key存在则使用原key，不存在则生成batch中字典序最小的key.
        Args:
            data_tensor: 待解析batch中的tensor
            data_batch: 解析后的batch
            key: batch中的key

        Returns: None

        """

        if key is not None:
            data_batch[key] = data_tensor
            return

        last_key = f"{sorted(data_batch)[-1]}_last_key"
        data_batch[last_key] = data_tensor

    # 开始解析old batch
    if isinstance(data_args, dict):
        for key, data_tensor in data_args.items():
            parse_batch(data_tensor, data_batch, key)
        return
    if isinstance(data_args, (list, tuple)):
        for data_arg in data_args:
            parse_batch(data_arg, data_batch, key)
        return
    if isinstance(data_args, Tensor):
        # 将old batch中的tensor加入到dict中
        parse_tensor(data_args, data_batch, key)
        return

    raise ValueError(f"Invalid batch type, expected: (dict, list, tuple, Tensor), got: {type(data_args)}.")


def get_input_index_list(
        cutting_point_list: List[Tensor],
        replacement_specs: DefaultDict[Tensor, List[Tuple[int, Operation]]],
        mapping_name_list: List[str],
        base_count: int,
        timestamp_index: int = None
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


def find_make_iterator_op(batch_tensor: Tensor) -> Operation:
    graph = tf.compat.v1.get_default_graph()
    operations = graph.get_operations()
    for each_op in operations:
        for input_tensor in batch_tensor.op.inputs:
            if input_tensor.op.outputs and input_tensor.op.outputs[0] in list(
                    each_op.inputs) and each_op.type == AnchorIteratorOp.MAKE_ITERATOR.value:
                logger.debug("Op MakeIterator '%s' was found.", each_op.name)
                return each_op

    raise ValueError(f"op MakeIterator was not found.")


@performance("find_target_dataset_op")
def find_target_dataset_op(base_ops: Operation, op_type: str) -> Operation:
    base_ops = check_input_list(base_ops, tf.Operation)
    parent_ops = base_ops

    while_num = 0
    while True:
        while_num += 1
        if while_num > MAX_WHILE_SIZE:
            raise RuntimeError(f"In find_target_dataset_op function, the maximum cycle depth is greater "
                               f"than {MAX_WHILE_SIZE}.")
        for parent_op in parent_ops:
            if parent_op.type == op_type:
                return parent_op

        base_ops = parent_ops
        parent_ops = []
        for base_op in base_ops:
            parent_ops.extend(find_parent_op(base_op))

        if not parent_ops:
            raise ValueError(f"op {op_type} was not found.")


def get_dataset_op(get_next_op: Operation) -> Operation:
    """
    根据`IteratorGetNext`算子从图中找到`OptimizeDataset`的dataset op.
    注: TF2没有`OptimizeDataset`，则找的是dataset的默认锚点.

    Args:
        get_next_op: `IteratorGetNext`算子

    Returns: TF1返回`OptimizeDataset`算子，TF2返回dataset默认锚点的算子

    """

    if get_next_op.type != AnchorIteratorOp.ITERATOR_GET_NEXT.value:
        raise TypeError(f"op '{get_next_op}' must be one instance of IteratorGetNext.")

    # looking for the MakeIterator operator which corresponds to given batch_tensor
    base_op = find_make_iterator_op(get_next_op.outputs[0])
    # looking for the op which is the one before OptimizeDataset operator
    if tf.__version__.startswith("1"):
        optimize_dataset_op = find_target_dataset_op(base_op, AnchorDatasetOp.MODEL_DATASET.value)
        target_op = find_parent_op(optimize_dataset_op)
        if not target_op:
            raise RuntimeError("the parent op for 'ModelDataset' op was not found.")
        if target_op[0].type != AnchorDatasetOp.OPTIMIZE_DATASET.value:
            raise TypeError("op OptimizeDataset was not found.")
        target_op = target_op[0]
    else:
        # 'OptimizeDataset' is not available in TensorFlow2.X
        target_op = find_target_dataset_op(base_op, AnchorDatasetOp.PREFETCH_DATASET.value)
    return target_op


def get_passing_tensor_list(
        src_tensors: List[Tensor],
        target_op: Operation
) -> Tuple[List[Tensor], List[int], List[Tensor]]:
    def get_passing_tensors(src_tensor):
        passing_tensors = []
        tensor_list = [src_tensor]
        while_num = 0
        while tensor_list:
            while_num += 1
            if while_num > MAX_WHILE_SIZE:
                raise RuntimeError(f"In get_passing_tensors function, the maximum cycle depth is greater "
                                   f"than {MAX_WHILE_SIZE}.")
            last_tensor = tensor_list.pop()
            if last_tensor.op is target_op:
                passing_tensors.append(last_tensor)
            else:
                tensor_list.extend(list(last_tensor.op.inputs))

        return passing_tensors

    src_tensors = check_input_list(src_tensors, Tensor)
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


def find_target_instance_dataset(variant_tensor: Tensor) -> DatasetV1Adapter:
    dataset_instance_list = tf.compat.v1.get_collection("dataset_group")
    for ins in dataset_instance_list:
        if ins._variant_tensor == variant_tensor:
            if not isinstance(ins, DatasetV1Adapter):
                ins = ins._input_dataset
            logger.debug("Find target instance '%s', whose variant_tensor is '%s'.", ins, variant_tensor)
            if not isinstance(ins.element_spec, dict) and not (
                    isinstance(ins.element_spec, (list, tuple)) and len(ins.element_spec) == 2 and isinstance(
                ins.element_spec[0], dict)):
                raise NotImplementedError("the found dataset does not return a valid layout.")

            return ins

    raise LookupError(f"Can not find target instance, whose variant_tensor is '{variant_tensor}' respectively.")


def get_sub_graph(
        input_tensors: List[Tensor],
        output_tensors: List[Tensor]
) -> Tuple[GraphDef, List[str], List[str]]:
    input_tensors = check_input_list(input_tensors, tf.Tensor)
    output_tensors = check_input_list(output_tensors, tf.Tensor)
    input_op_name_list = [tensor.op.name for tensor in input_tensors]
    output_op_name_list = [tensor.op.name for tensor in output_tensors]

    graph_def = tf.compat.v1.get_default_graph().as_graph_def()
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


def update_input_tensor_with_new_batch(replacement_specs: DefaultDict[Tensor, List[Tuple[int, Operation]]],
                                       new_get_next_op_name: str,
                                       new_batch: Dict[str, Tensor]):
    """
    用新batch中的IteratorGetNext替换计算图中老batch的IteratorGetNext.

    Args:
        replacement_specs: 记录待替换算子的dict，key为老batch的IteratorGetNext，value为以老batch作为输入的算子
        new_get_next_op_name: 新数据集的get_next算子名称
        new_batch: 新数据集的batch

    Returns: None

    """

    graph = tf.compat.v1.get_default_graph()
    for old_tensor, item in replacement_specs.items():
        for idx, operator in item:
            old_tensor_name = old_tensor.name
            output_index = old_tensor_name.split(":")[-1]
            new_tensor_name = f"{new_get_next_op_name}:{output_index}"
            new_tensor = graph.get_tensor_by_name(new_tensor_name)
            try:
                operator._update_input(idx, new_tensor)
            except InvalidArgumentError as err:
                logger.info("The replacement specs keys (old batch) is: %s. \n\t\t The new batch is: %s.",
                            replacement_specs.keys(), new_batch)
                raise RuntimeError(f"Cannot update edge, old tensor: {old_tensor}, new tensor: {new_tensor}.") from err


def get_dataset_tensor_count(dataset: DatasetV1Adapter) -> int:
    """
    获取数据集中batch的tensor数量.

    Args:
        dataset: 数据集实例

    Returns: 数据集batch中的tensor数量

    """

    src_element_spec = dataset.element_spec
    if not isinstance(src_element_spec, (list, tuple)):
        src_element_spec = [src_element_spec]
    src_sorted_keys = make_sorted_key_to_tensor_list(src_element_spec, [])

    return len(src_sorted_keys)


def change_ext_emb_size_by_opt(optimizer):
    for _, table_instance in ConfigInitializer.get_instance().sparse_embed_config.table_instance_dict.items():
        # When dynamic expansion mode, ext_emb_size is set by optimizer
        if ConfigInitializer.get_instance().use_dynamic_expansion or not table_instance.is_hbm:
            table_instance.ext_emb_size = table_instance.emb_size * (1 + optimizer.slot_num)
            logger.info("ext_emb_size is reset to be %s in change_ext_emb_size_by_opt", table_instance.ext_emb_size)


@para_checker_decorator(
    check_option_list=[("dump_graph", ClassValidator, {"classes": (bool,)})]
)
def modify_graph_and_start_emb_cache(dump_graph: bool = False):
    modify_graph_for_asc(dump_graph=dump_graph)
    start_asc_pipeline()


def generate_get_next_op_specs(
        cutting_point_list: List[Tensor],
        dump_graph: bool = False
) -> Dict[Tensor, AnchorRecord]:
    get_next_op_map = defaultdict(dict)

    for input_tensor in cutting_point_list:
        get_next_op = find_target_dataset_op(input_tensor.op, AnchorIteratorOp.ITERATOR_GET_NEXT.value)
        if get_next_op not in get_next_op_map:
            logger.debug("find a new get_next_op named '%s'", get_next_op.name)

            replacement_specs = record_ops_to_replace(get_next_op)
            passing_tensors, batch_tensor_indexs, sub_cutting_points = \
                get_passing_tensor_list(cutting_point_list, get_next_op)
            sub_graph_def, input_names, output_names = get_sub_graph(passing_tensors, sub_cutting_points)
            is_training = BaseSparseEmbedding.get_anchor_attribute(input_tensor, ASCAnchorAttr.IS_TRAINING)

            record = AnchorRecord(
                replacement_specs,
                passing_tensors,
                batch_tensor_indexs,
                sub_cutting_points,
                sub_graph_def,
                input_names,
                output_names,
                is_training
            )
            get_next_op_map[get_next_op] = record

            export_pb_graph(f"cut_graph_{get_next_op.name}.pb", dump_graph, graph_def=sub_graph_def)

    return get_next_op_map


def get_src_dataset(get_next_op: Operation, is_training: bool) -> DatasetV1Adapter:
    """
    根据`IteratorGetNext`算子在计算图中找出原始dataset.

    Args:
        get_next_op: `IteratorGetNext`算子
        is_training: 当前是否为训练模式，训练模式为True，否则为False

    Returns: 原始数据集

    """

    try:
        target_op = get_dataset_op(get_next_op)
    except (ValueError, TypeError, RuntimeError) as err:
        logger.warning("The dataset op was not found, the error is `%s`. Start to traverse the operations.", err)
        graph = tf.compat.v1.get_default_graph()
        dataset_op_list = [op for op in graph.get_operations() if AnchorDatasetOp.PREFETCH_DATASET.value in op.name]

        # WARN: Couple with NoGradSubGraphSlicer::_find_old_dataset.
        dataset_op_list = list(
            filter(lambda op: op not in tf.compat.v1.get_collection(DeprecatedOp.DEPRECATED_PREFETCH_DATASET),
            dataset_op_list)
        )
        dataset_op_list = sorted(dataset_op_list, key=lambda op: op.name)

        logger.debug("In get_src_dataset function, current mode(train: True, eval: False): %s, dataset_op_list: %s.",
                     is_training, dataset_op_list)

        if len(dataset_op_list) == 1:
            target_op = dataset_op_list[0]
        elif is_training and len(dataset_op_list) == 2:
            prefetch_dataset_op_list = sorted(dataset_op_list, key=lambda op: op.name)
            target_op = prefetch_dataset_op_list[0]
        elif not is_training and len(dataset_op_list) == 2:
            prefetch_dataset_op_list = sorted(dataset_op_list, key=lambda op: op.name)
            target_op = prefetch_dataset_op_list[1]
        elif not is_training and len(dataset_op_list) == 3:
            prefetch_dataset_op_list = sorted(dataset_op_list, key=lambda op: op.name)
            target_op = prefetch_dataset_op_list[1]
        else:
            raise RuntimeError(f"'{AnchorDatasetOp.PREFETCH_DATASET.value}' not found, got transformation datasets: "
                               f"{dataset_op_list}.") from err
    except Exception as err:
        raise RuntimeError(f"The dataset was not found, the error is `{err}`.") from err

    if not target_op.outputs:
        raise ValueError(f"The length of the outputs of target op `{target_op}` is 0.")
    logger.debug("Find target op `%s`, and output is `%s`.", target_op.name, target_op.outputs)
    src_dataset = find_target_instance_dataset(target_op.outputs[0])
    return src_dataset


def get_tgt_dataset(
        src_dataset: DatasetV1Adapter,
        sub_cutting_point_list: List[Tensor],
        record: AnchorRecord,
        dump_graph: bool = False,
        prefetch: int = 10
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
        record.is_training)
    # 在数据读取完时，通过EosDataset向acl数据通道发送end_of_sequence
    max_train_steps = ConfigInitializer.get_instance().max_steps
    max_eval_steps = ConfigInitializer.get_instance().eval_steps
    src_dataset = src_dataset.eos_map(librec, channel_id, max_train_steps, max_eval_steps)

    tgt_dataset = src_dataset.map(get_preprocessing_map_func(record.sub_graph_def,
                                                             record.input_names,
                                                             record.output_names,
                                                             pipeline_input_indexes=record.batch_tensor_indexs))

    feature_numbers = [BaseSparseEmbedding.get_anchor_attribute(cutting_point, ASCAnchorAttr.FEATURE_SPEC).feat_cnt for
                       cutting_point in sub_cutting_point_list]
    table_names = [BaseSparseEmbedding.get_anchor_attribute(cutting_point, ASCAnchorAttr.FEATURE_SPEC).table_name for
                   cutting_point in sub_cutting_point_list]
    tgt_dataset = tgt_dataset.map(get_asc_insert_func(feature_numbers=feature_numbers,
                                                      table_names=table_names,
                                                      args_index_list=record.input_indexs,
                                                      is_training=record.is_training,
                                                      dump_graph=dump_graph))

    tgt_dataset = tgt_dataset.prefetch(prefetch)
    return tgt_dataset


def update_iterator_getnext(get_next_op: Operation,
                            tgt_dataset: DatasetV1Adapter,
                            is_training: bool,
                            record: AnchorRecord):
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
        iterator_type = find_make_iterator_op(get_next_op.outputs[0]).type
    if iterator_type not in (AnchorIteratorOp.MAKE_ITERATOR.value, AnchorIteratorOp.ONE_SHOT_ITERATOR.value):
        raise RuntimeError(f"Only iterators `MakeIterator` and `OneShotIterator` are supported in `graph modify` mode, "
                           f"but the current iterator is `{iterator_type}`.")
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

    try:
        new_batch_tensor = list(new_batch.values())[0]
    except IndexError as err:
        raise IndexError("Cannot find a tensor from given batch.") from err
    new_get_next_op_name = find_target_dataset_op(new_batch_tensor.op, AnchorIteratorOp.ITERATOR_GET_NEXT.value).name
    update_input_tensor_with_new_batch(record.replacement_spec, new_get_next_op_name, new_batch)


def get_swap_info(table_instance: BaseSparseEmbedding, variable_and_slot_list: list, swap_len: int, swap_pos: list,
                  channel_id: int) -> list:
    """
    Get swap info if threshold is configured.
    :param table_instance: BaseSparseEmbedding
    :param variable_and_slot_list: [var + slots]
    :param swap_len: swap length
    :param swap_pos: swap position
    :param channel_id: train or predict
    :return: swap info
    """
    use_static = ConfigInitializer.get_instance().use_static
    max_lookup_vec_size = None
    if use_static:
        max_lookup_vec_size = table_instance.send_count * table_instance.rank_size

    if table_instance.is_hbm:
        swap_in = [tf.no_op()]
    else:
        with tf.compat.v1.variable_scope("h2d_emb"):
            logger.debug('Channel %s_h2d_%s was built for getnext', table_instance.table_name, channel_id)
            h2d_emb = npu_ops.gen_npu_ops.get_next(
                output_types=[tf.float32],
                output_shapes=[[max_lookup_vec_size, table_instance.ext_emb_size]],
                channel_name=f'{table_instance.table_name}_h2d_{channel_id}')[0]
        logger.debug("h2d_emb shape: %s", h2d_emb)
        if not isinstance(variable_and_slot_list, list):
            raise RuntimeError("When enable emb_transfer, optimizer should have slots")
        if use_static:
            swap_pos = swap_pos[0:swap_len]
            h2d_emb = h2d_emb[0:swap_len, :]
        swap_outs = [tf.gather(one_table, swap_pos) for one_table in variable_and_slot_list]
        swap_out = tf.concat(swap_outs, axis=1)
        logger.debug('Channel %s_d2h_%s was built for op outfeed.', table_instance.table_name, channel_id)
        swap_out_op = npu_ops.outfeed_enqueue_op(
            channel_name=f'{table_instance.table_name}_d2h_{channel_id}', inputs=[swap_out])
        with tf.control_dependencies([swap_out_op]):
            nd_swap_pos = tf.expand_dims(swap_pos, 1)
            table_num = len(variable_and_slot_list)
            h2d_emb_split = tf.split(h2d_emb, table_num, axis=1)
            optimizer = ConfigInitializer.get_instance().optimizer_config.get_optimizer_by_table_name(
                table_instance.table_name)
            if optimizer is None and channel_id == 1:
                swap_in = [tf.compat.v1.scatter_nd_update(variable_and_slot_list[0], nd_swap_pos, h2d_emb_split[0])]
            else:
                swap_in = [tf.compat.v1.scatter_nd_update(variable_and_slot_list[i], nd_swap_pos, h2d_emb_split[i])
                           for i in range(len(variable_and_slot_list))]
    return swap_in


def get_variable_and_slot_list(each_var, slot_num, table_name, channel_id):
    variable_and_slot_list = [each_var]
    if slot_num == 0:
        return variable_and_slot_list

    # 通过apply_gradients创建optimizer
    optimizer = ConfigInitializer.get_instance().optimizer_config.get_optimizer_by_table_name(table_name)
    if optimizer is None and channel_id == 0:
        raise RuntimeError("In training mode, table_instance should have been set_optimizer_for_table "
                           "before modify_graph, please check whether apply_gradients is performed")

    # predict不需要传优化器，但是如果客户创建了优化器，ddr模式加载的是维度ext_size的emb用作换入换出，所以需要给slot零值占位
    if optimizer is None and channel_id == 1:
        slot_place_holder = tf.zeros_like(each_var)
        for _ in range(slot_num):
            variable_and_slot_list.append(slot_place_holder)
    else:
        # opt name to slot dict
        for slot_dict in optimizer.values():
            for slot_val in slot_dict.values():
                variable_and_slot_list.append(slot_val)

    return variable_and_slot_list


def modify_graph_for_ddr(get_next_op_map):
    # 通过create_hash_optimizer创建optimizer_instance
    optimizer_instance = ConfigInitializer.get_instance().optimizer_config.optimizer_instance
    # predict
    if optimizer_instance is None:
        slot_num = 0
    else:
        # ddr和扩容需要在获取优化器后重置ext
        change_ext_emb_size_by_opt(optimizer_instance)
        slot_num = optimizer_instance.slot_num

    for _, record in get_next_op_map.items():
        is_training = record.is_training
        channel_id = 0 if is_training else 1

        swap_args = SwapArgs()
        sparse_variables = tf.compat.v1.get_collection(
            ConfigInitializer.get_instance().train_params_config.ascend_global_hashtable_collection)

        for each_var in sparse_variables:
            table_instance = ConfigInitializer.get_instance().sparse_embed_config.get_table_instance(each_var)
            if table_instance.is_hbm:
                continue
            swap_args_dict = swap_args.swap_config_dict[table_instance.table_name][channel_id]
            swap_pos = swap_args_dict['swap_pos']
            swap_len = swap_args_dict['swap_len']
            variable_and_slot_list = get_variable_and_slot_list(each_var, slot_num, table_instance.table_name,
                                                                channel_id)

            swap_op = get_swap_info(table_instance, variable_and_slot_list, swap_len, swap_pos, channel_id)
            swap_control_dict = swap_args.swap_control_dict[table_instance.table_name][channel_id]
            if "control_ops" not in swap_control_dict:
                raise ValueError("Missing Required key in modify_graph_for_asc: control_ops")
            control_ops = swap_control_dict['control_ops']
            replace_anchor_control(control_ops, swap_op)


@performance("graph_modifier")
def modify_graph_for_asc(dump_graph: bool = False, prefetch: int = 10):
    cutting_point_list = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_ENTRANCE)
    check_cutting_points(cutting_point_list)
    if not cutting_point_list:
        logger.warning("Nothing to revise.")
        return

    export_pb_graph("old_graph.pb", dump_graph)
    get_next_op_map = generate_get_next_op_specs(cutting_point_list, dump_graph)
    logger.debug("In modify_graph_for_asc function, get_next_op_map.len: %d, get_next_op_map.key: %s.",
                 len(get_next_op_map), get_next_op_map.keys())

    for get_next_op, record in get_next_op_map.items():
        is_training = record.is_training

        # get source dataset
        src_dataset = get_src_dataset(get_next_op, is_training)

        # generate target dataset
        timestamp_index = get_timestamp_index(get_next_op, is_training)
        original_batch_tensor_count = get_dataset_tensor_count(src_dataset)
        sub_cutting_points = record.sub_cutting_points
        input_index_list = get_input_index_list(sub_cutting_points,
                                                record.replacement_spec,
                                                record.output_names,
                                                original_batch_tensor_count, timestamp_index=timestamp_index)
        record.input_indexs = input_index_list
        tgt_dataset = get_tgt_dataset(src_dataset, sub_cutting_points, record,
                                      dump_graph=dump_graph, prefetch=prefetch)

        # update the batch of dataset
        update_iterator_getnext(get_next_op, tgt_dataset, is_training, record)

        # In eval mode, backward is not required. In addition, compute gradients is not executed when
        # only eval is used. Therefore, `do_merge_lookup` needs to be invoked during modify graph.
        if not is_training:
            do_merge_lookup(is_train=False)
            if 'evaluate' in ConfigInitializer.get_instance().train_params_config.bool_gauge_set:
                logger.debug("In estimator mode, eval re-creates graph each time, so the flag needs to be cleared.")
                ConfigInitializer.get_instance().train_params_config.insert_merged_multi_lookup(is_training, False)
        # In training mode, `do_merge_lookup` should have been executed in compute gradients phase.
        if is_training and not ConfigInitializer.get_instance().train_params_config.get_merged_multi_lookup(True):
            raise RuntimeError("In training mode, `do_merge_lookup` should have been executed in compute gradients "
                               "phase. Please check whether compute gradients is performed.")
    # ddr
    modify_graph_for_ddr(get_next_op_map)

    logger.info("Graph has been revised.")
    export_pb_graph("new_graph.pb", dump_graph)


def get_timestamp_index(get_next_op: Operation, is_training: bool) -> int:
    timestamp_tensor_list = tf.compat.v1.get_collection(ASCEND_TIMESTAMP)
    timestamp_index = None
    for timestamp in timestamp_tensor_list:
        if timestamp in get_next_op.outputs:
            timestamp_index = int(timestamp.name.split(":")[1])
            timestamp_feature_spec = ConfigInitializer.get_instance().feature_spec_config.get_feature_spec("timestamp")
            if timestamp_feature_spec is None:
                timestamp_feature_spec = FeatureSpec("timestamp", index_key=timestamp_index, is_timestamp=True)
                timestamp_feature_spec.include_timestamp(is_training)
                ConfigInitializer.get_instance().feature_spec_config.insert_feature_spec(timestamp_feature_spec,
                                                                                         is_training)
                break

            if timestamp_feature_spec.index_key != timestamp_index:
                raise ValueError(f"Given timestamp_index, which is {timestamp_index}, does not match index "
                                 f"key. Please double check.")
            timestamp_feature_spec.include_timestamp(is_training)
            break
    return timestamp_index


class GraphModifierHook(tf.estimator.SessionRunHook):
    @para_checker_decorator(
        check_option_list=[
            ("dump_graph", ClassValidator, {"classes": (bool,)}),
            ("modify_graph", ClassValidator, {"classes": (bool,)})
        ]
    )
    def __init__(self, dump_graph=False, modify_graph=True):
        self._dump_graph = dump_graph
        self._modify_graph = modify_graph
        self._iterator_type = ""
        ConfigInitializer.get_instance().train_params_config.is_graph_modify_hook_running = True

    def begin(self):
        if self._modify_graph:
            modify_graph_and_start_emb_cache(dump_graph=self._dump_graph)
        else:
            start_asc_pipeline()

        self._iterator_type = ConfigInitializer.get_instance().train_params_config.iterator_type
        if self._modify_graph and self._iterator_type not in (AnchorIteratorOp.MAKE_ITERATOR.value,
                                                              AnchorIteratorOp.ONE_SHOT_ITERATOR.value):
            raise ValueError("the value of iterator type should be like `MakeIterator` or `OneShotIterator`.")
        logger.debug("In GraphModifierHook, iterator type is `%s`.", self._iterator_type)

    def after_create_session(self, session, coord):
        if self._modify_graph and self._iterator_type == AnchorIteratorOp.MAKE_ITERATOR.value:
            session.run(tf.compat.v1.get_collection(ASCEND_CUTTING_POINT_INITIALIZER))
