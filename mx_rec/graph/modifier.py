#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import logging
from collections import defaultdict

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter
from tensorflow.python.framework.ops import Operation

from mx_rec.core.asc.helper import get_asc_insert_func
from mx_rec.core.asc.feature_spec import FeatureSpec
from mx_rec.core.asc.manager import start_asc_pipeline
from mx_rec.core.embedding import SparseEmbedding
from mx_rec.constants.constants import ASCEND_CUTTING_POINT_INITIALIZER, ASCEND_SPARSE_LOOKUP_ENTRANCE, \
    ASCAnchorAttr, ASCEND_TIMESTAMP, ANCHOR_DATASET_NAME
from mx_rec.util.initialize import get_feature_spec, insert_feature_spec, set_initializer, \
    terminate_config_initializer, set_is_graph_modify_hook_running, get_bool_gauge_set, increase_run_times, \
    get_is_last_round, insert_merged_multi_lookup, get_merged_multi_lookup, set_target_batch, get_iterator_type, \
    set_iterator_type
from mx_rec.util.perf import performance
from mx_rec.graph.utils import check_input_list, find_parent_op, check_cutting_points, record_ops_to_replace, \
    export_pb_graph, make_sorted_key_to_tensor_list
from mx_rec.graph.merge_lookup import do_merge_lookup


def get_preprocessing_map_func(graph_def, input_names, output_names, batch_tensor_names=None,
                               pipeline_input_indexes=None):
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
        def print_tensors(batch_id, tracker=None):
            if tracker is None:
                tracker = []
            if isinstance(batch_id, dict):
                for key, item in batch_id.items():
                    print_tensors(item, tracker + [key])
            if isinstance(batch_id, tf.Tensor):
                logging.debug(f"######## tracker: {tracker}, tensor: {batch_id} ########")

        for batch in args:
            print_tensors(batch)

        batch = args[0]

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

        output_list = tf.import_graph_def(graph_def, input_map=dict(zip(input_names, input_tensors)),
                                          return_elements=output_names)

        output_batch = list(args)
        output_batch.append(tuple(output_list))
        return tuple(output_batch)

    return map_func


def get_input_index_list(cutting_point_list, replacement_specs, mapping_name_list, base_count, timestamp_index=None):
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


def find_make_iterator_op(batch_tensor):
    graph = tf.compat.v1.get_default_graph()
    operations = graph.get_operations()
    for each_op in operations:
        for input_tensor in batch_tensor.op.inputs:
            if input_tensor.op.outputs and input_tensor.op.outputs[0] in list(
                    each_op.inputs) and each_op.type == "MakeIterator":
                logging.debug(f"Op MakeIterator '{each_op.name}' was found.")
                return each_op

    raise ValueError(f"Op MakeIterator was not found.")


@performance("find_target_dataset_op")
def find_target_dataset_op(base_ops, op_type):
    base_ops = check_input_list(base_ops, tf.Operation)
    parent_ops = base_ops

    while True:
        for parent_op in parent_ops:
            if parent_op.type == op_type:
                return parent_op

        base_ops = parent_ops
        parent_ops = []
        for base_op in base_ops:
            parent_ops.extend(find_parent_op(base_op))

        if not parent_ops:
            raise ValueError(f"Op {op_type} was not found.")


def get_dataset_op(get_next_op: Operation) -> Operation:
    """
    根据`IteratorGetNext`算子从图中找到`OptimizeDataset`的dataset op. 注: TF2没有`OptimizeDataset`，则找的是dataset的默认锚点.

    Args:
        get_next_op: `IteratorGetNext`算子

    Returns: TF1返回`OptimizeDataset`算子，TF2返回dataset默认锚点的算子

    """

    if get_next_op.type != "IteratorGetNext":
        raise TypeError("Op '{get_next_op}' must be one instance of IteratorGetNext.")

    # looking for the MakeIterator operator which corresponds to given batch_tensor
    base_op = find_make_iterator_op(get_next_op.outputs[0])
    # looking for the op which is the one before OptimizeDataset operator
    if tf.__version__.startswith("1"):
        optimize_dataset_op = find_target_dataset_op(base_op, "ModelDataset")
        target_op = find_parent_op(optimize_dataset_op)
        if not target_op:
            raise RuntimeError(f"The parent op for 'ModelDataset' op was not found.")
        if target_op[0].type != "OptimizeDataset":
            raise TypeError(f"Op OptimizeDataset was not found.")
        target_op = target_op[0]
    else:
        # 'OptimizeDataset' is not available in TensorFlow2.X
        target_op = find_target_dataset_op(base_op, ANCHOR_DATASET_NAME)
    return target_op


def get_passing_tensor_list(src_tensors, target_op):
    def get_passing_tensors(src_tensor):
        passing_tensors = []
        tensor_list = [src_tensor]
        while tensor_list:
            last_tensor = tensor_list.pop()
            if last_tensor.op is target_op:
                passing_tensors.append(last_tensor)
            else:
                tensor_list.extend(list(last_tensor.op.inputs))

        return passing_tensors

    src_tensors = check_input_list(src_tensors, tf.Tensor)
    passing_tensor_list = []
    sub_src_tensors = []
    for tensor in src_tensors:
        passing_tensors = get_passing_tensors(tensor)
        for passing_tensor in passing_tensors:
            if passing_tensor not in passing_tensor_list:
                passing_tensor_list.append(passing_tensor)
        if len(passing_tensors) != 0:
            logging.info(f"passing_tensors: {passing_tensors}")
            sub_src_tensors.append(tensor)
        else:
            logging.info(f"Cannot find passing tensor for given tensor '{tensor}'.")

    output_index_list = [int(tensor.name.split(":")[1]) for tensor in passing_tensor_list]

    return passing_tensor_list, output_index_list, sub_src_tensors


def find_target_instance_dataset(variant_tensor):
    dataset_instance_list = tf.compat.v1.get_collection("dataset_group")
    for ins in dataset_instance_list:
        if ins._variant_tensor == variant_tensor:
            if not isinstance(ins, DatasetV1Adapter):
                ins = ins._input_dataset
            logging.debug(f"Find target instance '{ins}', whose variant_tensor is '{variant_tensor}'.")
            if not isinstance(ins.element_spec, dict) and not (
                    isinstance(ins.element_spec, (list, tuple)) and len(ins.element_spec) == 2 and isinstance(
                ins.element_spec[0], dict)):
                raise NotImplementedError("The found dataset does not return a valid layout.")

            return ins

    raise LookupError(f"Can not find target instance, whose variant_tensor is '{variant_tensor}' respectively.")


def get_sub_graph(input_tensors, output_tensors):
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


def update_input_tensor_with_new_batch(replacement_specs, new_get_next_op_name):
    graph = tf.compat.v1.get_default_graph()
    for old_tensor, item in replacement_specs.items():
        for idx, operator in item:
            old_tensor_name = old_tensor.name
            output_index = old_tensor_name.split(":")[-1]
            new_tensor_name = f"{new_get_next_op_name}:{output_index}"
            new_tensor = graph.get_tensor_by_name(new_tensor_name)
            operator._update_input(idx, new_tensor)


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


def modify_graph_and_start_emb_cache(dump_graph=False):
    modify_graph_for_asc(dump_graph=dump_graph)
    start_asc_pipeline()


def generate_get_next_op_specs(cutting_point_list, dump_graph):
    get_next_op_map = defaultdict(dict)
    for input_tensor in cutting_point_list:
        get_next_op = find_target_dataset_op(input_tensor.op, "IteratorGetNext")
        if get_next_op not in get_next_op_map:
            logging.debug(f"find a new get_next_op named '{get_next_op.name}'")
            replacement_specs = record_ops_to_replace(get_next_op)
            get_next_op_map[get_next_op]["replacement_specs"] = replacement_specs
            passing_tensor_list, batch_tensor_index_list, sub_cutting_point_list = \
                get_passing_tensor_list(cutting_point_list, get_next_op)
            get_next_op_map[get_next_op]["passing_tensor_list"] = passing_tensor_list
            get_next_op_map[get_next_op]["batch_tensor_index_list"] = batch_tensor_index_list
            get_next_op_map[get_next_op]["sub_cutting_point_list"] = sub_cutting_point_list

            sub_graph_def, input_name_list, output_name_list = get_sub_graph(passing_tensor_list,
                                                                             sub_cutting_point_list)
            get_next_op_map[get_next_op]["sub_graph_def"] = sub_graph_def
            get_next_op_map[get_next_op]["input_name_list"] = input_name_list
            get_next_op_map[get_next_op]["output_name_list"] = output_name_list
            get_next_op_map[get_next_op]["is_training"] = \
                SparseEmbedding.get_anchor_attribute(input_tensor, ASCAnchorAttr.IS_TRAINING)

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
        logging.warning("The dataset op was not found, the error is `%s`. Start to traverse the operations.", err)
        graph = tf.compat.v1.get_default_graph()
        dataset_op_list = [op for op in graph.get_operations() if ANCHOR_DATASET_NAME in op.name]
        logging.debug("In get_src_dataset function, current mode(train: True, eval: False): %s, dataset_op_list: %s.",
                      is_training, dataset_op_list)

        if len(dataset_op_list) == 1:
            target_op = dataset_op_list[0]
        elif is_training and len(dataset_op_list) == 2:
            prefetch_dataset_op_list = sorted(dataset_op_list, key=lambda op: op.name)
            target_op = prefetch_dataset_op_list[0]
        elif not is_training and len(dataset_op_list) == 3:
            prefetch_dataset_op_list = sorted(dataset_op_list, key=lambda op: op.name)
            target_op = prefetch_dataset_op_list[1]
        else:
            raise RuntimeError(f"The `{ANCHOR_DATASET_NAME}` was not found from the operations, dataset_op_list: "
                               f"{dataset_op_list}.") from err
    except Exception as err:
        raise RuntimeError(f"The dataset was not found, the error is `{err}`.") from err

    if not target_op.outputs:
        raise ValueError(f"The length of the outputs of target op `{target_op}` is 0.")
    logging.debug("Find target op `%s`, and output is `%s`.", target_op.name, target_op.outputs)
    src_dataset = find_target_instance_dataset(target_op.outputs[0])
    return src_dataset


def get_tgt_dataset(src_dataset: DatasetV1Adapter, sub_cutting_point_list: list, records: dict,
                    dump_graph: bool = False, prefetch: int = 10) -> DatasetV1Adapter:
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

    tgt_dataset = src_dataset.map(get_preprocessing_map_func(records.get("sub_graph_def"),
                                                             records.get("input_name_list"),
                                                             records.get("output_name_list"),
                                                             pipeline_input_indexes=records.get(
                                                                 "batch_tensor_index_list")))

    feature_numbers = [SparseEmbedding.get_anchor_attribute(cutting_point, ASCAnchorAttr.FEATURE_SPEC).feat_cnt for
                       cutting_point in sub_cutting_point_list]
    table_names = [SparseEmbedding.get_anchor_attribute(cutting_point, ASCAnchorAttr.FEATURE_SPEC).table_name for
                   cutting_point in sub_cutting_point_list]
    tgt_dataset = tgt_dataset.map(get_asc_insert_func(feature_numbers=feature_numbers,
                                                      table_names=table_names,
                                                      args_index_list=records.get("input_index_list"),
                                                      is_training=records.get("is_training"),
                                                      dump_graph=dump_graph))

    tgt_dataset = tgt_dataset.prefetch(prefetch)
    return tgt_dataset


def update_iterator_getnext(get_next_op: Operation, tgt_dataset: DatasetV1Adapter, is_training: bool, records: dict):
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
        raise RuntimeError("There is no tensor in the dataset. Please check the dataset and data processing.")
    iterator_type = ""
    if get_next_op.outputs[0].op.inputs:
        iterator_type = get_next_op.outputs[0].op.inputs[0].op.type
    if iterator_type == "IteratorV2":
        iterator_type = find_make_iterator_op(get_next_op.outputs[0]).type
    if iterator_type not in ("MakeIterator", "OneShotIterator"):
        raise RuntimeError(f"Only iterators `MakeIterator` and `OneShotIterator` are supported in `graph modify` mode, "
                           f"but the current iterator is `{iterator_type}`.")
    set_iterator_type(iterator_type)
    logging.info("The iterator type of dataset is `%s`.", iterator_type)

    if iterator_type == "MakeIterator":
        new_iterator = tgt_dataset.make_initializable_iterator()
        tf.compat.v1.add_to_collection(ASCEND_CUTTING_POINT_INITIALIZER, new_iterator.initializer)
        set_initializer(is_training, new_iterator.initializer)
    else:
        new_iterator = tgt_dataset.make_one_shot_iterator()
    new_batch = new_iterator.get_next()
    set_target_batch(is_training, new_batch)

    try:
        new_batch_tensor = list(new_batch.values())[0]
    except IndexError as err:
        raise IndexError("Cannot find a tensor from given batch.") from err
    new_get_next_op_name = find_target_dataset_op(new_batch_tensor.op, "IteratorGetNext").name
    update_input_tensor_with_new_batch(records.get("replacement_specs"), new_get_next_op_name)


@performance("graph_modifier")
def modify_graph_for_asc(dump_graph=False, prefetch=10):
    cutting_point_list = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_ENTRANCE)
    check_cutting_points(cutting_point_list)
    if not cutting_point_list:
        logging.warning("Nothing to revise.")
        return

    export_pb_graph("old_graph.pb", dump_graph)
    get_next_op_map = generate_get_next_op_specs(cutting_point_list, dump_graph)
    logging.debug("In modify_graph_for_asc function, get_next_op_map.len: %d, get_next_op_map.key: %s.",
                  len(get_next_op_map), get_next_op_map.keys())

    for get_next_op, records in get_next_op_map.items():
        is_training = records.get("is_training")

        # get source dataset
        src_dataset = get_src_dataset(get_next_op, is_training)

        # generate target dataset
        timestamp_index = get_timestamp_index(get_next_op, is_training)
        original_batch_tensor_count = get_dataset_tensor_count(src_dataset)
        sub_cutting_point_list = records.get("sub_cutting_point_list")
        input_index_list = get_input_index_list(sub_cutting_point_list,
                                                records.get("replacement_specs"),
                                                records.get("output_name_list"),
                                                original_batch_tensor_count, timestamp_index=timestamp_index)
        records["input_index_list"] = input_index_list
        tgt_dataset = get_tgt_dataset(src_dataset, sub_cutting_point_list, records,
                                      dump_graph=dump_graph, prefetch=prefetch)

        # update the batch of dataset
        update_iterator_getnext(get_next_op, tgt_dataset, is_training, records)

        # In eval mode, backward is not required. In addition, compute gradients is not executed when
        # only eval is used. Therefore, `do_merge_lookup` needs to be invoked during modify graph.
        if not is_training:
            do_merge_lookup(is_train=False)
            if 'evaluate' in get_bool_gauge_set():
                logging.debug("In estimator mode, eval re-creates graph each time, so the flag needs to be cleared.")
                insert_merged_multi_lookup(is_training, False)
        # In training mode, `do_merge_lookup` should have been executed in compute gradients phase.
        if is_training and not get_merged_multi_lookup(True):
            raise RuntimeError("In training mode, `do_merge_lookup` should have been executed in compute gradients "
                               "phase. Please check whether compute gradients is performed.")

    logging.info("Graph has been revised.")
    export_pb_graph("new_graph.pb", dump_graph)


def get_timestamp_index(get_next_op, is_training):
    timestamp_tensor_list = tf.compat.v1.get_collection(ASCEND_TIMESTAMP)
    timestamp_index = None
    for timestamp in timestamp_tensor_list:
        if timestamp in get_next_op.outputs:
            timestamp_index = int(timestamp.name.split(":")[1])
            timestamp_feature_spec = get_feature_spec("timestamp")
            if timestamp_feature_spec is None:
                timestamp_feature_spec = FeatureSpec("timestamp", index_key=timestamp_index, is_timestamp=True)
                timestamp_feature_spec.include_timestamp(is_training)
                insert_feature_spec(timestamp_feature_spec, is_training)
                break

            if timestamp_feature_spec.index_key != timestamp_index:
                raise ValueError(f"Given timestamp_index, which is {timestamp_index}, does not match index "
                                 f"key. Please double check.")
            timestamp_feature_spec.include_timestamp(is_training)
            break
    return timestamp_index


class GraphModifierHook(tf.estimator.SessionRunHook):
    def __init__(self, dump_graph=True, modify_graph=True):
        self._dump_graph = dump_graph
        self._modify_graph = modify_graph
        self._iterator_type = ""
        set_is_graph_modify_hook_running(True)

    def begin(self):
        if self._modify_graph:
            modify_graph_and_start_emb_cache(dump_graph=self._dump_graph)
        else:
            start_asc_pipeline()

        self._iterator_type = get_iterator_type()
        if self._modify_graph and self._iterator_type not in ("MakeIterator", "OneShotIterator"):
            raise ValueError("The value of iterator type should be like `MakeIterator` or `OneShotIterator`.")
        logging.debug("In GraphModifierHook, iterator type is `%s`.", self._iterator_type)

    def after_create_session(self, session, coord):
        if self._modify_graph and self._iterator_type == "MakeIterator":
            session.run(tf.compat.v1.get_collection(ASCEND_CUTTING_POINT_INITIALIZER))

    def end(self, session):
        bool_gauge_set = get_bool_gauge_set()
        logging.debug(f"GraphModifierHook, bool_gauge_set: {bool_gauge_set}")

        # In eval or predict mode, the initializer can be directly terminated.
        if 'train' not in bool_gauge_set:
            logging.debug(f"In evaluate or predict case, GraphModifierHook call 'terminate_config_initializer'...")
            terminate_config_initializer()
            return

        if 'train_and_evaluate' in bool_gauge_set:
            increase_run_times()
            # In 'train_and_evaluate' mode, the terminate function should be executed last.
            if get_is_last_round():
                logging.debug(f"In train_and_evaluate case, GraphModifierHook call 'terminate_config_initializer'...")
                terminate_config_initializer()
