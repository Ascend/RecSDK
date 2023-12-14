#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import os
import shutil
import unittest
from unittest import mock
from typing import Callable

import tensorflow as tf

from mx_rec.core.asc.feature_spec import FeatureSpec
from tests.mx_rec.core.generator_dataset import generate_dataset, Config
from tests.mx_rec.core.mock_class import MockHostPipeLineOps


class TestGetAscInsertFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.helper.get_asc_insert_func'.
    """

    def test_get_asc_insert_func_case1(self):
        """
        case1: tgt_key_specs和args_index_list都为None，抛出异常
        """

        from mx_rec.core.asc.helper import get_asc_insert_func

        with self.assertRaises(ValueError):
            get_asc_insert_func(tgt_key_specs=None, args_index_list=None)

    def test_get_asc_insert_func_case2(self):
        """
        case2: tgt_key_specs和args_index_list都不为None，抛出异常
        """

        from mx_rec.core.asc.helper import get_asc_insert_func

        with self.assertRaises(ValueError):
            get_asc_insert_func(tgt_key_specs=[], args_index_list=[])

    def test_get_asc_insert_func_case3(self):
        """
        case3: tgt_key_specs不为None时，table_names应为None，否则抛出异常
        """

        from mx_rec.core.asc.helper import get_asc_insert_func

        with self.assertRaises(RuntimeError):
            get_asc_insert_func(tgt_key_specs=[], table_names=[])

    @mock.patch.multiple("mx_rec.core.asc.helper",
                         get_asc_insert_func_inner=mock.MagicMock(return_value=Callable))
    def test_get_asc_insert_func_case4(self):
        """
        case4: tgt_key_specs不为None
        """

        from mx_rec.core.asc.helper import get_asc_insert_func

        self.assertTrue(callable(get_asc_insert_func(tgt_key_specs=[])))

    def test_get_asc_insert_func_case5(self):
        """
        case5: args_index_list不为None时，table_names应不为None，否则抛出异常
        """

        from mx_rec.core.asc.helper import get_asc_insert_func

        with self.assertRaises(RuntimeError):
            get_asc_insert_func(args_index_list=[])

    @mock.patch.multiple("mx_rec.core.asc.helper",
                         get_asc_insert_func_inner=mock.MagicMock(return_value=Callable))
    def test_get_asc_insert_func_case6(self):
        """
        case6: args_index_list和table_names都不为None
        """

        from mx_rec.core.asc.helper import get_asc_insert_func

        self.assertTrue(callable(get_asc_insert_func(args_index_list=[], table_names=["xxx"])))


class TestGetAscInsertFuncInnerFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.helper.get_asc_insert_func_inner'.
    """

    @mock.patch.multiple("mx_rec.core.asc.helper",
                         get_target_tensors_with_feature_specs=mock.MagicMock(return_value=None))
    @mock.patch("mx_rec.core.asc.helper.do_insert")
    def test_get_asc_insert_func_inner_case1(self, mock_do_insert):
        """
        case1: tgt_key_specs不为None，args_index_list为None
        """

        from mx_rec.core.asc.helper import get_asc_insert_func_inner

        with tf.Graph().as_default():
            mock_do_insert.return_value = {"xxx": tf.constant(1, dtype=tf.int64)}
            map_fn = get_asc_insert_func_inner(tgt_key_specs="xxx", dump_graph=False)
            self.assertTrue(callable(map_fn))

            dataset = generate_dataset(Config(batch_size=2, batch_number=2))
            dataset = dataset.map(map_fn)
            iterator = dataset.make_initializable_iterator()
            batch = iterator.get_next()
            with tf.Session() as sess:
                sess.run(iterator.initializer)
                sess.run(tf.compat.v1.global_variables_initializer())
                self.assertEqual(sess.run(batch.get("xxx")), 1)

    @mock.patch.multiple("mx_rec.core.asc.helper",
                         get_target_tensors_with_feature_specs=mock.MagicMock(return_value=None),
                         find_dangling_table=mock.MagicMock(return_value=["table1"]))
    @mock.patch("mx_rec.core.asc.helper.get_target_tensors_with_args_indexes")
    @mock.patch("mx_rec.core.asc.helper.do_insert")
    def test_get_asc_insert_func_inner_case2(self, mock_do_insert, mock_get_target_tensors_with_args_indexes):
        """
        case2: args_index_list不为None，tgt_key_specs为None
        """

        from mx_rec.core.asc.helper import get_asc_insert_func_inner

        with tf.Graph().as_default():
            mock_do_insert.return_value = {"xxx": tf.constant(1, dtype=tf.int64)}
            mock_get_target_tensors_with_args_indexes.return_value = [
                tf.constant([2, 1], dtype=tf.int64), tf.constant([2, 2], dtype=tf.int64)
            ]
            FeatureSpec.use_timestamp_train = True
            map_fn = get_asc_insert_func_inner(args_index_list=[0], table_names=["table1", "table2"])
            self.assertTrue(callable(map_fn))

            dataset = generate_dataset(Config(batch_size=2, batch_number=2))
            dataset = dataset.map(map_fn)
            iterator = dataset.make_initializable_iterator()
            batch = iterator.get_next()
            with tf.Session() as sess:
                sess.run(iterator.initializer)
                sess.run(tf.compat.v1.global_variables_initializer())
                self.assertEqual(sess.run(batch.get("xxx")), 1)

    @mock.patch.multiple("mx_rec.core.asc.helper",
                         get_target_tensors_with_feature_specs=mock.MagicMock(return_value=None),
                         find_dangling_table=mock.MagicMock(return_value=["table1"]))
    @mock.patch("mx_rec.core.asc.helper.get_target_tensors_with_args_indexes")
    @mock.patch("mx_rec.core.asc.helper.do_insert")
    def test_get_asc_insert_func_inner_case3(self, mock_do_insert, mock_get_target_tensors_with_args_indexes):
        """
        case3: args_index_list不为None，tgt_key_specs为None，splits小于1抛出异常
        """

        from mx_rec.core.asc.helper import get_asc_insert_func_inner

        with tf.Graph().as_default():
            mock_do_insert.return_value = {"xxx": tf.constant(1, dtype=tf.int64)}
            mock_get_target_tensors_with_args_indexes.return_value = []
            FeatureSpec.use_timestamp_train = True
            map_fn = get_asc_insert_func_inner(args_index_list=[0], table_names=[])
            self.assertTrue(callable(map_fn))

            dataset = generate_dataset(Config(batch_size=2, batch_number=2))
            with self.assertRaises(ValueError):
                dataset.map(map_fn)


class TestDoInsertFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.helper.do_insert'.
    """

    def setUp(self) -> None:
        self._dir_name = "./export_graph"

    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()
        if os.path.isdir(self._dir_name):
            shutil.rmtree(self._dir_name)

    @mock.patch.multiple("mx_rec.core.asc.helper",
                         send_feature_id_request_async=mock.MagicMock(return_value=None),
                         export_read_emb_key_v2_op=mock.MagicMock(return_value=dict()))
    def test_do_insert_case(self):
        """
        case: test do_insert
        """

        from mx_rec.core.asc.helper import do_insert

        args = dict()
        insert_tensor = []
        splits = []
        table_names = []
        input_dict = dict(dump_graph=True)

        out_batch = do_insert(args, insert_tensor, splits, table_names, input_dict)
        self.assertIsInstance(out_batch, dict)


class TestSendFeatureIdRequestAsyncFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.helper.send_feature_id_request_async'.
    """

    @mock.patch.multiple("mx_rec.core.asc.helper",
                         get_use_static=mock.MagicMock(return_value=True),
                         get_training_mode_channel_id=mock.MagicMock(return_value=0))
    @mock.patch("mx_rec.core.asc.helper.merge_feature_id_request")
    @mock.patch("mx_rec.core.asc.helper.get_host_pipeline_ops")
    def test_send_feature_id_request_async_case1(self, mock_get_host_pipeline_ops, mock_merge_feature_id_request):
        """
        case1: 静态shape
        """

        from mx_rec.core.asc.helper import send_feature_id_request_async

        with tf.Graph().as_default():
            mock_get_host_pipeline_ops.return_value = MockHostPipeLineOps()
            feature_id_list = [tf.constant([2, ], dtype=tf.int64), tf.constant([3, ], dtype=tf.int64)]
            mock_merge_feature_id_request.return_value = dict(output_feature_id_list=[feature_id_list[1]],
                                                              output_split_list=[1], output_tensorshape_split_list=[1])
            split_list = [2, 3]
            table_name_list = ["table1", "table2"]
            input_dict = dict(is_training=True, timestamp=True)

            mock_res = send_feature_id_request_async(feature_id_list, split_list, table_name_list, input_dict)
            self.assertEqual(mock_res, 0)

    @mock.patch.multiple("mx_rec.core.asc.helper",
                         get_use_static=mock.MagicMock(return_value=False),
                         get_training_mode_channel_id=mock.MagicMock(return_value=0))
    @mock.patch("mx_rec.core.asc.helper.merge_feature_id_request")
    @mock.patch("mx_rec.core.asc.helper.get_host_pipeline_ops")
    def test_send_feature_id_request_async_case2(self, mock_get_host_pipeline_ops, mock_merge_feature_id_request):
        """
        case2: 动态shape
        """

        from mx_rec.core.asc.helper import send_feature_id_request_async

        with tf.Graph().as_default():
            mock_get_host_pipeline_ops.return_value = MockHostPipeLineOps()
            feature_id_list = [tf.constant([2, ], dtype=tf.int64), tf.constant([3, ], dtype=tf.int64)]
            mock_merge_feature_id_request.return_value = dict(output_feature_id_list=[feature_id_list[1]],
                                                              output_split_list=[1], output_tensorshape_split_list=[1])
            split_list = [2, 3]
            table_name_list = ["table1", "table2"]
            input_dict = dict(is_training=True, timestamp=True)

            mock_res = send_feature_id_request_async(feature_id_list, split_list, table_name_list, input_dict)
            self.assertEqual(mock_res, 1)

    @mock.patch.multiple("mx_rec.core.asc.helper",
                         get_use_static=mock.MagicMock(return_value=False),
                         get_training_mode_channel_id=mock.MagicMock(return_value=0))
    @mock.patch("mx_rec.core.asc.helper.merge_feature_id_request")
    @mock.patch("mx_rec.core.asc.helper.get_host_pipeline_ops")
    def test_send_feature_id_request_async_case3(self, mock_get_host_pipeline_ops, mock_merge_feature_id_request):
        """
        case3: split_list或tensorshape_split_list为空，抛出异常
        """

        from mx_rec.core.asc.helper import send_feature_id_request_async

        with tf.Graph().as_default():
            mock_get_host_pipeline_ops.return_value = MockHostPipeLineOps()
            feature_id_list = [tf.constant([2, ], dtype=tf.int64), tf.constant([3, ], dtype=tf.int64)]
            mock_merge_feature_id_request.return_value = dict(output_feature_id_list=[feature_id_list[1]],
                                                              output_split_list=[], output_tensorshape_split_list=[1])
            split_list = [2, 3]
            table_name_list = ["table1", "table2"]
            input_dict = dict(is_training=True, timestamp=True)

            with self.assertRaises(RuntimeError):
                send_feature_id_request_async(feature_id_list, split_list, table_name_list, input_dict)


class TestMergeFeatureIdRequestFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.helper.merge_feature_id_request'.
    """

    def test_merge_feature_id_request_case1(self):
        """
        case1: 入参长度不等，抛出异常
        """

        from mx_rec.core.asc.helper import merge_feature_id_request

        feature_id_list = [1, 2]
        split_list = [1, 2]
        table_name_list = ["table1"]

        with self.assertRaises(RuntimeError):
            merge_feature_id_request(feature_id_list, split_list, table_name_list)

    @mock.patch.multiple("mx_rec.core.asc.helper",
                         get_modify_graph=mock.MagicMock(return_value=False))
    def test_merge_feature_id_request_case2(self):
        """
        case2: 非自动改图
        """

        from mx_rec.core.asc.helper import merge_feature_id_request

        with tf.Graph().as_default():
            feature_id_list = [
                tf.constant([2, ], dtype=tf.int64),
                tf.constant([3, ], dtype=tf.int64),
                tf.constant([4, ], dtype=tf.int64)
            ]
            split_list = [2, 3, 4]
            table_name_list = ["table1", "table2", "table1"]

            output_dict = merge_feature_id_request(feature_id_list, split_list, table_name_list)
            '''
                经过排序后：table_name_list = ["table1", "table1", "table2"]
                           split_list = [2, 4, 3]
                           feature_id_list = [tf.constant([2, ], dtype=tf.int64), 
                                              tf.constant([4, ], dtype=tf.int64), 
                                              tf.constant([3, ], dtype=tf.int64)]
                经过merge后：output_feature_id_list = [tf.constant([2, ], dtype=tf.int64), 
                                                      tf.constant([4, ], dtype=tf.int64), 
                                                      tf.constant([3, ], dtype=tf.int64)]
                            output_split_list = [6, 3]
                            output_table_name_list = ["table1", "table2"]
                            output_tensorshape_split_list = [Tensor(2), Tensor(1)]  # shape
            '''
            output_feature_id_list = [
                tf.constant([2, ], dtype=tf.int64),
                tf.constant([4, ], dtype=tf.int64),
                tf.constant([3, ], dtype=tf.int64)
            ]
            output_split_list = [6, 3]
            output_table_name_list = ["table1", "table2"]
            output_tensorshape_split_list = [
                tf.math.reduce_prod(tf.shape(output_feature_id_list[0])) + tf.math.reduce_prod(
                    tf.shape(output_feature_id_list[1])),
                tf.math.reduce_prod(tf.shape(output_feature_id_list[2]))
            ]
            self.assertListEqual(output_dict.get("output_split_list"), output_split_list)
            self.assertListEqual(output_dict.get("output_table_name_list"), output_table_name_list)

            with tf.Session() as sess:
                for real_output_feature_id, except_output_feature_id in zip(output_dict.get("output_feature_id_list"),
                                                                            output_feature_id_list):
                    self.assertListEqual(sess.run(real_output_feature_id).tolist(),
                                         sess.run(except_output_feature_id).tolist())

                for real_output_tensorshape_split, except_output_tensorshape_split in zip(
                        output_dict.get("output_tensorshape_split_list"), output_tensorshape_split_list):
                    self.assertEqual(sess.run(real_output_tensorshape_split),
                                     sess.run(except_output_tensorshape_split))

    @mock.patch.multiple("mx_rec.core.asc.helper",
                         get_modify_graph=mock.MagicMock(return_value=True))
    def test_merge_feature_id_request_case3(self):
        """
        case3: 自动改图
        """

        from mx_rec.core.asc.helper import merge_feature_id_request

        with tf.Graph().as_default():
            feature_id_list = [
                tf.constant([2, ], dtype=tf.int64),
                tf.constant([3, ], dtype=tf.int64),
                tf.constant([4, ], dtype=tf.int64)
            ]
            split_list = [2, 3, 4]
            table_name_list = ["table1", "table2", "table1"]

            # 自动改图流程和非自动改图类似，只是排序的时候只根据table_name_list排序
            output_dict = merge_feature_id_request(feature_id_list, split_list, table_name_list)
            output_feature_id_list = [
                tf.constant([2, ], dtype=tf.int64),
                tf.constant([4, ], dtype=tf.int64),
                tf.constant([3, ], dtype=tf.int64)
            ]
            output_split_list = [6, 3]
            output_table_name_list = ["table1", "table2"]
            output_tensorshape_split_list = [
                tf.math.reduce_prod(tf.shape(output_feature_id_list[0])) + tf.math.reduce_prod(
                    tf.shape(output_feature_id_list[1])),
                tf.math.reduce_prod(tf.shape(output_feature_id_list[2]))
            ]
            self.assertListEqual(output_dict.get("output_split_list"), output_split_list)
            self.assertListEqual(output_dict.get("output_table_name_list"), output_table_name_list)

            with tf.Session() as sess:
                for real_output_feature_id, except_output_feature_id in zip(output_dict.get("output_feature_id_list"),
                                                                            output_feature_id_list):
                    self.assertListEqual(sess.run(real_output_feature_id).tolist(),
                                         sess.run(except_output_feature_id).tolist())

                for real_output_tensorshape_split, except_output_tensorshape_split in zip(
                        output_dict.get("output_tensorshape_split_list"), output_tensorshape_split_list):
                    self.assertEqual(sess.run(real_output_tensorshape_split),
                                     sess.run(except_output_tensorshape_split))


class TestExportReadEmbKeyV2OpFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.helper.export_read_emb_key_v2_op'.
    """

    def test_export_read_emb_key_v2_op_case1(self):
        """
        case1: 入参args长度小于1
        """

        from mx_rec.core.asc.helper import export_read_emb_key_v2_op

        with self.assertRaises(ValueError):
            export_read_emb_key_v2_op([], "")

    def test_export_read_emb_key_v2_op_case2(self):
        """
        case2: args = ({"user_ids": tensor1, "item_ids": tensor2}, (tensor3, tensor4))，改图常见场景
            走 `if isinstance(origin_batch[0], dict)` 分支
        """

        from mx_rec.core.asc.helper import export_read_emb_key_v2_op

        with tf.Graph().as_default():
            args = (
                dict(user_ids=tf.constant(1, dtype=tf.int64), item_ids=tf.constant(1, dtype=tf.int64)),
                (tf.constant(3, dtype=tf.int64), tf.constant(3, dtype=tf.int64))
            )
            pipeline_op = tf.constant(0, dtype=tf.int64)

            output_batch = export_read_emb_key_v2_op(args, pipeline_op)
            self.assertIsInstance(output_batch, dict)
            read_emb_key = f"{sorted(output_batch)[-1]}"
            self.assertEqual(read_emb_key, "user_ids_read_emb_key")
            with tf.Session() as sess:
                self.assertEqual(sess.run(output_batch.get(read_emb_key)), 0)

    def test_export_read_emb_key_v2_op_case3(self):
        """
        case3: args = [tensor1]
            走 `elif len(origin_batch) == 1 and isinstance(origin_batch[0], tf.Tensor)` 分支
        """

        from mx_rec.core.asc.helper import export_read_emb_key_v2_op

        with tf.Graph().as_default():
            args = [tf.constant(1, dtype=tf.int64)]
            pipeline_op = tf.constant(0, dtype=tf.int64)

            # pipeline_op插入到batch的最后一个
            output_batch = export_read_emb_key_v2_op(args, pipeline_op)
            self.assertIsInstance(output_batch, tuple)
            with tf.Session() as sess:
                self.assertEqual(sess.run(output_batch[-1]), 0)

    def test_export_read_emb_key_v2_op_case4(self):
        """
        case4: args = [[tensor1], tensor2]
            走 `elif len(origin_batch) == 2` 分支
                走 `if isinstance(origin_batch[0], (list, tuple))` 分支
        """

        from mx_rec.core.asc.helper import export_read_emb_key_v2_op

        with tf.Graph().as_default():
            args = [[tf.constant(1, dtype=tf.int64)], tf.constant(2, dtype=tf.int64)]
            pipeline_op = tf.constant(0, dtype=tf.int64)

            # pipeline_op插入到batch[0]的最后一个
            output_batch = export_read_emb_key_v2_op(args, pipeline_op)
            self.assertIsInstance(output_batch, tuple)
            with tf.Session() as sess:
                self.assertEqual(sess.run(output_batch[0][-1]), 0)

    def test_export_read_emb_key_v2_op_case5(self):
        """
        case5: args = [tensor1, tensor2]
            走 `elif len(origin_batch) == 2` 分支
                走 `elif isinstance(origin_batch[0], tf.Tensor)` 分支
        """

        from mx_rec.core.asc.helper import export_read_emb_key_v2_op

        with tf.Graph().as_default():
            args = [tf.constant(1, dtype=tf.int64), tf.constant(2, dtype=tf.int64)]
            pipeline_op = tf.constant(0, dtype=tf.int64)

            # pipeline_op插入到batch[0]的最后一个
            output_batch = export_read_emb_key_v2_op(args, pipeline_op)
            self.assertIsInstance(output_batch, tuple)
            with tf.Session() as sess:
                self.assertEqual(sess.run(output_batch[0][-1]), 0)

    def test_export_read_emb_key_v2_op_case6(self):
        """
        case6: args = [set(tensor1), tensor2]，args[0]不是list、tuple、tensor，抛出异常
        """

        from mx_rec.core.asc.helper import export_read_emb_key_v2_op

        with tf.Graph().as_default():
            args = [{tf.constant(1, dtype=tf.int64)}, tf.constant(2, dtype=tf.int64)]
            pipeline_op = tf.constant(0, dtype=tf.int64)

            with self.assertRaises(EnvironmentError):
                export_read_emb_key_v2_op(args, pipeline_op)

    def test_export_read_emb_key_v2_op_case7(self):
        """
        case7: args = [tensor1, tensor2, tensor3]，pipeline_op是列表[tensor]
        """

        from mx_rec.core.asc.helper import export_read_emb_key_v2_op

        with tf.Graph().as_default():
            args = [tf.constant(1, dtype=tf.int64), tf.constant(2, dtype=tf.int64), tf.constant(3, dtype=tf.int64)]
            pipeline_op = [tf.constant(0, dtype=tf.int64)]

            # (pipeline_op,)插入到batch的最后一个
            output_batch = export_read_emb_key_v2_op(args, pipeline_op)
            self.assertIsInstance(output_batch, tuple)
            with tf.Session() as sess:
                self.assertEqual(sess.run(output_batch[-1][0]), 0)


class TestGetTargetTensorsWithArgsIndexesFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.helper.get_target_tensors_with_args_indexes'.
    """

    def test_get_target_tensors_with_args_indexes(self):
        """
        case: 此函数只能通过map_fn进行测试，否则`graph.get_tensor_by_name("args_%d:0" % index)`会直接报错
        """

        from mx_rec.core.asc.helper import get_target_tensors_with_args_indexes

        with tf.Graph().as_default():
            args_index_list = [0, 1]
            batch_size = 2
            dataset_config = Config(batch_size=batch_size, batch_number=5)

            def _map_fn(args):
                insert_tensors = get_target_tensors_with_args_indexes(args_index_list)
                args["new_item_ids"] = insert_tensors[0]
                args["new_label_0"] = insert_tensors[1]
                return args

            dataset = generate_dataset(dataset_config)
            dataset = dataset.map(_map_fn)
            '''
                原始batch:
                    batch = {"item_ids": Tensor([2, 8]), "label_0": Tensor([2, ])}
                map后batch：
                    batch = {"item_ids": Tensor([2, 8]), "label_0": Tensor([2, ]),
                             "new_item_ids": Tensor([16, ]), "new_label_0": Tensor([2, ])}
            '''
            iterator = dataset.make_initializable_iterator()
            batch = iterator.get_next()
            with tf.Session() as sess:
                sess.run(iterator.initializer)
                sess.run(tf.compat.v1.global_variables_initializer())
                self.assertEqual(sess.run(tf.shape(batch.get("new_item_ids"))),
                                 batch_size * dataset_config.item_feat_cnt)
                self.assertEqual(sess.run(tf.shape(batch.get("new_label_0"))), batch_size)


class TestGetTargetTensorsWithFeatureSpecFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.helper.get_target_tensors_with_feature_specs'.
    """

    def setUp(self):
        # 每个测试方法执行前，将FeatureSpec的静态成员设为默认值
        FeatureSpec.instance_count_train = 0
        FeatureSpec.instance_count_eval = 0
        FeatureSpec.use_timestamp_train = False
        FeatureSpec.use_timestamp_eval = False

    def test_get_target_tensors_with_feature_specs_case1(self):
        """
        case1: tgt_key_specs不属于dict、list、tuple和FeatureSpec，抛出异常
        """

        from mx_rec.core.asc.helper import get_target_tensors_with_feature_specs

        with self.assertRaises(ValueError):
            get_target_tensors_with_feature_specs(tgt_key_specs=None, batch=dict(), is_training=True,
                                                  read_emb_key_inputs_dict=dict())

    @mock.patch.multiple("mx_rec.core.asc.helper",
                         is_feature_spec_list=mock.MagicMock(return_value=True))
    @mock.patch.multiple("mx_rec.core.asc.helper.FeatureSpec",
                         set_feat_attribute=mock.MagicMock(return_value={
                             "tensor": tf.constant(1, dtype=tf.int32), "table_name": "table1", "split": 1}))
    def test_get_target_tensors_with_feature_specs_case2(self):
        """
        case2: tgt_key_specs为list或tuple，并由FeatureSpec组成，batch为dict
        """

        from mx_rec.core.asc.helper import get_target_tensors_with_feature_specs

        with tf.Graph().as_default():
            batch = {"ids1": tf.constant(1, dtype=tf.int64), "timestamp": tf.constant(2, dtype=tf.int64)}
            tgt_key_specs = [
                FeatureSpec("ids1", table_name="table1"),
                FeatureSpec("timestamp", table_name="table2", is_timestamp=True)
            ]
            is_training = True
            read_emb_key_inputs_dict = {
                "insert_tensors": [],
                "table_names": [],
                "feature_spec_names": [],
                "splits": []
            }

            get_target_tensors_with_feature_specs(tgt_key_specs, batch, is_training, read_emb_key_inputs_dict)
            self.assertListEqual(read_emb_key_inputs_dict.get("table_names"), ["table1"])
            self.assertListEqual(read_emb_key_inputs_dict.get("feature_spec_names"), [tgt_key_specs[0].name])
            self.assertListEqual(read_emb_key_inputs_dict.get("splits"), [1])
            # insert_tensors中timestamp会在第一个，如：[timestamp, ids1]
            insert_tensors = read_emb_key_inputs_dict.get("insert_tensors")
            self.assertEqual(len(insert_tensors), len(tgt_key_specs))
            with tf.Session() as sess:
                self.assertEqual(sess.run(insert_tensors[0]), 2)  # timestamp

    @mock.patch.multiple("mx_rec.core.asc.helper",
                         is_feature_spec_list=mock.MagicMock(return_value=True))
    @mock.patch.multiple("mx_rec.core.asc.helper.FeatureSpec",
                         set_feat_attribute=mock.MagicMock(return_value={
                             "tensor": tf.constant(1, dtype=tf.int64), "table_name": "table1", "split": 1}))
    def test_get_target_tensors_with_feature_specs_case3(self):
        """
        case3: tgt_key_specs为list或tuple，并由FeatureSpec组成，batch为dict
            batch中某个value不为tensor，抛出异常
        """

        from mx_rec.core.asc.helper import get_target_tensors_with_feature_specs

        with tf.Graph().as_default():
            batch = {"ids1": 1, "timestamp": tf.constant(2, dtype=tf.int64)}
            tgt_key_specs = [
                FeatureSpec("ids1", table_name="table1"),
                FeatureSpec("timestamp", table_name="table2", is_timestamp=True)
            ]
            is_training = True
            read_emb_key_inputs_dict = {
                "insert_tensors": [],
                "table_names": [],
                "feature_spec_names": [],
                "splits": []
            }

            with self.assertRaises(TypeError):
                get_target_tensors_with_feature_specs(tgt_key_specs, batch, is_training, read_emb_key_inputs_dict)

    @mock.patch.multiple("mx_rec.core.asc.helper",
                         is_feature_spec_list=mock.MagicMock(return_value=True))
    @mock.patch.multiple("mx_rec.core.asc.helper.FeatureSpec",
                         set_feat_attribute=mock.MagicMock(return_value={
                             "tensor": tf.constant(1, dtype=tf.int64), "table_name": "table1", "split": 1}))
    def test_get_target_tensors_with_feature_specs_case4(self):
        """
        case4: tgt_key_specs为list或tuple，并由FeatureSpec组成，batch为dict
            tgt_key_specs中有timestamp，但batch中没有timestamp，抛出异常
        """

        from mx_rec.core.asc.helper import get_target_tensors_with_feature_specs

        with tf.Graph().as_default():
            batch = {"ids1": tf.constant(1, dtype=tf.int64)}
            tgt_key_specs = [
                FeatureSpec("ids1", table_name="table1"),
                FeatureSpec("timestamp", table_name="table2", is_timestamp=True)
            ]
            is_training = True
            read_emb_key_inputs_dict = {
                "insert_tensors": [],
                "table_names": [],
                "feature_spec_names": [],
                "splits": []
            }

            with self.assertRaises(KeyError):
                get_target_tensors_with_feature_specs(tgt_key_specs, batch, is_training, read_emb_key_inputs_dict)

    @mock.patch.multiple("mx_rec.core.asc.helper",
                         is_feature_spec_list=mock.MagicMock(return_value=True))
    @mock.patch.multiple("mx_rec.core.asc.helper.FeatureSpec",
                         set_feat_attribute=mock.MagicMock(return_value={
                             "tensor": tf.constant(1, dtype=tf.int64), "table_name": "table1", "split": 1}))
    def test_get_target_tensors_with_feature_specs_case5(self):
        """
        case5: tgt_key_specs为list或tuple，并由FeatureSpec组成，batch为set，抛出异常
        """

        from mx_rec.core.asc.helper import get_target_tensors_with_feature_specs

        with tf.Graph().as_default():
            batch = {tf.constant(1, dtype=tf.int64)}
            tgt_key_specs = [
                FeatureSpec("ids1", table_name="table1"),
                FeatureSpec("timestamp", table_name="table2", is_timestamp=True)
            ]
            is_training = True
            read_emb_key_inputs_dict = {
                "insert_tensors": [],
                "table_names": [],
                "feature_spec_names": [],
                "splits": []
            }

            with self.assertRaises(ValueError):
                get_target_tensors_with_feature_specs(tgt_key_specs, batch, is_training, read_emb_key_inputs_dict)

    @mock.patch.multiple("mx_rec.core.asc.helper",
                         is_feature_spec_list=mock.MagicMock(return_value=True))
    @mock.patch.multiple("mx_rec.core.asc.helper.FeatureSpec",
                         set_feat_attribute=mock.MagicMock(return_value={
                             "tensor": tf.constant(1, dtype=tf.int64), "table_name": "table1", "split": 1}))
    def test_get_target_tensors_with_feature_specs_case6(self):
        """
        case6: tgt_key_specs为list或tuple，并由FeatureSpec组成，batch为list
        """

        from mx_rec.core.asc.helper import get_target_tensors_with_feature_specs

        with tf.Graph().as_default():
            batch = [tf.constant(1, dtype=tf.int64), tf.constant(2, dtype=tf.int64)]
            tgt_key_specs = [
                FeatureSpec("ids1", index_key=0, table_name="table1"),
                FeatureSpec("timestamp", index_key=1, table_name="table2", is_timestamp=True)
            ]
            is_training = True
            read_emb_key_inputs_dict = {
                "insert_tensors": [],
                "table_names": [],
                "feature_spec_names": [],
                "splits": []
            }

            get_target_tensors_with_feature_specs(tgt_key_specs, batch, is_training, read_emb_key_inputs_dict)
            self.assertListEqual(read_emb_key_inputs_dict.get("table_names"), ["table1"])
            self.assertListEqual(read_emb_key_inputs_dict.get("feature_spec_names"), [tgt_key_specs[0].name])
            self.assertListEqual(read_emb_key_inputs_dict.get("splits"), [1])
            # insert_tensors中timestamp会在第一个，如：[timestamp, ids1]
            insert_tensors = read_emb_key_inputs_dict.get("insert_tensors")
            self.assertEqual(len(insert_tensors), len(tgt_key_specs))
            with tf.Session() as sess:
                self.assertEqual(sess.run(insert_tensors[0]), 2)  # timestamp

    @mock.patch.multiple("mx_rec.core.asc.helper",
                         is_feature_spec_list=mock.MagicMock(return_value=True))
    @mock.patch.multiple("mx_rec.core.asc.helper.FeatureSpec",
                         set_feat_attribute=mock.MagicMock(return_value={
                             "tensor": tf.constant(1, dtype=tf.int64), "table_name": "table1", "split": 1}))
    def test_get_target_tensors_with_feature_specs_case7(self):
        """
        case7: tgt_key_specs为list或tuple，并由FeatureSpec组成，batch为list
            index_key的长度大于len(batch)，抛出异常
        """

        from mx_rec.core.asc.helper import get_target_tensors_with_feature_specs

        with tf.Graph().as_default():
            batch = [tf.constant(1, dtype=tf.int64), tf.constant(2, dtype=tf.int64)]
            tgt_key_specs = [
                FeatureSpec("ids1", index_key=3, table_name="table1"),
                FeatureSpec("timestamp", index_key=1, table_name="table2", is_timestamp=True)
            ]
            is_training = True
            read_emb_key_inputs_dict = {
                "insert_tensors": [],
                "table_names": [],
                "feature_spec_names": [],
                "splits": []
            }

            with self.assertRaises(ValueError):
                get_target_tensors_with_feature_specs(tgt_key_specs, batch, is_training, read_emb_key_inputs_dict)

    @mock.patch.multiple("mx_rec.core.asc.helper",
                         is_feature_spec_list=mock.MagicMock(return_value=True))
    @mock.patch.multiple("mx_rec.core.asc.helper.FeatureSpec",
                         set_feat_attribute=mock.MagicMock(return_value={
                             "tensor": tf.constant(1, dtype=tf.int64), "table_name": "table1", "split": 1}))
    def test_get_target_tensors_with_feature_specs_case8(self):
        """
        case8: tgt_key_specs为list或tuple，并由FeatureSpec组成，batch为list
            batch中不为tensor，抛出异常
        """

        from mx_rec.core.asc.helper import get_target_tensors_with_feature_specs

        with tf.Graph().as_default():
            batch = [1, tf.constant(2, dtype=tf.int64)]
            tgt_key_specs = [
                FeatureSpec("ids1", index_key=0, table_name="table1"),
                FeatureSpec("timestamp", index_key=1, table_name="table2", is_timestamp=True)
            ]
            is_training = True
            read_emb_key_inputs_dict = {
                "insert_tensors": [],
                "table_names": [],
                "feature_spec_names": [],
                "splits": []
            }

            with self.assertRaises(TypeError):
                get_target_tensors_with_feature_specs(tgt_key_specs, batch, is_training, read_emb_key_inputs_dict)

    @mock.patch.multiple("mx_rec.core.asc.helper",
                         is_feature_spec_list=mock.MagicMock(return_value=True))
    @mock.patch.multiple("mx_rec.core.asc.helper.FeatureSpec",
                         set_feat_attribute=mock.MagicMock(return_value={
                             "tensor": tf.constant(1, dtype=tf.int64), "table_name": "table1", "split": 1}))
    def test_get_target_tensors_with_feature_specs_case9(self):
        """
        case9: tgt_key_specs为dict，并由FeatureSpec组成，batch为dict
        """

        from mx_rec.core.asc.helper import get_target_tensors_with_feature_specs

        with tf.Graph().as_default():
            batch = {
                "ids1": {"ids1": tf.constant(1, dtype=tf.int64), "timestamp": tf.constant(2, dtype=tf.int64)},
                "timestamp": {"ids1": tf.constant(1, dtype=tf.int64), "timestamp": tf.constant(2, dtype=tf.int64)}
            }
            tgt_key_specs = {
                "ids1": FeatureSpec("ids1", table_name="table1"),
                "timestamp": FeatureSpec("timestamp", table_name="table1", is_timestamp=True),
            }
            is_training = True
            read_emb_key_inputs_dict = {
                "insert_tensors": [],
                "table_names": [],
                "feature_spec_names": [],
                "splits": []
            }

            get_target_tensors_with_feature_specs(tgt_key_specs, batch, is_training, read_emb_key_inputs_dict)
            self.assertListEqual(read_emb_key_inputs_dict.get("table_names"), ["table1"])
            self.assertListEqual(read_emb_key_inputs_dict.get("feature_spec_names"), [tgt_key_specs.get("ids1").name])
            self.assertListEqual(read_emb_key_inputs_dict.get("splits"), [1])
            # insert_tensors中timestamp会在第一个，如：[timestamp, ids1]
            insert_tensors = read_emb_key_inputs_dict.get("insert_tensors")
            self.assertEqual(len(insert_tensors), len(tgt_key_specs))
            with tf.Session() as sess:
                self.assertEqual(sess.run(insert_tensors[0]), 2)  # timestamp

    @mock.patch.multiple("mx_rec.core.asc.helper",
                         is_feature_spec_list=mock.MagicMock(return_value=True))
    @mock.patch.multiple("mx_rec.core.asc.helper.FeatureSpec",
                         set_feat_attribute=mock.MagicMock(return_value={
                             "tensor": tf.constant(1, dtype=tf.int64), "table_name": "table1", "split": 1}))
    def test_get_target_tensors_with_feature_specs_case10(self):
        """
        case10: tgt_key_specs为list或tuple，并由FeatureSpec组成，batch也为list或tuple
        """

        from mx_rec.core.asc.helper import get_target_tensors_with_feature_specs

        with tf.Graph().as_default():
            batch = [tf.constant(1, dtype=tf.int64), tf.constant(2, dtype=tf.int64)]
            tgt_key_specs = [
                FeatureSpec("ids1", index_key=0, table_name="table1"),
                FeatureSpec("timestamp", index_key=1, table_name="table2", is_timestamp=True)
            ]
            is_training = True
            read_emb_key_inputs_dict = {
                "insert_tensors": [],
                "table_names": [],
                "feature_spec_names": [],
                "splits": []
            }

            get_target_tensors_with_feature_specs(tgt_key_specs, batch, is_training, read_emb_key_inputs_dict)
            self.assertListEqual(read_emb_key_inputs_dict.get("table_names"), ["table1"])
            self.assertListEqual(read_emb_key_inputs_dict.get("feature_spec_names"), [tgt_key_specs[0].name])
            self.assertListEqual(read_emb_key_inputs_dict.get("splits"), [1])
            # insert_tensors中timestamp会在第一个，如：[timestamp, ids1]
            insert_tensors = read_emb_key_inputs_dict.get("insert_tensors")
            self.assertEqual(len(insert_tensors), len(tgt_key_specs))
            with tf.Session() as sess:
                self.assertEqual(sess.run(insert_tensors[0]), 2)  # timestamp


class TestIsFeatureSpecListFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.helper.is_feature_spec_list'.
    """

    def test_is_feature_spec_list_case1(self):
        """
        case1: specs不为list或tuple，返回False
        """

        from mx_rec.core.asc.helper import is_feature_spec_list

        self.assertFalse(is_feature_spec_list(dict()))

    def test_is_feature_spec_list_case2(self):
        """
        case2: specs为list或tuple，但不为FeatureSpec，返回False
        """

        from mx_rec.core.asc.helper import is_feature_spec_list

        self.assertFalse(is_feature_spec_list(["xxx"]))

    def test_is_feature_spec_list_case3(self):
        """
        case3: specs为list或tuple，且为FeatureSpec，返回True
        """

        from mx_rec.core.asc.helper import is_feature_spec_list

        self.assertTrue(is_feature_spec_list([FeatureSpec("xxx")]))


if __name__ == '__main__':
    unittest.main()
