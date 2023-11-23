#!/usr/bin/env python3
# coding: UTF-8
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
import os
import sys
import unittest
from dataclasses import dataclass
from unittest import mock

import tensorflow as tf

from tests.mx_rec.core.mxrec_pybind_mock import MxRecPybindMock
from tests.mx_rec.core.initializer_mock import InitializerMock
from mx_rec.util.tf_version_adapter import npu_ops

sys.modules['mxrec_pybind'] = MxRecPybindMock
sys.modules['mx_rec.util.initialize'] = InitializerMock

os.environ[
    "HOST_PIPELINE_OPS_LIB_PATH"] = f"{os.getenv('so_path')}/libasc/libasc_ops.so"


@dataclass
class InputConfig:
    batch_size: int
    feat_cnt: int
    send_count: int
    rank_size: int
    channel_id: int
    table_name: str
    skip_emb_transfer: bool
    ext_emb_size: int
    emb_size: int
    use_hot: bool
    device_id: int
    use_dynamic_expansion: bool


class TestBuildGraph(unittest.TestCase):
    """
    Test Suite for Exception Checkpoint.
    """

    def setUp(self):
        """
        准备步骤
        :return:无
        """
        super().setUp()

    def tearDown(self):
        """
        销毁步骤
        :return: 无
        """
        super().tearDown()

    @staticmethod
    def get_next_mock():
        return tf.constant(value=1, name="inference/asecnd_lookup_one_big_embedding/all2all/mul", shape=[8, 8],
                           dtype=tf.int64)

    @staticmethod
    def get_id_offsets_mock():
        return tf.constant(value=1, shape=[270412, 8], dtype=tf.float32, name="inference/gather_for_id_offsets"), [], 0

    @staticmethod
    def get_all2all_mock():
        return tf.constant(value=1, shape=[8, ], dtype=tf.int64, name="mul")

    @staticmethod
    def get_restore_vector_mock():
        return [tf.constant(value=1, shape=[2908800], dtype=tf.int32, name="aicpu_getnext_restore_vector/GetNext"),
                None]

    @staticmethod
    def get_input_config(input_config_init: InputConfig):
        batch_size = input_config_init.batch_size
        feat_cnt = input_config_init.feat_cnt
        send_count = input_config_init.send_count
        rank_size = input_config_init.rank_size
        channel_id = input_config_init.channel_id
        table_name = input_config_init.table_name
        skip_emb_transfer = input_config_init.skip_emb_transfer
        ext_emb_size = input_config_init.ext_emb_size
        emb_size = input_config_init.emb_size
        use_hot = input_config_init.use_hot
        device_id = input_config_init.device_id
        use_dynamic_expansion = input_config_init.use_dynamic_expansion

        input_config = {'batch_size': batch_size,
                        'feat_cnt': feat_cnt,
                        'send_count': send_count,
                        'rank_size': rank_size,
                        'channel_id': channel_id,
                        'table_name': table_name,
                        'skip_emb_transfer': skip_emb_transfer,
                        'ext_emb_size': ext_emb_size,
                        'emb_size': emb_size,
                        'use_hot': use_hot,
                        'device_id': device_id,
                        'use_dynamic_expansion': use_dynamic_expansion}
        return input_config

    @staticmethod
    def get_input_table():
        input_table = tf.Variable(tf.zeros([875000, 8]), name="inference/one_ascend_hash_embedding:0",
                                  dtype=tf.float32)
        return input_table

    @mock.patch('npu_bridge.estimator.npu_ops')
    @mock.patch("npu_bridge.hccl.hccl_ops")
    @mock.patch("npu_bridge.estimator.npu.npu_hook.NPUCheckpointSaverHook")
    def test_get_restore_vector_use_hot(self, tf1_npu_ops_mock, tf1_hccl_ops_mock,
                                        tf1_save_mock):
        from mx_rec.core.asc.build_graph import get_restore_vector
        tf1_npu_ops_mock.return_value = None
        tf1_hccl_ops_mock.return_value = None
        tf1_save_mock.return_value = None
        input_config_instance = InputConfig(9600, 1, None, 8, 0, "one_ascend_hash_embedding", True, 8, "8", True, 6,
                                            False)
        input_config = self.get_input_config(input_config_instance)
        try:
            get_restore_vector(input_config)
        except TypeError as exp:
            self.assertEqual(type(exp), TypeError)
        else:
            self.fail("TypeError not raised.")

    @mock.patch("npu_bridge.hccl.hccl_ops")
    @mock.patch('npu_bridge.estimator.npu_ops')
    @mock.patch("npu_bridge.estimator.npu.npu_hook.NPUCheckpointSaverHook")
    def test_get_restore_vector_emb_size_value_error(self, tf1_hccl_ops_mock, tf1_npu_ops_mock,
                                                     tf1_save_mock):
        from mx_rec.core.asc.build_graph import get_restore_vector
        tf1_save_mock.return_value = None
        tf1_npu_ops_mock.return_value = None
        tf1_hccl_ops_mock.return_value = None
        input_config_instance = InputConfig(9600, 1, None, 8, 0, "one_ascend_hash_embedding", True, 8, -1, True, 6,
                                            False)
        input_config = self.get_input_config(input_config_instance)
        try:
            get_restore_vector(input_config)
        except TypeError as exp:
            self.assertEqual(type(exp), TypeError)
        else:
            self.fail("ValueError not raised.")

    @mock.patch('npu_bridge.estimator.npu_ops')
    @mock.patch("npu_bridge.hccl.hccl_ops")
    @mock.patch("npu_bridge.estimator.npu.npu_hook.NPUCheckpointSaverHook")
    def test_get_restore_vector_ext_emb_size_type_error(self, tf1_npu_ops_mock, tf1_hccl_ops_mock,
                                                        tf1_save_mock):
        from mx_rec.core.asc.build_graph import get_restore_vector
        tf1_npu_ops_mock.return_value = None
        tf1_hccl_ops_mock.return_value = None
        tf1_save_mock.return_value = None
        input_config_instance = InputConfig(9600, 1, None, 8, 0, "one_ascend_hash_embedding", False, "8", 8, True, 6,
                                            False)
        input_config = self.get_input_config(input_config_instance)
        try:
            get_restore_vector(input_config)
        except TypeError as exp:
            self.assertEqual(type(exp), TypeError)
        else:
            self.fail("TypeError not raised.")

    @mock.patch('npu_bridge.estimator.npu_ops')
    @mock.patch("npu_bridge.hccl.hccl_ops")
    @mock.patch("npu_bridge.estimator.npu.npu_hook.NPUCheckpointSaverHook")
    def test_get_restore_vector_ext_emb_size_value_error(self, tf1_npu_ops_mock, tf1_hccl_ops_mock,
                                                         tf1_save_mock):
        from mx_rec.core.asc.build_graph import get_restore_vector
        tf1_npu_ops_mock.return_value = None
        tf1_hccl_ops_mock.return_value = None
        tf1_save_mock.return_value = None
        input_config_instance = InputConfig(9600, 1, None, 8, 0, "one_ascend_hash_embedding", False, -1, 8, True, 6,
                                            False)
        input_config = self.get_input_config(input_config_instance)
        try:
            get_restore_vector(input_config)
        except TypeError as exp:
            self.assertEqual(type(exp), TypeError)
        else:
            self.fail("ValueError not raised.")

    @mock.patch("npu_bridge.hccl.hccl_ops")
    @mock.patch("npu_bridge.estimator.npu.npu_hook.NPUCheckpointSaverHook")
    def test_get_id_offsets(self, tf1_hccl_ops_mock,
                            tf1_save_mock):
        with mock.patch.object(npu_ops, "gen_npu_ops") as mock_npu_ops:
            from mx_rec.core.asc.build_graph import get_id_offsets
            id_offset_mock = tf.constant(value=1, shape=[270412, 8], dtype=tf.float32,
                                         name="inference/gather_for_id_offsets")
            mock_npu_ops.get_next.return_value = [id_offset_mock]
            tf1_hccl_ops_mock.return_value = None
            tf1_save_mock.return_value = None
            input_config_instance = InputConfig(9600, 1, None, 8, 0, "one_ascend_hash_embedding", True, 8, 8, True, 6,
                                                False)
            input_config = self.get_input_config(input_config_instance)
            max_lookup_vec_size = None
            res_id_offsets = get_id_offsets(max_lookup_vec_size, input_config)
            self.assertEqual(res_id_offsets[0], id_offset_mock)

    @mock.patch("npu_bridge.hccl.hccl_ops")
    @mock.patch("npu_bridge.estimator.npu.npu_hook.NPUCheckpointSaverHook")
    def test_get_all2all_args(self, tf1_hccl_ops_mock,
                              tf1_save_mock):
        with mock.patch.object(npu_ops, "gen_npu_ops") as mock_npu_ops:
            from mx_rec.core.asc.build_graph import get_all2all_args
            all2all_mock = tf.constant(
                value=1,
                name='mul',
                shape=[8, 8], dtype=tf.int64)
            mock_npu_ops.get_next.return_value = all2all_mock
            tf1_hccl_ops_mock.return_value = None
            tf1_save_mock.return_value = None
            input_config_instance = InputConfig(9600, 1, None, 8, 0, "one_ascend_hash_embedding", True, 8, 8, True, 6,
                                                False)
            input_config = self.get_input_config(input_config_instance)
            use_static = False
            res_all2all_args = get_all2all_args(use_static, input_config)
            self.assertEqual(res_all2all_args.shape, tf.constant(value=1, shape=[8, ], dtype=tf.int64,
                                                                 name="mul").shape)
            self.assertEqual(res_all2all_args.dtype, tf.constant(value=1, shape=[8, 8], dtype=tf.int64,
                                                                 name="mul").dtype)



if __name__ == '__main__':
    unittest.main()
