#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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
import time
from dataclasses import dataclass
import csv
import logging

import pytest
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

tf.compat.v1.disable_eager_execution()
if tf.__version__.startswith("1"):
    from npu_bridge.npu_init import NPURunConfig, NPUEstimator, npu_hooks_append, DumpConfig
else:
    import npu_device

mx_rec_package_path = os.getenv('MX_REC_PACKAGE_PATH')
if not mx_rec_package_path:
    raise EnvironmentError("please set env MX_REC_PACKAGE_PATH first")

op_lib_path = os.path.join(mx_rec_package_path, "libasc", "libasc_ops.so")
tfOpLib = tf.load_op_library(op_lib_path)


@dataclass
class CollectInfo:
    batch_size: int
    table_size: int
    dim_size: int
    lr: float
    weight_decay: float
    golden_avg: float
    npu_avg: float
    time_diff: float
    speedup: float
    res: bool


def write_to_csv(collect_info: CollectInfo):
    # 定义 CSV 文件名和表头
    csv_file = "performance_report.csv"
    headers = [
        "batch_size", "table_size", "dim_size", "lr", 
        "weight_decay", "golden_avg(s)", "npu_avg(s)", 
        "time_diff(s)", "speedup(x)", "compare_result"
    ]
    
    # 检查文件是否存在，不存在则写入表头
    file_exists = os.path.isfile(csv_file)

    if collect_info.weight_decay is None:
        collect_info.weight_decay = 0.0
    
    # 以追加模式打开文件
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writerow(headers)
        
        # 写入性能数据
        writer.writerow([
            collect_info.batch_size,
            collect_info.table_size,
            collect_info.dim_size,
            collect_info.lr,
            collect_info.weight_decay,
            f"{collect_info.golden_avg:.6f}",  # 保留6位小数
            f"{collect_info.npu_avg:.6f}",
            f"{collect_info.time_diff:.6f}",
            f"{collect_info.speedup:.2f}",
            f"{collect_info.res}"
        ])


def generate_unique_mask_array(batch_size, table_size):
    # 生成 [0, table_size) 范围内的不重复随机索引
    indices = np.random.choice(
        np.arange(table_size, dtype=np.int32),  # 索引范围
        size=batch_size,                        # 生成数量 = batch_size
        replace=False                           # 强制不重复[1,3,6](@ref)
    )
    
    # 调整形状为 (batch_size, 1) 并指定 dtype=int32
    return indices.reshape(-1, 1).astype(np.int32)


def gen_test_data(batch_size, table_size, dim_size, lr):
    # 使用Numpy的固定随机种子保证可复现性
    np.random.seed(42)
    grad = np.random.uniform(0.0, 1.0, size=(batch_size, dim_size)).astype(np.float32)
    var = np.random.uniform(0.0, 1.0, size=(table_size, dim_size)).astype(np.float32)
    lr_tensor = np.full((batch_size, dim_size), lr, dtype=np.float32)
    indices = generate_unique_mask_array(batch_size, table_size)

    return grad, indices, var, lr_tensor


def run_sgd_npu(grad, indices, var, lr, weight_decay):
    result = tfOpLib.sgd(grad, indices, var, lr, weight_decay)
    return result


def run_sgd_gloden(grad, indices, var, lr, weight_decay):
    if weight_decay is None:
        nd_value = grad * lr
    else:
        nd_value = (grad + weight_decay * tf.gather(var, tf.squeeze(indices, axis=1))) * lr
    var_update_op = tf.scatter_nd_add(var, indices, -nd_value)
    return var_update_op


@pytest.fixture(scope="session")
def tf_session():
    logging.getLogger().setLevel(logging.INFO)

    device_id = 4

    os.environ["DEVICE_ID"] = str(0)
    os.environ["ASCEND_DEVICE_ID"] = str(device_id)
    os.environ["JOB_ID"] = "10086"
    # 初始化 Session 配置
    config = tf.compat.v1.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    
    # 创建 Session
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.global_variables_initializer().run(session=sess)
    
    yield sess  # 将会话传递给测试用例
    
    # 清理资源
    sess.close()


@pytest.fixture(scope="function")
def static_graph():
    grad_ph = tf.placeholder(tf.float32, shape=[None, None], name="global_grad")
    indices_ph = tf.placeholder(tf.int32, shape=[None, 1], name="global_indices")
    lr_ph = tf.placeholder(tf.float32, shape=[None, None], name="global_lr")
    return grad_ph, indices_ph, lr_ph


@pytest.fixture(scope="function")
def tf_vars(tf_session, batch_size, table_size, dim_size, lr):
    input_grad, input_indices, input_var, input_lr = gen_test_data(batch_size, table_size, dim_size, lr)

    with tf.variable_scope("npu_vars"):
        var_npu = tf.Variable(input_var, dtype=tf.float32, trainable=False)
    with tf.variable_scope("golden_vars"): 
        var_golden = tf.Variable(input_var, dtype=tf.float32, trainable=False)

    tf_session.run(tf.variables_initializer([var_npu, var_golden]))
    return var_npu, var_golden, input_grad, input_indices, input_lr


@pytest.mark.parametrize("batch_size, table_size", [(b, 5 * b) for b in [1, 15, 100, 1000]])
@pytest.mark.parametrize("dim_size", [32, 88, 128, 1024])
@pytest.mark.parametrize("lr", [0.1, 1.2])
@pytest.mark.parametrize("weight_decay", [None, 0.012])
def test_sgd(tf_session, static_graph, tf_vars, batch_size, table_size, dim_size, lr, weight_decay):
    grad_ph, indices_ph, lr_ph = static_graph
    var_npu, var_golden, input_grad, input_indices, input_lr = tf_vars

     # 动态绑定维度
    grad = tf.reshape(grad_ph, [batch_size, dim_size])
    indices = tf.reshape(indices_ph, [batch_size, 1])
    lr_tensor = tf.reshape(lr_ph, [batch_size, dim_size])

    # 构建计算图
    golden_out = run_sgd_gloden(grad, indices, var_golden, input_lr, weight_decay)
    npu_out = run_sgd_npu(grad, indices, var_npu, input_lr, weight_decay)

    # 预热10次
    for _ in range(10):
        results = tf_session.run({
            "npu_result": npu_out,
            "golden_result": golden_out,
            "var_npu": var_npu,
            "var_golden": var_golden
        }, feed_dict={
            grad: input_grad,
            indices: input_indices,
            lr_tensor: input_lr
        })

    golden_times = []
    for _ in range(10):
        start = time.perf_counter()
        tf_session.run({"golden_result": golden_out, "var_golden": var_golden}, 
            feed_dict={grad: input_grad, indices: input_indices, lr_tensor: input_lr})
        golden_times.append(time.perf_counter() - start)
    golden_avg = np.mean(golden_times)

    npu_times = []
    for _ in range(10):
        start = time.perf_counter()
        tf_session.run({"npu_result": npu_out, "var_npu": var_npu}, 
            feed_dict={grad: input_grad, indices: input_indices, lr_tensor: input_lr})
        npu_times .append(time.perf_counter() - start)
    npu_avg = np.mean(npu_times)

    res = np.allclose(results["golden_result"], results["npu_result"], 1e-4, 1e-4)

    collect_info = CollectInfo(
        batch_size, table_size, dim_size, 
        lr, weight_decay, golden_avg, 
        npu_avg, golden_avg - npu_avg, 
        golden_avg / npu_avg, res
    )

    write_to_csv(collect_info)
    assert res

## pytest -v -s test.py 直接执行 对比精度和性能
## msprof --application="pytest test.py" --output=prof 用msprof工具测量性能