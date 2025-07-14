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

from typing import List, Union
import copy
import csv
import os
import shutil
import inspect

import toml
import torch
import torch.utils.benchmark as benchmark
import torch_npu

from utils.logger import default_logger
from pattern.constant import TORCHINDUCTOR_CACHE_ENV

STEP_COL_NAME = 'Step'
OP_TIME_COL_NAME = 'Computing'
PROF_STEP_DATA_FILE_NAME = "step_trace_time.csv"

g_prof_config = torch_npu.profiler._ExperimentalConfig(
    export_type=[
        torch_npu.profiler.ExportType.Text,
        torch_npu.profiler.ExportType.Db
        ],
    profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
    msprof_tx=False,
    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    l2_cache=False,
    op_attr=False,
    data_simplification=False,
    record_op_args=False,
    gc_detect_threshold=None)


def load_config():
    """
    加载配置文件。

    该函数通过读取当前文件所在目录下的 'config.toml' 文件来加载配置信息。
    使用 TOML 格式解析配置文件，并返回解析后的配置字典。

    Returns:
        dict: 解析后的配置文件内容，以字典形式返回。
    """
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)

    # 获取当前文件所在的目录
    current_directory = os.path.dirname(current_file_path)

    # 通过调用 toml.load 加载配置文件并返回配置信息
    config = toml.load(os.path.join(current_directory, "config.toml"))
    return config


def get_caller_module_name(level=4):
    """
    获取调用者模块的文件名。

    通过inspect模块获取调用者的信息，并使用os.path模块获取文件名部分。

    参数:
    level (int): 调用栈的深度，用于确定调用者。默认值为4。

    返回:
    str: 调用者模块的文件名。
    """
    # 获取调用者的帧对象，level用于指定调用栈的深度
    caller_frame = inspect.stack()[level]
    # 从帧对象中获取文件名，并使用os.path.basename移除路径，只保留文件名
    return os.path.basename(caller_frame.filename)


def perform_test(model: torch.nn.Module, input_list: list):
    """
    根据配置文件执行精度测试或性能测试。

    Args:
        model (torch.nn.Module): 要测试的模型。
        input_list (list): 输入数据列表。
    """
    config = load_config()
    if config["test_config"]["mode"] == "precision":
        model_list = [model, copy.deepcopy(model)]
        precision_test(model_list, input_list, config)

    elif config["test_config"]["mode"] == "performance":
        performance_test(model, input_list, config)
    else:
        raise ValueError("mode must be 'precision' or 'performance'")


def log_avg_time(config: dict, module_name: str, avg_time: float):
    """
    打印平均运行时间。

    参数:
    - config (dict): 包含模型配置信息的字典，包括设备信息、模式等。
    - avg_time (float): 模型平均耗时。
    """
    default_logger.info(
        "[{}] avg time over {} with {} mode: {}".format(
            module_name,
            config["model_config"]["device"],
            config["model_config"]["mode"],
            avg_time
        )
    )


def find_file(directory: str, filename: str) -> str:
    """
    指定目录下递归搜索文件路径。

    参数:
    - directory (str): 需要递归搜索的目录路径
    - filename (str): 待搜索的文件名
    """
    for root, _, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return ""


def get_op_avg_time(prof_output_dir: str, warmup_step_num: int, exec_step_num: int) -> float:
    """
    从指定的目录中读取算子耗时数据，并计算平均耗时。

    参数:
    - prof_output_dir (str): profile输出的目录
    - warmup_step_num (int): warmup执行的次数
    - exec_step_num (int): 待求平均的执行耗时

    返回:
    - float: 算子平均耗时
    """
    row_num = 0
    op_time_sum = 0
    prof_file = find_file(prof_output_dir, PROF_STEP_DATA_FILE_NAME)
    if prof_file == "":
        raise ValueError(f"未找到文件：{PROF_STEP_DATA_FILE_NAME}，profile数据生成异常")

    default_logger.info(f"找到prof文件路径：{prof_file}")
    # 读取CSV文件
    with open(prof_file, 'r') as file:
        # 创建CSV读取器
        csv_reader = csv.DictReader(file)
        # 读取数据行
        for row in csv_reader:
            if (int(row[STEP_COL_NAME]) < warmup_step_num):
                continue
            row_num += 1 # 统计行数
            op_time_sum += float(row[OP_TIME_COL_NAME]) # 累加算子耗时

        if (row_num != exec_step_num):
            raise ValueError(f"step数量：{exec_step_num}，实际统计数量：{row_num}，profile数据生成异常")

        return op_time_sum / row_num


def performance_test_on_npu(model: torch.nn.Module, input_list: list, config: dict, module_name: str):
    """
    测试模型在npu上的性能。

    此函数重点关注npu侧算子运行的总时间。

    参数:
    - model (torch.nn.Module): 需要测试性能的模型。
    - input_list (list): 模型的输入数据列表。
    - config (dict): 包含模型配置信息的字典，包括设备信息、模式等。

    返回:
    无返回值，直接打印模型在指定设备和模式下的平均运行时间。
    """
    # 切换为评估模式
    model.eval()
    with torch.no_grad():
        model(*input_list) # 提前跑一次，确保模型能正常执行，否则导致profiler异常

    warmup_step_num = int(config["test_config"]["warmup_step_num"])
    exec_step_num = int(config["test_config"]["exec_step_num"])
    prof_output_dir = os.path.join(config["test_config"]["prof_dir"], module_name + ".prof")
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
            ],
        # wait必须设置为0，repeat必须设置为1，使profiler期望执行次数与实际执行次数匹配；
        # 否则profiler对象无法正常析构，再次创建profiler将导致功能异常
        schedule=torch_npu.profiler.schedule(wait=0, warmup=warmup_step_num, active=exec_step_num, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_output_dir),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        with_flops=False,
        experimental_config=g_prof_config)

    # 执行推理
    all_step_num = warmup_step_num + exec_step_num
    prof.start()
    with torch.no_grad():
        for _ in range(all_step_num):
            model(*input_list)
            prof.step()
    prof.stop()

    # 读取算子耗时
    op_avg_time = get_op_avg_time(prof_output_dir, warmup_step_num, exec_step_num)
    # 打印平均运行时间
    log_avg_time(config, module_name, op_avg_time)


def performance_test(model: torch.nn.Module, input_list: list, config: dict):
    """
    测试给定模型的性能。

    将模型和输入数据移动到指定设备上，根据配置信息中的模式(graph 或 eager)进行模型的编译或直接运行，
    并测量模型的平均运行时间。

    参数:
    - model (torch.nn.Module): 需要测试性能的模型。
    - input_list (list): 模型的输入数据列表。
    - config (dict): 包含模型配置信息的字典，包括设备信息、模式等。

    返回:
    无返回值，直接打印模型在指定设备和模式下的平均运行时间。
    """
    # 将输入列表中的每个元素移动到指定设备
    input_list = [
        move_to_device(i, config["model_config"]["device"]) for i in input_list
    ]
    # 将模型移动到指定设备
    model = model.to(config["model_config"]["device"])

    # 根据配置信息中的模式进行模型编译或直接运行
    if config["model_config"]["mode"] == "graph":
        # 使用torch.compile编译模型，以提高性能
        model = torch.compile(
            model,
            backend=config["model_config"]["compile_backend"],
            mode=config["model_config"]["compile_mode"],
            dynamic=config["model_config"]["compile_dynamic"],
            fullgraph=config["model_config"]["compile_fullgraph"],
        )
    elif config["model_config"]["mode"] == "eager":
        # 如果模式为'eager'，则不进行编译
        pass
    else:
        # 如果模式不是'graph'或'eager'，抛出异常
        raise ValueError("mode must be 'graph' or 'eager'")

    module_name = get_caller_module_name()

    # npu采用算子耗时的统计方法，与其他硬件不同
    if (config["model_config"]["device"] == "npu"):
        performance_test_on_npu(model, input_list, config, module_name)
        return

    # 准备benchmark需要的全局变量
    timer_globals = {
        "model": model,
        "input_list": input_list,
    }

    # 创建benchmark计时器
    t0 = benchmark.Timer(stmt="model(*input_list)", globals=timer_globals)
    # 预热
    t0.timeit(config["test_config"]["warmup_step_num"])
    # 运行计时测试
    m = t0.timeit(config["test_config"]["exec_step_num"])
    # 打印平均运行时间
    log_avg_time(config, module_name, m.mean)


def precision_test(model_list: List[torch.nn.Module], input_list: list, config: dict):
    """
    精度测试函数，用于比较不同设备上模型的输出精度。

    参数:
    model_list: 包含两个模型的列表，这两个模型将被分别部署在不同的设备上进行测试。
    input_list: 模型输入数据的列表，这些数据将被分别发送到不同的设备上。
    config: 测试配置字典，包含精度测试的配置信息，如设备类型和编译配置。

    该函数没有返回值，但会打印出精度测试的结果，包括标准差(std)和均方根误差(rmse)。
    """
    # 获取精度测试的配置信息
    golden = config["test_config"]["precision"]["golden"]
    target = config["test_config"]["precision"]["target"]
    device_list = [golden, target]

    # 将模型移动到指定的设备上
    model_list = [model.to(device) for model, device in zip(model_list, device_list)]

    # 如果模式为"graph"，则编译模型
    if config["model_config"]["mode"] == "graph":
        model_list = [
            torch.compile(
                model_list[i],
                backend=config["model_config"]["compile_backend"],
                mode=config["model_config"]["compile_mode"],
                dynamic=config["model_config"]["compile_dynamic"],
                fullgraph=config["model_config"]["compile_fullgraph"],
            )
            for i in range(2)
        ]

    # 初始化输出列表
    output_list = []

    # 在每个设备上运行模型并收集输出
    for model, device in zip(model_list, device_list):
        input_list = move_to_device(input_list, device)
        output_list.append(model(*input_list))

    # 将输出拆分为golden输出和target输出，并确保它们都是元组形式
    golden_output, target_output = output_list
    if not isinstance(golden_output, tuple):
        golden_output = (golden_output,)
    if not isinstance(target_output, tuple):
        target_output = (target_output,)

    # 初始化统计字典
    stastics = dict()

    # 计算并记录每个输出的精度指标
    for g, t in zip(golden_output, target_output):
        g = g.to("cpu")
        t = t.to("cpu")
        stastics["std"] = torch.std(g - t).item()
        stastics["rmse"] = torch.sqrt(torch.nn.functional.mse_loss(t, g)).item()

    # 打印精度测试的结果
    module_name = get_caller_module_name()
    default_logger.info("[%s] Accuracy test result: %s", module_name, stastics)


def move_to_device(
    input_tensor: Union[list, torch.Tensor], device: str
) -> Union[list, torch.Tensor]:
    """
    将输入的数据移动到指定的设备上。

    该函数支持输入为列表或torch.Tensor类型，将输入数据移动到指定的计算设备（如NPU）上。
    如果输入是列表，函数将递归地将列表中的每个元素移动到指定设备。如果输入是torch.Tensor，
    函数将直接将其移动到指定设备。

    参数:
    input (Union[list, torch.Tensor]): 输入数据，可以是列表或torch.Tensor类型。
    device (str): 设备标识符，指定要将输入数据移动到的目标设备。

    返回:
    Union[list, torch.Tensor]: 移动到指定设备后的输入数据。
    """
    # 如果输入是列表，则递归调用move_to_device函数处理列表中的每个元素
    if isinstance(input_tensor, list):
        return [move_to_device(x, device) for x in input_tensor]
    # 如果输入是torch.Tensor，直接调用其to方法将其移动到指定设备
    return input_tensor.to(device)


def clean_up_inductor_cache():
    """
    清理缓存目录。

    该函数检查环境中是否存在 TORCHINDUCTOR_CACHE_ENV 变量，并删除该变量指定的缓存目录。
    """
    # 检查环境变量中是否存在缓存目录路径
    if TORCHINDUCTOR_CACHE_ENV in os.environ:
        # 删除缓存目录及其内容
        shutil.rmtree(os.environ[TORCHINDUCTOR_CACHE_ENV])
