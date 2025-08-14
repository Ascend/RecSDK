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

import sys
import os
from decimal import Decimal
from dataclasses import dataclass
import ijson

from .logger import default_logger


@dataclass
class StepInfo:
    step_id: int
    start_time: Decimal
    end_time: Decimal
    computing_num: dict
    computing_time: dict
    kernel_data: list


@dataclass
class KernelInfo:
    device_id: int
    name: str
    start_time: Decimal
    duration: Decimal


def proc_step_info(step_data, event_item):
    step_id = int(event_item['name'].split('#')[-1]) # name形如ProfilerStep#109
    start_time = Decimal(event_item['ts'])
    end_time = start_time + Decimal(event_item['dur'])
    step_data.append(StepInfo(step_id, start_time, end_time, {}, {}, [])) # kernel_data后续填充


def proc_kernel_info(step_data, unknow_data, event_item):
    kernel_name = event_item['name']
    start_time = Decimal(event_item['ts'])
    duration = Decimal(event_item['dur'])
    device = int(event_item['args']['device'])
    kernel_info = KernelInfo(device, kernel_name, start_time, duration)
    match_flag = False
    kernel_end_time = kernel_info.start_time + kernel_info.duration
    for step_info in step_data:
        if kernel_info.start_time >= step_info.start_time and \
            (kernel_end_time <= step_info.end_time):
            step_info.kernel_data.append(kernel_info)
            match_flag = True
            break
    if not match_flag:
        unknow_data.append(kernel_info)


def save_parse_result(file_dir, step_data):
    kernel_details_file_path = os.path.join(file_dir, 'kernel_details.csv')
    step_trace_file_path = os.path.join(file_dir, 'step_trace_time.csv')
    with open(kernel_details_file_path, 'w', encoding='utf-8') as kernel_details_file, \
        open(step_trace_file_path, 'w', encoding='utf-8') as step_trace_file:
        kernel_details_file.write('"Step Id","Device_id","Name","Start_time","Duration"\n')
        step_trace_file.write('"Device_id","Step","Computing"\n')
        for step_info in step_data:
            for kernel_info in step_info.kernel_data:
                kernel_details_file.write(f'"{step_info.step_id}","{kernel_info.device_id}","{kernel_info.name}",\
                    "{kernel_info.start_time}","{kernel_info.duration}"\n')
                if kernel_info.device_id in step_info.computing_time:
                    step_info.computing_time[kernel_info.device_id] += kernel_info.duration
                    step_info.computing_num[kernel_info.device_id] += 1
                else:
                    step_info.computing_time[kernel_info.device_id] = kernel_info.duration
                    step_info.computing_num[kernel_info.device_id] = 1

            for device_id, computing_time in step_info.computing_time.items():
                step_trace_file.write(f'"{device_id}","{step_info.step_id}","{computing_time}"\n')


def parse_json(file_path: str):
    file_dir = os.path.dirname(file_path)
    step_data = []
    unknow_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # 获取全部时间对象，这是一个json对象数组
        prof_data = ijson.items(f, 'traceEvents.item')
        default_logger.info(f'start parse {file_path}...')
        # 获取cpu侧的每个step的开始、结束时间，所有在范围内的算子，即需要被记录的对象
        for event_item in prof_data:
            if 'cat' in event_item and event_item['cat'] == 'user_annotation' and 'ProfilerStep' in event_item['name']:
                proc_step_info(step_data, event_item)

            if 'cat' in event_item and event_item['cat'] == 'kernel':
                proc_kernel_info(step_data, unknow_data, event_item)

        # 数据解析完成后，unknow_data应该为空，否则数据异常
        if unknow_data:
            num = len(unknow_data)
            print_num = min(5, num) # 至多打印5个，否则占满屏幕
            for i in range(print_num):
                default_logger.error(f'unknow_data {i} is {unknow_data[i]}')
            raise ValueError(f'data error, num {num}')

    # 此时数据已经全部解析完成，持久化
    save_parse_result(file_dir, step_data)
    default_logger.info('parse all down!')


if __name__ == "__main__":
    """
    解析gpu场景下，torch.profiler生成的profiling的json文件，并在同目录下生成统计文件

    参数:
    - profiling的json文件路径

    输出:
    - 统计文件: kernel_details.csv, step_trace_time.csv
    """
    if len(sys.argv) != 2:
        raise ValueError('Please input json path.')

    parse_json(sys.argv[1])