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

from logger import default_logger
import os
import re
import sys


TIME_PATTERN = r'\[(?P<module>\w+\.py)\] avg time over (?P<device>\w+) with (?P<mode>\w+) mode: (?P<time>\d+\.\d+)'
NUM_PATTERN = r'\[(?P<module>\w+\.py)\] op num over (?P<device>\w+) with (?P<mode>\w+) mode: (?P<num>\d+)'


def proc_time_info(log_info, line):
    if log_info is not None:
        raise ValueError("log parse error, log_info is not None!")

    match = re.search(TIME_PATTERN, line)
    if not match:
        raise ValueError(f"log parse error, can't match log_avg_time, log line:{line}")

    log_info = {}
    log_info["module_name"] = match.group("module")
    log_info["device"] = match.group("device")
    log_info["mode"] = match.group("mode")
    log_info["op_time_sum"] = match.group("time")


# 解析op num后，写入文件
def proc_num_info(log_info, line, result_file):
    if log_info is None:
        raise ValueError("log parse error, log_info is None!")

    match = re.search(NUM_PATTERN, line)
    if not match:
        raise ValueError(f"log parse error, can't match log_op_num, log line:{line}")
    if log_info["module_name"] != match.group("module"):
        raise ValueError(f"log parse error, module name not match, \
            {log_info["module_name"]} != {match.group("module")}")
    if log_info["device"] != match.group("device"):
        raise ValueError(f"log parse error, device not match, {log_info["device"]} != {match.group("device")}")
    if log_info["mode"] != match.group("mode"):
        raise ValueError(f"log parse error, mode not match, {log_info["mode"]} != {match.group("mode")}")

    log_info["op_num"] = match.group("num")
    result_file.write(f'"{log_info["module_name"]}","{log_info["op_num"]}",\
        "{log_info["op_time_sum"]}"\n')
    log_info = None


def proc_fail_info(line, result_file):
    fail_pattern = r'pattern_\d+\.py'
    match = re.findall(fail_pattern, line)
    if match:
        for module_name in match:
            module_name, _ = os.path.splitext(module_name)
            result_file.write(f'"{module_name}","0","0"\n')


def parse_log(log_path: str):
    default_logger.info(f'start parse log: {log_path}')
    dir_path = os.path.dirname(log_path)
    file_name = os.path.basename(log_path)
    base_name, _ = os.path.splitext(file_name)

    result_file_path = os.path.join(dir_path, base_name + '.csv')
    with open(log_path, 'r', encoding='utf-8') as log_file, \
         open(result_file_path, 'w', encoding='utf-8') as result_file:
            log_info: dict = None
            for line in log_file:
                if 'log_avg_time' in line:
                    proc_time_info(log_info, line)

                if 'log_op_num' in line:
                    proc_num_info(log_info, line, result_file)

                if '以下是失败的用例文件' in line:
                    proc_fail_info(line, result_file)

    default_logger.info('log parse done!')


if __name__ == "__main__":
    """
    解析pattern生成的日志文件，将每个pattern的算子数、算子总耗时统计输出为csv文件

    参数:
    - pattern生成的日志文件路径

    输出:
    - 同日志文件名的csv文件
    """
    if len(sys.argv) != 2:
        raise ValueError('Please input log path.')
    
    parse_log(sys.argv[1])
