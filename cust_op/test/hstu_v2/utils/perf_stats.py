#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
import glob
import argparse
import logging

import pandas as pd

# 配置 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 临时benchmark文件和结果文件路径
TMP_BENCHMARK_CSV = "tmp_benchmark.csv"
BENCHMARK_RESULT_CSV = "benchmark_result.csv"

# 解析命令行参数
parser = argparse.ArgumentParser(description='提取op_summary中的算子数据')
parser.add_argument('--input-dir', type=str, required=True, help='输入目录，递归查找op_summary_*.csv')
parser.add_argument('--target-ops', type=str, nargs='+', required=True, help='目标算子名称列表')
parser.add_argument('--output', type=str, default='benchmark_result.csv', help='输出文件路径')
args = parser.parse_args()

# 在指定目录中递归查找所有op_summary_*.csv文件
csv_pattern = os.path.join(args.input_dir, '**', 'op_summary_*.csv')
csv_files = glob.glob(csv_pattern, recursive=True)
logger.info(f"在目录 {args.input_dir} 中找到 {len(csv_files)} 个CSV文件:")
for f in csv_files:
    logger.info(f"  - {os.path.relpath(f, args.input_dir)}")

# 读取所有CSV文件并合并
dfs = []
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file, low_memory=False)
        dfs.append(df)
        logger.info(f"已加载: {os.path.basename(csv_file)}, {len(df)} 行")
    except Exception as e:
        logger.error(f"加载 {csv_file} 失败: {e}")

if not dfs:
    raise RuntimeError("没有找到任何CSV文件")

# 合并所有数据
all_data = pd.concat(dfs, ignore_index=True)
logger.info(f"\n总共加载了 {len(all_data)} 行数据")

# 筛选目标算子的记录
target_ops = args.target_ops
filtered = all_data[all_data['Op Name'].isin(target_ops)]
logger.info(f"筛选出 Op Name 为 {target_ops} 的记录: {len(filtered)} 行")

if len(filtered) == 0:
    raise RuntimeError("没有找到目标算子")

# 按 Op Name、Input Shapes 和 Input Data Types 进行分组，取 Task Start Time(us) 最小的记录
# 首先确保 Task Start Time(us) 列是数值类型
filtered = filtered.copy()
filtered['Task Start Time(us)'] = pd.to_numeric(filtered['Task Start Time(us)'], errors='coerce')
filtered['Task Duration(us)'] = pd.to_numeric(filtered['Task Duration(us)'], errors='coerce')

# 去除 Input Shapes 或 Input Data Types 为空的记录
filtered = filtered.dropna(subset=['Input Shapes', 'Input Data Types'])

# 按 Op Name、Input Shapes 和 Input Data Types 分组，取 Task Start Time(us) 最小的记录
result = filtered.loc[
    filtered.groupby(['Op Name', 'Input Shapes', 'Input Data Types'])['Task Start Time(us)'].idxmin()
]

# 选择需要的列
result = result[['Op Name', 'Input Shapes', 'Input Data Types', 'Task Start Time(us)', 'Task Duration(us)']]

# 按 Op Name 和 Task Start Time(us) 排序
result = result.sort_values(['Op Name', 'Task Start Time(us)'])

# 保存结果到临时CSV（不含表头追加模式）
TMP_OUTPUT = "tmp_benchmark_with Perf.csv"
result.to_csv(TMP_OUTPUT, index=False)
logger.info(f"\n临时结果已保存到: {TMP_OUTPUT}")
logger.info(f"共 {len(result)} 条记录")

# 读取原始tmp_benchmark.csv获取用例输入
if os.path.exists(TMP_BENCHMARK_CSV):
    input_df = pd.read_csv(TMP_BENCHMARK_CSV)
    logger.info(f"原始用例输入记录: {len(input_df)} 行")
else:
    logger.warning(f"找不到临时文件 {TMP_BENCHMARK_CSV}")
    input_df = pd.DataFrame()

# 验证行数一致性
if len(input_df) > 0:
    result_count = len(result)
    input_count = len(input_df)
    if result_count != input_count:
        raise ValueError(f"行数不一致: perf_stats去重后有 {result_count} 条记录, "
                         f"而tmp_benchmark.csv有 {input_count} 条记录")
    logger.info(f"行数校验通过: 两者都有 {input_count} 条记录")

# 合并结果: 将Input Shapes和Task Duration追加到原始输入数据
if len(input_df) > 0 and len(result) > 0:
    # 读取result数据
    perf_df = pd.read_csv(TMP_OUTPUT)

    # 合并数据，确保行数对应
    # 输入参数在前，性能数据在后
    merged_df = pd.concat([input_df, perf_df[['Input Shapes', 'Task Duration(us)']]], axis=1)

    # 保存最终结果
    output_path = args.output
    merged_df.to_csv(output_path, index=False)
    logger.info(f"\n最终结果已保存到: {output_path}")
    logger.info(f"共 {len(merged_df)} 条记录")

    # 显示结果预览
    logger.info("\n结果预览:")
    logger.info("\n" + merged_df.to_string(index=False))
else:
    # 如果没有输入数据，只保存perf结果
    output_path = args.output
    result.to_csv(output_path, index=False)
    logger.info(f"\n结果已保存到: {output_path}")
    logger.info(f"共 {len(result)} 条记录")
    logger.info("\n结果预览:")
    logger.info("\n" + result.to_string(index=False))

# 清理临时文件
if os.path.exists(TMP_OUTPUT):
    os.remove(TMP_OUTPUT)