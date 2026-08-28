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
from datetime import datetime
import shutil
import json
from contextlib import nullcontext
import argparse

import torch
from set_env import logger
from easydict import EasyDict as edict

def get_params():
    params = edict(
        {
            "data_dir": "",
            "num_epochs": 200,
            "device": "cpu",
            "mode": "train",
            "model_dir": "",
            "batch_size": 128,
            "learning_rate": 0.001,
            "embedding_size": 16,
            "field_size": 39,
            "extra_fields": 0,
        }
    )
    return params

def get_opts(argv, params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="eval",
        choices=["train", "test", "eval", "test_qps", "get_layer_result"],
        help="模型运行模型，支持train/test/eval/test_qps/get_layer_result",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "npu", "mlu"],
        help="指定训练设置，支持cpu/cuda/npu/mlu",
    )
    parser.add_argument("--device_id", type=int, default=0, help="指定设备id")
    parser.add_argument("--data_dir", type=str, default="", help="数据集路径")
    parser.add_argument("--test_dir", type=str, default="", help="测试数据集路径")
    parser.add_argument("--model_dir", type=str, default="", help="模型ckpt路径")
    parser.add_argument("--modelsave_dir", type=str, default="", help="训练模型保存路径")
    parser.add_argument(
        "--check_results",
        type=str,
        default="false",
        choices=["true", "false"],
        help="检查输入输出精度，是否保存输入输出",
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default="",
        help="记录模型QPS/Latency等信息路径，需要为test模型下",
    )

    parser.add_argument("--num_epochs", type=int, default=1, help="模型训练轮次")
    parser.add_argument("--batch_size", type=int, default=128, help="模型批处理大小")
    parser.add_argument("--step_num", type=int, default=200, help="模型训练批次")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="模型学习率大小"
    )
    parser.add_argument("--embedding_size", type=int, default=32, help="embedding大小")
    parser.add_argument(
        "--hf32",
        type=str,
        default="true",
        choices=["true", "false"],
        help="是否启用hf32(npu)/tf32(cuda)",
    )
    parser.add_argument(
        "--compile",
        type=str,
        default="true",
        choices=["true", "false"],
        help="是否启用inductor模式(torch.compile)",
    )
    parser.add_argument(
        "--enable_dynamic_compile",
        type=lambda value: {
            "false": False,
            "true": True,
            "none": None,
        }[value.lower()],
        default=False,
        help="torch.compile dynamic mode: False, True, or None",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="false",
        choices=["true", "false"],
        help="是否启用图下沉模式",
    )
    parser.add_argument(
        "--shape_handle",
        type=str,
        default="false",
        choices=["true", "false"],
        help="是否启用分档(torch.compile)，支持npu",
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=1, help="test_qps模式下batch_size大小"
    )
    parser.add_argument(
        "--profiling_mode",
        type=str,
        default="false",
        choices=["true", "false"],
        help="是否启用profiling",
    )
    parser.add_argument(
        "--profiling_path", type=str, default="./", help="profiling保存路径"
    )
    parser.add_argument(
        "--check_precision",
        type=str,
        default="false",
        choices=["true", "false"],
        help="是否校验compile精度",
    )
    parser.add_argument(
        "--dynamic_batch",
        type=str,
        default="false",
        choices=["true", "false"],
        help="是否启用动态batch_size输入，会忽略test_batch_size设置每个step采用大小batch_size输入",
    )
    parser.add_argument(
        "--check_mode",
        type=str,
        default="model",
        choices=["model", "layer", "op"],
        help="对比精度的等级，支持模型整体对比与每层对比",
    )
    parser.add_argument(
        "--gpu_data_path",
        type=str,
        help="gpu数据路径",
    )
    parser.add_argument("--seed", type=int, default=2025, help="随机数种子")
    parser.add_argument('--random_seqlen', nargs='+', type=int, default=[0],
                        help="动态seqlen的取值范围，为0则不启用动态seqlen")
    args = parser.parse_args(argv[1:])

    positive_check_dict = {
        "device_id": args.device_id,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "embedding_size": args.embedding_size,
    }

    range_check_dict = {
        "num_epochs": (args.num_epochs, 1, 1000),
        "batch_size": (args.batch_size, 1, 10000),
        "learning_rate": (args.learning_rate, 1e-8, 10),
        "embedding_size": (args.embedding_size, 4, 1024),
    }


    params.update(vars(args))
    params.profiling_mode = params.profiling_mode == "true"
    params.hf32 = params.hf32 == "true"
    params.compile = params.compile == "true"
    params.graph = params.graph == "true"
    params.shape_handle = params.shape_handle == "true"
    params.check_precision = params.check_precision == "true"
    params.dynamic_batch = params.dynamic_batch == "true"

    if args.device != "cpu":
        if args.device == "npu":
            import torch_npu

            torch.npu.set_device(args.device_id)
        params.device = f"{args.device}:{args.device_id}"

    logger.info(params)

    return params

def save_json(dic: dict, path: str, file_name: str, mode="w"):
    if path:
        if not os.path.exists(path):
            os.makedirs(path)
        js = json.dumps(dic)
        with open(os.path.join(path, file_name), "w") as file:
            file.write(js)


def output_report(times_range, batch_size,  graph_, compile_):
    times_range.sort()
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = { "model_name": "Decision-Transfomer"}
    tail_latency = round(times_range[int(len(times_range) * 0.99)] * 1000, 6)
    p90_latency = round(times_range[int(len(times_range) * 0.90)] * 1000, 6)
    avg_latency = round(sum(times_range) / len(times_range) * 1000, 6)
    qps = calculate_qps(times_range, batch_size)

    report["QPS"] = qps
    report["AVG Latency"] = avg_latency
    report["P99 Latency"] = tail_latency
    report["P90 Latency"] = p90_latency
    logger.info(f"[ scripts/common ] {report}")
    saved_path = os.path.join("report/")
    save_json(report, saved_path, f"report_graph{graph_}_compile{compile_}_{time_str}.json")
    logger.info(f"[ scripts/common ] Report json file saved in {saved_path}")


def calculate_qps(times_range, batches_list):
    return int(sum(batches_list) / sum(times_range))


def remove_directory_if_exists(path):
    if os.path.exists(path):
        try:
            # 使用shutil.rmtree删除文件夹及其所有内容
            shutil.rmtree(path)
            logger.info(f"删除文件夹: {path}")
        except Exception as e:
            logger.info(f"删除文件夹时出错: {e}")
    else:
        logger.info(f"路径不存在: {path}")


def get_loop_element(loop_list: list, idx):
    return loop_list[idx % len(loop_list)]


class Profiler:
    def __init__(self, param):
        self.params = param
        self.cur_batch_size = param.batch_size
        if "npu" in self.params.device:
            import torch_npu
            self.npu_experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=[torch_npu.profiler.ExportType.Text],
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
                l2_cache=False,
            )
        self.warmup = 5
        self.activate = 10
        self.skip_first = 10
        self.start_iter = self.skip_first + self.warmup
        self.end_iter = self.start_iter + self.activate
        self.graph = param.graph
        self.compile = param.compile

    def trace_handler(self, p):
        profiling_output_dir = os.path.join(
            self.params.profiling_path, self.params.model, f"{self.params.mode}_bs{self.cur_batch_size}"
        )
        if not os.path.exists(profiling_output_dir):
            os.makedirs(profiling_output_dir, mode=0o750)
        p.export_chrome_trace(
            os.path.join(profiling_output_dir, f"trace_{str(p.step_num)}_graph{self.graph}_compile{self.compile}.json")
        )
    def get_npu_profiler(self, profiling_output_dir):
        import torch_npu
        return torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                schedule=torch_npu.profiler.schedule(
                    wait=0, warmup=self.warmup, active=self.activate, repeat=1, skip_first=self.skip_first
                ),  # 与prof.step()配套使用
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    profiling_output_dir
                ),
                record_shapes=True,
                with_stack=True,
                profile_memory=False,
                with_modules=False,
                with_flops=False,
                experimental_config=self.npu_experimental_config,
            )

    def get_gpu_profiler(self, profiling_output_dir):
        return torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=0, warmup=self.warmup, active=self.activate, repeat=1, skip_first=self.skip_first
                ),  # 与prof.step()配套使用
                on_trace_ready=self.trace_handler,
                # 形状记录a
                record_shapes=True,
                with_stack=True,
                profile_memory=False,
                with_modules=False,
                with_flops=False,
            )

    def get_mlu_profiler(self, profiling_output_dir):
        return torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.MLU],
                schedule=torch.profiler.schedule(
                    wait=0, warmup=self.warmup, active=self.activate, repeat=1, skip_first=self.skip_first
                ),  # 与prof.step()配套使用
                on_trace_ready=self.trace_handler,
                profile_memory=False,
                with_stack=False,
                with_modules=False,
                with_flops=False,
            )

    def get_profiler(self):
        profiler = None
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        profiling_output_dir = os.path.join(
            self.params.profiling_path, self.params.model, 
            f"{self.params.mode}_bs{self.cur_batch_size}_graph{self.graph}_compile{self.compile}_{time_str}"
        )
        self.profiling_output_dir = profiling_output_dir
        remove_directory_if_exists(profiling_output_dir)
        if "npu" in self.params.device and self.params.profiling_mode:
            profiler = self.get_npu_profiler(profiling_output_dir)
        elif "cuda" in self.params.device and self.params.profiling_mode:
            profiler = self.get_gpu_profiler(profiling_output_dir)
        elif "mlu" in self.params.device and self.params.profiling_mode:
            profiler = self.get_mlu_profiler(profiling_output_dir)
        else:
            profiler = nullcontext()
        return profiler
