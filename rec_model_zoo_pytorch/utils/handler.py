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
import collections
import os
import random
import time
import argparse
import json
from contextlib import nullcontext

import numpy as np
import torch
import tensorflow as tf
from tqdm import tqdm
from easydict import EasyDict as edict
from torch.fx import symbolic_trace
import pandas as pd

from utils.common import evaluate_model, savSe_json, Profiler
from utils.logger import logger
from utils.capturing_interpreter import CapturingInterpreter, to_clean_cpu_tensor


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


def nonegative_int(value, name):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"{name}: {value} must be positive")
    return ivalue


def range_int(value, min_val, max_val, name):
    if value < min_val or value > max_val:
        logger.error(f"value {value} min_val {min_val} max_val{max_val}")
        raise argparse.ArgumentTypeError(
            f"{name} must between {min_val} and {max_val}, but get {value}."
        )
    return value


def check_positive_range(pos_dict: dict, range_dict: dict):
    for key, val in pos_dict.items():
        res = nonegative_int(val, key)

    for key, val in range_dict.items():
        res = range_int(val[0], val[1], val[2], key)
    return res


def get_fun_argument(parser):
    parser.add_argument(
        "--mode",
        type=str,
        default="eval",
        choices=["train", "test", "eval", "test_qps", "get_layer_result"],
        help="模型运行模型，支持train/test/eval/test_qps/get_layer_result",
    )
    parser.add_argument(
        "--shape_handle",
        type=str,
        default="false",
        choices=["true", "false"],
        help="是否启用分档(torch.compile)，支持npu",
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
    parser.add_argument("--seed", type=int, default=2025, help="随机数种子")
    parser.add_argument('--random_seqlen', nargs='+', type=int, default=[0],
                        help="动态seqlen的取值范围，为0则不启用动态seqlen")
    parser.add_argument(
        "--test_batch_size", type=int, default=1, help="test_qps模式下batch_size大小"
    )
    return parser


def get_opt_argument(parser):
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
        "--graph",
        type=str,
        default="false",
        choices=["true", "false"],
        help="是否启用图下沉模式",
    )
    return parser


def get_argument(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "npu", "mlu"],
        help="指定训练设置，支持cpu/cuda/npu/mlu",
    )
    parser.add_argument("--device_id", type=int, default=0, help="指定设备id")
    parser.add_argument("--data_dir", type=str, default="", help="数据集路径")
    parser.add_argument("--model_dir", type=str, default="", help="模型ckpt路径")
    parser.add_argument(
        "--report_dir",
        type=str,
        default="",
        help="记录模型QPS/Latency等信息路径，需要为test模型下",
    )

    parser.add_argument("--num_epochs", type=int, default=1, help="模型训练轮次")
    parser.add_argument("--batch_size", type=int, default=128, help="模型批处理大小")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="模型学习率大小"
    )
    parser.add_argument("--embedding_size", type=int, default=32, help="embedding大小")
    parser = get_fun_argument(parser)
    parser = get_opt_argument(parser)

    args = parser.parse_args(argv[1:])
    return args


def get_opts(argv, params):
    args = get_argument(argv)

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

    check_res = check_positive_range(positive_check_dict, range_check_dict)

    if args.mode != "test_qps" and not args.data_dir:
        raise argparse.ArgumentTypeError("The dataset path is not specified.")

    if args.mode != "test_qps" and not os.path.exists(args.data_dir):
        raise argparse.ArgumentTypeError(
            f"The dataset path {args.data_dir} does not exist."
        )

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


def set_all_seed(params):
    logger.info(f"set cpu random seed: {params.seed}")
    if params.seed is None:
        logger.warning("Random seed is None, reproducibility might not be guaranteed.")
        return
    torch.manual_seed(params.seed)  # PyTorch 的 CPU 随机数种子
    random.seed(params.seed)
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)
    # 如果存在 GPU/NPU，也设置它们的种子
    if "cuda" in params.device:
        logger.info(f"set cuda random seed: {params.seed}")
        torch.cuda.manual_seed(params.seed)
        torch.cuda.manual_seed_all(params.seed)  # 为所有 GPU 设置种子
    elif "npu" in params.device:
        import torch_npu

        logger.info(f"set npu random seed: {params.seed}")
        torch.npu.manual_seed(params.seed)
        torch.npu.manual_seed_all(params.seed)
        torch_npu.npu.manual_seed(params.seed)
        torch_npu.npu.manual_seed_all(params.seed)
    else:
        logger.info(f"cant set seed, params.device: {params.device}")
    # 设置环境变量以影响某些库或系统级别的随机性
    os.environ["PYTHONHASHSEED"] = str(params.seed)


def calculate_qps(times_range, batches_list):
    return int(sum(batches_list) / sum(times_range))


class TestHandler:
    def __init__(self, params):
        self.params = params
        self.batch_size = 1
        self.manual_graph = False
        self.dynamic_range = (1, self.params.test_batch_size)

        self.manual_graph = self.is_manual_graph()
        self.graphs = {}

    def is_manual_graph(self):
        return not self.params.compile and self.params.graph and \
            ("npu" in self.params.device or "cuda" in self.params.device)

    def generate_data(self, batch_size):
        raise NotImplementedError("TestHandler.generate_data(batch_size)")

    def test_qps(self, model):
        _, report = self.infer_with_generate_data(model)

    def save_pt(self, model, features):
        ts = torch.jit.trace(model, features, strict=False)
        save_path = os.path.join(self.params.model_dir, self.params.model)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        ts.save(os.path.join(save_path, "model.pt"))

    def model_infer(self, model, features):
        if self.manual_graph:
            return self.model_infer_graph(model, features, self.batch_size)
        return model(features)

    def model_infer_graph(self, model, features, batch_size):
        if batch_size not in self.graphs:
            # new batch size scenario, a warmup is required
            new_batch = {
                "graph": (
                    torch.npu.NPUGraph()
                    if "npu" in self.params.device
                    else torch.cuda.CUDAGraph()
                ),
                "stream": (
                    torch.npu.Stream(self.params.device)
                    if "npu" in self.params.device
                    else None
                ),
                "static_input": None,
                "static_output": None,
            }
            new_batch["static_input"] = {k: v.clone() for k, v in features.items()}
            if "npu" in self.params.device:
                with torch.npu.graph(new_batch["graph"], None, new_batch["stream"]):
                    new_batch["static_output"] = model(new_batch["static_input"])
            else:
                for _ in range(3):
                    _ = model(new_batch["static_input"])
                torch.cuda.synchronize()

                with torch.cuda.graph(new_batch["graph"]):
                    new_batch["static_output"] = model(new_batch["static_input"])
            self.graphs[batch_size] = new_batch
        else:
            for k in features.keys():
                self.graphs[batch_size]["static_input"][k].copy_(features[k])

        self.graphs[batch_size]["graph"].replay()
        if "cuda" in self.params.device:
            torch.cuda.synchronize()
        elif "npu" in self.params.device:
            torch.npu.synchronize()
        return self.graphs[batch_size]["static_output"]

    def synchronize(self):
        if "npu" in self.params.device:
            torch.npu.synchronize()
        elif "cuda" in self.params.device:
            torch.cuda.synchronize()
        elif "mlu" in self.params.device:
            torch.mlu.synchronize()

    def get_batch_size(self):
        if self.params.dynamic_batch:
            return random.randint(self.dynamic_range[0], self.dynamic_range[1])
        return self.params.test_batch_size


    def infer_with_generate_data(self, model):
        self.batch_size = self.get_batch_size()

        model.eval()
        features = self.generate_data(self.dynamic_range[1])
        iteration = 210
        times_range = []
        batches_list = []
        with torch.no_grad():
            # 预热
            if self.params.check_precision:
                logger.info("check precision between eager and compile model")
                compile_pred = self.model_infer(model, features)
                eager_pred = self.eager_model(features)
                for key in eager_pred.keys():
                    tensor1 = eager_pred[key]
                    tensor2 = compile_pred[key]
                    logger.info(f"*********************precision compare:{key}**********************")

                    torch.testing.assert_close(tensor1, tensor2, rtol=1e-04, atol=1e-04, equal_nan=True)

                logger.info("Precision check pass!")
            for _ in range(30):
                if self.params.dynamic_batch:
                    self.batch_size = self.get_batch_size()
                    features = self.generate_data(self.batch_size)
                pred = self.model_infer(model, features)
        profiler = Profiler(self.params)
        bs_str = self.batch_size if not self.params.dynamic_batch else "Dynamic"
        profiler.cur_batch_size = bs_str
        profile = profiler.get_profiler()
        with profile as prof:
            with torch.no_grad():
                for it in range(iteration):
                    self.batch_size = self.get_batch_size()
                    if self.params.dynamic_batch:
                        features = self.generate_data(self.batch_size)
                    self.synchronize()
                    start_time = time.time()
                    pred = self.model_infer(model, features)
                    self.synchronize()
                    end_time = time.time()
                    if self.params.profiling_mode:
                        prof.step()
                    if it < profiler.start_iter or it >= profiler.end_iter:
                        cur_range = end_time - start_time
                        batches_list.append(self.batch_size)
                        times_range.append(cur_range)

        report = {"Batch_size": bs_str, "model_name": self.params.model}
        if self.params.report_dir:
            df = pd.DataFrame({
                'time(s)': times_range,
                'batch size': batches_list,
            })
            saved_path = os.path.join(self.params.report_dir, self.params.model)
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)
            df.to_csv(os.path.join(saved_path, f'timeranges_bs{bs_str}.csv'))

        times_range.sort()
        tail_latency = round(times_range[int(len(times_range) * 0.99)] * 1000, 6)
        p90_latency = round(times_range[int(len(times_range) * 0.90)] * 1000, 6)
        avg_latency = round(sum(times_range) / len(times_range) * 1000, 6)
        qps = calculate_qps(times_range, batches_list)

        report["QPS"] = qps
        report["AVG Latency"] = avg_latency
        report["P99 Latency"] = tail_latency
        report["P90 Latency"] = p90_latency
        logger.info(report)
        if self.params.report_dir:
            saved_path = os.path.join(self.params.report_dir, self.params.model)
            save_json(report, saved_path, f"report_bs{bs_str}.json")
            logger.info(f"Report json file saved in {saved_path}")

        return pred, report

transform_keys = []


def transform_pre_fn(*args, **kwargs):
    transform_inputs = []
    for key, value in args[0].items():
        transform_inputs.append(value)
        transform_keys.append(key)
    return transform_inputs


def transform_post_fn(trans_outputs, **kwargs):
    arg_list = []
    for trans_output in trans_outputs:
        arg = {}
        for idx, tensor in enumerate(trans_output):
            arg[transform_keys[idx]] = tensor
        arg_list.append((arg,))
    kwargs_list = [{}] * len(arg_list)
    return arg_list, kwargs_list


recover_keys = []


def recover_pre_fn(groups):
    recover_inputs = []
    for group in groups:
        recover_input = []
        for value in group.values():
            recover_input.append(value)
        recover_inputs.append(recover_input)
    for key in groups[0].keys():
        recover_keys.append(key)

    return recover_inputs


def recover_post_fn(re_outputs):
    real_output = {}
    for idx, re_output in enumerate(re_outputs):
        real_output[recover_keys[idx]] = re_output
    return real_output


class ModelHandler:
    def __init__(
        self,
        params,
        model: torch.nn.Module,
        optimizer,
        load_data_func,
        test_handler=None,
    ) -> None:
        self.params = params
        self.manual_graph = False
        self.static_inputs = {}
        self.static_mode = None
        self.static_outputs = None
        self.graph_prepared = True
        self.stream = None
        self.graph = None
        self.model = model
        self.saved_dir = self.get_saved_path()
        self.load_check_point()
        logger.info(
            f"*********************compile:{params.compile}**********************"
        )
        logger.info(f"*********************graph:{params.graph}**********************")
        if params.shape_handle:
            self.shape_options = {
                "enable_shape_handling": True,
                "shape_handling_min_size": 1,
                "shape_handling_max_size": 1024,
                "shape_handling_dict": {
                    "trans_pre_fn": transform_pre_fn,
                    "trans_post_fn": transform_post_fn,
                    "re_pre_fn": recover_pre_fn,
                    "re_post_fn": recover_post_fn,
                },
            }

        if params.check_precision:
            test_handler.eager_model = self.model
        
        self.set_compile_model()

        self.set_hf32()

        self.optimizer = optimizer
        self.loss_fn = model.loss
        if self.params.mode != "test_qps":
            self.train_loader, self.test_loader, self.val_loader = load_data_func(
                self.params
            )
        self.cur_batch_size = params.batch_size
        self.test_handler = test_handler
        torch.manual_seed(params.seed)
        self.npu_experimental_config = None
        
    def set_compile_model(self):
        if self.params.compile:
            if self.params.graph and self.params.shape_handle and "npu" in self.params.device:
                self.shape_options["triton.cudagraphs"] = True
                self.model = torch.compile(
                    self.model, backend="inductor", dynamic=False, options=self.shape_options
                )
            elif self.params.shape_handle and "npu" in self.params.device:
                self.model = torch.compile(
                    self.model, backend="inductor", dynamic=False, options=self.shape_options
                )
            elif self.params.graph:
                self.model = torch.compile(
                    self.model, backend="inductor", dynamic=False, mode="reduce-overhead"
                )
            else:
                self.model = torch.compile(self.model, backend="inductor", dynamic=False)
        else:
            self.model = self.model
            if self.params.graph and ("npu" in self.params.device or "cuda" in self.params.device):
                self.manual_graph = True
                self.graph_prepared = False

    def set_hf32(self):
        if "npu" in self.params.device:
            import torch_npu

            torch_npu.npu.aclnn.allow_hf32 = self.params.hf32
            torch_npu.npu.conv.allow_hf32 = self.params.hf32
            torch_npu.npu.matmul.allow_hf32 = self.params.hf32
            logger.info(f"*************hf32: {self.params.hf32}*******************")
        elif "cuda" in self.params.device:
            torch.backends.cuda.matmul.allow_tf32 = self.params.hf32
            torch.backends.cudnn.allow_tf32 = self.params.hf32
            logger.info(f"*************tf32: {self.params.hf32}*******************")
        elif "mlu" in self.params.device:
            torch.backends.mlu.matmul.allow_tf32 = self.params.hf32
            torch.backends.cnnl.allow_tf32 = self.params.hf32
            logger.info(f"*************tf32: {self.params.hf32}*******************")

    def load_check_point(self):
        if self.params.get("model_dir", ""):
            if os.path.exists(self.params.model_dir):
                try:
                    pth_path = os.path.join(self.saved_dir, "best_val.pth")
                    self.model.load_state_dict(torch.load(pth_path, 
                                                          map_location=self.params.device, 
                                                          weights_only=False))
                    logger.info(f"load {pth_path} success !")
                except FileNotFoundError as e:
                    if self.params.model not in ['resnet50', 'bert']:
                        logger.info("no saved pth file found !")
                        self.initialize_weights()
                    else:
                        logger.info('load from pretrained weight, skip initialize.')
            else:
                logger.info("model path not exist!")

    def get_saved_path(self):
        saved_path = None
        if self.params.get("model_dir", ""):
            saved_path = os.path.join(self.params["model_dir"], self.params.model)
            if not os.path.exists(saved_path):
                os.makedirs(saved_path, mode=0o750)
        return saved_path

    def trace_handler(self, p):
        profiling_output_dir = os.path.join(
            self.params.profiling_path, self.params.model, f"bs{self.cur_batch_size}"
        )
        if not os.path.exists(profiling_output_dir):
            os.makedirs(profiling_output_dir, mode=0o750)
        p.export_chrome_trace(
            os.path.join(profiling_output_dir, f"trace_{str(p.step_num)}.json")
        )

    def is_on_device(self):
        return "cuda" in self.params.device or "npu" in self.params.device \
            or "mlu" in self.params.device

    def run_train_one_epoch(self, epoch):
        total_loss = 0
        idx = 0
        profiler = Profiler(self.params)
        profiler = profiler.get_profiler()
        with profiler as prof:
            for inputs, labels in tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.params.num_epochs} - Train",
            ):
                idx += 1
                # 前向传播
                outputs = self.model(inputs, "train")
                loss = self.loss_fn(outputs, labels)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # 统计信息
                total_loss += loss.item()
                if self.is_on_device() and self.params.profiling_mode:
                    prof.step()
        return total_loss

    def graph_warmup(self, inputs, mode):
        self.static_inputs = {k: v.clone() for k, v in inputs.items()}
        self.static_mode = mode
        self.static_outputs = None
        self.graph_prepared = True
        if "npu" in self.params.device:
            self.stream = torch.npu.Stream(self.params.device)
            self.graph = torch.npu.NPUGraph()
            with torch.npu.graph(self.graph, None, self.stream):
                self.static_outputs = self.model(self.static_inputs, self.static_mode)
        elif "cuda" in self.params.device:
            for _ in range(3):
                with torch.no_grad():
                    _ = self.model(self.static_inputs, self.static_mode)
            torch.cuda.synchronize()

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_outputs = self.model(self.static_inputs, self.static_mode)

    def run_manual_graph(self, inputs, mode):
        if not self.graph_prepared:
            self.graph_warmup(inputs, "val")
        else:
            for key, val in inputs.items():
                self.static_inputs[key].copy_(val)

        self.graph.replay()
        if "cuda" in self.params.device:
            torch.cuda.synchronize()
        elif "npu" in self.params.device:
            torch.npu.synchronize()
        return self.static_outputs

    def infer_model(self, inputs, labels, prof):
        outputs = (
            self.run_manual_graph(inputs, "val")
            if self.manual_graph
            else self.model(inputs, "val")
        )
        loss = self.loss_fn(outputs, labels)
        total_loss += loss.item()
        if self.is_on_device() and self.params.profiling_mode:
            prof.step()

    def eval(self, epoch=0):
        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            idx = 0
            profiler = Profiler(self.params)
            profiler = profiler.get_profiler()
            with profiler as prof:
                for inputs, labels in tqdm(
                    self.val_loader,
                    desc=f"Epoch {epoch + 1}/{self.params.num_epochs} - Val",
                ):
                    self.infer_model(inputs, labels, prof)
                    idx += 1
                logger.info(idx)
        logger.info(f"total loss: {total_loss}")

    def test(self, epoch=0):
        total_loss = 0
        self.model.eval()

        report = {"Batch_size": self.params.batch_size, "model_name": self.params.model}
        with torch.no_grad():
            idx = 0
            profiler = Profiler(self.params)
            profiler = profiler.get_profiler()
            with profiler as prof:
                for inputs, labels in tqdm(
                    self.test_loader,
                    desc=f"Epoch {epoch + 1}/{self.params.num_epochs} - Test",
                ):
                    self.infer_model(inputs, labels, prof)
                    idx += 1
                logger.info(idx)

        report["Total Loss"] = total_loss
        if self.params.report_dir:
            saved_path = os.path.join(self.params.report_dir, self.params.model)
            save_json(report, saved_path, f"report_loss.json")
            logger.info(f"Report json file saved in {saved_path}")
        logger.info(report)
        logger.info(f"total loss: {total_loss}")

    def train(self):
        self.model.train()
        train_losses = []
        val_losses = []
        min_val_loss = 1e8
        if self.saved_dir:
            saved_path = os.path.join(self.saved_dir, "best_val.pth")
        else:
            saved_path = ""

        for epoch in range(self.params.num_epochs):
            # 训练阶段
            self.model.train()
            running_loss = 0.0

            running_loss = self.run_train_one_epoch(epoch)

            logger.info(f"total loss: {running_loss}")
            train_loss = running_loss / (
                len(self.train_loader) * self.params.num_epochs
            )
            train_losses.append(train_loss)

            # 验证阶段
            val_running_loss = 0.0
            val_running_loss = self.eval(epoch)

            val_loss = val_running_loss / (
                len(self.val_loader) * self.params.num_epochs
            )
            if saved_path is not None and val_loss < min_val_loss:
                min_val_loss = val_loss
                logger.info(f"saving checkpoint to {saved_path}!")
                torch.save(self.model.state_dict(), saved_path)

            val_losses.append(val_loss)

            # 打印统计信息
            logger.info(f"Epoch {epoch + 1}/{self.params.num_epochs}")
            logger.info("-" * 50)

    def initialize_weights(self):
        logger.info("Initializing model weights with Xavier Uniform...")
        for m in self.model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                # 判断模块的类型
                if hasattr(m, "weight") and m.weight is not None:
                    torch.nn.init.kaiming_uniform_(m.weight)
                    logger.debug(f"Initialized weight of {m.__class__.__name__}")
                    logger.debug(f"Initialized weight of {m.weight}")

                # 检查是否存在偏置项
                if hasattr(m, "bias") and m.bias is not None:
                    # 将偏置初始化为 0
                    torch.nn.init.constant_(m.bias, 0)
                    logger.debug(f"Initialized bias of {m.__class__.__name__} to 0")

    def get_layer_output_with_hook(self):
        self.model.eval()
        captured_data = collections.OrderedDict()
        # 注册hook
        register_forward_output_hook(self.model, captured_data)
        # 获取1批次数据
        loader_iterator = iter(self.test_loader)
        inputs, labels = next(loader_iterator)
        logger.info(f"input: {inputs}")

        logger.info("\nrunning model...")
        with torch.no_grad():
            output = self.model(inputs)
        return captured_data, output

    def get_layer_result(self):
        # 保存初始权重
        saved_path = os.path.join(self.saved_dir, "init_val.pth")
        logger.info(f"saving init checkpoint to {saved_path}!")
        torch.save(self.model.state_dict(), saved_path)
        self.model.eval()  # 设置为评估模式
        loader_iterator = iter(self.test_loader)
        inputs, labels = next(loader_iterator)
        logger.info(f"input: {inputs}")
        layer_result = None
        if self.params.graph or self.params.compile:
            logger.info(f"graph: {self.params.graph}, compile: {self.params.compile}, get total data only.")
            output = (
                self.run_manual_graph(inputs, "val")
                if self.manual_graph
                else self.model(inputs, "val")
            )
        else:
            try:
                # 尝试使用torch.fx对模型进行符号化追踪
                logger.info("try to trace model")
                traced_model = symbolic_trace(self.model)
                logger.info("model trace success!")
                # 使用Interpreter运行模型
                capture_interpreter = CapturingInterpreter(traced_model)
                logger.info("\nrunning model with interpreter...")
                with torch.no_grad():
                    output = capture_interpreter.run(inputs)
                layer_result = capture_interpreter.captured_data
            except Exception as e:
                logger.info(f"model trace failed: {e}, capture data with hook")
                layer_result, output = self.get_layer_output_with_hook()

        logger.info(f"output: {output}")
        total_result = {
            self.params.model: {
                "type": "total_model",
                "input": to_clean_cpu_tensor(inputs),
                "output": to_clean_cpu_tensor(output),
            }
        }

        logger.info(f"\n--- saving data... ---")
        layer_result_dir = "./infer_result/"
        os.makedirs(layer_result_dir, exist_ok=True)
        result_filename = self.params.device + "_" + self.params.model
        result_path = os.path.join(layer_result_dir, result_filename)
        torch.save(total_result, result_path)
        logger.info(f"\n--- total data saved to: {result_path} ---")

        if layer_result is not None:
            layer_result_filename = result_filename + "_layer"
            layer_result_path = os.path.join(layer_result_dir, layer_result_filename)
            torch.save(layer_result, layer_result_path)
            logger.info(f"\n--- layer data saved to:{layer_result_path} ---")

    def run(self):
        if self.params["mode"] == "train":
            self.train()
        elif self.params["mode"] == "eval":
            self.eval()
        elif self.params["mode"] == "test":
            self.test()
        elif self.params["mode"] == "test_qps":
            self.test_handler.test_qps(self.model)
        elif self.params["mode"] == "get_layer_result":
            self.get_layer_result()


def get_hook(name, captured_data, module):
    def hook(model, inputs, output):
        layer_name = f"{name} ({module.__class__.__name__})"
        captured_data[layer_name] = {
            "type": "named_module",
            "input": to_clean_cpu_tensor(inputs),
            "output": to_clean_cpu_tensor(output),
            "parameter": to_clean_cpu_tensor(module.state_dict()),
        }
        logger.info(f"Hook captured data from: {layer_name}")

    return hook


def register_forward_output_hook(model, captured_data):
    for name, layer in model.named_modules():
        if not list(layer.children()):
            layer.register_forward_hook(get_hook(name, captured_data, layer))
            logger.info(
                f"Registered hook for layer: {name} ({layer.__class__.__name__})"
            )