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
import shutil
import json
from contextlib import nullcontext

import torch
from sklearn.metrics import roc_auc_score

from utils.logger import logger


def evaluate_model(model, features, labels, device):
    """评估模型性能"""
    model.eval()
    with torch.no_grad():
        # 将数据移到设备
        features = {k: v.to(device) for k, v in features.items()}
        labels = {k: v.to(device) for k, v in labels.items()}

        predictions = model(features)

        # 计算AUC
        res = {}
        if features.get("ctr", None) is not None:
            ctr_auc = roc_auc_score(
                labels["y"].cpu().numpy(), predictions["ctr"].cpu().numpy()
            )
            res["ctr_auc"] = ctr_auc

        if features.get("ctcvr", None) is not None:
            ctcvr_auc = roc_auc_score(
                labels["z"].cpu().numpy(), predictions["ctcvr"].cpu().numpy()
            )
            res["ctcvr_auc"] = ctcvr_auc

        # 计算CVR AUC（只在点击样本上）
        if features.get("cvr", None) is None:
            return res

        ctr_mask = labels["y"] > 0
        if ctr_mask.sum() <= 0:
            return res

        cvr_labels = labels["z"][ctr_mask]
        cvr_predictions = predictions["cvr"][ctr_mask]
        if len(torch.unique(cvr_labels)) > 1:  # 确保有正负样本
            cvr_auc = roc_auc_score(
                cvr_labels.cpu().numpy(), cvr_predictions.cpu().numpy()
            )
            res["cvr_auc"] = cvr_auc

        return res


def save_json(dic: dict, path: str, file_name: str, mode="w"):
    if path:
        if not os.path.exists(path):
            os.makedirs(path)
        js = json.dumps(dic)
        with open(os.path.join(path, file_name), "w") as file:
            file.write(js)


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
        self.warmup = 1
        self.activate = 10
        self.skip_first = 100
        self.start_iter = self.skip_first + self.warmup
        self.end_iter = self.start_iter + self.activate

    def trace_handler(self, p):
        profiling_output_dir = os.path.join(
            self.params.profiling_path, self.params.model, f"bs{self.cur_batch_size}"
        )
        if not os.path.exists(profiling_output_dir):
            os.makedirs(profiling_output_dir, mode=0o750)
        p.export_chrome_trace(
            os.path.join(profiling_output_dir, f"trace_{str(p.step_num)}.json")
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
                # 形状记录
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
        profiling_output_dir = os.path.join(
            self.params.profiling_path, self.params.model, f"bs{self.cur_batch_size}"
        )
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