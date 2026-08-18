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
import torch
import numpy as np
from set_env import logger

class IoChecker:
    def __init__(self, params):
        self.path = params["modelsave_dir"]
        self.device = params["device"]
        self.check = params["check_results"] == "true"

    def load_or_save_inputs(self, inputs, t):
        if t == 0 and self.check:
            save_path = os.path.normpath(self.path + '/dt.pt')
            if os.path.exists(self.path):
                data = torch.load(save_path, map_location="cpu")
                logger.info(
                    f"[ scripts/check_results ] loading input from {save_path}"
                )
                return self._to_device(data, self.device) if self.device else data

            os.makedirs(self.path, exist_ok=True)
            torch.save(self._to_cpu_detached(inputs), save_path)
            logger.info(
                f"[ scripts/check_results ] save input to {self.path}"
            )
            return inputs
        return inputs

    def save_outputs(self, outputs, t):
        if t == 0 and self.check:
            save_path = os.path.normpath(self.path + '/../results/outputs.pt')
            output_dir = os.path.dirname(save_path)
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self._to_cpu_detached(outputs), save_path)
            logger.info(
                f"[ scripts/check_results ] save output to {save_path}"
            )
        return outputs
    
    def _to_cpu_detached(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: self._to_cpu_detached(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = [self._to_cpu_detached(v) for v in obj]
            return type(obj)(t)
        return obj

    def _to_device(self,obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: self._to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = [self._to_device(v) for v in obj]
            return type(obj)(t)
        return obj


def load_value(path):
    x = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).cpu().float()
    return torch.tensor(x).cpu().float()

def compare_tensor_outputs(gpu_path, npu_path, atol=1e-5, rtol=1e-5):
    gpu = load_value(gpu_path)
    npu = load_value(npu_path)

    if gpu.shape != npu.shape:
        raise ValueError(f"Shape mismatch: gpu={gpu.shape}, npu={npu.shape}")

    diff = (npu - gpu).abs()

    p99 = torch.quantile(diff, 0.99).item()
    eps = 1e-12
    rel = diff / (gpu.abs() + eps)

    all_ok = torch.allclose(npu, gpu, atol=atol, rtol=rtol)

    py_path = "[ scripts/compare_results ]"
    logger.info(f"{py_path} ==== NPU vs GPU output====")
    logger.info(f"{py_path} files: gpu={gpu_path}")
    logger.info(f"{py_path} files: npu={npu_path}")
    logger.info(f"{py_path} shape: {tuple(gpu.shape)} | numel: {gpu.numel()}")
    logger.info(f"{py_path} allclose(atol={atol}, rtol={rtol}): {all_ok}")
    logger.info(f"{py_path} mean_abs_err  : {diff.mean().item():.6g}")
    logger.info(f"{py_path} mean_rel_err  : {rel.mean().item():.6g}")
    logger.info(f"{py_path} p99_abs_err   : {p99:.6g}")


if __name__ == "__main__":
    compare_tensor_outputs(
        gpu_path="/gpu.pt",
        npu_path="/npu.pt",
        atol=2e-3,
        rtol=2e-3,
    )