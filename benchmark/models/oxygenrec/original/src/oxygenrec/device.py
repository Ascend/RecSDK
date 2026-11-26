"""GPU/NPU共用的最小运行时设备适配层。

模型和损失函数不应按平台分叉；这里只处理设备注册、可用性、随机种子、同步和
显存统计。NPU插件仅在显式请求``npu``设备时导入，避免GPU环境依赖torch_npu。
"""

from __future__ import annotations

import importlib

import torch


SUPPORTED_DEVICE_TYPES = ("cpu", "cuda", "npu")


def resolve_device(device_spec: str) -> torch.device:
    """解析并验证CPU、CUDA或NPU设备；NPU注册发生在构造torch.device之前。"""

    device_type = device_spec.split(":", 1)[0]
    if device_type not in SUPPORTED_DEVICE_TYPES:
        raise ValueError(
            f"unsupported device type {device_type!r}; expected one of "
            f"{SUPPORTED_DEVICE_TYPES}"
        )
    if device_type == "npu":
        try:
            importlib.import_module("torch_npu") # 请求 NPU 时动态导入 torch_npu
        except ImportError as error:
            raise RuntimeError(
                "npu device requested but torch_npu cannot be imported"
            ) from error

    device = torch.device(device_spec)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda device requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(device)
    elif device.type == "npu":
        npu = getattr(torch, "npu", None)
        if npu is None or not npu.is_available():
            raise RuntimeError("npu device requested but torch.npu.is_available() is false")
        npu.set_device(device)
    return device


def seed_torch(seed: int, device: torch.device) -> None:
    """设置CPU与所选加速器的PyTorch随机种子。"""

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "npu":
        torch.npu.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    """在需要计时或读取主机值时同步当前加速器。"""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        torch.npu.synchronize(device)


def reset_peak_memory_stats(device: torch.device) -> bool:
    """重置峰值显存统计；CPU或后端未提供接口时返回False。"""

    backend = getattr(torch, device.type, None)
    function = getattr(backend, "reset_peak_memory_stats", None)
    if not callable(function):
        return False
    function(device)
    return True


def max_memory_allocated(device: torch.device) -> int | None:
    """返回峰值已分配显存字节数；不支持时返回None。"""

    backend = getattr(torch, device.type, None)
    function = getattr(backend, "max_memory_allocated", None)
    if not callable(function):
        return None
    return int(function(device))


def device_name(device: torch.device) -> str:
    """返回可公开记录的设备名称。"""

    if device.type == "cpu":
        return "CPU"
    backend = getattr(torch, device.type)
    try:
        return str(backend.get_device_name(device))
    except (TypeError, ValueError):
        return str(backend.get_device_name(device.index or 0))
