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

"""KVCacheManager 测试脚本共用的性能打点工具（可选开启）。

默认关闭。关闭时 ``instrument()`` 原样返回 manager，不做任何 monkey-patch、
不做设备同步、不注册退出钩子，功能用例行为与打点前完全一致。

开启方式：
    KV_PERF=1 python test/test_native.py

也可把下面的 ``ENABLED_BY_DEFAULT`` 改为 True 强制开启。

打点内容：对 manager 的公共接口在调用前后各做一次设备同步，记录端到端耗时
（含 kernel / D2H / H2D），按“阶段 + 接口名”聚合，进程退出时打印。
"""

import atexit
import os
import time

_TRUTHY = {"1", "true", "yes", "on"}

# 代码级总开关：环境变量 KV_PERF 未设置时使用该值。
ENABLED_BY_DEFAULT = False

# 在 KVCacheManager 上打点的接口。
_MANAGER_IFACES = (
    "lookup_kvcache",
    "allocate_kvcache",
    "onboard_launch",
    "offload_launch",
    "evict",
)
# 在设备侧子 manager上打点的接口。
_DEVICE_IFACES = ("put", "get")
# 纯 CPU 轮询原语：只计自身耗时，不做设备同步，避免同步开销淹没轮询。
_UNSYNCED_IFACES = ("offload_try_wait",)

_ENABLED = False
_CONFIGURED = False
_SYNC = None
_DEVICE = ""
_PHASE = "-"
_STATS = {}


def is_enabled():
    """环境变量 KV_PERF 优先；未设置时回退到 ENABLED_BY_DEFAULT。"""
    value = os.environ.get("KV_PERF")
    if value is None:
        return ENABLED_BY_DEFAULT
    return value.strip().lower() in _TRUTHY


def configure(device_sync, device_name="", enabled=None):
    """初始化打点（幂等）。应在测试模块导入时调用一次。

    device_sync: 设备同步函数，如 torch.cuda.synchronize / torch.npu.synchronize
    device_name: 报表中的设备标签，如 "GPU" / "NPU"
    enabled:     显式开关；为 None 时由 is_enabled() 决定
    """
    global _ENABLED, _CONFIGURED, _SYNC, _DEVICE
    if _CONFIGURED:
        return _ENABLED
    _ENABLED = is_enabled() if enabled is None else bool(enabled)
    _SYNC = device_sync
    _DEVICE = device_name
    _CONFIGURED = True
    if _ENABLED:
        atexit.register(report)
    return _ENABLED


def set_phase(name):
    """设置当前阶段标签，用于把打点结果按阶段聚合。"""
    global _PHASE
    if _ENABLED:
        _PHASE = name


def _record(iface, dt):
    rec = _STATS.setdefault((_PHASE, iface), [0, 0.0, float("inf"), 0.0])
    rec[0] += 1
    rec[1] += dt
    rec[2] = min(rec[2], dt)
    rec[3] = max(rec[3], dt)


def _timed(iface, fn, sync=True):
    def wrapper(*args, **kwargs):
        if sync:
            _SYNC()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        if sync:
            _SYNC()
        _record(iface, time.perf_counter() - t0)
        return out

    return wrapper


def instrument(mgr):
    """对 manager 实例的接口打点；关闭时原样返回，不做任何修改。"""
    if not _ENABLED:
        return mgr
    if not _CONFIGURED:
        raise RuntimeError("perf_probe.configure() must be called before instrument()")
    for name in _MANAGER_IFACES:
        setattr(mgr, name, _timed(name, getattr(mgr, name)))
    for name in _UNSYNCED_IFACES:
        setattr(mgr, name, _timed(name, getattr(mgr, name), sync=False))
    dev_mgr = getattr(mgr, "npu_kvcache_mgr", None)
    if dev_mgr is None:
        dev_mgr = getattr(mgr, "gpu_kvcache_mgr")
    for name in _DEVICE_IFACES:
        setattr(dev_mgr, name, _timed(name, getattr(dev_mgr, name)))
    return mgr


def report():
    """打印聚合结果：先分阶段，再给每个接口一个全程汇总。关闭或无数据时不输出。"""
    if not _STATS:
        return
    print("\n[PERF] device=%s" % (_DEVICE or "unknown"))
    print(
        "[PERF] %-6s%-20s%8s%14s%14s%14s%14s"
        % ("phase", "interface", "count", "total_ms", "mean_ms", "min_ms", "max_ms")
    )
    for (phase, iface), (count, total, minimum, maximum) in _STATS.items():
        print(
            "[PERF] %-6s%-20s%8d%14.3f%14.3f%14.3f%14.3f"
            % (phase, iface, count, total * 1e3, total / count * 1e3, minimum * 1e3, maximum * 1e3)
        )

    # 全程汇总：按接口合并所有阶段的样本，均值 = 总耗时 / 总调用次数。
    overall = {}
    for (_, iface), (count, total, minimum, maximum) in _STATS.items():
        rec = overall.setdefault(iface, [0, 0.0, float("inf"), 0.0])
        rec[0] += count
        rec[1] += total
        rec[2] = min(rec[2], minimum)
        rec[3] = max(rec[3], maximum)

    print("[PERF] ---- overall (all phases) ----")
    print(
        "[PERF] %-6s%-20s%8s%14s%14s%14s%14s"
        % ("phase", "interface", "count", "total_ms", "mean_ms", "min_ms", "max_ms")
    )
    for iface, (count, total, minimum, maximum) in overall.items():
        print(
            "[PERF] %-6s%-20s%8d%14.3f%14.3f%14.3f%14.3f"
            % ("ALL", iface, count, total * 1e3, total / count * 1e3, minimum * 1e3, maximum * 1e3)
        )
