# 昇腾 950PR 单卡环境运行 RecSDK TorchRec-V2 Little Demo 验证报告

> **脱敏说明**：本报告及配套终端日志（`origin_console_output.txt`）中的 SSH 跳板机凭据、IP 地址、实例 ID、容器主机名等敏感信息均已脱敏，以 `<...>` 形式的占位符表示。

**一句话结论**：未达成目标——样例在 HCCL 通信域初始化阶段即失败，训练未开始；关键阻塞点：有，容器分配的物理卡（`/dev/davinci4`）与 HCCL/HDC 实际访问的 `phyId(0)` 不一致，HCCP 取 VNIC IP 时 HDC 会话为空，HCCL 返回 error 19；关键指标：HCCL 初始化 0/1 通过（11 种启动方式与规避变量组合全部失败），基础 NPU 算力 1/1 通过。

## 1 文档信息

| 项目 | 内容 |
| --- | --- |
| 文档类型 | 验证报告 |
| 提交人 | GitCode ID：HungryLuo712 |
| 文档版本 / 日期 | v1 / 2026-11-20 |

> 说明：本报告未附平台页面与终端截图（原截图占位符已删除），以避免账号、实例等敏感信息外泄；全部结论均依据终端日志 `origin_console_output.txt` 中的命令原文与关键输出，可直接复现核对。

## 2 任务背景与目标

- **任务目标**：在 HiDevLab 昇腾 950PR 单卡开发环境（RecSDK MeetUp 镜像）中，按官方文档加载 CANN 与 Python 环境，克隆 RecSDK 仓库，使用 MovieLens-1M 数据集跑通 `torch_rec_v2_examples/little_demo`，并输出可用的训练/导出/加载全流程日志。
- **任务结果**：未达成。`bash run.sh` 在 `main.py` 首个 `dist.barrier()` 处失败，训练一个 step 都未执行；阻塞点为平台侧设备映射缺陷（HCCL/HDC 层），非样例代码或环境变量配置问题。关键指标：HCCL 通信域初始化成功率 **0/1**；基础 NPU 张量运算 **1/1 通过**（`torch.randn(1024,1024).npu()` 矩阵乘正常）。

## 3 环境准备

### 3.1 环境创建步骤

> 按实际创建顺序记录如下。

**步骤 1：申请 / 创建算力环境**

- 操作：HiDevLab 平台创建单卡开发环境（实例 `<实例ID>`，主机名 `<容器主机名>`），1 × Ascend 950PR（HBM 128 GB，`npu-smi` 可见 `5238 / 131072 MB` 已用），256 vCPU / 754 GB 内存。

**步骤 2：选择镜像**

- 操作：选择 RecSDK MeetUp 配套镜像（内含 CANN 9.1.0-beta.3、驱动 25.7.rc1.10、`/opt/buildtools` 下 4 套 torch 环境）。

**步骤 3：启动并登录开发环境**

- 操作：容器启动后通过 SSH 登录（跳板机 + 密钥），登录后提示符：`root@<容器主机名>:/workspace#`，已预装 code-server（6988 端口）。

### 3.2 环境信息

```text
$ npu-smi info
+-------------------------------------------------------------------------------------------------+
| npu-smi 25.7.rc1.10                              Version: 25.7.rc1.10                           |
+--------+------------------+---------------+-----------------------------------------------------+
| NPU ID | Name             | Health        | Power(W)     Temp(C)         Hugepages-Usage(page) |
|        |                  | Bus-Id        | NPU Util(%)  Memory-Usage(MB) HBM-Usage(MB)        |
+========+==================+===============+=====================================================+
| 0      | Ascend950PR      | OK            | 205.5        55              0     / 0             |
|        |                  | 0000:D7:00.0  | 0            0    / 0       5238  / 131072        |
+========+==================+===============+=====================================================+

$ id; uname -a
uid=0(root) gid=0(root) groups=0(root)
Linux <容器主机名> 6.6.0-159.4.3.154.oe2403sp4.x86_64 #1 SMP ... x86_64 GNU/Linux

$ cat /etc/os-release | head -2
PRETTY_NAME="Ubuntu 22.04.5 LTS"
VERSION_ID="22.04"

$ ls -l /dev/davinci* /dev/hisi_hdc
crw-rw-rw-. 1 1000 1000 237, 4 Nov 20 09:01 /dev/davinci4
crw-rw-rw-. 1 1000 1000 238, 0 Nov 20 09:01 /dev/davinci_manager
crw-rw-rw-. 1 1000 1000 234, 0 Nov 20 09:01 /dev/hisi_hdc

$ cat /usr/local/Ascend/driver/version.info | head -2
Version=25.7.rc1.10
ascendhal_version=7.35.23
```

## 4 实际执行步骤

- **任务入口文档**：RecSDK MeetUp quick_start（`source /usr/local/set_cann_env.sh a5` 方案）
- **样例 / 代码路径**：`/workspace/RecSDK/torch_rec_v2_examples/little_demo`
- **是否修改样例代码**：否（仅在 `/tmp` 下另写最小复现脚本，未改动仓库任何文件）

### 步骤 1：加载 CANN 环境

```bash
source /usr/local/set_cann_env.sh a5
```

- **预期**：切入 Ascend 950（A5）配套 Toolkit，`ASCEND_HOME_PATH` 指向 CANN 安装目录。
- **实际**：失败——文档要求的 `/usr/local/set_cann_env.sh` 在本镜像中**不存在**（`No such file or directory`）。改用等价命令后成功，但发现 `a5` 参数被脚本忽略（`set_env.sh` 未处理任何入参）：

```bash
$ source /usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh a5
$ echo $ASCEND_HOME_PATH
/usr/local/Ascend/cann-9.1.0-beta.3
```

### 步骤 2：激活 Python 虚拟环境

```bash
source /opt/buildtools/torch_v2_pt2.7.1/bin/activate
python3 -c "import torch,torch_npu;print(torch.__version__, torch_npu.__version__)"
```

- **预期**：`torch 2.7.1` + `torch_npu 2.7.1.post4`（与文档标注的 2.7.1 略有出入，请以镜像为准）。
- **实际**：成功，输出 `torch 2.7.1+cpu torch_npu 2.7.1.post4`。

### 步骤 3：克隆样例仓库

```bash
cd /workspace
git clone -b develop_examples_and_tools https://gitcode.com/Ascend/RecSDK.git
cd RecSDK/torch_rec_v2_examples/little_demo
```

- **预期**：仓库克隆完成，`little_demo` 目录含 `main.py`、`model.py`、`dataset.py`、`run.sh`。
- **实际**：成功。

### 步骤 4：准备 MovieLens-1M 数据集

```bash
curl -o ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip -q ml-1m.zip
```

- **预期**：得到 `./ml-1m` 目录，`run.sh` 默认数据路径即为该目录。
- **实际**：成功（`ml-1m.zip` 5.9 MB，解压后 `ml-1m/` 就绪）。

### 步骤 5：运行样例

```bash
bash run.sh
```

- **预期**：单卡 1 进程完成训练 → dump → load 三段流程，正常退出。
- **实际**：**失败**，训练未开始。异常栈原文：

```text
[rank0]: Traceback (most recent call last):
[rank0]:   File "/workspace/RecSDK/torch_rec_v2_examples/little_demo/main.py", line 307, in <module>
[rank0]:     main()
[rank0]:   File "/workspace/RecSDK/torch_rec_v2_examples/little_demo/main.py", line 296, in main
[rank0]:     dist.barrier(device_ids=[local_rank])
[rank0]:   File ".../torch/distributed/distributed_c10d.py", line 4635, in barrier
[rank0]:     work = group.barrier(opts=opts)
[rank0]: RuntimeError: create_config:../torch_npu/csrc/distributed/HCCLUtils.cpp:140 NPU function error: device error type 0, error code is 19
[rank0]: [ERROR] 2026-11-20-09:55:38 (PID:19841, Device:0, RankID:0) ERR00100 PTA call acl api failed
```

### 步骤 6：最小化复现（剥离样例代码）

```bash
cat > /tmp/hccl_test.py <<'EOF'
import os, torch, torch_npu
import torch.distributed as dist
print("torch", torch.__version__, "torch_npu", torch_npu.__version__)
dist.init_process_group("hccl")
torch.npu.set_device(0)
dist.barrier(device_ids=[0])
print("BARRIER_OK")
EOF

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:6001 \
  --nnodes=1 --nproc-per-node=1 /tmp/hccl_test.py
```

- **预期**：打印 `BARRIER_OK`，证明 HCCL 通信域可建。
- **实际**：失败，与样例**完全相同的错误码**，且失败点更靠前（`init_process_group` 阶段，尚未进入任何集合通信）：

```text
[rank0]: RuntimeError: create_config:../torch_npu/csrc/distributed/HCCLUtils.cpp:140
NPU function error: device error type 0, error code is 19
```

- **结论**：排除样例代码、`dynamic_emb`、数据加载因素，问题在 HCCL 层。

### 步骤 7：打开 Ascend plog，定位真实根因

```bash
export ASCEND_PROCESS_LOG_PATH=/tmp/ascend_log
source /usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh
source /opt/buildtools/torch_v2_pt2.7.1/bin/activate
timeout 180 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:6001 \
  --nnodes=1 --nproc-per-node=1 /tmp/hccl_test.py
grep -rhE '\[ERROR\]' /tmp/ascend_log | tail -40
```

- **预期**：定位到具体失败的子系统。
- **实际**：plog 给出完整因果链——HDC 会话为空 → 取 VNIC IP 失败 → 网络 API 异常 → 通信域创建失败 → 19：

```text
[ERROR] HCCP [ra_hdc.c:390] HdcSendRecvPkt : [send_recv][pkt]session is NULL!, phyId(0)
[ERROR] HCCP [ra_hdc.c:494] RaHdcProcessMsg : [process][ra_hdc_msg]hdc_send_recv_pkt opcode(94) failed ret(-22) phyId(0)
[ERROR] HCCP [ra_hdc_socket.c:515] RaHdcGetVnicIpInfos : [get][ip_infos]ra hdc message process failed ret(-22), phyId(0)
[ERROR] HCCL [orion_adapter_hccp.cc:809][hrtRaGetSocketVnicIpInfo]ra get VnicIp fail. ret[128303]
[ERROR] HCCL [communicator_impl.cc:217]Network api exception: call hrtRaGetSocketVnicIpInfo failed, ret=128303
[ERROR] HCCL [op_base_v2.cc:236][CreateCommConfigRootInfo]opbasedCommInfoV2.pComm->Init failed, errNo[0x0000000005000013]
[ERROR] HCCL [op_base_v2.cc:1831][HcclCommInitRootInfoConfigV2]errNo[0x0000000005000013] and create comm failed.
[ERROR] HCCL [op_base.cc:1974][operator()]call trace: hcclRet -> 19
```

- **对照**：容器实际拿到的设备节点是 `/dev/davinci4`（物理卡 4），而 HCCL/HDC 全程按 `phyId(0)` 访问会话，`gRaHdc[0]` 从未初始化 → `session is NULL`。基础算力走的是 ACL 通路，不经过 HDC 会话，所以矩阵乘正常、只有通信域建不起来。

### 步骤 8：规避尝试矩阵（均无效，供后人省时间）

| # | 尝试方式 | 结果 |
| --- | --- | --- |
| 1 | 原样 `bash run.sh`（torchrun + c10d） | 失败，error 19 |
| 2 | 最小脚本 `torchrun` | 失败，error 19 |
| 3 | 最小脚本，普通 python + 手工 `RANK/WORLD_SIZE/MASTER_ADDR` | 失败，error 19 |
| 4 | `init_process_group(device_id=torch.device("npu:0"))` + 先 `set_device` | 失败，error 19 |
| 5 | `HCCL_INTRA_ROCE_ENABLE=0` | 失败，error 19 |
| 6 | `HCCL_INTRA_PCIE_ENABLE=1` | 失败，error 19 |
| 7 | 5 + 6 组合 | 失败，error 19 |
| 8 | `HCCL_ALGO=Ring` | 失败，error 1（HCCL_E_PARA） |
| 9 | `HCCL_OP_EXPANSION_MODE=HOST` | 失败，error 1（HCCL_E_PARA） |
| 10 | `pkill -9 python` 清残留后重跑 | 失败，error 19 |
| 11 | `HCCL_SOCKET_IFNAME=lo` / `unset HCCL_IF_IP` 等网络变量 | 失败，error 19 |

> 说明：`npu-smi` Health 为 OK、无残留进程，说明不是卡故障；`HCCL_*` 环境变量无法绕过，因为失败点在 HDC 会话层，早于网络选卡。

## 5 执行结果

- **结论**：**不通过**
- **判据**：任务未正常结束，进程在 HCCL 通信域初始化阶段退出（exit code 1），训练 step 数为 0，无 checkpoint 产出。

关键输出（日志原文）：

```text
[rank0]: RuntimeError: create_config:../torch_npu/csrc/distributed/HCCLUtils.cpp:140
NPU function error: device error type 0, error code is 19
[ERROR] 2026-11-20-09:55:38 (PID:19841, Device:0, RankID:0) ERR00100 PTA call acl api failed

[ERROR] HCCP [ra_hdc.c:390] HdcSendRecvPkt : [send_recv][pkt]session is NULL!, phyId(0)
[ERROR] HCCL [orion_adapter_hccp.cc:809] ra get VnicIp fail. ret[128303]
[ERROR] HCCL [op_base_v2.cc:1831] errNo[0x0000000005000013] and create comm failed.
```

## 6 发现的问题

| 问题 | 提交 issue |
| --- | --- |
| 950PR 单卡容器分配的物理卡（`/dev/davinci4`）与 HCCL/HDC 使用的 `phyId(0)` 不一致，HCCP 取 VNIC IP 时 HDC 会话为空，任何 HCCL 通信域初始化都返回 error 19（与样例无关，最小 all_reduce 同样复现） | 上游根因分析：[Ascend/pytorch#5324](https://gitcode.com/Ascend/pytorch/issues/5324)；修复 PR：[cann/hcomm#4985](https://atomgit.com/cann/hcomm/pull/4985) |
| RecSDK quick_start 文档要求 `source /usr/local/set_cann_env.sh a5`，但 MeetUp 镜像中该文件不存在；且 `set_env.sh` 不识别 `a5` 参数，文档未给替代命令与说明 | [Ascend/RecSDK#1455](https://gitcode.com/Ascend/RecSDK/issues/1455)（同一现象，1 小时前提交） |
| 报错可读性差：`error code is 19` 只提示"HCCL 调用失败"，真正原因（`session is NULL, phyId(0)`）藏在 `ASCEND_PROCESS_LOG_PATH` 的 plog 里；文档未提供环境自检与 error 19 FAQ | 同上，建议在该 issue 中补充 |
| 本环境同现象（建议提交人另提一条，正文引用上述两条即可） | <待提交> |

## 7 总结

1. 样例代码与环境配置本身没有问题：纯 NPU 计算（1024×1024 矩阵乘）通过，`torch 2.7.1 + torch_npu 2.7.1.post4` 加载正常，失败点 100% 落在 HCCL 通信域初始化，训练 step 数 0。
2. 根因是平台侧设备映射缺陷：容器拿到 `/dev/davinci4`（物理卡 4），HCCL 却按 `phyId(0)` 建立 HDC 会话 → `session is NULL` → `hrtRaGetSocketVnicIpInfo` 返回 128303 → error 19；11 种启动方式/环境变量组合全部无法绕过。
3. 可用规避路径只有两条：① 重建环境直到分配到物理卡 0（官方技术支持确认卡 0 可通过）；② 升级 CANN/950-ops 到含 hcomm #4985 修复的版本（issue #320 记录 9.3.0 weekly 构建实测通过）。在当前容器内无法自行修复。

## 8 体验总结

| 维度 | 评价 | 说明 |
| --- | --- | --- |
| 文档易懂度 | 差 | 第一步 `source /usr/local/set_cann_env.sh a5` 就因文件不存在而卡住；文档未说明该文件与 `set_env.sh` 的关系，也未提示 `a5` 参数无效 |
| 流程顺畅度 | 差 | 申请环境、选镜像、克隆、下数据都顺利（约 10 分钟），但最后一步 100% 失败，且失败与操作无关，无法通过重试或调参绕过 |
| 报错可读性 | 差 | `error code is 19` / `PTA call acl api failed` 指向"HCCL 调用失败"，容易把人引向 `HCCL_SOCKET_IFNAME`、`GLOO_SOCKET_IFNAME` 等网络变量（实测均无效）；真正原因 `session is NULL, phyId(0)` 只在 plog 中可见 |
| 整体评价 | 2 / 5 | 环境与样例配套完整、算力可用，但单卡 HCCL 不可用导致 Demo 无法跑通，属平台侧阻塞 |

- **最耗时的一步**：步骤 7（打开 plog 定位根因）——约 40 分钟；期间在步骤 5/6/8 反复排错约 30 分钟，全部无效。
- **最值得改进的点**：① 平台侧修复物理卡映射（或允许用户指定物理卡），并在镜像发布前跑一遍单卡 HCCL all_reduce 自检；② quick_start 增加"环境自检"小节（`npu-smi` + 10 行 HCCL 探针）与 error 19 FAQ；③ 修正 `set_cann_env.sh a5` 的命令与说明。

---

### 附录：环境自检 / 最小复现脚本（10 秒出结论）

```bash
source /usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh
source /opt/buildtools/torch_v2_pt2.7.1/bin/activate

cat > /tmp/hccl_test.py <<'EOF'
import torch, torch_npu
import torch.distributed as dist
torch.npu.set_device(0)
x = torch.randn(64, 64).npu()
print("BASIC_NPU_MATH=", float((x @ x).sum()), flush=True)   # 基础算力
dist.init_process_group("hccl")                              # 通信域
dist.barrier(device_ids=[0])
print("HCCL_BARRIER_OK")                                     # 期望看到这一行
EOF

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:6001 \
  --nnodes=1 --nproc-per-node=1 /tmp/hccl_test.py
```

判定：出现 `HCCL_BARRIER_OK` 即环境已修复，可直接 `cd /workspace/RecSDK/torch_rec_v2_examples/little_demo && bash run.sh`；若仍报 `error code is 19`，按第 7 节两条规避路径处理。
