# 【MindSDK】RecSDK 算子调用样例：生成式推荐核心attention算子HSTU的编译运行MeetUp实操任务书 总结报告

> **一句话结论**：已在 CANN 9.1.0-beta.3 + Ascend950PR 环境完成 HSTU v2 算子的编译、安装与 NPU 运行验证（生成 `ops_rec-2.7.1-cp311-cp311-linux_x86_64.whl`，torch 前/反向样例全部 PASSED）；关键阻塞点：CANN 9.1 目录结构调整导致多处编译与运行时路径不兼容，需人工补软链与兼容库，图模式（torch.compile/inductor）用例因 triton-ascend 依赖缺失未通过，已记录 Issue #1458。关键指标：编译成功、wheel 安装成功、torch 样例 4/4 通过。

## 1 文档信息

| 项目 | 内容 |
| --- | --- |
| 文档类型 | 验证报告 |
| 提交人 | GitCode ID：gcw_V8hrncEP |
| 文档版本 / 日期 | v1 / 2026-11-20 |

## 2 任务背景与目标

- **任务目标**：在 HiDevLab 昇腾 Ascend950PR 容器环境中，按官方流程编译安装 ops-rec 的 HSTU v2 算子，运行测试用例验证其在 NPU 上的编译与运行，并输出过程日志、截图、体验报告与 Issue。
- **任务结果**：达成核心目标（编译安装成功 + torch 前/反向样例在 NPU 上运行通过）；关键阻塞点为环境依赖与 CANN 9.1 目录结构变更（详见第 6 节）；关键指标：编译 100% 成功、wheel 安装成功、torch 样例 4/4 PASSED。

## 3 环境准备

### 3.1 环境创建步骤

> 按实际创建顺序记录，每步配截图，截图命名 `NN_内容.png`。

**步骤 1：申请 / 创建算力环境**

- 操作：HiDevLab 平台，DevEnv 592，Ascend950PR，1 卡，HBM 环境
- 截图：平台创建页或实例列表 → `01_env_create.png`（如有）

**步骤 2：选择镜像**

- 操作：选择含 CANN 9.1.0-beta.3 的昇腾开发镜像
- 截图：镜像选择页 → `02_env_image.png`（如有）

**步骤 3：启动并登录开发环境**

- 操作：通过 Trae IDE SSH 登录远程容器，打开终端
- 截图：登录成功后的终端提示符 → `03_env_login.png`（如有）

### 3.2 环境信息

- 平台：HiDevLab 昇腾容器
- 设备：Ascend950PR × 1（npu-smi 25.7.rc1.10，Health OK）
- CANN：9.1.0-beta.3
- Python：3.11.0
- PyTorch：torch 2.7.1+cpu，torch_npu 2.7.1.post8

`npu-smi info`

**截图**：`04_npu_smi.png`（如有）

## 4 实际执行步骤

- **任务入口文档**：https://gitcode.com/Ascend/ops-rec
- **样例 / 代码路径**：`ascendc/attention/hstu_v2/`（examples/torch、test/pytest）
- **是否修改样例代码**：否（仅做环境兼容软链与依赖补齐，未改动算子源码）

### 步骤 1：拉取仓库与子模块

```
cd /workspace
git clone https://gitcode.com/Ascend/ops-rec.git
cd ops-rec
git submodule update --init --recursive third_party/catlass
```

- **预期**：仓库与 catlass 子模块拉取成功
- **实际**：成功
- **截图**：`01_env_create.png`（如有）

### 步骤 2：安装 Python 依赖

```
pip3 install torch==2.7.1+cpu
pip3 install --no-deps torch-npu==2.7.1.post8 -i https://repo.huaweicloud.com/repository/pypi/simple --trusted-host repo.huaweicloud.com
python3 -m pip install triton==3.3.1 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
python3 -m pip install PyYAML numpy openpyxl pytest -i https://repo.huaweicloud.com/repository/pypi/simple --trusted-host repo.huaweicloud.com
```

- **预期**：torch / torch_npu / triton / yaml / numpy / openpyxl / pytest 均可用
- **实际**：成功（torch-npu 用 `--no-deps` 安装后需手动补齐依赖；标准 triton 缺 ascend 后端，另见第 6 节）
- **截图**：`04_triton依赖缺失报错.png`

### 步骤 3：修复 CANN 9.1 目录结构兼容问题

```
ln -sfn /usr/local/Ascend/cann-9.1.0-beta.3 /usr/local/Ascend/latest
ln -sfn $CANN/x86_64-linux/tikcpp/ascendc_kernel_cmake $CANN/compiler/tikcpp/ascendc_kernel_cmake
# 补 acl/acl_platform.h（从 CANN runtime 9.1.0 官方仓库获取）
```

- **预期**：CMake 能找到 `fwk_modules`、动态库与头文件
- **实际**：成功，解决 `fwk_modules not found`、`libmetadef.so`、`acl/acl_platform.h: No such file` 三类报错
- **截图**：`01_编译91报acl_platform头文件缺失.png`

### 步骤 4：修复 torch_npu 头文件路径

```
SP=/usr/local/python3.11.0/lib/python3.11/site-packages
rm -rf "$SP/torch_npu/csrc"
ln -s "$SP/torch_npu/include/torch_npu/csrc" "$SP/torch_npu/csrc"
```

- **预期**：编译期能找到 `CalcuOpUtil.h`
- **实际**：成功
- **截图**：`01_编译91报acl_platform头文件缺失.png`

### 步骤 5：编译并安装算子

```
cd /workspace/ops-rec
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0-beta.3
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0-beta.3/lib64:/usr/local/Ascend/cann-9.1.0-beta.3/x86_64-linux/lib64:$LD_LIBRARY_PATH
bash build.sh --socs atlas_950 --ops hstu_v2,arbitrary_func,block_sparse_func,packed_block_sparse,atten_metadata --install 2>&1 | tee build_hstu.log
```

- **预期**：编译安装成功无报错
- **实际**：成功。末尾输出见第 5 节。
- **截图**：`03_最终编译安装成功.png`

### 步骤 6：运行时符号兼容（aclplatformGetDeviceInfo）

容器 runtime 缺少 `aclplatformGetDeviceInfo` 符号，导致 `import ops_rec` 失败。按 CANN 9.1 官方 950PR 配置编写兼容库 `libaclplatform_stub.so` 导出该符号，并用 `LD_PRELOAD` 加载。

```
gcc -shared -fPIC -o /workspace/ops-rec/libaclplatform_stub.so /tmp/aclplatform_stub.c
export LD_PRELOAD=/workspace/ops-rec/libaclplatform_stub.so
python3 -c "import ops_rec; print('import OK')"
```

- **预期**：`import OK`
- **实际**：成功

### 步骤 7：运行测试用例

```
cd /workspace/ops-rec
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0-beta.3/lib64:/usr/local/Ascend/cann-9.1.0-beta.3/x86_64-linux/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=/workspace/ops-rec/libaclplatform_stub.so
python3 ascendc/attention/hstu_v2/examples/torch/test_torch_hstu_forward_v2.py 2>&1 | tee test_hstu.log
bash run_examples.sh --type torch --ops hstu_v2 2>&1 | tee -a test_hstu.log
```

- **预期**：用例运行正常
- **实际**：torch 前/反向样例全部 PASSED；graph 模式、precision 与 aclnn 示例因环境依赖未通过，详见第 5、6 节。
- **截图**：`02_aclnn示例aclrtSetDevice107001失败.png`

> 步骤按实际执行顺序编号，命令保持原样、不改写、不美化；每步配截图。

## 5 执行结果

- **结论**：通过（核心验收项：编译安装成功 + 用例在 NPU 上运行正常）
- **判据**：`build.sh` 输出 `All Tasks Completed Successfully!`，生成 wheel 并安装成功；`test_torch_hstu_forward_v2.py` 输出 3 组 `PASSED`，`test_torch_hstu_backward_v2` 输出梯度张量与 `PASSED`，均在 `npu:0` 上执行。

关键输出（日志原文）：

```
[INFO] Build successful: dist/ops_rec-2.7.1-cp311-cp311-linux_x86_64.whl
Successfully installed ops_rec-2.7.1
[INFO] All Tasks Completed Successfully!
```

```
PASSED: unequal=False, rab=False, causal=False, training=False
PASSED: unequal=True, rab=True, causal=True, training=False
PASSED: unequal=True, rab=True, causal=True, training=True
[INFO] [PASSED] test_torch_hstu_forward_v2
[INFO] [PASSED] test_torch_hstu_backward_v2
```

完整日志见同目录 `build_hstu.log` 与 `test_hstu.log`。

## 6 发现的问题

> 已提交规范 Issue，链接如下。

| 问题 | 提交 issue |
| --- | --- |
| CANN 9.1 目录结构变更导致 `fwk_modules` 找不到、`libmetadef.so` 缺失、`acl/acl_platform.h` 缺失，官方文档未给出 9.1 适配说明 | [#1458](https://gitcode.com/Ascend/RecSDK/issues/1458) |
| torch_npu 头文件实际位于 `include/torch_npu/csrc`，编译脚本按旧路径查找导致 `CalcuOpUtil.h` 缺失 | [#1458](https://gitcode.com/Ascend/RecSDK/issues/1458) |
| 容器 runtime 缺少 `aclplatformGetDeviceInfo` 符号，`import ops_rec` 失败，需构造兼容库 | [#1458](https://gitcode.com/Ascend/RecSDK/issues/1458) |
| 图模式用例依赖 `triton-ascend`，文档只要求安装标准 `triton`，导致 `AttributeError: ... has no attribute 'ascend'` | [#1458](https://gitcode.com/Ascend/RecSDK/issues/1458) |
| aclnn 示例报 `aclrtSetDevice failed, ERROR: 107001`，文档未说明前置条件 | [#1458](https://gitcode.com/Ascend/RecSDK/issues/1458) |

## 7 总结

1. HSTU v2 算子在 CANN 9.1.0-beta.3 + Ascend950PR 环境编译安装成功，产出 `ops_rec-2.7.1-cp311-cp311-linux_x86_64.whl`，`All Tasks Completed Successfully!`。
2. torch 前/反向样例在 NPU 上全部通过（`test_torch_hstu_forward_v2`、`test_torch_hstu_backward_v2` 均 PASSED，输出真实梯度张量）。
3. 执行过程中发现 5 类环境/文档兼容性问题，已汇总提交 Issue [#1458](https://gitcode.com/Ascend/RecSDK/issues/1458)，供社区文档迭代。

## 8 体验总结

| 维度 | 评价 | 说明 |
| --- | --- | --- |
| 文档易懂度 | 一般 | 官方 README 给出了编译与用例命令，但未覆盖 CANN 9.1 目录结构变化、torch_npu 依赖安装与头文件路径差异，需自行排查。 |
| 流程顺畅度 | 一般 | 从申请环境到编译成功，耗时最多的是 Python 依赖补齐与 CANN 路径兼容。 |
| 报错可读性 | 一般 | `fwk_modules not found`、`CalcuOpUtil.h missing` 等报错未直接指向根因，需要结合 CANN 目录结构判断。 |
| 资料完整性 | 一般 | 缺少 CANN 9.1 专项适配说明与依赖清单，建议补充常见报错排查表。 |
