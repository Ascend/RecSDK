# 完成Rec SDK HSTU算子编译体验验证
> **一句话结论**：HSTU-v1、HSTU-v2 的 forward / backward 算子均已编译成功，产出 `ops_rec-2.10.0` whl 包；过程中定位并解决了 `tla/layout.hpp` 头文件缺失与 `torch_npu` 缺失两类环境问题；修复后 `test_aclnn_hstu` 在 Ascend950PR 上 4/4 用例 PASSED。

## 1 文档信息
| 项目 | 内容 |
| ---- | ---- |
| 文档类型 | 验证报告 |
| 提交人 | GitCode ID: gcw_Yowt27rj |
| 文档版本 / 日期 | v2 / 2026-11-21 |

## 2 任务背景与目标
- **任务目标**：在开发环境按照官方文档跑通HSTU AscendC算子编译流程，输出编译产物与验证日志
- **任务结果**：HSTU-v1 / HSTU-v2 的 forward 与 backward 算子全部编译通过，生成 `ops_rec-2.10.0-cp311-cp311-linux_x86_64.whl`；补齐依赖与运行环境后，`test_aclnn_hstu` 的 4 个用例（causal × rab 组合）全部 PASSED。
- **结论变化说明**：初版报告记录的“HSTU-v2 编译失败”是在 `third_party/catlass` 子模块未同步、且环境缺少 `torch_npu` 的条件下得到的，属于环境问题而非源码缺陷；依赖补齐后同一份源码可正常编译与运行，本版已按复测结果更新。

## 3 环境准备
### 3.1 环境创建步骤
> 按实际创建顺序记录，每一步配截图，截图命名 `NN_内容.png`
**步骤 1：申请 / 创建算力环境**
- 操作：OpenLab算力实例，Ascend950PR，单卡
- 截图：平台创建页或实例列表 → `01_env_create.png`

**步骤 2：选择镜像**
- 操作：基础ascend-toolkit开发镜像（仅含编译工具链，不带torch_npu）
- 截图：镜像选择页 → `02_env_image.png`

**步骤 3：启动并登录开发环境**
- 操作：启动实例，WebIDE登录开发环境
- 截图：WebIDE开发界面 → `03_env_webide.png`

### 3.2 环境信息
- Python版本：3.11.0
- CANN版本：cann-9.1.0-beta.3
- 默认PyTorch版本：torch 2.15.0+cpu（不含 `torch_npu`，编译 torch 扩展不可用）
- 验证用PyTorch版本：torch 2.10.0 + torch_npu 2.10.0（`/opt/buildtools/torch_v2_pt2.10.0`）
- npu-smi版本：25.7.rc1.10
- NPU芯片：Ascend950PR，单卡，Health OK

## 4 任务执行过程
1. 拉取RecSDK仓库，切换至experimental分支
2. 加载CANN环境变量：`source /usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh`
3. **同步子模块**：`git submodule update --init --recursive`，补齐 `third_party/catlass`（`tla/layout.hpp` 实际位于 `third_party/catlass/include/tla/layout.hpp`）
4. **配置 torch 编译环境**：切换至 `torch 2.10.0 + torch_npu 2.10.0`；关闭后端自动加载避免 import 失败；注入 `acl/acl_platform.h` 兼容头文件
5. 进入ops-rec目录，执行单算子编译：`bash build.sh --ops hstu_v2 --socs atlas_950 -j1`
6. 编译结果：`HstuForwardV2_ascend950`、`HstuBackwardV2_ascend950` 均 `[100%] Built target`，适配层 `ops_rec_atlas_950.so` 产出，最终 `All Tasks Completed Successfully!`
7. **运行验证**：将 `hstu_v2` 加入 CANN `opp/vendors/config.ini` 的 `load_priority`，执行 `test_aclnn_hstu`，4/4 用例 PASSED

完整编译日志节选见附件 `compile_log.txt`（原始完整日志 50276 行）。

## 5 问题与阻塞分析
### 问题1：`fatal error: 'tla/layout.hpp' file not found`
**现象**：编译 HSTU-v2 kernel 时 TBE 预编译报错：
```
fatal error: 'tla/layout.hpp' file not found
#include "tla/layout.hpp"
```
**根因**：`tla/layout.hpp` 来自 `third_party/catlass` 子模块，仓库 `op_kernel/CMakeLists.txt` 已经通过 `-I${CATLASS_INCLUDE_DIR}` 挂载了该路径，但初次编译时子模块内容为空，导致头文件找不到；此外 `catlass_hstu/tla_hstu/layout.hpp` 内部同样 `#include "tla/layout.hpp"`，依赖同一搜索路径。
**处理**：执行 `git submodule update --init --recursive` 后重新编译，`HstuForwardV2` 与 `HstuBackwardV2` 均编译通过，无需修改源码。
**结论**：该问题**不是** HSTU-v2 算子源码缺少 `tla` 依赖，而是子模块未初始化的构建环境问题。

### 问题2：torch 扩展编译失败（torch_npu 缺失 / C++20 要求）
**现象**：
- `fatal error: torch_npu/csrc/.../NPUFunctions.h: No such file or directory`
- `#error C++20 or later compatible compiler is required to use ATen`
- `fatal error: acl/acl_platform.h: No such file or directory`
**根因**：默认 python 环境为 `torch 2.15.0+cpu`，未安装 `torch_npu`，且该版本 torch 头文件要求 C++20，而工程按 `cxx_std_17` 编译。
**处理**：改用 `/opt/buildtools/torch_v2_pt2.10.0`（torch 2.10.0 + torch_npu 2.10.0），通过 `PYTHONPATH` 注入；设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 规避后端自动加载失败；用 `CPATH` 提供 `acl/acl_platform.h`。处理后适配层 `.so` 正常编译。

### 问题3：运行时 `binary_info_config.json ... does not support opType [HstuForwardV2]`
**现象**：`aclnnHstuForwardV2GetWorkspaceSize failed. ERROR: 161001`。
**根因**：自定义算子目录已就位，但 CANN `opp/vendors/config.ini` 的 `load_priority` 中没有 `hstu_v2`，vendor 未被加载。
**处理**：将 `hstu_v2` 追加到 `load_priority`，并把 `vendors/hstu_v2/op_api/lib` 加入 `LD_LIBRARY_PATH`。处理后 4/4 用例 PASSED。

## 6 总结与建议
本次体验：HSTU-v1 与 HSTU-v2 的 forward / backward 算子均完成 AscendC 编译，产出完整 whl 包；补齐子模块与 `torch_npu` 环境后，`test_aclnn_hstu` 四个用例（causal=0/1 × rab=0/1）全部 PASSED（`output sample` 与参考实现一致，进程返回 0）。此前的编译失败均由环境引起，已在本文档记录定位过程与修复方式。

建议：
1. 代码/构建侧：在 README 中显式提示首次编译需执行 `git submodule update --init --recursive`，否则 `tla/layout.hpp` 会报缺失；同时说明 `tla` 头文件实际路径为 `third_party/catlass/include/tla`。
2. 文档侧：在环境准备章节增加镜像/环境选型说明：
    - 纯编译镜像：只带 ascend-toolkit，可编 kernel，但装不了 torch 扩展；
    - 全量验证镜像：需配套 `torch` 与 `torch_npu` 版本匹配（本次为 torch 2.10.0 + torch_npu 2.10.0），才能加载算子并执行 pytest / aclnn 用例。
3. 用例侧：补充运行前置检查（`torch`/`torch_npu` 版本、`opp/vendors/config.ini` 是否包含目标 vendor），避免出现“编译成功但运行找不到算子”的偏差。
