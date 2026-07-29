# RecSDK Dev Container 使用说明

本目录包含 VS Code Dev Container 配置，用于在容器内编译和调试 RecSDK。

## 环境规格

| 软件          | 版本          |
| ------------- | ------------- |
| 基础镜像      | Debian 12     |
| PyTorch       | 2.6.0         |
| Python        | 3.11.0        |
| fbgemm-gpu    | 1.1.0         |
| GCC           | 11.2.0        |
| CMake         | 3.22.6        |
| CANN          | 9.0.0-beta.2  |
| torch_npu     | 7.1.0         |
| RecSDK-torch  | 26.0.0        |

## 快速开始

### 1. 启动容器

使用 VS Code 打开本仓目录，按 `F1` → **Dev Containers: Reopen in Container**，等待镜像拉取完成。

### 2. 编译 RecSDK

默认已安装torch_rec_v1相关软件包，如需源码编译安装，请参考[源码安装章节](https://gitcode.com/Ascend/RecSDK/blob/develop/docs/zh/torch/torch_rec_v1/recsdk_torch_installation_guide.md#%E6%BA%90%E7%A0%81%E5%AE%89%E8%A3%85)。

## 注意事项

- 容器以 **root** 用户运行，与 CANN toolkit 安装路径 `/usr/local/Ascend` 的权限匹配
- 完整编译及运行需要 **NPU 驱动** 和 **CANN toolkit**（镜像已预装 toolkit，驱动需宿主机提供）
- 工作区目录 `${localWorkspaceFolder}` 以 bind mount 方式挂载到容器内 `/workspace`，修改会实时同步
