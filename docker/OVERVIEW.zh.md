# RecSDK Docker 镜像构建概览与说明

本文档基于 `docker` 目录下的提供的 Dockerfile 文件 (`Dockerfile.ubuntu20.04-tf-py37` 和 `Dockerfile.ubuntu20.04-pt-py311`) 整理得出，旨在帮助开发者快速了解并使用这些容器镜像。

## a) 快速参考

- **基础镜像**: `ubuntu:20.04`
- **支持架构**: `x86_64` (x86) / `aarch64` (ARM)
- **预置编译环境与依赖**: 
  - `GCC` 11.2.0, `CMake` 3.22.6
  - `UCX` & `OpenMPI` 4.1.5
  - `OpenSSL` 1.1.1w & `Python` 3.7.5 
  - `HDF5` 1.10.5 & `h5py` 3.1.0 
- **AI 硬件与驱动支持**: 华为 Ascend CANN 9.0 (并在容器内部提供对应硬件版本的 [Toolkit和相应的算子包](https://www.hiascend.com/developer/download/commercial/result?module=cann))
- **核心功能**: 提供一套开箱即用、依赖环境相互隔离的框架（如 TensorFlow / PyTorch 等）推进推荐系统 SDK (RecSDK) 运行与构建的容器化方案。

## b) 镜像 tag 中关键字段描述

镜像标签遵循以下字段组合规范，便于直观区分镜像内包含的软硬件栈版本细节：

- **OS版本**: 基础操作系统的代号或版本号（例如 `ubuntu20.04`）
- **框架标识**: 应用支持的机器学习框架（例如 `tf` 包含 TF_v1 与 TF_v2 环境，或者是 `torch`）
- **Python版本**: 核心运行的解释器版本（例如 `py37`）
- **CANN版本**: 涵盖的华为计算包大板本级别（例如 `cann9.0`）

> **Tag 命名示例**: `recsdk:ubuntu20.04-tf-py37-x86_64`

## c) Dockerfile 归档路径

目前工程中支持由于底层框架不同而划分的容器场景，对应构建脚本存放于以下路径：

> [!NOTE]
> Dockerfile文件会与昇腾社区以及昇腾镜像仓库同步上线，目前暂未更新

- **TensorFlow 版本**: `docker/Dockerfile.ubuntu20.04-tf-py37`
  - 通过 Python 的原生虚拟环境 (`venv`) 特性，在一个镜像中同时隔离出涵盖 TensorFlow 1.15.0 (`/opt/buildtools/tf1_env`) 与 TensorFlow 2.6.5 (`/opt/buildtools/tf2_env`) 的两套环境。通过安装编译输出产物，集成了适配各版本的 RecSDK 。
- **PyTorch 版本**: `docker/Dockerfile.Dockerfile.ubuntu20.04-pt-py311`
  - 整体环境筹备与资源拉取的流程与 TensorFlow 镜像相仿，核心目标定位于作为支持基于 PyTorch 等相应依赖框架执行构建、装载包和调试 RecSDK 系列工作流的基础容器。

## d) 快速开始

### 运行现有镜像

若您已经成功拉取或者自行构建完成了可用镜像，可以参考如下命令快速启动并进入 Shell 环境：

```bash
docker run -it \
    --name {容器名} \
    --net=host \
    -e ASCEND_VISIBLE_DEVICES=0-7 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v {挂载目录}:{挂载目录} \
    {镜像名}:{镜像tag} \
    /bin/bash
```

参数说明：

- ASCEND_VISIBLE_DEVICES=0-7 当前机器所拥有的npu卡为0-7卡，如果是16卡可以设置为0-15，用户可根据实际情况挂载
- -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ 容器挂载的驱动目录，用户可按照实际驱动的安装目录进行配置
- -v /etc/ascend_install.info:/etc/ascend_install.info 容器需挂载驱动固件安装信息
- -m 300g 设置容器内可用内存，可根据使用情况进行配置

### 本地构建镜像

开发人员可自行在当前系统环境中本地打包构建业务镜像。在此之前需确保外层目录下存在对应的打包产物资源（例：`output/tf_rec_v1_*.tar.gz` 等），因为 Dockerfile 使用了 `COPY` 动作引入它们。

构建基于 TensorFlow 支持版本的镜像：

```bash
docker build -t recsdk_tf:latest -f docker/Dockerfile.Ubuntu.tfrec .
```

若您在编译算子或载入框架包时需要强制适配特定类型的核心类型 (默认为 `a2`)，可以在构建中通过 `CORE_TYPE` 参数显式声明：

```bash
docker build --build-arg CORE_TYPE=a2 -t recsdk_tf_a3:latest -f docker/Dockerfile.Ubuntu.tfrec .
```

### 二次开发与底层环境切换

镜像内部为您准备了便捷的环境切换脚本，适用于需要在多种底层依赖框架或指定芯片类型之间切换的研发及 Debug 工作：

- **切换 Python 框架环境 (TensorFlow 容器)**:

  ```bash
  # 激活切换到 TF 1.15.0 的 RecSDK 开发环境
  source /opt/buildtools/tf1_env/bin/activate  
  # 退出环境
  deactivate tf1_env

  # 激活切换到 TF 2.6.5 的 RecSDK 开发环境
  source /opt/buildtools/tf2_env/bin/activate
  # 退出环境
  deactivate tf2_env
  ```

- **切换 Python 框架环境 (PyTorch 容器)**:

  ```bash
  # 激活切换到 torch_rec_v1 PT 2.6.0 的 RecSDK 开发环境
  source /opt/buildtools/torch_v1_pt2.6.0/bin/activate  
  # 退出环境
  deactivate torch_v1_pt2.6.0

  # 激活切换到 torch_rec_v1 PT 2.7.1 的 RecSDK 开发环境
  source /opt/buildtools/torch_v1_pt2.7.1/bin/activate  
  # 退出环境
  deactivate torch_v1_pt2.7.1

  # 激活切换到 torch_rec_v2 PT 2.7.1 的 RecSDK 开发环境
  source /opt/buildtools/torch_v1_pt2.7.1/bin/activate  
  # 退出环境
  deactivate torch_v2_pt2.7.1
  ```

- **切换 CANN 硬件支持包**:
  
  借助内置切换脚本，您可以动态调整全局与 CANN 包绑定的环境变量及软链接地址：

  ```bash
  source /usr/local/set_cann_env.sh a2  # 切换并生效 Atlas 800T2 训练服务器 配套Toolkit及相关环境变量
  source /usr/local/set_cann_env.sh a3  # 切换并生效 Atlas 800T3 超节点服务器 配套Toolkit及相关环境变量
  source /usr/local/set_cann_env.sh a5  # 切换并生效 Ascend 950 配套Toolkit及相关环境变量(系统默认状态)
  ```

## e) 支持信息与变更说明

### 硬件支持信息

- **物理架构自动适配**: 各个版本的 Dockerfile 天然支持宿主机架构鉴别，不仅提供了涵盖 `x86` 及 `ARM` 的硬件处理，更在构建镜像早期借机 `ARCH=$(uname -m)` 识别所处架构系统，根据逻辑分支抓取和装配对应操作系统的 `GCC` 配置、系统动态链接库和框架分发版本（即不同的 `whl` 安装包、`run` 包）。
- **芯片全兼容部署**: 构建过程规避了仅对单点显卡/算力板卡的局限性。利用预置多版本算力部署方案，将 A2/A3/A5 的相应算力操作栈 (ops 包和 toolkit 包) 一并沉淀安装至镜像底座内，极大提升了容器的分发扩展性与业务泛用能力。

### 兼容性变更说明

- 放弃在全局系统 Python 下混装多个高排他性框架模块并由此产生的依赖灾难（如以往常见的 `npu_bridge` 与 `npu_device` 碰撞以及 Slim 兼容性等问题）风险。新版本脚本全网采取 `python3.7 -m venv` 为基础展开虚拟环境隔离级部署，各框架模块运行相互无感知、互不影响。
- 其中高度耦合硬件逻辑运算的依赖性套件（如涵盖在内的 GCC 11.2, CMake, OpenMPI, Python 等重要运行库），不再使用系统的包管理器发行版，均转而使用了自源码 `make install` 的编译手段构建。

## f) 许可证/免责声明

- 本容器所依赖并打包进内部的所有外部开源层应用源代码、解释器、中间件，均受到各版权方原发行的所属开源许可证（可能包括但不限于 GPL, MIT, Apache 2.0 序列等协议）制约。
- 涉及 Ascend （昇腾）设备的运行套件 `CANN Toolkit` 以及相关的 NPU 的基础依赖驱动资源均由相应产品所有方提供限度使用与约束授权。最终解释权与运行许可范围基于原有提供条款制定，使用者出于任何形式应用（包含且不仅局限于训练、生产系统部署或商业分析时），由此镜像使用过程中产生的技术不兼容以及业务级潜在风险后果，须由最终应用方自主评估及承担责任。
