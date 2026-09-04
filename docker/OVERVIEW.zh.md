# RecSDK Docker 镜像构建概览与说明

> [English](https://gitcode.com/Ascend/RecSDK/blob/develop/docker/OVERVIEW.md) | 中文

本文档基于 [docker](https://gitcode.com/Ascend/RecSDK/blob/develop/docker) 目录下的提供的 Dockerfile 文件整理得出，旨在帮助开发者快速了解并使用这些容器镜像。

## 快速参考

- 从哪里获取帮助
  - [issue 反馈](https://gitcode.com/Ascend/RecSDK/issues)
  - [RecSDK 代码](https://gitcode.com/Ascend/RecSDK)
  - [RecSDK 文档](https://gitcode.com/Ascend/RecSDK/tree/develop/docs)

## RecSDK 简介

RecSDK作为面向互联网市场搜索推荐广告的应用使能SDK产品，对于搜索推荐广告模型训练的应用场景需求，提供基于昇腾平台的搜索推荐广告框架，支撑大规模搜推广场景，助力完成搜推广模型的高效训练。

RecSDK的功能涉及：

1. 模型训练基础功能。支持单机单卡训练、多机多卡分布式训练。
2. 推荐场景特有功能。基于RecSDK的稀疏表方案，RecSDK提供必备功能，如特征保存和加载、特征准入、特征淘汰等。
3. 大规模稀疏表特有功能。支持加速卡内存、主机内存、主机磁盘多级存储、支持多机存储、支持动态扩容。规模可超10TB。

## 镜像 tag 中关键字段描述

镜像标签遵循以下字段组合规范，便于直观区分镜像内包含的软硬件栈版本细节：

- **RecSDK版本**: RecSDK版本号（例如`26.1.0`）
- **CANN版本**: CANN软件包版本（例如 `cann9.1.0`）
- **芯片标识**: 目标昇腾芯片平台（`910b` 对应 Atlas 800T A2 / a2，`a3` 对应 Atlas 800T A3，`950` 对应昇腾950代际 / a5）
- **OS版本**: 基础操作系统的代号或版本号（例如 `ubuntu20.04`）
- **Python版本**: 核心运行的解释器版本（例如 `py3.7`）
- **框架标识**: 镜像支持的机器学习框架（例如 `tf` 为TensorFlow环境，或者是 `pt` 为PyTorch环境）

> **注意**：同一 Dockerfile 通过 `--build-arg CORE_TYPE=a2/a3/a5` 可构建出不同芯片平台的镜像，构建时请根据目标平台替换 tag 中的芯片标识字段。例如：
>
> - `CORE_TYPE=a2` → tag 中使用 `910b`
> - `CORE_TYPE=a3` → tag 中使用 `a3`
> - `CORE_TYPE=a5` → tag 中使用 `950`

## Dockerfile 归档路径

目前工程中支持由于底层框架不同而划分的容器场景，对应构建脚本存放于以下路径：

### RecSDK 26.1.0

| Tag | Dockerfile |
|-----|------------|
|26.1.0-cann9.1.0-{chip}-ubuntu20.04-py3.7-tf|[Dockerfile](https://gitcode.com/Ascend/RecSDK/blob/develop/docker/Dockerfile.26.1.0-cann9.1.0-ubuntu20.04-py3.7-tf)|
|26.1.0-cann9.1.0-{chip}-ubuntu22.04-py3.11-pt|[Dockerfile](https://gitcode.com/Ascend/RecSDK/blob/develop/docker/Dockerfile.26.1.0-cann9.1.0-ubuntu22.04-py3.11-pt)|
|26.1.0-cann9.1.0-{chip}-openEuler22.03-py3.7-tf|[Dockerfile](https://gitcode.com/Ascend/RecSDK/blob/develop/docker/Dockerfile.26.1.0-cann9.1.0-openEuler22.03-py3.7-tf)|
|26.1.0-cann9.1.0-{chip}-openEuler22.03-py3.11-pt|[Dockerfile](https://gitcode.com/Ascend/RecSDK/blob/develop/docker/Dockerfile.26.1.0-cann9.1.0-openEuler22.03-py3.11-pt)|

> `{chip}` 替换为目标芯片标识：`910b`（Atlas 800T A2）、`a3`（Atlas 800T A3）、`950`（昇腾950代际产品）。历史版本所有Tag请参考[Support Tags](https://gitcode.com/Ascend/RecSDK/blob/develop/docker/supported_tags.md)。

## 快速开始

### 前提条件

| 项目 | 要求 |
|------|------|
| Docker 版本 | 建议 20.10 及以上，需支持 `--net=host` 网络模式 |
| 宿主操作系统 | Ubuntu 20.04 / 22.04（x86_64 或 ARM）、openEuler 22.03（x86_64 或 ARM） |
| 昇腾驱动与固件 | 宿主机需安装 Ascend NPU 驱动及固件，驱动路径默认为 `/usr/local/Ascend/driver` |
| CANN 版本 | 建议 CANN 9.1.0 及以上（可参照Dockerfile配置下载链接，默认下载9.1.0版本） |
| 磁盘空间 | 构建镜像建议预留至少 60 GB 可用空间 |
| 网络 | 构建过程中需访问外部网络以下载依赖包 |

### 运行现有镜像

若您已经成功拉取或者自行构建完成了可用镜像，可以参考如下命令快速启动并进入 Shell 环境：

```bash
docker run -it \
    --name {容器名} \
    --net=host \
    -m 300g \
    -e ASCEND_VISIBLE_DEVICES=0-7 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v {挂载目录}:{挂载目录} \
    {镜像名}:{镜像tag} \
    /bin/bash
```

参数说明：

- ASCEND_VISIBLE_DEVICES=0-7 当前机器所拥有的npu卡为0-7卡，如果是16卡可以设置为0-15，用户可根据实际情况挂载
- -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro 容器挂载的驱动目录（只读），用户可按照实际驱动的安装目录进行配置
- -v /etc/ascend_install.info:/etc/ascend_install.info 容器需挂载驱动固件安装信息
- -m 300g 设置容器内可用内存，可根据使用情况进行配置

### 本地构建镜像

开发人员可自行在当前系统环境中本地打包构建业务镜像，Dockerfile 已内置软件包的下载与安装逻辑，无需额外准备外部资源。用户可根据需要自行修改 Dockerfile 中的软件包版本（CANN软件包版本及RecSDK软件包版本）。

- **PyTorch 镜像（以 ubuntu22.04 为例）**

```bash
# 构建时通过 --build-arg CORE_TYPE 指定芯片平台，tag 中对应替换 {chip} 为 910b/a3/950
docker build --build-arg CORE_TYPE=a2 -t recsdk_pt:26.1.0-cann9.1.0-910b-ubuntu22.04-py3.11-pt -f docker/Dockerfile.26.1.0-cann9.1.0-ubuntu22.04-py3.11-pt .
```

- **TensorFlow 镜像（以 ubuntu20.04 为例）**

```bash
docker build --build-arg CORE_TYPE=a2 -t recsdk_tf:26.1.0-cann9.1.0-910b-ubuntu20.04-py3.7-tf -f docker/Dockerfile.26.1.0-cann9.1.0-ubuntu20.04-py3.7-tf .
```

若您在编译算子或载入框架包时需要强制适配特定核心类型，可以在构建中通过 `CORE_TYPE` 参数显式声明（默认为 `a2`）：

| CORE_TYPE | 适用平台 | tag 芯片标识 |
|-----------|----------|-------------|
| `a2` | Atlas 800T A2 训练服务器 | `910b` |
| `a3` | Atlas 800T A3 超节点服务器 | `a3` |
| `a5` | 昇腾950代际产品 | `950` |

```bash
# CORE_TYPE可选a2/a3/a5，以下以 TensorFlow 镜像为例，PyTorch 镜像同理
docker build --build-arg CORE_TYPE=a2 -t recsdk_tf:26.1.0-cann9.1.0-910b-ubuntu20.04-py3.7-tf -f docker/Dockerfile.26.1.0-cann9.1.0-ubuntu20.04-py3.7-tf .
```

- **openEuler 镜像**

openEuler 22.03 镜像的构建方式与上述 Ubuntu 示例一致。

### 二次开发与底层环境切换

镜像内部为您准备了便捷的环境切换脚本，适用于需要在多种底层依赖框架或指定芯片类型之间切换的研发及 Debug 工作：

- **如何进行二次开发**:

  ```dockerfile
  # 以recsdk_tf:26.1.0-cann9.1.0-910b-ubuntu20.04-py3.7-tf镜像为基础镜像，叠加用户软件
  FROM recsdk_tf:26.1.0-cann9.1.0-910b-ubuntu20.04-py3.7-tf
  RUN apt update -y && \
      apt install ...
  ```

- **切换 Python 框架环境 (TensorFlow 容器)**:

  ```bash
  # 激活切换到 TF 1.15.0 的 RecSDK 开发环境
  source /opt/buildtools/tf1_env/bin/activate

  # 激活切换到 TF 2.6.5 的 RecSDK 开发环境
  source /opt/buildtools/tf2_env/bin/activate
  ```

- **切换 Python 框架环境 (PyTorch 容器)**:

  ```bash
  # 激活切换到 torch_rec_v1 PT 2.6.0 的 RecSDK 开发环境（仅26.1.0版本镜像支持）
  source /opt/buildtools/torch_v1_pt2.6.0/bin/activate

  # 激活切换到 torch_rec_v1 PT 2.7.1 的 RecSDK 开发环境
  source /opt/buildtools/torch_v1_pt2.7.1/bin/activate

  # 激活切换到 torch_rec_v1 PT 2.10.0 的 RecSDK 开发环境（仅26.2.0及之后版本镜像支持）
  source /opt/buildtools/torch_v1_pt2.10.0/bin/activate

  # 激活切换到 torch_rec_v2 PT 2.7.1 的 RecSDK 开发环境
  source /opt/buildtools/torch_v2_pt2.7.1/bin/activate

  # 激活切换到 torch_rec_v2 PT 2.10.0 的 RecSDK 开发环境
  source /opt/buildtools/torch_v2_pt2.10.0/bin/activate
  ```

  使用 `deactivate` 退出当前虚拟环境。

## 支持信息与变更说明

### 硬件支持信息

- **物理架构自动适配**: 各个版本的 Dockerfile 天然支持宿主机架构鉴别，不仅提供了涵盖 `x86` 及 `ARM` 的硬件处理，更在构建镜像早期借助 `ARCH=$(uname -m)` 识别所处架构系统，根据逻辑分支抓取和装配对应操作系统的 `GCC` 配置、系统动态链接库和框架分发版本（即不同的 `whl` 安装包、`run` 包）。
- **芯片按需构建**: 构建过程通过 `--build-arg CORE_TYPE=a2/a3/a5` 指定目标芯片平台，仅安装对应芯片架构的 CANN toolkit 和 ops 包，避免镜像体积膨胀。

### 兼容性变更说明

- 放弃在全局系统 Python 下混装多个框架模块的做法，避免不同模块之间的依赖冲突。新版本采用 `python3.7 -m venv` 为每个框架创建独立的虚拟环境，各框架模块运行相互隔离、互不影响。
- 其中高度耦合硬件逻辑运算的依赖性套件（如涵盖在内的 GCC 11.2, CMake, OpenMPI, Python 等重要运行库），不再使用系统的包管理器发行版，均转而使用了自源码 `make install` 的编译手段构建。

## 快速验证

### 端到端验证流程示例

以下以 TensorFlow 镜像为例，展示从构建到验证的完整流程：

```bash
# 1. 构建镜像
docker build --build-arg CORE_TYPE=a2 -t recsdk_tf:26.1.0-cann9.1.0-910b-ubuntu20.04-py3.7-tf \
  -f docker/Dockerfile.26.1.0-cann9.1.0-ubuntu20.04-py3.7-tf .

# 2. 启动容器
docker run -it --name recsdk_test \
  --net=host \
  -m 300g \
  -e ASCEND_VISIBLE_DEVICES=0-7 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  recsdk_tf:26.1.0-cann9.1.0-910b-ubuntu20.04-py3.7-tf /bin/bash

# 3. 容器内验证 NPU 可用
npu-smi info

# 4. 激活环境并运行 little demo
source /opt/buildtools/tf1_env/bin/activate
# 参考下方 little demo 链接运行验证
```

### TensorFlow镜像验证

可参考[little demo](https://gitcode.com/Ascend/RecSDK/blob/develop_examples_and_tools/examples/demo/README.md)进行little demo验证，验证前可通过[章节 - 二次开发与底层环境切换](#二次开发与底层环境切换)切换容器内python虚拟环境。

### PyTorch镜像验证

可参考[little demo](https://gitcode.com/Ascend/RecSDK/blob/develop_examples_and_tools/torch_examples/little_demo/README.md)进行little demo验证，验证前可通过[章节 - 二次开发与底层环境切换](#二次开发与底层环境切换)切换容器内python虚拟环境。

## 常见问题与故障排查

> [!NOTE]
> 如遇到其他问题，请在 [RecSDK Issues](https://gitcode.com/Ascend/RecSDK/issues) 提交反馈。

### 运行容器时 NPU 不可见

**现象**：容器内执行 `npu-smi info` 无输出或报错。

**原因**：

- 宿主机未安装 NPU 驱动或驱动版本不匹配
- 未挂载驱动目录 `/usr/local/Ascend/driver`
- `ASCEND_VISIBLE_DEVICES` 未设置或设置为不存在的卡号

**解决**：

1. 在宿主机执行 `npu-smi info` 确认驱动正常
2. 检查 `docker run` 是否包含 `-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro` 和 `-v /etc/ascend_install.info:/etc/ascend_install.info`
3. 确认 `ASCEND_VISIBLE_DEVICES` 的值在宿主机 NPU 数量范围内

### 非特权容器NPU资源占用冲突<a id="non_privileged_container_npu_resource_conflict"></a>

**现象**：容器内执行 `npu-smi info` 时报错：“dcmi model initialized failed, because the device is used. ret is -8020”。

**原因**：存在多个非特权容器挂载了相同的NPU卡。非特权容器对 NPU 设备的访问采用独占式挂载机制，多个容器挂载了相同的NPU卡时，容器内无法访问NPU设备资源。

**解决**：

先使用`docker stop <container_name>`停止NPU挂载冲突的非特权容器，再使用如下方案修改容器启动指令并重新启动容器。

- 方案1：NPU卡差异挂载。各个非特权容器分别挂载不同的NPU卡，避免资源冲突。可通过**修改容器启动指令**中的ASCEND_VISIBLE_DEVICES参数值并重新创建容器。例如`ASCEND_VISIBLE_DEVICES=0-1`，表示容器仅挂载NPU 0卡和1卡。
- 方案2：使用特权容器模式启动容器。特权容器可避免设备独占问题，适用于对隔离性要求不高的场景。**该模式下可能存在安全风险，需使用者自行评估**！启动特权容器方式：在容器启动指令中添加`--privileged`参数。例如`docker run xxx`改为`docker run --privileged xxx`。

### 容器内存不足

**现象**：训练任务 OOM 或运行缓慢。

**原因**：`-m`参数限制的可用内存过小。

**解决**：

1. 增大`-m`值（如`-m 500g`），或移除该参数以使用宿主机全部内存
2. 确认宿主机有足够空闲内存

### 环境切换后框架版本未生效

**现象**：`source activate` 后 `python -c "import torch; print(torch.__version__)"` 仍显示旧版本。

**原因**：其他 Python 环境在 `PATH` 中优先匹配。

**解决**：

1. 先用 `deactivate` 退出当前虚拟环境，再 `source` 新环境
2. 用 `which python` 确认路径指向 `/opt/buildtools/` 下的虚拟环境

## 许可证/免责声明

- 查看这些镜像中包含的 RecSDK 和 Mind 系列软件的[许可证信息](https://gitcode.com/Ascend/RecSDK/blob/develop/LICENSE)。
- 与所有容器镜像一样，预装软件包（Python，系统库等）可能受其自身许可证约束。
