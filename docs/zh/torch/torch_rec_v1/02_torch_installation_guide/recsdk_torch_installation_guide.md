# 安装部署<a name="ZH-CN_TOPIC_0000002336148865"></a>

## 安装说明<a name="ZH-CN_TOPIC_0000002302229632"></a>

### 配套版本<a id="section146113514599"></a>

当前Rec SDK Torch支持两种配套版本，后续安装时请安装对应配套版本的软件包。

| 配套版本 | Python | PyTorch | TorchNPU | fbgemm\_gpu | Rec SDK Torch | 备注 |
| ---- | ------ | ------- | ---------- | ----------- | ------------- |----|
| 方案一  | 3.11+  | 2.6.0   | 2.6.0      | 1.1.0+cpu   | 1.1.0         |此方案暂不支持Ascend 950系列产品|
| 方案二  | 3.11+  | 2.7.1   | 2.7.1      | 1.2.0+cpu   | 1.2.0         ||

> [!NOTE]
>
> - PyTorch：深度学习训练框架，相关资料请参见[PyTorch文档](https://docs.pytorch.org/docs/stable/index.html)。
> - TorchNPU：PyTorch框架适配NPU设备的扩展插件，相关资料请参见[TorchNPU](https://gitcode.com/Ascend/pytorch)。
> - fbgemm\_gpu：TorchRec框架依赖的加速库，相关资料请参见[fbgemm\_gpu](https://github.com/pytorch/FBGEMM)。

### 依赖软件说明

本章节提前声明依赖软件列表和安装方式，详细安装步骤请参见后续[安装Rec SDK Torch](#section182972951211)章节并按需选择。

**依赖关系介绍**

各层级软件依赖关系如下表（除宿主机层外，其他层级均在容器内）。

| 层级         | 组件名称            | 依赖项                                 | 说明               |
| ---------- | --------------- | ----------------------------------- | ---------------- |
| **应用层**    | Rec SDK Torch   | hybrid\_torchrec、torchrec\_embcache | Rec SDK Torch组件  |
| **自定义算子层** | 自定义算子相关包        | rec\_ops、fbgemm\_ascend             | 自定义算子实现          |
| **适配层**    | Torchrec框架适配NPU | torchrec\_npu                       | TorchRec的适配NPU版本 |
| **依赖层**    | Torchrec依赖      | fbgemm\_gpu                         | TorchRec依赖的底层加速库 |
| **框架层**    | 训练框架            | PyTorch、TorchNPU                 | 深度学习训练框架         |
| **使能层**    | NPU使能           | CANN                                | 提供NPU底层支持        |
| **宿主机层**   | 宿主机依赖           | NPU固件、驱动、Device配置                   | 硬件层面的基础依赖        |

#### 宿主机依赖

Rec SDK Torch基于NPU环境运行，如下为宿主机依赖软件说明。若宿主机未安装相关软件，请根据说明安装。

| 依赖名称/操作               | 推荐版本                     | 获取方式/安装说明                                                                                                                                                                                                                                                            |
| --------------------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 昇腾硬件产品驱动          | Ascend HDK 26.0.RC1及补丁版本 | 请参考[《CANN快速安装》](https://www.hiascend.com/cann/download)安装昇腾NPU驱动，并配置环境变量。         |
| Ascend Docker Runtime | MindCluster 7.3.0        | 若宿主机未安装Docker，请参见[Docker社区或官网](https://docs.docker.com/engine/install/)先安装Docker。请参见《MindCluster 集群调度用户指南》的“安装 > [安装部署](https://www.hiascend.com/document/detail/zh/mindcluster/730/clustersched/dlug/dlug_installation_009.html)”章节下载和安装`Ascend Docker Runtime`软件包。 |

#### 容器内训练框架依赖

| 依赖名称/操作            | 推荐版本        | 获取方式/安装说明                                                                                                                                                                                                                                                                                                                                    |
| ------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CANN软件包            | CANN 9.0.0  | 请参考[《CANN快速安装》](https://www.hiascend.com/cann/download)安装昇腾CANN软件包（包含Toolkit和ops包），并配置环境变量。                                                                                                                                                                                                                                                                                      |
| PyTorch和TorchNPU | 2.6.0/2.7.1 | 容器内依赖，若容器内未安装，请在容器内安装。<br>2.6.0版本：请前往[TorchNPU2.6.0下载](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=168&ids=8e2241c30951478a85bb6f2fc282c568%2C97%2C108%2C1%2C6%2C3%2C)页面进行获取。<br>2.7.1版本：请前往[TorchNPU2.7.1下载](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=175&ids=89dda9ba9de741349efa03687a487678%2C96%2C108%2C1%2C6%2C177%2C)页面获取。<br>安装时请根据PyTorch版本、Python版本（建议使用Python 3.11）、CPU架构选择对应的安装指令。<br>如需卸载，可通过`pip3 uninstall -y torch_npu torch`指令进行卸载。 |

#### 容器内训练加速库依赖<a id="section146113514600"></a>

原生TorchRec框架依赖fbgemm\_gpu库。基于NPU环境运行时，需安装fbgemm\_gpu库的CPU版本。

由于其CPU版本无法通过requirements依赖安装，因此需手动安装并在安装时指定安装源。

根据[配套版本](#section146113514599)，安装对应版本的fbgemm\_gpu软件包。

| 软件版本      | 安装指令                                                                         |
| --------- | ---------------------------------------------------------------------------- |
| 1.1.0+cpu | `pip3 install fbgemm_gpu==1.1.0+cpu -i https://download.pytorch.org/whl/cpu` |
| 1.2.0+cpu | `pip3 install fbgemm_gpu==1.2.0+cpu -i https://download.pytorch.org/whl/cpu` |

### Rec SDK Torch软件包说明

Rec SDK Torch软件包如下表：

| 名称                       | 说明                                       |
| ------------------------ | ---------------------------------------- |
| torch\_rec\_v1-\*.tar.gz | Rec SDK Torch一键安装部署软件包（已包含TorchRec昇腾注册包） |
| rec\_ops                 | 自定义算子包                                   |
| fbgemm\_ascend           | fbgemm自定义算子包及PyTorch框架适配层                |

## 安装Rec SDK Torch<a id="section182972951211"></a>

可通过如下三种方案安装Rec SDK Torch软件包：

- 方案一：镜像安装
  - 基于宿主机配置、镜像制作，并在容器内安装Rec SDK Torch软件包。介绍宿主机环境配置、基础容器镜像制作、运行容器和安装Rec SDK Torch软件包的完整操作说明。
  - **推荐使用该方案**。
- 方案二：源码安装
  - 基于**容器内**安装Rec SDK Torch软件包，默认已配置完成宿主机环境并进入Docker容器内。介绍如何通过源码编译的方式安装Rec SDK Torch软件包。
  - 若使用的Docker容器镜像不是参考[基础镜像构建](../../build_torch_rec_images/README.md)制作，**可能存在cmake、glibc等基础软件版本不兼容**的情况，需自行处理。
- 方案三：离线安装
  - 基于**容器内**安装Rec SDK Torch软件包，默认已配置完成宿主机环境并进入Docker容器内，介绍如何通过下载Release版本的二进制包安装Rec SDK Torch软件包及自定义算子包。
  - 若使用的Docker容器镜像不是参考[基础镜像构建](../../build_torch_rec_images/README.md)制作，**可能存在cmake、glibc等基础软件版本不兼容**的情况，需自行处理。

如需查看Rec SDK Torch软件包的历史安装记录，请参见[查看Rec SDK Torch安装与卸载记录](../06_common_operations/common_operations.md)。

> \[!NOTE]
> 对于用户集成的开源和第三方软件，漏洞和问题请自行跟踪社区并及时进行修复；可以并且不限于通过[CVE（通用漏洞字典）官网](https://www.cve.org/)确认对应开源软件版本的已知漏洞，并通过版本升级、使用patch补丁包更新等方式修复。

### 镜像安装<a id="ZH-CN_TOPIC_0000002302389237"></a>

#### 简要步骤说明

1. 配置宿主机环境。
2. 构建基础镜像。
3. 启动Docker容器。
4. 安装Rec SDK Torch。

#### 操作流程图

基于容器部署Rec SDK Torch，操作流程图请参见[图1](#fig1345216415476)。

**图 1**  配置容器内的开发环境及训练镜像构建<a id="fig1345216415476"></a>

![](../../../figures/torch_rec_v1/配置容器内的开发环境及训练镜像构建.png "配置容器内的开发环境及训练镜像构建")

#### 详细操作步骤

1. 宿主机环境配置

   请参见[宿主机依赖](#宿主机依赖)章节完成宿主机环境配置。
2. 制作基础训练镜像<a id="section104919392501"></a>

   可直接下载已经制作好的基础训练镜像，请参见[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5)中的“镜像下载”标签页，下载26.0.0*及之后版本的镜像。

   从[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5)获取镜像的Docker指令示例如下：

   - 获取x86_64镜像: `docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/rec_sdk-torch:26.0.0_debian12-x86`
   - 获取aarch_64镜像: `docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/rec_sdk-torch:26.0.0_openeuler2203-arm`

   也可手动制作基础训练镜像，请参见[基础镜像构建](../../build_torch_rec_images/README.md)里的Dockerfile和README制作镜像。

   制作基础镜像时，会同时安装[容器内训练框架依赖](#容器内训练框架依赖)和[容器内训练加速库依赖](#容器内训练加速库依赖)中的依赖软件（CANN、PyTorch、TorchNPU、fbgemm\_gpu）。后续安装Rec SDK Torch时可跳过`依赖软件安装`步骤。
3. 运行容器<a id="ZH-CN_TOPIC_0000002302389300"></a>

   创建启动脚本run\_docker.sh，参考如下：

   ```bash
   #!/bin/bash
   container_name=$1
   image_name=$2
   docker run \
   -it \
   --name ${container_name} \
   --shm-size="300g" \
   -m 300g \
   -v /etc/localtime:/etc/localtime:ro \
   -e ASCEND_VISIBLE_DEVICES=0-7 \
   -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
   ${image_name} \
   /bin/bash
   ```

   > \[!NOTE]
   > 部分参数含义：
   >
   > - -m 300g 表示设置容器内可以使用的内存大小上限为300G，可根据实际情况进行配置。
   > - -e ASCEND\_VISIBLE\_DEVICES=0-7 表示将服务器上编号为device0\~device7的NPU设备挂载到容器内，可根据实际情况进行配置。

   执行如下命令新建容器并进入容器内：

   ```shell
   bash run_docker.sh 容器名 镜像名称:镜像版本
   ```

   > \[!NOTE]
   >
   > 1. 上述指令中Docker容器为前台运行，退出交互后容器将停止。更多Docker容器使用请参见[Docker社区文档](https://docs.docker.com/)。
   > 2. 若创建容器时，使用的镜像是从昇腾镜像仓库下载，且镜像版本为26.1.0及之后版本，需参考[刷新环境变量](../03_quick_start/quick_start.md#refresh_container_env)章节，手动刷新容器内环境变量。
   > 3. 如需更新容器内依赖软件版本，请参见[容器内训练框架依赖](#容器内训练框架依赖)和[容器内训练加速库依赖](#容器内训练加速库依赖)中的说明，卸载依赖软件后重新安装。
4. 安装Rec SDK Torch软件包

   安装Rec SDK Torch可参考后文中[源码安装](#源码安装)或者[离线安装](#离线安装)章节，并可跳过其中的`依赖软件安装`步骤（制作基础镜像时已安装容器内的相关依赖）。

### 源码安装

该方案默认已配置完成宿主机环境并进入Docker容器内。

1. 依赖软件安装

   请参见[容器内训练框架依赖](#容器内训练框架依赖)和[容器内训练加速库依赖](#容器内训练加速库依赖)完成容器内的依赖软件安装。
2. 安装TorchRec昇腾注册包

   TorchRec昇腾注册包是基于TorchRec源码做的NPU设备适配。可通过Rec SDK Torch提供的patch文件和TorchRec源码的固定分支编译出该注册包。

   请参见[README](https://gitcode.com/Ascend/RecSDK/blob/develop/training/torch_rec_v1/torchrec_npu/README.md)进行源码编译和安装。
3. 安装Rec SDK Torch推荐算法框架包<a id="source_build_hybrid_torchrec"></a>

   请参见如下指令进行编译安装：

   ```bash
   git clone https://gitcode.com/ascend/RecSDK.git
   cd RecSDK/training/torch_rec_v1/hybrid_torchrec
   bash build_whl.sh
   cd dist
   pip3 uninstall -y hybrid_torchrec
   pip3 install hybrid_torchrec-*.whl
   pip3 install -r requirements.txt
   pip3 uninstall -y torchrec_embcache
   pip3 install torchrec_embcache-*-py3-none-linux*.whl
   ```

4. 安装自定义算子相关包<a id="install_custom_op"></a>

   下载[RecSDK](https://gitcode.com/Ascend/RecSDK)源码，按如下指令进行算子相关包的编译和安装：

   > [!NOTE]
   > 编译脚本 `build_ai_core_op.sh` 的参数需根据实际芯片架构配置，默认以 `A2` 为例。详细参数和更多说明请参考 [算子编译README](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/build/README.md)。

   ```bash
   # 编译算子前，需使能CANN环境变量。默认路径安装CANN包时，使能CANN环境变量指令如下：
   source /usr/local/Ascend/cann/set_env.sh
   unset ASCEND_CUSTOM_OPP_PATH

   # 编译并安装算子包（Ascend-recsdk-npu-ops-\*-linux-\*.tar.gz）。
   cd RecSDK/cust_op/ascendc_op/build
   bash build_ai_core_op.sh A2

   # 可选：若仅需安装部分算子，可在其他容器内编译，并将build/output/recsdk_ops路径下所需算子包拷贝到当前环境，参考如下指令安装：
   # bash mxrec_opp_split_embedding_codegen_forward_unweighted.run

   # 安装算子适配层（libfbgemm_npu_api.so）
   cd ../../framework/torch_plugin/torch_library/common/
   bash build_ops.sh
   ```

   安装算子run包的参数如[表1](#table14435173717221)所示。

   **表 1**  参数说明 <a id="table14435173717221"></a>

   | 输入参数                   | 说明                                     |
   | ---------------------- | -------------------------------------- |
   | --help \| -h           | 查询帮助信息。                                |
   | --info                 | 查询安装包的信息。                              |
   | --list                 | 查询安装包的文件列表。                            |
   | --check                | 查询压缩包完整性。                              |
   | --quiet                | 静默安装方式。                                |
   | --nox11                | 不启动xterm终端。                            |
   | --noexec               | 不执行嵌入的安装脚本。                            |
   | --extract=\<path>      | 直接解压到目标目录，通常与--noexec配合使用，仅解压文件而不运行脚本。 |
   | --tar arg1 \[arg2 ...] | 通过**tar**命令访问压缩包内容。                    |
   | --install-path         | 安装到指定目录路径。                             |

   > \[!NOTE]
   >
   > 安装算子后，/usr/local/Ascend/cann/opp/vendors/目录下会生成split\_embedding\_codegen\_forward\_unweighted、backward\_codegen\_adagrad\_unweighted\_exact、asynchronous\_complete\_cumsum、permute2d\_sparse\_data等文件夹。如果没有相关文件夹，请使用**unset ASCEND\_CUSTOM\_OPP\_PATH**取消环境变量后重新安装算子。

### 离线安装

该方案默认已配置完成宿主机环境并进入Docker容器内。

该方案的安装步骤和[源码安装](#源码安装)章节类似。区别在于Rec SDK Torch一键安装部署软件包可直接从Release版本获取，无需从源码编译。

1. 依赖软件安装

   请参见[容器内训练框架依赖](#容器内训练框架依赖)和[容器内训练加速库依赖](#容器内训练加速库依赖)完成容器内的依赖软件安装。
2. 安装Rec SDK Torch一键安装部署软件包

   **下载软件包<a name="section1852417242717"></a>**

   请参考本章获取所需软件包和对应的数字签名文件，下载本软件即表示您同意[华为企业业务最终用户许可协议（EULA）](https://e.huawei.com/cn/about/eula)的条款和条件。

   > \[!NOTE]
   > 当前Release软件包为Rec SDK whl包，一键安装部署软件包待资源下载中心上线后更新。

   | 组件名称                   | 软件包                      | 获取链接                                               |
   | ---------------------- | ------------------------ | -------------------------------------------------- |
   | Rec SDK Torch一键安装部署软件包 | torch\_rec\_v1-\*.tar.gz | [Release下载页](https://gitcode.com/Ascend/RecSDK/releases) |

   > \[!NOTE]
   > 当前提供的Rec SDK一键安装部署软件包基于Python 3.11版本编译，**请在相同的Python版本环境下安装使用**。若需在其他Python版本环境下安装使用，请参见[源码编译 - 安装Rec SDK Torch一键安装部署软件包](#source_build_hybrid_torchrec)进行源码编译。

   **软件包Hash值校验<a name="section10830205518487"></a>**

   为了防止软件包在传递过程中或存储期间被恶意篡改，请同时下载软件包和软件包对应的`.sha256sum`的文件到同一目录，并执行如下指令进行完整行校验：

   ```bash
   # FILE_NAME需替换为软件包文件名
   sha256sum -c FILE_NAME.sha256sum
   ```

   **安装软件包**

   将下载的软件包上传到Docker容器内，实现方式可参考:
   - 方式1：在启动容器时，指定一个宿主机目录挂载到容器内，并将下载的软件包放在其中，使容器可以访问到下载的软件包。

     将宿主机目录挂载到Docker容器内，在[启动容器](#ZH-CN_TOPIC_0000002302389300)时增加如下参数（`dir1`修改为需要挂载的目录）：

     ```bash
     -v /dir1:/dir1
     ```

   - 方式2：在宿主机上，使用**docker cp**指令将软件包拷贝到容器内。

     **docker cp**指令示例：

     ```bash
     docker cp host_file_path container_name:container_file_path
     ```

     其中，host\_file\_path为宿主机文件路径，container\_name为待拷入的docker容器名称，container\_file\_path为待拷入的docker容器内的文件路径。

   执行如下指令进行安装：

   ```shell
   # 如已安装，请先卸载
   pip3 uninstall -y torch_rec_v1
   # 安装软件包
   pip3 install torch_rec_v1-{version}-{arch}.tar.gz
   ```

   （其中 `{version}` 代表版本号，`{arch}` 代表操作系统架构，请根据实际安装包替换）
3. 安装自定义算子相关包

   参考fbgemm\_ascend的[README](https://gitcode.com/Ascend/fbgemm-ascend/blob/main/README.md)获取fbgemm\_ascend软件包。

   rec\_ops自定义算子包参考[Release下载页](https://gitcode.com/Ascend/RecSDK/releases)获取。

   ```shell
   # 安装框架依赖算子包 fbgemm_ascend
   pip3 install fbgemm_ascend-*.whl
   # 安装自定义算子包 rec_ops
   pip3 install rec_ops*.whl
   ```

## 安装验证

可通过执行已有用例/模型验证Rec SDK Torch是否安装成功。

**单卡验证**

请参见[快速入门](../03_quick_start/quick_start.md#搭建模型)中“搭建模型”、“启动模型训练”章节，进行模型搭建及启动单卡训练。

**多卡验证**

请参见[快速入门](../03_quick_start/quick_start.md#搭建模型)中“搭建模型”、“启动模型训练”章节，进行模型搭建及启动多卡训练。

**框架用例验证**

hybrid\_torchrec用例列表和运行方式请参见[README](../../../../../training/torch_rec_v1/hybrid_torchrec/test/st/README.md)。

torchrec\_embcache用例列表和运行方式请参见[README](../../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/README.md)。

用例执行后，若显示`xx passed`且没有`xx failed`则说明用例执行通过，Rec SDK Torch安装成功。

## 配置环境变量<a name="ZH-CN_TOPIC_0000002336268805"></a>

Rec SDK Torch环境变量的说明如[表1](#table126401659163820)所示。

**表 1**  环境变量 <a id="table126401659163820"></a>

| 环境变量名                               | 含义                                                  | 可选/必选 | 说明                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ----------------------------------- | --------------------------------------------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| INPUT\_DIST\_THREADS                | Rec SDK Torch使用分桶任务的线程池并发数量。                        | 可选    | 整数，默认为6，取值范围：\[1, 12]。                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| POST\_INPUT\_THREADS                | Rec SDK Torch使用哈希去重任务的线程池并发数量。                      | 可选    | 整数，默认为6，取值范围：\[1, 12]。                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| MASTER\_ADDR                        | 用于指定分布式训练中主节点的IP地址。                                 | 可选    | IPv4地址，单机时推荐使用127.0.0.1，多机时需修改为可进行多机通信的IPv4地址。                                                                                                                                                                                                                                                                                                                                                                                                           |
| MASTER\_PORT                        | 用于指定分布式训练中的侦听端口。                                    | 可选    | 整数，取值范围：\[0，65520]。                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| LOCAL\_RANK                         | 当前进程在本机上的NPU编号。                                     | 可选    | 整数，取值范围：\[0，world\_size -1]。                                                                                                                                                                                                                                                                                                                                                                                                                             |
| WORLD\_SIZE                         | 参与训练的device数量。                                      | 可选    | 整数，取值范围：\[1，8]。                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ASCEND\_VISIBLE\_DEVICES            | 昇腾处理器可见的设备，来指定程序只使用其中的部分设备。                         | 必选    | 使用ASCEND\_VISIBLE\_DEVICES环境变量指定训练中的NPU设备（用户可执行ls /dev/ \| grep davinci\*命令查询宿主机的NPU设备），使用设备序号指定设备，支持单个和范围指定且支持混用。例如：<ul> <li>ASCEND\_VISIBLE\_DEVICES=0：表示将0号设备（/dev/davinci0）挂载入容器中。</li><li>ASCEND\_VISIBLE\_DEVICES=1,3：表示将1、3号设备挂载入容器中。</li><li>ASCEND\_VISIBLE\_DEVICES=0-2：表示将0号至2号设备（包含0号和2号）挂载入容器中，效果同ASCEND\_VISIBLE\_DEVICES=0,1,2。</li><li>ASCEND\_VISIBLE\_DEVICES=0-2,4：表示将0号至2号以及4号设备挂载入容器，效果同ASCEND\_VISIBLE\_DEVICES=0,1,2,4。</li></ul> |
| ASCEND\_OPP\_PATH                   | 算子库根目录。                                             | 必选    | 执行CANN环境变量配置脚本时设置，不建议用户修改。默认为/usr/local/Ascend/cann/opp。                                                                                                                                                                                                                                                                                                                                                                                                 |
| GLOO\_SOCKET\_IFNAME                | gloo通信网卡配置。                                         | 可选    | 使用**ifconfig**或**ip a**命令查看服务器网卡名称，推荐配置为lo，多机时需改为可进行多机通信的网卡名称。                                                                                                                                                                                                                                                                                                                                                                                           |
| ENABLE\_FAST\_HASHMAP               | 是否启用快速哈希表                                           | 可选    | 字符串，支持"true"、"yes"、"1"表示启用，其他值表示不启用，默认为false。                                                                                                                                                                                                                                                                                                                                                                                                            |
| EMB\_MEMORY\_POOL\_SIZE             | 快速哈希表的embedding内存池大小                                | 可选    | 整数，默认为102400，取值范围：\[1, 200000]。                                                                                                                                                                                                                                                                                                                                                                                                                          |
| FAST\_HASHMAP\_RESERVE\_BUCKET\_NUM | 快速哈希表预留桶数量                                          | 可选    | 整数，默认为2097152，取值范围：\[128, 4294967291]。                                                                                                                                                                                                                                                                                                                                                                                                                   |
| EMB\_MEMORY\_POOL\_THREAD\_NUM      | 快速哈希表embedding内存池处理线程数                              | 可选    | 整数，默认为4，取值范围：\[1, 1024]。                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| EMBCACHE\_SIZE\_ON\_DEVICE\_MEM     | HBM embedding缓存大小（单位：字节）                            | 可选    | 整数，默认为17179869184（16GB），取值范围：\[1, 设备可用内存]。                                                                                                                                                                                                                                                                                                                                                                                                               |
| DO\_EC\_LOCAL\_UNIQUE               | 多级缓存是否启用EC local unique                             | 可选    | 字符串，支持"true"、"1"、"yes"表示启用，其他值表示不启用。默认为false。                                                                                                                                                                                                                                                                                                                                                                                                            |
| LOCAL\_UNIQUE\_PARALLEL\_BATCH\_NUM | EmbCacheTrainPipelineSparseDist中Local unique并行处理批次数 | 可选    | 整数，默认为2，取值范围：\[1, 24]。                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ENABLE\_PARALLEL\_GLOBAL\_UNIQUE    | 是否启用并行Global Unique处理                               | 可选    | 字符串，1表示启用，其他值表示不启用。默认为0，表示不启用。                                                                                                                                                                                                                                                                                                                                                                                                                           |
| GLOG\_stderrthreshold               | 设置多级缓存C++模块的日志级别。                                   | 可选    | 整数，默认为1（WARN）。取值范围：<ul><li>-2：TRACE</li><li>-1：DEBUG</li><li>0：INFO</li><li>1：WARN</li><li>2：ERROR</li></ul>                                                                                                                                                                                                                                                                                                                                                   |

## 卸载<a name="ZH-CN_TOPIC_0000002302389376"></a>

用户如需移除Rec SDK Torch软件包和TorchRec昇腾注册包，可根据安装方式参考以下命令进行卸载。

- **基于一键安装部署软件包（离线安装）方式安装**：

    ```bash
    # 卸载Rec SDK Torch主包，并连带卸载TorchRec昇腾注册包
    pip3 uninstall torch_rec_v1 -y
    ```

- **基于源码编译方式安装**：

    ```bash
    # 卸载Rec SDK Torch推荐算法框架包
    pip3 uninstall hybrid_torchrec -y
    pip3 uninstall torchrec_embcache -y
    # 卸载TorchRec昇腾注册包
    pip3 uninstall torchrec -y
    ```

用户如需移除Rec SDK Torch自定义算子相关包，可参考以下命令进行卸载。其中，

- 自定义算子在CANN中的默认安装路径为/usr/local/Ascend/cann/opp/vendors/
- 卸载自定义算子时，删除vendors路径下自定义算子名称对应的文件夹即可。可以通过[ai\_core\_op](https://gitcode.com/Ascend/RecSDK/tree/develop/cust_op/ascendc_op/ai_core_op)查看Rec SDK的自定义算子目录。

```bash
# 源码编译安装方式卸载
# 卸载算子指令示例（仅列出部分，其他算子卸载指令同理）
rm -rf /usr/local/Ascend/cann/opp/vendors/asynchronous_complete_cumsum
rm -rf /usr/local/Ascend/cann/opp/vendors/backward_codegen_adagrad_unweighted_exact
rm -rf /usr/local/Ascend/cann/opp/vendors/permute2d_sparse_data
rm -rf /usr/local/Ascend/cann/opp/vendors/split_embedding_codegen_forward_unweighted
# 卸载libfbgemm_npu_api.so
PACKAGE_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")
if [ -d "$PACKAGE_PATH" ]; then
  cd ${PACKAGE_PATH}
  rm -rf libfbgemm_npu_api.so
else
  echo "no site-package"
fi

# 基于Release版本安装方式卸载
pip3 uninstall rec_ops -y
pip3 uninstall fbgemm_ascend -y
```

## 升级<a name="ZH-CN_TOPIC_0000002302389340"></a>

用户如需将当前版本的Rec SDK Torch升级至最新版本，可将最新的Rec SDK Torch软件包上传至安装环境后，在软件包所在目录下使用命令进行升级，具体命令如下。

1. 升级Rec SDK Torch新版本时，需要先手动卸载旧版本。具体操作请参考[卸载](#卸载)。
2. 参考[安装Rec SDK Torch](#section182972951211)重新安装Rec SDK Torch。
