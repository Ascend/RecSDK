# 安装部署<a name="zh-cn_topic_0000001580166488"></a>

## 安装说明<a name="zh-cn_topic_0000001579847264"></a>

- Rec SDK TensorFlow支持物理机部署开发环境或通过容器部署开发环境，用户可根据实际业务情况选择其中一种进行部署。
- 建议通过普通用户进行安装、运行。

    >[!NOTE] 说明
    >如果需要查看Rec SDK TensorFlow的历史安装记录，请参见[查看Rec SDK TensorFlow安装与卸载记录](common_operations.md#zh-cn_topic_0000001683896117)。

## 安装依赖<a name="zh-cn_topic_0000001580007100"></a>

安装Rec SDK TensorFlow软件包前需准备以下环境依赖及操作，请参见[表1](#table18461141184116)准备安装环境。

**表 1** Rec SDK TensorFlow环境依赖
<a id="table18461141184116"></a>

|依赖名称/操作|推荐版本|获取方式|
|--|--|--|
|CANN软件包和TensorFlow适配昇腾插件|CANN 9.0.0|请参考[《CANN快速安装》](https://www.hiascend.com/cann/download)安装昇腾CANN软件包（包含Toolkit和ops包），并配置环境变量。<br>TensorFlow适配昇腾插件，单击[获取链接](https://gitee.com/ascend/tensorflow/releases/tag/tfa_v0.0.44_8.3.RC1)。npu_bridge-1.15.0\*适配TensorFlow 1.15.0的版本。|
|昇腾硬件产品驱动和固件|Ascend HDK 26.0.RC1及补丁版本|单击[获取链接](https://www.hiascend.com/developer/download/commercial/result?module=cann)，在左侧配套资源的“编辑资源选择”中进行配置，筛选配套的软件包，确认版本信息后获取所需软件包。安装驱动与固件请参见相关硬件产品配套的[《驱动和固件安装升级指南》](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743)。|
|配置Device网卡|-|请参考《Ascend Training Solution 组网指南》的参数面网络配置示例配置训练节点章节，通过HCCN_Tool配置NPU网口的Device IP。|
|TensorFlow|TensorFlow 1.15.0|请从[TensorFlow](https://github.com/tensorflow/tensorflow)仓库获取源码。Arm环境下TensorFlow官方未提供对应的whl包，如需在Arm环境下使用，可以从[链接](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/python/index.html)获取Arm的TensorFlow whl包。<br>[!NOTE] 说明<br>若whl包下载受阻，可复制其链接并在新标签页中打开，即可顺利完成下载。|
|Python 3.7.5|Python 3.7.5|请从[Python官网](https://www.python.org/)获取依赖软件包。|

>[!NOTE] 须知
>对于用户集成的开源和第三方软件，漏洞和问题请自行跟踪社区并及时进行修复；可以但不限于通过[CVE（通用漏洞字典）官网](https://www.cve.org/)确认对应开源软件版本的已知漏洞，并通过版本升级、使用patch补丁包更新等方式修复。

## 获取Rec SDK TensorFlow软件包<a name="zh-cn_topic_0000001630127085"></a>

可参见本章节获取Rec SDK TensorFlow的软件，并进行软件数字签名验证。

如果用户需要了解Rec SDK TensorFlow的源码，则可在[源码地址](https://gitcode.com/Ascend/RecSDK/tree/develop/training/tf_rec_v2)获取组件源码（注意当前暂未发布正式版本，仅供参考）；也支持用户自行编译源码，具体操作可参考如下：

### 源码编译安装

源码编译前，请参考[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Ubuntu)安装CANN开发套件软件包；参考[TF Adapter安装指南](https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/900beta1/migration/tfmigr1/tfmigr1_000009.html)安装TensorFlow适配昇腾的框架插件包。

编译环境依赖：

- Python3.7.5
- GCC 11.2.0
- CMake 3.22.6

开源依赖：

- [pybind11](https://github.com/pybind/pybind11/archive/refs/tags/v2.10.3.zip)
- [securec](https://github.com/huaweicloud/huaweicloud-sdk-c-obs/archive/refs/tags/v3.23.9.zip)
- [openmpi 4.1.5](https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz)：请参考软件文档在编译环境完成安装
- tensorflow 1.15：根据实际需求选择对应版本

将pybind11（>= 2.10.3）与securec的压缩包（huaweicloud-sdk-c-obs-3.23.9.zip）置于RecSDK代码同级的opensource目录下。

**编译方法**

进入Rec SDK代码目录：

- setup.py：执行脚本 `python3.7 setup.py bdist_wheel` 构建tf1版本whl包。构建成功后，whl包在`build`子目录下。

## 使用物理机部署开发环境<a name="zh-cn_topic_0000001630046437"></a>

>[!NOTE] 须知
>
>- 当前支持在Ubuntu  20.04、CentOS  7系统中进行物理机开发环境部署。
>- 用户请勿修改编译目录下除run.sh文件外的其他文件代码。

1. 参考《CANN 软件安装指南》安装CANN软件包和TensorFlow适配昇腾插件。
2. 配置环境变量。

    CANN软件提供进程级环境变量设置脚本，供用户在进程中引用，以自动完成环境变量设置。用户进程结束后自动失效。

    可在程序启动的Shell脚本中使用如下命令设置CANN的相关环境变量，也可通过命令行执行如下命令（以root用户默认安装路径“/usr/local/Ascend”为例）：

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

3. 通过如下命令安装软件包。

    ```bash
    # 安装软件包
    pip install tf_rec_v2-{version}-{arch}.tar.gz
    ```

    （其中 `{version}` 代表版本号，`{arch}` 代表操作系统架构，请根据实际安装包替换）

    软件包默认安装在Python的“site-packages”路径，若通过“--target”参数指定目录，在安装完成后需要将Rec SDK TensorFlow路径加入“PYTHONPATH”环境变量。

    ```bash
    export PYTHONPATH={rec_install_path}:{rec_install_path}/tf_rec_v2:$PYTHONPATH
    ```

4. 安装依赖，若未构建镜像，直接在物理机上进行开发，则须安装以下Python依赖。

    ```bash
    pip3.7 install numpy decorator sympy cffi pyyaml pathlib2 grpcio grpcio-tools protobuf==3.20.3 scipy requests mpi4py easydict scikit-learn attrs toml
    ```

    horovod依赖安装前需配置“HOROVOD\_WITH\_MPI”、“HOROVOD\_WITH\_TENSORFLOW”，依赖安装命令参考如下。

    ```bash
    HOROVOD_WITH_MPI=1 HOROVOD_WITH_TENSORFLOW=1 pip3.7 install horovod --no-cache-dir
    ```

5. 如需使用Hadoop分布式文件系统，请参考[Hadoop官方文档](https://hadoop.apache.org/docs/r1.0.4/cn/quickstart.html)进行环境部署和集群搭建。推荐使用Hadoop-2.7.5版本。

## 部署容器内的开发环境<a name="zh-cn_topic_0000002175060298"></a>

### 使用容器部署开发环境<a name="zh-cn_topic_0000001579847292"></a>

>[!NOTE] 须知
>
>- 如需使用CentOS系统进行配置（包括宿主机及容器），libstdc++版本需要高于libstdc++.so.6.0.24。
>- 出于安全保护，用户仅能使用非root用户启动容器进行使用。

基于容器部署Rec SDK TensorFlow开发环境，可参考如[图1](#fig2687191413442)步骤完成配置。

**图 1**  配置容器内的开发环境及训练镜像构建<a id="fig2687191413442"></a>
![](../../figures/tf_rec_v1/配置容器内的开发环境及训练镜像构建.png "配置容器内的开发环境及训练镜像构建")

**关键步骤说明<a name="section15488921175211"></a>**

1. 宿主机环境准备。

    请参见[安装依赖](#zh-cn_topic_0000001580007100)完成宿主机环境的部署。

2. 获取训练镜像，启动容器。可参考[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub)完成基础镜像的制作，以及Rec SDK TensorFlow的安装。
3. 如需在容器中使用动态扩容功能，请参见[（可选）片上内存侧动态扩容算子包安装](common_operations.md#zh-cn_topic_0000001630046409)，编译安装动态扩容算子包。
4. 如需使用Hadoop分布式文件系统，请参考[Hadoop官方文档](https://hadoop.apache.org/docs/r1.0.4/cn/quickstart.html)进行环境部署和集群搭建。推荐使用Hadoop-2.7.5版本。

    >[!NOTE] 说明
    >根据Hadoop官方文档部署环境之后，环境中/usr/local/hadoop-2.7.5/sbin文件属主为20415（非root用户），该属主有重命名、创建新文件来替换root用户的PATH环境变量中的可执行文件的权限，存在越权风险。

### 制作Rec SDK TensorFlow训练镜像<a name="zh-cn_topic_0000001787827420"></a>

本章节旨在指导用户根据已有基础镜像制作Rec SDK TensorFlow的训练镜像。

**前提条件<a name="section9622175114313"></a>**

1. 已经参考[安装依赖](#zh-cn_topic_0000001580007100)，在物理机上安装对应CANN版本的驱动和固件。
2. 物理机上已经安装Docker，并且Docker网络可用。
3. 准备基础镜像。可以从[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub)获取基础镜像，或者使用用户已有的基础镜像。
    - （推荐）从昇腾镜像仓库获取Rec SDK TensorFlow训练镜像。昇腾镜像仓库上的Rec SDK TensorFlow训练镜像中已经安装gcc、cmake等基础依赖，无需再次安装；只需更新其中的CANN和Rec SDK软件包即可使用。
    - 准备一个镜像作为基础镜像，建议以Ubuntu 20.04镜像为基础。

4. 执行如下命令，将基础镜像加载到docker中。

    ```bash
    docker load --input xxx.tar
    ```

5. 创建一个制作镜像使用的文件夹（以build\_images为例）。
    1. 仅将制作镜像过程中要使用到的文件放至该文件夹中，如对应架构的Ascend-cann-toolkit\_\*.run、tf-plugin、Rec SDK软件包。
    2. 若需安装tf-plugin软件包，还需将/usr/local/Ascend/driver/version.info和/etc/ascend\_install.info两个文件拷贝到build\_images目录下。

        （请勿在build\_images目录下放入无关文件，制作镜像时会将该目录下文件拷贝到镜像内。）

6. 制作镜像过程中需使用docker指令及从物理机拷贝文件，请确保用户有执行指令和访问文件权限。

**使用Rec SDK TensorFlow基础镜像制作训练镜像<a name="section1373503012391"></a>**

1. 参考[获取Rec SDK TensorFlow软件包](#zh-cn_topic_0000001630127085)，获取Rec SDK软件包，以及配套的CANN软件包和TensorFlow适配昇腾插件。
2. 在build\_images目录下创建Dockerfile配置文件（以Dockerfile名称为例），使用vi Dockerfile命令编辑文件，插入如下内容。

    ```bash
    # 请根据实际情况修改基础镜像名称及镜像tag
    FROM rec_sdk-tf1:7.3.0

    # CANN相关参数
    ARG TOOLKIT_PKG=Ascend-cann-toolkit*.run
    ARG KERNEL_PKG=Ascend-cann-*-ops*.run
    ARG TF1_PLUGIN=npu_bridge-1.15.0-*.whl
    # Rec SDK TensorFlow包
    ARG REC_SDK_PKG=tf_rec_v2*.tar.gz

    # 设置安装路径环境变量
    ARG ASCEND_BASE=/usr/local/Ascend

    # 删除旧的CANN
    RUN rm -rf $ASCEND_BASE/ascend-toolkit
    RUN rm -rf $ASCEND_BASE/cann*

    # 请根据实际情况选择安装需要拷贝和安装的依赖，若无需执行可将指令行删除
    WORKDIR /tmp
    COPY $TOOLKIT_PKG .
    COPY $KERNEL_PKG .
    COPY $TF1_PLUGIN .
    COPY $REC_SDK_PKG .
    COPY version.info .
    COPY ascend_install.info .

    # 安装ascend-toolkit和ops算子包
    RUN umask 0027 && \
        mkdir -p $ASCEND_BASE/driver && \
        /usr/bin/cp -f version.info $ASCEND_BASE/driver/ && \
        /usr/bin/cp -f ascend_install.info /etc/ && \
        chmod +x $TOOLKIT_PKG && \
        echo Y | bash $TOOLKIT_PKG --quiet --install --install-path=$ASCEND_BASE && \
        chmod +x $KERNEL_PKG && \
        echo Y | bash $KERNEL_PKG --quiet --install && \
        source $ASCEND_BASE/cann/set_env.sh && \
        rm -rf /root/.cache/pip && \
        rm -f $TOOLKIT_PKG && \
        rm -f $KERNEL_PKG && \
        rm -rf $ASCEND_BASE/driver && \
        rm -rf /etc/ascend_install.info

    # 安装Rec SDK TensorFlow
    RUN pip3.7 install $TF1_PLUGIN --force-reinstall && \
        pip3.7 install $REC_SDK_PKG --force-reinstall && \
        rm -rf /root/.cache/pip && \
        rm -f $REC_SDK_PKG
    ```

3. 进入build\_images路径，执行如下指令构建Rec SDK TensorFlow镜像。

    ```bash
    docker build -t {镜像名称}:{镜像tag} -f Dockerfile .
    ```

**使用Ubuntu 20.04或用户镜像制作训练镜像<a name="section104919392501"></a>**

1. 确认镜像中是否已经安装以下依赖，将未安装的依赖软件包下载到build\_images目录下。

    |依赖名称| 下载链接                                                                                         |
    |--|----------------------------------------------------------------------------------------------|
    |gcc-11.2.0| [链接](https://mirrors.ustc.edu.cn/gnu/gcc/gcc-11.2.0/gcc-11.2.0.tar.gz)                         |
    |cmake-3.22.6| [链接](https://cmake.org/files/v3.22/cmake-3.22.6.tar.gz)                                      |
    |ucx| [链接](https://github.com/openucx/ucx/archive/master.zip)                                      |
    |openmpi-4.1.5| [链接](https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz)               |
    |python-3.7.5| [链接](https://repo.huaweicloud.com/python/3.7.5/Python-3.7.5.tar.xz)                          |
    |hdf5-1.10.5| [链接](https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz) |
    |CANN软件包、TensorFlow适配昇腾插件以及Rec SDK软件包| 参见[安装依赖](#zh-cn_topic_0000001580007100)                                                                                   |
    |TensorFlow（1.15.0）| [链接](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/python/index.html) |

2. 请参见[OVERVIEW](../../../../docker/OVERVIEW.md)制作镜像。

## 配置环境变量<a name="zh-cn_topic_0000001580326424"></a>

Rec SDK TensorFlow环境变量的说明如[表1](#table126401659163820)所示。在需要使用C/C++编译时，需要设置编译环境变量，如C++语言编写的算子编译等，具体请参见[表2](#table20242918114315)。

**表 1**  环境变量
<a id="table126401659163820"></a>

| 环境变量名                  | 含义                          | 可选/必选 | 说明                                                                                                                                                                                                                                                                                                                                                                                                                           |
|------------------------|-----------------------------|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| RANK_TABLE_FILE        | 用于配置昇腾芯片的通信集合文件。            | 可选    | 集合通信文件路径，默认为""。当使用rank table方案时为必选。                                                                                                                                                                                                                                                                                                                                                                                          |
| HCCL_TOPO_FILE_PATH    | 用于适配昇腾芯片的通信拓扑文件。            | 可选    | 通信拓扑文件路径，默认为""。当使用rank table方案时为必选。                                                                                                                                                                                                                                                                                                                                                                                          |
| ASCEND_VISIBLE_DEVICES | 昇腾处理器可见的设备，来指定程序只使用其中的部分设备。 | 必选    | 使用ASCEND_VISIBLE_DEVICES环境变量指定训练中的NPU设备（用户可执行ls /dev/ \| grep davinci\*命令查询宿主机的NPU设备），使用设备序号指定设备，支持单个和范围指定且支持混用。例如：<li>ASCEND_VISIBLE_DEVICES=0表示将0号设备（/dev/davinci0）挂载入容器中。</li><li>ASCEND_VISIBLE_DEVICES=1,3表示将1、3号设备挂载入容器中。</li><li>ASCEND_VISIBLE_DEVICES=0-2表示将0号至2号设备（包含0号和2号）挂载入容器中，效果同ASCEND_VISIBLE_DEVICES=0,1,2。</li><li>ASCEND_VISIBLE_DEVICES=0-2,4表示将0号至2号以及4号设备挂载入容器，效果同ASCEND_VISIBLE_DEVICES=0,1,2,4。</li> |

>[!NOTE] 说明
>Rec SDK TensorFlow在OpenMPI启动的分布式训练和推理场景中需要依赖OMPI_COMM_WORLD_SIZE、OMPI_COMM_WORLD_LOCAL_SIZE和OMPI_COMM_WORLD_RANK环境变量。这些环境变量由OpenMPI启动器自动注入，用户无需手动注入。

**表 2**  C++编译环境变量
<a name="table20242918114315"></a>

|环境变量名|含义|可选/必选|说明|
|--|--|--|--|
|CC|C语言编译器|必选|设置为gcc|
|CXX|C++语言编译器|必选|设置为g++|

## 升级<a name="ZH-CN_TOPIC_0000001723715541"></a>

用户如需将当前版本的Rec SDK TensorFlow升级至最新版本，可将最新的Rec SDK软件包上传至安装环境后，在软件包所在目录下使用命令进行版本升级，具体命令参见如下。

- 升级Rec SDK TensorFlow新版本时，需要先手动卸载旧版本。

    ```bash
    pip3 uninstall mxrec -y
    ```

- 使用<b>--upgrade</b>命令升级Rec SDK TensorFlow。

    ```bash
    pip3 install --upgrade mxrec_for_lingqu-{version}-py3-none-{arch}.whl
    ```

其中，{version}为版本号，{arch}为操作系统架构。

## 卸载<a name="ZH-CN_TOPIC_0000001629887081"></a>

用户如需移除Rec SDK TensorFlow软件包部署，可参考以下命令进行卸载。

```bash
pip3 uninstall mxrec -y
```
