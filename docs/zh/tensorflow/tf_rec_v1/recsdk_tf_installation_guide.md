# 安装部署<a name="ZH-CN_TOPIC_0000001580166488"></a>

## 安装说明<a name="ZH-CN_TOPIC_0000001579847264"></a>

- Rec SDK TensorFlow支持物理机部署开发环境或通过容器部署开发环境，用户可根据实际业务情况选择其中一种进行部署。
- 建议通过普通用户进行安装、运行。

    >[!NOTE]
    >如果需要查看Rec SDK TensorFlow的历史安装记录，请参见[查看Rec SDK TensorFlow安装与卸载记录](common_operations.md#查看rec-sdk-tensorflow安装与卸载记录)。

## 安装依赖<a name="ZH-CN_TOPIC_0000001580007100"></a>

安装Rec SDK TensorFlow软件包前需准备以下环境依赖及操作，请参见[表1](#table18461141184116)准备安装环境。

**表 1** Rec SDK TensorFlow环境依赖
<a id="table18461141184116"></a>

|依赖名称/操作|推荐版本|获取方式|
|--|--|--|
|CANN软件包和TensorFlow适配昇腾插件|CANN 9.0.0|<li>请参考[《CANN快速安装》](https://www.hiascend.com/cann/download)安装昇腾CANN软件包（包含Toolkit和ops包），并配置环境变量。</li><li>TensorFlow适配昇腾插件单击[获取链接](https://gitee.com/ascend/tensorflow/releases/tag/tfa_v0.0.44_8.3.RC1)。npu_device-2.6.5\*适配TensorFlow 2.6.5的版本；npu_bridge-1.15.0\*适配TensorFlow 1.15.0的版本。</li>|
|昇腾硬件产品驱动和固件|Ascend HDK 26.0.RC1及补丁版本|单击[获取链接](https://www.hiascend.com/developer/download/commercial/result?module=cann)，在左侧配套资源的“编辑资源选择”中进行配置，筛选配套的软件包，确认版本信息后获取所需软件包。安装驱动与固件请参见相关硬件产品配套的[《驱动和固件安装升级指南》](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743)。|
|Ascend Docker Runtime|MindCluster 7.3.0|请参见《MindCluster 集群调度用户指南》的“安装 > 安装部署”章节进行安装。|
|TensorFlow|TensorFlow 1.15.0和TensorFlow 2.6.5|请从[TensorFlow](https://github.com/tensorflow/tensorflow)仓库获取源码。Arm环境下TensorFlow官方未提供对应的whl包，如需在Arm环境下使用，可以从[链接](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/python/index.html)获取Arm的TensorFlow whl包。<br>若whl包下载受阻，可复制其链接并在新标签页中打开，即可顺利完成下载。|
|Python 3.7.5|Python 3.7.5|请从[Python官网](https://www.python.org/)获取依赖软件包。|

>[!NOTE]
>对于用户集成的开源和第三方软件，漏洞和问题请自行跟踪社区并及时进行修复；可以但不限于通过[CVE（通用漏洞字典）官网](https://www.cve.org/)确认对应开源软件版本的已知漏洞，并通过版本升级、使用patch补丁包更新等方式修复。

## 获取Rec SDK TensorFlow软件包<a name="ZH-CN_TOPIC_0000001630127085"></a>

可参见本章节获取Rec SDK TensorFlow的软件，并进行软件数字签名验证。

如果用户需要了解Rec SDK TensorFlow的源码，则可在[源码地址](https://gitcode.com/Ascend/RecSDK/tree/develop)获取组件源码；也支持用户自行编译源码，具体操作可参考如下：

### 源码编译安装

源码编译前，请参考[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Ubuntu)安装CANN开发套件软件包；参考[TF Adapter安装指南](https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/900beta1/migration/tfmigr1/tfmigr1_000009.html)安装TensorFlow适配昇腾的框架插件包。

1. 编译环境依赖：
   - Python3.7.5
   - GCC 11.2.0
   - CMake 3.22.6

2. 开源依赖：
   - [pybind11](https://github.com/pybind/pybind11/archive/refs/tags/v2.10.3.zip)
   - [securec](https://github.com/huaweicloud/huaweicloud-sdk-c-obs/archive/refs/tags/v3.23.9.zip)
   - [openmpi 4.1.5](https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz): 请参考软件文档在编译环境完成安装
   - tensorflow 1.15/2.6.5：根据实际需求选择对应版本

   将pybind11（>= 2.10.3）与securec的压缩包（huaweicloud-sdk-c-obs-3.23.9.zip）置于RecSDK代码同级的opensource目录下。

3. 编译方法：

   进入Rec SDK代码目录：
   - setup.py：此脚本供内部使用，用于同时构建tf1和tf2的Rec SDK包，用户通常只需要其中一个，所以建议使用下面两个脚本构建。
   - setup_tf1.py：执行脚本 `python3.7 setup_tf1.py bdist_wheel` 完成tf1版本whl包的构建，构建成功后，whl包在build/mindxsdk-mxrec/tf1_whl子目录下。
   - setup_tf2.py：执行脚本 `python3.7 setup_tf2.py bdist_wheel` 完成tf2版本whl包的构建，构建成功后，whl包在build/mindxsdk-mxrec/tf2_whl子目录下。

   如需使用动态扩容功能，进入“RecSDK/cust_op/ascendc_op/ai_core_op/cust_op_by_addr”目录中。执行命令 `bash run.sh` 编译并安装动态扩容算子包。

4. 测试用例

   **Python侧测试用例**

   运行Python测试用例所需依赖：

   - pytest 7.1.1
   - pytest-cov 4.1.0
   - pytest-html

   如需运行python测试用例，完成上述依赖项的安装，并验证tf1环境可正常进行源码编译。然后进入RecSDK/training/tf_rec_v1/python/tests目录，参考以下命令执行python侧测试用例：

   ```shell
   bash run_python_dt.sh
   ```

   **C++侧测试用例**

   运行C++侧测试用例所需依赖：

   - [googletest 1.8.1](https://github.com/google/googletest/archive/refs/tags/release-1.8.1.zip)
   - [emock 0.9.0](https://github.com/ez8-co/emock/archive/refs/tags/v0.9.0.zip)
   - [pybind11 v2.10.3](https://github.com/pybind/pybind11/archive/refs/tags/v2.10.3.zip)
   - [securec](https://github.com/huaweicloud/huaweicloud-sdk-c-obs/archive/refs/tags/v3.23.9.zip)

   将googletest（googletest-release-1.8.1.zip）、emock（emock-0.9.0.zip）、pybind11（>= 2.10.3）与securec的压缩包（huaweicloud-sdk-c-obs-3.23.9.zip）置于RecSDK代码同级的opensource目录下。

   如需运行C++测试用例，完成上述依赖项的安装。然后进入RecSDK/training/tf_rec_v1/src目录，参考以下命令执行C++测试用例：

   tf1环境下使用如下命令：

   ```shell
   bash test_ut.sh tf1
   ```

   tf2环境下使用如下命令：

   ```shell
   bash test_ut.sh tf2
   ```

**下载软件包<a name="section1852417242717"></a>**

请参考本章获取所需软件包和对应的数字签名文件，下载本软件即表示您同意[华为企业业务最终用户许可协议（EULA）](https://e.huawei.com/cn/about/eula)的条款和条件。

|组件名称|软件包|获取链接|
|--|--|--|
|Rec SDK|tf\_rec\_v1-\*.tar.gz|[获取链接](https://www.hiascend.com/zh/developer/download/community/result?module=sdk+cann)。|

**软件数字签名验证<a name="section10830205518487"></a>**

为了防止软件包在传递过程中或存储期间被恶意篡改，下载软件包时请下载对应的数字签名文件用于完整性验证。

在软件包下载之后，请参考《OpenPGP签名验证指南》，对下载的软件包进行PGP数字签名校验。如果校验失败，请勿使用该软件包并联系华为技术支持工程师解决。

使用软件包安装/升级前，也需要按照上述过程，验证软件包的数字签名，确保软件包未被篡改。

运营商客户请访问：[https://support.huawei.com/carrier/digitalSignatureAction](https://support.huawei.com/carrier/digitalSignatureAction)

企业客户请访问：[https://support.huawei.com/enterprise/zh/tool/software-digital-signature-openpgp-validation-tool-TL1000000054](https://support.huawei.com/enterprise/zh/tool/software-digital-signature-openpgp-validation-tool-TL1000000054)

## 部署容器内的开发环境<a name="ZH-CN_TOPIC_0000002175060298"></a>

### 使用容器部署开发环境<a name="ZH-CN_TOPIC_0000001579847292"></a>

>[!NOTE]
>
>- 如需使用CentOS系统进行配置（包括宿主机及容器），libstdc++版本需要高于libstdc++.so.6.0.24。
>- 出于安全保护，用户仅能使用非root用户启动容器进行使用。

基于容器部署Rec SDK TensorFlow开发环境，可参考如[图1](#fig2687191413442)步骤完成配置。

**图 1**  配置容器内的开发环境及训练镜像构建<a id="fig2687191413442"></a>
![](../../figures/tf_rec_v1/配置容器内的开发环境及训练镜像构建.png "配置容器内的开发环境及训练镜像构建")

**关键步骤说明<a name="section15488921175211"></a>**

1. 宿主机环境准备。

    请参见[安装依赖](#安装依赖)完成宿主机环境的部署。

2. 获取训练镜像，启动容器。可参考[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub)完成基础镜像的制作，以及Rec SDK TensorFlow的安装。
3. 如需在容器中使用动态扩容功能，请参见[（可选）片上内存侧动态扩容算子包安装](common_operations.md#可选片上内存侧动态扩容算子包安装)，编译安装动态扩容算子包。
4. 如需使用Hadoop分布式文件系统，请参考[Hadoop官方文档](https://hadoop.apache.org/docs/r1.0.4/cn/quickstart.html)进行环境部署和集群搭建。推荐使用Hadoop-2.7.5版本。

    >[!NOTE]
    >根据Hadoop官方文档部署环境之后，环境中/usr/local/hadoop-2.7.5/sbin文件属主为20415（非root用户），该属主有重命名、创建新文件来替换root用户的PATH环境变量中的可执行文件的权限，存在越权风险。

### 制作Rec SDK TensorFlow训练镜像<a name="ZH-CN_TOPIC_0000001787827420"></a>

请参考 [docker 目录](https://gitcode.com/Ascend/RecSDK/tree/develop/docker) 下的 Dockerfile 制作训练镜像，包含以下文件：

- `Dockerfile.26.1.0-ubuntu20.04-tf-py3.7`：基于 Ubuntu 20.04 的 TensorFlow 训练镜像
- `Dockerfile.26.1.0-openEuler22.03-tf-py3.7`：基于 openEuler 22.03 的 TensorFlow 训练镜像

构建时可通过 `--build-arg CORE_TYPE=a2|a3|a5` 指定芯片架构。

## 配置环境变量<a name="ZH-CN_TOPIC_0000001580326424"></a>

Rec SDK TensorFlow环境变量的说明如[表1](#table126401659163820)所示。在需要使用C/C++编译时，需要设置编译环境变量，如C++语言编写的算子编译等，具体请参见[表2](#table20242918114315)。

**表 1**  环境变量
<a id="table126401659163820"></a>

|环境变量名|含义|可选/必选|说明|
|--|--|--|--|
|MXREC_LOG_LEVEL|框架日志等级。|可选|取值范围：INFO、DEBUG或者ERROR，默认值为INFO。|
|TF_DEVICE|是否进行合表判断。|可选|取值范围：NPU、GPU、CPU或者NONE，默认值为NONE。<li>当取值为GPU、CPU、NONE时，不进行合表判断。</li><li>当取值为NPU时，进行合表判断。</li>|
|AclTimeout|Acl超时时间。|可选|取值范围：[-1,int32的最大值2147483647]，默认值为-1。|
|HD_CHANNEL_SIZE|CPU处理的数据通道深度。|可选|取值范围：[2,8192]，默认值为40。|
|KEY_PROCESS_THREAD_NUM|KEY_PROCESS线程数量。|可选|取值范围：[1,10]，默认值为6。|
|MAX_UNIQUE_THREAD_NUM|最大UNIQUE线程数。|可选|取值范围：[1,8]，默认值为8。|
|FAST_UNIQUE|是否自实现的优化去重编码算法。|可选|取值范围：0或者1，默认值为0。取值范围以外的值，将产生不可预期的行为。<li>0：表示不生效。</li><li>1：表示生效。</li>|
|HOT_EMB_UPDATE_STEP|Hot Embedding更新步数。|可选|取值范围：[1,1000]，默认值为1000。|
|GLOG_stderrthreshold|glog日志等级。|可选|取值范围：[-2,2]，默认值为0。<li>-2：表示TRACE。</li><li>-1：表示DEBUG。</li><li>0：表示INFO。</li><li>1：表示WARNING。</li><li>2：表示ERROR。</li>|
|USE_COMBINE_FAAE|控制是否合表统计次数。|可选|取值范围：0或者1，默认值为0。取值范围以外的值，将产生不可预期的行为。<li>如果USE_COMBINE_FAAE=0，表示分表统计，每张表key的count记录是独立的。</li><li>如果USE_COMBINE_FAAE=1，表示合表统计，多张表维护一个count记录。</li>|
|CM_CHIEF_IP|主节点IP。|可选|当使用去rank table方案时为必选。|
|CM_CHIEF_PORT|主节点侦听端口，比如60000。|可选|当使用去rank table方案时为必选。<br><li>可使用如下命令指定一组本地保留端口，这些端口将被系统保留，不会被其他应用程序使用：<br>```sysctl -w net.ipv4.ip_local_reserved_ports=60000-60015```<br>然后将CM_CHIEF_PORT设置为上述命令指定范围的端口。</li><li>检查端口是否被占用：<br>``` netstat -anp \| grep 端口号 ``` <br>如果端口号被占用，会显示出占用该端口的进程ID和进程名称。</li>|
|CM_CHIEF_DEVICE|主节点Device ID。|可选|指定Master节点中统计Server端集群信息的Device逻辑ID。<br>取值范围：[0,环境可见Device数量<b>-1</b>]。当使用去rank table方案时为必选。|
|CM_WORKER_IP|当前节点IP。|可选|当使用去rank table方案时为必选。|
|CM_WORKER_SIZE|参与集群训练的device数量。|可选|取值范围：[0,512]。当使用去rank table方案时为必选。|
|RANK_TABLE_FILE|用于配置昇腾芯片的通信集合文件。|可选|集合通信文件路径，默认为""。当使用rank table方案时为必选。|
|ASCEND_VISIBLE_DEVICES|昇腾处理器可见的设备，来指定程序只使用其中的部分设备。|必选|使用ASCEND_VISIBLE_DEVICES环境变量指定训练中的NPU设备（用户可执行ls /dev/ \| grep davinci\*命令查询宿主机的NPU设备），使用设备序号指定设备，支持单个和范围指定且支持混用。例如：<li>ASCEND_VISIBLE_DEVICES=0表示将0号设备（/dev/davinci0）挂载入容器中。</li><li>ASCEND_VISIBLE_DEVICES=1,3表示将1、3号设备挂载入容器中。</li><li>ASCEND_VISIBLE_DEVICES=0-2表示将0号至2号设备（包含0号和2号）挂载入容器中，效果同ASCEND_VISIBLE_DEVICES=0,1,2。</li><li>ASCEND_VISIBLE_DEVICES=0-2,4表示将0号至2号以及4号设备挂载入容器，效果同ASCEND_VISIBLE_DEVICES=0,1,2,4。</li>|
|RECORD_KEY_COUNT|控制是否记录key及key出现的数量count的开关。|可选|取值范围：0或者1，默认值为0。取值范围以外的值，将产生不可预期的行为。<li>0：表示不开启记录key及count信息。</li><li>1：表示开启记录key及count信息。</li>|
|LCAL_COMM_ID|指定LCAL元信息交换主节点|可选|基于socket通信，格式为ip:port。不指定时，则默认通信主节点为当前任务最小rank id对应的进程，默认端口为10067。|
|LCCL_DETERMINISTIC|开启LCCL确定性计算|可选|默认值为0，表示关闭LCCL确定性计算。<br>需要确定性计算时，可配置值为1。GatherUss算子将确保计算有序。<br>取值范围以外的值，将产生不可预期的行为。|
|USE_SHM_SWAP|PCIE through性能提升|可选|取值范围：0或者1，默认值为0。取值范围以外的值，将产生不可预期的行为。<li>0：表示关闭该特性。</li><li>1：表示开启该特性。</li>|
|HUGE_TLB_ENABLE|大页内存|可选|取值范围：0或者1，默认值为0。取值范围以外的值，将产生不可预期的行为。<li>0：表示关闭该特性。</li><li>1：表示开启该特性。</li>|
|SSD_SAVE_COMPACT_LEVEL|SSD保存时的压缩等级|可选|取值范围：[0,2]，默认值为2。<li>0：表示不压缩。</li><li>1：表示仅压缩超阈值文件。</li><li>2：表示压缩所有文件。</li>|

>[!NOTE]
>Rec SDK TensorFlow在OpenMPI启动的分布式训练和推理场景中需要依赖OMPI_COMM_WORLD_SIZE、OMPI_COMM_WORLD_LOCAL_SIZE、和OMPI_COMM_WORLD_RANK环境变量。这些环境变量由OpenMPI启动器自动注入，用户无需手动注入。

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
    pip3 uninstall mx_rec -y
    ```

- 使用<b>--upgrade</b>命令升级Rec SDK TensorFlow。

    ```bash
    pip3 install --upgrade mx_rec-{version}-py3-none-{arch}.whl
    ```

\{version\}为版本号，\{arch\}为操作系统架构。

## 卸载<a name="ZH-CN_TOPIC_0000001629887081"></a>

用户如需移除Rec SDK TensorFlow软件包部署，可参考以下命令进行卸载。

```bash
pip3 uninstall mx_rec -y
```

如需卸载动态扩容算子，请参见[（可选）片上内存侧动态扩容算子包安装](common_operations.md#可选片上内存侧动态扩容算子包安装)中的卸载说明。
