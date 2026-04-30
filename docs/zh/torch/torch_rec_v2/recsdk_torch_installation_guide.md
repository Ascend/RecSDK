# 安装部署<a name="ZH-CN_TOPIC_0000002336148865"></a>

## 安装说明<a name="ZH-CN_TOPIC_0000002302229632"></a>

推荐Rec SDK Torch基于容器部署开发环境。如果需要查看Rec SDK Torch的历史安装记录，请参见[查看Rec SDK Torch安装与卸载记录](../../torch/torch_rec_v2/common_operations.md)。

**注意事项<a name="section1297475493911"></a>**

如需安装Rec SDK Torch软件包以外的第三方软件，请注意及时升级最新版本，关注并修补存在的漏洞。

## 安装依赖<a name="ZH-CN_TOPIC_0000002302389236"></a>

安装Rec SDK Torch软件包前需准备以下环境依赖及操作，请参见[表1](#table18461141184116)准备安装环境。

**表 1** Rec SDK Torch环境依赖
<a id="table18461141184116"></a>

|依赖名称/操作|推荐版本|获取方式|
|--|--|--|
|昇腾硬件产品驱动和固件|Ascend HDK 25.5.0|单击[获取链接](https://www.hiascend.com/developer/download/commercial/result?module=cann)，在左侧配套资源的“编辑资源选择”中进行配置，筛选配套的软件包，确认版本信息后获取所需软件包。<br>安装驱动与固件请参见相关硬件产品配套的《[驱动和固件安装升级指南](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743)》。|
|Ascend Docker Runtime|MindCluster 7.3.0|请参见《[MindCluster 集群调度用户指南](https://www.hiascend.com/document/detail/zh/mindcluster/730/clustersched/dlug/dlug_installation_017.html)》的“安装 > 安装部署”章节进行安装。|
|配置Device网卡|-|请参考《[Ascend Training Solution 23.0.0 组网指南](https://support.huawei.com/enterprise/zh/doc/EDOC1100349028/d9914967)》的参数面网络配置示例配置示例配置训练节点章节，通过HCCN_Tool配置NPU网口的Device IP。|
|CANN软件包|CANN 9.0.0|单击[获取链接](https://www.hiascend.com/developer/download/commercial/result?module=cann)，在左侧配套资源的“编辑资源选择”中进行配置，筛选配套的软件包，确认版本信息后获取所需软件包。<br>根据设备架构获取Ascend-cann-toolkit_<i>{version}</i>_linux-<i>{arch}</i>.run和Ascend-cann-<i>{chip_type}</i>-ops-<i>{version}</i>_linux-<i>{arch}</i>.run，并参考《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/softwareinst/instg/instg_0008.html?Mode=PmIns&OS=Debian&InstallType=local)》的“安装CANN”章节在容器内进行安装。|
|PyTorch昇腾适配插件|2.7.1|单击[链接](https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/Daily/v2.7.1/20260327.4/pytorch_v2.7.1_py311.tar.gz)，根据设备架构获取torch_npu-2.7.1*-cp311-*.whl软件包并在容器内安装。|

> [!NOTE]须知 
>对于用户集成的开源和第三方软件，漏洞和问题请自行跟踪社区并及时进行修复；可以并且不限于通过[CVE（通用漏洞字典）官网](https://www.cve.org/)确认对应开源软件版本的已知漏洞，并通过版本升级、使用patch补丁包更新等方式修复。

## 获取Rec SDK Torch软件包<a name="ZH-CN_TOPIC_0000002336148981"></a>

### 源码编译安装

源码编译前，请参考[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Ubuntu)安装CANN开发套件软件包；参考[Ascend Extension for PyTorch安装指南](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)安装PyTorch适配昇腾的框架插件包。 

需要编译的源码包：

| 名称                                       | 说明              |
|------------------------------------------|-----------------|
| Ascend-mindxsdk-torchrec-*-npu-*.tar.gz  | torchrec昇腾适配包   |
| Ascend-mindxsdk-dynamic-emb-*.tar.gz     | RecSDK-Torch软件包 |
| Ascend-recsdk-npu-ops-\*-linux-\*.tar.gz | 算子包             |
| libfbgemm_npu_api.so                     | 算子适配层           |

1. 编译环境
 
   容器环境编译，参考[README](../build_torch_rec_v2_images/README.md)。
 
2. 编译Ascend-mindxsdk-torchrec-\*-npu-\*.tar.gz
 
   参考[README](../../../../training/torch_rec_v2/torchrec_npu/README.md)。
 
   生成的tar包在 `RecSDK/training/torch_rec_v2/torchrec_npu/torchrec/` 路径下。
 
   **安装方法**
 
   ```bash
   tar zxvf Ascend-mindxsdk-torchrec-*-npu-*.tar.gz
   pip3 install torchrec-*+npu-py3-none-linux_*.whl
   ```
 
3. 编译Ascend-mindxsdk-dynamic-emb-*.tar.gz
 
   参考对应[README](../../../../training/torch_rec_v2/dynamic_emb/README.md)。
 
   生成的tar包在 `RecSDK/training/torch_rec_v2/output/` 路径下。
 
   **安装方法**
 
   ```bash
    tar -zxvf Ascend-mindxsdk-dynamic-emb-*.tar.gz
    pip3 install dynamic_emb-*.whl
   ```
 
4. 编译Ascend-recsdk-npu-ops-\*-linux-\*.tar.gz
 
   参考对应[README](../../../../cust_op/ascendc_op/build/README.md)。
 
5. 编译libfbgemm_npu_api.so
 
   ```bash
   cd RecSDK/cust_op/framework/torch_plugin/torch_library/common/
   bash build_ops.sh
   ```
 
   生成的so包在 `RecSDK/cust_op/framework/torch_plugin/torch_library/common/build` 路径下。
 
   **安装方法**
 
   编译完成后，会在common/build下生成so，并自动拷贝到python默认site-packages路径下。
 
   在 Python 代码中添加以下语句，即可自动从 Python 默认 site-packages 目录加载 libfbgemm_npu_api.so：

   ```python
   torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
   ```

**配套版本<a name="section146113514599"></a>**

当前动态稀疏表Rec SDK Torch支持的配套版本如下所示：

|PyTorch|PyTorch昇腾适配插件|Rec SDK Torch|
|--|--|--|
|2.7.1|2.7.1|25.09|

其他软件配套版本信息请参考[基础镜像构建](../build_torch_rec_v2_images/README.md)。

**下载软件包<a name="section1852417242717"></a>**

请参考本章获取所需软件包和对应的数字签名文件，下载本软件即表示您同意[华为企业业务最终用户许可协议（EULA）](https://e.huawei.com/cn/about/eula)的条款和条件。

|组件名称|软件包|获取链接|
|--|--|--|
|Rec SDK|推荐算法框架开发套件包|[获取链接](https://gitcode.com/Ascend/RecSDK/releases)|

>[!NOTE]说明 
>当前提供的Rec SDK推荐算法框架开发套件包基于Python 3.11版本编译，请在相同的Python版本环境下安装使用。若需要在其他Python版本环境下安装使用，请参见[README](https://gitcode.com/Ascend/RecSDK/blob/develop/training/torch_rec_v2/dynamic_emb/README.md)进行源码编译。

**软件数字签名验证<a name="section10830205518487"></a>**

为了防止软件包在传递过程中或存储期间被恶意篡改，在软件包下载之后请使用**sha256sum**命令校验Hash值。

**软件包内容<a name="section635921752112"></a>**

|组件名称|软件包|
|--|--|
|Ascend-mindxsdk-dynamic-emb-*.tar.gz|Rec SDK Torch功能包|

## 部署容器内的开发环境<a name="ZH-CN_TOPIC_0000002336148877"></a>

### 使用容器部署开发环境<a name="ZH-CN_TOPIC_0000002302229684"></a>

基于容器部署Rec SDK Torch开发环境，可参考如[图1](#fig1345216415476)完成配置。

**图 1**  配置容器内的开发环境及训练镜像构建<a id="fig1345216415476"></a>  
![](../../figures/torch_rec_v1/配置容器内的开发环境及训练镜像构建.png "配置容器内的开发环境及训练镜像构建")

**关键步骤说明<a name="section15488921175211"></a>**

1. 准备宿主机环境。请参见[安装依赖](#安装依赖)完成宿主机环境的部署。
2. 构建基础镜像，请参见[使用Debian 12制作训练镜像](../build_torch_rec_v2_images/README.md)。
3. 启动容器，请参见[启动容器](#section12808621121114)。
4. 安装Rec SDK Torch，请参见[安装Rec SDK Torch](#section182972951211)。

### 制作Rec SDK Torch训练镜像<a name="ZH-CN_TOPIC_0000002336268705"></a>

**使用Debian 12制作训练镜像<a id="section104919392501"></a>**

参考[基础镜像构建](../build_torch_rec_v2_images/README.md)里的Dockerfile和Readme制作镜像。

**启动容器<a id="section12808621121114"></a>**

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

>[!NOTE]说明 
>部分参数说明如下：
>
>- -m 300g表示设置容器内可以使用的内存大小上限为300G，可根据实际情况进行配置。
>- -e ASCEND\_VISIBLE\_DEVICES=0-7表示将服务器上编号为device0-device7的NPU设备挂载到容器内。可根据实际情况进行配置。

**安装Rec SDK Torch<a id="section182972951211"></a>**

1. 参考[获取Rec SDK Torch软件包](#获取rec-sdk-torch软件包)获取Rec SDK Torch软件包。
2. 将软件包拷贝到容器中。可通过以下方式：
    - 在启动容器时，指定一个宿主机目录挂载到容器内，并将下载的软件包放在其中，使容器可以访问到下载的软件包。

        宿主机目录挂载到容器的docker参数示例：

        ```bash
        -v /dir1:/dir1
        ```

    - 在宿主机上，使用**docker cp**指令将软件包拷贝到容器内。

        **docker cp**指令示例：

        ```bash
        docker cp host_file_path container_name:container_file_path
        ```

        其中，host\_file\_path为宿主机文件路径，container\_name为待拷入的docker容器名称，container\_file\_path为待拷入的docker容器内的文件路径。

3. 按照如下步骤进行编译和安装包。

    1. 安装TorchRec昇腾注册包

        TorchRec昇腾注册包是基于TorchRec源码做的NPU设备适配。可通过Rec SDK Torch提供的patch文件和TorchRec源码的固定分支编译出该注册包。

        具体的源码编译和安装可参考[README](https://gitcode.com/Ascend/RecSDK/blob/develop/training/torch_rec_v2/torchrec_npu/README.md)（该包的源码编译不区分PyTorch版本）。

    2. 安装Ascend-mindxsdk-dynamic-emb-\*-linux-\*.tar.gz

        ```bash
        # 安装Ascend-mindxsdk-dynamic-emb-*-linux-*.tar.gz
        tar zxvf Ascend-mindxsdk-dynamic-emb-*-linux-*.tar.gz
        pip3 install dynamic_emb-*-cp11-cp11-linux_*.whl
        ```

## 配置环境变量<a name="ZH-CN_TOPIC_0000002336268805"></a>

Rec SDK Torch环境变量的说明如[表1](#table126401659163820)所示。

**表 1**  环境变量
<a id="table126401659163820"></a>

|环境变量名|含义|可选/必选|说明|
|--|--|--|--|
|ASCEND_RT_VISIBLE_DEVICES|昇腾处理器可见的设备，来指定程序只使用其中的部分设备。|可选|使用ASCEND_RT_VISIBLE_DEVICES环境变量指定训练中的NPU设备（用户可执行ls /dev/ \| grep davinci*命令查询宿主机的NPU设备），使用设备序号指定设备，支持单个和范围指定且支持混用。例如：<ul> <li>ASCEND_RT_VISIBLE_DEVICES=0：表示将0号设备（/dev/davinci0）挂载入容器中。</li><li>ASCEND_RT_VISIBLE_DEVICES=1,3：表示将1、3号设备挂载入容器中。</li><li>ASCEND_RT_VISIBLE_DEVICES=0-2：表示将0号至2号设备（包含0号和2号）挂载入容器中，效果同ASCEND_RT_VISIBLE_DEVICES=0,1,2。</li><li>ASCEND_RT_VISIBLE_DEVICES=0-2,4：表示将0号至2号以及4号设备挂载入容器，效果同ASCEND_RT_VISIBLE_DEVICES=0,1,2,4。</li></ul>|
|ASCEND_CANN_PACKAGE_PATH ASCEND_HOME_PATH|CANN包实际安装路径。|必选|编译算子所需指定的CANN包实际安装目录。默认为/usr/local/Ascend/ascend-toolkit/latest。|

## 卸载<a name="ZH-CN_TOPIC_0000002302389376"></a>

用户如需移除Rec SDK Torch软件包，可参考以下命令进行卸载。

```bash
# 卸载dynamic_emb
pip3 uninstall dynamic_emb -y
# 卸载torchrec
pip3 uninstall torchrec -y
```

## 升级<a name="ZH-CN_TOPIC_0000002302389340"></a>

用户如需将当前版本的Rec SDK Torch升级至最新版本，可将最新的Rec SDK Torch软件包上传至安装环境后，在软件包所在目录下使用命令进行升级，具体命令如下。

1. 升级Rec SDK Torch新版本时，需要先手动卸载旧版本。具体操作请参考[卸载](#卸载)。
2. 参考[安装Rec SDK Torch](#section182972951211)重新安装Rec SDK Torch。
