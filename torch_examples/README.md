# Rec SDK Torch 模型样例运行环境说明

## 版本配套说明

本模型迁移依赖特定版本的CANN、PyTorch、驱动和固件，源码编译需使用指定版本的Python、GCC、CMake等工具，仅支持昇腾平台（Atlas 800T A2），基础软件版本以Rec SDK Torch提供的基础镜像环境为准，主要配套软件版本如下：

| 软件名称  | PyTorch | TorchNPU | torchrec  | fbgemm_gpu | hybrid_torchrec | torchrec_embcache |
|-------|---------|-----------|-----------|------------|-----------------|-------------------|
| 配套版本 | 2.7.1+cpu | 2.7.1    | 1.2.0+npu | 1.2.0+cpu  | 1.2.0           | 1.2.0             |

注：在26.1.0及之后版本的镜像中，hybrid_torchrec、torchrec_embcache、torchrec三个包作为子包统一被torch-rec-v1软件包包装，此时无法直接使用pip3查看3个子包信息，但可直接在Python脚本中进行import使用。

## 基础镜像

基础镜像请参见[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5)中“镜像下载”页面，下载最新版本的RecSDK-Torch镜像。

注：当前基础镜像中提供的软件版本为PyTorch 2.6.0配套，请在**启动容器**后参见[升级PyTorch 2.7.1版本配套](#升级pytorch-271版本配套)进行软件版本升级。

## 启动容器

创建run_docker.sh脚本，写入如下命令并保存：

```shell
#!/bin/bash
container_name=$1
image_name=$2
free_devices=$(npu-smi info | grep 'No running processes found in NPU' | grep -o '[0-9]\+' | paste -sd ',' -)

if [ -z "${free_devices}" ]; then
    echo "No free devices! Stop docker running."
    exit 1
fi

docker run \
-it \
--name "${container_name}" \
-e ASCEND_VISIBLE_DEVICES="${free_devices}" \
--shm-size="300g" \
-m 300g \
-v /etc/localtime:/etc/localtime:ro \
-v /etc/ascend_install.info:/etc/ascend_install.info:ro \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
"${image_name}" \
/bin/bash
```

部分参数说明：

- free_devices：检测当前空闲NPU卡号。
- -m 300g：设置容器内使用内存大小，可根据实际情况进行配置。
- -e ASCEND_VISIBLE_DEVICES="${free_devices}"：将服务器上空闲的NPU设备挂载到容器内，可根据实际情况进行配置。
- -v /xx:/xx:ro 表示以只读的方式将宿主机目录挂载到容器内。driver目录建议挂载，源码编译时可能使用到，其余目录可按需挂载。
- image_name：镜像名称，该参数为`REPOSITORY:TAG`形式，例如[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5)中26.0.0版本的x86镜像的镜像名称为：`swr.cn-south-1.myhuaweicloud.com/ascendhub/rec_sdk-torch:26.0.0_debian12-x86`。

执行如下命令新建容器：

```shell
bash run_docker.sh 容器名 镜像名称
```

### 刷新容器内环境变量

```shell
# 配置CANN环境变量
source /usr/local/Ascend/cann/set_env.sh
# Python虚拟环境存在时激活虚拟环境。使用完后若需退出 Python 虚拟环境，执行命令： deactivate 即可退出。
[ -f /opt/buildtools/torch_v1_pt2.7.1/bin/activate ] && source /opt/buildtools/torch_v1_pt2.7.1/bin/activate
```

> [!NOTE]
> [昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5)中26.1.0及之后的镜像中内置了Python虚拟环境，使用前需激活虚拟环境。该虚拟环境中已默认安装好Rec SDK Torch及相关依赖。

### FAQ

1. 容器内执行git指令或Python脚本时报错 <a id="container_git_python_error"></a>
   
   报错示例：

        git执行时报错：`error, cannot create async thread: Operation not permitted`

        Python脚本执行时报错：`PyCapsule_Import could not import module "datatime"`

   原因：宿主机Docker版本较低时，和容器内OS存在兼容性问题，容器内无法访问系统底层指令，导致git/Python执行失败。

   处理：升级宿主机Docker版本到20.10.10及以上。或者启动Docker容器时，增加`--security-opt seccomp=unconfined`参数。示例：修改前启动容器指令为`docker run xxx`，修改后为`docker run --security-opt seccomp=unconfined xxx`。

## 环境可用性验证

```shell
# 执行命令，正常回显NPU卡信息说明环境可用
npu-smi info
```

说明：容器启动命令推荐为非特权模式，如果有其他容器也挂载了卡可能导致容器内卡不能使用，常见错误提示为：“dcmi model initialized failed, because the device is used. ret is -8020”，此时请停止其他容器，确保环境可用。

## 模型样例验证

如需了解模型构建细节可进入little_demo目录并参考对应目录下README文件进行模型验证。也可直接运行以下命令进行little_demo模型验证。

```shell
# 克隆用例仓库
git clone -b develop_examples_and_tools https://gitcode.com/Ascend/RecSDK.git
# 进入目录
cd RecSDK/torch_examples/little_demo/
# 执行单卡训练
WORLD_SIZE=1 RANK=0 python main.py  
# 执行多卡训练
bash bash.sh 
```

执行成功后出现demo done字样说明基础模型跑通。

## 开源模型迁移验证

| 模型名称                                      | 开源模型参考 | 模型简介 | 说明 | 适配路径                              |
|-------------------------------------------|---------|---------|---------|-----------------------------------|
| DLRM (Deep Learning Recommendation Model) | [facebookresearch/dlrm](https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm/) | Facebook开源的深度学习推荐模型，广泛应用于推荐系统 | 适配torchrec框架并在NPU上进行训练。 | [dlrm/README](./dlrm/README.md)   |

## 安装依赖（可选）

说明：从[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5)获取的基础镜像中已经安装好torchrec、hybrid_torchrec、torchrec_embcache及算子等依赖。如需重新安装依赖需确保网络通畅。

### 1. 安装TorchRec昇腾注册包

TorchRec昇腾注册包为基于torchrec开源代码固定分支，进行NPU设备适配后的包，请参见[README](https://gitcode.com/Ascend/RecSDK/blob/develop/training/torch_rec_v1/torchrec_npu/README.md)进行源码编译安装。

### 2. 安装Rec SDK Torch训练框架包

请参见[源码编译](https://gitcode.com/Ascend/RecSDK/blob/develop/docs/zh/torch/torch_rec_v1/02_torch_installation_guide/recsdk_torch_installation_guide.md#source_build_hybrid_torchrec)章节进行编译安装。

不建议使用Rec SDK的release包进行安装，有可能会导致版本不匹配问题。

### 3. 安装自定义算子和算子适配层

#### 3.1 安装fbgemm_ascend

从[fbgemm_ascend](https://gitcode.com/Ascend/fbgemm-ascend/releases)获取最新v1.2.0版本（配套PyTorch 2.7.1）的fbgemm_ascend算子whl包并参考如下指令安装：

```bash
pip3 uninstall -y fbgemm_ascend
pip3 install fbgemm_ascend-*.whl
```

#### 3.2 安装rec_cust_ops

下载[RecSDK](https://gitcode.com/Ascend/RecSDK)源码，按如下指令进行算子相关包的编译和安装：

```bash
# 配置CANN环境变量
source /usr/local/Ascend/cann/set_env.sh

# 编译并安装rec_cust_ops算子包
cd RecSDK/cust_op
git submodule update --init --recursive 
bash build_whl.sh
cd dist
pip3 uninstall -y rec_cust_ops
pip3 install rec_cust_ops*.whl
```

rec_cust_ops算子包更多编译安装说明请参见[rec_cust_ops编译安装](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/README.md#build_recsdk_cust_ops)。

## 附录

### 升级PyTorch 2.7.1版本配套

1. 升级PyTorch，TorchNPU，fbgemm_gpu

    卸载已安装的包：

    ```bash
    pip3 uninstall -y fbgemm_gpu torch_npu torch
    ```

    安装PyTorch和TorchNPU：请参见[PyTroch框架下载](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=175&ids=89dda9ba9de741349efa03687a487678%2C96%2C108%2C1%2C6%2C177%2C)进行安装。

    安装fbgemm_gpu:

    ```bash
    pip3 install fbgemm_gpu==1.2.0+cpu -i https://download.pytorch.org/whl/cpu 
    ```

    注：若安装fbgemm_gpu时出现SSL验证失败，可在安装指令末尾添加` --trusted-host download-r2.pytorch.org `进行临时规避。

2. 升级RecSDK相关包
    
    卸载已安装的包：

    ```bash
    pip3 uninstall -y torchrec_embcache hybrid_torchrec torchrec
    ```

    参考[安装依赖](#安装依赖可选)章节进行源码编译。
