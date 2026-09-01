# RecSDK-Torch 模型样例适配介绍

| 模型名称 | 开源模型参考 | 模型简介 | 说明 | 适配路径 |
|---------|---------|---------|---------|---------|
| DIN (Deep Interest Network) | [alibaba/TorchEasyRec](https://github.com/alibaba/TorchEasyRec.git) | 深度兴趣网络模型，用于推荐系统中的兴趣建模 | 使用NPU算子进行训练加速。<br>开源代码依赖的三方库只有x86架构，该迁移样例只支持在x86上运行。 | [din/README](./din/README.md) |
| DLRM (Deep Learning Recommendation Model)  | [facebookresearch/dlrm](https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm/) | Facebook开源的深度学习推荐模型，广泛应用于推荐系统 | 适配torchrec框架并在NPU上进行训练。 | [dlrm/README](./dlrm/README.md) |
| GR (Generative Recommendations) - Meta版 | [facebookresearch/generative-recommenders](https://github.com/facebookresearch/generative-recommenders) | Meta开源的生成式推荐模型 | 使用NPU的HSTU融合算子来实现性能优化。 | [gr/README](./gr/gr_meta/README.md) |
| GR (Generative Recommendations) - NV版 | [NVIDIA/recsys-examples](https://github.com/NVIDIA/recsys-examples/tree/main/examples) | NVIDIA基于recsys-gr的生成式推荐模型 | 使用NPU的HSTU融合算子来实现性能优化。 | [gr_nv/README](./gr_nv/README.md) |
| MMoE (Multi-gate Mixture-of-Experts) | [shenweichen/DeepCTR](https://github.com/shenweichen/DeepCTR/) | 多门控混合专家模型，常用于多任务学习场景 | 包含数据预处理和模型训练脚本。 | [README](./model_zoo/README.md) |
| ETA (End-to-end Target Attention) | [End-to-End User Behavior Retrieval in Click-Through Rate Prediction Model](https://arxiv.org/pdf/2108.04468.pdf) | 适用于超长序列建模，同时具有较高的预测准确率和较低的计算复杂度 | 包含数据预处理和模型训练脚本。 | [README](./model_zoo/README.md) |

# RecSDK-Torch 模型样例运行环境说明

## 版本配套说明

本模型迁移依赖特定版本的CANN、PyTorch、驱动和固件，源码编译需使用指定版本的Python、GCC、CMake等工具，仅支持昇腾平台（Atlas 800T A2），基础软件版本以Rec SDK Torch提供的基础镜像环境为准，主要配套软件版本如下：

| 软件名称  | PyTorch | TorchNPU | torchrec  | fbgemm_gpu | hybrid_torchrec | torchrec_embcache |
|-------|---------|-----------|-----------|------------|-----------------|-------------------|
| 配套版本 | 2.7.1+cpu   | 2.7.1     | 1.2.0+npu | 1.2.0+cpu      | 1.2.0           | 1.2.0             |

注：在26.1.0及之后版本的镜像中，hybrid_torchrec、torchrec_embcache、torchrec三个包作为子包统一被torch-rec-v1软件包包装，此时无法直接使用pip3查看3个子包信息，但可直接在Python脚本中进行import使用。

## 基础镜像

基础镜像请参见[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5)中“镜像下载”页面，下载最新版本的RecSDK-Torch镜像。

注：当前基础镜像中提供的软件版本为PyTorch 2.6.0配套，请在**启动容器**后参见[升级PyTorch 2.7.1版本配套](#升级pytorch-271版本配套)进行软件版本升级。

## 启动容器

创建启动脚本run_docker.sh，参考如下（启动命令仅作参考，按需挂载目录）：

```shell
#!/bin/bash
container_name=$1
image_name=$2
docker run \
-it \
--name "${container_name}" \
-e ASCEND_VISIBLE_DEVICES=0-7 \
--shm-size="300g" \
-m 300g \
-v /etc/localtime:/etc/localtime:ro \
-v /etc/ascend_install.info:/etc/ascend_install.info:ro \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
"${image_name}" \
/bin/bash
```

部分参数说明：

- -m 300g：设置容器内使用内存大小，可根据实际情况进行配置。
- -e ASCEND_VISIBLE_DEVICES=0-7：将服务器上编号为device0-device7的NPU设备挂载到容器内，可根据实际情况进行配置。
- image_name：镜像名称，该参数为`REPOSITORY:TAG`形式，例如[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5)中26.0.0版本的x86镜像的镜像名称为：`swr.cn-south-1.myhuaweicloud.com/ascendhub/rec_sdk-torch:26.0.0_debian12-x86`。

执行如下命令新建容器：

```shell
bash run_docker.sh 容器名 镜像名称
```

## 刷新容器内环境变量

```shell
# 配置CANN环境变量
source /usr/local/Ascend/cann/set_env.sh
# Python虚拟环境存在时激活虚拟环境。使用完后若需退出 Python 虚拟环境，执行命令： deactivate 即可退出。
[ -f /opt/buildtools/torch_v1_pt2.7.1/bin/activate ] && source /opt/buildtools/torch_v1_pt2.7.1/bin/activate
```

> [!NOTE]
> [昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5)中26.1.0及之后的镜像中内置了Python虚拟环境，使用前需激活虚拟环境，该虚拟环境中已默认安装好Rec SDK Torch及相关依赖。

## 安装依赖

说明：[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5)提供镜像中已经安装好torchrec、hybrid_torchrec、torchrec_embcache及算子等依赖，**如无需重新安装可跳过后续步骤**。如需重新安装依赖可参考如下步骤，并确保网络通畅。

### 1. 安装TorchRec昇腾注册包

TorchRec昇腾注册包是基于TorchRec源码做的NPU设备适配。可通过Rec SDK Torch提供的patch文件和TorchRec源码的固定分支编译出该注册包。
   
请参见[README](https://gitcode.com/Ascend/RecSDK/blob/develop/training/torch_rec_v1/torchrec_npu/README.md)进行源码编译和安装。

### 2. 安装Rec SDK Torch推荐算法框架包

请参见如下指令进行编译安装：

```shell
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

### 3. 安装自定义算子相关包

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

    参考[安装依赖](#安装依赖)章节进行源码编译。
