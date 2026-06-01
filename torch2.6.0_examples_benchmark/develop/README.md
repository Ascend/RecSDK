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

本模型迁移依赖特定版本的CANN、PyTorch、驱动和固件，源码编译需使用指定版本的Python、GCC、CMake等工具，仅支持昇腾平台（Atlas 800T A2），软件环境以Rec SDK Torch提供的基础镜像环境为准。

| 配套版本  | PyTorch | torch-npu | torchrec  | fbgemm_gpu | hybrid_torchrec | torchrec_embcache |
|-------|---------|-----------|-----------|------------|-----------------|-------------------|
| 配套版本1 | 2.6.0   | 2.6.0     | 1.1.0+npu | 1.1.0      | 1.1.0           | 1.1.0             |

## 基础镜像

基础镜像请参见[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5)中“镜像下载”页面，下载最新版本的RecSDK-Torch镜像。

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

执行如下命令新建容器：

```shell
bash run_docker.sh 容器名 {镜像名称}:{版本名称}
```

## 设置环境变量

进入容器后，设置环境变量：

```shell
source /etc/profile
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 安装依赖

说明：最新的容器已经安装好torchrec、hybrid_torchrec、torchrec_embcache及算子等依赖，**如无需重新安装可跳过后续步骤**。如需重新安装依赖可参考如下步骤，并确保网络通畅。

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

下载[RecSDK](https://gitcode.com/Ascend/RecSDK)源码，按如下指令进行算子相关包的编译和安装：

```bash
# 编译算子前，需使能CANN环境变量。默认路径安装CANN包时，使能CANN环境变量指令如下：
source /usr/local/Ascend/cann/set_env.sh
unset ASCEND_CUSTOM_OPP_PATH

# 编译并安装算子包（Ascend-recsdk-npu-ops-\*-linux-\*.tar.gz）。
cd RecSDK
git submodule update --init --recursive 
cd cust_op/ascendc_op/build
bash build_ai_core_op.sh A2

# 可选：若仅需安装部分算子，可在其他容器内编译，并将build/output/recsdk_ops路径下所需算子包拷贝到当前环境，参考如下指令安装：
# bash mxrec_opp_split_embedding_codegen_forward_unweighted.run

# 安装算子适配层（libfbgemm_npu_api.so）
cd ../../framework/torch_plugin/torch_library/common/
bash build_ops.sh
```

注意：执行完"安装算子适配层"步骤后，融合算子的依赖包libfbgemm_npu_api.so会生成在同目录下的build文件夹下，并自动拷贝到python默认的site-package目录。也可以将该so包拷贝到指定的目录，在后续模型运行时会配置该文件的路径。
