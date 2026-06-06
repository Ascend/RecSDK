# RecSDK-Torch 模型样例运行环境说明

## 版本配套说明

本模型迁移依赖特定版本的CANN、PyTorch、驱动和固件,源码编译需使用指定版本的Python、GCC、CMake等工具,仅支持昇腾平台（Atlas 800T A2）,软件环境以Rec SDK Torch提供的基础镜像环境为准。

基于PyTorch开源软件版本，支持两种软件版本配套，可根据需要自行选择。

| 配套版本  | PyTorch | torch-npu | torchrec  | fbgemm_gpu | hybrid_torchrec | torchrec_embcache |
|-------|---------|-----------|-----------|------------|-----------------|-------------------|
| 配套版本1 | 2.6.0   | 2.6.0     | 1.1.0+npu | 1.1.0      | 1.1.0           | 1.1.0             |
| 配套版本2 | 2.7.1   | 2.7.1     | 1.2.0+npu | 1.2.0      | 1.2.0           | 1.2.0             |

## 基础镜像

下载基础镜像地址为：https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5

下载镜像版本为:26.0.0_openeuler2203-arm

注：该镜像环境已经包含以上“配套版本1”中依赖和必要的NPU算子,如需要使用PyTorch 2.7.1配套，可参考[README](https://gitcode.com/Ascend/RecSDK/blob/develop/docs/zh/torch/build_torch_rec_images/README.md)中"版本配套说明"章节，下载对应软件重新安装。
安装步骤可参考以下"安装依赖"章节

## 启动容器

说明：以下启动命令仅作参考，按需挂载目录。

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

- free_devices：检测当前空闲NPU卡号
- -m 300g：设置容器内使用内存大小，可根据实际情况进行配置。
- -e ASCEND_VISIBLE_DEVICES="${free_devices}"：将服务器上空闲的NPU设备挂载到容器内，可根据实际情况进行配置。

执行如下命令新建容器：

```shell
bash run_docker.sh 容器名 {镜像名称}:{版本名称}
```

## 设置环境变量

进入容器后，设置环境变量

```shell
source /etc/profile
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 环境可用性验证

```shell
# 执行命令,正常回显卡信息说明环境可用。
npu-smi info
```

说明：容器启动命令推荐为非特权模式，如果有其他容器也挂载了卡可能导致容器内卡不能使用，常见错误提示为："dcmi model initialized failed, because the device is used. ret is -8020",
此时请停止其他容器,确保环境可用。

## 模型样例验证

如需了解模型构建细节可进入little_demo目录参考README.md，也可直接运行以下命令进行验证。

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

## 安装依赖（如需要）

说明：容器中已经安装好torchrec,hybrid_torchrec,torchrec_embcache及算子等依赖。如需重新安装依赖需确保网络通畅。

### 1. 安装TorchRec昇腾注册包

TorchRec昇腾注册包为基于torchrec开源代码固定分支，进行NPU设备适配后的包，支持源码编译安装。

参考：https://gitcode.com/Ascend/RecSDK/blob/develop/training/torch_rec_v1/torchrec_npu/README.md

### 2. 安装Rec SDK Torch训练框架包

提供通过安装包安装、源码编译安装两种方式，选择其一即可。

#### 2.1 通过安装包安装

从[RecSDK release版本](https://gitcode.com/Ascend/RecSDK/releases)，选择最新版本，下载Ascend-mindxsdk-hybrid-torchrec-*.tar.gz软件包。

说明：hybrid_torchrec软件包在7.x.x版本同时支持torch 2.6.0和torch 2.7.1两种配套版本。之前的版本包仅支持torch 2.6.0配套。

```shell
tar zxvf Ascend-mindxsdk-hybrid-torchrec*.tar.gz
# 如果已安装，请先卸载
pip3 uninstall -y hybrid_torchrec torchrec_embcache
# 安装软件包
pip3 install hybrid_torchrec-*-py3-none-linux*.whl
pip3 install -r requirements.txt
pip3 install torchrec_embcache-*-py3-none-linux*.whl
```

#### 2.2 源码编译安装

通过源码编译方式安装Rec SDK Torch训练框架包。

参考：https://gitcode.com/Ascend/RecSDK/blob/develop/training/torch_rec_v1/hybrid_torchrec/README.md

参考README编译完成后，会在编译脚本build_whl.sh的同层级目录下生成tar.gz包。其中同时包含hybrid_torchrec、torchrec_embcache的whl包，解压安装即可。

### 3. 安装自定义算子和算子适配层

源码编译Ascend-recsdk-npu-ops*.tar.gz软件包，参考：[算子编译安装说明](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/build/README.md)

```shell
# 安装所需算子
tar -zxvf Ascend-recsdk-npu-ops-*.tar.gz
cd recsdk-npu-ops/recsdk_ops/
unset ASCEND_CUSTOM_OPP_PATH
bash mxrec_opp_backward_codegen_adagrad_unweighted_exact.run
bash mxrec_opp_split_embedding_codegen_forward_unweighted.run
bash mxrec_opp_permute2d_sparse_data.run
bash mxrec_opp_asynchronous_complete_cumsum.run
bash mxrec_opp_dense_to_jagged.run
bash mxrec_opp_jagged_to_padded_dense.run
bash mxrec_opp_index_select_for_rank1_backward.run
bash mxrec_opp_gather_for_rank1.run
bash mxrec_opp_hstu_dense_forward.run
bash mxrec_opp_hstu_dense_backward.run

# 编译算子适配文件
cd ../../
cd recsdk-npu-ops/torch_plugin/torch_library/common
bash build_ops.sh
```

注意：执行完"编译算子适配文件"步骤后，融合算子的依赖包libfbgemm_npu_api.so会生成在同目录下的build文件夹下，同时也会生成在python默认安装的site-package路径中，也可以将该so包拷贝到指定的目录下，在后续模型运行时会配置该文件的路径 。
