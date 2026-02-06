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

注：该链接中镜像环境为基于PyTorch 2.6.0版本配套。（不包含hybrid_torchrec和torchrec_embcache）

如需要使用PyTorch 2.7.1配套，可参考[README](https://gitcode.com/Ascend/RecSDK/blob/develop/docs/zh/torch/build_torch_rec_images/README.md)中"版本配套说明"章节，下载对应软件重新安装。

## 启动容器
说明：以下启动命令仅作参考，按需挂载目录。
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
进入容器后，设置环境变量
```shell
source /etc/profile
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 安装依赖
说明：容器中已经安装好torchrec,hybrid_torchrec,torchrec_embcache及算子等依赖。如需重新安装依赖需确保网络通畅。

### 1. 安装TorchRec昇腾注册包
TorchRec昇腾注册包为基于torchrec开源代码固定分支，进行NPU设备适配后的包，支持源码编译安装。

参考：https://gitcode.com/Ascend/RecSDK/blob/develop/training/torch_rec_v1/torchrec_npu/README.md

### 2. 安装Rec SDK Torch训练框架包
提供通过安装包安装、源码编译安装两种方式，选择其一即可。

#### 2.1 通过安装包安装
从[RecSDK release版本](https://gitcode.com/Ascend/RecSDK/releases)，选择最新版本，下载对应配套版本的Ascend-mindxsdk-hybrid-torchrec-*.tar.gz软件包。

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

```
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
注意：执行完"编译算子适配文件"步骤后，融合算子的依赖包libfbgemm_npu_api.so会生成在同目录下的build文件夹下，同时也会生成在python默认安装的site-package路径中，也可以将该so包拷贝到指定的目录下，在后续模型运行时会配置该文件的路径。
