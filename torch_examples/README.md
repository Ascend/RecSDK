# Rec SDK Torch 模型样例运行环境说明

## 版本配套说明

模型迁移依赖特定版本的CANN、PyTorch、驱动和固件，源码编译需使用指定版本的Python、GCC、CMake等工具，软件环境和支持平台以Rec SDK Torch提供的基础镜像环境资料为准。

## 基础镜像

请参见[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5)中“镜像下载”页签，**根据环境架构**获取已经制作好的**最新运行镜像**（26.0.0及之后版本）。

上述镜像中已包含Rec SDK Torch及相关软件包，其中软件版本如下：

| 软件名称  | PyTorch | torch_npu | torchrec  | fbgemm_gpu | hybrid_torchrec | torchrec_embcache |
|-------|---------|-----------|-----------|------------|-----------------|-------------------|
| 配套版本 | 2.6.0   | 2.6.0     | 1.1.0+npu | 1.1.0      | 1.1.0           | 1.1.0             |

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

执行如下命令新建容器：

```shell
bash run_docker.sh 容器名 镜像名称
```

注：镜像名称参数为`REPOSITORY:TAG`形式，例如[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5)中26.0.0版本的x86镜像的镜像名称为：`swr.cn-south-1.myhuaweicloud.com/ascendhub/rec_sdk-torch:26.0.0_debian12-x86`。

### 刷新容器内环境变量

26.1.0及之后的镜像中内置了Python虚拟环境，需要刷新环境变量。使用`ls -l /opt/buildtools/torch_v1_pt2.6.0/bin/activate`指令判断Python虚拟环境是否存在：若回显包含文件详细属性表示存在，若回显包含“No such file or directory”表示不存在。

若Python虚拟环境存在则执行如下命令刷新容器内环境变量：

```shell
# 激活 torch_rec_v1 PyTorch 2.6.0 版本Python虚拟环境，该环境内已安装好Rec SDK Torch及相关软件包。
source /opt/buildtools/torch_v1_pt2.6.0/bin/activate
# 使用完成后，若需退出 Python 虚拟环境，执行命令： deactivate 即可退出。

# 切换并生效 Atlas A2 系列服务器配套CANN Toolkit及相关环境变量
source /usr/local/set_cann_env.sh a2
```

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

说明：容器中已经安装好torchrec、hybrid_torchrec、torchrec_embcache及算子等依赖。如需重新安装依赖需确保网络通畅。

### 1. 安装TorchRec昇腾注册包

TorchRec昇腾注册包为基于torchrec开源代码固定分支，进行NPU设备适配后的包，请参见[README](https://gitcode.com/Ascend/RecSDK/blob/develop/training/torch_rec_v1/torchrec_npu/README.md)进行源码编译安装。

### 2. 安装Rec SDK Torch训练框架包

请参见[源码编译](https://gitcode.com/Ascend/RecSDK/blob/develop/docs/zh/torch/torch_rec_v1/recsdk_torch_installation_guide.md#source_build_hybrid_torchrec)章节进行编译安装。

不建议使用Rec SDK的release包进行安装，有可能会导致版本不匹配问题。

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

注意：执行完"编译算子适配文件"步骤后，融合算子的依赖包libfbgemm_npu_api.so会生成在同目录下的build文件夹下，同时也会生成在python默认安装的site-package路径中。
