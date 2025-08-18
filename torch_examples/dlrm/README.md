# DLRM模型迁移样例说明

## 适配说明

本样例以DLRM模型为例,适配torchrec框架并在NPU上进行训练。 模型参考的开源链接为:https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm/ 克隆源码并固定版本为:Commits on Jun 7 , 2024，提交的SHA-1 hash值（提交ID）：b631a99

## 代码结构说明

```shell
├── dlrm_npu.patch         # 模型迁移适配patch文件
├── generate_data.patch    # 随机生成模型样例
├── README.md              # 样例迁移说明文档
└── run.sh                 # 模型运行脚本
```

## 版本配套说明
本模型迁移依赖特定版本的CANN、PyTorch、驱动和固件,源码编译需使用指定版本的Python、GCC、CMake等工具,仅支持昇腾平台（Atlas 800T A2）,基于软件环境以RecSDK-Torch提供的基础镜像环境为准，主要的配套依赖如下表所示：

| Python版本   | 主要配套依赖                                                                                            |
|------------|---------------------------------------------------------------------------------------------------|
| Python3.11 | torch==2.6.0<br/>torch_npu==2.6.0<br/>fbgemm+gpu==1.1.0+cpu<br/>torchrec==1.1.0+npu<br/>hybrid_torchrec==1.1.0 |

### 基础镜像
下载基础镜像地址为：

### 启动容器
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
-v /etc/localtime:/etc/localtime:ro \
-v /etc/ascend_install.info:/etc/ascend_install.info:ro \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
"${image_name}" \
/bin/bash
```
执行如下命令新建容器：
```shell
bash run_docker.sh 容器名 {镜像名称}:{版本名称}
```

### 设置环境变量
进入容器后，设置环境变量
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 如果是arm镜像启动容器后手动设置python环境
export LD_LIBRARY_PATH=/usr/local/python3.11.0/lib/:$LD_LIBRARY_PATH
export PATH=/usr/local/python3.11.0/bin:$PATH
```

### 安装依赖
说明：容器中已经安装好torchrec,hybrid_torchrec以及算子等依赖。如需重新安装依赖需确保网络通畅，请选择（1）通过获取安装包或者（2）源码编译的方式。

1.获取安装包安装
```shell
# 如果已经安装,请先卸载
pip3 uninstall -y hybrid_torchrec torchrec
# 安装torchrec
tar -zxvf Ascend-mindxsdk-torchrec-1.1.0-npu-*.tar.gz
pip3 install torchrec-1.1.0+npu-*.whl
pip3 install -r requirements.txt

# 安装hybrid_torchrec
tar -zxvf Ascend-mindxsdk-hybrid-torchrec-1.1.0-*.tar.gz
pip3 install hybrid_torchrec-1.1.0-*.whl

# 安装算子
tar -zxvf Ascend-mindxsdk-mxrec-add-ons-*.tar.gz
cd mindxsdk-mxrec-add-ons/mxrec_ops/
bash mxrec_opp_backward_codegen_adagrad_unweighted_exact.run
bash mxrec_opp_split_embedding_codegen_forward_unweighted.run
bash mxrec_opp_permute2d_sparse_data.run
bash mxrec_opp_asynchronous_complete_cumsum.run

# 编译算子适配文件
cd ../../
cd mindxsdk-mxrec-add-ons/torch_plugin/torch_library/2.6.0/common
bash build_ops.sh
```
2.源码编译安装

（1）编译安装torchrec
参考：https://gitee.com/ascend/RecSDK/blob/develop/torchrec/README.md

（2）编译安装hybrid_torchrec
参考：https://gitee.com/ascend/RecSDK/blob/develop/torchrec/hybrid_torchrec/README.MD

（3）编译安装算子和适配文件

```shell
# 克隆源码仓
git clone -b develop https://gitee.com/ascend/RecSDK.git  # 如果已经克隆此分支请忽略
# 算子编译
cd RecSDK/mxrec_add_ons/build
bash build.sh
# 进入打包文件
cd ../../../RecSDK/mxrec_add_ons/output
# 解压安装包
tar -zvxf Ascend-mindxsdk-mxrec-add-ons-*.tar.gz
# 进入算子目录
cd mindxsdk-mxrec-add-ons/mxrec_ops
# 安装算子--参考以上安装方法
# 编译算子适配文件--参考以上编译方法
```

注意：执行完"编译算子适配文件"步骤后，融合算子的依赖包libfbgemm_npu_api.so会生成在同目录下的build文件夹下，同时也会生成在python默认安装的site-package路径中，也可以将该so包拷贝到指定的目录下，在后续模型运行时会配置该文件的路径 。


## dlrm源码适配

进入当前目录，下载官方模型代码后，并使用patch文件进行修改。
```shell
git clone -b main https://github.com/facebookresearch/dlrm.git
cd dlrm && git checkout b631a99 
cp -f ../dlrm_npu.patch ./
git apply dlrm_npu.patch
```

### 数据集下载
说明：本样例提供两种获取数据集的方式：使用官网数据集可验证模型性能和精度，若仅验证模型功能跑通可使用随机数据集。

1.官网数据集

官网提供两种方式跑通demo：

（1）下载原始数据处理后，提前进行mutil-hot的合成，产生4T的数据

（2）下载原始数据处理后，在训练的过程中生成mutil-hot数据，使用690gb数据集

由于(1)需要的条件苛刻，大部分机器很难满足条件，本次演示使用(2)中的条件。无host瓶颈的情况下，对性能影响较小。需要修改模型脚本代码，让host生成的数据在pin_memory上。

进入[开源模型官网](https://github.com/facebookresearch/dlrm/blob/main/torchrec_dlrm/README.MD)，按照指引下载数据集到指定目录。该数据集已经托管到HuggingFace:https://huggingface.co/datasets/criteo/CriteoClickLogs 也可直接前往下载。


2.使用生成的数据集
```shell
mkdir generate_data
cp generate_data.py generate_data
cd generate_data
python3 generate_data.py
```

数据集准备完成后的格式如下，后续模型运行时会配置该数据集文件路径。
```shell
day_0_sparse.npy
day_0_dense.npy
day_0_labels.npy
...
day_23_sparse.npy
day_23_dense.npy
day_23_labels.npy
```
说明：数据集较大，数据下载时间较长，请预留时间和磁盘空间，官网数据集大约690GB,随机生成数据集大约71GB


## 修改脚本并运行

修改run.sh文件中的参数，后拷贝到torchrec_dlrm目录下(与dlrm_main.py同级目录)，然后运行模型。

```shell
# 环境参数配置说明（根据实际情况修改）
export LIB_FBGEMM_NPU_API_SO_PATH="/path/to/libfbgemm_npu_api.so"                     # 算子适配文件libfbgemm_npu_api.so文件路径
export PREPROCESSED_DATASET="/path/to/data"                                           # 数据集文件路径
export WORLD_SIZE=8                                                                   # 运行npu卡数，默认8卡
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7                                      # 可用npu卡编号，与WORLD_SIZE数量保持一致

# 运行代码
bash run.sh
```

## 精度、性能对比

为与开源模型比较性能和精度，模型的默认配置参数不建议修改，对比结果如下表所示。

| Device Type | Number of GPUs/NPUs |Collective Size of Embedding Tables (GiB)|Local Batch Size|Global Batch Size|Learning Rate|Interaction Type|Optimizer| AUROC Over Test Set After 1 Epoch | Training speed                        | Time to Train 1 Epoch |Unique Flags|
|-------------|---------------------| --- | --- | --- | --- | --- | --- |-----------------------------------|---------------------------------------|-----------------------| --- |
| GPU         | 8                   |104.54|2,048|16,384|0.006|DCN v2|Adagrad| 0.7973                            | ~55.0 batches/s == ~901,120 samples/s | 1h20m21s              |`--batch_size 2048 --learning_rate 0.006 --adagrad --interaction_type=dcn` |
| NPU         | 8                   |104.54|2,048|16,384|0.006|DCN v2|Adagrad| 0.7975                            | ~59.0 batches/s == ~966,656 samples/s | 1h12m03s              |`--batch_size 2048 --learning_rate 0.006 --adagrad --interaction_type=dcn`|

说明：NPU测试结果为在参考镜像的X86环境上的测试结果。GPU测试数据参考: https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm/ 。
