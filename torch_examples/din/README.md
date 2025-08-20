# Din模型NPU适配说明

## 适配说明

本样例的适配对象为DIN(Deep Interest Network)模型, 将其迁移至NPU侧训练，并使用NPU的算子进行训练加速。
模型参考的开源链接为: https://github.com/alibaba/TorchEasyRec.git 因开源代码依赖的三方库只支持x86架构的版本，本样例迁移也暂只支持在x86上运行。
克隆源码并固定版本为:Commits on May 30, 2025，提交的SHA-1 hash值（提交ID）：9ffe1f09d336d3a5cdb5bb6970aa8cc8bc648b2e
验证运行的算力平台：Atlas A2训练系列产品

## 代码结构说明

```shell
├── din_npu.patch    # 模型迁移适配patch文件
├── README.md        # 样例迁移说明文档
└── run.sh           # 模型运行脚本
```

## 版本配套说明
本模型迁移依赖特定版本的CANN、PyTorch、驱动和固件,源码编译需使用指定版本的Python、GCC、CMake等工具,仅支持昇腾平台（Atlas 800T A2）,基于软件环境以RecSDK-Torch提供的基础镜像环境为准，主要的配套依赖如下表所示：

| Python版本   | 主要配套依赖                                                                                            |
|------------|---------------------------------------------------------------------------------------------------|
| Python3.11 | torch==2.6.0<br/>torch_npu==2.6.0<br/>fbgemm+gpu==1.1.0+cpu<br/>torchrec==1.1.0+npu<br/>hybrid_torchrec==1.1.0 |

### 基础镜像
下载基础镜像地址为：

### 启动容器
说明：以下启动命令仅作参考，按需挂在目录。
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
source /etc/profile
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 安装Pytorch配套
说明：容器中已经安装好torchrec,hybrid_torchrec以及算子等依赖，如需重新安装需确保网络通畅，请选择（1）通过获取安装包或者（2）源码编译的方式。

1.获取安装包安装
```shell
# 如果已经安装，请先卸载
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

## DIN源码适配

将DIN(Deep Interest Network)模型迁移到NPU上并适配NPU算子，代码修改部分已经编写在`din_npu.patch`中，载入命令如下：

```bash
# 进入到当前目录，克隆源码并通过patch文件适配修改。
git clone https://github.com/alibaba/TorchEasyRec.git
cd TorchEasyRec && git checkout 9ffe1f09d336d3a5cdb5bb6970aa8cc8bc648b2e
cp ../din_npu.patch ./ && git apply din_npu.patch
```
将代码仓中.proto定义文件编译为python代码。
注意：需先安装Protocl Buffers编译器，如基于Debian/Ubuntu系统参考命令:
```bash
apt-get install protobuf-compiler
```
```bash
protoc --proto_path=./ --python_out=./ tzrec/protos/*.proto
protoc --proto_path=./ --python_out=./ tzrec/protos/models/*.proto
```

### 安装依赖
```bash
pip3 install -r requirements/runtime.txt
```
说明：部分三方库需要在指定地址安装，如遇网络问题，可按runtime.txt描述手动下载依赖库安装。


### 生成并安装源码框架
```bash
python3 setup.py sdist bdist_wheel
cd dist && pip3 install tzrec-0.7.14-*.whl
cd ..
```

## 数据集准备
下载训练数据和评估数据，配置文件以multi_tower_din_taobao_local.config为例。
```shell
# 下载并解压
mkdir -p data
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_train.tar.gz
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_eval.tar.gz
wget https://tzrec.oss-cn-beijing.aliyuncs.com/config/quick_start/multi_tower_din_taobao_local.config
tar xf taobao_data_train.tar.gz -C data
tar xf taobao_data_eval.tar.gz -C data
```

## 模型运行

修改run.sh 脚本,然后执行。

```shell
export LIB_FBGEMM_NPU_API_SO_PATH="/path/to/libfbgemm_npu_api.so"      # 根据实际情况修改

torchrun --master_addr=localhost --master_port=32555 \
         --nnodes=1 --nproc-per-node=2 --node_rank=0 \
         -m tzrec.train_eval \
         --pipeline_config_path multi_tower_din_taobao_local.config \
         --train_input_path data/taobao_data_train/\*.parquet \
         --eval_input_path data/taobao_data_eval/\*.parquet \
         --model_dir experiments/multi_tower_din_taobao_local
```
–pipeline_config_path: 训练用的配置文件

–train_input_path: 训练数据的输入路径

–eval_input_path: 评估数据的输入路径

–model_dir: 模型训练目录

## FAQ
