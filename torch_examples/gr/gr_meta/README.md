# GR模型迁移样例说明

## 适配说明

本样例的适配对象为Meta开源的Generative Recommendations模型, 将其迁移至NPU侧训练，并使用NPU的HSTU融合算子来实现性能的优化。
模型参考的开源链接为 https://github.com/facebookresearch/generative-recommenders
克隆源码并固定版本为:Commits on Dec 16, 2024，提交的SHA-1 hash值（提交ID）：bb389f9539b054e7268528efcd35457a6ad52439


## 代码结构说明

```shell
├── gr_npu.patch    # 模型迁移适配patch文件
├── README.md       # 样例迁移说明文档
└── run.sh          # 模型运行脚本
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
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 如果是arm镜像启动容器后手动设置python环境
export LD_LIBRARY_PATH=/usr/local/python3.11.0/lib/:$LD_LIBRARY_PATH
export PATH=/usr/local/python3.11.0/bin:$PATH
```

### 安装依赖
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
bash mxrec_opp_asynchronous_complete_cumsum.run
bash mxrec_opp_dense_to_jagged.run
bash mxrec_opp_index_select_for_rank1_backward.run
bash mxrec_opp_jagged_to_padded_dense.run
bash mxrec_opp_gather_for_rank1.run
bash mxrec_opp_hstu_dense_forward.run
bash mxrec_opp_hstu_dense_backward.run

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


## GR源码适配

将 Generative Recommendations 模型迁移到NPU上并适配NPU融合HSTU算子，代码修改部分已经编写在`gr_npu.patch`中，载入命令如下：

```bash
# 进入当前目录，克隆meta开源GR模型源代码
git clone https://github.com/meta-recsys/generative-recommenders.git
cd generative-recommenders && git checkout bb389f9539b054e7268528efcd35457a6ad52439
cp ../gr_npu.patch ./ && git apply gr_npu.patch
```

## 安装模型依赖python包
```bash
pip3 install -r requirements.txt
```
说明:本模型样例是迁移NPU适配，并在pytorch框架下2.6.0配套版本运行，已忽略nvidia和tensorflow相关安装包,并调整部分配套依赖包版本。
## 数据集准备
参考源码，在preprocess_public_data.py同级目录下执行如下命令。
```shell
mkdir -p tmp/ && python3 preprocess_public_data.py
```
说明：本次测试基于ml-1m数据集，整体使用hstu-sampled-softmax-n128-large-final.gin参数配置，为适配npu算子调整个别参数见patch文件。

## 模型运行

```shell
# 拷贝运行脚本到当前目录
cp ../run.sh ./
```
修改run.sh 脚本：
```shell
export USE_NPU_HSTU=1                                                                 # 是否使用hstu算子加速
export ENABLE_RAB=0                                                                   # 是否带RAB
export LIB_FBGEMM_NPU_API_SO_PATH="/path/to/libfbgemm_npu_api.so"                     # 根据实际情况修改
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3                                              # 根据实际情况修改
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
python3 main.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin --master_port=12345 | tee temp.log
```

拷贝run.sh与main.py同级目录，执行命令：
```shell
bash run.sh
```

### 整网精度
MovieLens-1M (ML-1M):

| Method   | NDCG@10 | NDCG@50 | HR@10  | HR@50  | MRR    |
|-------|---------|---------|--------|--------|--------|
|HSTU-large| 0.1531  | 0.2142  | 0.2772 | 0.5531 | 0.1312 |

说明:以上为hstu-sampled-softmax-n128-large-final.gin参数配置,训练一轮数据的测试精度。


### 性能参考
| Steps | NPU适配 | HSTU算子加速 |
|-------|-------|----------|
| 100   | 68.59 | 16.96    | 

说明：以上表示每100步耗时，单位：秒。
## FAQ
