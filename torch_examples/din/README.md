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

## 运行环境准备
请参考：https://gitcode.com/Ascend/RecSDK/tree/develop/torch_examples/README.md

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
说明：protoc版本需要>=3.X，如默认安装的版本过低，请手动升级安装。
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
