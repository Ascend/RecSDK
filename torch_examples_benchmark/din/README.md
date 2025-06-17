# Din模型NPU适配

## 适配说明

本样例的适配对象为DIN(Deep Interest Network)模型, 将其迁移至NPU侧训练，并使用NPU的算子进行训练加速。

模型参考的开源链接为: https://github.com/alibaba/TorchEasyRec.git 因开源代码依赖的三方库只支持x86架构的版本，本样例迁移也暂只支持在x86上运行。

克隆源码并固定版本为:Commits on May 30, 2025，提交的SHA-1 hash值（提交ID）：9ffe1f09d336d3a5cdb5bb6970aa8cc8bc648b2e

验证运行的算力平台：Atlas A2训练系列产品

## 版本配套说明

### 软件配套说明

| 软件包简称   | 配套版本   |
|---------|--------|
| Python  | 3.11.0 |
| Pytorch | 2.6.0  |
| Fbgemm  | 1.1.0  |

### CANN、驱动、Kernels包

| 软件           | 版本           | 下载链接                                                                                                                 |
|--------------|--------------|----------------------------------------------------------------------------------------------------------------------|
| CANN-toolkit | 8.0.0.beta1  | https://www.hiascend.com/developer/download/community/result?module=pt+cann                                          |
| CANN-kernels | 8.0.0.beta1  | https://www.hiascend.com/developer/download/community/result?module=pt+cann                                          |
| driver       | 1.0.28.alpha | https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.0.0.beta1&driver=1.0.28.alpha |

## 安装依赖

请根据机器架构、机器型号选择合适的安装包进行安装。

### 安装Pytorch配套

```shell
# torch版本
pip3 install torch==2.6.0+cpu  --index-url https://download.pytorch.org/whl/cpu # x86
# fbgemm_gpu版本
pip3 install fbgemm_gpu==1.1.0+cpu -i https://download.pytorch.org/whl/cpu
# torch_npu
pip3 install torch_npu-2.6.0.*.whl
# torchrec
pip3 install torchrec-1.1.0+npu-*.whl
```
### 安装算子

进入mindxsdk-mxec-add-ons/mxrec_ops文件夹, 安装需要的昇腾适配算子。

```shell
cd mindxsdk-mxec-add-ons/mxrec_ops
bash mxrec_opp_backward_codegen_adagrad_unweighted_exact.run
bash mxrec_opp_split_embedding_codegen_forward_unweighted.run
bash mxrec_opp_asynchronous_complete_cumsum.run
```

### 编译算子适配层文件

进入mindxsdk-mxec-add-ons/torch_plugin/torch_library/2.6.0/common文件夹,编译适配层文件。

```shell
cd mindxsdk-mxec-add-ons/torch_plugin/torch_library/2.6.0/common
bash build_ops.sh
```

执行完以上命令之后，融合算子的依赖包libfbgemm_npu_api.so会生成在同目录下的build文件夹下，以及python默认安装的site-package路径。也可将该so包拷贝到某固定目录下便于在代码中加载。示例如下：
```shell
torch.ops.load_library(/path/to/libfbgemm_npu_api.so) # 根据实际路径修改
```

## DIN源码适配

将DIN(Deep Interest Network)模型迁移到NPU上并适配NPU算子，代码修改部分已经编写在`din_npu.patch`中，载入命令如下：

```bash
git clone https://github.com/alibaba/TorchEasyRec.git
cd TorchEasyRec && git checkout 9ffe1f09d336d3a5cdb5bb6970aa8cc8bc648b2e
cp ../din_npu.patch ./ && git apply din_npu.patch
```
将代码仓中.proto定义文件编译为python代码
```bash
protoc --proto_path=./ --python_out=./ tzrec/protos/*.proto
protoc --proto_path=./ --python_out=./ tzrec/protos/models/*.proto
```
注意：需先安装Protocl Buffers编译器，如基于Debian/Ubuntu系统参考命令:
```bash
apt-get install protobuf-compiler
```

## 生成wheel包并安装
```bash
python3 setup.py sdist bdist_wheel
cd dist && pip3 install tzrec-0.7.14-*.whl
cd ..
```
注意：使用pip3安装tzrec三方库时，会默认安装requirements/runtime.txt中依赖，其中部分三方库需要在指定地址安装，如遇网络问题，请手动安装依赖库，然后使用--no-deps选项安装tzrec库。

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
