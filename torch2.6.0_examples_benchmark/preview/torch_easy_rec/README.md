# torchEasyRec仓模型适配NPU
## 软件配套说明

| 软件包简称   | 配套版本   |
|---------|--------|
| Python  | 3.11.0 |
| Pytorch | 2.6.0  |
| Fbgemm  | 1.1.0  |

## CANN、驱动、Kernels包

| 软件           | 版本           | 下载链接                                                                                                                 |
|--------------|--------------|----------------------------------------------------------------------------------------------------------------------|
| CANN-toolkit | 8.2.RC1  | https://www.hiascend.com/developer/download/community/result?module=pt+cann                                          |
| CANN-kernels | 8.2.RC1  | https://www.hiascend.com/developer/download/community/result?module=pt+cann                                          |
| driver       | 1.0.28.alpha | https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=32&cann=8.2.RC1&driver=Ascend+HDK+25.2.0

## 安装依赖

请根据机器架构、机器型号选择合适的安装包进行安装。

参考：https://gitcode.com/Ascend/RecSDK/blob/develop/training/torch_rec_v1/hybrid_torchrec/README.md

## torchEasyRec源码适配

代码修改部分已经编写在`torch_easy_rec_npu.patch`中，载入命令如下：

```bash
git clone https://github.com/alibaba/TorchEasyRec.git
cd TorchEasyRec && git checkout 9ffe1f09d336d3a5cdb5bb6970aa8cc8bc648b2e
cp ../torch_easy_rec_npu.patch ./ && git apply torch_easy_rec_npu.patch
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
下载训练数据和评估数据。
```shell
# 下载并解压
mkdir -p data
mkdir model_configs
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_train.tar.gz
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_eval.tar.gz
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_recall_train_transformed.tar.gz
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_recall_eval_transformed.tar.gz
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_ad_feature_transformed_fill.tar.gz
tar xf taobao_data_train.tar.gz -C data
tar xf taobao_data_eval.tar.gz -C data
tar xf taobao_data_recall_train_transformed.tar.gz -C data
tar xf taobao_data_recall_eval_transformed.tar.gz -C data
tar xf taobao_ad_feature_transformed_fill.tar.gz -C data
```
## 建立初始树
命令详细解释可参考：https://github.com/alibaba/TorchEasyRec/blob/master/docs/source/quick_start/local_tutorial_tdm.md
```shell
python -m tzrec.tools.tdm.init_tree \
    --item_input_path data/taobao_ad_feature_transformed_fill/\*.parquet \
    --item_id_field adgroup_id \
    --cate_id_field cate_id \
    --attr_fields cate_id,campaign_id,customer,brand,price \
    --node_edge_output_file data/init_tree \
    --tree_output_dir data/init_tree
```