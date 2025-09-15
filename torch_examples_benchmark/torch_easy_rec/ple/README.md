# ple模型NPU适配

## 适配说明

本样例的适配对象为Progressive Layered Extraction (简称PLE)模型, 将其迁移至NPU侧训练/推理。

模型参考的开源链接为: https://github.com/alibaba/TorchEasyRec.git 因开源代码依赖的三方库只支持x86架构的版本，本样例迁移也暂只支持在x86上运行。

克隆源码并固定版本为:Commits on May 30, 2025，提交的SHA-1 hash值（提交ID）：9ffe1f09d336d3a5cdb5bb6970aa8cc8bc648b2e

验证运行的算力平台：Atlas A2系列产品

## 配置文件准备
```shell
wget https://tzrec.oss-cn-beijing.aliyuncs.com/config/models/ple_taobao.config -O ../TorchEasyRec/model_configs/ple_taobao.config
sed -i '1,2d' ../TorchEasyRec/model_configs/ple_taobao.config
sed -i '1i train_input_path: "/data/taobao_data_train/*.parquet"' ../TorchEasyRec/model_configs/ple_taobao.config
sed -i '2i eval_input_path: "/data/taobao_data_eval/*.parquet"' ../TorchEasyRec/model_configs/ple_taobao.config
sed -i 's@OdpsDataset@ParquetDataset@' ../TorchEasyRec/model_configs/ple_taobao.config
```

如果需要profiling，则增加如下配置：
```shell
sed -i '19i is_profiling:True' ../TorchEasyRec/model_configs/ple_taobao.config
```

## 模型运行
```shell
cp run_ple.sh ../TorchEasyRec
cd ../TorchEasyRec
bash run_ple.sh
```
