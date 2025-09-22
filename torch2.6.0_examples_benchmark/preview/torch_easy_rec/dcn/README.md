# dcn模型NPU适配

## 适配说明

本样例的适配对象为Deep＆Cross Network（DCN）模型, 将其迁移至NPU侧训练/推理。

模型参考的开源链接为: https://github.com/alibaba/TorchEasyRec.git 因开源代码依赖的三方库只支持x86架构的版本，本样例迁移也暂只支持在x86上运行。

克隆源码并固定版本为:Commits on May 30, 2025，提交的SHA-1 hash值（提交ID）：9ffe1f09d336d3a5cdb5bb6970aa8cc8bc648b2e

验证运行的算力平台：Atlas A2系列产品

## 配置文件准备

```shell
cp dcn_taobao.config ../TorchEasyRec/model_configs/dcn_taobao.config
```

如果需要profiling，则增加如下配置：
```shell
sed -i '20i is_profiling:True' ../TorchEasyRec/model_configs/dcn_taobao.config
```

## 模型运行
```shell
cp run_dcn.sh ../TorchEasyRec
cd ../TorchEasyRec
bash run_dcn.sh
```
