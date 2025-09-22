# tdm模型NPU适配

## 适配说明

本样例的适配对象为tdm召回模型, 将其迁移至NPU侧训练/推理。

模型参考的开源链接为: https://github.com/alibaba/TorchEasyRec.git 因开源代码依赖的三方库只支持x86架构的版本，本样例迁移也暂只支持在x86上运行。

克隆源码并固定版本为:Commits on May 30, 2025，提交的SHA-1 hash值（提交ID）：9ffe1f09d336d3a5cdb5bb6970aa8cc8bc648b2e

验证运行的算力平台：Atlas A2系列产品

## 配置文件准备
```shell
wget https://tzrec.oss-cn-beijing.aliyuncs.com/config/quick_start/tdm_taobao_local.config -O ../TorchEasyRec/model_configs/tdm_taobao.config
sed -i '19,21d' ../TorchEasyRec/model_configs/tdm_taobao.config
sed -i '19i num_steps: 50' ../TorchEasyRec/model_configs/tdm_taobao.config
sed -i '19i log_step_count_steps: 1' ../TorchEasyRec/model_configs/tdm_taobao.config
sed -i '19i save_checkpoints_steps: 10' ../TorchEasyRec/model_configs/tdm_taobao.config
```

如果需要profiling，则增加如下配置：
```shell
sed -i '19i is_profiling:True' ../TorchEasyRec/model_configs/tdm_taobao.config
```

## 模型运行
```shell
cp run_tdm.sh ../TorchEasyRec
cd ../TorchEasyRec
bash run_tdm.sh
```
