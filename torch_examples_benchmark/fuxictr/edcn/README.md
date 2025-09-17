# EDCN模型NPU适配说明

本样例的适配对象为FuxiCTR提供用于点击率预测的开源库里 Huawei开源的EDCN(Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models)模型, 将其迁移至NPU侧训练。
模型参考的开源链接为: https://github.com/reczoo/FuxiCTR/tree/main/model_zoo/EDCN
克隆源码并固定版本为:Commits on Jun 17, 2025，提交的SHA-1 hash值（提交ID）：b7dff73

## 可修改配置文件
修改config文件：
增大epochs数量方便采集正确数据，Profiler收集到的数据结果会自动保存至EDCN同级目录下生成的 result 文件夹内。
在根目录下 ./FuxiCTR
```shell
cd model_zoo/EDCN/config
sed -i '33s/epochs 1/epochs 10/' model_config.yaml
```

## 在微小数据上运行模型
此外，用户可以修改数据集配置和模型配置文件，以在自己的数据集上运行或使用新的超参数运行。更多详细信息可以在 README 中找到。
在根目录下 ./FuxiCTR
```shell
cd model_zoo/EDCN
python run_expid.py --expid EDCN_test --gpu 0
```
