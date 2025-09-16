# EulerNet模型NPU适配说明

## 适配说明

本样例的适配对象为FuxiCTR提供用于点击率预测的开源库里 Huawei开源的EulerNet(EulerNet: Adaptive Feature Interaction Learning via Euler's Formula for CTR Prediction)模型, 将其迁移至NPU侧训练。
模型参考的开源链接为: https://github.com/reczoo/FuxiCTR/tree/main/model_zoo/EulerNet
克隆源码并固定版本为:Commits on Jun 17, 2025，提交的SHA-1 hash值（提交ID）：b7dff73

## 在微小数据上运行模型
此外，用户可以修改数据集配置和模型配置文件，以在自己的数据集上运行或使用新的超参数运行。更多详细信息可以在 README 中找到。
```shell
cd model_zoo/EulerNet
python run_expid.py --expid EulerNet_test --gpu 0
```

Profiler收集到的数据结果会自动保存至EulerNet同级目录下生成的 result 文件夹内

## 可修改配置文件
修改config文件：
增大epochs数量方便采集正确数据
```shell
cd /model_zoo/EulerNet/config
sed -i '53s/epochs 1/epochs 10/' model_config.yaml
```