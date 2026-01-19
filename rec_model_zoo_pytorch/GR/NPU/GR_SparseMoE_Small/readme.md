# RecModelZoo(pytorch版本)

本样例的适配对象为 recsys-gr 模型
模型参考的开源链接为 https://github.com/NVIDIA/recsys-examples/tree/main/examples
该文档介绍如何在NPU进行GR - MOE模型的推理适用于(0.05B/1B/2B等参数GR)。

## 主要依赖
**Python:** 3.12.3
**Pytorch:** 2.7.0
**torch_npu:** 2.7.1
**triton-ascend:**:3.4.0

# 模型代码准备
运行launch.sh准备代码
 ```shell
 bash launch.sh
 ```

## 模型推理
 ```shell
cd examples/hstu
bash run_random_2k_05k.sh
 ```

通过设置以下参数设置MOE参数
| 参数名称 | 参数说明                              |
|-|-----------------------------------|
| USE_MOE | 是否开启MOE，0关闭，1启动            |
| LAYER_NUM | GR的层数      |
| MOE_EXPERT_TOTAL_NUM | MOE总专家数量                |
| MOE_EXPORT_ACT_NUM | 激活的MOE专家数                   |
| MOE_HIDDEN_SIZE| moe隐藏层大小                         |
| USE_FFN | 是否只使用FFN，0启用 1关闭  和USE_MOE不能同时开启                |
| USE_FFN_NUMBER |         使用FFN的数量       |      



## Profiling

保存在脚本里PROF_SAVE_PATH值的目录下

1. 打开*report_bs1.json可看到运行info
例如{'Batch_size': 1, 'model_name': 'GR', 'QPS': 26, 'AVG Latency': 37.919027, 'P99 Latency': 38.726255, 'P90 Latency': 38.726255}

2. 打开op_statistic.csv可看到算子信息
3. 将输出的*trace.json文件拖入chrome://tracing页面即可查看运行时详细信息
    


