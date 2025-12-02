# RecModelZoo(pytorch版本)

该文档介绍如何在NPU进行RecsysGR模型的推理和打印profiling，统计性能。


## 主要依赖
**Python:** 3.12.3
**Pytorch:** 2.7.0
**torch_npu:** 2.7.0

# 模型代码准备
运行launch.sh准备代码
 ```shell
 bash launch.sh
 ```

## 模型推理
 ```shell
cd examples/hstu
bash run_random_2k.sh
 ```

| 参数名称 | 参数说明                              |
|-|-----------------------------------|
| USE_COMPILE | 是否启动inductor模式，0关闭，1启动            |
| USE_GRAPH | USE_GRAPH是否开启graph模式，0关闭，1启动      |
| RUN_NUM | 运行轮数(运行多少轮采集性能数据)                 |
| USE_E2E | 是否开启端到端时间统计 1开启                   |
| USE_PROFILING | 是否采集profiling 1采集                 |
| USE_GPU_INPUT | 是否加载GPU输入,维持和GPU输入一致  0关闭1打开      |
| SAVE_INPUT_AND_PUTPUT | 是否保存输入和输出tensor, 和GPU对比精度  0关闭1打开 |
| MODEL_PATH | 模型所在路径                            |
| LIB_FBGEMM_NPU_API_SO_PATH | libfbgemm_npu_api.so库所在路径         |
| PROF_SAVE_PATH | PROF_SAVE_PATH存放的位置               |
| BATCH_SIZE | 可选 1 2 4                          |


## Profiling

保存在脚本里PROF_SAVE_PATH值的目录下

1. 打开*report_bs1.json可看到运行info
例如{'Batch_size': 1, 'model_name': 'GR', 'QPS': 26, 'AVG Latency': 37.919027, 'P99 Latency': 38.726255, 'P90 Latency': 38.726255}

2. 打开op_statistic.csv可看到算子信息
3. 将输出的*trace.json文件拖入chrome://tracing页面即可查看运行时详细信息
    


