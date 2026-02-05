# RecModelZoo(pytorch版本)

本样例的适配对象为 recsys-gr 模型

模型参考的开源链接为 https://github.com/NVIDIA/recsys-examples/tree/main/examples
用于NPU适配对比。

# 模型代码准备
运行launch.sh准备代码
 ```shell
 bash launch.sh
 ```

# 参数说明
| 参数名称 | 参数说明                              |
|-|-----------------------------------|
| USE_COMPILE | 是否启动inductor模式，0关闭，1启动            |
| USE_GRAPH | USE_GRAPH是否开启graph模式，0关闭，1启动      |
| RUN_NUM | 运行轮数(运行多少轮采集性能数据)                 |
| USE_E2E | 是否开启端到端时间统计 1开启                   |
| USE_PROFILING | 是否采集profiling 1采集                 |
| USE_GPU_INPUT | 是否加载GPU输入,维持和GPU输入一致  0关闭1打开      |
| SAVE_INPUT_AND_OUTPUT | 是否保存输入和输出tensor, 和GPU对比精度  0关闭1打开 |
| MODEL_PATH | 模型所在路径                            |
| SAVE_MODEL | 是否导出模型参数，1启用，0关闭 |
| PROF_SAVE_PATH | PROF_SAVE_PATH存放的位置               |
| BATCH_SIZE | 可选 1 2 4                          |
| SEQ_LEN | 序列长度，当前默认2048           |
| CANDIDATE_LEN | 候选序列长度，当前默认长度100      |
| HIDDEN_DIM | 隐藏层大小 |
| HEAD_NUM| 注意力头的个数 |
| HEAD_DIM| 注意力维度大小，当前HEAD_NUM*HEAD_DIM=HIDDEN_DIM |
| USE_NEW_HSTU_OP |   RecSDK 7.3.0后及之后的版本设置为1      |
| USE_HSTU_OP |   是否使用hstu融合算子      |
| USE_MOE_OP |   是否使用groupMatmul算子     |
| USE_MOE | 是否开启MOE，1启用，0关闭           |
| LAYER_NUM | GR的层数      |
| MOE_EXPERT_TOTAL_NUM | MOE总专家数量                |
| MOE_EXPORT_ACT_NUM | 激活的MOE专家数                   |
| MOE_HIDDEN_SIZE| moe隐藏层大小                         |
| USE_FFN | 是否只使用FFN，1启用 0关闭  和USE_MOE不能同时开启                |
| USE_FFN_NUMBER |         使用FFN的数量       |   
| USE_RANDOM_PARM | 是否使用随机生成的模型参数，1启用 |   
| PARM_MAX_VALUE | 随机值范围，与USE_RANDOM_PARM搭配使用 |
| DYNAMIC_SHAPE_TEST | 是否生成动态序列长度的测试数据 |


# 模型推理执行命令
 ```shell
cd examples/hstu
# 0.05B推理
bash run_random_2k_05B.sh
# 1B推理
bash run_random_2k_1B.sh
# 2B推理
bash run_random_2k_2B.sh
 ```

# Profiling

保存在脚本里PROF_SAVE_PATH值的目录下

## E2E信息

USE_E2E=1时，打开*report_bs{batch_size}.json可看到运行info
例如{'Batch_size': 1, 'model_name': 'GR', 'QPS': 26, 'AVG Latency': 37.919027, 'P999 Latency': 39.726255, 'P99 Latency': 38.726255, 'P95 Latency': 38.716255， 'P90 Latency': 37.998}

## 算子profiling

USE_PROFILING=1时
1. 将输出的*trace.json文件拖入chrome://tracing页面即可查看运行时详细信息
    

## 精度验证

| 参数名称 | 参数说明   |参数取值                           |
|-|-----------------------------------|------|
| USE_RANDOM_PARM | 是否随机生成参数| 0            |
| MODEL_PATH | 模型参数加载路径      | 模型保存路径 |
| SAVE_INPUT_AND_OUTPUT | 保存计算结果  | 1 |
| USE_GPU_INPUT | 使用GPU的输入  | 1 |
| PERCISE_VALIDATE | 精度验证模式，打开确定性开关，对比inductor和eager模式  | 1 |

