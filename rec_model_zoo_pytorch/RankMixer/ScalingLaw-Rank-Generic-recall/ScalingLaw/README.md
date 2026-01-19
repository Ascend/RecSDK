# 论文
https://www.arxiv.org/pdf/2507.15551

# 输入处理
1. 参考[Meta GR](https://github.com/meta-recsys/generative-recommenders)的序列数据处理方法,生成sasrec格式的输入数据`sasrec_format.csv`
2. 修改`run.sh`的`data_dir`路径为第1步生成`sasrec_format.csv`的父目录

# 运行实例
跑NPU：
```shell
bash run.sh npu
```
跑GPU:
```shell
bash run.sh gpu
```
默认采集profiling，如果需要采集，需要配置环境变量：
```shell
export PROFILING_FLAG=1
```

# 参数配置
当前提供0.2B、0.5B、1B的配置文件，详见sample_configs目录，使用时需要在run.sh文件中修改config_file，选择不同的配置文件
