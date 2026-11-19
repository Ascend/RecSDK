
# RankMixer-ShareMOE

首次运行时，[benchmark 配置](../../configs/RANKMIXER_SHAREMOE.json)会在该模型独立克隆的 RecSDK 中应用 ShareMOE 补丁，再将打好补丁的 `ScalingLaw` 复制到 ShareMOE 的运行目录。

# 论文

https://www.arxiv.org/pdf/2507.15551

# 输入处理

1. 参考[Meta GR](https://github.com/meta-recsys/generative-recommenders)的序列数据处理方法,生成sasrec格式的输入数据 `sasrec_format.csv`
2. 将环境变量 `data_dir` 设置为第 1 步生成的 CSV 所在目录的绝对路径。

# 运行实例

从 RecSDK 仓库根目录启动 NPU 推理：

```shell
cd benchmark
python run.py RANKMIXER_SHAREMOE.json
```

运行 GPU 推理时，将本地配置 `run_cmd` 末尾的 `bash run.sh npu` 改为 `bash run.sh gpu`，再执行同一条 benchmark 命令。

# 参数配置

当前提供0.2B、0.5B、1B的配置文件，详见sample_configs目录，使用时需要在run.sh文件中修改config_file，选择不同的配置文件
