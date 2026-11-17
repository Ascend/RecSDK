# tf_rec_v2 demo模型运行说明

## 代码结构

```shell
.
├── config.py        # 模型配置文件
├── dataset.py       # 数据集生成脚本
├── demo.toml        # Rec SDK训练框架和模型配置文件
├── logger.py        # 日志封装
├── main.py          # 主函数
├── model.py         # demo模型
├── op_precision.ini # 算子精度配置文件
├── README.md        # demo模型运行说明
├── run.sh           # demo运行脚本
└── runner.py        # 训练、推理流程封装
```

## 1.准备数据

demo样例无需从其他地方下载数据集，在demo样例中Rec SDK会自动生成数据集，详情见dataset.py。

## 2.准备运行环境

运行环境可以参考[Rec SDK用户指南 - 安装指南（tf_rec_v2）](https://gitcode.com/Ascend/RecSDK/blob/develop/docs/zh/tensorflow/tf_rec_v2/02_tf_installation_guide/recsdk_tf_installation_guide.md)
章节进行准备。更多关于demo的介绍也可以参考[Rec SDK快速入门（tf_rec_v2）](https://gitcode.com/Ascend/RecSDK/blob/develop/docs/zh/tensorflow/tf_rec_v2/03_quick_start/quick_start.md)。

## 3.安装Rec SDK

Rec SDK软件包可以通过[获取Rec SDK TensorFlow软件包](https://gitcode.com/Ascend/RecSDK/blob/develop/docs/zh/tensorflow/tf_rec_v2/02_tf_installation_guide/recsdk_tf_installation_guide.md)
章节进行获取，按章节指导完成安装即可。

> [!NOTE]
> run.sh脚本中通过 `which python3.7` 自动推导Rec SDK的安装路径（tf_rec_v2所在目录），
> 并设置 `PYTHONPATH` 和 `LD_LIBRARY_PATH` 环境变量。若python3.7的安装路径与默认推导结果不一致，请根据实际环境修改run.sh中的路径。

## 4.安装demo依赖

demo解析toml配置文件时需要安装toml库：

```shell
pip install toml
```

## 5.运行demo模型

执行完以上步骤之后，接下来就可以运行demo模型，其中run.sh就是运行的脚本，启动时需要传入toml配置文件路径，运行命令如：

```shell
bash run.sh demo.toml
```

demo默认使用 `demo.toml` 中的 `local_rank_size = 1` 和 `num_server = 1` 配置，即单机单卡运行。

run.sh脚本内部会通过horovod拉起训练进程，并读取toml文件中的 `interface`（训练网卡名）等参数。
可通过 `ip addr` 或 `ifconfig` 命令查询本机可用的网卡名，根据实际查询结果修改 `demo.toml` 中的 `interface`（单机场景可使用默认的 `lo`）。另外run.sh的mpi_args参数中写死了 `NCCL_SOCKET_IFNAME=docker0`，
若运行环境中不存在 `docker0` 网卡，会导致进程间通信失败，同样需要按查询结果修改run.sh中的该配置。

默认配置下 `use_ranktable = true`，run.sh会自动使用上级目录 `ranktable_samples` 中的ranktable和topo文件，
单卡场景使用 `ranktable_1p.json` 和 `topo_1p.json`。
若 `use_ranktable = false`，则会读取 `demo.toml` 中 `[mxrec.cm-node-info]` 的节点配置信息。

训练启动后，日志会通过tee落盘到当前目录下形如 `temp_{num_process}p_时间戳.log` 的文件中，训练结束后会打印 `Demo done.` 字样。

## 6.模型运行模式

通过 `demo.toml` 中 `[model]` 下的 `mode` 字段可以指定demo的运行模式：

- `train_and_evaluate`：执行训练并周期性进行评估、保存模型（默认模式）。模型保存至 `saved_path` 指定的目录。
- `load_and_train`：加载已保存的模型继续训练。
- `predict`：加载已保存的模型进行推理。

其中 `train_steps`、`eval_steps`、`train_interval`、`batch_number` 分别控制训练步数、评估步数、评估/保存间隔以及生成的batch数量。

## 7.开启精度对齐模式

在 `demo.toml` 中将 `deterministic` 设置为 `true` 即可开启精度对齐（确定性计算）功能：

```toml
[model]
# Set to true for precision alignment mode.
deterministic = true
```

开启后，Session会以确定性计算模式运行（算子精度模式为 `must_keep_origin_dtype`，并打开 `deterministic` 开关，
详见config.py中的sess_config函数），相同输入下训练过程中每一步的loss可保持一致，便于与GPU/CPU进行精度对齐。
