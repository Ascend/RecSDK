## perf.py
```
usage: perf.py [-h] --perf_data PERF_DATA --flamegraph_path FLAMEGRAPH_PATH
               [--perf_bin PERF_BIN] [--output_svg OUTPUT_SVG]

Generate a Flamegraph from perf.data.

optional arguments:
  -h, --help            show this help message and exit
  --perf_data PERF_DATA
                        Path to the perf.data file.
  --flamegraph_path FLAMEGRAPH_PATH
                        Path to the Flamegraph Perl scripts directory.
  --perf_bin PERF_BIN   Path to perf exacutable binary file. (default: perf)
  --output_svg OUTPUT_SVG
                        Path to the output SVG file. (default: flamegraph.svg)
```
#### 使用示例

参考以下脚本使用`perf`采集数据。
```bash
pid=$(top -b -n 1 | head -n 8 | tail -n 1 | awk '{print $1}')
if [ -z "$pid" ];then
    echo "未获取到进程ID"
    exit 1
fi
perf record -F 99 -p $pid -a -g -- sleep 60
if [ $? -ne 0 ]; then
    echo "perf record执行失败"
    exit 1
fi
echo "perf.data 采集完成"
```

使用本工具生成火焰图和耗时函数分析。
```bash
python perf.py --perf_data perf.data --flamegraph_path /ws/FlameGraph 
```
#### 可选配置
```toml
# config.toml

[perf]
# Filter percentage of time cost
threshold = 0.05
# Ignore function list
ignores = ["[libc.so.6]"]
```

## fusion_tracing.py
```
usage: fusion_tracing.py [-h] --debug_log DEBUG_LOG
                         [--msprof_output MSPROF_OUTPUT]

Generate CPU/NPU fusion tracing json.

optional arguments:
  -h, --help            show this help message and exit
  --debug_log DEBUG_LOG
                        MxRec DEBUG level log flie path.
  --msprof_output MSPROF_OUTPUT
                        msprof output path.
```
#### 使用示例
```bash
# only cpu
python fusion_tracing.py --debug_log ../../example/demo/little_demo/temp.log
# cpu + npu
python fusion_tracing.py --debug_log ../../example/demo/little_demo/temp.log --msprof_output ../../example/demo/little_demo/msprof
```
#### 可选配置
```toml
# config.toml

[mxrec]
# Pipe name and time cost name
key_process = ["getBatchData", "getAndProcess"]
process_emb_info = ["getAndSendTensors"]
lookup_swap_addr = ["lookupAddrs"]
embedding_recv = ["EmbeddingRecv", "EmbeddingUpdate", "SendH2DEmb"]
```
