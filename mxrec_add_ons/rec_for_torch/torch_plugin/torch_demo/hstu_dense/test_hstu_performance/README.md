# 测试hstu npu, gpu性能对比
npu tag: https://gitee.com/ascend/RecSDK.git branch_v7.1.0-RC1<br>
gpu tag: https://github.com/NVIDIA/recsys-examples.git v25.05

## 1. 准备一台GPU, 一台NPU 服务器
## 2. GPU服务器下载编译 recsys-example

```
git clone https://github.com/NVIDIA/recsys-examples.git
cd recsys-examples/corelib/hstu
git checkout v25.05
make install
```
配置GPU服务器端的sftp server服务

## 3. 配置NFS共享路径

## 4. NPU算子编译
git clone https://gitee.com/ascend/RecSDK.git<br>
cd RecSDK/mxrec_add_ons/rec_for_torch/operators/<br>
去掉hstu_dense_forward/op_host/tiling_policy.cpp 函数GeneralShapeCheck 的校验，直接返回true.  编译算子安装<br>
去掉hstu_dense_backward/op_host/hstu_dense_backward_tiling_common.cpp 函数BasicShapeCheck 的校验，直接返回true.  编译算子安装<br>


## 5. 配置当前工程config.py 填写所有x的位置
```
# Absolute directory path of the recsys-example project recsys-examples/corelib/hstu (replace with actual path)
RECSYS_DIR="x"

# Network File System (NFS) mount directory path (replace with actual path)
NFS_DIR="x"

# IP address of the GPU server (replace with actual IP)
GPU_IP="x"

# Login username for the GPU server (replace with actual username)  
GPU_USER="x"

# Absolute path to Python3 interpreter (replace with actual path, e.g. /usr/bin/python3)
PYTHON3="x"
```
## 6. 上传test_npu_performance 到NPU 共享目录NFS_DIR

## 7. 登录NPU服务器，执行性能测试命令
输入参数在benchmark.csv, 结果保存在result.csv中。
```
cd test_npu_performance
pip install -r requirements.txt
python3 test_benchmark.py  # 需交互输入GPU服务器密码
```


### Benchmark文件头说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| index | int | 测试用例的唯一标识符 |
| shape_info | str | 张量形状描述信息 |
| total_len | int | 输入序列总长度 |
| batch_size | int | 批量大小 |
| heads | int | 注意力头数 |
| heads_rab | int | 使用RAB时的头数 |
| max_seq_len_q | int | 查询序列最大长度 |
| max_seq_len_k | int | 键序列最大长度 |
| max_context_len | int | 上下文序列最大长度 |
| max_target_len | int | 目标序列最大长度 |
| target_group_size | int | 目标序列分组大小 |
| attn_dim | int | 注意力机制维度 |
| hidden_dim | int | 隐藏层维度 |
| alpha | float | 缩放因子 |
| has_rab | bool | 是否使用RAB |
| has_drab | bool | 是否使用DRAB |
| window_size | int | 滑动窗口大小 |
| run_benchmark | bool | 是否运行基准测试 |
| dtype | str | 数据类型 |
| full_batch | bool | 是否使用完整批次 |
| is_delta_q | bool | 是否使用增量查询 |
| format | str | 数据格式 |
| npu_fw_time | float | NPU前向时间(ms) |
| npu_bw_time | float | NPU反向时间(ms) |
| gpu_fw_time | float | GPU前向时间(ms) |
| gpu_bw_time | float | GPU反向时间(ms) |
| precision | bool | 计算精度是否对其 |
| npu_fw/gpu_fw | float | NPU/GPU前向时间比 |
| npu_bw/gpu_bw | float | NPU/GPU反向时间比 |
| npu_fw+bw/gpu_fw+bw | float | NPU/GPU总时间比 |

### Result文件头说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| precision | bool | 计算精度是否对其 |
| npu_fw/gpu_fw | float | NPU/GPU前向时间比 |
| npu_bw/gpu_bw | float | NPU/GPU反向时间比 |
| npu_fw+bw/gpu_fw+bw | float | NPU/GPU总时间比 |
| npu_fw/benchmark | float | NPU/基准前向时间比 |
| npu_bw/benchmark | float | NPU/基准反向时间比 |
| npu_fw+bw/benchmark | float | NPU/基准总时间比 |

### 使用说明

1. **性能对比**：
   - 比值>1表示NPU更快
   - 比值<1表示GPU更快

2. **基准验证**：
   - 比值≈1表示符合预期
   - 显著偏离1需检查