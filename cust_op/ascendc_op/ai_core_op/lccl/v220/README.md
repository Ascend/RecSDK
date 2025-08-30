# LCCL通信优化算子及样例说明

名词术语：

LCCL（Low Latency Collective Communication Library）

LCAL（Low Latency Collective Acceleration Library）

## 相关文件目录结构

若用户期望单独使用LCCL算子，可参考以下目录中的实现：

```shell

├──cust_op
├────lccl
├────── tf_test              # 单算子测试用例
├────── emb_custom.json      # 算子原型配置
├────── op_host              # LCCL算子Host侧实现
├────── op_kernel            # LCCL算子Kernel侧实现
├────── README.md            # LCCL算子说明文档
├────── run.sh               # LCCL算子安装脚本
├──src
├────lccl                    # 共享内存申请、元信息同步库
├────── include
├────── src
├────── CMakeLists.txt
├────ops_tf                  # tf算子注册
├────── tf_ops.h
├────── hybrid_dataset_ops.cpp
├────── CMakeLists.txt
├────pybind                  # 共享内存申请的python接口绑定（GetPeerMem接口）
├────── CMakeLists.txt
└────── module_main.cpp
```

## 使用约束

硬件：A2单机

软件：

* CANN 8.0及以上配套软件，需安装kernels包
* 安装Rec SDK 7.0及以上。或参考目录自行编译算子库、pybind库，参考单算子测试用例调用方式

技术限制：

* 对于Rec SDK，当前仅支持片上内存非扩容模式
* AllToAll通信量小于2GB
* 使用时确保卡独占，若中途有其他进程抢占卡会导致算子被阻塞，最终超时导致运行失败

## 环境变量配置

### LCCL_DETERMINISTIC

用于控制AllUss算子是否使用确定性计算模式。

- **取值范围**: 0 或 1
- **默认值**: 0（不开启确定性计算）
- **说明**:
  - `0`: 不开启确定性计算模式，使用标准AllUss算法
  - `1`: 开启确定性计算模式，使用AllUssDeterministic算法
  - `other value`: 其他无效值，默认不开启确定性计算
- **设置方法**:
  ```bash
  export LCCL_DETERMINISTIC=1  # 开启确定性计算
  export LCCL_DETERMINISTIC=0  # 关闭确定性计算（默认）
  ```

## 使用方法（基于Rec SDK）

1. 上传lccl文件夹到目标环境，并进入当前目录，执行指令对LCCL算子进行编译和部署

```shell
bash run.sh
```

注：需先在环境中设置CANN相关环境变量，再执行算子编译和安装指令。使用默认路径安装CANN时设置环境变量指令如下：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. 在init接口添加参数use_lccl并设置为True。代码示例：

```python
from mx_rec.util.initialize import init

# 详细使用指导请参考Rec SDK用户指南。
init(use_lccl=True, ...)
```

## 功能介绍

算子的主要功能是利用aicore mte直接访问对端片上内存的能力，使用内存语义进行集合通信，大幅减少传输启动开销，使用内存同步，提升性能。

算子依赖LCAL库申请片上共享内存（源码位于src/lccl）。

### 申请片上共享内存

参考src/lccl中的LCAL源码和mxRec::GetPeerMem接口，各进程基于acl接口申请片上共享内存后，通过训练任务的主进程进行元信息交换，各进程获取所有rank的共享内存地址。

在进行通信时，需给算子传入共享内存地址。

### AllToAll

#### 算子参数说明

输入：

* send_data（float）: 输入的数据
* send_count_matrix（int64）: 通信矩阵
* shape_vec（int）：数据个数
* peer_mem（int64）: 共享内存地址
* rank（int）: 当前进程的rank id
* rank_size（int）: 总通信卡数
* dim（int）：单条数据的长度

返回：

* recv_data（float）：输出数据

#### Host侧算子实现

Host侧算子实现在目录 lccl/op_host下，其中包括：lccl_all_to_all.cpp和lccl_all_to_all_tiling.h。

##### Tiling实现

namespace optiling域中的TilingFunc函数，主要实现从context中获取外部入参信息，校验有效性。

##### Shape推导

InferShape：

* output第0维表示数据条数，由shape_vec提供。
* output第1维表示单条数据长度，由send_data第1维提供。
* output第2维用于占位，无其他含义。

InferDataType：输出类型同send_data，为浮点数。

##### 原型注册

namespace ops域中的LcclAllToAll类定义了算子原型，并将算子注册到GE。

#### Kernel侧算子实现

Kernel侧算子实现在目录lccl/op_kernel下，其中包括：all2all.h、lccl_all_to_all.cpp。

* 核函数的入口：extern "C" __global__ __aicore__ void lccl_all_to_all

* 解析tiling参数：GET_TILING_DATA(tilingData, tiling)从TilingData中获取host侧传入的数据

* Init方法：进行算子运行数据的初始化；按生产者、消费者对core进行分组

* Process方法：进行数据搬入、同步、搬出，并且计算完成后将计算结果数据分别更新到对应入参中

### AllUss

#### 算子参数说明

输入：

* send_data（float）: 输入的数据
* send_count_matrix（int64）: 通信矩阵
* shape_vec（int）：数据个数
* peer_mem（int64）: 共享内存地址
* restore（int）：用于USS的索引向量
* rank（int）: 当前进程的rank id
* rank_size（int）: 总通信卡数
* dim（int）：单条数据的长度

返回：

* recv_data（float）：输出数据

#### Host侧算子实现

Host侧算子实现在目录 lccl/op_host下，其中包括：lccl_all_uss.cpp和lccl_all_uss_tiling.h。

##### Tiling实现

namespace optiling域中的TilingFunc函数，主要实现从context中获取外部入参信息，校验有效性。

##### Shape推导

InferShape：

* output第0维表示数据条数，由shape_vec提供。
* output第1维表示单条数据长度，由send_data第1维提供。
* output第2维用于占位，无其他含义。

InferDataType：输出类型同send_data，都是浮点数。

##### 原型注册

namespace ops域中的LcclAllUss类定义了算子原型，并将算子注册到GE。

#### Kernel侧算子实现

Kernel侧算子实现在目录lccl/op_kernel下，其中包括：all2all_uss.h、all_uss_deterministic.h、lccl_all_uss.cpp。

* 核函数的入口：extern "C" __global__ __aicore__ void lccl_all_uss，根据环境变量LCCL_DETERMINISTIC决定使用AllUss或AllUssDeterministic。
* 解析tiling参数：GET_TILING_DATA(tilingData, tiling)从TilingData中获取host侧传入的数据。
* Init方法：进行算子运行数据的初始化；按生产者、消费者对core进行分组
* Process方法：进行数据搬入、同步、计算、搬出，并且计算完成后将计算结果数据分别更新到对应入参中。

### GatherAll

#### 算子参数说明

输入：

* emb_table（float）: 稀疏表
* lookup（int64）：查表的索引
* send_count_matrix（int64）: 通信矩阵
* shape_vec（int）：数据个数
* peer_mem（int64）: 共享内存地址
* rank（int）: 当前进程的rank id
* rank_size（int）: 总通信卡数
* dim（int）：单条数据的长度

返回：

* recv_data（float）：输出数据

#### Host侧算子实现

Host侧算子实现在目录 lccl/op_host下，其中包括：lccl_gather_all.cpp和lccl_gather_all_tiling.h。

##### Tiling实现

namespace optiling域中的TilingFunc函数，主要实现从context中获取外部入参信息，校验有效性。

##### Shape推导

InferShape：

* output第0维表示数据条数，由shape_vec提供。
* output第1维表示单条数据长度，由emb_table第1维提供。
* output第2维用于占位，无其他含义。

InferDataType：输出类型同emb_table，都是浮点数。

##### 原型注册

namespace ops域中的LcclGatherAll类定义了算子原型，并将算子注册到GE。

#### Kernel侧算子实现

Kernel侧算子实现在目录lccl/op_kernel下，其中包括：gather_all.h、lccl_gather_all.cpp。

* 核函数的入口：extern "C" __global__ __aicore__ void lccl_gather_all
* 解析tiling参数：GET_TILING_DATA(tilingData, tiling)从TilingData中获取host侧传入的数据
* Init方法：进行算子运行数据的初始化；按生产者、消费者对core进行分组
* Process方法：进行数据搬入、索引、同步、搬出，并且计算完成后将计算结果数据分别更新到对应入参中

## 单算子测试参考设计


### 前置条件

1. 完成算子的编译部署，可参考以下两种方式：

   * 使用本目录下的run.sh直接编译安装

   * [算子编译部署](https://www.hiascend.com/document/detail/zh/canncommercial/800/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0068.html)

2. 编译并安装Rec SDK

3. 生成hccl路由表，参考[Atlas A2 中心推理和训练硬件 24.1.RC3 HCCN Tool 接口参考 03 - 华为](https://support.huawei.com/enterprise/zh/doc/EDOC1100422625/284725ac?idPath=23710424|251366513|254884019|261408772|252764743)

### LCCL算子的调用实现

核心代码如下，以AllToAll算子为例，详细代码参见cust_op/test/lccl_op_test/tf/all2all.py：

```python
# 使用get_peer_mem接口申请共享内存
import mxrec_pybind
peer_mem = mxrec_pybind.get_peer_mem(rank_id, comm_server_rank_id, rank_size)

# 加载算子库，用于调用lccl算子
import tensorflow as tf
ops_so = tf.load_op_library("/usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec/libasc/librecsdk_tf_npu_ops.so")

# 构建单算子模型
class WideDeep:
    def __init__(self, input_data, matrix):
        self.lbl_hldr = input_data
        self.matrix = matrix
        self.forward()

    def forward(self):
        with tf.control_dependencies([self.lbl_hldr]):
            all2all_result_ = ops_so.lccl_all_to_all(...)
            self.all2all_result = tf.reshape(all2all_result_, [-1, dim])
        return self.all2all_result
```

详细实践请参考cust_op/test/lccl_op_test/tf目录下的文件。

### 运行脚本

以8卡测试AllToAll算子为例，执行：

```shell
./run.sh 8 all2all.py
```
主要日志：
```shell
# 表示当前已完成的执行步数和耗时
current steps: ..., time cost(ms): ...

# 精度测试通过
all2all precision test pass
```
如果期望观察性能，可以调整`stop_steps`参数的值，观察日志中的`time cost`。
