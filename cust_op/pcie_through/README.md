# pcie_through算子及样例说明

## 相关文件目录结构

若用户期望单独使用pcie_through算子，可参考以下目录中的实现：

```shell
├── emb_custom.json    # 算子原型配置
├── op_host    # pcie_through算子Host侧实现
├── op_kernel  # pcie_through算子Kernel侧实现
├── tf-test    # 单算子测试代码
├── README.md  # pcie_through算子说明文档
└── run.sh     # pcie_through算子安装脚本
```

## 使用约束

硬件：A2

软件：

* CANN 8.0及以上配套软件，需安装kernels包
* 安装Rec SDK 7.0及以上。或参考目录自行编译算子库、pybind库，参考单算子测试用例调用方式

技术限制：
* 对于Rec SDK，当前仅支持DDR模式
* 使用时确保显存足够，若中途有其他进程抢占卡会导致算子被阻塞，最终超时导致运行失败

## 使用方法（基于Rec SDK）

1. 上传pcie_through算子文件夹到目标环境，并进入当前目录，执行指令对pcie_through算子进行编译和部署

```shell
bash run.sh
```

注：需先在环境中设置CANN相关环境变量，再执行算子编译和安装指令。使用默认路径安装CANN时设置环境变量指令如下：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. 在init函数中传入use_shm_swap，代码示例：

```python
from mx_rec.util.initialize import init

# 详细使用指导请参考Rec SDK用户指南。
init(use_shm_swap=True, ...)
```

## 功能介绍

算子的主要功能是利用pcie_through, 在host和device交换数据量较大的场景，提升换入换出的性能。

算子依赖host侧处理（源码位于src/core/hd_transfer）。

### host处理共享队列

参考src/core/hd_transfer中的rma_shm_svm源码和mxRec::GetShmAddr接口，调用后将在host申请共享内存，并使用共享队列初始化这段内存，该共享内存将用于数据交换。

### RmaSwapMultiTables

#### 算子参数说明

输入：

* swap_in_index（int64）: 换入的key的索引。
* swap_out_index（int64）: 换出的key的索引。
* table_a ~ table_f（float）: 换入换出的表，目前设定最多6个。
* table_num（int64）: 换入换出的表数量。
* shm_swap_in（string）: 换入的共享内存地址。
* shm_swap_out（string）: 换出的共享内存地址。

返回：

* output（int64）：标志每个核上执行换入换出操作是否存在异常。

#### Host侧算子实现

Host侧算子实现在目录 pcie_through/op_host下，其中包括：rma_swap_multi_tables.cpp和rma_swap_multi_tables_tiling.h。

##### Tiling实现

namespace optiling域中的TilingFunc函数，主要实现从context中获取外部入参信息，校验有效性。

##### Shape推导

InferShape：

* output第0维表示每个核的状态标志。

InferDataType：输出类型int64。

##### 原型注册

namespace ops域中的RmaSwapMultiTables类定义了算子原型，并将算子注册到GE。

#### Kernel侧算子实现

Kernel侧算子实现在目录pcie_through/op_kernel下。

* 核函数的入口：extern "C" __global__ __aicore__ void rma_swap_multi_tables。

* 解析tiling参数：GET_TILING_DATA(tilingData, tiling)从TilingData中获取host侧传入的数据。

* Init方法：进行算子运行数据的初始化；按生产者、消费者对core进行分组。

* Process方法：进行数据搬入、同步、搬出，并且计算完成后将计算结果数据分别更新到对应入参中。

## 单算子参考设计


### 前置条件

1. 完成算子的编译部署，可参考以下两种方式：

   * 使用本目录下的run.sh直接编译安装。

   * [算子编译部署](https://www.hiascend.com/document/detail/zh/canncommercial/800/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0068.html)。

2. 编译并安装Rec SDK。

3. 生成hccl路由表，参考[Atlas A2 中心推理和训练硬件 24.1.RC3 HCCN Tool 接口参考 03 - 华为](https://support.huawei.com/enterprise/zh/doc/EDOC1100422625/284725ac?idPath=23710424|251366513|254884019|261408772|252764743)。

### RmaSwapMultiTables算子的调用实现

核心代码如下：

```python
# 使用get_peer_mem接口申请共享内存并初始化共享队列
import mxrec_pybind
peer_mem = mxrec_pybind.get_shm_mem(d2h_name_id, device_id, capacity)

# 加载算子库，用于调用RmaSwapMultiTables算子
import tensorflow as tf
ops_so = tf.load_op_library("/usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec/libasc/libasc_ops.so")

# 构建单算子模型
class WideDeep:
    def __init__(self, input_data):
        self.lbl_hldr = input_data
        self.forward()

    def forward(self):
        with tf.control_dependencies([self.lbl_hldr]):
            result_ = ops_so.rma_swap_multi_tables(...)
        return result_
```

### 运行脚本

以8卡测试example/DCNv2为例：

```shell
设置环境变量export USE_SHM_SWAP=1
设置环境变量export HUGE_TLB_ENABLE=1
./run.sh
```
主要日志：
```shell
# 创建pcie_through数据交换通道
Device x start alloc shm for xxxx
