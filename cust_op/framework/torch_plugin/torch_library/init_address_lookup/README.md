# init_address_lookup 算子适配层

## 功能说明

该目录包含 `init_address_lookup` 算子的 PyTorch 适配层实现，用于在 PyTorch 框架下调用 Ascend NPU 上的自定义算子。

## 目录结构

```
init_address_lookup/
├── CMakeLists.txt           # CMake 构建配置
├── build_ops.sh             # 编译脚本
├── init_address_lookup.cpp  # 算子适配层实现
└── README.md                # 说明文档
```

## 编译方法

```bash
# 确保已设置 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 编译算子适配层
bash build_ops.sh
```

算子完整编译步骤请参考[RecSDK\cust_op\README.md](../../../../README.md)。

编译成功后，会在 `build/` 目录下生成 `libinit_address_lookup.so` 文件。

## 使用方法

完成所有编译后，调用方式：

```python
torch.ops.mxrec.init_address_lookup(self.address_lookup, self.buffer_offsets, self.emb_sizes)
```

具体示例：

```python
import sysconfig
import torch
import torch_npu
import fbgemm_gpu

device = "npu:0"

# 多算子编译
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
# 单算子编译
# torch.ops.load_library("build/libinit_address_lookup.so")

# 2 张嵌入表：
#   表 0 的 buffer 大小为 5，逻辑行数（emb_size）为 3
#   表 1 的 buffer 大小为 4，逻辑行数（emb_size）为 4
buffer_offsets = torch.tensor([0, 5, 9], dtype=torch.int64, device=device)
emb_sizes      = torch.tensor([3, 4],   dtype=torch.int64, device=device)
total_rows     = int(buffer_offsets[-1])  # 9

address_lookup = torch.empty(total_rows, dtype=torch.int64, device=device)
# also can use torch.ops.fbgemm
torch.ops.mxrec.init_address_lookup(address_lookup, buffer_offsets, emb_sizes)
# result: [0, 1, 2, 0, 0, 0, 1, 2, 3]
print(address_lookup.cpu())
```


> **提示**  
> 上述用例为通用场景执行，更详细精度、多场景测试用例，请参考完整测试文件：  
> [`RecSDK/cust_op/test/init_address_lookup_test/torch/test_init_address_lookup.py`](../../../../test/init_address_lookup_test/torch/test_init_address_lookup.py)

## 依赖

- PyTorch 2.1+
- torch_npu
- CANN toolkit

本示例在 PyTorch 2.7.1 + Python 3.11.0 + CANN 9.0 环境通过验证。

## 注意事项

1. 使用前需要先编译并部署底层 Ascend C 算子（参见 `cust_op/ascendc_op/ai_core_op/init_address_lookup/c310/`）
2. 确保 NPU 设备可用且环境变量配置正确
3. `buffer_offsets` 的大小应为 `emb_sizes` 大小 + 1
4. `address_lookup` 的数据类型应与 `emb_sizes` 相同
5. 为避免溢出问题，不建议使用 int32 类型的 `emb_sizes` 。后续算子可能只接受 int64 类型。