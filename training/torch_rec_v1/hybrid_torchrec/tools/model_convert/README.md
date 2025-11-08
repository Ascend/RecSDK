# NPU稀疏表保存数据转换工具说明
作用：将Rec SDK Torch框架NPU训练时保存的稀疏表数据加载到指定模型中，实现模型数据转换功能。

# 文件说明
```shell
model_convert_2_gpu.py
   |-- model_convert_2_gpu.py  # 转换脚本
   |-- test.py                 # 转换功能测试脚本
   |-- README.md               # 说明文档
```

# 使用示例

伪代码如下：

```python
import torch
import torchrec
from torchrec import EmbeddingBagCollection, EmbeddingBagConfig

from model_convert_2_gpu import ModelConverter

# == 创建模型 ==
ebc_configs = [EmbeddingBagConfig(num_embeddings=10000, embedding_dim=16, name="table0")]
ebc_model = EmbeddingBagCollection(ebc_configs)
optimizer_class = torch.optim.Adam
dmp_model = torchrec.distributed.DistributedModelParallel(ebc_model, "xxx")

# == 模型转换 ==
rank = 0  # 需获取到实际的rank id
path = "xxx"  # NPU 稀疏表数据保存路径，相对路径和绝对路径均可
model_converter = ModelConverter(rank)
# 调用load接口，会直接将NPU保存的稀疏表数据（embedding, optimizer）加载到dmp_model中
model_converter.load(dmp_model, path, optimizer_class)
```

# test.py脚本说明
## 使用方式说明
1. 先修改NPU纯显存模式的测试脚本[test_save_load.py](test/st/test_save_and_load.py)并执行（当前其main方法中已将参数改为和test.py脚本参数一致，可手动确认）：
   1. 将WORLD_SIZE改成1；
   2. 将params字典中`num_embeddings`、`embedding_dims`、`lookup_len`参数值改成和test.py脚本一样；
   3. 搜索`params["optim"] =`，将其值改为`[Adagrad]`；
   4. 将字典中params`ids_repeat_rate`的值修改为：`None`；(保证两边生成数据一致)
   5. 进入对应测试脚本目录，执行一次保存加载测试： `python3 test_save_load.py`；
2. 再将执行后生成的save_dir整个目录数据拷贝到当前目录；
3. 再运行test.py脚本： `python3 test.py`。运行后会输出对比结果，当结果不一致时会抛出异常。

## 测试逻辑说明
1. 脚本中使用的单卡，cpu，table_wise模式做的测试；(因为使用cpu设备时，不支持多进程场景下的row_wise分表模式)
2. golden: train模式执行100步，再用eval模式执行100步，并记录eval的100步查表结果；
3. load: 新创建模型，并加载数据到dmp模型中。切换到eval模式，执行200步，记录后100步的查表结果；
4. 对比两次查表结果数据是否一致；
