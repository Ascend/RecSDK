# 昇腾适配运行

### 1. pytorch-lightning昇腾适配 


```
git apply npu.patch
```



### 2. hllm代码昇腾适配

2.1 下载源码

```
https://github.com/bytedance/HLLM
```

2.2 code/REC/trainer/trainer.py代码修改点：

去掉第32行

```python
self.gpu_available = torch.cuda.is_available() and config['use_gpu']
```

修改第324行

```python
修改前：
self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)

修改后：
self.lite = L.Fabric(accelerator='npu', strategy=strategy, precision=precision, num_nodes=nnodes)
```

修改第328行

```python
修改前：
self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)

修改后：
self.lite = L.Fabric(accelerator='npu', strategy=strategy, precision=precision, num_nodes=nnodes)
```

修改第457行

```python
修改前：
self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)

修改后：
self.lite = L.Fabric(accelerator='npu', strategy=strategy, precision=precision, num_nodes=nnodes)
```

修改第464行

```python
修改前：
self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)

修改后：
self.lite = L.Fabric(accelerator='npu', strategy=strategy, precision=precision, num_nodes=nnodes)
```

2.3  code/run.py代码修改点：

在import中增加内容：

```python
import torch_npu
```

修改第50行

```python
修改前：
device = torch.device("cuda", local_rank)

修改后：
device = f"npu:{local_rank}"
```

修改第136-137行

```python
修改前：
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')

修改后：
device = torch.device("npu:{}".format(local_rank))
torch.distributed.init_process_group(backend="hccl", rank=local_rank)  # 将通信方式设置为hccl
torch.npu.set_device(local_rank)

torch_npu.npu.set_compile_mode(jit_compile=False)
torch_npu.npu.matmul.allow_hf32 = True
torch_npu.npu.conv.allow_hf32 = True
```

