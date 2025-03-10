# dlrm 300IDUO 推理适配说明

## 迁移说明
1. 源码地址：[DLRM](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM)
2. 提交的SHA-1 hash值 729963dd47e7c8bd462ad10bfac7a7b0b604e6dd

## 启动容器
1. 镜像地址： https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f
2. 创建启动脚本如下：
```bash
docker run \
-u root \
-it \
--name ${container_name} \
--net=host \
--shm-size="300g" \
--privileged \
-v /etc/localtime:/etc/localtime \
-e ASCEND_VISIBLE_DEVICES=0-7 \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /home:/home \
-v /root/ascend:/root/ascend \
-v /root/.ssh:/root/.ssh \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
${image_name} \
/bin/bash
```

## python、pytorch、CANN版本
| 名称 | 版本 |
| ----- | ----- |
| python | 3.11.6 |
| pytorch | 2.1.0 |
| CANN | 8.0.0 |

## python 安装依赖
| 名称 | 版本 |
| ----- | ----- |
| [dllogger](https://github.com/NVIDIA/dllogger) | 1.0.0 |
| tqdm | 4.66.5 |
| torch_npu | 2.1.0.post10 |
| [Apex Patch](https://gitee.com/ascend/apex.git) | 0.1+ascend |

## 设置环境变量
`source /usr/local/Ascend/ascend-toolkit/set_env.sh`

## 适配
1. dlrm/cuda_ext/dot_based_interact.py  
删除19-22行
2. dlrm/cuda_ext/fused_gather_embedding.py   
删除23行，添加如下代码
```python
import torch_npu
from torch_npu.contrib import transfer_to_npu


def fusedEmbeddingNpu(embedding, categorical_inputs, offsets, amp_train):
    offsets = offsets[:-1]
    indices = categorical_inputs + offsets.unsqueeze(0)
    return embedding[indices][:]

if hasattr(torch, "npu"):
    fused_embedding_fwd = fusedEmbeddingNpu
else:
    from dlrm.cuda_ext import fused_embedding
    fused_embedding_fwd = fused_embedding.gather_gpu_fused_fwd
```
原31行替换成`output = fused_embedding_fwd(embedding, indices, offsets, amp_train)`

3. dlrm/cuda_ext/sparse_embedding.py  
删除19行，69行替换成如下
```python
if hasattr(torch, "npu"):
    def embedding_gather(weights: torch.Tensor, indices: torch.Tensor):
        embedding = weights.index_select(0, indices.flatten())
        return embedding.reshape(*indices.shape, -1)
else:
    embedding_gather = EmbeddingGatherFunction.apply
```
4. dlrm/model/distributed.py(分布式需要用到，若不使用分布式可忽略)  
64-68行替换成如下
```python
        gathered_outputs = [torch.empty(
            batch_sizes_per_gpu[rank], n, vector_dim,
            device=local_bottom_outputs.device, dtype=local_bottom_outputs.dtype) for n in vectors_per_gpu]

        torch.distributed.all_gather(gathered_outputs, local_bottom_outputs)

        bottom_output_buffer = []

        for i in range(ctx.world_size):
            split_tensors = list(gathered_outputs[i].split(batch_sizes_per_gpu, dim=0))
            bottom_output_buffer.append(split_tensors[rank])
```

5. dlrm/nn/factories.py  
删除22行的CppMlp  
添加import torch  
27行修改成`return TorchMlp(input_dim, sizes)`  
58-60替换成
```python
        if hasattr(torch, "npu"):
            return DotInteraction(embedding_num, embedding_dim)
        else:
            return CudaDotInteraction(
                DotInteraction(embedding_num, embedding_dim)
            )
```
6. /dlrm/nn/mlps.py  
删除18、23-44、117-140行
7. /dlrm/nn/parts.py  
删除59行
8. 添加文件/dlrm/scripts/caculate_loss_auc.py(不在推理阶段进行AUC和loss计算，单独脚本计算)  
内容详见caculate_loss_auc.py
9. dlrm/scripts/main.py
添加
```python
import re
import torch
import torch_npu
```
添加flag参数`flags.DEFINE_integer("total_devices", 1, "Total number of devices to run the test on")`
validate_flags函数添加如下判断
```python
    if "npu" in FLAGS.base_device:
        if FLAGS.amp:
            print('WARNING: Automatic mixed precision not supported on NPU')
            FLAGS.amp = False
```
原389行即main函数一开始，添加一行`torch.npu.set_compile_mode(jit_compile=False)`  
原468-471行修改成
```python
    if hasattr(torch, "npu"):
        if FLAGS.Adam_MLP_optimizer:
            mlp_optimizer = apex_optim.NpuFusedAdam(mlp_params)
        else:
            mlp_optimizer = apex_optim.NpuFusedSGD(mlp_params)
    else:
        if FLAGS.Adam_MLP_optimizer:
            mlp_optimizer = apex_optim.FusedAdam(mlp_params)
        else:
            mlp_optimizer = apex_optim.FusedSGD(mlp_params)
```
原508行修改成
```python
        device_id = re.findall(r'\d+', device)
        if device_id:
            device_id = int(device_id[0])
            total_devices = FLAGS.total_devices
        else:
            device_id = 0
            total_devices = 1
        auc, valid_loss = dist_evaluate(model, data_loader_test, device_id, total_devices)
```
原733行修改成`def dist_evaluate(model, data_loader, device_id=0, total_devices=1):`  
原755行修改成`    steps_per_epoch = len(data_loader)//total_devices + int(bool(len(data_loader) % total_devices > device_id))`  
原772-773行修改成
```python
        for step in range(len(data_loader)//total_devices + int(bool(len(data_loader) % total_devices > device_id))):
            try:
                for i in range(total_devices):
                    if i == device_id:
                        numerical_features, categorical_features, click = next(batch_iter)
                    else:
                        next(batch_iter)
            except StopIteration:
                break
```
原820-835行修改成
```python
        output_click = [y_true, y_score]

        if not os.path.exists('output'):
            os.makedirs('output')
        
        torch.save(output_click, f'output/output_click_{device_id}.pt')

        if world_size > 1:
            torch.distributed.barrier()

    model.train()

    return None, None
```
10. 添加文件dlrm/scripts/test_main.py(替代分布式，用多进程开启多个芯片跑数据)
文件内容详见test_main.py
11. dlrm/preproc/run_spark_cpu.sh  
将64行改成 export MASTER=local[1]   
这一步是在CPU使用spark处理数据时可能需要的步骤

12. dlrm/data/factories.py  
若启动在线推理且不准备train数据，则需要将105-112注释，且123行改成`return None dataset_test`

13. dlrm/scripts/generate_data_for_om.py  
创建新文件，文件内容详见generate_data_for_om.py  
该文件为生成离线推理数据脚本

14. dlrm/scripts/pth2onnx.py
新建该文件，文件内容详见pth2onnx.py

## 数据预处理（CPU处理）
需安装spark及相关依赖
| 名称 | 版本 |
| ----- | ----- |
| spark | 3.4.4-with-hadop3 |
| java | 1.8.0 |
| scala | 2.13.0 |
### 相关适配
1. bind.sh  
75行需要修改成npu数量  
98-101行需要根据lscpu数量决定，不能让num_gpus小于num_nodes  
启动命令
```bash
cd dlrm/preproc
bash prepare_dataset.sh 15 CPU Spark
```
预估需要一天时间处理数据
2. 需要设置spark/sbin文件夹路径到系统环境变量中  
3. 需要设置环境变量`export DGX_VERSION=DGX-2`  
4. 也可以基于原生提供的GPU处理方式处理数据

## 启动命令
### 在线推理
单芯片
```bash
cd /workspace/DeepLeaningExamples/PyTorch/Recommendation/DLRM
python3 dlrm/scripts/main.py --mode test --dataset /your/path/to/data --load_checkpoint_path /your/path/to/checkpoint --base_device npu --test_batch_size 65000
```
- --mode: 指定运行模式为测试模式。可选train/test/inference_benchmark，推理阶段选择test。
- --dataset: 数据集路径。
- --load_checkpoint_path: 加载checkpoint路径
- --base_device: 指定基础设备为NPU。可选gpu/npu/cpu。
- --test_batch_size: 测试批处理大小。
更多参数可参考原生main.py里的参数详解
多芯片
```bash
python3 dlrm/scripts/test_main.py --args "--mode test --dataset /your/path/to/data --load_checkpoint_path /your/path/to/checkpoint --base_device npu --test_batch_size 65000 --total_devices=2"
```
- --args: 参数字符串
- --mode: 指定运行模式为测试模式。可选train/test/inference_benchmark，推理阶段选择test。
- --dataset: 数据集路径。
- --load_checkpoint_path: 加载checkpoint路径
- --base_device: 指定基础设备为NPU。可选gpu/npu/cpu。
- --test_batch_size: 测试批处理大小。
- --total_devices: 使用的设备总数。
更多参数可参考原生main.py里的参数详解
#### auc计算
```bash
python3 dlrm/scripts/caculate_loss_auc.py --path /your/path/to/output_click.pt
```
- --path 在线推理输出的output_click.pt文件
### 离线推理
1. 先生成onnx文件
`python dlrm/scripts/pth2onnx.py --mode test --dataset /your/path/to/data --load_checkpoint_path /your/path/to/checkpoint --base_device npu --test_batch_size 65000`
- --mode: 指定运行模式为测试模式。可选train/test/inference_benchmark。
- --dataset: 数据集路径。
- --load_checkpoint_path: 加载checkpoint路径
- --base_device: 指定基础设备为NPU。可选GPU/NPU/CPU。
- --test_batch_size: 测试批处理大小。
更多参数可参考原生main.py里的参数详解
2. onnx转om
由于pytorch1.10之后的特性，onnx文件会被分割成多个文件，转om时只需要加载onnx单个文件即可，其他文件atc会自动加载
`atc --model output/dlrm.onnx --framework=5 --input_shape="numerical_features:65000,13;categorical_features:65000,26" --soc_version=Ascend310P3 --output=dlrm_output`
- --model 模型输出地址
- --framework 原始网络模型框架类型，5代表ONNX
- --input_shape 模型输入数据的shape
- --soc_version 模型转换时指定芯片版本
更多参数可参考： https://www.hiascend.com/document/detail/zh/canncommercial/800/devaids/devtools/atc/atlasatc_16_0039.html
3. 生成离线推理数据
`python dlrm/scripts/generate_data_for_om.py --mode test --dataset /your/path/to/data --base_device cpu --test_batch_size 65000 --output_dir ./`
- --mode: 指定运行模式为测试模式。可选train/test/inference_benchmark。
- --dataset: 数据集路径。
- --base_device: 指定基础设备为NPU。可选gpu/npu/cpu。
- --test_batch_size: 测试批处理大小。
- --output_dir: 输出文件夹地址，默认当前路径下
更多参数可参考原生main.py里的参数详解
4. 离线推理
`python -m ais_bench --model path/to/om --input "path/to/numerical_features,path/to/categorical_features" --output path/to/output`
- --model 模型地址
- --input generate_data_for_om.py生成的数据文件路径
- --output 推理输出文件地址
更多参数可参考：https://www.hiascend.com/document/detail/zh/Atlas%20300I%20Duo%20Inference%20Card/Atlas%20300I%20Duo%20Inference%20Card/apn_032.html
## note
- main.py的step time含义：单step消耗时间
