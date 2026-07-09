# FAQ

## 容器使用

### 容器内执行git指令或Python脚本时报错 <a id="container_git_python_error"></a>

**问题现象**

创建容器后，容器内执行git指令或者Python脚本时，出现报错。

git执行时报错：`error, cannot create async thread: Operation not permitted`。

或者Python脚本执行时报错：`PyCapsule_Import could not import module "datatime"`。

**可能原因**

宿主机Docker版本较低时，和容器内OS存在兼容性问题，容器内无法访问系统底层指令，导致git/Python执行失败。

**解决方案**

- 升级宿主机Docker版本到20.10.10及以上。
- 或者启动Docker容器时，增加`--security-opt seccomp=unconfined`参数。修改启动Docker容器指令示例：修改前启动容器指令为`docker run xxx`，修改后为`docker run --security-opt seccomp=unconfined xxx`。

## 环境相关

### fbgemm_gpu安装错误导致报错

**问题现象**

运行Rec SDK Torch用例时报错“Extension modules: numpy.core._multiarray_umath, numpy.core._multiarray_tests, numpy.linalg._umath_linalg, numpy.linalg._umath_linalg xxx”。

**可能原因**

fbgemm_gpu软件包版本安装错误，导致和Rec SDK Torch中算子PTA层定义的ops schema不一致，在PyTorch加载ops时报错。

**解决方案**

fbgemm_gpu软件包为TorchRec框架依赖，需安装CPU版本，且需和PyTorch保持[配套版本](./recsdk_torch_installation_guide.md#section146113514599)一致。

可参考[容器内训练加速库依赖](./recsdk_torch_installation_guide.md#section146113514600)重新安装fbgemm_gpu软件包。

### 编译算子PTA层时报错

**问题现象**

编译算子PTA层代码时报错“Could NOT find Python3 (missing: Python3_EXECUTABLE Interpreter Development)”。

**可能原因**

容器环境中没有python3的可执行文件。

**解决方案**

使用Rec SDK Torch的[基础训练镜像](./recsdk_torch_installation_guide.md#section104919392501)创建容器再进行编译。

或者手动在环境内使用`which python`搜索Python可执行文件路径，并创建软链接。示例：假设环境中Python可执行文件路径为`/usr/bin/python`，则创建软链接指令为`ln -s /usr/bin/python /usr/bin/python3`。

## 模型训练

### 训练时embedding_bag反向算子fall back to CPU

**问题现象**

模型训练时，出现告警信息“The operator 'aten::_embedding_bag_backward' is not currently supported on the NPU backend and will fall back to run on the CPU。 This may have performance implications.”。表示embedding_bag反向算子没有NPU后端实现将回退到CPU执行，并会导致训练性能下降。

**可能原因**

直接原因是embedding_bag反向算子没有NPU后端实现。根因为使用Rec SDK Torch的Collection创建稀疏表后，训练时直接调用`model.forward(xx)`，未使用pipeline进行流水线查表，导致直接使用的TorchRec原生实现，而不是Rec SDK Torch的融合算子。

**解决方案**

使用torchrec.distributed import DistributedModelParallel对稀疏表进行分表，再使用pipeline进行流水线查表，此时会自动使用Rec SDK Torch的融合算子，不会走到embedding_bag反向算子。

详细实现可参考：[纯显存模式查表](./migration_and_training.md#basic_usage_device_memory)，[多级缓存模式查表](./migration_and_training.md#basic_usage_embcache)。

### permute算子执行时报错tensor device不在NPU上

**问题现象**

稀疏表查表时，执行到了permute算子，且因算子输入数据不在NPU device，导致算子执行报错。

**可能原因**

创建稀疏表（xxxCollection）时传入的feature_names列表，和模型前向输入的KeyedJaggedTensor的key列表不一致，导致走到了features.permute(xx)逻辑。

**解决方案**

保持feature_names列表和模型前向输入的KeyedJaggedTensor的key列表一致。
