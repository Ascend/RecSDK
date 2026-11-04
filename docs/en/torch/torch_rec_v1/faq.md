# FAQ

## Container Usage

### Error When Running Git Commands or Python Scripts in a Container <a id="container_git_python_error"></a>

**Symptom**

After you create a container, errors are reported when you run git commands or Python scripts in the container.

When you run git commands, the following error is reported: `error, cannot create async thread: Operation not permitted`.

When you run Python scripts, the following error is reported: `PyCapsule_Import could not import module "datetime"`.

**Possible Causes**

If the Docker version on the host is too low, compatibility issues occur with the OS in the container. The container cannot access the underlying system commands, which causes git or Python execution to fail.

**Solution**

- Upgrade the Docker version on the host to 20.10.10 or later.
- Alternatively, add the `--security-opt seccomp=unconfined` parameter when starting the Docker container. For example, change the container start command from `docker run xxx` to `docker run --security-opt seccomp=unconfined xxx`.

## Environment

### fbgemm_gpu Installation Error Causes a Failure

**Symptom**

When you run a Rec SDK Torch use case, the following error is reported: `Extension modules: numpy.core._multiarray_umath, numpy.core._multiarray_tests, numpy.linalg._umath_linalg, numpy.linalg._umath_linalg xxx`.

**Possible Causes**

The fbgemm_gpu package is installed with an incorrect version, which causes inconsistency with the ops schema defined in the operator PTA layer of Rec SDK Torch. Therefore, an error is reported when PyTorch loads the ops.

**Solution**

The fbgemm_gpu package is a dependency of the TorchRec framework. Install the CPU version and keep it consistent with the [matching version../02_torch_installation_guide/recsdk_torch_installation_guide.md#section146113514599 of PyTorch.

You can reinstall the fbgemm_gpu package by referring to [Container Training Acceleration Library Dependencies../02_torch_installation_guide/recsdk_torch_installation_guide.md#section146113514600.

### Error When Compiling the Operator PTA Layer

**Symptom**

When you compile the operator PTA layer code, the following error is reported: `Could NOT find Python3 (missing: Python3_EXECUTABLE Interpreter Development)`.

**Possible Causes**

The container environment does not have the `python3` executable file.

**Solution**

Use the [base training image../02_torch_installation_guide/recsdk_torch_installation_guide.md#section104919392501 of Rec SDK Torch to create a container, and then compile the code.

Alternatively, manually use the `which python` command in the environment to search for the Python executable file path, and then create a symbolic link. For example, if the Python executable file path in the environment is `/usr/bin/python`, create the symbolic link by running `ln -s /usr/bin/python /usr/bin/python3`.

## Model Training

### `embedding_bag` Backward Operator Falls Back to CPU During Training

**Symptom**

During model training, the warning `The operator 'aten::_embedding_bag_backward' is not currently supported on the NPU backend and will fall back to run on the CPU. This may have performance implications.` is displayed. This indicates that the `embedding_bag` backward operator has no NPU backend implementation and falls back to running on the CPU, which degrades training performance.

**Possible Causes**

The direct cause is that the `embedding_bag` backward operator has no NPU backend implementation. The root cause is that after you use the Collection of Rec SDK Torch to create a sparse table, you directly call `model.forward(xx)` during training instead of using the pipeline for table lookup. Therefore, the native TorchRec implementation is used directly instead of the fused operators of Rec SDK Torch.

**Solution**

Use `torchrec.distributed import DistributedModelParallel` to shard the sparse table, and then use the pipeline for table lookup. In this case, the fused operators of Rec SDK Torch are automatically used, and the `embedding_bag` backward operator is not invoked.

For detailed implementation, refer to [Table Lookup in Full-Device Memory Mode../04_migration_and_training/migration_and_training.md#basic_usage_device_memory and [Table Lookup in Multi-Level Cache Mode../04_migration_and_training/migration_and_training.md#basic_usage_embcache.

### permute Operator Execution Error: Tensor Device Is Not on the NPU

**Symptom**

During sparse table lookup, execution reaches the `permute` operator. Because the input data of the operator is not on the NPU device, an error is reported when the operator executes.

**Possible Causes**

The `feature_names` list passed when you create the sparse table (`xxxCollection`) is inconsistent with the key list of the `KeyedJaggedTensor` in the model forward input, which causes execution to enter the `features.permute(xx)` code path.

**Solution**

Keep the `feature_names` list consistent with the key list of the `KeyedJaggedTensor` in the model forward input.
