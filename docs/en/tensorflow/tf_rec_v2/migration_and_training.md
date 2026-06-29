# Migration and Training

## Training Scenarios

**Introduction to Training Scenarios**

Rec SDK TensorFlow provides the `tf.Session` training scenario (currently, the Estimator training scenario is not supported).

- `tf.Session` training scenario: Start model running by creating a new `Session` instance, which returns tensor examples for customized model training.

>[!NOTE]
>
> - Rec SDK TensorFlow does not support Keras currently.
> - Rec SDK TensorFlow currently supports only the migration of model training scripts using TensorFlow native APIs. It does not support third-party frameworks (such as tf_adapter, HugeCTR, and DeepRec).
> - When large and small loops are enabled, the total number of training iterations must be an integer multiple of the small loop (that is, `iterations_per_loop`).

**Mapping Between TensorFlow and Rec SDK TensorFlow APIs**

During model migration, determine whether the sparse table APIs are used based on the actual model code and context. If a TensorFlow API is related to sparse tables, change it to the corresponding Rec SDK TensorFlow API as shown in [Table 1](#table16435142101913).

**Table 1** API mapping
<a id="table16435142101913"></a>

|TensorFlow API| Rec SDK TensorFlow API|Description|
|--|----------------------|--|
|<li>MutableHashTable</li><li>tf.Variable</li>| get_embedding_table  |Creates a sparse table.|
|<li>tf.embedding_lookup</li><li>mutable_hash_table.lookup (where mutable_hash_table is an instance of MutableHashTable), and so on</li>| embedding_lookup     |Queries a sparse table.|

API examples:

- TensorFlow example

    ```python
    import tensorflow as tf
    from tensorflow.contrib.lookup import MutableHashTable
    # .......
    user_id = features["user_ids"]
    user_emb_table = MutableHashTable(key_dtype=tf.int64, value_dtype=tf.float32, default_value=0.0)
    user_emb = user_emb_table.lookup(user_id)
    ```

- Rec SDK TensorFlow example

    ```python
    import tensorflow as tf
    import mxrec
    # .......
    user_id = features["user_ids"]
    user_emb_table = mxrec.get_embedding_table(
        name="user_table",
        dimension=8,
        device_vocabulary_size=10000,
        key_dtype=tf.int64,
        value_dtype=tf.float32,
    )
    user_emb = mxrec.embedding_lookup(user_emb_table, user_id)
    ```

## `sess.run` Migration and Training

### `sess.run` Migration

If the original TensorFlow network is constructed based on the `sess.run` API, refer to this section for the manual migration process.

You are advised to use the model training samples provided by Rec SDK TensorFlow for adaptation. If you use open-source recommendation projects, direct API migration may cause compatibility issues.

**`sess.run` Overview**

As a low-level API of TensorFlow, `sess.run` appears more flexible than Estimator. On the flip side, using it for model implementation could be complex.

The process of developing a training script using `sess.run` is as follows:

1. [Data preprocessing](#section3602537142311)
2. [Model construction, loss calculation, and gradient update](#section177071912713)
3. [Creating a session and initializing resources](#section19387465248)
4. [Performing training](#section20549115411413)

The following describes how to migrate a `sess.run` training script for training on Ascend AI Processors.

**Adding Header Files**

For Python files modified in the following steps, add the following header file references to import NPU-related libraries.

```python
from npu_bridge.npu_init import *
```

>[!NOTE]
>After you import the preceding header file, the training script runs on the Ascend AI Processor by default.

**Data Preprocessing<a id="section3602537142311"></a>**

Generally, this part of code does not need to be modified. Adapt the code in the following scenario:

If the original network script uses `dataset.batch(batch_size)` to return a dynamic shape, the number of remaining samples in the data stream may be smaller than the batch size. This makes the shape of the last step inconsistent with preceding shapes, triggering the dynamic shape compilation process. To improve compilation performance, set `drop_remainder` to `True` to discard the last few samples in the file and ensure consistent shapes for every step in the network.

```python
  dataset = dataset.batch(batch_size, drop_remainder=True)
```

Note: During inference, if the amount of inference data in the last iteration is smaller than the batch size, you must pad the data to the batch size. This is because some scripts include an assertion at the end to verify that the number of results matches the number of validation data.

```python
 assert num_written_lines == num_actual_predict_examples
```

**Model Construction, Loss Calculation, and Gradient Update<a id="section177071912713"></a>**

Generally, this part of code does not need to be modified. Adapt the code in the following scenario:

- For `dropout` in the original network, replace it with the corresponding CANN API implementation for better performance, while paying attention to the impact on network accuracy.
    - If `tf.nn.dropout` exists, change it to:

        ```python
        layers = npu_ops.dropout()
        ```

    - If `tf.layers.dropout`, `tf.layers.Dropout`, `tf.keras.layers.Dropout`, `tf.keras.layers.SpatialDropout1D`, `tf.keras.layers.SpatialDropout2D`, or `tf.keras.layers.SpatialDropout3D` exists, add the following header file reference:

        ```python
        from npu_bridge.estimator.npu import npu_convert_dropout
        ```

- For `gelu` in the original network, replace it with the corresponding CANN API implementation for better performance.

    Original TensorFlow code:

    ```python
    def gelu(x):
      cdf = 0.5 * (1.0 + tf.tanh(
         (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
      return x*cdf
    layers = gelu()
    ```

    Migrated code:

    ```python
    layers = npu_unary_ops.gelu(x)
    ```

**Creating a Session and Initializing Resources<a id="section19387465248"></a>**

When you run a training script on the Ascend AI Processor in `sess.run` mode, note the following configurations:

- The following configuration is disabled by default and should remain disabled:

    rewrite_options.disable_model_pruning

- The following configurations are enabled by default and should remain enabled:
    - `rewrite_options.function_optimization`
    - `rewrite_options.constant_folding`
    - `rewrite_options.shape_optimization`
    - `rewrite_options.arithmetic_optimization`
    - `rewrite_options.loop_optimization`
    - `rewrite_options.dependency_optimization`
    - `rewrite_options.layout_optimizer`

- The following configuration is enabled by default and should be disabled explicitly:
  - `rewrite_options.remapping`
  - `rewrite_options.memory_optimization`

- If code related to `tf.device` is used in the original network, add the session configuration `allow_soft_placement=True` to allow TensorFlow to automatically allocate devices.

Original TensorFlow code:

```python
#Construct an iterator.
iterator=Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

#Obtain the batch data.
next_batch=iterator.get_next()

#Initialize the iterator.
training_init_op=iterator.make_initializer(train_dataset)

#Initialize the variables.
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

#Get the number of training/validation steps per epoch
train_batches_per_epoch=int(np.floor(train_size/batch_size))
```

Migrated code:

```python
#Construct an iterator.
iterator=Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

#Obtain the batch data.
next_batch=iterator.get_next()

#Initialize the iterator.
training_init_op=iterator.make_initializer(train_dataset)

#Initialize the variables.
init=tf.global_variables_initializer()

#Create a session. If code related to tf.device is used in the original network, add the session configuration "allow_soft_placement=True" to allow TensorFlow to automatically allocate devices.
config = tf.ConfigProto(allow_soft_placement=True)
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
# You must explicitly disable the remapping and memory_optimization functions of TensorFlow to avoid conflicts with functions in the NPU.
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # Explicitly disable.
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # Explicitly disable.
sess = tf.Session(config=config)
sess.run(init)

#Get the number of training/validation steps per epoch
train_batches_per_epoch=int(np.floor(train_size/batch_size))
```

All native `tf.Session` functions are supported on the Ascend platform.

Additionally, the Ascend platform supports functions such as automatic mixed precision. To enable these functions, see "Session Configuration" in the *TF Adapter API (1.x)*.

**Performing Training<a id="section20549115411413"></a>**

This part of the code does not need modification. For example:

```python
#Start loop iterations.
for epoch in range(num_epochs):
  ##Initialize the iterator with the training dataset.
  sess.run(training_init_op)
  for step in range(train_batches_per_epoch):
    #Get the next batch of data.
    img_batch,label_batch=sess.run(next_batch)
    #Run the training operation.
    _,train_loss = sess.run([train_op, loss],feed_dict={x:img_batch, y_:label_batch, is_training:True})
```

After the session object created by `tf.Session` is used, you must explicitly call `session.close()` or use the with statement to create the session (which calls `close()` automatically through the context). See the following examples:

Example 1: Explicitly call `sess.close()`.

```python
sess = tf.Session(config=config)
sess.run(...)
sess.close()
```

Example 2: Use `with` to create a session.

```python
with tf.Session(config=config) as sess:
    sess.run(...)
```

>[!NOTE]
>If an error occurs during migration and training, see the [FAQ](faq.md) or contact technical support.

## Distributed Training Script Migration

### Data Parallelism Support (AllReduce)

AllReduce is a mainstream data parallelism architecture where each node works collaboratively based on algorithms. It is suitable for scenarios with high training power requirements and large scale of devices. This section describes how to perform distributed training of a TensorFlow training script on the Ascend AI Processor using the AllReduce architecture.

**Implementation Principle of AllReduce**

In large-scale AI training clusters, data parallelism is usually used for training. In data parallelism, each device uses the same model but different training samples. Gradient data calculated by each device must be aggregated before parameter updates.

**Figure 1** Data parallelism training
![](../../figures/tf_rec_v1/data-parallelism-training.png "Data parallelism training")

Mainstream implementations of data parallelism include the **parameter server-worker (PS-worker) architecture** and the **AllReduce architecture** based on gradient aggregation methods. In the **AllReduce architecture**, each device participating in training forms a ring, and there is no central node to aggregate all calculated gradients. The AllReduce algorithm places participating devices in a logical ring. Each device receives data from the upstream device and sends data to the downstream device, fully utilizing the upstream and downstream bandwidth of each device.

The AllReduce architecture is an improved architecture proposed to solve the linear scaling issue of the PS-worker architecture. Each node works collaboratively according to an algorithm designed to reduce the volume of transmitted data and make full use of hardware communication bandwidth. It is generally suitable for scenarios with high training power requirements and large scale of devices. The following figure shows the implementation of the AllReduce architecture.

**Figure 2** AllReduce mode<a id="fig1321114115499"></a>
![](../../figures/tf_rec_v1/all-reduce-mode.png "AllReduce mode")

The Ring algorithm (called Ring-AllReduce) is used to introduce the Allreduce mode. As shown in [Figure 2](#fig1321114115499), in the Ring-AllReduce architecture, each device is a worker and forms a ring without a central node to aggregate gradients calculated by all workers. During an iteration, each worker completes the forward calculation and backward calculation of a mini-batch sample data to obtain gradient data, and then uses the Ring-AllReduce algorithm to synchronize gradient data. The Ring-AllReduce algorithm includes Scatter-Reduce and AllGather. Gradient data is passed to the next worker in the ring in multiple steps, while the worker also receives gradient data from the preceding worker multiple times. For a ring containing *N* workers, each worker needs to receive gradient data 2 × (*N* - 1) times from other workers (1/*N* of the data each time) and send gradient data 2 × (*N* - 1) times to other nodes (1/*N* of the data each time).

**APIs Used**

In TensorFlow, `tf.distribute.Strategy` is generally used for distributed training. For details, see [this link](https://www.tensorflow.org/guide/distributed_training). However, Ascend AI Processors do not support the aforementioned distributed strategy. TF Adapter provides the distributed API `npu_distributed_optimizer_wrapper` to add the AllReduce operation of the NPU to the passed optimizer gradient function, and finally returns the input optimizer. This supports gradient aggregation between devices in networking modes such as single-node multi-card and multi-node multi-card. After you call this function, AllReduce operator nodes are inserted between gradient calculation and update operators in the generated training graph.

**Figure 3** APIs used
![](../../figures/tf_rec_v1/apis-used.png "APIs used")

Therefore, the original TensorFlow training script must be modified to support distributed training on the Ascend AI Processor.

**Dataset Sharding**

During distributed training, you can use the TensorFlow API for dataset sharding. If processor resource information is required for dataset sharding, obtain the number of Ascend AI Processors through the collective communication API `get_rank_size` and obtain the processor ID through `get_rank_id`. For example:

```bash
  dataset = dataset.shard(get_rank_size(),get_rank_id())
```

**Script Migration in `sess.run` Mode**

You must manually implement the broadcast function in the `sess.run` training script. The method is as follows:

1. After variable initialization and before training, broadcast variables through the collective communication API broadcast. For details about the broadcast API, see the *HCCL APIs*.

    ```python
    from npu_bridge.npu_init import *

    def broadcast_global_variables(root_rank, index):
        """Broadcasts all global variables from root rank to all other processes.
        Arguments:
        root_rank: rank of the process from which global variables will be broadcasted
        to all other processes.
        index: rank_id
        """
        op_list = []
        for var in tf.trainable_variables():
            # the input and out tensor of HCOMBroadcast interface are list
            if "float" in var.dtype.name:
                inputs = [var]
                outputs=hccl_ops.broadcast(tensor=inputs,root_rank=root_rank)
            if outputs is not None:
                op_list.append(outputs[0].op)
                op_list.append(tf.assign(var, outputs[0]))

        return tf.group(op_list)

    ...
    bcast_op = broadcast_global_variables(root_rank, index)
    sess = tf.Session()
    ...
    sess.run(bcast_op)
    ```

    In addition, the broadcast API includes a graph modification operation. If the graph cannot be modified (for example, the graph is finalized or a session is created using `tf.train.Supervisor`), you need to unfreeze the graph first:

    ```python
    with sv.managed_session() as sess:
      sess.graph._unsafe_unfinalize() # Unfreeze the graph.
      sess.run(bcast_op)
    ```

2. During training, call `npu_distributed_optimizer_wrapper` to aggregate gradient data after calculating the data for each device using the gradient optimizer:

    ```python
    from npu_bridge.npu_init import *
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001) # Use the SGD optimizer.
    distributedOptimizer=npu_distributed_optimizer_wrapper(optimizer) # Use NPU distributed computing to update gradients.
    ```

    >[!NOTE]
    >The NPUDistributedOptimizer distributed optimizer is compatible in the current version.

    If the original script uses the TensorFlow API to calculate gradients, such as `grads = tf.gradients(loss, tvars)`, you need to call the `npu_AllReduce` API to perform AllReduce on the gradients after calculation.

    ```bash
    grads = npu_allreduce(tf.gradients(a + b, [a, b], stop_gradients=[a, b]))
    ```

### Data Parallelism Support (PS-Worker)

In recommendation networks, feature data is saved in an embedding table. The data volume can reach the terabyte TB level and cannot be saved on the device side. Therefore, data must be saved in the host memory through the PS-worker mode. This section describes how to perform distributed training of a TensorFlow training script on the Ascend AI Processor using the PS-worker architecture.

**Implementation Principle of PS-Worker**

**Figure 1** PS-worker mode
![](../../figures/tf_rec_v1/ps-worker-mode.png "PS-worker mode")

In the PS-worker architecture, nodes in a cluster are divided into two types: parameter servers and workers. Parameter servers store model parameters, and workers are responsible for calculating parameter gradients. In each iteration, workers obtain parameters from the parameter servers and return calculated gradients to the parameter servers. The parameter servers aggregate the gradients returned from the workers, update the parameters, and broadcast the new parameters to the workers.

The following describes how to perform distributed training on the Ascend AI Processor through the PS-worker architecture using a training script developed based on the TensorFlow Python API.

**Configuring Cluster Information**

> [!NOTICE]NOTICE
>
> - Distributed training through the PS-worker architecture on the Ascend AI Processor currently supports only the NPUEstimator mode.
> - Currently, only one worker process is supported on one device.
> - In the PS-worker cluster scenario, you are advised to use high-speed NICs.
>

In the PS-worker architecture, cluster information is configured through the environment variable `TF_CONFIG`. `TF_CONFIG` consists of two parts: `cluster` and `task`. `cluster` provides information about the entire cluster, that is, workers and parameter servers in the cluster. `task` provides information about the current task. For details, see the [TensorFlow official website](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_estimator).

The following uses two servers, each with one parameter server and eight workers, as an example.

1. Set `TF_CONFIG` information.

    ```python
    os.environ['TF_CONFIG'] = json.dumps({
            'cluster': {
                ##'chief':chief_hosts, # This parameter is optional.
                'worker': worker_hosts,
                'ps': ps_hosts,
                'evaluator': evaluator_hosts, # This parameter is optional if evaluation is not performed.
            },
            'task': {'type': job_name, 'index': task_index}
    })
    ```

2. `ps_hosts` and `worker_hosts` information can be configured using `FLAGS` as follows:

    ```python
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    evaluator_hosts = FLAGS.evaluator_hosts.split(',')
    task_index = FLAGS.task_index
    job_name = FLAGS.job_name
    flags.DEFINE_string("ps_hosts", '192.168.1.100:2222,192.168.1.200:2222',)
    flags.DEFINE_string("worker_hosts",
                        '192.168.1.100:2223,192.168.1.100:2224,192.168.1.100:2225,192.168.1.100:2226,'
                        '192.168.1.100:2227,192.168.1.100:2228,192.168.1.100:2229,192.168.1.100:2230,'
                        '192.168.1.200:2223,192.168.1.200:2224,192.168.1.200:2225,192.168.1.200:2226,'
                        '192.168.1.200:2227,192.168.1.200:2228,192.168.1.200:2229,192.168.1.200:2230',)
    flags.DEFINE_string("evaluator_hosts", '192.168.1.100:2231',)
    flags.DEFINE_string("job_name", '', "One of 'ps', 'worker', 'evaluator', chief")
    flags.DEFINE_integer("task_index", 0, "Index of task within the job")
    ```

    Configuration description:

    - `worker_hosts`/`ps_hosts`: Separate entries with commas (,). Do not add a space after the comma.
    - `chief_hosts`: There can be only one chief host. It can also be omitted as in this example. If `chief` is not set, the first worker is the chief by default. Like other workers, the chief performs model training. In addition to model training, the chief worker manages other tasks (such as saving and restoring checkpoints and writing summary information).
    - `evaluator_hosts`: There can be only one evaluator host. Omit it if evaluation is not performed.

        Next, correctly set the environment variable `TF_CONFIG` for all workers.

**Defining a `ParameterServerStrategy` Instance**

To support distributed training under the PS-worker architecture, you need to define a `tf.distribute.experimental.ParameterServerStrategy` instance first. For more details about this strategy, see [this link](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/ParameterServerStrategy).

```python
strategy = tf.distribute.experimental.ParameterServerStrategy()
```

**Running the Script**

If you run the script according to `ps_hosts`, `worker_hosts`, and other information defined in the Python script itself, with no `chief` defined in the Python script:

```bash
python resnet50_ps_strategy.py --job_name=ps --task_index=0
python resnet50_ps_strategy.py --job_name=ps --task_index=1
python resnet50_ps_strategy.py --job_name=worker --task_index=0
python resnet50_ps_strategy.py --job_name=worker --task_index=1
python resnet50_ps_strategy.py --job_name=worker --task_index=2
python resnet50_ps_strategy.py --job_name=worker --task_index=3
python resnet50_ps_strategy.py --job_name=worker --task_index=4
python resnet50_ps_strategy.py --job_name=worker --task_index=5
python resnet50_ps_strategy.py --job_name=worker --task_index=6
python resnet50_ps_strategy.py --job_name=worker --task_index=7
```

If you need to redefine information such as `ps_hosts` and `worker_hosts`, with no `chief` defined in the Python script:

```bash
python resnet50_ps_strategy.py \
       --ps_hosts=192.168.1.79:2222,192.168.1.80:2222 \
       --worker_hosts=192.168.1.79:2223,192.168.1.79:2224,192.168.1.79:2225,192.168.1.79:2226,192.168.1.79:2227,192.168.1.79:2228,192.168.1.79:2229,192.168.1.79:2230,192.168.1.80:2223,192.168.1.80:2224,192.168.1.80:2225,192.168.1.80:2226,192.168.1.80:2227,192.168.1.80:2228,192.168.1.80:2229,192.168.1.80:2230 \
       --job_name=ps \
       --task_index=0
```

If you need to run `chief` and `evaluator`, change `job_name` to the corresponding defined value:

```bash
python resnet50_ps_strategy.py --job_name=chief --task_index=0
python resnet50_ps_strategy.py --job_name=evaluator --task_index=0
```

>[!NOTE]
>For the environment variables required to run the script, see "Single-Device Training Execution" in the *TensorFlow 1.15 Model Porting Guide*.

### Horovod Script Migration

Horovod is a distributed training framework built on TensorFlow, Keras, PyTorch, and MXNet. It aims to improve distributed training performance. Unlike the traditional PS-worker distributed training architecture in TensorFlow, Horovod uses AllReduce to aggregate gradients, which better utilizes bandwidth and helps remove the PS-worker bottleneck. This section describes how to migrate a distributed training script developed with Horovod so it can run distributed training on the Ascend AI Processor.

For more information about Horovod, see the official [Horovod](https://horovod.readthedocs.io/en/stable/tensorflow.html) website.

Original Horovod code:

```python
import tensorflow as tf
import horovod.tensorflow as hvd

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())

# Build model...
loss = ...
opt = tf.train.AdagradOptimizer(0.01 * hvd.size())

# Add Horovod Distributed Optimizer
opt = hvd.DistributedOptimizer(opt)

# Add hook to broadcast variables from rank 0 to all other processes during
# initialization.
hooks = [hvd.BroadcastGlobalVariablesHook(0)]

# Make training operation
train_op = opt.minimize(loss)

# Save checkpoints only on worker 0 to prevent other workers from corrupting them.
checkpoint_dir = '/tmp/train_logs' if hvd.rank() == 0 else None

# The MonitoredTrainingSession takes care of session initialization,
# restoring from a checkpoint, saving to a checkpoint, and closing when done
# or an error occurs.
with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                       config=config,
                                       hooks=hooks) as mon_sess:
  while not mon_sess.should_stop():
    # Perform synchronous training.
    mon_sess.run(train_op)
```

Migrated code:

```python
# Import the NPU library.
import tensorflow as tf
from npu_bridge.npu_init import *

# This example uses the HCCL group management API, so you need to start a separate session to initialize HCCL. For more information, see the "Collective Communication Initialization" section of the TensorFlow 1.15 Model Portal Guide.
npu_int = npu_ops.initialize_system()
npu_shutdown = npu_ops.shutdown_system()
config = tf.ConfigProto()
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
init_sess = tf.Session(config=config)
init_sess.run(npu_int)

# Pin GPU to be used to process local rank (one GPU per process)
config.gpu_options.visible_device_list = str(get_local_rank_id())  # Replace hvd.local_rank with get_local_rank_id.

# Build model...
loss = ...
opt = tf.train.AdagradOptimizer(0.01 * get_rank_size())   # Replace hvd.size with get_rank_size.

# NPU Allreduce
# Replace hvd.DistributedOptimizer with npu_distributed_optimizer_wrapper.
opt = npu_distributed_optimizer_wrapper(opt)
# Add hook to broadcast variables from rank 0 to all other processes during initialization.
hooks = [NPUBroadcastGlobalVariablesHook(0)]

# Use the broadcast interface of collective communication in session.run mode to broadcast variables.
input = tf.trainable_variables()
bcast_global_variables_op = hccl_ops.broadcast(input, 0)

# Make training operation
train_op = opt.minimize(loss)

# Save checkpoints only on worker 0 to prevent other workers from corrupting them.
checkpoint_dir = '/tmp/train_logs' if get_rank_id() == 0 else None  # Replace hvd.rank with get_rank_id.

# The MonitoredTrainingSession takes care of session initialization,
# restoring from a checkpoint, saving to a checkpoint, and closing when done
# or an error occurs.
with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                       config=config,
                                       hooks=hooks) as mon_sess:
  # Broadcast variables.
  mon_sess.run(bcast_global_variables_op)
  while not mon_sess.should_stop():
    # Perform synchronous training.
    mon_sess.run(train_op)

# Run shutdown_system after training finishes, then close the session.
init_sess.run(npu_shutdown)
init_sess.close()
```

>[!NOTE]
>The NPUDistributedOptimizer distributed optimizer is compatible in the current version.

## Precision Tuning

### Precision Tuning Scenarios

If a model is migrated from GPU/CPU training to NPU training, or if precision does not meet expectations during version iterations on NPU training, for example when the loss curve does not match expectations or the validation accuracy does not match expectations, refer to this section to tune precision and identify the problematic operator or component.

The scenarios where precision does not meet expectations are mainly divided into four categories:

- If a model is migrated from GPU/CPU training to NPU training and precision does not meet expectations, refer to [End-to-End Network Comparison between GPU/CPU and NPU](#end-to-end-network-comparison-between-gpucpu-and-npu) for precision tuning.
- If a model is continuously trained on NPU and precision does not meet expectations after the software version or configuration changes, refer to [End-to-End Network Comparison between NPUs](#end-to-end-network-comparison-between-npus) for precision tuning.
- If a model trained on NPU produces `NaN` output values, refer to [NaN Overflow Localization](#nan-overflow-localization) for precision tuning.
- If a model trained on NPU shows random precision errors, that is, precision may occasionally fail to meet expectations after multiple training runs, refer to [Random Error Localization](#random-error-localization) for precision tuning.

If you find any issues during tuning or encounter difficulties while operating, you are advised to visit the [GitCode community](https://gitcode.com/Ascend/RecSDK) and submit feedback or ask for help.

### Preparations Before Tuning

#### Migration Check

For scenarios where a model is migrated from GPU/CPU training to NPU training, perform the following checks to rule out possible issues introduced during migration.

1. Check the consistency of precision across multiple training runs.

    Train the model multiple times on the GPU/CPU and multiple times on the NPU. If the results of the multiple runs fluctuate within the same range for both the GPU/CPU and the NPU, you cannot conclude that NPU training has a precision issue. If the average precision of multiple GPU/CPU runs is significantly higher than the average precision of multiple NPU runs and exceeds the normal fluctuation range, you can conclude that NPU training has a precision issue.

2. Check the migrated model configuration.
    - Make sure the mixed precision mode on the NPU matches that on the GPU. Use the `precision_mode_v2` option with the value `origin`. For details, see the "Session Configuration Parameter Description" section in [TF Adapter API (1.x)](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/800/API/tfadapter1x/tfmigr1_tfadapi_0020.html).
    - Make sure the loss scaling function is enabled correctly on the NPU. If the GPU/CPU version uses `LossScaleManager` for dynamic LossScale calculation, you need to migrate it to `NPULossScaleOptimizer` on the NPU. For details, see the "Loss Scale" section in the *TensorFlow 1.15 Model Porting Guide*.
    - Make sure the GPU/CPU training and NPU training use the same dataset, data preprocessing method, and model hyperparameters, except for the interface changes involved in migration.

3. Use the high-precision mode.

    If precision issues still exist after the preceding checks, you can enable the high-precision mode for NPU training and train again to check whether the issue was introduced by the operator precision mode.

    Example configuration for `session.run` mode training:

    ```python
    custom_op.parameter_map["op_select_implmode"].s = tf.compat.as_bytes("high_precision")
    ```

    For details, see the "Session Configuration Parameter Description" section in [TF Adapter API (1.x)](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/800/API/tfadapter1x/tfmigr1_tfadapi_0020.html).

    Example configuration for Estimator mode training:

    ```python
    config = NPURunConfig(op_select_implmode="high_precision")
    ```

    For details, see the "NPURunConfig Configuration Parameter Description" section in [TF Adapter API (1.x)](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/800/API/tfadapter1x/tfmigr1_tfadapi_0020.html).

#### Fixed Randomness Removal

##### Removing Data Randomness

If you first use interfaces like `os.listdir` to get a file list when reading data, call `sort` on the file list to ensure the same file order is obtained across different devices or repeated runs.

If `dataset.shuffle` is used in the dataset to randomly shuffle the data, comment out that line of code.

Remove or deterministically modify other kinds of random data processing as well.

##### Removing Initialization Randomness

- If the code uses the `random` module for random initialization, call `random.seed` before the random initialization to set a fixed seed. You are advised to change the initialization directly to a fixed value.
- If the code uses NumPy for random initialization, for example `random` initialization, call `numpy.random.seed` before the initialization to set a fixed seed. You are advised to change the random initialization directly to a fixed value, for example by using `numpy.full` to fill a fixed value.
- If the code uses TensorFlow random initialization, for example `tf.truncated_normal_initializer`, call `tf.set_random_seed` in TensorFlow 1 or `tf.random.set_seed` in TensorFlow 2 before the initialization to set a fixed seed. You are advised to change the random initialization directly to a fixed value, for example by using `tf.constant_initializer` to fill a fixed value.
- If the code loads a pre-trained model for initialization, make sure the same pre-trained model is loaded across different devices or repeated runs.
- Remove or deterministically modify other kinds of random initialization as well.

##### Removing Randomness in the Network Structure

If the network uses dropout, such as `tf.nn.dropout`, change the `rate` input parameter to `0`. If it uses `slim.dropout`, change the `keep_prob` input parameter to `1`.

If the network calls TensorFlow random modules, such as `tf.random_uniform`, you are advised to replace the random value directly with a constant such as `tf.constant`.

Remove or deterministically modify other kinds of random network structures as well.

##### Enabling Deterministic Computing

When you train on the GPU or NPU, repeated runs may produce different results. This difference usually comes from asynchronous multithreaded execution in operator implementations, which changes the order of floating-point accumulation. On the NPU, you can enable deterministic computation to make repeated results the same and improve the accuracy of precision comparison. However, operator execution becomes slower and performance drops, so you can choose whether to enable it as required.

- Example configuration for `session.run` mode training:

    ```python
    custom_op.parameter_map["deterministic"].i = 1
    ```

    For details, see the "Session Configuration Parameter Description" section in [TF Adapter API (1.x)](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/800/API/tfadapter1x/tfmigr1_tfadapi_0020.html).

- Example configuration for Estimator mode training:

    ```python
    config = NPURunConfig(deterministic=1)
    ```

    For details, see the "NPURunConfig Configuration Parameter Description" section in [TF Adapter API (1.x)](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/800/API/tfadapter1x/tfmigr1_tfadapi_0020.html).

##### Validating Fixed Randomness

To determine whether randomness has been removed, perform the following checks:

- After training the same model multiple times, compare the loss values of a run against itself. If the first-step loss is exactly the same and later loss values differ only slightly, but the difference is much smaller than when randomness is not removed, for example if the loss difference is less than 0.001, you can conclude that randomness has been removed.
- If you train on the CPU, or if you enable deterministic computation on the NPU, the multi-step loss values across repeated runs must be exactly the same. If the loss difference is large, the network may still contain randomness, and you need to keep checking whether all randomness has been fixed.

#### Tool Deployment

For scenarios that require full-network comparison, deploy the precision analysis tool. Upload the `precision_tool` folder from the [Ascend tools](https://gitee.com/ascend/tools) project to any directory in the training working directory. You need this tool both for collecting training precision data and for performing precision comparison analysis after training.

### End-to-End Network Comparison between GPU/CPU and NPU

#### GPU/CPU Data Dump

1. Install the dump tool dependencies.

    ```bash
    pip3 install gnureadline pexpect
    ```

2. Modify the training script and insert the dump configuration.

    - Example configuration for `session.run` mode training:

        ```python
        import precision_tool.tf_config as npu_tf_config
        sess = npu_tf_config.sess_dump(sess=sess)
        ```

    - Example configuration for Estimator mode training:

        ```python
        import precision_tool.tf_config as npu_tf_config
        estim_specs = tf.estimator.EstimatorSpec(training_hooks=[npu_tf_config.estimator_dump()])
        ```

    > [!NOTE]NOTE
    > - In `session.run` mode, you cannot use dump configuration and Rec SDK TensorFlow model saving at the same time.
    > - During multi-card training, you only need to add dump configuration to the training of one card. Otherwise, saving data from multiple cards at the same time will cause data conflicts.

3. Perform training.

    Change the maximum number of training steps to 1 and run training. Dump data will be generated in the `precision_data/tf/tf_debug/` directory.

4. Parse the dump data.

    After you run `python3 precision_tool/cli.py tf_dump`, parsed dump data will be generated in the `precision_data/tf/dump/` directory. If you need to regenerate dump data, delete the generated data and run training and parsing again.

#### NPU Data Dump

1. Modify the training script and insert the dump configuration.

    - Example configuration for `session.run` mode training:

        ```python
        import precision_tool.tf_config as npu_tf_config
        config = npu_tf_config.session_dump_config(config, action='dump')
        sess = tf.Session(config)
        ```

    - Example configuration for Estimator mode training:

        ```python
        import precision_tool.tf_config as npu_tf_config
        dump_config = npu_tf_config.estimator_dump_config(action='dump')
        npu_config = NPURunConfig(dump_config=dump_config)
        ```

    > [!NOTE]NOTE
    > - In `session.run` mode, you cannot use dump configuration and Rec SDK TensorFlow model saving at the same time.
    > - During multi-card training, you only need to add dump configuration to the training of one card. Otherwise, saving data from multiple cards at the same time will cause data conflicts.

2. Perform training.

    Change the maximum number of training steps to 1 and run training. Dump data files are generated in the directory specified by `precision_data/npu/debug_0/`, that is, the `precision_data/npu/debug_0/dump/{time}/{deviceid}/{model_name}/{model_id}/{data_index}` directory. Example directory structure:

    ```bash
    precision_data/npu/debug_0/dump/20240125153144/0/ge_default_20240125153322_41/6/0/
    ```

    **Table 1** Dump data file path format

    |Path Key|Description|Remarks|
    |--|--|--|
    |dump_path|Path for storing dump data. If a relative path is set, the full path after concatenation is used.|-|
    |time|Time when the dump data files are written to the drive.|The format is *YYYYMMDDHHMMSS*.|
    |deviceid|Device ID.|-|
    |model_name|Subgraph name.|There may be multiple folders at the `model_name` level. Use the data under the directory that corresponds to the computational graph name.<br>If `model_name` contains dots (.), slashes (/), backslashes (\), or spaces, they are converted into underscores (_).|
    |model_id|Subgraph ID.|-|
    |data_index|Iteration number used to save the dump data for the corresponding iteration.|If `dump_step` is specified, `data_index` is the same as `dump_step`. If `dump_step` is not specified, `data_index` starts at `0` and increments by 1 for each dumped iteration.|

#### Dump Data Comparison

**Preparing Data**

Upload the `precision_tool` and `precision_data` folders, including the GPU/CPU benchmark data and the NPU precision data, to any directory in the Toolkit installation environment. Example directory structure:

```bash
├── precision_tool
│    ├── cli.py
│    ├── ...
├── precision_data
│    ├── npu
│    │    ├── debug_0  // Stores NPU dump data.
│    ├── tf
│    │    ├── dump     // Stores GPU/CPU dump data.
│    ├── ...
```

**Installing Dependencies**

```bash
# Graphviz is an optional dependency. Install it only when you need to draw operator subgraphs.
pip3 install rich graphviz
# ubuntu/Debian
sudo apt-get install graphviz
# fedora/CentOS
sudo yum install graphviz
Modify config.py in the precision_tool/lib/config directory.
# This tool depends on the atc and msaccucmp.py tools in the Toolkit package, so configure the Toolkit installation directory.
# By default, the Toolkit package is installed in /usr/local/Ascend, so you do not need to modify it unless you installed it in a different directory.
CMD_ROOT_PATH = '/usr/local/Ascend'
```

**Comparing Data**

1. Start the PrecisionTool interactive CLI:

    ```bash
    python3 ./precision\_tool/cli.py
    ```

2. Enter the interactive CLI. To exit, press `Ctrl + C`.

    ```bash
    PrecisionTool \>
    ```

3. Run the `ac -l [limit_num] (-c)` command to perform full-network precision comparison. For details, see the "precision_tool Command Reference" section of the TensorFlow 1.15 Model Migration Guide.

    ```bash
    PrecisionTool > ac -c
    ```

    The comparison takes different amounts of time depending on the data volume.

    The comparison results are stored in CSV format in the `precision_data/temp/vector_compare` directory.

**Analyzing Precision**

Open the CSV file in the `precision_data/temp/vector_compare` directory and search from top to bottom for the first operator whose output cosine similarity is less than 0.98. For details, see the "Full-Network Precision Comparison Result File Description" section of the *TensorFlow 1.15 Model Porting Guide*. You can also use the `vcs -f [file_name] -c [cos_sim_threshold] -l [limit]` command to filter comparison results. See the "precision_tool Command Reference" section of the *TensorFlow 1.15 Model Porting Guide*. If the 0.98 threshold does not filter any results, you can increase the threshold and keep filtering.

The following figure shows an example `vcs` command result.

![](../../figures/tf_rec_v1/image_0000002210421105.png)

- **Left** indicates the name of the operator that generated dump data based on NPU execution.
- **Right** indicates the name of the operator that generated npy or dump data based on GPU/CPU execution.
- **Input** and **Output** indicate the cosine similarity comparison result for each input and output of the operator. The range is [-1, 1]. The closer the result is to 1, the more similar the two values are. The closer it is to -1, the more opposite the two values are.

From the comparison result in the preceding figure, you can see that the operator inputs are basically the same, but the first output differs significantly from the benchmark, with a cosine similarity of 0.806927, which is less than 0.98. This indicates that the operator may have a precision issue. If the operator inputs already differ significantly, you need to continue checking the comparison result of the input node.

> [!NOTE]NOTE
> Run the `ni (-n) [op_name] -g [graph] -a [attr] -s [save sub graph depth]` command to query the input and output node information of the operator. For details, see the "precision_tool Command Reference" section of the *TensorFlow 1.15 Model Porting Guide*.
> <br>![](../../figures/tf_rec_v1/image_0000002210306721.png)
> <br>The `ni` command can obtain the following key information based on the operator name you pass in.
> <br>The text inside `[]` is the operator type. In the preceding figure, the operator type is `Add`. If `PassName` is included, the operator is a fusion operator, and the corresponding value is the name of the fusion rule. `OriginOp` indicates the operator before fusion.

#### Problem Source Isolation

1. After you use the preceding commands to find the first operator whose input is similar but whose output differs, you can determine that the issue lies in that operator. The following is an example.

    **Figure 1** Unique operator output differs in the vcs command output
    ![](../../figures/tf_rec_v1/unique-operator-output-differs-in-the-vcs-command-output.png "Unique operator output differs in the vcs command output")

2. If the operator is a fusion operator, the precision issue is caused by operator fusion. You can disable the fusion and perform precision comparison again to determine whether other issues still exist. The following is an example.

    **Figure 2** AutomaticBufferFusionOp output differs in the vcs command output
    ![](../../figures/tf_rec_v1/automatic-buffer-fusion-op-output-differs-in-the-vcs-command-output.png "AutomaticBufferFusionOp output differs in the vcs command output")

3. If the input or output of the operator contains an embedding variable and the embedding variable differs, the issue is a precision issue in Rec SDK TensorFlow table lookup. The following is an example.

    **Figure 3** Output differs for a Rec SDK TensorFlow table lookup operator in the CSV file generated by the vcs command
    ![](../../figures/tf_rec_v1/output-differs-for-a-rec-sdk-tensor-flow-table-lookup-operator-in-the-csv-file-generated-by-the-vcs-command.png "Output differs for a Rec SDK TensorFlow table lookup operator in the CSV file generated by the vcs command")

### End-to-End Network Comparison between NPUs

#### NPU Data Dump

Follow the instructions in [NPU Data Dump](#npu-data-dump). The data is saved by default in the `precision_data/npu/debug_0` directory. Copy the data to the `precision_data/npu/debug_1` directory with the copy command `mv precision_data/npu/debug_0/ precision_data/npu/debug_1`, then run training again in the NPU environment and collect dump data. The new data is saved by default in the `precision_data/npu/debug_0` directory.

#### Comparing Dump Data

1. Start the PrecisionTool interactive CLI:

    ```bash
    python3 ./precision_tool/cli.py
    ```

2. Enter the interactive CLI. To exit, press `Ctrl + C`.

    ```bash
    PrecisionTool >
    ```

3. Run the `vc -lt [left_path] -rt [right_path] -g [graph]` command to perform full-network data comparison:

    ```bash
    vc -lt precision_data/npu/debug_1/dump/20211016164504/1/ge_default_20211016164504_1/1/0 -rt precision_data/npu/debug_0/dump/20211016180613/1/ge_default_20211016180613_1/1/0
    ```

    Precision comparison results are generated in the `out_dir` directory. For data analysis, see the "Full-Network Precision Comparison Result File Description" section of the *TensorFlow 1.15 Model Porting Guide*. Open the CSV file in the directory and search from top to bottom for the first operator whose output cosine similarity is less than 0.98.

4. For the preceding results, you can also use the `precision_tool` command `ni (-n) [op_name] -g [graph] -a [attr] -s [save sub graph deep]` for single-layer data comparison analysis. For details, see the "precision_tool Command Reference" section of the *TensorFlow 1.15 Model Porting Guide*.

5. When both `debug_0` and `debug_1` exist in the `precision_data/npu/` directory, the `ni` command parses the dump files with the same operator name in both folders at the same time. From the parsing result, you can clearly see the data differences.

    ![](../../figures/tf_rec_v1/image_0000002210306725.png)

    `Op` is the operator type. In the preceding figure, the operator name is `trans_Cast_4940`.

#### Problem Source Isolation

1. After you find the first operator whose input is similar but whose output differs, you can determine that the issue lies in that operator.
2. If the operator is a fusion operator, the precision issue is caused by operator fusion. You can disable the fusion and perform precision comparison again to determine whether other issues still exist.
3. If the input or output of the operator contains an embedding variable and the embedding variable differs, the issue is a precision issue in Rec SDK TensorFlow table lookup.

### NaN Overflow Localization

#### Determine the Overflow Step

Overflow localization depends on precision data dumps. If you can reliably reproduce the model output `NaN` issue at a certain training iteration after fixed randomness is removed, you can specify the dump step for training:

1. Modify `config.py` in the `precision_tool/lib/config` directory and specify the step for which you want to dump data.

    ```bash
    # Dump data for a specific step. In general, you only need to compare the first dumped layer, that is, keep the default value. If you need to specify a particular step, you can change it, for example to '0|5|10'.
    TF_DUMP_STEP = '0'
    ```

2. Change `TF_DUMP_STEP` to the step where `NaN` appears. Note that `TF_DUMP_STEP=0` corresponds to the first training step of the dumped model.

    If the loss `NaN` issue cannot be stably reproduced at a certain training iteration, you can modify `TF_DUMP_STEP` to a range or run multiple times as needed so that the precision data for the corresponding step is dumped before you proceed to the next analysis step. Because dump data uses a large amount of memory, make sure you do not dump too much data and delete useless dump data in time.

#### NPU Data Dump

Follow the instructions in [NPU Data Dump](#npu-data-dump). The data is saved by default in the `precision_data/npu/debug_0` directory.

#### Parsing Dump Data

1. Run the command to parse the original dump binary data files into `npy` files that NumPy can read.

    ```bash
    find precision_data/npu/debug_0 -type f -name "*" | xargs -i python3 /usr/local/Ascend/ascend-toolkit/latest/tools/operator_cmp/compare/msaccucmp.py convert -d {} -out dump_data_npy/ -v 2
    ```

    This command converts the files in the `precision_data/npu/debug_0` directory by using the `msaccucmp.py` script and saves them to the `dump_data_npy` directory. The `/usr/local/Ascend/ascend-toolkit/` directory is the CANN installation directory. You can change it as needed. For details about the `msaccucmp.py` command, see the "Dump Data File Format Conversion" section in the *CANN Model Accuracy Analyzer User Guide*.

2. Find the source of `NaN` in the `npy` files.

    The converted `npy` data is stored in the `dump_data_npy` directory and contains input and output data for all operators in the network. The five positions after the file name is split by periods (.) are the timestamp. Sort all files by timestamp in ascending order and determine whether `NaN` exists in the files one by one. Find the first data file that contains `NaN`. If one exists, print the file name and stop the loop. If nothing is printed, the file does not contain `NaN`, and you need to check whether the dump step ran correctly. For details about the file naming format, see the "Data Format Requirements" section in the *CANN Model Accuracy Analyzer User Guide*.

    Run the `python3 find_nan.py` command. The `find_nan.py` file is as follows:

    ```python
    import glob
    import numpy as np

    files = glob.glob("dump_data_npy/*")
    files.sort(key = lambda x : int(x.split(".")[4]))
    for i in files:
          f = np.load(i)
          if np.isnan(f).any():
          print(i)
          break
    ```

#### Problem Source Isolation

After you find the dump data file for the first operator that contains `NaN`, determine the root cause as follows:

1. If `NaN` appears in the output data of an operator, take the input data of that operator and run the same operator logic on the CPU to obtain the CPU operator output. If the CPU operator output does not match the NPU operator output, the `NaN` was caused by incorrect operator execution, and you can isolate the issue to that operator. If the CPU operator output matches the NPU operator output, the `NaN` was caused by normal operator execution, and you need to keep tracing upstream to determine whether an upstream operator has an execution issue or other issues.
2. If `NaN` appears in the input data of an operator, continue tracing upstream to find the source of `NaN`. If the upstream operator output is normal but the current operator input is abnormal, you can isolate the issue to memory corruption that occurred in the interval between the upstream operator execution and the current operator. Otherwise, keep checking whether the upstream operator has other issues.
3. If `NaN` appears in an embedding variable, the issue is in Rec SDK TensorFlow table lookup.

### Random Error Localization

#### Localization Approach

In scenarios where occasional errors during NPU training cause precision to fall below expectations, for example when the loss suddenly changes in a certain iteration of one training run, the issue may not be reproducible in a stable way. Because dump data takes time to collect and consumes a lot of memory, it is difficult to use a dump-based comparison solution. In this case, you can use model file comparison to investigate the issue. Compare all variables in the model file from the abnormal run and the model file from a normal run. Find the variable with the lowest cosine similarity. If its cosine similarity is below a certain value, for example 0.98, you can conclude that the issue was introduced by the operator that outputs that variable.

#### Model Training and Saving

After you follow [Fixed Randomness Removal](#fixed-randomness-removal) without necessarily enabling deterministic computation, reduce the model save interval as much as possible, for example save every five steps, so that you can reproduce the precision anomaly within a reasonable amount of time. Then take the model file from the step immediately after the loss anomaly occurs and compare it with the model file from the same step in a normal training run.

#### Model Comparison

Change `checkpoint_path1` in `ckpt_compare.py` to the path of the abnormal model file and `checkpoint_path2` to the path of the normal model file. Then run the following `python ckpt_compare.py` script in the TensorFlow 1.15 environment. The script outputs variable names and cosine similarities from low to high.

```python
from tensorflow.python import pywrap_tensorflow
import numpy as np
checkpoint_path1 = "path1/model-200"
checkpoint_path2 = "path2/model-200"
reader1 = pywrap_tensorflow.NewCheckpointReader(checkpoint_path1)
reader2 = pywrap_tensorflow.NewCheckpointReader(checkpoint_path2)
var_to_shape_map = reader1.get_variable_to_shape_map()
key_cos = {}
for key in var_to_shape_map:
     tensor1 = reader1.get_tensor(key).reshape(-1)
     tensor2 = reader2.get_tensor(key).reshape(-1)
     key_cos[key] = np.dot(tensor1,tensor2)/(np.linalg.norm(tensor1)*np.linalg.norm(tensor2))
key_cos = list(key_cos.items())
key_cos.sort(key = lambda x : x[1])
for key, cos in key_cos:
     print(key, cos)
```

![](../../figures/tf_rec_v1/image_0000002175060386.png)

#### Problem Source Isolation

1. Find the first variable name. If the cosine similarity of that variable is low, you can conclude that the precision issue was introduced by the operator that outputs that variable.
2. If the operator is a fusion operator, the precision issue is caused by operator fusion. You can disable the fusion and perform precision comparison again to determine whether other issues still exist.
3. If the operator related to that variable is associated with the Rec SDK TensorFlow table lookup component, the issue is a precision issue in Rec SDK TensorFlow table lookup.

### References

[Precision Tuning Process](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/800/migration/tfmigr1/tfmigr1_tfprecision_0002.html)

[precision_tool Command Reference](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/800/migration/tfmigr1/tfmigr1_tfprecision_0050.html)

[Full-Network Precision Comparison Result File Description](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/800/migration/tfmigr1/tfmigr1_tfprecision_0051.html)

## Rec SDK TensorFlow Migration Examples

Rec SDK TensorFlow supports migration and adaptation of open-source TensorFlow recommendation models. You can refer to the following migration examples:

- [DLRM Example](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/tf_rec_v2_examples/DLRM/model)
- [LittleDemo Example](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/tf_rec_v2_examples/little_demo)
