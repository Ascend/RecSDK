# 迁移与训练<a name="ZH-CN_TOPIC_0000001630127069"></a>

## 训练场景介绍<a name="ZH-CN_TOPIC_0000001580012140"></a>

**训练场景介绍<a name="section16627105015515"></a>**

Rec SDK TensorFlow提供使用tf.Session训练场景（当前暂不支持Estimator训练场景）。

-   tf.Session训练场景。通过新建的Session实例启动模型运行，返回Tensor示例，进行定制化模型训练。

>[!NOTE] 说明 
>-   Rec SDK TensorFlow暂时不支持Keras。
>-   Rec SDK TensorFlow目前仅支持使用TensorFlow原生API模型训练脚本迁移，不支持使用第三方框架（tf\_adapter、HugeCTR、DeepRec等）。
>-   启用大小循环的情况下，训练迭代的总次数必须是小循环（即iterations\_per\_loop）的整数倍。

**TensorFlow与Rec SDK TensorFlow接口对应关系<a name="section9248145363514"></a>**

在进行模型迁移时，需要根据实际的模型代码以及代码上下文判断是否使用到稀疏表相关的接口，如果是与稀疏表相关的TensorFlow接口，需要修改为Rec SDK TensorFlow的接口，接口对应关系如[表1](#table16435142101913)所示。

**表 1**  接口对应关系
<a id="table16435142101913"></a>

|TensorFlow接口| Rec SDK TensorFlow接口 |接口功能描述|
|--|----------------------|--|
|<li>MutableHashTable</li><li>tf.Variable</li>| get_embedding_table  |创建稀疏表|
|<li>tf.embedding_lookup</li><li>mutable_hash_table.lookup（mutable_hash_table是MutableHashTable的实例）等</li>| embedding_lookup     |查询稀疏表|


接口示例：

-   TensorFlow示例：

    ```python
    import tensorflow as tf
    from tensorflow.contrib.lookup import MutableHashTable
    # .......
    user_id = features["user_ids"]
    user_emb_table = MutableHashTable(key_dtype=tf.int64, value_dtype=tf.float32, default_value=0.0)
    user_emb = user_emb_table.lookup(user_id)
    ```

-   Rec SDK TensorFlow示例：

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

## sess.run迁移与训练<a name="ZH-CN_TOPIC_0000001836561129"></a>

### sess.run迁移<a name="ZH-CN_TOPIC_0000001835670309"></a>

若原始TensorFlow网络基于sess.run API构造，可参见本节了解手工迁移全流程。

建议用户直接使用Rec SDK TensorFlow提供的模型训练样例进行其他模型适配，如需使用开源推荐项目，直接进行对应API迁移可能存在兼容性问题。

**sess.run简介<a name="section12840202717412"></a>**

sess.run API属于TensorFlow的低阶API，相对于Estimator来讲，灵活性较高，但模型的实现较为复杂。

使用sess.run API进行训练脚本开发的流程为：

1.  [数据预处理](#section3602537142311)。
2.  [模型搭建/计算Loss/梯度更新](#section177071912713)。
3.  [创建session并初始化资源](#section19387465248)。
4.  [执行训练](#section20549115411413)。

下面介绍如何迁移sess.run训练脚本，以便在昇腾AI处理器上进行训练。

**头文件增加<a name="section20907117165512"></a>**

对于以下步骤中涉及修改的Python文件，新增以下头文件引用，用于导入NPU相关库。

```python
from npu_bridge.npu_init import *
```

>[!NOTE] 说明 
>引入上述头文件后，训练脚本默认在昇腾AI处理器执行。

**数据预处理<a id="section3602537142311"></a>**

一般情况下，此部分代码无需改造。如下情况需要进行适配修改：

当原始网络脚本中使用`dataset.batch(batch_size)`返回动态形状时，由于数据流中剩余的样本数可能小于batch大小，导致网络中最后一个step的shape与之前的shape不一致，此种场景下会进入动态shape编译流程。为提升网络编译性能，建议将`drop_remainder`设置为True，丢弃文件中的最后几个样本，确保网络中每个step的shape一致。

```python
  dataset = dataset.batch(batch_size, drop_remainder=True)
```

但需要注意的是：推理时，当最后一次迭代的推理数据量小于batch_size时，需要补齐空白数据到batch_size，因为有些脚本最后会加个断言，验证结果的数量要和验证数据的数量一致。

```python
 assert num_written_lines == num_actual_predict_examples
```

**模型搭建/计算Loss/梯度更新<a id="section177071912713"></a>**

一般情况下，此部分代码无需改造。如下情况需要进行适配修改：

-   对于原始网络中的dropout，请替换为CANN对应的API实现，以获得更优性能，但需关注对网络精度的影响。
    -   如果存在tf.nn.dropout，请修改为：

        ```python
        layers = npu_ops.dropout()
        ```

    -   如果存在tf.layers.dropout/tf.layers.Dropout/tf.keras.layers.Dropout/tf.keras.layers.SpatialDropout1D/tf.keras.layers.SpatialDropout2D/tf.keras.layers.SpatialDropout3D，请增加头文件引用：

        ```python
        from npu_bridge.estimator.npu import npu_convert_dropout
        ```

-   对于原始网络中的gelu，请替换为CANN对应的API实现，以获得更优性能。

    TensorFlow原始代码：

    ```python
    def gelu(x): 
      cdf = 0.5 * (1.0 + tf.tanh(
         (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))) 
      return x*cdf
    layers = gelu()
    ```

    迁移后的代码：

    ```python
    layers = npu_unary_ops.gelu(x)
    ```

**创建session并初始化资源<a id="section19387465248"></a>**

在昇腾AI处理器上通过sess.run模式执行训练脚本时，相关配置说明：

-   以下配置默认关闭，请勿开启：

    rewrite_options.disable_model_pruning

-   以下配置默认开启，请勿关闭：
    -   rewrite\_options.function\_optimization
    -   rewrite\_options.constant\_folding
    -   rewrite\_options.shape\_optimization
    -   rewrite\_options.arithmetic\_optimization
    -   rewrite\_options.loop\_optimization
    -   rewrite\_options.dependency\_optimization
    -   rewrite\_options.layout\_optimizer

-   以下配置默认开启，必须显式关闭：
    -   rewrite\_options.remapping
    -   rewrite\_options.memory\_optimization

-   如果原始网络中使用了tf.device相关代码，需要增加session配置“allow\_soft\_placement=True”，允许TensorFlow自动分配设备。

TensorFlow原始代码：

```python
#构造迭代器
iterator=Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

#取batch数据
next_batch=iterator.get_next()

#迭代器初始化
training_init_op=iterator.make_initializer(train_dataset)
 
#变量初始化
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
 
#Get the number of training/validation steps per epoch
train_batches_per_epoch=int(np.floor(train_size/batch_size))
```

迁移后的代码：

```python
#构造迭代器
iterator=Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

#取batch数据
next_batch=iterator.get_next()

#迭代器初始化
training_init_op=iterator.make_initializer(train_dataset)
 
#变量初始化
init=tf.global_variables_initializer()

#创建session，如果原始网络中使用了tf.device相关代码，则需要增加session配置“allow_soft_placement=True”，允许TensorFlow自动分配设备。
config = tf.ConfigProto(allow_soft_placement=True)
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
# 必须显式关闭TensorFlow的remapping、memory_optimization功能，避免与NPU中的功能冲突。
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 显式关闭
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 显式关闭
sess = tf.Session(config=config)
sess.run(init)
 
#Get the number of training/validation steps per epoch
train_batches_per_epoch=int(np.floor(train_size/batch_size))
```

tf.Session原生功能在Ascend平台上全部支持。

另外，Ascend平台还支持自动混合精度等功能，如果用户需要进行相关使能，可以参考《TF Adapter 接口（1.x）》的“session配置”章节。

**执行训练<a id="section20549115411413"></a>**

此部分代码无需改造，例如：

```python
#开始循环迭代
for epoch in range(num_epochs):
  ##Initialize iterator with the training dataset
  sess.run(training_init_op)
  for step in range(train_batches_per_epoch):  
    #get next batch of data
    img_batch,label_batch=sess.run(next_batch)
    #run the training op
    _,train_loss = sess.run([train_op, loss],feed_dict={x:img_batch, y_:label_batch, is_training:True})
```

tf.Session创建的session对象使用后需显式调用`session.close()`，或使用with创建session（通过上下文自动调用`close()`）。详情参考示例：

示例1：显式调用sess.close\(\)

```python
sess = tf.Session(config=config)
sess.run(...)
sess.close()
```

示例2：使用with创建session

```python
with tf.Session(config=config) as sess:
    sess.run(...)
```

>[!NOTE] 说明
>如果在迁移与训练过程中遇到报错，请参考[FAQ](faq.md)进行解决，或者联系技术支持。

## 分布式训练脚本迁移<a name="ZH-CN_TOPIC_0000001788871148"></a>

### 支持数据并行（Allreduce）<a name="ZH-CN_TOPIC_0000001835670317"></a>

Allreduce是主流的数据并行架构，各个节点按照算法协同工作，适用于对训练算力要求高、设备规模大的场景。本节介绍如何将TensorFlow训练脚本在昇腾AI处理器上通过Allreduce架构进行分布式训练。

**Allreduce实现原理<a name="section87591042174619"></a>**

大规模AI训练集群中，通常采用数据并行的方式完成训练。数据并行即每个设备使用相同的模型、不同的训练样本，每个Device计算得到的梯度数据需要聚合之后进行参数更新。

**图 1**  数据并行方式训练的示意图<a name="fig086734164810"></a>  
![](../../figures/tf_rec_v1/数据并行方式训练的示意图.png "数据并行方式训练的示意图")

如果按照梯度聚合方式进行分类，数据并行的主流实现有**PS-workers架构**和**Allreduce架构**两种。在**Allreduce架构**中，每个参与训练的Device形成一个环，没有中心节点来聚合所有计算梯度。Allreduce算法将参与训练的Device放置在一个逻辑环路（logical ring）中。每个Device从上行的Device接收数据，并向下行的Device发送数据，可充分利用每个Device的上下行带宽。

Allreduce架构是为了解决了PS-workers架构无法线性扩展问题而提出的改良架构。各个节点按照算法协同工作，算法的目标是减少传输数据量，并充分利用硬件通信带宽。一般适合训练算力要求高、设备规模大的场景。Allreduce架构的实现原理如下图所示。

**图 2**  Allreduce模式<a id="fig1321114115499"></a>  
![](../../figures/tf_rec_v1/Allreduce模式.png "Allreduce模式")

以Ring算法为例介绍Allreduce模式（称为Ring-Allreduce），如[图2](#fig1321114115499)所示，在Ring-Allreduce架构下，每个设备都是worker，并且形成一个环，不需要中心节点来聚合所有worker计算的梯度。在一个迭代过程中，每个worker完成一份mini-batch样本数据的前向计算、反向计算，得到梯度数据，然后使用Ring-Allreduce算法完成梯度数据的同步。Ring-Allreduce算法包括scatter-reduce和allgather两部分，梯度数据分多个步骤传递给环中的下一个worker，同时它也多次接收上一个worker的梯度数据。对于一个包含N个worker的环，每个worker需要从其它worker接收2\*（N-1）次梯度数据（每次接收1/N的数据），并向其他节点发送2\*（N-1）次梯度数据（每次发送1/N的数据）。

**使用的接口<a name="section291012110287"></a>**

在TensorFlow中，一般使用tf.distribute.Strategy进行分布式训练，具体请参考[链接](https://www.tensorflow.org/guide/distributed_training)。而昇腾AI处理器暂不支持上述分布式策略，TF Adapter提供了分布式接口npu\_distributed\_optimizer\_wrapper，对传入的optimizer梯度函数添加NPU的Allreduce操作，最终返回输入的优化器，从而支持单机多卡、多机多卡等组网形式下，各个Device之间计算梯度后执行梯度聚合操作。用户调用该函数后，在生成的训练图中，梯度计算和更新算子之间插入了Allreduce算子节点。

**图 3**  使用的接口<a name="fig1792101713010"></a>  
![](../../figures/tf_rec_v1/使用的接口.png "使用的接口")

因此，对于原始TensorFlow训练脚本，需要经过修改后，才可在昇腾AI处理器上支持分布式训练。

**数据集切分<a name="section949271371011"></a>**

分布式训练时，用户可以使用TensorFlow接口进行数据集切分。如果数据集切分时需要获取处理器资源信息，用户可以通过集合通信接口`get_rank_size`获取昇腾AI处理器数量，通过`get_rank_id`获取处理器id，例如：

```bash
  dataset = dataset.shard(get_rank_size(),get_rank_id())
```

**sess.run模式下脚本迁移<a name="section1177815188519"></a>**

sess.run模式的训练脚本需要用户手写实现broadcast功能。具体方法为：

1.  在变量初始化之后，训练之前，通过集合通信接口broadcast进行变量广播，关于broadcast接口的详细介绍请参见《HCCL集合通信库接口参考》。

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

    此外，broadcast接口中有改图的操作，如果图无法修改（例如冻结了图或者使用tf.train.Supervisor创建session等），则需要先取消图冻结：

    ```python
    with sv.managed_session() as sess:
      sess.graph._unsafe_unfinalize() # 取消冻结的Graph
      sess.run(bcast_op)
    ```

2.  执行训练时，在使用梯度优化器计算完各Device数据后，直接调用`npu_distributed_optimizer_wrapper`进行梯度数据聚合：

    ```python
    from npu_bridge.npu_init import *
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001) # 使用SGD优化器
    distributedOptimizer=npu_distributed_optimizer_wrapper(optimizer) # 使用NPU分布式计算，更新梯度
    ```

    >[!NOTE] 说明
    >NPUDistributedOptimizer分布式优化器在当前版本依然兼容。

    如果原始脚本使用TensorFlow接口计算梯度，例如`grads = tf.gradients(loss, tvars)`，需要在计算完梯度之后，调用npu\_allreduce接口对梯度进行Allreduce：

    ```bash
    grads = npu_allreduce(tf.gradients(a + b, [a, b], stop_gradients=[a, b]))
    ```


### 支持数据并行（PS-Worker）<a name="ZH-CN_TOPIC_0000001789030812"></a>

在推荐网络中，特征数据通过Embedding table保存，数据量最大可能达到TB（Terabyte，太字节，是一种信息计量单位，1TB=10<sup>12</sup>字节）级别，无法在Device侧保存，因此需要通过PS-Worker方式将数据保存在Host侧的内存中。本节介绍如何将TensorFlow训练脚本在昇腾AI处理器上通过PS-Worker架构进行分布式训练。

**PS-Worker实现原理<a name="section1464316265102"></a>**

**图 1**  PS-Worker模式<a name="fig42561512148"></a>  
![](../../figures/tf_rec_v1/PS-Worker模式.png "PS-Worker模式")

在PS-Worker架构中，集群中的节点被分为两类：参数服务器（parameter server）和工作服务器（worker）。其中参数服务器存放模型的参数，而工作服务器负责计算参数的梯度。在每个迭代过程，工作服务器从参数服务器中获得参数，然后将计算的梯度返回给参数服务器，参数服务器聚合从工作服务器传回的梯度，然后更新参数，并将新的参数广播给工作服务器。

下面介绍基于TensorFlow的Python API开发的训练脚本，如何在昇腾AI处理器通过PS-Worker架构进行分布式训练。

**配置集群信息<a name="section644513616218"></a>**

>[!NOTICE] 须知 
>-   在昇腾AI处理器通过PS-Worker架构进行分布式训练当前仅支持NPUEstimator模式。
>-   当前仅支持一个worker进程对应在一个device上执行。
>-   PS-Worker集群场景下，建议用户选择高速率网卡。

PS-Worker架构下通过环境变量TF\_CONFIG配置集群信息，TF\_CONFIG里包括了两个部分：cluster和task。cluster提供了关于整个集群的信息，也就是集群中的工作服务器和参数服务器。task提供了关于当前任务的信息，详细使用说明请参考[TensorFlow官网](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_estimator)。

下面以两台Server，每台Server上各1个ps，8个worker为例进行说明。

1.  设置TF\_CONFIG信息。

    ```python
    os.environ['TF_CONFIG'] = json.dumps({
            'cluster': {
                #'chief':chief_hosts, # 可不设置
                'worker': worker_hosts,
                'ps': ps_hosts,
                'evaluator':evaluator_hosts, # 不做评估的话，可不设置
            },
            'task': {'type': job_name, 'index': task_index}
    })
    ```

2.  ps\_hosts、worker\_hosts信息可以采用Flags方式配置，配置如下：

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

    配置说明：

    -   worker\_hosts/ps\_hosts：每条信息用“,”分开，“,”后不能加空格。
    -   chief\_hosts：只能有一个，也可像当前示例一样不设置。若chief不设置，则默认第一个worker为chief，chief与其他worker一样，也进行模型训练。chief worker除了进行模型训练，还管理一些其它work（例如：checkpoint保存/恢复，写入summary信息等）。
    -   evaluator\_hosts：只能有一个，如果不做评估，可以不设置。

        下面需要做的就是正确地设置所有worker的环境变量TF\_CONFIG。

**定义ParameterServerStrategy实例<a name="section475154714210"></a>**

为支持PS-Worker架构下的分布式训练，需要先定义tf.distribute.experimental.ParameterServerStrategy实例，该策略的更多细节请参考[链接](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/ParameterServerStrategy)。

```python
strategy = tf.distribute.experimental.ParameterServerStrategy()
```

**脚本运行<a name="section13115183161016"></a>**

若按python脚本内的ps_hosts，worker_hosts等信息运行（python脚本内未定义chief）：

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

若需要重新定义ps_hosts，worker_hosts等信息（python脚本内未定义chief）：

```bash
python resnet50_ps_strategy.py \
       --ps_hosts=192.168.1.79:2222,192.168.1.80:2222 \       
       --worker_hosts=192.168.1.79:2223,192.168.1.79:2224,192.168.1.79:2225,192.168.1.79:2226,192.168.1.79:2227,192.168.1.79:2228,192.168.1.79:2229,192.168.1.79:2230,192.168.1.80:2223,192.168.1.80:2224,192.168.1.80:2225,192.168.1.80:2226,192.168.1.80:2227,192.168.1.80:2228,192.168.1.80:2229,192.168.1.80:2230 \
       --job_name=ps \
       --task_index=0
```

若需运行chief和evaluator，将`job_name`更改为定义的类型值即可，即：

```bash
python resnet50_ps_strategy.py --job_name=chief --task_index=0
python resnet50_ps_strategy.py --job_name=evaluator --task_index=0
```

>[!NOTE] 说明 
>脚本运行依赖的环境变量请参考《TensorFlow 1.15模型迁移指南》的“执行单Device训练”章节。


### Horovod脚本迁移<a name="ZH-CN_TOPIC_0000001835790373"></a>

Horovod是基于TensorFlow、Keras、PyTorch以及MXNet的分布式训练框架，目的是提升分布式训练的性能。不同于传统的TensorFlow分布式训练采用PS-Worker架构，Horovod使用Allreduce进行聚合梯度，能够更好地利用带宽，解决PS worker的瓶颈问题。本节介绍如何迁移基于Horovod开发的分布式训练脚本，使其在昇腾AI处理器进行分布式训练。

关于Horovod的介绍，可参见[Horovod](https://horovod.readthedocs.io/en/stable/tensorflow.html)官网。

Horovod原始代码：

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

迁移后的代码：

```python
# 导入NPU库
import tensorflow as tf
from npu_bridge.npu_init import *

# 本示例调用了HCCL的group管理接口，因此需要另起session进行HCCL初始化，更多介绍请参考《TensorFlow 1.15模型迁移指南》的“集合通信初始化”章节
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
config.gpu_options.visible_device_list = str(get_local_rank_id())  # "hvd.local_rank"修改为"get_local_rank_id"

# Build model...
loss = ...
opt = tf.train.AdagradOptimizer(0.01 * get_rank_size())   # "hvd.size"修改为"get_rank_size"

# NPU Allreduce
# 将"hvd.DistributedOptimizer"修改为"npu_distributed_optimizer_wrapper"
opt = npu_distributed_optimizer_wrapper(opt)   
# Add hook to broadcast variables from rank 0 to all other processes during initialization.
hooks = [NPUBroadcastGlobalVariablesHook(0)]

# 在session run模式下调用集合通信接口broadcast进行变量广播：
input = tf.trainable_variables()
bcast_global_variables_op = hccl_ops.broadcast(input, 0)

# Make training operation
train_op = opt.minimize(loss)

# Save checkpoints only on worker 0 to prevent other workers from corrupting them.
checkpoint_dir = '/tmp/train_logs' if get_rank_id() == 0 else None  # "hvd.rank"修改为"get_rank_id"

# The MonitoredTrainingSession takes care of session initialization,
# restoring from a checkpoint, saving to a checkpoint, and closing when done
# or an error occurs.
with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                       config=config,
                                       hooks=hooks) as mon_sess:
  # 变量广播
  mon_sess.run(bcast_global_variables_op)  
  while not mon_sess.should_stop():
    # Perform synchronous training.
    mon_sess.run(train_op) 
  
# 训练结束后执行shutdown_system，同时关闭session
init_sess.run(npu_shutdown)
init_sess.close()
```

>[!NOTE] 说明 
>NPUDistributedOptimizer分布式优化器在当前版本依然兼容。

## 精度调优<a name="ZH-CN_TOPIC_0000002210421029"></a>

### 精度调优场景<a name="ZH-CN_TOPIC_0000002210306641"></a>

模型从GPU/CPU迁移到NPU训练，或者在NPU训练版本迭代的过程中，如果出现精度不达标的场景，例如loss曲线不符合预期或者验证精度不符合预期，可参考本章内容进行精度调优，找出存在问题的算子或者组件。

当前精度不达标的场景主要分为四类：

-   模型从GPU/CPU迁移到NPU训练，出现精度不达标的情况，可以参考[GPU/CPU与NPU整网比对](#gpucpu与npu整网比对)进行精度调优；
-   模型在NPU上持续迭代训练的过程中，软件版本或者配置发生变化后，出现精度不达标的情况，可以参考[NPU与NPU整网比对](#npu与npu整网比对)进行精度调优；
-   模型在NPU上训练出现模型输出值为NaN的情况，可以参考[NaN溢出定位](#nan溢出定位)进行精度调优；
-   模型在NPU上训练出现随机精度误差，即多次训练，可能随机出现精度不达标的情况，可以参考[随机误差定位](#随机误差定位)进行精度调优。

若在调优过程中发现任何问题或在操作时遇到困难，建议访问[GitCode社区](https://gitcode.com/Ascend/RecSDK)提交反馈或寻求帮助。


### 调优前准备<a name="ZH-CN_TOPIC_0000002175060302"></a>

#### 迁移检查<a name="ZH-CN_TOPIC_0000002174900594"></a>

对于模型从GPU/CPU迁移到NPU训练的场景，需要做以下检查，排除迁移过程中可能存在的问题。

1.  检查模型多次训练精度一致性。

    在GPU/CPU下多次训练，并且在NPU下多次训练；如果多次训练的结果中，GPU/CPU的精度和NPU的精度均在相同范围内波动，那么不能认为NPU训练存在精度问题；如果GPU/CPU多次训练的平均精度明显高于NPU多次训练的平均精度，并且超出正常波动范围，可以认为NPU训练存在精度异常。

2.  检查迁移后的模型配置。
    -   确保NPU下的混合精度模式和GPU下相同，使用“precision\_mode\_v2”选项，取值为“origin”。具体可参考《TF Adapter 接口（1.x）》的“[session配置参数说明](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/800/API/tfadapter1x/tfmigr1_tfadapi_0020.html)”章节。
    -   确保NPU下正确使能Loss Scale功能，如果GPU/CPU下使用了LossScaleManager进行动态LossScale计算，在NPU下需要迁移为NPULossScaleOptimizer。具体可参考《TensorFlow 1.15模型迁移指南》的“Loss Scale”章节。
    -   确保除迁移过程中涉及到的接口修改外，GPU/CPU训练和NPU训练使用的数据集，数据预处理方式，模型超参等配置相同。

3.  使用高精度模式。

    如果以上检查完成后仍然存在精度问题，可以打开NPU训练的高精度模式后再次训练，检查是否是由算子精度模式引入的问题。

    session.run模式训练配置示例：

    ```python
    custom_op.parameter_map["op_select_implmode"].s = tf.compat.as_bytes("high_precision")
    ```

    具体可参考《TF Adapter 接口（1.x）》的“[session配置参数说明](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/800/API/tfadapter1x/tfmigr1_tfadapi_0020.html)”章节。

    Estimator模式训练配置示例：

    ```python
    config = NPURunConfig(op_select_implmode="high_precision")
    ```

    具体可参考《TF Adapter 接口（1.x）》的“NPURunConfig配置参数说明”章节。


#### 去除固定随机性<a name="ZH-CN_TOPIC_0000002210421033"></a>

##### 去除数据随机性<a name="ZH-CN_TOPIC_0000002210306645"></a>

如果读取数据时首先使用os.listdir接口或其他接口获取到文件列表，调用sort接口对文件列表进行排序，保证不同设备或多次运行时获取到的文件顺序相同。

如果数据集中使用了dataset.shuffle操作对数据集进行随机打乱操作，将该行代码注释。

其他随机性的数据处理同样进行去除或者确定性修改。


##### 去除初始化随机性<a name="ZH-CN_TOPIC_0000002175060306"></a>

-   如果代码中使用了random模块的随机初始化，需要在随机初始化之前调用random.seed设置固定随机种子，推荐直接改为固定值初始化。
-   如果代码中使用了numpy的随机初始化，例如random初始化，需要在初始化之前调用numpy.random.seed函数设定固定的随机种子，推荐直接将random初始化改为固定值初始化，例如numpy.full填充固定值。
-   如果代码中使用了TensorFlow的随机初始化，例如tf.truncated\_normal\_initializer等，需要在初始化之前调用tf.set\_random\_seed\(TF1\)函数或者tf.random.set\_seed\(TF2\)函数设定固定的随机种子，推荐直接将随机初始化改为固定值初始化，例如tf.constant\_initializer填充固定值。
-   如果代码中加载了预训练模型进行初始化，确保不同设备或多次运行时加载的是相同的预训练模型。
-   其他随机性的初始化同样进行去除或者确定性修改。


##### 去除网络结构中的随机性<a name="ZH-CN_TOPIC_0000002174900598"></a>

如果网络中使用了dropout，例如`tf.nn.dropout`函数，将函数入参中的rate改为0，如果调用了`slim.dropout`函数，将函数入参中的`keep_prob`入参改为1。

如果网络中调用了tf的random随机模块，例如`tf.random_uniform`等函数，建议直接使用常数`tf.constant`等替代该随机生成值。

其他随机性的网络结构同样进行去除或者确定性修改。


##### 开启确定性计算<a name="ZH-CN_TOPIC_0000002210421037"></a>

在GPU/NPU设备下训练时，多次执行的结果可能不同。这个差异的来源，一般是因为在算子实现中，存在异步的多线程执行，会导致浮点数累加的顺序变化。NPU下可以开启确定性计算，保证多次执行结果相同，提高精度比对的准确性，但算子执行时间会变慢，导致性能下降，可根据实际情况选择是否开启。

-   session.run模式训练配置示例：

    ```python
    custom_op.parameter_map["deterministic"].i = 1
    ```

    具体可参考《TF Adapter 接口（1.x）》的“[session配置参数说明](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/800/API/tfadapter1x/tfmigr1_tfadapi_0020.html)”章节。

-   Estimator模式训练配置示例：

    ```python
    config = NPURunConfig(deterministic=1)
    ```

    具体可参考《TF Adapter 接口（1.x）》的“NPURunConfig配置参数说明”章节。


##### 固定随机性检验<a name="ZH-CN_TOPIC_0000002210306649"></a>

为判断随机性是否消除，可以进行以下验证：

-   相同模型多次训练后，自身与自身进行loss对比，第一步loss完全相同，后续loss差异存在细微差异，但差异值明显小于未消除随机性的情况，例如loss差异小于0.001，则可认为随机性消除。
-   如果CPU训练，或者NPU开启确定性计算训练，自身多次训练的多步loss对比需要完全相同；如果loss差异较大，则网络中可能还存在随机性，需要继续排查随机性是否完全固定。



#### 工具部署<a name="ZH-CN_TOPIC_0000002175060314"></a>

对于需要进行整网比对的场景，需要部署精度分析工具进行精度分析。将[Ascend](https://gitee.com/ascend/tools)项目中的`precision_tool`文件夹上传到训练工作目录下，训练精度数据采集以及训练完成后的精度比对分析均需要用到该工具。



### GPU/CPU与NPU整网比对<a name="ZH-CN_TOPIC_0000002174900602"></a>

#### GPU/CPU数据dump<a name="ZH-CN_TOPIC_0000002210421041"></a>

1.  安装dump工具依赖。

    ```bash
    pip3 install gnureadline pexpect
    ```

2.  修改训练脚本，插入dump配置。

    -   session.run模式训练配置示例：

        ```python
        import precision_tool.tf_config as npu_tf_config
        sess = npu_tf_config.sess_dump(sess=sess)
        ```

    -   Estimator模式训练配置示例：

        ```python
        import precision_tool.tf_config as npu_tf_config
        estim_specs = tf.estimator.EstimatorSpec(training_hooks=[npu_tf_config.estimator_dump()])
        ```

    >[!NOTE] 说明 
    >-   session.run模式下，不支持dump配置和Rec SDK TensorFlow模型保存功能同时使用。
    >-   多卡训练时，仅需在某一张卡的训练中增加dump配置，否则多卡同时保存会导致数据冲突。

3.  执行训练。

    将训练最大步数修改为1后执行训练，会在`precision_data/tf/tf_debug/`目录生成dump数据。

4.  解析dump数据。

    执行`python3 precision_tool/cli.py tf_dump`后，会在`precision_data/tf/dump/`目录生成解析好的dump数据。如果需要重新生成dump数据，将已生成的数据删除再重新执行训练和解析操作即可。


#### NPU数据dump<a name="ZH-CN_TOPIC_0000002210306653"></a>

1.  修改训练脚本，插入dump配置。

    -   session.run模式训练配置示例：

        ```python
        import precision_tool.tf_config as npu_tf_config
        config = npu_tf_config.session_dump_config(config, action='dump')
        sess = tf.Session(config)
        ```

    -   Estimator模式训练配置示例：

        ```python
        import precision_tool.tf_config as npu_tf_config
        dump_config = npu_tf_config.estimator_dump_config(action='dump')
        npu_config = NPURunConfig(dump_config=dump_config)
        ```

    >[!NOTE] 说明 
    >-   session.run模式下，不支持dump配置和Rec SDK TensorFlow模型保存功能同时使用。
    >-   多卡训练时，仅需在某一张卡的训练中增加dump配置，否则多卡同时保存会导致数据冲突。

2.  执行训练。

    将训练最大步数修改为1后执行训练。Dump数据文件会生成在`precision_data/npu/debug_0/`指定的目录下，即`precision_data/npu/debug_0/dump/{time}/{deviceid}/{model_name}/{model_id}/{data_index}`目录下。文件目录结构示例：

    ```bash
    precision_data/npu/debug_0/dump/20240125153144/0/ge_default_20240125153322_41/6/0/
    ```

    **表 1**  dump数据文件路径格式说明

    |路径key|说明|备注|
    |--|--|--|
    |dump_path|dump数据存放路径（如果设置的是相对路径，则为拼接后的全路径）。|-|
    |time|dump数据文件落盘的时间。|格式为：YYYYMMDDHHMMSS|
    |deviceid|Device设备ID号。|-|
    |model_name|子图名称。|model_name层可能存在多个文件夹，dump数据取计算图名称对应目录下的数据。<br>如果model_name出现了“.”、“/”、“\”以及空格时，转换为下划线表示。|
    |model_id|子图ID号。|-|
    |data_index|迭代数，用于保存对应迭代的dump数据。|如果指定了dump_step，则data_index和dump_step一致；如果不指定dump_step，则data_index序号从0开始计数，每dump一个迭代的数据，序号递增1。|



#### Dump数据对比<a name="ZH-CN_TOPIC_0000002175060318"></a>

**数据准备<a name="section109541144122610"></a>**

将`precision_tool`和`precision_data`（包括GPU/CPU标杆数据和NPU的精度数据）文件夹上传到Toolkit安装环境的任意目录下，目录结构示例：

```bash
├── precision_tool
│    ├── cli.py
│    ├── ...
├── precision_data
│    ├── npu
│    │    ├── debug_0  // 存放NPU dump数据
│    ├── tf
│    │    ├── dump     // 存放GPU/CPU dump数据
│    ├── ...
```

**安装依赖<a name="section1894611032716"></a>**

```bash
# graphviz为可选依赖，只有当需要绘制算子子图时才需要安装
pip3 install rich graphviz
# ubuntu/Debian
sudo apt-get install graphviz
# fedora/CentOS
sudo yum install graphviz
修改工具precision_tool/lib/config目录下的config.py
# 依赖Toolkit包中的atc和msaccucmp.py工具，配置为Toolkit包安装目录
# 默认Toolkit包安装在/usr/local/Ascend，可以不用修改，指定目录安装则需要修改
CMD_ROOT_PATH = '/usr/local/Ascend'
```

**数据对比<a name="section106461849142715"></a>**

1.  启动PrecisionTool交互命令行：

    ```bash
    python3 ./precision\_tool/cli.py
    ```

2.  进入交互命令行界面（如需退出，可执行Ctrl + c）。

    ```bash
    PrecisionTool \>
    ```

3.  执行<b>ac -l \[limit\_num\] \(-c\)</b>命令进行整网精度比对，具体可参考《TensorFlow 1.15模型迁移指南》的“precision\_tool命令参考”章节。

    ```bash
    PrecisionTool > ac -c
    ```

    根据数据量大小，比对过程需要时间不同。

    对比结果会以csv的格式存放在`precision_data/temp/vector_compare`目录中。

**精度分析<a name="section1599411334298"></a>**

打开`precision_data/temp/vector_compare`目录下的csv文件，从上向下查找第一个输出余弦相似度小于0.98的算子，具体可参考《TensorFlow 1.15模型迁移指南》的“整网精度比对结果文件说明”章节；或者使用`vcs -f [file_name] -c [cos_sim_threshold] -l [limit]`命令（可参考《TensorFlow 1.15模型迁移指南》的“`precision_tool`命令参考”章节）筛选比对结果，vcs命令默认筛选余弦相似度小于0.98的结果，找到输出的第一个算子。如果阈值0.98无法筛选出结果，可以提高阈值继续筛选。

下图为vcs命令结果举例：

![](../../figures/tf_rec_v1/zh-cn_image_0000002210421105.png)

-   Left：表示基于NPU运行生成的dump数据的算子名。
-   Right：表示基于GPU/CPU运行生成的npy或dump数据的算子名。
-   Input和Output：表示该算子各输入输出的余弦相似度算法比对结果，范围是\[-1,1\]，比对的结果如果越接近1，表示两者的值越相近，越接近-1意味着两者的值越相反。

从上图的比对结果可以看到，算子的输入基本一致，但第一个输出与标杆存在明显差异（余弦相似度为0.806927，小于0.98），说明该算子可能存在精度问题；如果算子的输入就存在明显差异，需要继续找输入节点的比对结果。

>[!NOTE] 说明 
>执行`ni (-n) [op_name] -g [graph] -a [attr] -s [save sub graph depth]`命令，可以查询算子的输入输出节点信息，具体可参考《TensorFlow 1.15模型迁移指南》的“`precision_tool`命令参考”章节。
><br>![](../../figures/tf_rec_v1/zh-cn_image_0000002210306721.png)
><br>ni命令可以根据传入的算子名称，得到如下关键信息：
><br>`[]`内部为算子类型，以上图为例，算子类型为Add，如果包含PassName，表示该算子为融合算子，对应值表示融合规则名称，OriginOp为融合前的算子。


#### 问题来源定界<a name="ZH-CN_TOPIC_0000002174900606"></a>

1.  使用以上命令找到首个输入相似，输出存在差异的算子后，可以确定是该算子存在问题。示例如下：

    **图 1**  vcs命令输出中Unique算子输出存在差异<a name="fig74041256134714"></a>  
    ![](../../figures/tf_rec_v1/vcs命令输出中Unique算子输出存在差异.png "vcs命令输出中Unique算子输出存在差异")

2.  如果该算子为融合算子，表明是由于算子融合导致精度问题，可以关闭该融合后重新进行精度比对，确定是否还存在其他问题。示例如下：

    **图 2**  vcs命令输出中AutomaticBufferFusionOp输出存在差异<a name="fig4431239174810"></a>  
    ![](../../figures/tf_rec_v1/vcs命令输出中AutomaticBufferFusionOp输出存在差异.png "vcs命令输出中AutomaticBufferFusionOp输出存在差异")

3.  如果定位到算子的输入或输出中包含embedding variable，并且embedding variable存在差异，表明是Rec SDK TensorFlow查表存在精度问题。示例如下：

    **图 3**  vcs命令生成的csv文件中Rec SDK TensorFlow查表算子输出存在差异<a name="fig14804312114920"></a>  
    ![](../../figures/tf_rec_v1/vcs命令生成的csv文件中Rec-SDK-TensorFlow查表算子输出存在差异.png "vcs命令生成的csv文件中Rec-SDK-TensorFlow查表算子输出存在差异")



### NPU与NPU整网比对<a name="ZH-CN_TOPIC_0000002210421045"></a>

#### NPU数据dump<a name="ZH-CN_TOPIC_0000002210306657"></a>

参考[NPU数据dump](#npu数据dump)执行，该数据默认保存在“precision_data/npu/debug_0”目录下。将以上数据转存到“precision_data/npu/debug_1”目录下（转存命令：**mv precision\_data/npu/debug\_0/ precision\_data/npu/debug\_1**），再次在NPU环境执行训练，采集dump数据，该数据默认保存在“precision_data/npu/debug_0”目录下。


#### Dump数据对比<a name="ZH-CN_TOPIC_0000002175060322"></a>

1.  启动PrecisionTool交互命令行：

    ```bash
    python3 ./precision_tool/cli.py
    ```

2.  进入交互命令行界面（如需退出，可执行ctrl + c）：

    ```bash
    PrecisionTool >
    ```

3.  使用<b>vc -lt \[left\_path\] -rt \[right\_path\] -g \[graph\]</b>命令进行整网数据比对：

    ```bash
    vc -lt precision_data/npu/debug_1/dump/20211016164504/1/ge_default_20211016164504_1/1/0 -rt precision_data/npu/debug_0/dump/20211016180613/1/ge_default_20211016180613_1/1/0
    ```

    在out\_dir目录生成精度比对结果，可参考《TensorFlow 1.15模型迁移指南》的“整网精度比对结果文件说明”章节进行数据分析，打开目录下的csv文件，从上向下查找第一个输出余弦相似度小于0.98的算子。

4.  针对以上结果，还可以使用precision\_tool的<b>ni \(-n\) \[op\_name\] -g \[graph\] -a \[attr\] -s \[save sub graph deep\]</b>命令进行单层数据比对分析，具体可参考《TensorFlow 1.15模型迁移指南》的“precision\_tool命令参考”章节。
5.  当precision\_data/npu/目录下同时存在debug\_0和debug\_1的时候，ni命令会同时解析两个文件夹下相同算子名的dump文件，从该解析结果中，可以比较直观的看出数据差异。

    ![](../../figures/tf_rec_v1/zh-cn_image_0000002210306725.png)

    Op为算子类型，以上图为例，算子名为`trans_Cast_4940`。


#### 问题来源定界<a name="ZH-CN_TOPIC_0000002174900610"></a>

1.  找到首个输入相似，输出存在差异的算子后，可以确定是该算子存在问题；
2.  如果该算子为融合算子，表明是由于算子融合导致精度问题，可以关闭该融合后重新进行精度比对，确定是否还存在其他问题；
3.  如果定位到算子的输入或输出中包含embedding variable，并且embedding variable存在差异，表明是Rec SDK TensorFlow查表存在精度问题。



### NaN溢出定位<a name="ZH-CN_TOPIC_0000002210421049"></a>

#### 确认溢出步数<a name="ZH-CN_TOPIC_0000002210306661"></a>

溢出定位依赖于精度数据dump，如果固定随机性后，能够在训练某一步迭代稳定复现模型输出NaN问题，那么可以指定dump步数进行训练：

1.  修改工具precision\_tool/lib/config目录下的config.py，指定需要dump数据的step。

    ```bash
    # dump特定step的数据，一般对比分析dump首层即可，即保持默认值，如需指定特定step可以修改，例如 '0|5|10'
    TF_DUMP_STEP = '0'
    ```

2.  将TF\_DUMP\_STEP修改为出现NaN的步数，需要注意TF\_DUMP\_STEP=0对应dump模型训练的第1步。

    如果loss nan问题无法稳定复现在训练的某一步迭代，可根据实际情况修改`TF_DUMP_STEP`为一定范围，或者多次执行，保证dump到了对应步数的精度数据后才能进行下一步分析。由于dump数据占用内存较大，需要注意不要dump过多数据，并且及时删除无用的dump数据。


#### NPU数据dump<a name="ZH-CN_TOPIC_0000002175060326"></a>

参考[NPU数据dump](#npu数据dump)执行，该数据默认保存在“precision_data/npu/debug_0”目录下。


#### 解析dump数据<a name="ZH-CN_TOPIC_0000002174900614"></a>

1.  执行命令将原始的dump二进制数据文件解析为numpy可读取的npy文件。

    ```bash
    find precision_data/npu/debug_0 -type f -name "*" | xargs -i python3 /usr/local/Ascend/ascend-toolkit/latest/tools/operator_cmp/compare/msaccucmp.py convert -d {} -out dump_data_npy/ -v 2
    ```

    该命令将“precision_data/npu/debug_0”目录下的文件通过msaccucmp.py脚本进行格式转换，并保存到dump_data_npy目录下。其中“/usr/local/Ascend/ascend-toolkit/”目录为CANN安装目录，可根据实际修改，msaccucmp.py脚本命令使用可参考《CANN 精度调试工具用户指南》中“dump数据文件Format转换”章节。

2.  从npy文件中查找NaN来源。

    转换后的npy数据保存在“dump_data_npy”目录下，包含网络中所有算子的输入和输出数据，文件按照“.”分割后的5个位置是时间戳，将所有文件按照时间戳从小到大排序，依次判断文件中是否存在NaN，从中找到最先出现NaN的数据文件，如果存在则打印文件名称，并终止循环；如果没有打印，说明文件中不存在NaN，需要检查dump步骤是否正确执行。文件的命名格式说明可参考《CANN 精度调试工具用户指南》中“数据格式要求”章节。

    执行**python3 find\_nan.py**命令，find_nan.py内容如下：

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


#### 问题来源定界<a name="ZH-CN_TOPIC_0000002210421053"></a>

找到最先出现NaN的算子dump数据文件后，根据情况判断问题根源：

1.  如果NaN出现在某个算子的输出数据中，取对应算子的输入数据，在CPU下用相同的算子逻辑执行，得到CPU算子输出，如果CPU算子输出和NPU算子输出不匹配，那么说明NaN是该算子错误执行产生，可以定界为该算子问题；如果CPU算子输出和NPU算子输出匹配，那么说明NaN是算子正常执行产生，需要继续向上找到该算子的上游算子，判断是否上游算子存在执行问题或其他问题；
2.  如果NaN出现在某个算子的输入数据中，需要继续向上找该算子的上游算子，找到NaN的来源；如果上游算子的输出正常，而当前算子的输入异常，那么定界为上游算子执行到当前算子的间隔时间中出现了内存踩踏；否则继续找上游算子是否存在其他问题。
3.  如果NaN出现在embedding variable中，那么定界为Rec SDK TensorFlow查表存在问题。



### 随机误差定位<a name="ZH-CN_TOPIC_0000002210306665"></a>

#### 定位思路<a name="ZH-CN_TOPIC_0000002175060330"></a>

在NPU训练过程中偶现误差导致精度不达标的场景，例如在某一次训练中的某一轮迭代步数出现loss突变，由于无法稳定复现，并且dump数据的时间消耗和内存占用较高，难以通过dump数据的方案进行精度比对，可以采取对比模型文件的思路进行排查。对比发生异常时的模型文件和正常训练的模型文件中的所有变量，找到余弦相似度最低的变量，如果其余弦相似度低于一定值后，例如0.98，可以认为问题由该输出该变量的算子引入。


#### 模型训练和保存<a name="ZH-CN_TOPIC_0000002174900618"></a>

按照[去除固定随机性](#去除固定随机性)（可不开启确定性计算）执行后，尽量缩小模型的保存步数间隔，例如间隔5步保存一次，保证能够在合理的时间内复现精度异常问题，取得loss异常发生后的最近一步的模型文件，和正常训练时相同步数的模型文件进行对比。


#### 模型对比<a name="ZH-CN_TOPIC_0000002210421057"></a>

修改ckpt_compare.py文件（内容如下面代码块所示）中的checkpoint_path1为异常的模型文件路径，checkpoint_path2为正常的模型文件路径后，在TensorFlow  1.15环境中执行如下python ckpt_compare.py脚本后，会按照余弦相似度从低到高输出变量名和余弦相似度。

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

![](../../figures/tf_rec_v1/zh-cn_image_0000002175060386.png)

#### 问题来源定界<a name="ZH-CN_TOPIC_0000002210306669"></a>

1.  找到第一个变量名，如果该变量的余弦相似度较低，那么可以认为精度问题由输出为该变量的算子引入。
2.  如果该算子为融合算子，表明是由于算子融合导致精度问题，可以关闭该融合后重新进行精度比对，确定是否还存在其他问题；
3.  如果定位到该变量涉及的算子和Rec SDK TensorFlow查表组件相关，表明是Rec SDK TensorFlow查表存在精度问题。

### 相关参考<a name="ZH-CN_TOPIC_0000002175060334"></a>

[精度调优流程](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/800/migration/tfmigr1/tfmigr1_tfprecision_0002.html)

[precision\_tool命令参考](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/800/migration/tfmigr1/tfmigr1_tfprecision_0050.html)

[整网精度比对结果文件说明](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/800/migration/tfmigr1/tfmigr1_tfprecision_0051.html)

## Rec SDK TensorFlow迁移样例<a name="ZH-CN_TOPIC_0000001941855729"></a>

Rec SDK TensorFlow支持TensorFlow开源推荐模型迁移适配，迁移步骤可以参考如下：

-   [DLRM样例](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/tf_rec_v2_examples/DLRM/model)
-   [LittleDemo样例](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/tf_rec_v2_examples/little_demo)
