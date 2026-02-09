# 快速入门<a name="ZH-CN_TOPIC_0000001580166524"></a>

## 使用前说明<a name="ZH-CN_TOPIC_0000001848119673"></a>

本章节指导用户基于Rec SDK TensorFlow为用户提供的little-demo样例，快速理解一个使用tf.Session进行模型训练需要准备的相关文件和关键接口适配。little-demo仅是一个代码示例，并介绍了调用相关接口的逻辑，不包含具体的模型，没有实现具体的功能。

little-demo仅作参考学习，不支持在little-demo上适配用户自己的模型。little-demo存放路径为：[链接](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/tf_rec_v2_examples/little_demo)。

**表 1**  little-demo文件说明

| 文件名              | 说明                             |
|------------------|--------------------------------|
| config.py        | 训练相关配置。                        |
| dataset.py       | 数据集生成。                         |
| main.py          | 模型训练入口。                        |
| model.py         | 模型搭建。                          |
| op_precision.ini | 算子配置文件。                        |
| runner.py        | 训练、推理流程封装。                     |
| logger.py        | 日志封装。                          |
| demo.toml        | Rec SDK TensorFlow训练框架和模型配置文件。 |
| run.sh           | 模型训练启动脚本。                      |

## 接口调用介绍<a name="ZH-CN_TOPIC_0000001801480550"></a>

1.  初始化框架。在`main.py`中调用`init`接口，传入初始化框架需要的相关参数。相关参数请参见[接口说明](./api/initialization_of_the_training_framework.md#inita-namezh-cn_topic_0000001630046449a)。

    ```python
    mxrec.init(params.path)
    ```

2.  建立稀疏表。在`runner.py`中调用`get_embedding_table`接口，建立稀疏表，创建稀疏网络层。相关参数请参见[接口说明](./api/model_apis.md#get_embedding_tablea-namezh-cn_topic_0000001630246521a)。

    ```python
    user_table = mxrec.get_embedding_table(
        name="user_table",
        dimension=Config.user_hashtable_dim,
        device_vocabulary_size=Config.user_vocab_size,
        initializer=tf.truncated_normal_initializer(-0.01, 0.01, Config.random_seed),
    )
    item_table = mxrec.get_embedding_table(
        name="item_table",
        dimension=Config.item_hashtable_dim,
        device_vocabulary_size=Config.item_vocab_size,
        initializer=tf.truncated_normal_initializer(-0.01, 0.01, Config.random_seed),
    )
    ```

3.  建立前向计算图。传入稀疏网络层和特征列表，创建模型计算图，在`runner.py`中调用`embedding_lookup`进行特征查询和误差计算。相关参数请参见[接口说明](./api/model_apis.md#embedding_lookupa-namezh-cn_topic_0000001630246521a)。

    ```python
    embedding_list = []
    for table, ids in zip([user_table, item_table], [batch.get(Config.user_ids), batch.get(Config.item_ids)]):
        embedding = mxrec.embedding_lookup(table, ids)
        reduced_embedding = tf.reduce_sum(embedding, axis=1, keepdims=False)
        embedding_list.append(reduced_embedding)

    model = Model()
    model(embedding_list, batch.get(Config.label_0), batch.get(Config.label_1))
    ```

4.  定义优化器。在`runner.py`中定义优化器，支持的优化器类型和相关参数请参见[优化器](./api/optimizers_apis.md)。

    ```python
    sparse_optimizer = mxrec.AdamWOptimizer(learning_rate=Config.learning_rate)
    ```

5.  定义梯度计算和优化过程。在`runner.py`中调用`get_sparse_embedding`接口，得到稀疏网络层的参数，通过优化器计算梯度并执行优化。接口说明请参见[接口说明](./api/model_apis.md#get_sparse_embeddinga-namezh-cn_topic_0000001630246521a)。

    ```python
    def _get_train_ops(train_model: Model) -> List[tf.Tensor]:
        train_ops = []

        # Do dense optimization.
        dense_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        dense_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=Config.learning_rate)
        grads = dense_optimizer.compute_gradients(train_model.loss, var_list=dense_variables)
        avg_grads = []
        for grad, var in grads:
           if Config.rank_size > 1:
               grad = hccl_ops.allreduce(grad, "sum") if grad is not None else None
           if grad is not None:
               avg_grads.append((grad, var))
        train_ops.append(dense_optimizer.apply_gradients(avg_grads))
        
        # Do sparse optimization.
        sparse_optimizer = mxrec.AdamWOptimizer(learning_rate=Config.learning_rate)
        sparse_embeddings = mxrec.get_sparse_embedding()
        sparse_grads = tf.gradients(train_model.loss, sparse_embeddings)
        train_ops.append(sparse_optimizer.apply_gradients(zip(sparse_grads, sparse_embeddings)))
        
        return train_ops
    ```

6.  启动Session计算并在训练过程中保存模型。在`runner.py`中调用`EmbeddingTableSaver`接口，启动Session计算并在训练过程中保存模型。接口说明请参见[接口说明](./api/model_apis.md#embeddingtablesavera-namezh-cn_topic_0000001630246521a)。

    ```python
    def _train_and_evaluate(self, saved_path: str):
        train_model, train_iterator = self._model_forward()
        eval_model, eval_iterator = self._model_forward()
    
        train_ops = _get_train_ops(train_model)
        init_hashtable_op = mxrec.get_init_hashtable_op()
        self._sess.run(init_hashtable_op)
        self._sess.run(train_iterator.initializer)
        self._sess.run(tf.compat.v1.global_variables_initializer())
    
        saver = tf.compat.v1.train.Saver()
        tf_save_path = os.path.join(saved_path, Config.ckpt_name)
        embedding_table_saver = mxrec.EmbeddingTableSaver(mxrec.get_existing_tables())
    
        for i in range(self._train_steps):
            logger.info("################    training at step %d    ################", i + 1)
            try:
                _, loss = self._sess.run([train_ops, train_model.loss])
                logger.info("Training loss: %s.", loss)
            except tf.errors.OutOfRangeError:
                logger.info("Encounter the end of Sequence for training.")
                break
    
            if (i + 1) % self._train_interval == 0:
                self._evaluate(eval_iterator, eval_model)
    
                saver.save(self._sess, tf_save_path, global_step=i + 1)
                embedding_table_saver.save(self._sess, saved_path, i + 1)
    
                logger.info("The saved path: %s.", saved_path)
    ```

## 启动模型训练<a name="ZH-CN_TOPIC_0000002255637701"></a>

### 单机单卡和单机多卡训练<a name="ZH-CN_TOPIC_0000001801320778"></a>

本章节介绍通过环境变量设置资源信息，启动训练任务，包含单机单卡和单机多卡场景。

**前提条件<a name="section16252115164"></a>**

使用该方案启动训练任务，需要设置如下环境变量。详细的配置环境变量的方法可参考[little-demo的启动脚本](https://gitcode.com/Ascend/RecSDK/blob/develop_examples_and_tools/tf_rec_v2_examples/little_demo/run.sh)；关于环境变量的说明可参见[配置环境变量](recsdk_tf_installation_guide.md#配置环境变量a-namezh-cn_topic_0000001580326424a)。 

**启动训练<a name="section16252115164"></a>**

模型训练启动命令：
```bash
bash run.sh
```

训练结束后，会打印`Demo done.`字样。
