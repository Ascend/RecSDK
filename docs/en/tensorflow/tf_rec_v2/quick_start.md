# Quick Start

## Before You Start

This document guides you through the related files and key interface adaptations required for model training using `tf.Session`, based on the `little-demo` sample provided by Rec SDK TensorFlow. `little-demo` is a code example that introduces the logic of calling related interfaces. It does not contain specific models or implement specific functions.

`little-demo` is for reference and learning only. Adapting your own models on `little-demo` is not supported. You can find `little-demo` at the following [link](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/tf_rec_v2_examples/little_demo).

**Table 1** little-demo file description

| File             | Description                            |
|------------------|--------------------------------|
| config.py        | Training-related configurations                       |
| dataset.py       | Dataset generation                        |
| main.py          | Entry point for model training                       |
| model.py         | Model construction                         |
| op_precision.ini | Operator configuration file                       |
| runner.py        | Encapsulation of training and inference processes                    |
| logger.py        | Log encapsulation                         |
| demo.toml        | Rec SDK TensorFlow training framework and model configuration file|
| run.sh           | Startup script for model training                     |

## Interface Call Introduction

1. Initialize the framework. In `main.py`, call the `init` interface and pass the parameters required for framework initialization. For details about the parameters, see the [API description](./api/initialization_of_the_training_framework.md).

    ```python
    mxrec.init(params.path)
    ```

2. Create a sparse table. In `runner.py`, call `get_embedding_table` to create a sparse table and a sparse network layer. For details about the parameters, see the [API description](./api/model_apis.md#get_embedding_table).

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

3. Create a forward computational graph. Pass the sparse network layer and feature list to create the model computational graph. Call `embedding_lookup` in `runner.py` for feature lookup and error calculation. For details about the parameters, see the [API description](./api/model_apis.md#embedding_lookup).

    ```python
    embedding_list = []
    for table, ids in zip([user_table, item_table], [batch.get(Config.user_ids), batch.get(Config.item_ids)]):
        embedding = mxrec.embedding_lookup(table, ids)
        reduced_embedding = tf.reduce_sum(embedding, axis=1, keepdims=False)
        embedding_list.append(reduced_embedding)

    model = Model()
    model(embedding_list, batch.get(Config.label_0), batch.get(Config.label_1))
    ```

4. Define the optimizer. Define the optimizer in `runner.py`. For supported optimizer types and related parameters, see [Optimizers](./api/optimizers_apis.md).

    ```python
    sparse_optimizer = mxrec.AdamWOptimizer(learning_rate=Config.learning_rate)
    ```

5. Define gradient calculation and the optimization process. In `runner.py`, call `get_sparse_embedding` to obtain parameters for the sparse network layer. Calculate gradients and perform optimization through the optimizer. For details about the API, see the [API description](./api/model_apis.md#get_sparse_embedding).

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

6. Start `Session` calculation and save the model during training. In `runner.py`, call `EmbeddingTableSaver` to start `Session` calculation and save the model during training. For details about the API, see the [API description](./api/model_apis.md#embeddingtablesaver).

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

## Model Training Startup

### Single-Node Single-Card and Single-Node Multi-Card Training

This section describes how to set resource information through environment variables to start training tasks, including single-node single-card and single-node multi-card scenarios.

**Prerequisites**

To start a training task using this solution, set the following environment variables. For details about how to configure environment variables, see the [startup script for `little-demo`](https://gitcode.com/Ascend/RecSDK/blob/develop_examples_and_tools/tf_rec_v2_examples/little_demo/run.sh). For descriptions of environment variables, see [Configuring environment variables](recsdk_tf_installation_guide.md#configuring-environment-variables).

**Starting Training**

Run the following command to start model training:

```bash
bash run.sh
```

After the training is complete, `Demo done.` is displayed.
