# xDeepFM迁移样例

## 模型参考开源链接

1. https://github.com/Leavingseason/xDeepFM

2. Commits on Oct 15, 2018，提交的SHA-1 hash值（提交ID）：114c4c45b1cb6144b2540f92a2b357c3f445e98e

3. 只保留执行所需要的代码及文件，其他已删除。
4. config/network.yaml配置文件，data/dnn/infer.userid.txt、res/infer.userid.txt等数据文件由用户从开源链接下载导入

## 迁移NPU

请参照昇腾社区CANN商用版文档先使用迁移工具进行NPU自动迁移：https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldev/tfmigr1/tfmigr1_000009.html


## 迁移Rec SDK

1、修改IO/iterator.py，把第30~41行


```python
        _fm_feat_indices, _fm_feat_values,
        _fm_feat_shape, _labels, _dnn_feat_indices,
        _dnn_feat_values, _dnn_feat_weights, _dnn_feat_shape = iterator.get_next()
        self.initializer = iterator.initializer
        self.fm_feat_indices = _fm_feat_indices
        self.fm_feat_values = _fm_feat_values
        self.fm_feat_shape = _fm_feat_shape
        self.labels = _labels
        self.dnn_feat_indices = _dnn_feat_indices
        self.dnn_feat_values = _dnn_feat_values
        self.dnn_feat_weights = _dnn_feat_weights
        self.dnn_feat_shape = _dnn_feat_shape
```
` ` ` `改为：
```python
        batch = iterator.get_next()
        self.initializer = iterator.initializer
        self.fm_feat_indices = batch.get('fm_feat_indices')
        self.fm_feat_values = batch.get('fm_feat_values')
        self.fm_feat_shape = batch.get('fm_feat_shape')
        self.labels = batch.get('labels')
        self.dnn_feat_indices = batch.get('dnn_feat_indices')
        self.dnn_feat_values = batch.get('dnn_feat_values')
        self.dnn_feat_weights = batch.get('dnn_feat_weights')
        self.dnn_feat_shape = batch.get('dnn_feat_shape')
```

` ` ` `第63~65行
```python
        return fm_feat_indices, fm_feat_values,
               fm_feat_shape, labels, dnn_feat_indices,
               dnn_feat_values, dnn_feat_weights, dnn_feat_shape
```
` ` ` `改为：
```python
        return {
            'fm_feat_indices': fm_feat_indices, 'fm_feat_values': fm_feat_values, 'fm_feat_shape': fm_feat_shape,
            'labels': labels, 'dnn_feat_indices': dnn_feat_indices, 'dnn_feat_values': dnn_feat_values,
            'dnn_feat_weights': dnn_feat_weights, 'dnn_feat_shape': dnn_feat_shape
        }
```

2、修改src/base_model.py。把embedding初始化值设成tf.zeros_initializer()，把84行
```python
        return tf.truncated_normal_initializer(stddev=hparams.init_value)
```
` ` ` `改为（为了对比CPU，xDeepFM源代码这里也要一起修改）：
```python
        return tf.zeros_initializer()
```

` ` ` `更新自动改图模式下生成新数据集中batch的label记录，把188~189行
```python
    def eval(self, sess):
        return sess.run([self.loss, self.data_loss, self.pred, self.iterator.labels], \
```
` ` ` `改为：
```python
    def eval(self, sess, eval_label):
        return sess.run([self.loss, self.data_loss, self.pred, eval_label], \
```

3、修改src/exDeepFM.py。在第6行添加
```python
from mx_rec.core.embedding import create_table
from mx_rec.core.embedding import sparse_lookup
```
` ` ` `把40~43行
```python
        w_fm_nn_input_orgin = tf.nn.embedding_lookup_sparse(self.embedding,
                                                            fm_sparse_index,
                                                            fm_sparse_weight,
                                                            combiner="sum")
```
` ` ` `改为：
```python
        dense_indices = tf.sparse.to_dense(fm_sparse_index, default_value=0)
        dense_weights = tf.sparse.to_dense(fm_sparse_weight, default_value=0)
        
        sparse_hashtable = create_table(key_dtype=tf.int32,
                                        dim=tf.TensorShape([hparams.dim]),
                                        name='sparse_embeddings_table',
                                        emb_initializer=tf.zeros_initializer(),
                                        device_vocabulary_size=hparams.FEATURE_COUNT,
                                        host_vocabulary_size=0
                                        )
        embedded_values = sparse_lookup(sparse_hashtable,
                                        dense_indices,
                                        is_train=True,
                                        name="sparse_embeddings",
                                        modify_graph=True)
        w_fm_nn_input_orgin = tf.reduce_sum(embedded_values * tf.expand_dims(dense_weights, axis=-1), axis=1)
```

4、修改main.py。在第176行添加
```python
    # init
    from mx_rec.util.initialize import init
    init(use_dynamic=True,
         use_dynamic_expansion=False)
```

5、修改train.py。在第11行添加
```python
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.graph.modifier import modify_graph_and_start_emb_cache
```
把第35~57行
```python
    graph = tf.Graph()
    with graph.as_default():
        # feed train file name, valid file name, or test file name
        filenames = tf.placeholder(tf.string, shape=[None])
        #src_dataset = tf.contrib.data.TFRecordDataset(filenames)
        src_dataset = tf.data.TFRecordDataset(filenames)
    
        if hparams.data_format == 'ffm':
            batch_input = FfmIterator(src_dataset)
        elif hparams.data_format == 'din':
            batch_input = DinIterator(src_dataset)
        elif hparams.data_format == 'cccfnet':
            batch_input = CCCFNetIterator(src_dataset)
        else:
            raise ValueError("not support {0} format data".format(hparams.data_format))
        # build model
        model = model_creator(
            hparams,
            iterator=batch_input,
            scope=scope)
    
    return TrainModel(
        graph=graph,
```
` ` ` `改为：
```python
    # feed train file name, valid file name, or test file name
    filenames = tf.placeholder(tf.string, shape=[None])
    # src_dataset = tf.contrib.data.TFRecordDataset(filenames)
    src_dataset = tf.data.TFRecordDataset(filenames)
    
    if hparams.data_format == 'ffm':
        batch_input = FfmIterator(src_dataset)
    elif hparams.data_format == 'din':
        batch_input = DinIterator(src_dataset)
    elif hparams.data_format == 'cccfnet':
        batch_input = CCCFNetIterator(src_dataset)
    else:
        raise ValueError("not support {0} format data".format(hparams.data_format))
    # build model
    model = model_creator(
        hparams,
        iterator=batch_input,
        scope=scope)
    
    return TrainModel(
        graph=tf.get_default_graph(),
```
` ` ` `把第68~73行
```python
    load_sess.run(load_model.iterator.initializer, feed_dict={load_model.filenames: [filename]})
    preds = []
    labels = []
    while True:
        try:
            _, _, step_pred, step_labels = load_model.model.eval(load_sess)
```
` ` ` `改为：
```python
    eval_label = ConfigInitializer.get_instance().train_params_config.get_target_batch(True).get("labels")
    initializer = ConfigInitializer.get_instance().train_params_config.get_initializer(True)
    load_sess.run(initializer, feed_dict={load_model.filenames: [filename]})
    preds = []
    labels = []
    while True:
        try:
            _, _, step_pred, step_labels = load_model.model.eval(load_sess, eval_label)
```

` ` ` `在第223行添加
```python
    modify_graph_and_start_emb_cache(dump_graph=True)
```
` ` ` `把第239行
```python
        train_sess.run(train_model.iterator.initializer, feed_dict={train_model.filenames: [hparams.train_file_cache]})
```
` ` ` `改为：
```python
        initializer = ConfigInitializer.get_instance().train_params_config.get_initializer(True)
        train_sess.run(initializer, feed_dict={train_model.filenames: [hparams.train_file_cache]})
```
6、为了适配Rec SDK运行环境，添加了run.sh。

## 适配其他代码

1、修改utils/util.py。把第63行


```python
            config = yaml.load(f)
```
` ` ` `改为（为了xDeepFM源代码在CPU上能跑通，这里也要一起修改）：
```python
             config = yaml.safe_load(f)
```

2、由于去掉了无关代码src/CIN.py，修改main.py适配。把第156~158行

```python
                                             'opnn', 'fm', 'lr', 'din', 'cccfnet', 'deepcross', 'exDeepFM', "cross", "CIN"]:
        raise ValueError(
            "model type must be cccfnet, deepFM, deepWide, dnn, ipnn, opnn, fm, lr, din, deepcross, exDeepFM, cross, CIN but you set is {0}".format(
```
` ` ` `改为：
```python
                                             'opnn', 'fm', 'lr', 'din', 'cccfnet', 'deepcross', 'exDeepFM', "cross"]:
        raise ValueError(
            "model type must be cccfnet, deepFM, deepWide, dnn, ipnn, opnn, fm, lr, din, deepcross, exDeepFM, "
            "cross, but you set is {0}".format(config['model']['model_type']))
```

` ` ` `修改train.py适配。删除第21行代码
```python
from src.CIN import CINModel
```

` ` ` `删除第210~212行代码
```python
    elif hparams.model_type == 'CIN':
        print("run extreme cin model!")
        model_creator = CINModel
```

## 运行命令
```shell
bash run.sh main.py 10.10.10.10
```
其中，10.10.10.10为服务器IP，请替换成对应服务器IP。

## 验证结果
1、CPU:
```log
step 1 , total_loss: 0.6931, data_loss: 0.6931
step 2 , total_loss: 0.6905, data_loss: 0.6905
finish one epoch!
at epoch 0 train info: loss:0.6918214857578278 eval info: auc:0.4867, logloss:0.6865 test info: auc:0.4867, logloss:0.6865
at epoch 0 , train time: 0.6 eval time: 0.3
step 1 , total_loss: 0.6845, data_loss: 0.6845
step 2 , total_loss: 0.6818, data_loss: 0.6818
finish one epoch!
at epoch 1 train info: loss:0.6831814646720886 eval info: auc:0.485, logloss:0.6801 test info: auc:0.485, logloss:0.6801
at epoch 1 , train time: 0.2 eval time: 0.1
step 1 , total_loss: 0.6766, data_loss: 0.6766
step 2 , total_loss: 0.6732, data_loss: 0.6732
finish one epoch!
at epoch 2 train info: loss:0.6748818755149841 eval info: auc:0.4832, logloss:0.6738 test info: auc:0.4832, logloss:0.6738
at epoch 2 , train time: 0.1 eval time: 0.1
```
2、Rec SDK:
```log
[1,0]<stdout>:step 1 , total_loss: 0.6931, data_loss: 0.6931
[1,0]<stdout>:step 2 , total_loss: 0.6905, data_loss: 0.6905
[1,0]<stdout>:finish one epoch!
[1,0]<stdout>:at epoch 0 train info: loss:0.6918215453624725 eval info: auc:0.4867, logloss:0.6865 test info: auc:0.4867, logloss:0.6865
[1,0]<stdout>:at epoch 0 , train time: 15.9 eval time: 3.1
[1,0]<stdout>:step 1 , total_loss: 0.6845, data_loss: 0.6845
[1,0]<stdout>:step 2 , total_loss: 0.6818, data_loss: 0.6818
[1,0]<stdout>:finish one epoch!
[1,0]<stdout>:at epoch 1 train info: loss:0.6831814646720886 eval info: auc:0.485, logloss:0.6801 test info: auc:0.485, logloss:0.6801
[1,0]<stdout>:at epoch 1 , train time: 7.8 eval time: 0.7
[1,0]<stdout>:step 1 , total_loss: 0.6766, data_loss: 0.6766
[1,0]<stdout>:step 2 , total_loss: 0.6732, data_loss: 0.6732
[1,0]<stdout>:finish one epoch!
[1,0]<stdout>:at epoch 2 train info: loss:0.6748818457126617 eval info: auc:0.4832, logloss:0.6738 test info: auc:0.4832, logloss:0.6738
[1,0]<stdout>:at epoch 2 , train time: 0.5 eval time: 0.7
```
