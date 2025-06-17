# wide&deep模型 迁移样例（基于DLRM模型框架）

开源项目在保证原有结构不变的情况下，可采用替换相关API接口的方式将项目由GPU >> NPU >> Rec SDK。在模型迁移适配过程中可能因兼容性问题而导致模型迁移失败，此处提供另一种模型适配方案。  

***
## 开源项目链接
Commits on Apr 29, 2022, 提交的SHA-1 hash值（提交ID）：4bbfb492b872c5a3290a2bce1ed5c160162558a3
commit的链接: https://github.com/ZiyaoGeng/RecLearn/tree/4bbfb492b872c5a3290a2bce1ed5c160162558a3
```shell
https://github.com/ZiyaoGeng/RecLearn
```
***
## 数据集

```shell
Criteo4500w数据集: 如下链接中的Kaggle Display Advertising dataset
https://ailab.criteo.com/ressources/
```
***
## 数据集预处理

### 解压文件列表
- train.txt
- test.txt
- readme.txt

text.txt因缺少label列无法使用，将train.txt数据集切分为10份，train_01.txt~train_09.txt为训练集，train_10.txt为测试集。数据预处理文件：criteo.py。  

***
### 数据预处理运行脚本
```shell
python critro.py --data_path data_path --output_path output_path 
```
参数说明：
- dataset_path: train.txt的路径，如："D:\dat\train.txt"
- output_path: tfrecord存放路径，如："D:\dat\tfrecord\ "
***

### criteo.py
#### 1. 分割数据集
调用`criteo.py`文件中的`get_split_file_path(parent_path, dataset_path, sample_num=4600000)`方法将数据集分割，`sample_num=4600000`是每个子数据集的样本数量。返回包含全部子数据集名称的列表。

```python
# get txt_list
file_split_list = get_split_file_path(dataset_path=data_path)
```
***
#### 2. 建立特征映射
调用`criteo.py`文件中的`get_fea_map()`方法，以`{'C1':{}, 'C2':{},..., 'I1':{},...}`形式储存dense_feature的最大最小值以及sparse_feature去重后的特征映射。

```python
# get feature_map
feature_map = get_fea_map(split_file_list=file_split_list)
```
***
#### 3. dense_feature分桶离散化
调用`criteo.py`文件中的`rec_kbins_discretizer(data_df, n_bins, min_max_dict)`方法将dense_feature分桶化离散化，`nbins=1000`。

```python
# dense feature: Bin continuous data into intervals.
data_df[dense_features] = rec_kbins_discretizer(data_df[dense_features], 1000, feature_map)
```
***
#### 4. sparse_feature特征映射
通过如下操作将原始的字符串数据映射为0~max的int64数据。

```python
# sparse feature: mapping
for col in sparse_features:
    try:
        data_df[col] = data_df[col].map(lambda x: feature_map[col][x])
    except KeyError as e:
        raise KeyError("Feature {} not found in dataset".format(col)) from e
```
***
#### 5. 39个特征增加偏移项
开源项目deep部分对39个特征分别作了embedding，即建了39个表。本项目只建了一张表，因此需要对每个特征对应的值作偏移。`slot_size_array`中的值分别对应各特征去重后的类别数。 

```python
# add offsets
slot_size_array = [
                1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001,
                1462, 585, 10131228, 2202609, 307, 25, 12519, 635, 5, 93147, 5685, 8351594, 3196,
                29, 14994, 5461307, 12, 5654, 2174, 5, 7046548, 19, 17, 286182, 106, 142573
]
offset_size_list = np.cumsum([0] + slot_size_array[:-1])
for col_index in range(1, len(offset_size_list) + 1):
    data_df.iloc[:, col_index] += offset_size_list[col_index - 1]
```
***
#### 6. 数据集格式转换：txt >> tfrecord
调用`criteo.py`文件中的`convert_input2tfrd(in_file_path, out_file_path)`方法将txt文件转换为tfrecord文件。

```python
# txt to tfrecords
convert_input2tfrd(in_file_path=file, out_file_path=output_path)
```
***

## 模型运行

参考Rec SDK的`README.md`文件在NPU服务器上配置环境并安装镜像创建容器后，可参考DLRM模型运行命令启动模型训练。模型运行脚本是run.sh，运行此脚本需要四个参数：so_path、rec_package_path、hccl_cfg_json以及dlrm_criteo_data_path。其中，   
- so_path: Rec SDK中libasc所在路径，在镜像中已经安装过Rec SDK，所以so_path是：/usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec/libasc/
- rec_package_path: Rec SDK这个包的安装路径，镜像中是：/usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec/  
- hccl_cfg_json:  hccl配置文件所在路径，一般是当前路径下的hccl文件
- dlrm_criteo_data_path: Wide&Deep模型需要的数据所在路径，根据实际情况进行配置

运行Rec SDK有两种方式，一种是使用hccl配置文件（rank table方案），一种是不使用hccl配置文件（去rank table方案）。
- 使用hccl配置文件（rank table方案）
```shell
bash run.sh {so_path} {rec_package_path} {hccl_cfg_json} {dlrm_criteo_data_path}
```
***
- 不使用hccl配置文件（去rank table方案）
```shell
bash run.sh {so_path} {rec_package_path} {hccl_cfg_json} {dlrm_criteo_data_path} {IP}
```
如：bash run.sh /usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec/libasc/ /usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec/ hccl_json_8p.json /dataset 10.10.10.10。  
**注意:** 去rank table方案，当前路径下不存在hccl文件，模型仍可正常运行。


## 模型结果
[开源项目](https://github.com/ZiyaoGeng/RecLearn)使用Criteo4500W数据集在GPU上训练模型，结果为`Log Loss=0.4692`、`AUC=0.7930`。适配完成模型后，固定`CACHE_MODE="HBM"`、`USE_FAAE=0`，在`run.sh`中配置其他选项卡，运行结果如下。

<table style="text-align:center;margin:auto">
  <tr></tr>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="4">Options</th>
    <th colspan="2">Criteo4500W</th>
  </tr>
  <tr>
    <th>Use_Dynamic</th>
    <th>Use_Dynamic_Expansion</th>
    <th>Use_Multi_Lookup</th>
    <th>Use_Modify_Graph</th>
    <th>Log Loss</th>
    <th>AUC</th>
  </tr>
  <tr><td> WDL </td><td> 0 </td><td> 0 </td><td> 0 </td><td> 0 </td><td> 0.4592 </td><td> 0.7934 </td></tr>
  <tr><td> WDL </td><td> 0 </td><td> 1 </td><td> 0 </td><td> 0 </td><td> 0.4593 </td><td> 0.7933 </td></tr>
  <tr><td> WDL </td><td> 1 </td><td> 0 </td><td> 0 </td><td> 0 </td><td> 0.4594 </td><td> 0.7932 </td></tr>
  <tr><td> WDL </td><td> 1 </td><td> 1 </td><td> 0 </td><td> 0 </td><td> 0.4594 </td><td> 0.7932 </td></tr>
  <tr><td> WDL </td><td> 1 </td><td> 1 </td><td> 1 </td><td> 0 </td><td> 0.4590 </td><td> 0.7937 </td></tr>
  <tr><td> WDL </td><td> 0 </td><td> 0 </td><td> 0 </td><td> 1 </td><td> 0.4593 </td><td> 0.7934 </td></tr>
  <tr><td> WDL </td><td> 0 </td><td> 1 </td><td> 0 </td><td> 1 </td><td> 0.4593 </td><td> 0.7933 </td></tr>
  <tr><td> WDL </td><td> 1 </td><td> 0 </td><td> 0 </td><td> 1 </td><td> 0.4593 </td><td> 0.7933 </td></tr>
  <tr><td> WDL </td><td> 1 </td><td> 1 </td><td> 0 </td><td> 1 </td><td> 0.4594 </td><td> 0.7932 </td></tr>
  <tr><td> WDL </td><td> 1 </td><td> 1 </td><td> 1 </td><td> 1 </td><td> 0.4589 </td><td> 0.7937 </td></tr>
</table>


***
## 模型迁移

**迁移思路：** 在现有已适配好的dlrm模型框架下，改动相关代码逻辑，完成Wide&deep模型的适配。**核心：根据开源项目model代码修改`model.py`；数据处理操作一部分放入`criteo.py`,一部分放入`main_mxrec.py`中`make_batch_and_iterator()`内；`main_mxrec.py`中其他相关代码改动主要是为了适配Rec SDK提供的相关特性。**
详细改动见[https://gitee.com/ascend/mxrec/pulls/171/commits](https://gitee.com/ascend/mxrec/pulls/171/commits)，Commits ID：7a05b033d41af51df9aed7414ad04216dff821cc。  
下文所提到的`动态扩容`、`动态shape`、`自动改图`、`一表多查`是Rec SDK提供的相关特性，开关选项见`run.sh`。

```shell
# run.sh: 32~37行
export USE_DYNAMIC=0            # 0：静态shape；1：动态shape
export CACHE_MODE="HBM"         # HBM；DDR；SSD
export USE_FAAE=0               # 0：关闭准入淘汰；1：开启准入淘汰
export USE_DYNAMIC_EXPANSION=0  # 0：关闭动态扩容；1: 开启动态扩容
export USE_MULTI_LOOKUP=0       # 0：一表一查；1：一表多查
export USE_MODIFY_GRAPH=0       # 0：feature spec模式；1：自动改图模式
```

***
### DLRM模型框架  
**迁移说明：** 迁移过程中未使用`gradient_descent_w.py`、`mean_auc.py`。

- config.py
- delay_loss_scale.py
- demo_logger.py
- gradient_descent_w.py
- main_mxrec.py
- mean_auc.py
- model.py
- optimizer.py
- run.sh

***

### 代码改动说明
#### 1. config.py
实验超参数配置如下：取消动态学习率逻辑，学习率固定为0.001。

```python
# 88~89行
lr_sparse = self.base_lr_sparse * lr_factor_constant
lr_dense = self.base_lr_dense * lr_factor_constant
# 140~146行
_lr_scheduler = LearningRateScheduler(
    0.001,
    0.001,
    LR_SCHEDULE_STEPS[0],
    LR_SCHEDULE_STEPS[1],
    LR_SCHEDULE_STEPS[2],
)
# 超参数
self.batch_size = 4096
self.line_per_sample = 1
self.train_epoch = 1
self.test_epoch = 9
self.emb_dim = 8           
```
***


#### 2. model.py
迁移过程中，`model.py`需参考开源项目文件`reclearn/models/ranking/wdl.py`的代码逻辑，使用tensorflow的低阶API重新编写。输出参数必须包括`loss`,`prediction`,`label`,`trainable_variables`。**迁移重点：Rec SDK对推荐模型中sparse_feature的创表查表操作作了加速，使用`create_table`与`sparse_lookup`接口替换tensorflow中的`tf.nn.embedding_lookup`接口。** 因此在适配开源项目时，会将sparse_feature的embedding操作放在模型结构外。 

**reclearn开源项目原始代码：**
```python
# wdl.py
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Dropout, Input
from tensorflow.keras.regularizers import l2

from reclearn.layers import Linear, MLP
from reclearn.layers.utils import index_mapping

class WideDeep(Model):
    def __init__(self, feature_columns, hidden_units, activation='relu',
                 dnn_dropout=0., embed_reg=0., w_reg=0.):
        """Wide&Deep.
        Args:
            :param feature_columns: A list. [{'feat_name':, 'feat_num':, 'embed_dim':}, ...]
            :param hidden_units: A list. Neural network hidden units.
            :param activation: A string. Activation function of MLP.
            :param dnn_dropout: A scalar. Dropout of MLP.
            :param embed_reg: A scalar. The regularization coefficient of embedding.
            :param w_reg: A scalar. The regularization coefficient of Linear.
        :return
        """
        super(WideDeep, self).__init__()
        self.feature_columns = feature_columns
        self.embed_layers = {
            feat['feat_name']: Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_normal',
                                         embeddings_regularizer=l2(embed_reg))
            for feat in self.feature_columns
        }
        self.map_dict = {}
        self.feature_length = 0
        for feat in self.feature_columns:
            self.map_dict[feat['feat_name']] = self.feature_length
            self.feature_length += feat['feat_num']
        self.dnn_network = MLP(hidden_units, activation, dnn_dropout)
        self.linear = Linear(self.feature_length, w_reg=w_reg)
        self.final_dense = Dense(1, activation=None)

    def call(self, inputs):
        sparse_embed = tf.concat([self.embed_layers[feat_name](value) for feat_name, value in inputs.items()], axis=-1)
        x = sparse_embed  # (batch_size, field * embed_dim)
        # Wide
        wide_inputs = index_mapping(inputs, self.map_dict)
        wide_inputs = tf.concat([value for _, value in wide_inputs.items()], axis=-1)
        wide_out = self.linear(wide_inputs)
        # Deep
        deep_out = self.dnn_network(x)
        deep_out = self.final_dense(deep_out)
        # out
        outputs = tf.nn.sigmoid(0.5 * wide_out + 0.5 * deep_out)
        return outputs

    def summary(self):
        inputs = {
            feat['feat_name']: Input(shape=(), dtype=tf.int32, name=feat['feat_name'])
            for feat in self.feature_columns
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()

```
`self.embed_layers`是对数据集中39个特征分别建表作embedding的操作，迁移后对应的代码逻辑见`main_mxrec.py`。  
`self.map_dict`统计了各特征需增加的偏移量。  
`index_mapping`是对数据增加偏移量的操作，迁移后对应的代码逻辑见`criteo.py`。

**迁移后代码：**
```python
# model.py
import time
from easydict import EasyDict as edict

import tensorflow as tf


model_cfg = edict()
model_cfg.loss_mode = "batch"
LOSS_OP_NAME = "loss"
LABEL_OP_NAME = "label"
VAR_LIST = "variable"
PRED_OP_NAME = "pred"


class MyModel:
    def __init__(self):
        self.kernel_init = None
        self._loss_fn = None
        self.is_training = None

    def build_model(self,
                    wide_embedding=None,
                    deep_embedding=None,
                    label=None,
                    is_training=True,
                    seed=None,
                    dropout_rate=None,
                    batch_norm=False):

        with tf.variable_scope("wide_deep", reuse=tf.AUTO_REUSE):
            self._loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            self.is_training = is_training

            # wide
            batch_size, wide_num, wide_emb_dim = wide_embedding.shape
            wide_input = tf.reshape(wide_embedding[:,0], shape=(batch_size, wide_num * 1))
            wide_output = tf.reshape(tf.reduce_sum(wide_input, axis=1), shape=(-1,1))

            # deep
            batch_size, deep_num, deep_emb_dim = deep_embedding.shape
            deep_input = tf.reshape(deep_embedding, shape=(batch_size, deep_num * deep_emb_dim))

            ## MLP
            hidden_units = [256,128,64]
            net = deep_input
            for i,unit in enumerate(hidden_units):

                net = tf.layers.dense(net, units=unit, activation='relu', name=f'hidden_layer_{i}',
                                      kernel_initializer=tf.glorot_uniform_initializer(seed=seed),
                                      bias_initializer=tf.zeros_initializer())

                if dropout_rate is not None and 0.0 < dropout_rate < 1.0:
                    net = tf.layers.dropout(net,dropout_rate,training=self.is_training)
                if batch_norm:
                    net = tf.layers.batch_normalization(net, training=self.is_training)
                    
            deep_output = tf.layers.dense(net, units=1, activation=None, name='deep_output',
                                          kernel_initializer=tf.glorot_uniform_initializer(seed=seed),
                                          bias_initializer=tf.zeros_initializer())
            
            total_logits = 0.5 * tf.add(wide_output,deep_output,name='total_logits')
            loss = self._loss_fn(label, total_logits)
            prediction = tf.sigmoid(total_logits)
            trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wide_deep')
            return {LOSS_OP_NAME: loss,
                    PRED_OP_NAME: prediction,
                    LABEL_OP_NAME: label,
                    VAR_LIST: trainable_variables}


my_model = MyModel()

```
***
#### 3. main_mxrec.py

`main_mxrec.py`文件中的函数如下所示。`make_batch_and_iterator()`是读取数据集以及对数据作处理的函数；`model_forward()`是前向过程函数；`evaluate()`与`evaluate_fix()`是评估函数，用于计算测试集的AUC与loss。`add_timestamp_func()`与特征准入、淘汰有关；`create_feature_spec_list()`是生成元素为FeatureSpec类的列表的函数，其返回值是`make_batch_and_iterator()`所需的传参。特征准入与淘汰、FeatureSpec类、自动改图等解释见[Rec SDK用户指南](https://www.hiascend.com/document/detail/zh/mind-sdk/60rc3/mxRec/mxrecug/mxrecug_0004.html)。  

- `add_timestamp_func()`
- `make_batch_and_iterator()`
- `model_forward()`
- `evaluate()`
- `evaluate_fix()`
- `create_feature_spec_list()`

**迁移代码改动说明：** `add_timestamp_func()`、`evaluate()`、`evaluate_fix()`未作修改。  
<br>

3.1 读取数据集：`make_batch_and_iterator()` 

```python
# main_mxrec.py：100~104行
def map_fn(batch):
    new_batch = batch
    new_batch['sparse_feature'] = tf.concat([batch['dense_feature'], batch['sparse_feature']], axis=1)
    return new_batch
dataset = dataset.map(map_fn, num_parallel_calls=num_parallel)
```
`map_fn()`：该函数是将分桶后的dense_feature与sparse_feature合并为新sparse_feature。该操作主要与`FeatureSpec()`、`sparse_lookup()`传入参数有关。

```python
# main_mxrec.py：109~118行
if not MODIFY_GRAPH_FLAG:

    # Enable EOSDataset manually.
    librec = import_host_pipeline_ops(LIBREC_EOS_OPS_SO)
    channel_id = 0 if is_training else 1
    # 此处eos_map的调用必须先于insert_func,避免多卡数据不均匀的情况
    dataset = dataset.eos_map(librec, channel_id, kwargs.get("max_train_steps", max_train_steps),
                              kwargs.get("max_eval_steps", eval_steps))
    insert_fn = get_asc_insert_func(tgt_key_specs=feature_spec_list, is_training=is_training, dump_graph=dump_graph)
    dataset = dataset.map(insert_fn)
```
`dataset.eos_map()`：该函数主要是为了解决FeatureSpec模式下开`动态shape`选项卡，训练结束无法正常退出的问题。

***
3.2 模型前向传播过程

```python
# main_mxrec.py：127~179行
def model_forward(feature_list, wide_hash_table_list, deep_hash_table_list, batch, is_train, modify_graph, is_use_faae=False):
    wide_embedding_list = []
    deep_embedding_list = []
    wide_feature_list = []
    deep_feature_list = []
    if is_use_faae:
        feature_list_copy = feature_list[:-1]
    else:
        feature_list_copy = feature_list

    for i,item in enumerate(feature_list_copy):
        if i % 2 == 0:
            wide_feature_list.append(item)
        else:
            deep_feature_list.append(item)

    logger.debug(f"In model_forward function, is_train: {is_train}, feature_list: {len(feature_list)}, "
                 f"wide_hash_table_list: {len(wide_hash_table_list)}, deep_hash_table_list: {len(deep_hash_table_list)}")

    # wide
    for wide_feature, wide_hash_table in zip(wide_feature_list, wide_hash_table_list):
        if MODIFY_GRAPH_FLAG:
            wide_feature = batch["sparse_feature"]
        wide_embedding = sparse_lookup(wide_hash_table, wide_feature, cfg.send_count, dim=None, is_train=is_train,
                                  name="wide_embedding_lookup", modify_graph=modify_graph, batch=batch,
                                  access_and_evict_config=None)
        wide_embedding_list.append(wide_embedding)

    # deep
    for deep_feature, deep_hash_table in zip(deep_feature_list, deep_hash_table_list):
        if MODIFY_GRAPH_FLAG:
            deep_feature = batch["sparse_feature"]
        deep_embedding = sparse_lookup(deep_hash_table, deep_feature, cfg.send_count, dim=None, is_train=is_train,
                                  name="deep_embedding_lookup", modify_graph=modify_graph, batch=batch,
                                  access_and_evict_config=None)
        deep_embedding_list.append(deep_embedding)

    if len(wide_embedding_list) == 1:
        wide_emb = wide_embedding_list[0]
        deep_emb = deep_embedding_list[0]
    elif len(wide_embedding_list) > 1:
        wide_emb = tf.reduce_sum(wide_embedding_list, axis=0, keepdims=False)
        deep_emb = tf.reduce_sum(deep_embedding_list, axis=0, keepdims=False)
    else:
        raise ValueError("the length of embedding_list must be greater than or equal to 1.")
    my_model = MyModel()
    model_output = my_model.build_model(wide_embedding=wide_emb,
                                        deep_embedding=deep_emb,
                                        label=batch["label"],
                                        is_training=is_train,
                                        seed=dense_hashtable_seed,
                                        dropout_rate=0.5)
    return model_output
```
该函数是前向传播函数，主要包括sparse_feature的embedding操作（查表）与model前向操作。130-141行代码是预处理`sparse_lookup`传参的逻辑。147-162行代码对应开源项目中wide部分`self.linear`与deep部分`self.embed_layers`对39个特征作embedding的逻辑。164-171行是配置Rec SDK中`一表多查`特性的逻辑。

***
3.3 创表操作

```python
# main_mxrec.py: 273~296行
def create_feature_spec_list(use_timestamp=False):
    access_threshold = None
    eviction_threshold = None
    if use_timestamp:
        access_threshold = 1000
        eviction_threshold = 180

    feature_spec_list = [FeatureSpec("sparse_feature", table_name="wide_embeddings", batch_size=cfg.batch_size,
                                     access_threshold=access_threshold, eviction_threshold=eviction_threshold),
                         FeatureSpec("sparse_feature", table_name="deep_embeddings", batch_size=cfg.batch_size,
                                     access_threshold=access_threshold, eviction_threshold=eviction_threshold)]

    if use_multi_lookup:
        feature_spec_list.extend([FeatureSpec("sparse_feature", table_name="wide_embeddings",
                                             batch_size=cfg.batch_size,
                                             access_threshold=access_threshold,
                                             eviction_threshold=eviction_threshold),
                                  FeatureSpec("sparse_feature", table_name="deep_embeddings",
                                             batch_size=cfg.batch_size,
                                             access_threshold=access_threshold,
                                             eviction_threshold=eviction_threshold)])
    if use_timestamp:
        feature_spec_list.append(FeatureSpec("timestamp", is_timestamp=True))
    return feature_spec_list

```

```python
# main_mxrec.py: 379~397行
# 创表操作
wide_emb_initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.05, seed=sparse_hashtable_seed)
deep_emb_initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.05, seed=sparse_hashtable_seed)

sparse_hashtable_wide = create_table(
    key_dtype=cfg.key_type,
    dim=tf.TensorShape([cfg.emb_dim]),
    name="wide_embeddings",
    emb_initializer=wide_emb_initializer,
    **cfg.get_emb_table_cfg()
)

sparse_hashtable_deep = create_table(
    key_dtype=cfg.key_type,
    dim=tf.TensorShape([cfg.emb_dim]),
    name="deep_embeddings",
    emb_initializer=deep_emb_initializer,
    **cfg.get_emb_table_cfg()
)
```
`create_feature_spec_list()`的返回值是`make_batch_and_iterator()`、`model_forward()`的传参;`create_table()`的返回值是`sparse_lookup()`的传参。  
**注意：`len(feature_spec_list)`应与使用`create_table()`接口创建的表数相等；开启`一表多查`选项卡，feature_spec_list中的元素重复添加一次；开启`特征淘汰`选项卡，feature_spec_list增加时间戳的FeatureSpec类元素**。  

***

3.4 模型反向传播过程
```python
# main_mxrec.py: 410~442行
train_variables, emb_variables = get_dense_and_sparse_variable()

rank_size = mxrec_util.communication.hccl_ops.get_rank_size()
train_ops = []
# multi task training
for loss, (model_optimizer, emb_optimizer) in zip([train_model.get("loss")], optimizer_list):
    # do model optimization
    grads = model_optimizer.compute_gradients(loss, var_list=train_variables)
    avg_grads = []
    for grad, var in grads:
        if rank_size > 1:
            grad = hccl_ops.allreduce(grad, "sum") if grad is not None else None
        if grad is not None:
            avg_grads.append((grad / 8.0, var))
    # apply gradients: update variables
    train_ops.append(model_optimizer.apply_gradients(avg_grads))

    if use_dynamic_expansion:
        train_address_list = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_ID_OFFSET)
        train_emb_list = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_LOCAL_EMB)
        # do embedding optimization by addr
        sparse_grads = emb_optimizer.compute_gradients(loss, train_emb_list)  # local_embedding
        grads_and_vars = [(grad, address) for grad, address in zip(sparse_grads, train_address_list)]
        train_ops.append(emb_optimizer.apply_gradients(grads_and_vars))
    else:
        # do embedding optimization
        sparse_grads = emb_optimizer.compute_gradients(loss, emb_variables)
        print("sparse_grads_tensor:", sparse_grads)
        grads_and_vars = [(grad, variable) for grad, variable in zip(sparse_grads, emb_variables)]
        train_ops.append(emb_optimizer.apply_gradients(grads_and_vars))

# 动态学习率更新
train_ops.extend([cfg.global_step.assign(cfg.global_step + 1), cfg.learning_rate[0], cfg.learning_rate[1]])
```
410-442行代码是模型的反向过程操作。Rec SDK对推荐模型中sparse_feature的创表查表操作作了加速，使用`create_table`与`sparse_lookup`接口替换tensorflow中的`tf.nn.embedding_lookup`接口。因此模型反向更新分为两部分：417-425行代码是对`model.py`内的模型部分的反向；427-439行代码是对sparse_feature作embedding操作部分的反向过程，根据是否开启`动态扩容`选择不同的参数计算梯度并更新权重。

***

#### 4. optimizer.py
如上所述，模型反向过程分为`model.py`与`embedding`两部分；`model.py`可使用tf原生的优化器，`embedding`部分选择Rec SDK提供的`lazy_adam`或`lazy_adam_by_addr`优化器。`delay_loss_scale.py`包装`dense_optimizer`与`sparse_optimizer`并对其应用损失缩放技术，该技术主要作用于混合精度训练过程中。 

```python
import tensorflow as tf
from delay_loss_scale import DenseLossScaleOptimizer, SparseLossScaleOptimizer
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.optimizers.lazy_adam import create_hash_optimizer
from mx_rec.optimizers.lazy_adam_by_addr import create_hash_optimizer_by_address


def get_dense_and_sparse_optimizer(cfg):
    dense_optimizer = tf.train.AdamOptimizer(learning_rate=cfg.learning_rate[0])
    use_dynamic_expansion = ConfigInitializer.get_instance().use_dynamic_expansion
    if use_dynamic_expansion:
        sparse_optimizer = create_hash_optimizer_by_address(learning_rate=cfg.learning_rate[1])
    else:
        sparse_optimizer = create_hash_optimizer(learning_rate=cfg.learning_rate[1])
    sparse_optimizer = SparseLossScaleOptimizer(sparse_optimizer, 1)
    dense_optimizer = DenseLossScaleOptimizer(dense_optimizer, 1)

    return dense_optimizer, sparse_optimizer
```


