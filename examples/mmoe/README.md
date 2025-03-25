# mmoe模型 迁移样例（基于DLRM模型框架）

开源项目在保证原有结构不变的情况下，可采用替换相关API接口的方式将项目由GPU >> NPU >> Rec SDK。在模型迁移适配过程中可能因兼容性问题而导致模型迁移失败，此处提供另一种模型适配方案。  

***
## 开源项目链接
```
wangzhen38 committed on May 19, 2022 hash值（提交ID）：54b6a96abae574a04c7fcac53df190f885970c3f
commit的链接: https://github.com/PaddlePaddle/PaddleRec/commit/54b6a96abae574a04c7fcac53df190f885970c3f
https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/mmoe
```
***
## 数据集

```url
Census-Income-KDD数据集:  
https://archive.ics.uci.edu/static/public/117/census+income+kdd.zip
```
***
## 数据集预处理

### 解压文件列表
- census-income.data 训练数据
- census-income.test 测试数据
- census-income.name 

数据预处理文件：census.py。  

***
### 数据预处理运行脚本
```python
python census.py --train_data_path train_data_path --test_data_path test_data_path --output_path output_path 
```
参数说明：
- train_data_path: census-income.data的路径，如："D:\dat\census-income.data"
- test_data_path: census-income.test的路径，如："D:\dat\census-income.test"
- output_path: tfrecord存放路径，如："D:\dat\tfrecord\ "
***

### census.py
***
#### 1. 建立特征映射
调用`census.py`文件中的`get_fea_map()`方法，以`{'C1':{}, 'C2':{},..., 'I1':{},...}`形式储存sparse_feature去重后的特征映射。

```python
# get feature_map
feature_map = get_fea_map(split_file_list=list(file_path_dict.values()))
```
***

#### 2. sparse_feature特征映射
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

#### 3. 数据集格式转换：txt >> tfrecord
调用`census.py`文件中的`convert_input2tfrd()`方法将txt文件转换为tfrecord文件。

```python
# txt to tfrecords
convert_input2tfrd(data_frame=data_df, in_file_path=file_path, out_file_path=output_path_)
```
***

## 模型运行

参考Rec SDK的`README.md`文件在NPU服务器上配置环境并安装镜像创建容器后，可参考DLRM模型运行命令启动模型训练。模型运行脚本是run.sh，运行此脚本需要四个参数：so_path、rec_package_path、hccl_cfg_json以及dlrm_criteo_data_path。其中，   
- so_path: Rec SDK中libasc所在路径，在镜像中已经安装过Rec SDK，所以so_path是：/usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec/libasc/
- rec_package_path: Rec SDK这个包的安装路径，镜像中是：/usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec/  
- hccl_cfg_json:  hccl配置文件所在路径，一般是当前路径下的hccl文件
- dlrm_criteo_data_path: mmoe模型需要的数据所在路径，根据实际情况进行配置

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


***
## 模型迁移

**迁移思路：** 在现有已适配好的dlrm模型框架下，改动相关代码逻辑，完成mmoe模型的适配。**核心：根据开源项目model代码修改`model.py`；数据处理操作一部分放入`census.py`,一部分放入`main_Rec SDK.py`中`make_batch_and_iterator()`内；`main_mxrec.py`中其他相关代码改动主要是为了适配Rec SDK提供的相关特性。**
详细改动见https://gitee.com/ascend/mxrec/pulls/171/commits，Commits ID：7a05b033d41af51df9aed7414ad04216dff821cc。  
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
    0.001
)
# 超参数
self.batch_size = 32
self.line_per_sample = 1
self.train_epoch = 100
self.test_epoch = 100
self.expert_num = 8 #专家网络数量
self.gate_num = 2 #门控网络数量
self.expert_size = 16
self.tower_size = 8      
# emb dim大小计算
self.emb_dim = self.expert_num * self.expert_size + self.gate_num * self.expert_num    
```
***


#### 2. model.py
迁移过程中，`model.py`需参考开源项目文件`models/multitask/mmoe/net.py`的代码逻辑，使用tensorflow的低阶API重新编写。输出参数必须包括`loss`,`prediction`,`label`,`trainable_variables`。**迁移重点：Rec SDK对推荐模型中sparse_feature的创表查表操作作了加速，使用`create_table`与`sparse_lookup`接口替换tensorflow中的`tf.nn.embedding_lookup`接口。** 因此在适配开源项目时，会将sparse_feature的embedding操作放在模型结构外。 

**reclearn开源项目原始代码：**
```python
# net.py
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class MMoELayer(nn.Layer):
    def __init__(self, feature_size, expert_num, expert_size, tower_size,
                 gate_num):
        super(MMoELayer, self).__init__()

        self.expert_num = expert_num
        self.expert_size = expert_size
        self.tower_size = tower_size
        self.gate_num = gate_num

        self._param_expert = []
        expert_init = [pow(10, -i) for i in range(1, self.expert_num + 1)]
        for i in range(0, self.expert_num):
            linear = self.add_sublayer(
                name='expert_' + str(i),
                sublayer=nn.Linear(
                    feature_size,
                    expert_size,
                    #initialize each expert respectly
                    weight_attr=nn.initializer.Constant(value=expert_init[i]),
                    bias_attr=nn.initializer.Constant(value=0.1),
                    #bias_attr=paddle.ParamAttr(learning_rate=1.0),
                    name='expert_' + str(i)))
            self._param_expert.append(linear)

        self._param_gate = []
        self._param_tower = []
        self._param_tower_out = []
        gate_init = [pow(10, -i) for i in range(1, self.gate_num + 1)]
        for i in range(0, self.gate_num):
            linear = self.add_sublayer(
                name='gate_' + str(i),
                sublayer=nn.Linear(
                    feature_size,
                    expert_num,
                    #initialize every gate respectly
                    weight_attr=nn.initializer.Constant(value=gate_init[i]),
                    bias_attr=nn.initializer.Constant(value=0.1),
                    #bias_attr=paddle.ParamAttr(learning_rate=1.0),
                    name='gate_' + str(i)))
            self._param_gate.append(linear)

            linear = self.add_sublayer(
                name='tower_' + str(i),
                sublayer=nn.Linear(
                    expert_size,
                    tower_size,
                    #initialize each gate respectly
                    weight_attr=nn.initializer.Constant(value=gate_init[i]),
                    bias_attr=nn.initializer.Constant(value=0.1),
                    #bias_attr=paddle.ParamAttr(learning_rate=1.0),
                    name='tower_' + str(i)))
            self._param_tower.append(linear)

            linear = self.add_sublayer(
                name='tower_out_' + str(i),
                sublayer=nn.Linear(
                    tower_size,
                    2,
                    #initialize each gate respectly
                    weight_attr=nn.initializer.Constant(value=gate_init[i]),
                    bias_attr=nn.initializer.Constant(value=0.1),
                    name='tower_out_' + str(i)))
            self._param_tower_out.append(linear)

    def forward(self, input_data):
        expert_outputs = []
        for i in range(0, self.expert_num):
            linear_out = self._param_expert[i](input_data)
            expert_output = F.relu(linear_out)
            expert_outputs.append(expert_output)
        expert_concat = paddle.concat(x=expert_outputs, axis=1)
        expert_concat = paddle.reshape(
            expert_concat, [-1, self.expert_num, self.expert_size])

        output_layers = []
        for i in range(0, self.gate_num):
            cur_gate_linear = self._param_gate[i](input_data)
            cur_gate = F.softmax(cur_gate_linear)
            cur_gate = paddle.reshape(cur_gate, [-1, self.expert_num, 1])
            cur_gate_expert = paddle.multiply(x=expert_concat, y=cur_gate)
            cur_gate_expert = paddle.sum(x=cur_gate_expert, axis=1)
            cur_tower = self._param_tower[i](cur_gate_expert)
            cur_tower = F.relu(cur_tower)
            out = self._param_tower_out[i](cur_tower)
            out = F.softmax(out)
            out = paddle.clip(out, min=1e-15, max=1.0 - 1e-15)
            output_layers.append(out)

        return output_layers

```
`_input`未把sparse特征和dense特征区分，迁移后将sparse数据单独做了处理，详见`main_mxrec.py`,而后将sparse查表数据embedding与dense数据分别传入模型中，之后再切片聚合，还原原代码操作。  

**迁移后代码：**
```python
# model.py
from easydict import EasyDict as edict

import tensorflow as tf


model_cfg = edict()
model_cfg.loss_mode = "batch"
LOSS_OP_NAME = "loss"
LABEL_OP_NAME = "label"
VAR_LIST = "variable"
PRED_OP_NAME = "pred"


class MyModel:
    def __init__(self, expert_num=8, expert_size=16, tower_size=8, gate_num=2):

        self.expert_num = expert_num
        self.expert_size = expert_size
        self.tower_size = tower_size
        self.gate_num = gate_num

    
    def expert_layer(self, _input):
        param_expert = []
        for i in range(0, self.expert_num):
            expert_linear = tf.layers.dense(_input, units=self.expert_size, activation=None, name=f'expert_layer_{i}', 
                                            kernel_initializer=tf.constant_initializer(value=0.1), 
                                            bias_initializer=tf.constant_initializer(value=0.1))
            
            param_expert.append(expert_linear)
        return param_expert
    
    
    def gate_layer(self, _input):
        param_gate = []
        for i in range(0, self.gate_num):
            gate_linear = tf.layers.dense(_input, units=self.expert_num, activation=None, name=f'gate_layer_{i}', 
                                            kernel_initializer=tf.constant_initializer(value=0.1), 
                                            bias_initializer=tf.constant_initializer(value=0.1))
            
            param_gate.append(gate_linear)
        return param_gate
    
    
    def tower_layer(self, _input, layer_name):
        tower_linear = tf.layers.dense(_input, units=self.tower_size, activation='relu', 
                                            name=f'tower_layer_{layer_name}', 
                                            kernel_initializer=tf.constant_initializer(value=0.1), 
                                            bias_initializer=tf.constant_initializer(value=0.1))
        
        tower_linear_out = tf.layers.dense(tower_linear, units=2, activation=None, 
                                            name=f'tower_payer_out_{layer_name}', 
                                            kernel_initializer=tf.constant_initializer(value=0.1), 
                                            bias_initializer=tf.constant_initializer(value=0.1))
        
        return tower_linear_out
        
        

    
    def build_model(self,
                    embedding=None,
                    dense_feature=None,
                    label=None,
                    is_training=True,
                    seed=None):

        with tf.variable_scope("mmoe", reuse=tf.AUTO_REUSE):

            dense_expert = self.expert_layer(dense_feature)
            dense_gate = self.gate_layer(dense_feature)

            all_expert = []
            _slice_num = 0
            for i in range(0, self.expert_num):
                slice_num_end = _slice_num + self.expert_size
                cur_expert = tf.add(dense_expert[i], embedding[:, _slice_num:slice_num_end])
                cur_expert = tf.nn.relu(cur_expert)
                all_expert.append(cur_expert)
                _slice_num = slice_num_end

            expert_concat = tf.concat(all_expert, axis=1)
            expert_concat = tf.reshape(expert_concat, [-1, self.expert_num, self.expert_size])

            output_layers = []
            out_pred = []
            for i in range(0, self.gate_num):
                slice_gate_end = _slice_num + self.expert_num
                cur_gate = tf.add(dense_gate[i], embedding[:, _slice_num:slice_gate_end])
                cur_gate = tf.nn.softmax(cur_gate)

                cur_gate = tf.reshape(cur_gate, [-1, self.expert_num, 1])

                cur_gate_expert = tf.multiply(x=expert_concat, y=cur_gate)
                cur_gate_expert = tf.reduce_sum(cur_gate_expert, axis=1)
                
                out = self.tower_layer(cur_gate_expert, i)
                out = tf.nn.softmax(out)
                out = tf.clip_by_value(out, clip_value_min=1e-15, clip_value_max=1.0 - 1e-15)
                output_layers.append(out)
                out_pred.append(tf.nn.softmax(out[:, 1]))
                _slice_num = slice_gate_end
            trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='mmoe')

            label_income = label[:, 0:1]
            label_mat = label[:, 1:]

            pred_income_1 = tf.slice(output_layers[0], [0, 1], [-1, 1])
            pred_marital_1 = tf.slice(output_layers[1], [0, 1], [-1, 1])

            cost_income = tf.losses.log_loss(labels=tf.cast(label_income, tf.float32), predictions=pred_income_1,
                                             epsilon=1e-4)
            cost_marital = tf.losses.log_loss(labels=tf.cast(label_mat, tf.float32), predictions=pred_marital_1,
                                              epsilon=1e-4)

            avg_cost_income = tf.reduce_mean(cost_income)
            avg_cost_marital = tf.reduce_mean(cost_marital)

            loss = 0.5 * (avg_cost_income + avg_cost_marital)
            
            return {LOSS_OP_NAME: loss,
                    PRED_OP_NAME: out_pred,
                    LABEL_OP_NAME: label,
                    VAR_LIST: trainable_variables}


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
# main_mxrec.py：65~79行
def extract_fn(data_record):
        features = {
            # Extract features using the keys set during creation
            'label': tf.compat.v1.FixedLenFeature(shape=(2 * config.line_per_sample,), dtype=tf.int64),
            'sparse_feature': tf.compat.v1.FixedLenFeature(shape=(29 * config.line_per_sample,), dtype=tf.int64),
            'dense_feature': tf.compat.v1.FixedLenFeature(shape=(11 * config.line_per_sample,), dtype=tf.float32),
        }
        sample = tf.compat.v1.parse_single_example(data_record, features)
        return sample

def reshape_fn(batch):
    batch['label'] = tf.reshape(batch['label'], [-1, 2])
    batch['dense_feature'] = tf.reshape(batch['dense_feature'], [-1, 11])
    batch['sparse_feature'] = tf.reshape(batch['sparse_feature'], [-1, 29])
    return batch
```


***
3.2 模型前向传播过程

```python
# main_mxrec.py：112~137行
def model_forward(feature_list, hash_table_list, batch, is_train, modify_graph):
    embedding_list = []
    logger.debug(f"In model_forward function, is_train: {is_train}, feature_list: {len(feature_list)}, "
                 f"hash_table_list: {len(hash_table_list)}")
    for feature, hash_table in zip(feature_list, hash_table_list):
        if MODIFY_GRAPH_FLAG:
            feature = batch["sparse_feature"]
        embedding = sparse_lookup(hash_table, feature, cfg.send_count, dim=None, is_train=is_train,
                                  name="user_embedding_lookup", modify_graph=modify_graph, batch=batch,
                                  access_and_evict_config=None)
        embedding_list.append(embedding)

    if len(embedding_list) == 1:
        emb = embedding_list[0]
    elif len(embedding_list) > 1:
        emb = tf.reduce_sum(embedding_list, axis=0, keepdims=False)
    else:
        raise ValueError("the length of embedding_list must be greater than or equal to 1.")
    emb = tf.reduce_sum(emb, axis=1)
    my_model = MyModel()
    model_output = my_model.build_model(embedding=emb,
                                        dense_feature=batch["dense_feature"],
                                        label=batch["label"],
                                        is_training=is_train,
                                        seed=dense_hashtable_seed)
    return model_output
```
该函数是前向传播函数，主要包括sparse_feature的embedding操作（查表）与model前向操作。

***


