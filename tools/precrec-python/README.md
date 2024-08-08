precrec-python精度比对工具使用说明
注意：精度工具暂只支持配合 little demo 精度对齐模式使用。

# 代码结构
```shell
.
├── data_set.py               # 数据集 解析对比模块
├── dense_ckpt.py             # dense ckpt 解析对比模块
├── dump_info.py              # 任务信息 解析对比模块
├── loss.py                   # loss 解析对比模块
├── ops.py                    # 算子 解析对比模块
├── precision_check.py        # 精度比对工具 总体入口
├── sparse_ckpt.py            # sparse ckpt 解析对比模块
└── utils.py                  # 公共函数
```

# 使用配置
使用前请根据需要修改 precision_check.py 中对应配置参数

## 日志级别
不同级别日志展示中间过程信息不同， 建议使用INFO_LEVEL
```python
"""
Please choose your log level before comparison.
Could be DEBUG_LEVEL, INFO_LEVEL, WARNING_LEVEL, ERROR_LEVEL
"""
init_logger(INFO_LEVEL)
```

## 解析内容
解析文件结构如下
```shell
precision_check
└── 20240807_091347
    ├── 01dump_dataset # 数据集数据
    ├── 02dump_model # dense sparse 落盘模型
    ├── 03dump_loss # 每步loss
    ├── 04dump_op # 查表 更新表相关数据
    └── dump_info.json # 模型配置信息
```

其中解析、比对内容可配置，直接在global_func_key_list中添加即可
```python
"""
Choose what data you want to compare.
Could be any one or any combination of following choices.
BATCH_DATASET_FUNC_KEY, SPARSE_MODEL_FUNC_KEY, DENSE_MODEL_FUNC_KEY, OP_DATA_FUNC_KEY
"""
global_func_key_list = [
    BATCH_DATASET_FUNC_KEY,
    SPARSE_MODEL_FUNC_KEY,
    DENSE_MODEL_FUNC_KEY,
    LOSS,
    OP_DATA_FUNC_KEY,
]
```

## 解析step
具体解析哪些step, 在global_step_list添加即可。默认为解析数据中所有的step
```python
"""
Set up the steps of data you want to compare, could only be a list of int.
"""
global_step_list = [1, 2]
```

## 解析rank
具体解析哪些rank, 在global_rank_list添加即可。默认为解析数据中所有的rank
```python
"""
Set up the ranks of data you want to compare, could only be a list of int.
"""
global_rank_list = list(range(8))
```

# 使用方法
```shell
python precision_check.py /home/little_demo/precision_check/20240807_091347 /home/little_demo/precision_check/20240807_101855
```

其中为/home/little_demo/precision_check/20240807_091347 little demo开启精度对齐模式运行后产生数据
precision_check.py将对它进行解析并比对两次任务结果

# 结果展示
## 对比结果展示
```
1: 
1.0: 
1.0.BatchDataSet: True
1.0.SparseModel: True
1.0.DenseModel: True
1.0.Loss: True
1.0.Opdata: True
```
其中 1: 代表 step 1
其中 1.0: 代表 step 1, rank 0
其中 1.0.SparseModel: 代表 step 1, rank 0 中的SparseModel部分是否相等， True为比较结果相同

## 解析内容展示查询
可解开一下代码注释进入pdb来进行内容查询
```python
# Use pdb can look into the parsing data without running the whole process repeatedly.
"""
test_data_result = nested_dict_to_str(glob_test_data.data_dict)
golden_data_result = nested_dict_to_str(glob_golden_data.data_dict)
logging.info("Test Data Dict[test_data_result]:\n %s", test_data_result)
logging.info("Golden Data Dict[golden_data_result]:\n %s", golden_data_result)
import pdb
pdb.set_trace()
"""
```

会直接展示已解析的内容
```
[PrecCheck][INFO][2024/08/07 10:56:44] Test Data Dict[test_data_result]:
1: 
1.0: 
1.0.BatchDataSet: <data_set.BatchDataSet object at 0x7f99fe027a10>
1.0.SparseModel: <sparse_ckpt.SparseModel object at 0x7f99fe05f2d0>
1.0.DenseModel: <dense_ckpt.DenseModel object at 0x7f9a00187c50>
1.0.Loss: <loss.Loss object at 0x7f99fe0847d0>
1.0.Opdata: <ops.OpData object at 0x7f99fe084390>
```

可通过 dir(glob_test_data_dict[1][0]["BatchDataSet"])来取得step1 rank0的dataset数据的哪些类，并依次查看
```
(Pdb) dir(glob_test_data_dict[1][0]["BatchDataSet"])
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'batch_data', 'batch_data_name_list', 'batch_data_path', 'data_pattern', 'get_dump_data_names', 'parse_batch_data']
(Pdb) glob_test_data.data_dict[1][0]["BatchDataSet"].batch_data
{'data_rank_0_batch_1_category_ids.npy': array([[36416, 30080, 25696],
       [27957, 12169, 21778],
       [20431,  3941, 35953],
       ...,
       [21304, 23233, 19757],
       [25426, 11084, 11182],
       [26194, 13630, 14365]]),
}
```
