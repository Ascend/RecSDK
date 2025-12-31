# 常量参考<a name="ZH-CN_TOPIC_0000001627638374"></a>

## ASCEND\_TIMESTAMP<a name="ZH-CN_TOPIC_0000001627158698"></a>

**功能描述<a name="section123217321652"></a>**

集合名常量，在TensorFlow构建图时作为key调用，用于设定数据集是否启用时间戳。

**使用示例<a name="section7851532751"></a>**

```bash
from mx_rec.constants.constants import ASCEND_TIMESTAMP
tf.compat.v1.add_to_collection(ASCEND_TIMESTAMP, batch["timestamp"])   #batch为数据集的迭代器对象
```


