# Constant Reference

## `ASCEND_TIMESTAMP`

**Description**

This constant represents a collection name used as a key when you build a TensorFlow graph to determine whether to enable timestamps for a dataset.

**Example**

```python
from mx_rec.constants.constants import ASCEND_TIMESTAMP
tf.compat.v1.add_to_collection(ASCEND_TIMESTAMP, batch["timestamp"])   # batch is an iterator object of the dataset.
```
