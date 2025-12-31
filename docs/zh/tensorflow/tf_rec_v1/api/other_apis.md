# 其他接口<a name="ZH-CN_TOPIC_0000001630046429"></a>

## version<a name="ZH-CN_TOPIC_0000001580326428"></a>

**功能描述<a name="section634582619155"></a>**

Rec SDK TensorFlow框架版本号查询。

**函数原型<a name="section1483104721911"></a>**

```bash
def version()
```

**返回值说明<a name="section651195312311"></a>**

-   成功：返回版本号。
-   失败：抛出异常。

**使用示例<a name="section2553042232"></a>**

用户可以通过version\(\)或\_\_version\_\_查询Rec SDK TensorFlow框架的版本号。示例如下：

```bash
import mx_rec
print(mx_rec.version())
print(mx_rec.__version__)
```


## ascend\_global\_hashtable\_collection<a name="ZH-CN_TOPIC_0000001649963549"></a>

**功能描述<a name="section1217131745816"></a>**

修改、获取哈希表集合的名字。

**函数原型<a name="section858517176587"></a>**

```bash
# 获取哈希表集合名字
@property
def ascend_global_hashtable_collection(self):
    return self._ascend_global_hashtable_collection
# 修改哈希表集合名字
@ascend_global_hashtable_collection.setter
def ascend_global_hashtable_collection(self, name):
    self._ascend_global_hashtable_collection = name
```

**参数说明<a name="section65810189587"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|name|string|可选|哈希集合修改后的名称。长度范围：[1, 255]。|


**返回值说明<a name="section10745722145816"></a>**

-   成功：返回None。
-   失败：抛出异常。

**使用示例<a name="section09971948135814"></a>**

```bash
# 获取哈希表集合名字
from mx_rec.util.initialize import ConfigInitializer
hashtable_collection = ConfigInitializer.get_instance().train_params_config.ascend_global_hashtable_collection
# 修改哈希表集合名字
from mx_rec.util.initialize import ConfigInitializer
ConfigInitializer.get_instance().train_params_config.ascend_global_hashtable_collection = "test"
```


## get\_rank\_id<a name="ZH-CN_TOPIC_0000001675558225"></a>

**功能描述<a name="section123217321652"></a>**

返回当前进程在MPI通信中的序号。

**函数原型<a name="section8465133218513"></a>**

```bash
def get_rank_id()
```

**返回值说明<a name="section46439326512"></a>**

-   成功：返回当前进程在MPI通信中的序号。
-   失败：抛出异常。

**使用示例<a name="section7851532751"></a>**

```bash
from rec_sdk_common.communication.hccl.hccl_info import get_rank_id
rank_id = get_rank_id()
```


## get\_rank\_size<a name="ZH-CN_TOPIC_0000001627318510"></a>

**功能描述<a name="section123217321652"></a>**

返回MPI通信器中的总进程数。

**函数原型<a name="section8465133218513"></a>**

```bash
def get_rank_size()
```

**返回值说明<a name="section46439326512"></a>**

-   成功：返回MPI通信器中的总进程数。
-   失败：抛出异常。

**使用示例<a name="section7851532751"></a>**

```bash
from rec_sdk_common.communication.hccl.hccl_info import get_rank_size
rank_size = get_rank_size()
```


## import\_host\_pipeline\_ops<a name="ZH-CN_TOPIC_0000001676679253"></a>

**功能描述<a name="section14715151895416"></a>**

获得Rec SDK TensorFlow中自定义的TensorFlow算子。

**函数原型<a name="section18854625145510"></a>**

```bash
def import_host_pipeline_ops(so_pkg_name: str = LIBASC_OPS_SO) -> ModuleType
```

**参数说明<a name="section65810189587"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|so_pkg_name|string|可选|Rec SDK TensorFlow so包名。长度范围：[1, 100]。|


**返回值说明<a name="section1828451185519"></a>**

-   成功：返回一个包含so中Rec SDK TensorFlow定义的TensorFlow算子的Python封装模块。
-   失败：返回“RuntimeError: when unable to load the library or get the python wrappers.”

**使用示例<a name="section148539625910"></a>**

```bash
from mx_rec.util.ops import import_host_pipeline_ops
host_pipeline_ops = import_host_pipeline_ops()
```

**set\_threshold算子介绍<a name="section82914222915"></a>**

<a name="table191438421290"></a>
<table><tbody><tr id="row181439428296"><th class="firstcol" valign="top" width="13.98%" id="mcps1.1.3.1.1"><p id="p1144642112915"><a name="p1144642112915"></a><a name="p1144642112915"></a>算子名称</p>
</th>
<td class="cellrowborder" valign="top" width="86.02%" headers="mcps1.1.3.1.1 "><p id="p160744914308"><a name="p160744914308"></a><a name="p160744914308"></a>set_threshold</p>
</td>
</tr>
<tr id="row6144242202912"><th class="firstcol" valign="top" width="13.98%" id="mcps1.1.3.2.1"><p id="p1214414202911"><a name="p1214414202911"></a><a name="p1214414202911"></a>算子功能</p>
</th>
<td class="cellrowborder" valign="top" width="86.02%" headers="mcps1.1.3.2.1 "><p id="p76061449173019"><a name="p76061449173019"></a><a name="p76061449173019"></a>更改特征准入阈值</p>
</td>
</tr>
<tr id="row11441042102913"><th class="firstcol" valign="top" width="13.98%" id="mcps1.1.3.3.1"><p id="p9144124217290"><a name="p9144124217290"></a><a name="p9144124217290"></a>参数说明</p>
</th>
<td class="cellrowborder" valign="top" width="86.02%" headers="mcps1.1.3.3.1 "><a name="ul1065662883114"></a><a name="ul1065662883114"></a><ul id="ul1065662883114"><li>第一个入参为TF上下文中上层的tf.Tensor。</li><li>emb_name: List[str]：需要修改准入阈值的特征表。</li><li>ids_name: List[str]：基于兼容性原因遗留，目前无作用。</li></ul>
</td>
</tr>
<tr id="row183421350113111"><th class="firstcol" valign="top" width="13.98%" id="mcps1.1.3.4.1"><p id="p1534235018317"><a name="p1534235018317"></a><a name="p1534235018317"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="86.02%" headers="mcps1.1.3.4.1 "><p id="p1034245020313"><a name="p1034245020313"></a><a name="p1034245020313"></a>set_threshold的第一个入参值为“0”，表示修改对应的emb表为特征不累加模式（准入阈值不变，但是特征计数不再累加，使用历史值）。</p>
</td>
</tr>
</tbody>
</table>


## use\_dynamic\_expansion<a name="ZH-CN_TOPIC_0000001676359469"></a>

**功能描述<a name="section193914186216"></a>**

返回是否使用了动态扩容。

**函数原型<a name="section844715301121"></a>**

```bash
@property
def use_dynamic_expansion(self)
```

**返回值说明<a name="section122575425214"></a>**

-   True：使用了动态扩容。
-   False：没有使用动态扩容。

**使用示例<a name="section1215012571213"></a>**

```bash
from mx_rec.util.initialize import ConfigInitializer
use_dynamic_expansion = ConfigInitializer.get_instance().use_dynamic_expansion
```


## hccl\_ops<a name="ZH-CN_TOPIC_0000001676439373"></a>

**功能描述<a name="section193914186216"></a>**

适配不同版本的HCCL算子。

**返回值说明<a name="section122575425214"></a>**

返回相应版本的HCCL算子。

**使用示例<a name="section1215012571213"></a>**

```bash
from mx_rec.util.tf_version_adapter import hccl_ops
```


## get\_target\_batch<a name="ZH-CN_TOPIC_0000001700857132"></a>

**功能描述<a name="section971782218537"></a>**

返回自动改图模式下生成新数据集中batch的记录。

**函数原型<a name="section1597420420530"></a>**

```bash
def get_target_batch(self, is_training: bool)
```

**参数说明<a name="section65810189587"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|is_training|bool|必选|是否为训练模式。<li>True：表示是训练模式。</li><li>False：表示不是训练模式。</li>|


**返回值说明<a name="section1150317065419"></a>**

-   成功：返回自动改图模式下生成新数据集中batch的记录。
-   失败：抛出异常。

**使用示例<a name="section11613181513543"></a>**

```bash
from mx_rec.util.initialize import ConfigInitializer
ConfigInitializer.get_instance().train_params_config.get_target_batch(False)
```


