# 训练框架初始化<a name="ZH-CN_TOPIC_0000001630127073"></a>

## init<a name="ZH-CN_TOPIC_0000001630046449"></a>

**功能描述<a name="section634582619155"></a>**

初始化Rec SDK TensorFlow模型训练框架，在调用其他接口前，需先进行初始化。

**函数原型<a name="section1483104721911"></a>**

```bash
def init(toml_path: str)
```

| 参数名     | 类型 | 可选/必选 | 说明                        |
|---------|---|--|---------------------------|
|toml_path|str|必选|初始化配置文件的路径。路径长度范围：[1, 1024]|

**返回值说明<a name="section651195312311"></a>**

-   成功：None。
-   失败：抛出异常。

**使用示例<a name="zh-cn_topic_0000001422098394_section653575124718"></a>**

```python
import mxrec
mxrec.init("toml_path")
```

**参考资源<a name="section426664933312"></a>**

初始化配置文件（.toml文件）可参考如下示例进行配置：
```toml
[mxrec]
# Only ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL') are allowed.
log_level = "INFO"
# If use_ranktable is true, the environment variable "RANK_TABLE_FILE" will be read; if it is false, the configuration
# of "mxrec.cm-node-info" in the toml file will be read.
use_ranktable = true

[mxrec.cm-node-info]
# Used to configure the listening host ip of the master node.
cm_chief_ip = "127.0.0.1"
# Used to configure the listening port of the master node.
cm_chief_port = 60001
# Used to specify the logical id of the device for collecting cluster information of the server side 
# within the master node.
cm_chief_device = 0
# Used to configure the network card ip used for information exchange between the current devices and the master node.
cm_worker_ip = "127.0.0.1"
# Used to configure the number of devices for this business communication domain.
cm_worker_size = 1

[model]
# Support "train_and_evaluate", "load_and_train", "predict".
mode = "train_and_evaluate"
# Set to true for precision alignment mode.
deterministic = false

saved_path = "saved_model"
train_steps = 200
train_interval = 100
eval_steps = 10
batch_number = 200

[model.distribution]
interface = "lo"
local_rank_size = 1
num_server = 1
```

>[!NOTE] 说明 
>- `[mxrec]`字段下的内容为必须配置项，其中`[mxrec.cm-node-info]`字段可参考：https://www.hiascend.com/document/detail/zh/canncommercial/850/commlib/hcclug/hcclug_000070.html

