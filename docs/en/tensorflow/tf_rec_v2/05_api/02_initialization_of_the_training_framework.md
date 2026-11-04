# Initialization of the Training Framework

## `init`

**Description**

Initializes the Rec SDK TensorFlow model training framework. Before calling other APIs, you need to initialize the framework.

**Function Prototype**

```python
def init(toml_path: str)
```

| Parameter    | Type| Mandatory/Optional| Description              |
|---------|---|--|---------------------------|
|toml_path|str|Mandatory|Path of the initialization configuration file. The path length range is [1, 1024].|

**Returns**

- Success: No value is returned.
- Failure: An exception is thrown.

**Example**

```python
import mxrec
mxrec.init("toml_path")
```

**References**

The following is an example of the initialization configuration file (.toml format):

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

>[!NOTE]
>
>- The content under the `[mxrec]` field is mandatory. For details about the `[mxrec.cm-node-info]` field, see <https://www.hiascend.com/document/detail/zh/canncommercial/850/commlib/hcclug/hcclug_000070.html>.
