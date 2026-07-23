# Release Notes

## Version Mapping

### Product Version Information

<table><tbody><tr id="topic_0000001938532254_topic_0000001935094108_row244mcpsimp"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.1.1"><p id="topic_0000001938532254_topic_0000001935094108_p246mcpsimp">Product</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.1.1 "><p id="topic_0000001938532254_topic_0000001935094108_p1684675795511"><span id="ph925512229126">MindSDK</span></p>
</td>
</tr>
<tr id="topic_0000001938532254_topic_0000001935094108_row255mcpsimp"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.2.1"><p id="topic_0000001938532254_topic_0000001935094108_p257mcpsimp">Version</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.2.1 "><p id="topic_0000001938532254_topic_0000001935094108_p233mcpsimp">26.0.0</p>
</td>
</tr>
<tr id="topic_0000001938532254_topic_0000001935094108_row7259721105019"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.3.1"><p id="topic_0000001938532254_topic_0000001935094108_p7260182135013">Version Type</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.3.1 "><p id="topic_0000001938532254_topic_0000001935094108_p72606219501">Release version</p>
</td>
</tr>
</tbody>
</table>

### Related Product Version Mapping

| Product  | Version    |
| ---------- | ---------- |
| Ascend HDK | <ul><li>Atlas 350: 1.0.RC1</li><li>Other products: 26.0.RC1</li></ul>|
| CANN       | 9.0.0    |

### Virus Scan Results

Virus scan passed.

## Version Compatibility

- Rec SDK Torch (torch_rec_v1): After the upgrade, recompile `torchrec_npu` and the packages related to custom operators.

**Table 1** Software version compatibility

| MindSDK Version| MindSDK Version to Upgrade To                                                        | CANN Version Compatibility                                                                                   | Ascend HDK Version Compatibility                                                                                       |
| --------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Rec SDK 26.0.0  | <ul><li>MindSDK 7.3.0 and 7.3.0.x</li></ul>| <ul><li>CANN 8.3.RC1 and patch versions</li><li>CANN 8.5.0 and patch versions</li><li>CANN 9.0.0 and patch versions</li></ul>| <ul><li>Ascend HDK 25.5.0 and patch versions</li><li>Ascend HDK 25.1.RC1 and patch versions</li><li>Ascend HDK 26.0.RC1 and patch versions</li></ul>|

> [!NOTE]NOTE
>Software version compatibility means that when you upgrade the product software version, related software does not need to be upgraded or patched, and existing features remain supported.

## Usage Precautions

None

## Change Description

### New Features

|Feature|Description|Compatible Product Models|
|--|--|--|
|Rec SDK TensorFlow(tf_rec_v1)|<ul><li>Adapted for the Atlas 350 PCIe card.</li></ul>|Atlas 800T A2 training server<br>Atlas 200T A2 Box16 heterogeneous subrack<br>Atlas 800T A3 SuperPoD server<br>Atlas 350 PCIe card|
|Rec SDK Torch(torch_rec_v1)|<ul><li>Multi-level cache saving and loading: Incremental saving and loading and differential card loading are supported. </li><li>Multi-level cache admission and eviction: The `showclick` admission and eviction policy is supported. </li><li>Recommendation models now support inference on computing power-partitioned devices. </li><li>Adapted for the Atlas 350 PCIe card.</li></ul>|Atlas 800T A2 training server<br>Atlas 200T A2 Box16 heterogeneous subrack<br>Atlas 800T A3 SuperPoD server<br>Atlas 350 PCIe card|
|Rec SDK TensorFlow(tf_rec_v2)|<ul><li>The basic sparse table functions are implemented, including table creation, table lookup, saving, loading, feature admission, and feature eviction.</li></ul>|Atlas 350 PCIe card|
|Rec SDK Torch(torch_rec_v2)|<ul><li>The basic sparse table functions are implemented, including table creation, table lookup, saving, loading, feature admission, and feature eviction.</li></ul>|Atlas 350 PCIe card|
|Rec SDK operators|<ul><li>Fused operators for generative recommendation models, including `in_linear_silu` and `reverse_sequence`, are completed and adapted for the Atlas 350 PCIe card. </li><li>Fused operators for generative recommendation models, including `norm_multiply_dropout` and `concat_2d_jagged`, are completed but not adapted for the Atlas 350 PCIe card. </li><li>Enhanced HSTU operators: The performance of the backward operator is optimized. The forward and backward operators now support the int32 type. </li><li>Refactored operators: The table lookup backward operator and the HSTU forward and backward operators are refactored.</li></ul>|Atlas 800T A2 training server<br>Atlas 200T A2 Box16 heterogeneous subrack<br>Atlas 800T A3 SuperPoD server<br>Atlas 350 PCIe card|
|fbgemm-npu|<ul><li>Adapted for the Atlas 350 PCIe card.</li></ul>|Atlas 800T A2 training server<br>Atlas 200T A2 Box16 heterogeneous subrack<br>Atlas 800T A3 SuperPoD server<br>Atlas 350 PCIe card|
|HKV|<ul><li>Adapted for the Atlas 350 PCIe card.</li></ul>|Atlas 350 PCIe card|

### Interface Changes

**Rec SDK**

- Rec SDK TensorFlow (tf_rec_v1): No interface changes.
- Rec SDK Torch (torch_rec_v1): No interface changes.
- Rec SDK TensorFlow (tf_rec_v2): No interface changes.
- Rec SDK Torch (torch_rec_v2): No interface changes.

### Key Feature Changes

**Rec SDK**

- Rec SDK TensorFlow (tf_rec_v1): No key feature changes.
- Rec SDK Torch (torch_rec_v1): No key feature changes.
- Rec SDK TensorFlow (tf_rec_v2): No key feature changes.
- Rec SDK Torch (torch_rec_v2): No key feature changes.

### Resolved Issues

None

### Known Issues

None

## Upgrade Impact

### Impact on the System During the Upgrade

None

### Impact on the System After the Upgrade

None

## 26.0.0 Documentation

|Document|Description|Update Description|
|--|--|--|
|[Rec SDK 26.0.0 User Guide](../../README.md)|Introduces Rec SDK, its installation and deployment process, functions, model adaptation, and API reference.|For details, see *Rec SDK 26.0.0 User Guide*.|

## Vulnerability Patch List

None
