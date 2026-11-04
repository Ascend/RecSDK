# Release Notes

## Version Mapping

### Product Version Information

<a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108__Ref249955742"></a>
<table><tbody><tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row244mcpsimp"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"></a>Product</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"></a><span id="ph925512229126"><a name="ph925512229126"></a><a name="ph925512229126"></a>Rec SDK</span></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row255mcpsimp"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"></a>Version</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"></a>26.1.0</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row7259721105019"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"></a>Version Type</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"></a>Release version</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row7259721105019"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"></a>Maintenance Period</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"></a>See the <a href="../../README.md#torch_rec_v1-框架维护策略">maintenance policy</a></p>
</td>
</tr>
</tbody>
</table>

### Related Product Version Mapping

**Table 1** Rec SDK software version mapping

| Rec SDK | CANN Version | Ascend HDK Version | TorchNPU Version |
| --- | --- | --- | --- |
| 26.1.0 | 9.1.0 | <ul><li>Atlas A2/A3 series products: 26.1.0</li><li>Atlas 350 series products: 25.7.RC1</li></ul> | 26.1.0 |

## Version Compatibility

> [!NOTE]
>
> In the tables in this section, "/" indicates that the versions are not compatible, and "Y" indicates that they are compatible.
>
> Software version compatibility means that when you upgrade the product software version, related software does not need to be upgraded or patched, and existing features remain supported.
>
> Rec SDK Torch (torch_rec_v1): After the upgrade, recompile `torchrec_npu` and the packages related to custom operators.
>
> Rec SDK Torch (torch_rec_v2): After the upgrade, recompile `torchrec_npu` and the packages related to custom operators.

### Atlas A2/A3 Series Products

**Table 2** Rec SDK and CANN version compatibility

<table style="table-layout: fixed; width: 433px"><colgroup>
<col style="width: 156px">
<col style="width: 88px">
<col style="width: 91px">
<col style="width: 98px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Rec SDK</th>
    <th colspan="3">CANN Version</th>
  </tr>
  <tr>
    <th>8.5.0</th>
    <th>9.0.0</th>
    <th>9.1.0</th>
  </tr></thead>
<tbody>
  <tr>
    <td>7.3.0</td>
    <td>Y</td>
    <td>Y</td>
    <td>Y</td>
  </tr>
  <tr>
    <td>26.0.0</td>
    <td>/</td>
    <td>Y</td>
    <td>Y</td>
  </tr>
  <tr>
    <td>26.1.0</td>
    <td>/</td>
    <td>Y</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

**Table 3** Rec SDK and Ascend HDK version compatibility

<table style="table-layout: fixed; width: 433px"><colgroup>
<col style="width: 156px">
<col style="width: 88px">
<col style="width: 91px">
<col style="width: 98px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Rec SDK</th>
    <th colspan="3">Ascend HDK Version</th>
  </tr>
  <tr>
    <th>25.5.0</th>
    <th>26.0.RC1</th>
    <th>26.1.0</th>
  </tr></thead>
<tbody>
  <tr>
    <td>7.3.0</td>
    <td>Y</td>
    <td>Y</td>
    <td>Y</td>
  </tr>
  <tr>
    <td>26.0.0</td>
    <td>/</td>
    <td>Y</td>
    <td>Y</td>
  </tr>
  <tr>
    <td>26.1.0</td>
    <td>/</td>
    <td>Y</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

**Table 4** Rec SDK and TorchNPU version compatibility

<table style="table-layout: fixed; width: 433px"><colgroup>
<col style="width: 156px">
<col style="width: 88px">
<col style="width: 91px">
<col style="width: 98px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Rec SDK</th>
    <th colspan="3">TorchNPU Version</th>
  </tr>
  <tr>
    <th>7.3.0</th>
    <th>26.0.0</th>
    <th>26.1.0</th>
  </tr></thead>
<tbody>
  <tr>
    <td>7.3.0</td>
    <td>Y</td>
    <td>Y</td>
    <td>Y</td>
  </tr>
  <tr>
    <td>26.0.0</td>
    <td>/</td>
    <td>Y</td>
    <td>Y</td>
  </tr>
  <tr>
    <td>26.1.0</td>
    <td>/</td>
    <td>Y</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

### Atlas 350 Series Products

**Table 5** Rec SDK and CANN version compatibility

<table style="table-layout: fixed; width: 433px"><colgroup>
<col style="width: 156px">
<col style="width: 88px">
<col style="width: 91px">
<col style="width: 98px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Rec SDK</th>
    <th colspan="2">CANN Version</th>
  </tr>
  <tr>
    <th>9.0.0</th>
    <th>9.1.0</th>
  </tr></thead>
<tbody>
  <tr>
    <td>26.0.0</td>
    <td>Y</td>
    <td>Y</td>
  </tr>
  <tr>
    <td>26.1.0</td>
    <td>/</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

**Table 6** Rec SDK and Ascend HDK version compatibility

<table style="table-layout: fixed; width: 433px"><colgroup>
<col style="width: 156px">
<col style="width: 156px">
<col style="width: 91px">
<col style="width: 98px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Rec SDK</th>
    <th colspan="1">Ascend HDK Version</th>
  </tr>
  <tr>
    <th>25.7.RC1</th>
  </tr></thead>
<tbody>
  <tr>
    <td>26.0.0</td>
    <td>Y</td>
  </tr>
  <tr>
    <td>26.1.0</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

**Table 7** Rec SDK and TorchNPU version compatibility

<table style="table-layout: fixed; width: 433px"><colgroup>
<col style="width: 156px">
<col style="width: 88px">
<col style="width: 91px">
<col style="width: 98px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Rec SDK</th>
    <th colspan="2">TorchNPU Version</th>
  </tr>
  <tr>
    <th>26.0.0</th>
    <th>26.1.0</th>
  </tr></thead>
<tbody>
  <tr>
    <td>26.0.0</td>
    <td>Y</td>
    <td>Y</td>
  </tr>
  <tr>
    <td>26.1.0</td>
    <td>/</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

## Usage Precautions

None

## Change Description

### New Features

|Feature|Description|Compatible Product Models|
|--|--|--|
|Rec SDK TensorFlow(`tf_rec_v1`)|<ul><li>Adapted for the Atlas 350 series products.</li></ul>|Atlas A2 series products<br>Atlas A3 series products<br>Atlas 350 series products|
|Rec SDK Torch(`torch_rec_v1`)|<ul><li>Adapted for the Atlas 350 series products.</li></ul>|Atlas A2 series products<br>Atlas A3 series products<br>Atlas 350 series products|
|Rec SDK TensorFlow(`tf_rec_v2`)|<ul><li>Adapted for the Atlas 350 series products.</li></ul>|Atlas 350 series products|
|Rec SDK Torch(`torch_rec_v2`)|<ul><li>Dynamic table eviction policies are supported: `TIMESTAMP`, `STEP`, `CUSTOMIZED`, and `LFU`.</li><li>The `HBM+DDR` sparse table cache mode is supported.</li><li>Incremental saving is supported.</li><li>The `EmbeddingBagCollection` scenario is supported.</li><li>The SGD, ADAM, AdaGrad, and RowWiseAdaGrad optimizers are supported.</li></ul>|Atlas 350 series products|
|Rec SDK operators|<ul><li>Fused operators for generative recommendation models, including `split_2d_jagged`, `concat_2d_jagged`, and `norm_multiply_dropout`, are supplemented and adapted for the Atlas 350 series products.</li><li>Operator performance optimization for the Atlas 350 series products: `hstu_v2`.</li></ul>|Atlas A2 series products<br>Atlas A3 series products<br>Atlas 350 series products|

### Interface Changes

**Rec SDK**

- Rec SDK TensorFlow (`tf_rec_v1`): No interface changes.
- Rec SDK Torch (`torch_rec_v1`): No interface changes.
- Rec SDK TensorFlow (`tf_rec_v2`): No interface changes.
- Rec SDK Torch (`torch_rec_v2`): No interface changes.

### Key Feature Changes

**Rec SDK**

- Rec SDK TensorFlow (`tf_rec_v1`): No key feature changes.
- Rec SDK Torch (`torch_rec_v1`): No key feature changes.
- Rec SDK TensorFlow (`tf_rec_v2`): No key feature changes.
- Rec SDK Torch (`torch_rec_v2`): No key feature changes.

### Resolved Issues

None

### Known Issues

None

## Upgrade Impact

### Impact on the System During the Upgrade

None

### Impact on the System After the Upgrade

None

## 26.1.0 Documentation

|Document|Description|Update Description|
|--|--|--|
|[Rec SDK 26.1.0 User Guide](../../README.md)|Introduces Rec SDK, its installation and deployment process, functions, model adaptation, and API reference.|For details, see *Rec SDK 26.1.0 User Guide*.|

## Virus Scan Results

Virus scan passed.

## Vulnerability Patch List

For details, see the [RecSDK Vulnerability Patch List](../zh/resources/RecSDK漏洞修补列表.xlsx).
