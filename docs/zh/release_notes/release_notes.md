# 版本说明

Rec SDK 26.1.0版本核心要点如下：

- 要点1：适配 Atlas 350 标卡；
- 要点2：新增动态表淘汰策略（TIMESTAMP/STEP/CUSTOMIZED/LFU）与 HBM+DDR 缓存模式；
- 要点3：算子补齐，如 split_2d_jagged 等；
- 要点4：兼容 CANN 9.1.0、Ascend HDK 26.1.0、TorchNPU 26.1.0。

## 版本配套说明

### 产品版本信息

<a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108__Ref249955742"></a>
<table><tbody><tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row244mcpsimp"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"></a>产品名称</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"></a><span id="ph925512229126"><a name="ph925512229126"></a><a name="ph925512229126"></a>Rec SDK</span></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row255mcpsimp"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"></a>产品版本</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"></a>26.1.0</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row7259721105019"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"></a>版本类型</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"></a>Release版本</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row7259721105020"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135014"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135014"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135014"></a>维护周期</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219502"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219502"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219502"></a>参考<a href="../../../README.md#torch_rec_v1-框架维护策略">维护策略</a></p>
</td>
</tr>
</tbody>
</table>

### 相关产品版本配套说明

**表 1**  Rec SDK软件版本配套表

| Rec SDK   | CANN版本 | Ascend HDK版本 |TorchNPU版本     |
| ------------ | ------------- | ------------ | -----------  |
| 26.1.0       | 9.1.0           | 26.1.0       |  26.1.0     |

## 版本兼容性说明

> [!NOTE]
>
> 本节表格中“/”表示不可配套，“Y”表示可配套。
>
> 软件版本兼容性是指产品软件版本升级时，其他关联软件不需要联动升级或打补丁，仍然可以支持已有功能。
>
> Rec SDK Torch（torch_rec_v1/v2）：在升级版本后，需要重新编译torchrec_npu和自定义算子相关包。

**表 2**  Rec SDK与CANN版本兼容

<table style="table-layout: fixed; width: 433px"><colgroup>
<col style="width: 156px">
<col style="width: 88px">
<col style="width: 91px">
<col style="width: 98px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Rec SDK</th>
    <th colspan="3">CANN版本</th>
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
    <td>/</td>
    <td>/</td>
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
    <td>/</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

**表 3**  Rec SDK与Ascend HDK版本兼容

<table style="table-layout: fixed; width: 433px"><colgroup>
<col style="width: 156px">
<col style="width: 88px">
<col style="width: 91px">
<col style="width: 98px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Rec SDK</th>
    <th colspan="3">Ascend HDK版本</th>
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
    <td>/</td>
    <td>/</td>
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
    <td>/</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

**表 4**  Rec SDK与TorchNPU版本兼容

<table style="table-layout: fixed; width: 433px"><colgroup>
<col style="width: 156px">
<col style="width: 88px">
<col style="width: 91px">
<col style="width: 98px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Rec SDK</th>
    <th colspan="3">TorchNPU版本</th>
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
    <td>/</td>
    <td>/</td>
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
    <td>/</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

## 版本使用注意事项

无

## 更新说明

### 新增特性

|特性名称|特性描述|配套产品型号|
|--|--|--|
|Rec SDK TensorFlow(tf_rec_v1)|<ul><li>适配Atlas 350 标卡。</li></ul>|Atlas 800T A2 训练服务器<br>Atlas 200T A2 Box16 异构子框<br>Atlas 800T A3 超节点服务器<br>Atlas 350 标卡|
|Rec SDK Torch(torch_rec_v1)|<ul><li>适配Atlas 350 标卡。</li></ul>|Atlas 800T A2 训练服务器<br>Atlas 200T A2 Box16 异构子框<br>Atlas 800T A3 超节点服务器<br>Atlas 350 标卡|
|Rec SDK TensorFlow(tf_rec_v2)|<ul><li>适配Atlas 350 标卡。</li></ul>|Atlas 350 标卡|
|Rec SDK Torch(torch_rec_v2)|<ul><li>支持动态表淘汰策略：TIMESTAMP、STEP、CUSTOMIZED、LFU</li><li>支持稀疏表缓存模式：HBM+DDR</li><li>支持增量保存功能</li><li>支持EmbeddingBagCollection场景</li><li>支持SGD、ADAM、AdaGrad、RowWiseAdaGrad优化器</li></ul>|Atlas 350 标卡|
|Rec SDK 算子|<ul><li>生成式推荐模型融合算子补齐，并适配Atlas 350标卡：split_2d_jagged、concat_2d_jagged、norm_multiply_dropout</li><li>Atlas 350标卡算子性能优化：hstu_v2</li></ul>|Atlas 800T A2 训练服务器<br>Atlas 200T A2 Box16 异构子框<br>Atlas 800T A3 超节点服务器<br>Atlas 350 标卡|

### 业务接口变更

**Rec SDK**

- Rec SDK TensorFlow(tf_rec_v1)：不涉及接口变更。
- Rec SDK Torch(torch_rec_v1)：不涉及接口变更。
- Rec SDK TensorFlow(tf_rec_v2)：不涉及接口变更。
- Rec SDK Torch(torch_rec_v2)：不涉及接口变更。

### 关键特性变更

**Rec SDK**

- Rec SDK TensorFlow(tf_rec_v1)：不涉及关键特性变更。
- Rec SDK Torch(torch_rec_v1)：不涉及关键特性变更。
- Rec SDK TensorFlow(tf_rec_v2)：不涉及关键特性变更。
- Rec SDK Torch(torch_rec_v2)：不涉及关键特性变更。

### 已解决的问题

无

### 遗留问题

无

## 升级影响

### 升级过程对现行系统的影响

无

### 升级后对现行系统的影响

无

## 26.1.0版本配套文档

|文档名称|内容简介|更新说明|
|--|--|--|
|《[Rec SDK 26.1.0 用户指南](../../../README.md)》|主要包括Rec SDK的简介、软件安装部署、功能特性、模型适配和相关的API接口参考。|变更详见《Rec SDK 26.1.0 用户指南》。|

## 病毒扫描结果

病毒扫描通过。

## 漏洞修补列表

详见《[RecSDK漏洞修补列表](../resources/RecSDK漏洞修补列表.xlsx)》。
