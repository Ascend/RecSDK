# Rec SDK 26.2.0 版本说明书

# 关键特性

Rec SDK 26.2.0 版本核心要点如下：

- 要点1：torch_rec_v1 框架 torch 版本升级重构，移除 PyTorch/torch_npu 2.6.0 版本支持，官方支持矩阵收敛到 2.7.1 与 2.10.0；
- 要点2：torch_rec_v2 完善 TorchRec API 支持度，实现 TorchRec 1.2.0/1.5.0 100% API 支持，并适配 TorchRec 1.6.0/1.7.0 双版本；
- 要点3：算子仓重构，FBGEMM 类算子迁出至 fbgemm-ascend 仓库，构建独立 ops-rec 算子库；
- 要点4：配套 CANN 9.2.0、Ascend HDK 26.2.0、TorchNPU 26.2.0。

# 版本配套说明

## 产品版本信息

| 字段 | 取值 |
| --- | --- |
| 产品名称 | Rec SDK |
| 产品版本 | 26.2.0 |
| 版本类型 | Release版本 |
| 维护周期 | 参考[维护策略](https://gitcode.com/Ascend/RecSDK#torch_rec_v1-框架维护策略) |

## 相关产品版本配套说明

**表 1**  Rec SDK软件版本配套表

| Rec SDK | CANN版本 | Ascend HDK版本 | TorchNPU版本 |
| --- | --- | --- | --- |
| 26.2.0 | 9.2.0 | 26.2.0 | 26.2.0 |

## 与操作系统/数据库配套说明

Rec SDK 26.2.0 镜像支持操作系统：Ubuntu 22.04、openEuler 22.03（x86_64 / aarch64）。

# 版本兼容性说明

> [!NOTE]
>
> 本节表格中“/”表示不可配套，“Y”表示可配套。
>
> 软件版本兼容性是指产品软件版本升级时，其他关联软件不需要联动升级或打补丁，仍然可以支持已有功能。
>
> Rec SDK Torch（torch_rec_v1）：torch_rec_v1 自 26.2.0 起移除对 PyTorch/torch_npu 2.6.0 的支持，官方支持矩阵收敛到 PyTorch 2.7.1 与 2.10.0；升级后需重新编译 torchrec_npu 和自定义算子相关包。
>
> Rec SDK Torch（torch_rec_v2）：在升级版本后，需要重新编译 torchrec_npu 和自定义算子相关包。

## Atlas A2/A3系列产品

**表 2**  Rec SDK与CANN版本兼容

<table style="table-layout: fixed; width: 520px"><colgroup>
<col style="width: 156px">
<col style="width: 91px">
<col style="width: 91px">
<col style="width: 91px">
<col style="width: 91px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Rec SDK</th>
    <th colspan="4">CANN版本</th>
  </tr>
  <tr>
    <th>8.5.0</th>
    <th>9.0.0</th>
    <th>9.1.0</th>
    <th>9.2.0</th>
  </tr></thead>
<tbody>
  <tr><td>7.3.0</td><td>Y</td><td>Y</td><td>Y</td><td>Y</td></tr>
  <tr><td>26.0.0</td><td>/</td><td>Y</td><td>Y</td><td>Y</td></tr>
  <tr><td>26.1.0</td><td>/</td><td>Y</td><td>Y</td><td>Y</td></tr>
  <tr><td>26.2.0</td><td>/</td><td>/</td><td>Y</td><td>Y</td></tr>
</tbody>
</table>

**表 3**  Rec SDK与Ascend HDK版本兼容

<table style="table-layout: fixed; width: 520px"><colgroup>
<col style="width: 156px">
<col style="width: 91px">
<col style="width: 91px">
<col style="width: 91px">
<col style="width: 91px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Rec SDK</th>
    <th colspan="4">Ascend HDK版本</th>
  </tr>
  <tr>
    <th>25.5.0</th>
    <th>26.0.RC1</th>
    <th>26.1.0</th>
    <th>26.2.0</th>
  </tr></thead>
<tbody>
  <tr><td>7.3.0</td><td>Y</td><td>Y</td><td>Y</td><td>Y</td></tr>
  <tr><td>26.0.0</td><td>/</td><td>Y</td><td>Y</td><td>Y</td></tr>
  <tr><td>26.1.0</td><td>/</td><td>Y</td><td>Y</td><td>Y</td></tr>
  <tr><td>26.2.0</td><td>/</td><td>/</td><td>Y</td><td>Y</td></tr>
</tbody>
</table>

**表 4**  Rec SDK与TorchNPU版本兼容

<table style="table-layout: fixed; width: 520px"><colgroup>
<col style="width: 156px">
<col style="width: 91px">
<col style="width: 91px">
<col style="width: 91px">
<col style="width: 91px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Rec SDK</th>
    <th colspan="4">TorchNPU版本</th>
  </tr>
  <tr>
    <th>7.3.0</th>
    <th>26.0.0</th>
    <th>26.1.0</th>
    <th>26.2.0</th>
  </tr></thead>
<tbody>
  <tr><td>7.3.0</td><td>Y</td><td>Y</td><td>Y</td><td>Y</td></tr>
  <tr><td>26.0.0</td><td>/</td><td>Y</td><td>Y</td><td>Y</td></tr>
  <tr><td>26.1.0</td><td>/</td><td>Y</td><td>Y</td><td>Y</td></tr>
  <tr><td>26.2.0</td><td>/</td><td>/</td><td>Y</td><td>Y</td></tr>
</tbody>
</table>

## Ascend 950PR&950DT系列产品

**表 5**  Rec SDK与CANN版本兼容

<table style="table-layout: fixed; width: 520px"><colgroup>
<col style="width: 156px">
<col style="width: 91px">
<col style="width: 91px">
<col style="width: 91px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Rec SDK</th>
    <th colspan="3">CANN版本</th>
  </tr>
  <tr>
    <th>9.0.0</th>
    <th>9.1.0</th>
    <th>9.2.0</th>
  </tr></thead>
<tbody>
  <tr><td>26.0.0</td><td>Y</td><td>Y</td><td>Y</td></tr>
  <tr><td>26.1.0</td><td>/</td><td>Y</td><td>Y</td></tr>
  <tr><td>26.2.0</td><td>/</td><td>Y</td><td>Y</td></tr>
</tbody>
</table>

**表 6**  Rec SDK与Ascend HDK版本兼容

<table style="table-layout: fixed; width: 520px"><colgroup>
<col style="width: 156px">
<col style="width: 91px">
<col style="width: 91px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Rec SDK</th>
    <th colspan="2">Ascend HDK版本</th>
  </tr>
  <tr>
    <th>25.7.RC1</th>
    <th>26.2.0</th>
  </tr></thead>
<tbody>
  <tr><td>26.0.0</td><td>Y</td><td>Y</td></tr>
  <tr><td>26.1.0</td><td>Y</td><td>Y</td></tr>
  <tr><td>26.2.0</td><td>Y</td><td>Y</td></tr>
</tbody>
</table>

**表 7**  Rec SDK与TorchNPU版本兼容

<table style="table-layout: fixed; width: 520px"><colgroup>
<col style="width: 156px">
<col style="width: 91px">
<col style="width: 91px">
<col style="width: 91px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Rec SDK</th>
    <th colspan="3">TorchNPU版本</th>
  </tr>
  <tr>
    <th>26.0.0</th>
    <th>26.1.0</th>
    <th>26.2.0</th>
  </tr></thead>
<tbody>
  <tr><td>26.0.0</td><td>Y</td><td>Y</td><td>Y</td></tr>
  <tr><td>26.1.0</td><td>/</td><td>Y</td><td>Y</td></tr>
  <tr><td>26.2.0</td><td>/</td><td>Y</td><td>Y</td></tr>
</tbody>
</table>

# 版本使用注意事项

- torch_rec_v1 自 26.2.0 起移除对 PyTorch/torch_npu 2.6.0 的支持，使用 2.6.0 的用户需升级至 PyTorch 2.7.1 或 2.10.0。
- torch_rec_v1 采用单一 `hybrid_torchrec` 包适配 TorchRec v1.2.0（配套 PyTorch 2.7.1 + fbgemm_gpu 1.2.0）与 TorchRec v1.5.0（配套 PyTorch 2.10.0 + fbgemm_gpu 1.5.0）。
- 自 26.2.0 起，torch_rec_v1 框架中自定义算子通过导入 `rec_cust_ops` + `fbgemm_ascend` 的方式使用，FBGEMM 类算子已迁出至 fbgemm-ascend 仓库。

# 更新说明

## 新增特性说明

|特性名称|特性描述|配套产品型号|
|--|--|--|
|Rec SDK TensorFlow(tf_rec_v1)|例行镜像、配套表更新。|Atlas A2训练系列产品<br>Atlas A3训练系列产品<br>Ascend 950PR&950DT系列产品|
|Rec SDK TensorFlow(tf_rec_v2)|例行镜像、配套表更新。|Ascend 950PR&950DT系列产品|
|Rec SDK Torch(torch_rec_v1)|<ul><li>torch_rec_v1 版本升级重构，移除 PyTorch/torch_npu 2.6.0 支持，收敛到 2.7.1 与 2.10.0，采用单一 hybrid_torchrec 包适配 torchrec v1.2.0 与 v1.5.0。</li></ul>|Atlas A2训练系列产品<br>Atlas A3训练系列产品<br>Ascend 950PR&950DT系列产品|
|Rec SDK Torch(torch_rec_v2)|<ul><li>完善 TorchRec API 功能支持，实现 TorchRec 1.2.0/1.5.0 100% API 支持度，适配 TorchRec 1.6.0/1.7.0 双版本。</li></ul>|Atlas A2训练系列产品<br>Atlas A3训练系列产品<br>Ascend 950PR&950DT系列产品|
|Rec SDK 算子|<ul><li>FBGEMM 类算子迁出至 fbgemm-ascend。</li><li>构建独立 ops-rec 算子库。</li></ul>|Atlas A2训练系列产品<br>Atlas A3训练系列产品<br>Ascend 950PR&950DT系列产品|

## 关键特性变更

**Rec SDK**

- Rec SDK Torch(torch_rec_v1)：移除对 PyTorch/torch_npu 2.6.0 版本的支持，官方支持矩阵收敛到 PyTorch 2.7.1 与 2.10.0；版本重构为单一 `hybrid_torchrec` 包。
- Rec SDK 算子：FBGEMM 类算子迁出至 fbgemm-ascend 仓库，cust_op/framework/torch_plugin/torch_library 下的 FBGEMM 类算子改由 `fbgemm_ascend` 提供。
- Rec SDK TensorFlow(tf_rec_v1)：不涉及关键特性变更。
- Rec SDK TensorFlow(tf_rec_v2)：不涉及关键特性变更。
- Rec SDK Torch(torch_rec_v2)：不涉及关键特性变更。

## 业务接口变更

**Rec SDK**

- Rec SDK Torch(torch_rec_v1)：自 26.2.0 起采用导入 `rec_cust_ops` + `fbgemm_ascend` 的方式使用自定义算子。
- Rec SDK TensorFlow(tf_rec_v1)：不涉及接口变更。
- Rec SDK TensorFlow(tf_rec_v2)：不涉及接口变更。
- Rec SDK Torch(torch_rec_v2)：不涉及接口变更。

## 已解决的问题

无

## 遗留问题

无

# 升级影响

## 升级过程对现行系统的影响

无

## 升级后对现行系统的影响

torch_rec_v1 自 26.2.0 起不再支持 PyTorch/torch_npu 2.6.0，基于 2.6.0 的存量业务需升级至 PyTorch 2.7.1 或 2.10.0 并重新编译相关组件包。

# 版本配套文档

| 文档名称 | 内容简介 | 更新说明 |
| --- | --- | --- |
| 《Rec SDK 26.2.0 用户指南》 | 主要包括 Rec SDK 的简介、软件安装部署、功能特性、模型适配和相关的 API 接口参考。 | 变更详见《Rec SDK 26.2.0 用户指南》。 |

# 病毒扫描结果

病毒扫描通过。

# 漏洞修补列表

详见《[RecSDK漏洞修补列表](<../resources/RecSDK 漏洞修补列表.xlsx>)》。

# 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| --- | --- | --- |
| 01 | 2026-09-30 | 第一次正式发布。 |
