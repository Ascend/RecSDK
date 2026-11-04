# Rec SDK 26.2.0 Release Notes

## Key Features

The key highlights of Rec SDK 26.2.0 are as follows:

- Key point 1: The torch_rec_v1 framework is upgraded and refactored for torch versions, removing support for PyTorch/torch_npu 2.6.0 and narrowing the official support matrix to 2.7.1 and 2.10.0.
- Key point 2: torch_rec_v2 improves TorchRec API support, achieving 100% API support for TorchRec 1.2.0/1.5.0 and adapting to both TorchRec 1.6.0/1.7.0.
- Key point 3: The operator repository is refactored. FBGEMM operators are moved out to the fbgemm-ascend repository, and an independent ops-rec operator library is built.
- Key point 4: Compatible with CANN 9.2.0, Ascend HDK 26.2.0, and TorchNPU 26.2.0.

## Version Mapping

### Product Version Information

| Field | Value |
| --- | --- |
| Product | Rec SDK |
| Version | 26.2.0 |
| Version Type | Release version |
| Maintenance Period | See the [maintenance policy](https://gitcode.com/Ascend/RecSDK#torch_rec_v1-version-maintenance) |

### Related Product Version Mapping

**Table 1** Rec SDK software version mapping

| Rec SDK | CANN Version | Ascend HDK Version | TorchNPU Version |
| --- | --- | --- | --- |
| 26.2.0 | 9.2.0 | 26.2.0 | 26.2.0 |

### OS/Database Mapping

Rec SDK 26.2.0 images support the following OSs: Ubuntu 22.04 and openEuler 22.03 (x86_64/aarch64).

## Version Compatibility

> [!NOTE]
>
> In the tables in this section, "/" indicates that the versions are not compatible, and "Y" indicates that they are compatible.
>
> Software version compatibility means that when you upgrade the product software version, related software does not need to be upgraded or patched, and existing features remain supported.
>
> Rec SDK Torch (torch_rec_v1): Since 26.2.0, torch_rec_v1 no longer supports PyTorch/torch_npu 2.6.0, and the official support matrix is narrowed to PyTorch 2.7.1 and 2.10.0. After the upgrade, recompile `torchrec_npu` and the packages related to custom operators.
>
> Rec SDK Torch (torch_rec_v2): After the upgrade, recompile `torchrec_npu` and the packages related to custom operators.

<!-- npu="A3,910b" id1 -->
### Atlas A2/A3 Products

**Table 2** Rec SDK and CANN version compatibility

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
    <th colspan="4">CANN Version</th>
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

**Table 3** Rec SDK and Ascend HDK version compatibility

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
    <th colspan="4">Ascend HDK Version</th>
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

**Table 4** Rec SDK and TorchNPU version compatibility

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
    <th colspan="4">TorchNPU Version</th>
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
<!-- end id1 -->

<!-- npu="950" id2 -->
### Atlas 950PR Products

**Table 5** Rec SDK and CANN version compatibility

<table style="table-layout: fixed; width: 520px"><colgroup>
<col style="width: 156px">
<col style="width: 91px">
<col style="width: 91px">
<col style="width: 91px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Rec SDK</th>
    <th colspan="3">CANN Version</th>
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

**Table 6** Rec SDK and Ascend HDK version compatibility

<table style="table-layout: fixed; width: 520px"><colgroup>
<col style="width: 156px">
<col style="width: 91px">
<col style="width: 91px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Rec SDK</th>
    <th colspan="2">Ascend HDK Version</th>
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

**Table 7** Rec SDK and TorchNPU version compatibility

<table style="table-layout: fixed; width: 520px"><colgroup>
<col style="width: 156px">
<col style="width: 91px">
<col style="width: 91px">
<col style="width: 91px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Rec SDK</th>
    <th colspan="3">TorchNPU Version</th>
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
<!-- end id2 -->

## Usage Precautions

- Since 26.2.0, torch_rec_v1 no longer supports PyTorch/torch_npu 2.6.0. Users of 2.6.0 need to upgrade to PyTorch 2.7.1 or 2.10.0.
- torch_rec_v1 uses a single `hybrid_torchrec` package to adapt to TorchRec v1.2.0 (with PyTorch 2.7.1 + fbgemm_gpu 1.2.0) and TorchRec v1.5.0 (with PyTorch 2.10.0 + fbgemm_gpu 1.5.0).
- Since 26.2.0, custom operators in the torch_rec_v1 framework are used by importing `rec_cust_ops` + `fbgemm_ascend`. FBGEMM operators have been moved out to the fbgemm-ascend repository.

## Change Description

### New Features

|Feature|Description|Compatible Product Models|
|--|--|--|
|Rec SDK TensorFlow(`tf_rec_v1`)|Routine image and mapping table updates.|Atlas A2 training products<br>Atlas A3 training products<br>Atlas 950PR products|
|Rec SDK TensorFlow(`tf_rec_v2`)|Routine image and mapping table updates.|Atlas 950PR products|
|Rec SDK Torch(`torch_rec_v1`)|<ul><li>torch_rec_v1 is upgraded and refactored, removing support for PyTorch/torch_npu 2.6.0 and narrowing the support matrix to 2.7.1 and 2.10.0, and uses a single `hybrid_torchrec` package to adapt to torchrec v1.2.0 and v1.5.0.</li></ul>|Atlas A2 training products<br>Atlas A3 training products<br>Atlas 950PR products|
|Rec SDK Torch(`torch_rec_v2`)|<ul><li>TorchRec API support is improved, achieving 100% API support for TorchRec 1.2.0/1.5.0 and adapting to both TorchRec 1.6.0/1.7.0.</li></ul>|Atlas 950PR products|
|Rec SDK operators|<ul><li>FBGEMM operators are moved out to fbgemm-ascend.</li><li>An independent ops-rec operator library is built.</li></ul>|Atlas A2 training products<br>Atlas A3 training products<br>Atlas 950PR products|

### Key Feature Changes

**Rec SDK**

- Rec SDK Torch(`torch_rec_v1`): Support for PyTorch/torch_npu 2.6.0 is removed, and the official support matrix is narrowed to PyTorch 2.7.1 and 2.10.0. The version is refactored into a single `hybrid_torchrec` package.
- Rec SDK operators: FBGEMM operators are moved out to the fbgemm-ascend repository. The FBGEMM operators under `cust_op/framework/torch_plugin/torch_library` are now provided by `fbgemm_ascend`.
- Rec SDK TensorFlow(`tf_rec_v1`): No key feature changes
- Rec SDK TensorFlow(`tf_rec_v2`): No key feature changes
- Rec SDK Torch(`torch_rec_v2`): No key feature changes

### Interface Changes

**Rec SDK**

- Rec SDK Torch(`torch_rec_v1`): Since 26.2.0, custom operators are used by importing `rec_cust_ops` + `fbgemm_ascend`.
- Rec SDK TensorFlow(`tf_rec_v1`): No interface changes
- Rec SDK TensorFlow(`tf_rec_v2`): No interface changes
- Rec SDK Torch(`torch_rec_v2`): No interface changes

### Resolved Issues

None

### Known Issues

None

## Upgrade Impact

### Impact on the System During the Upgrade

None

### Impact on the System After the Upgrade

Since 26.2.0, torch_rec_v1 no longer supports PyTorch/torch_npu 2.6.0. Existing services based on 2.6.0 need to upgrade to PyTorch 2.7.1 or 2.10.0 and recompile the related component packages.

## Documentation

| Document | Description | Update Description |
| --- | --- | --- |
| [Rec SDK 26.2.0 User Guide](../README.md) | Introduces Rec SDK, its installation and deployment process, functions, model adaptation, and API reference. | For details, see *Rec SDK 26.2.0 User Guide*. |

## Virus Scan Results

Virus scan passed.

## Vulnerability Patch List

For details, see the [RecSDK Vulnerability Patch List](../resources/RecSDK_vulnerability_patch_list.xlsx).

## Revision History

| Document Version | Release Date | Change Description |
| --- | --- | --- |
| 01 | 2026-09-30 | First official release. |
