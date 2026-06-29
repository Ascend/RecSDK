# RecOps Running Case Documentation

## Introduction

### What Is RecOps

RecOps is a set of custom operators for recommendation scenarios built on Ascend C. It provides basic operator capabilities for framework components (`tf_rec_v1`, `tf_rec_v2`, `torch_rec_v1`, `torch_rec_v2`) and supports Atlas A2/A3/A5 devices.

**Core positioning:**

- A custom operator set for recommendation scenarios built on Ascend C.
- Focused on fusing and accelerating key compute paths such as `Embedding`, `Attention`, and `Optimizer`.
- Supports mainstream recommendation frameworks such as `tf_rec` and `torch_rec`.

**Repository:** [RecSDK repository on GitCode](https://gitcode.com/Ascend/RecSDK)

---

## Running Case

This case briefly introduces the HSTU_V1 forward operator `hstu_dense_forward` and how to compile and run it in an NPU environment.

### Introduction to the HSTU_V1 Forward Operator

**Hierarchical Sparse Transformer Unit (HSTU)** is a sparse attention fusion operator for recommendation scenarios. It fuses QK matrix multiplication, SiLU activation, scaling, mask application, and matrix multiplication with V into a single fused operator. This greatly reduces memory access overhead and kernel scheduling overhead, which enables high-performance training of recommendation models on Ascend NPUs.

In the HSTU_V1 version, we implement the **hstu_dense_forward** forward operator as the concrete Ascend NPU implementation of the HSTU fused operator. The following table compares its features with NV HSTU_V3.

<table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: center; font-size:14px;">
  <thead>
    <tr>
      <th></th>
      <th>Feature</th>
      <th>Functionality</th>
      <th>ASCEND HSTU_V1 (Atlas A2/A3 Ascend950PR/DT)</th>
      <th>NV HSTU_V3 (hopper, Ada, Ampere)<br>Releases/v25.11/7492d4b</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="13">HSTU forward features</td>
      <td rowspan="6">Basic attention functions</td>
      <td>Attention mechanism</td>
      <td>Supported</td>
      <td>Supported</td>
      <td>Implements the core part of HSTU attention.<br>S = Q * KT<br>P = silu(S)<br>O = P * V</td>
    </tr>
    <tr>
      <td>Multi-head attention (MHA)</td>
      <td>Supported</td>
      <td>Supported</td>
      <td>The number of query heads corresponds one to one with the number of key heads.</td>
    </tr>
    <tr>
      <td>Grouped-query attention (GQA)</td>
      <td>Supported</td>
      <td>Supported</td>
      <td>The number of query heads corresponds to the number of key heads in a many-to-one relationship.</td>
    </tr>
    <tr>
      <td>Variable-length sequence format</td>
      <td>Supported</td>
      <td>Supported</td>
      <td>Implements HSTU attention support for variable-length sequences.</td>
    </tr>
    <tr>
      <td>Floating-point calculation types</td>
      <td>Supports FP32, FP16, BF16</td>
      <td>Supports FP16, BF16</td>
      <td>The reference NV HSTU_V3 supports only FP16 and BF16 floating-point types.</td>
    </tr>
    <tr>
      <td>FP8 quantization types</td>
      <td>Supports <code>cast</code></td>
      <td>Supports <code>cast</code>,<br><code>per-block</code>,<br><code>per-head</code>,<br><code>per-batch</code>,<br><code>per-tensor</code></td>
      <td>Performs FP8 quantization at the element, block, head, batch, and tensor levels.</td>
    </tr>
    <tr>
      <td rowspan="2">Relative attention bias</td>
      <td>Whether <code>rab</code> is passed in</td>
      <td>Supports passing <code>rab</code>,<br>or omitting <code>rab</code>.</td>
      <td>Supports passing <code>rab</code> in <code>[b, n, s, s]</code> format,<br>or omitting <code>rab</code>.</td>
      <td>Implements the relative bias part of HSTU attention.</td>
    </tr>
    <tr>
      <td>Support for <code>rab</code> broadcasting</td>
      <td>Not supported</td>
      <td>Supports passing <code>rab</code> in <code>[b, 1, s, s]</code> format.</td>
      <td>Implements the <code>rab</code> broadcast function.</td>
    </tr>
    <tr>
      <td rowspan="4">Mask</td>
      <td>Support no mask</td>
      <td>Supported</td>
      <td>Supported</td>
      <td>Bidirectional mask.</td>
    </tr>
    <tr>
      <td>Support causal mask</td>
      <td>Supported (context + history + target)</td>
      <td>Supported (context + history + target)</td>
      <td>The causal mask is built from <code>num_context</code>, <code>num_targets</code>, and <code>target_group_size</code>. Any one of them can be <code>None</code>. For the specific parameter combinations, see the feature analysis.</td>
    </tr>
    <tr>
      <td>Support local mask</td>
      <td>Not supported</td>
      <td>Supported</td>
      <td>Local mask.</td>
    </tr>
    <tr>
      <td>Support arbitrary mask</td>
      <td>Not supported</td>
      <td>Supported</td>
      <td>User-defined mask parameters.</td>
    </tr>
    <tr>
      <td>Page</td>
      <td>Support Page feature</td>
      <td>Supported</td>
      <td>Supported</td>
      <td>Used to implement the page HSTU feature of the KV cache.</td>
    </tr>
  </tbody>
</table>

### Computing Principles

![alt text](./pic/hstu_v1_forward_image.png)

### hstu_dense_forward Operator File Structure

```shell
-- hstu_dense_forward
   |-- c310
      |-- op_kernel                # Kernel-side implementation of the hstu_dense_forward operator for A5.
      |-- run.sh                   # Installation script for the hstu_dense_forward operator for A5.
   |-- onnx_plugin                 # Supports ONNX model conversion for hstu_dense_forward.
   |-- v220
      |-- op_host                  # Host-side implementation of the hstu_dense_forward operator.
      |-- op_kernel                # Kernel-side implementation of the hstu_dense_forward operator.
      |-- pic                      # Operator implementation diagram.
      |-- hstu_dense_forward.json  # Operator prototype configuration.
      |-- run.sh                   # Installation script for the hstu_dense_forward operator for A2/A3.
   |-- README.md                   # Operator documentation for hstu_dense_forward.
```

### hstu_dense_forward Forward Inference Inputs and Outputs

| Name                | Input/Output | Data Type                             | Data Format                            | Range                                                                                                     | Description                                                                                                              |
|-------------------|-------|----------------------------------|---------------------------------|--------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| q                 | Input    | Tensor[float32/float16/bfloat16] | [B, S, N_q, D]/<br>[s_b, N, D]  | B∈[1, 2048]<br>S∈[1, 20480]<br>N_q∈[1, 16]<br>D∈[1, 512]                                       | B: batch_size, indicates the batch size.<br>S: seq_len, indicates the sequence length.<br>N: head_num, indicates the number of heads.<br>D: head_dim, indicates the dimension.<br>s_b is the sum of the actual sequence lengths of each batch in jagged format. |
| k                 | Input    | Tensor[float32/float16/bfloat16] | [B, S, N_k, D]/<br>[s_b, N, D]  | Same as q                                                                                                     | **GQA support**: The number of K heads can be smaller than the number of Q heads, but N_q must be divisible by N_k.                                                                          |
| v                 | Input    | Tensor[float32/float16/bfloat16] | [B, S, N_k, D]/<br>[s_b, N, D]  | B∈[1, 2048]<br>S∈[1, 20480]<br>N_q∈[1, 16]<br>D∈[16, 512] and is a multiple of 16                                                                                                     | Same as k.                                                                                                              |
| mask              | Input    | Tensor[float32/float16/bfloat16] | [B, N, S, S]                    | NA                                                                                                     | `S` is the maximum sequence length of the model, `max_seq_len`.<br>Pass `None` when you do not use a mask. The type must match `q`.<br>`N` must stay consistent with `q`.                                                      |
| attn_bias         | Input    | Tensor[float32/float16/bfloat16] | [B, N, S, S]                    | NA                                                                                                     | `S` is the maximum sequence length of the model, `max_seq_len`.<br>Pass `None` when you do not use `attn_bias`. The type must match `q`.<br>`N` must stay consistent with `q`.                                                |
| seq_offsets_q     | Input    | Tensor[int32_t/int64_t]                  | [B + 1]                         | NA                                                                                                     | Indicates the actual Q sequence-length offset of each batch, increasing from 0. The user must ensure validity. This takes effect only in jagged format.                                                             |
| seq_offsets_k     | Input    | Tensor[int32_t/int64_t]                  | [B + 1]                         | NA                                                                                                     | Indicates the actual K sequence-length offset of each batch, increasing from 0. The user must ensure validity. This takes effect only in jagged format.                                                             |
| seq_offsets_t     | Input    | Tensor[int32_t/int64_t]                  | [B + 1]                         | NA                                                                                                     | Target sequence offset tensor.                                                                                                       |
| kv_cache          | Input    | Tensor[float32/float16/bfloat16] | [num_pages, 2, page_size, N, D] | page_size∈{32, 128, 256}                                                                               | KV cache tensor used to store historical Key-Value pairs.                                                                                         |
| page_offsets      | Input    | Tensor[int32_t/int64_t]                  | [B + 1]                         | NA                                                                                                     | Page offset tensor.                                                                                                         |
| page_ids          | Input    | Tensor[int32_t/int64_t]                  | [page_offsets[-1]]              | NA                                                                                                     | Page ID tensor.                                                                                                          |
| last_page_len     | Input    | Tensor[int32_t/int64_t]                  | [B]                             | NA                                                                                                     | Last page length tensor.                                                                                                        |
| num_context       | Input    | Tensor[int32_t/int64_t]                  | [B]                             | The value range is [0, 256]. Other values are not restricted or monitored.                                                                               | Context count tensor.                                                                                                         |
| num_target        | Input    | Tensor[int32_t/int64_t]                  | [B]                             | The value range is [0, 512]. Other values are not restricted or monitored.                                                                               | Target count tensor.                                                                                                          |
| mask_type         | Input    | int                              | NA                              | 0: Use the built-in lower triangular mask. No mask needs to be passed in.<br>1: Use the built-in upper triangular mask. No mask needs to be passed in. Currently not supported.<br>2: Do not use a mask.<br>3: Use a custom mask. In this case, the user must define and pass in the mask. | NA                                                                                                              |
| max_seq_len_q     | Input    | int                              | NA                              | [1, 20480]                                                                                             | Indicates the maximum Q sequence length of the model.                                                                                                     |
| max_seq_len_k     | Input    | int                              | NA                              | [1, 20480]                                                                                             | Indicates the maximum K sequence length of the model.                                                                                                     |
| silu_scale        | Input    | float                            | NA                              | NA                                                                                                     | Supports a user-defined value. If no value is passed in, the default value is `1/max_seq_len`.                                                                                  |
| layout            | Input    | string                           | NA                              | `"normal"`: indicates that the data format of q, k, and v is `[B, S, N, D]`.<br>`"jagged"`: indicates that the data format of q, k, and v is `[s_b, N, D]`.                                  | NA                                                                                                              |
| target_group_size | Input    | int                              | NA                              | Currently, only `{0, 1, 3}` is monitored. Other values are not restricted or monitored.                                                                             | Used when the built-in target mask is created. When `target_group_size` is 0, no target mask is created.                                                           |
| is_delta_qk       | Input    | int                              | NA                              | NA                                                                                                     | Whether the QK sequences have the same length: `0` = same length, `1` = different lengths.                                                                                             |
| alpha             | Input    | float                            | NA                              | NA                                                                                                     | Alpha scaling parameter.                                                                                                       |
| attn_output       | Output   | Tensor[float32/float16/bfloat16] | [B, S, N, D]/<br>[s_b, N, D]    | Same as q                                                                                                     | Same as q.                                                                                                              |

Note:

* The data in the B, S, N, and D dimensions cannot be 0. If any of them is 0, the operator input is empty and the operator does not run.
* The B, S, and N parameters affect the device memory occupied by bias and mask. Set the parameter values based on actual memory usage.

### Operating Environment Dependencies

#### Hardware Environment

| Hardware Model | Supported |
|---------|---------|
| Atlas A2 training series products | Yes |
| Atlas A3 training series products | Yes |
| Atlas A5 training series products | Yes |
| Atlas inference series products | Yes |

#### Software Dependencies

**CANN**: Ascend CANN toolkit. Ensure that the environment variables are set correctly.

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Currently, two software version pairings are supported: PyTorch 2.6.0 and PyTorch 2.7.1. Before calling the operator, you must install the matching software stack and the required operators. The detailed mappings are as follows:

| Compatible Version | PyTorch | torch-npu | torchrec  | fbgemm_gpu | hybrid_torchrec |
| ------------------ | ------- | --------- | --------- | ---------- | --------------- |
| 1                  | 2.6.0   | 2.6.0     | 1.1.0+npu | 1.1.0      | 1.1.0           |
| 2                  | 2.7.1   | 2.7.1     | 1.2.0+npu | 1.2.0      | 1.2.0           |

### Single-Operator Usage Instructions

#### Operator Compilation

Enter the implementation directory of the HSTU_V1 forward operator (`cust_op/ascendc_op/ai_core_op/hstu_dense_forward`, A5 under `c310`, A2/A3 under `v220`) and run the compilation and deployment command. By default, this compiles and installs the AI Core type for Atlas A2 training series products.

If you specify the AI Core type for compilation:

```shell
bash run.sh --ai-core ai_core-(soc_version)
```

> Obtain `soc_version` for the AI processor model as follows:
>
> - Run the `npu-smi info` command on the server where the Ascend AI processor is installed to query the `Chip Name` field. The actual configuration value is `Ascend` plus the chip name. For example, if `Chip Name` is `xxxyy`, the actual configuration value is `Ascendxxxyy`.
>
> Operator projects created for AI processor models in the same series share the same basic functions for operator development, compilation, and deployment.

#### Operator Adaptation Layer Compilation

Enter the adaptation layer directory of the HSTU_V1 forward operator (`cust_op/framework/torch_plugin/torch_library/hstu`) and compile the operator adaptation layer.

```shell
bash build_ops.sh
```

After the command finishes, the `xxx.so` file is generated in the current `build` directory. When you call the operator, run the following command to load it.

```python
import torch
torch.ops.load_library("path/to/build/xxx.so")  # Replace this with the absolute path of the .so file.
```

#### Single-Operator Running Case

```python
import torch
import torch_npu
torch.ops.load_library("path/to/build/xxx.so")  # Replace this with the absolute path of the previously generated .so file.

# GQA configuration: 8 Q heads and 2 K/V heads.
batch_size = 2
seq_len = 256
num_heads_q = 8    # Number of Q heads.
num_heads_k = 2    # Number of K/V heads in GQA mode.
head_dim = 64

# Generate data.
q = torch.randn(batch_size * seq_len, num_heads_q, head_dim, dtype=torch.float16).npu()
k = torch.randn(batch_size * seq_len, num_heads_k, head_dim, dtype=torch.float16).npu()  # The number of K heads is smaller than the number of Q heads.
v = torch.randn(batch_size * seq_len, num_heads_k, head_dim, dtype=torch.float16).npu()  # The number of V heads is equal to the number of K heads.

# Call the operator in jagged format.
seq_offsets_q = torch.tensor([0, 128, 256], dtype=torch.int64).npu()
seq_offsets_k = torch.tensor([0, 128, 256], dtype=torch.int64).npu()

output = torch.ops.mxrec.hstu_jagged(
    q=q,
    k=k,
    v=v,
    mask=None,
    attn_bias=None,
    mask_type=0,  # Lower triangular mask.
    max_seq_len=256,
    max_seq_len_k=256,
    silu_scale=1.0/256,
    seq_offset=seq_offsets_q,
    seq_offset_k=seq_offsets_k
)

# Output shape: [batch_size * seq_len, num_heads_q, head_dim]
print(output.shape)  # torch.Size([512, 8, 64])
```

Note:

* For details about the `hstu_dense_forward` operator, see `cust_op/ascendc_op/ai_core_op/hstu_dense_forward/README.md`.
* For test cases of the `hstu_dense_forward` operator, see the `cust_op/test/hstu_dense/torch` directory.
