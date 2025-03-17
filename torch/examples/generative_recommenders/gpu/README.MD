# triton适配说明
## 适配说明
1. 源码地址：[generative-recommenders](https://github.com/facebookresearch/generative-recommenders)
2. commit: bb389f9539b054e7268528efcd35457a6ad52439
## Python、pytorch、fbgemm-gpu、CUDA版本
| 名称 | 版本 |
| ----- | ----- |
| python | 3.11.6 |
| pytorch | 2.4.1 |
| CUDA | 12.1 |
| fbgemm-gpu | 0.8.0 |
## 依赖安装
修改requirements.txt
```
torch==2.4.0
torch_tensorrt==2.4.0
torchaudio==2.4.0
torchvison==0.19
fbgemm-gpu==0.8.0
```
执行命令
`pip install -r requirements.txt`
## 数据下载
`mkdir -p tmp/ && python3 preprocess_public_data.py`
## 适配
1. generative_recommenders/modeling/sequential/hstu.py  
原46行替换成如下代码
```python
from generative_recommenders.ops.triton.triton_hstu_attention import (
    triton_hstu_mha,
    triton_hstu_attention_fwd,
    triton_hstu_attention_bwd,
)
from generative_recommenders.ops.pytorch.pt_hstu_attention import (
    pytorch_hstu_mha
)
from generative_recommenders.common  import (
    generate_sparse_seq_len,
    gpu_unavailable,
    HammerKernel,
    set_dev_mode,
)
from generative_recommenders.ops.hstu_attention import hstu_mha
TIMESTAMPS_KEY = "timestamps"
ref_kernel=HammerKernel.PYTORCH,
real_kernel=HammerKernel.TRITON,

class AttentionFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        N: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
        causal: bool,
        num_targets: Optional[torch.Tensor],
        max_attn_len: Optional[int],
        contextual_seq_len: int,
        sort_by_length: bool,
    ) -> torch.Tensor:
        sort_by_length_indices = None
        if sort_by_length:
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            _, sort_by_length_indices = torch.sort(
                seq_lengths, descending=True, stable=False
            )
        saved_tensors = [q, k, v, seq_offsets]
        if num_targets is not None:
            saved_tensors.append(num_targets)
        max_attn_len = max_attn_len or 0
        if sort_by_length_indices is not None:
            saved_tensors.append(sort_by_length_indices)
        ctx.save_for_backward(*saved_tensors)
        ctx.alpha = alpha
        ctx.causal = causal
        ctx.has_multiple_targets = num_targets is not None
        ctx.max_attn_len = max_attn_len
        ctx.N = N
        ctx.contextual_seq_len = contextual_seq_len
        ctx.sort_by_length = sort_by_length
        return triton_hstu_attention_fwd(
            N=N,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=causal,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length_indices=sort_by_length_indices,
        )
    
    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dout: torch.Tensor
    ) -> Tuple[
        None,
        None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        with torch.inference_mode():
            q, k, v, seq_offsets = ctx.saved_tensors[:4]
            idx = 4
            if ctx.has_multiple_targets:
                num_targets = ctx.saved_tensors[idx]
                idx += 1
            else:
                num_targets = None
            if ctx.sort_by_length:
                sort_by_length_indices = ctx.saved_tensors[idx]
            else:
                sort_by_length_indices = None
            
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            dq, dk, dv = triton_hstu_attention_bwd(
                dout=dout,
                q=q,
                k=k,
                v=v,
                dq=dq,
                dk=dk,
                dv=dv,
                seq_offsets=seq_offsets,
                num_targets=num_targets,
                N=ctx.N,
                alpha=ctx.alpha,
                max_attn_len=ctx.max_attn_len,
                causal=ctx.causal,
                contextual_seq_len=ctx.contextual_seq_len,
                sort_by_length_indices=sort_by_length_indices,
            )
            return (
                None,
                None,
                dq,
                dk,
                dv,
                None,
                None,
                None,
                None,
                None,
                None,
            )
```
SequentialTransductionUnitJagged类里，初始化__init__里`self,`下添加`max_seq_len: int,`，`super().__init__()`下添加`self._max_seq_len: int = max_seq_len`  
SequentialTransductionUnitJagged类里，forward函数里, `cached_k = None`下添加
```python
        alpha = 1.0
        B: int = x_offsets.size(0) - 1
```
将
```
        B: int = x_offsets.size(0) - 1
        if self._normalization == "rel_bias" or self._normalization == "hstu_rel_bias":
            assert self._rel_attn_bias is not None
            attn_output, padded_q, padded_k = _hstu_attention_maybe_from_cache(
                num_heads=self._num_heads,
                attention_dim=self._attention_dim,
                linear_dim=self._linear_dim,
                q=q,
                k=k,
                v=v,
                cached_q=cached_q,
                cached_k=cached_k,
                delta_x_offsets=delta_x_offsets,
                x_offsets=x_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                rel_attn_bias=self._rel_attn_bias,
            )
        elif self._normalization == "softmax_rel_bias":
            if delta_x_offsets is not None:
                B = x_offsets.size(0) - 1
                padded_q, padded_k = cached_q, cached_k
                flattened_offsets = delta_x_offsets[1] + torch.arange(
                    start=0,
                    end=B * n,
                    step=n,
                    device=delta_x_offsets[1].device,
                    dtype=delta_x_offsets[1].dtype,
                )
                assert padded_q is not None
                assert padded_k is not None
                padded_q = (
                    padded_q.view(B * n, -1)
                    .index_copy_(
                        dim=0,
                        index=flattened_offsets,
                        source=q,
                    )
                    .view(B, n, -1)
                )
                padded_k = (
                    padded_k.view(B * n, -1)
                    .index_copy_(
                        dim=0,
                        index=flattened_offsets,
                        source=k,
                    )
                    .view(B, n, -1)
                )
            else:
                padded_q = torch.ops.fbgemm.jagged_to_padded_dense(
                    values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
                )
                padded_k = torch.ops.fbgemm.jagged_to_padded_dense(
                    values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
                )

            qk_attn = torch.einsum("bnd,bmd->bnm", padded_q, padded_k)
            if self._rel_attn_bias is not None:
                qk_attn = qk_attn + self._rel_attn_bias(all_timestamps)
            qk_attn = F.softmax(qk_attn / math.sqrt(self._attention_dim), dim=-1)
            qk_attn = qk_attn * invalid_attn_mask
            attn_output = torch.ops.fbgemm.dense_to_jagged(
                torch.bmm(
                    qk_attn,
                    torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]),
                ),
                [x_offsets],
            )[0]
        else:
            raise ValueError(f"Unknown normalization method {self._normalization}")
```
替换成
```python
        bs = q.size(0)
        q = q.contiguous().view(bs, self._num_heads, self._attention_dim)
        k = k.contiguous().view(bs, self._num_heads, self._attention_dim)
        v = v.contiguous().view(bs, self._num_heads, self._linear_dim)

        attn_op = AttentionFunction()
        attn_output = attn_op.apply(
            self._max_seq_len,
            alpha,
            q,
            k,
            v,
            x_offsets,
            True,
            None,
            None,
            0,
            False,
        )

        attn_output = attn_output.contiguous().view(bs, self._num_heads*self._attention_dim)
        B: int = x_offsets.size(0) - 1
```
最后替换return值，原442行改成`return new_outputs, (v, cached_q, cached_k, new_outputs)`  
HSTU类初始化里  
将
```
        self._hstu = HSTUJagged(
            modules=[
                SequentialTransductionUnitJagged(
                    embedding_dim=self._embedding_dim,
                    linear_hidden_dim=linear_dim,
                    attention_dim=attention_dim,
                    normalization=normalization,
                    linear_config=linear_config,
                    linear_activation=linear_activation,
                    num_heads=num_heads,
                    # TODO: change to lambda x.
                    relative_attention_bias_module=(
                        RelativeBucketedTimeAndPositionBasedBias(
                            max_seq_len=max_sequence_len
                            + max_output_len,  # accounts for next item.
                            num_buckets=128,
                            bucketization_fn=lambda x: (
                                torch.log(torch.abs(x).clamp(min=1)) / 0.301
                            ).long(),
                        )
                        if enable_relative_attention_bias
                        else None
                    ),
                    dropout_ratio=linear_dropout_rate,
                    attn_dropout_ratio=attn_dropout_rate,
                    concat_ua=concat_ua,
                    precision=precision,
                )
                for _ in range(num_blocks)
            ],
            autocast_dtype=None,
        )
```
换成
```
        self._hstu = HSTUJagged(
            modules=[
                SequentialTransductionUnitJagged(
                    max_seq_len=max_sequence_len+max_output_len, 
                    embedding_dim=self._embedding_dim,
                    linear_hidden_dim=linear_dim,
                    attention_dim=attention_dim,
                    normalization=normalization,
                    linear_config=linear_config,
                    linear_activation=linear_activation,
                    num_heads=num_heads,
                    dropout_ratio=linear_dropout_rate,
                    attn_dropout_ratio=attn_dropout_rate,
                    concat_ua=concat_ua,
                    precision=precision,
                )
                for _ in range(num_blocks)
            ],
            autocast_dtype=None,
        )
```

2. generative_recommenders/common.py  
注释83-84行
```
# pre_hook=None
# post_hook=None
```

3. trainer/train.py  
原69行添加代码
```
from torch.utils.data import Dataset

class RepeatDataset(Dataset):
    def __init__(self, dataset: Dataset, num_repeats: int):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return len(self.dataset) * self.num_repeats

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]
```
将
```
    train_data_sampler, train_data_loader = create_data_loader(
        dataset.train_dataset,
        batch_size=local_batch_size,
        world_size=world_size,
        rank=rank,
        shuffle=True,
        drop_last=world_size > 1,
    )
```
改成
```
    train_dataset = RepeatDataset(dataset.train_dataset, num_repeats=100)
    train_data_sampler, train_data_loader = create_data_loader(
        train_dataset,
        batch_size=local_batch_size,
        world_size=world_size,
        rank=rank,
        shuffle=True,
        drop_last=world_size > 1,
    )
```
注释`torch.autograd.set_detect_anomaly(True)`

4. generative_recommenders/ops/triton/triton_hstu_attention.py
第1119行，将  
`    for start_m in tl.range(low, high, BLOCK_M, loop_unroll_factor=UNROLL):`  
修改成  
`    for start_m in tl.range(low, high, BLOCK_M, num_stages=UNROLL):`

### precision_mode模式
### 相关依赖  
安装 [mindstudio-probe](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/01.installation.md)  

### 精度对比
使用mindstudio-probe开启debugger，可对整网进行精度对比，使用参考链接： https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/devaids/quickstart/pttraingquickstart/pttools_qucikstart_0006.html

### 适配
1. generative_recommenders/modeling/sequential/embedding_modules.py  
LocaEmbeddingModule类__init__函数下  
添加参数precision_mode: bool = False  
把`self.reset_params`替换成
```
        if precision_mode:
            self._item_emb.weight.data.fill_(1)
            self._item_emb.weight.data[0] = 0
        else:
            self.reset_params()
```
2. generative_recommenders/modeling/sequential/hstu.py  
SequentialTransductionUnitJagged类__init__函数下  
添加参数precision_mode: bool = False  
将
```
            self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
              torch.empty(
                     (
                         embedding_dim,
                         linear_hidden_dim * 2 * num_heads
                         + attention_dim * num_heads * 2,
                     )
                 ).normal_(mean=0, std=0.02),
            )
```
替换成
```
            if not precision_mode:
                self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
                    torch.empty(
                        (
                            embedding_dim,
                            linear_hidden_dim * 2 * num_heads
                            + attention_dim * num_heads * 2,
                        )
                    ).normal_(mean=0, std=0.02),
                )
            else:
                self._uvqk = torch.nn.Parameter(
                    torch.empty(
                        (
                            embedding_dim,
                            linear_hidden_dim * 2 * num_heads
                            + attention_dim * num_heads * 2,
                        )
                    ).fill_(0.01),
                )
```
HSTU __init__添加参数precision_mode: bool = False  
初始化self._hstu下的SequentialTransductionUnitJagged里添加precision_mode=precision_mode参数  
__init__初始化函数，`self.reset_params()`上面添加`self.precision_mode = precision_mode`
HSTU reset_params函数下  
将`torch.nn.init.xavier_normal_(params.data)`替换成
```
                if not self.precision_mode:
                    torch.nn.init.xavier_normal_(params.data)
                else:
                    fixed_value = 1
                    params.data.fill_(fixed_value)
```

3. trainer/train.py  
原69行添加
```
from msprobe.pytorch import seed_all
```
train_fn函数  
添加参数precision_mode: bool = False  
在train_fn函数
将`random.seed(random_seed)`  
替换成
```
if precision_mode:
    shuffle = False
    seed_all(mode=True, rm_dropout=True, seed=random_seed)
else:
    shuffle = True
    random.seed(random_seed)
```
将
```
    train_data_sampler, train_data_loader = create_data_loader(
        train_dataset,
        batch_size=local_batch_size,
        world_size=world_size,
        rank=rank,
        shuffle=True,
        drop_last=world_size > 1,
    )
    eval_data_sampler, eval_data_loader = create_data_loader(
        dataset.eval_dataset,
        batch_size=eval_batch_size,
        world_size=world_size,
        rank=rank,
        shuffle=True,  # needed for partial eval
        drop_last=world_size > 1,
    )
```
改成
```
    train_data_sampler, train_data_loader = create_data_loader(
        train_dataset,
        batch_size=local_batch_size,
        world_size=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=world_size > 1,
    )
    eval_data_sampler, eval_data_loader = create_data_loader(
        dataset.eval_dataset,
        batch_size=eval_batch_size,
        world_size=world_size,
        rank=rank,
        shuffle=shuffle,  
        drop_last=world_size > 1,
    )
```
将
```
    if embedding_module_type == "local":
        embedding_module: EmbeddingModule = LocalEmbeddingModule(
            num_items=dataset.max_item_id,
            item_embedding_dim=item_embedding_dim,
        )
```
替换成
```
    if embedding_module_type == "local":
        embedding_module: EmbeddingModule = LocalEmbeddingModule(
            num_items=dataset.max_item_id,
            item_embedding_dim=item_embedding_dim,
            precision_mode=precision_mode,
        )
```
model = get_sequential_encoder下添加参数precision_mode=precision_mode  
input_preproc_module = LearnablePositionalEmbeddingInputFeaturesPreprocessor下添加参数precision_mode=precision_mode

4. generative_recommenders/modeling/sequential/encoder_utils.py  
hstu_encoder函数添加参数precision_mode: bool = False  
return HSTU下添加precision_mode=precision_mode参数  
get_sequential_encoder函数添加参数precision_mode: bool = False  
model = hstu_encoder下添加参数precision_mode=precision_mode

5. generative_recommenders/modeling/sequential/input_features_preprocessors.py  
LearnablePositionalEmbeddingInputFeaturesPreprocessor类添加参数precision_mode: bool = False  
并将`self.reset_state()`替换成
 ```
        if precision_mode:
            self._pos_emb.weight.data.fill_(1)
        else:
            self.reset_state()
```
通过开启precision_mode，开启精度模式，可以消除随机性对模型精度进行对比定位 

### 精度测试示例
本次测试基于ml-1m数据集，使用GPU的triton(去rab, 下三角mask)， 基于以下配置config文件进行测试：

config文件：
创建一个hstu-mt-3400.gin文件，文件内容如下。将该gin文件放置在 generative-recommenders/configs/ml-1m/ 目录下，并开启精度模式

```
train_fn.dataset_name = "ml-1m"
train_fn.max_sequence_length = 3389
train_fn.local_batch_size = 32

train_fn.main_module = "HSTU"
train_fn.dropout_rate = 0.2
train_fn.user_embedding_norm = "l2_norm"
train_fn.num_epochs = 1
train_fn.item_embedding_dim = 512

hstu_encoder.num_blocks = 3
hstu_encoder.num_heads = 2
hstu_encoder.dqk = 256
hstu_encoder.dv = 256
hstu_encoder.linear_dropout_rate = 0.2

train_fn.learning_rate = 1e-3
train_fn.weight_decay = 0
train_fn.num_warmup_steps = 0

train_fn.interaction_module_type = "DotProduct"
train_fn.top_k_method = "MIPSBruteForceTopK"

train_fn.loss_module = "SampledSoftmaxLoss"
train_fn.num_negatives = 128
train_fn.eval_interval = 10
train_fn.sampling_strategy = "local"
train_fn.temperature = 0.05
train_fn.item_l2_norm = True
train_fn.l2_norm_eps = 1e-6

train_fn.enable_tf32 = True

create_data_loader.prefetch_factor = 128
create_data_loader.num_workers = 8

train_fn.precision_mode = True
```

执行命令:  
`python main.py --gin_config_file=your/path/to/config.gin --master_port=master_port`

### note
由于数据处理部分存在随机性，对比精度时需要将npu和gpu的数据进行对齐 

### GPU triton和NPU融合算子整网loss精度对比结果

| step | GPU loss | NPU loss | loss diff | 
| ----- | ----- | ----- | ----- |
|  0  | 4.859579 | 4.859569 | 1.0000000000509601e-05 |
|  10  | 4.85863 | 4.858632 | 2.000000000279556e-06 |
|  20  | 4.855053 | 4.855056 | 3.000000000419334e-06 |
|  30  | 4.850527 | 4.850548 | 2.1000000000270802e-05 |
|  40  | 4.837311 | 4.837361 | 4.999999999988347e-05 |
|  50  | 4.821918 | 4.822002 | 8.400000000019503e-05 |
|  60  | 4.802688 | 4.80268 | 8.000000000230045e-06 |
|  70  | 4.776829 | 4.777043 | 0.00021399999999971442 |
|  80  | 4.760865 | 4.760607 | 0.0002579999999996474 |
|  90  | 4.713118 | 4.712983 | 0.0001349999999993301 |
|  100  | 4.680532 | 4.68044 | 9.200000000042508e-05 |