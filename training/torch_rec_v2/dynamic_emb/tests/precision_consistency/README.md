# DynamicEmb 精度一致性测试

对比本仓 `training/torch_rec_v2/dynamic_emb`（Ascend）与上游
[recsys-examples v25.09 `corelib/dynamicemb`](https://github.com/NVIDIA/recsys-examples/tree/v25.09/corelib/dynamicemb)
在同一份**确定性训练场景**下的数值，覆盖 `dynamic_emb` 支持的全部优化器。

只做精度：接口面 / 参数校验 / score 接口 / `external_storage` 等不在本目录（见 Issue #1404–#1411）。

## 覆盖的优化器

`EmbOptimType` 有 5 个取值、4 个实现（`optimizers/base_dynamicemb_optimizer.py::convert_optimizer_type`）：

| `EmbOptimType` | 实现族 | 超参 |
| --- | --- | --- |
| `SGD` / `EXACT_SGD` | SGD | `lr=0.1` |
| `ADAM` | Adam | `lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0` |
| `EXACT_ADAGRAD` | AdaGrad | `lr=0.1, eps=1e-8, initial_accumulator_value=0` |
| `EXACT_ROWWISE_ADAGRAD` | RowWiseAdaGrad | 同上 |

`ADAMW` / `LARS_SGD` / `PARTIAL_ROW_WISE_ADAM` 不在表内：`convert_optimizer_type()` 对它们
抛 `Not supported optimizer type`。`test_unsupported_optimizer_is_rejected` **直接调用库的
`convert_optimizer_type()`**，逐个断言该后端 `EmbOptimType` 中未实现的成员都会被拒（不是断言本套件自己的白名单）。

损失用 **平方和**（`loss = Σ pooled²`）：plain sum 的梯度恒为 1，Adam/AdaGrad 在这种梯度下与 SGD **数值上不可区分**，
矩阵就失去意义；平方损失的梯度随权重变化，各优化器的更新式才会分叉。

每个优化器验证三件事：**5 轮 loss**（与参照逐轮比对）、
**初始化/训练后池化输出**（`sum/mean/abs_sum`）、**dump→load 往返**（`optim=True`，
有状态优化器因此同时验证优化器状态的落盘与回读，`loaded_matches_trained` 必须为 `True`）。

## 场景为什么确定

`DEBUG` 初始化（每个 key 独立取值 `key % 100000`，非均匀但跨端逐位相同）、`STEP` 打分、自研 LCG 生成输入（不碰 `torch` RNG）、
`init_capacity=2**16` 且 `max_load_factor=0.5`（不触发换入换出）、`output_dtype=FP32`、
SUM 池化、单卡 `world_size=1`。

## 目录

```
precision_consistency/
├── backend.py                     逻辑 API 名 → 两端导入路径；设备/分布式抽象
├── scenario.py                    确定性场景 + OPTIMIZER_SPECS
├── record_reference.py            在参照端（GPU）录 golden
├── test_precision_consistency.py  pytest 用例（参数化全部优化器）
└── golden/reference_v25.09.json   参照结果（含环境与来源）
```

## 运行

```bash
# 受测端（CI / 本地）
cd training/torch_rec_v2/dynamic_emb/tests
python3 -m pytest precision_consistency/test_precision_consistency.py -v
```

容器内需先 `source` CANN 的 `set_env.sh`；后端由 `DYNAMICEMB_BACKEND` 指定（未设置时按
`torch_npu` 是否可导入自动判定）。

参照结果随用例提交，**仅在参照版本升级或场景配置变化时重录**：

```bash
DYNAMICEMB_BACKEND=gpu python3 record_reference.py --out golden/reference_v25.09.json
```

录制环境（python / torch / device / 上游 tag / 时间）写进 JSON 头部，便于审核来源。

## 判据

| 检查 | 判据 |
| --- | --- |
| 每轮 loss | `abs(diff) <= rtol*abs(ref)`；SGD / EXACT_SGD / AdaGrad `rtol=1e-6`（实测仅差 1 个 fp32 ULP），`ADAM` `1e-3` |
| `initial_probe` / `post_training_probe` | `abs(diff) <= atol + rtol*abs(ref)`，`atol` 1e-4（`mean` 1e-6），`rtol` 默认 1e-6、`ADAM` 2e-3 |
| dump/load | 独立用例 `test_dump_load_roundtrip`；`load_changed_state` 与 `loaded_matches_trained` 均为 `True`（硬断言，Issue #1409 的修复已在源码） |
| 矩阵完整性 | 缺任一已支持优化器 → FAIL；仅"已列入矩阵但参照未录"才是 skip |

容差是**测试侧策略**（`TOLERANCE` 表），golden 只存数字。实测（2026-10-26，A100 vs Ascend950PR，DEBUG 初值）：
SGD / EXACT_SGD / AdaGrad 与参照仅差 **1 个 fp32 ULP**（相对 8.5e-8 / 1.1e-7，归约顺序差异），`ADAM` 相对漂移 4e-5。

## 边界

单卡；无性能指标；池化固定 SUM、dim=8、2 张表、`num_embeddings=1024`（能暴露优化器算术差异，不代表大表换入换出）；
不覆盖 `caching=True`。扩维度：改 `scenario.build_case()` 配置并重录 golden。
