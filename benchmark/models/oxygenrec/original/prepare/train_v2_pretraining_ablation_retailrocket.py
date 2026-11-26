#!/usr/bin/env python3
"""在同一RetailRocket daily cohort上配对复现v2预训练三组消融。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from oxygenrec.data import (
    Split,
    TemporalBoundaries,
    build_daily_listwise_samples,
    build_listwise_sid_model_batch,
    load_retailrocket_events,
)
from oxygenrec.evaluation import evaluate_sid_token_match
from oxygenrec.device import resolve_device, seed_torch
from oxygenrec.model import OxygenRECConfig, OxygenRECModel
from oxygenrec.sid import PrefixTrie, SIDRegistry
from oxygenrec.migration_alignment import BEHAVIOR_WEIGHTS


VARIANTS = ("base", "ib", "full")


def chunks(items, size):
    """把列表切成连续且不超过size的mini-batch。"""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def sid_chunks(flat_path: list[int], levels: int) -> list[list[int]]:
    """把展平的3N token恢复为N个SID，便于评测和审阅。"""
    return [
        flat_path[start : start + levels]
        for start in range(0, len(flat_path), levels)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--sid-registry", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/retailrocket_v2_pretraining_ablation_smoke"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--list-size", type=int, default=2)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--max-train-samples", type=int, default=5_000)
    parser.add_argument("--max-validation-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--geometric-decay", type=float, default=0.9)
    return parser.parse_args()


def model_config(registry, args, *, use_behavior_instruction: bool):
    return OxygenRECConfig(
        sid_width=registry.width,
        sid_levels=registry.levels,
        hidden_size=args.hidden_size,
        attention_heads=args.attention_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        feedforward_size=args.hidden_size * 4,
        max_history_items=args.max_history,
        max_target_items=args.list_size,
        behavior_vocab_size=3,
        behavior_instruction_vocab_size=3 if use_behavior_instruction else 0,
    )


def copy_common_initialization(base, ib_model, sid_width: int) -> None:
    """把可对应的主干初值从+Ib模型逐值复制到w/o-Ib模型。"""
    base_state = base.state_dict()
    ib_state = ib_model.state_dict()
    for name, target in base_state.items():
        source = ib_state.get(name)
        if source is None:
            continue
        if source.shape == target.shape:
            target.copy_(source)
        elif name == "sid_embeddings.0.weight":
            target.copy_(source[:sid_width])
        elif name == "decoder_positions.weight":
            # w/o-Ib前三行仍是BOS/scene/reasoning；后续target prefix在+Ib中右移一位。
            target[:3].copy_(source[:3])
            target[3:].copy_(source[4:])
        else:
            raise RuntimeError(f"unexpected common parameter shape mismatch: {name}")
    base.load_state_dict(base_state, strict=True)


def common_initialization_matches(base, ib_model, sid_width: int) -> bool:
    base_state = base.state_dict()
    ib_state = ib_model.state_dict()
    for name, target in base_state.items():
        source = ib_state.get(name)
        if source is None:
            return False
        if source.shape == target.shape:
            expected = source
        elif name == "sid_embeddings.0.weight":
            expected = source[:sid_width]
        elif name == "decoder_positions.weight":
            expected = torch.cat((source[:3], source[4:]), dim=0)
        else:
            return False
        if not torch.equal(target, expected):
            return False
    return True


def make_batch(samples, registry, args, device, variant: str) -> dict:
    raw = build_listwise_sid_model_batch(
        samples, registry, max_history_items=args.max_history
    )
    batch = {
        "history_sids": torch.tensor(
            raw.history_sids, dtype=torch.long, device=device
        ),
        "history_padding_mask": torch.tensor(
            raw.history_padding_mask, dtype=torch.bool, device=device
        ),
        "history_behavior_ids": torch.tensor(
            raw.history_behavior_ids, dtype=torch.long, device=device
        ),
        "target_sids": torch.tensor(
            raw.target_sids, dtype=torch.long, device=device
        ),
    }
    if variant != "base":
        behavior_ids = torch.tensor(
            raw.target_behavior_ids, dtype=torch.long, device=device
        )
        batch["behavior_instruction_ids"] = behavior_ids
        if variant == "full":
            weights = torch.tensor(
                BEHAVIOR_WEIGHTS, dtype=torch.float32, device=device
            )
            sid_tokens = args.list_size * registry.levels
            batch["token_weights"] = weights[behavior_ids].unsqueeze(1).expand(
                -1, sid_tokens
            )
    return batch


def model_inputs(batch: dict) -> dict:
    return {
        key: value
        for key, value in batch.items()
        if not key.startswith("labs_")
    }


def generation_inputs(batch: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
    optional = {}
    for name in ("history_behavior_ids", "behavior_instruction_ids"):
        if name in batch:
            optional[name] = batch[name]
    return batch["history_sids"], batch["history_padding_mask"], optional


def evaluate(model, samples, registry, trie, args, device, variant: str):
    model.eval()
    totals = Counter()
    behavior_totals: dict[str, Counter] = defaultdict(Counter)
    rows = []
    with torch.no_grad():
        for sample_batch in chunks(samples, args.batch_size):
            batch = make_batch(sample_batch, registry, args, device, variant)
            targets = batch["target_sids"].reshape(len(sample_batch), -1)
            history, padding, optional = generation_inputs(batch)
            generated = model.generate(
                history,
                padding,
                trie,
                output_items=args.list_size,
                **optional,
            )
            for sample, target, prediction in zip(
                sample_batch,
                targets.cpu().tolist(),
                generated.cpu().tolist(),
                strict=True,
            ):
                target_items = sid_chunks(target, registry.levels)
                predicted_items = sid_chunks(prediction, registry.levels)
                target_set = {tuple(item) for item in target_items}
                predicted_set = {tuple(item) for item in predicted_items}
                token_match = evaluate_sid_token_match(
                    prediction,
                    target,
                    geometric_decay=args.geometric_decay,
                )
                overlap = len(target_set & predicted_set)
                legal = sum(trie.contains(item) for item in predicted_items)
                unique = len(predicted_set)
                behavior = sample.target_behavior.value
                totals.update({
                    "lists": 1,
                    "targets": args.list_size,
                    "target_unique": len(target_set),
                    "generated_unique": unique,
                    "all_unique_lists": int(unique == args.list_size),
                    "sid_overlap": overlap,
                    "token_hits": sum(token_match.hits),
                    "tokens": len(token_match.hits),
                    "geometric_reward": token_match.geometric_reward,
                    "legal_items": legal,
                    "exact_lists": int(prediction == target),
                })
                behavior_totals[behavior].update({
                    "lists": 1,
                    "targets": args.list_size,
                    "sid_overlap": overlap,
                    "token_hits": sum(token_match.hits),
                    "tokens": len(token_match.hits),
                })
                rows.append({
                    "behavior": behavior,
                    "utc_day": sample.utc_day,
                    "history_length": len(sample.history),
                    "history_behavior_counts": dict(sorted(Counter(
                        event.behavior.value for event in sample.history
                    ).items())),
                    "target_sids": target_items,
                    "generated_sids": predicted_items,
                    "sid_token_hits": list(token_match.hits),
                    "sid_token_accuracy": token_match.accuracy,
                    "geometric_token_reward": token_match.geometric_reward,
                    "sid_recall": overlap / args.list_size,
                    "all_generated_items_legal": legal == args.list_size,
                    "generated_items_unique": unique == args.list_size,
                })
    if not totals["lists"]:
        raise RuntimeError("validation produced no listwise rows")
    metrics = {
        "lists": totals["lists"],
        "list_size": args.list_size,
        "target_sid_unique_rate": totals["target_unique"] / totals["targets"],
        "generated_sid_unique_rate": (
            totals["generated_unique"] / totals["targets"]
        ),
        "all_unique_list_rate": totals["all_unique_lists"] / totals["lists"],
        "sid_recall": totals["sid_overlap"] / totals["targets"],
        "sid_token_accuracy": totals["token_hits"] / totals["tokens"],
        "geometric_token_reward": totals["geometric_reward"] / totals["lists"],
        "legal_item_rate": totals["legal_items"] / totals["targets"],
        "exact_list_rate": totals["exact_lists"] / totals["lists"],
        "behavior": {
            behavior: {
                "lists": counts["lists"],
                "sid_recall": counts["sid_overlap"] / counts["targets"],
                "sid_token_accuracy": counts["token_hits"] / counts["tokens"],
            }
            for behavior, counts in sorted(behavior_totals.items())
        },
    }
    return metrics, rows


def compare_rows(rows_by_variant: dict[str, list[dict]]) -> list[dict]:
    counts = {name: len(rows) for name, rows in rows_by_variant.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"variant validation row counts differ: {counts}")
    compared = []
    for base, ib, full in zip(
        rows_by_variant["base"],
        rows_by_variant["ib"],
        rows_by_variant["full"],
        strict=True,
    ):
        identity = (base["behavior"], base["utc_day"], base["target_sids"])
        if identity != (ib["behavior"], ib["utc_day"], ib["target_sids"]):
            raise RuntimeError("base and +Ib validation rows are not paired")
        if identity != (full["behavior"], full["utc_day"], full["target_sids"]):
            raise RuntimeError("base and full validation rows are not paired")
        base_generated = [
            token for sid in base["generated_sids"] for token in sid
        ]
        ib_generated = [
            token for sid in ib["generated_sids"] for token in sid
        ]
        full_generated = [
            token for sid in full["generated_sids"] for token in sid
        ]
        target_sids_unique = (
            len({tuple(sid) for sid in base["target_sids"]})
            == len(base["target_sids"])
        )
        compared.append({
            "behavior": base["behavior"],
            "utc_day": base["utc_day"],
            "history_length": base["history_length"],
            "history_behavior_counts": base["history_behavior_counts"],
            "target_sids": base["target_sids"],
            "target_sids_unique": target_sids_unique,
            "base": {
                key: base[key]
                for key in (
                    "generated_sids", "sid_token_hits", "sid_token_accuracy",
                    "geometric_token_reward", "sid_recall",
                    "all_generated_items_legal", "generated_items_unique",
                )
            },
            "ib": {
                key: ib[key]
                for key in (
                    "generated_sids", "sid_token_hits", "sid_token_accuracy",
                    "geometric_token_reward", "sid_recall",
                    "all_generated_items_legal", "generated_items_unique",
                )
            },
            "full": {
                key: full[key]
                for key in (
                    "generated_sids", "sid_token_hits", "sid_token_accuracy",
                    "geometric_token_reward", "sid_recall",
                    "all_generated_items_legal", "generated_items_unique",
                )
            },
            "ib_minus_base_token_accuracy": (
                ib["sid_token_accuracy"] - base["sid_token_accuracy"]
            ),
            "full_minus_ib_token_accuracy": (
                full["sid_token_accuracy"] - ib["sid_token_accuracy"]
            ),
            "full_minus_base_token_accuracy": (
                full["sid_token_accuracy"] - base["sid_token_accuracy"]
            ),
            "ib_vs_base_generated_token_changes": sum(
                left != right
                for left, right in zip(
                    base_generated, ib_generated, strict=True
                )
            ),
            "full_vs_ib_generated_token_changes": sum(
                left != right
                for left, right in zip(
                    ib_generated, full_generated, strict=True
                )
            ),
        })
    return compared


def select_review_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    specs = (
        (
            "largest_ib_gain",
            lambda row: row["ib_minus_base_token_accuracy"] > 0.0,
            "ib_minus_base_token_accuracy",
            True,
        ),
        (
            "largest_full_gain",
            lambda row: row["full_minus_base_token_accuracy"] > 0.0,
            "full_minus_base_token_accuracy",
            True,
        ),
        (
            "largest_full_drop",
            lambda row: row["full_minus_base_token_accuracy"] < 0.0,
            "full_minus_base_token_accuracy",
            False,
        ),
        (
            "weighting_gain",
            lambda row: row["full_minus_ib_token_accuracy"] > 0.0,
            "full_minus_ib_token_accuracy",
            True,
        ),
        (
            "ib_output_changed",
            lambda row: row["ib_vs_base_generated_token_changes"] > 0,
            "ib_vs_base_generated_token_changes",
            True,
        ),
        (
            "weighting_output_changed",
            lambda row: row["full_vs_ib_generated_token_changes"] > 0,
            "full_vs_ib_generated_token_changes",
            True,
        ),
        (
            "ib_duplicate_introduced",
            lambda row: (
                row["target_sids_unique"]
                and row["base"]["generated_items_unique"]
                and not row["ib"]["generated_items_unique"]
            ),
            "history_length",
            True,
        ),
        (
            "weighting_duplicate_introduced",
            lambda row: (
                row["target_sids_unique"]
                and row["ib"]["generated_items_unique"]
                and not row["full"]["generated_items_unique"]
            ),
            "history_length",
            True,
        ),
        (
            "target_sid_collision",
            lambda row: not row["target_sids_unique"],
            "history_length",
            True,
        ),
        (
            "transaction",
            lambda row: row["behavior"] == "transaction",
            "history_length",
            True,
        ),
        (
            "generated_duplicate",
            lambda row: any(
                not row[name]["generated_items_unique"] for name in VARIANTS
            ),
            "history_length",
            True,
        ),
    )
    chosen = {}
    coverage = {}
    for role, predicate, key, maximize in specs:
        candidates = [
            (index, row) for index, row in enumerate(rows) if predicate(row)
        ]
        if not candidates:
            coverage[role] = None
            continue
        chooser = max if maximize else min
        index, row = chooser(candidates, key=lambda item: (item[1][key], -item[0]))
        chosen.setdefault(index, {**row, "roles": []})["roles"].append(role)
        coverage[role] = index
    selected = []
    case_ids = {}
    for number, (index, row) in enumerate(sorted(chosen.items()), start=1):
        case_id = f"v2-pretrain-ablation-review-{number:03d}"
        selected.append({"case_id": case_id, **row})
        case_ids[index] = case_id
    return selected, {
        role: case_ids[index] if index is not None else None
        for role, index in coverage.items()
    }


def metric_delta(left: dict, right: dict) -> dict:
    names = (
        "sid_recall",
        "sid_token_accuracy",
        "geometric_token_reward",
        "generated_sid_unique_rate",
        "all_unique_list_rate",
    )
    return {name: right[name] - left[name] for name in names}


def write_artifacts(
    output_dir: Path, summary: dict, compared_rows: list[dict]
) -> dict:
    selected, coverage = select_review_rows(compared_rows)
    output_changes = {
        "ib_vs_base_changed_lists": sum(
            row["ib_vs_base_generated_token_changes"] > 0
            for row in compared_rows
        ),
        "full_vs_ib_changed_lists": sum(
            row["full_vs_ib_generated_token_changes"] > 0
            for row in compared_rows
        ),
        "target_sid_collision_lists": sum(
            not row["target_sids_unique"] for row in compared_rows
        ),
    }
    summary = {
        **summary,
        "coverage": coverage,
        "representative_cases": len(selected),
        "output_changes": output_changes,
    }
    (output_dir / "v2_pretraining_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "v2_pretraining_ablation_cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    (output_dir / "v2_pretraining_ablation_all_rows.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in compared_rows
        ),
        encoding="utf-8",
    )
    lines = [
        "# OxygenREC-v2预训练三组配对消融",
        "",
        "> Base不使用目标行为指令且使用统一NTP；+Ib加入目标行为指令但仍使用统一NTP；",
        "> Full进一步加入view/cart/order=1.2/1.5/2.0的行为加权损失。",
        "> 三组共用daily cohort和训练预算；这是公开代理smoke，不是论文私有指标复现。",
        "",
        "## 聚合指标",
        "",
        "| 阶段/变体 | SID recall | token accuracy | geometric reward | target SID unique | generated SID unique | all-unique lists | legal items |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in ("before", "after"):
        for variant in VARIANTS:
            values = summary[f"{stage}_metrics"][variant]
            lines.append(
                f"| {stage}/{variant} | {values['sid_recall']:.6f} | "
                f"{values['sid_token_accuracy']:.6f} | "
                f"{values['geometric_token_reward']:.6f} | "
                f"{values['target_sid_unique_rate']:.6f} | "
                f"{values['generated_sid_unique_rate']:.6f} | "
                f"{values['all_unique_list_rate']:.6f} | "
                f"{values['legal_item_rate']:.6f} |"
            )
    lines.extend([
        "",
        "## 训练后差值",
        "",
        f"- +Ib - Base：`{summary['after_delta']['ib_minus_base']}`",
        f"- Full - +Ib：`{summary['after_delta']['full_minus_ib']}`",
        f"- Full - Base：`{summary['after_delta']['full_minus_base']}`",
        f"- +Ib相对Base生成变化列表数：{output_changes['ib_vs_base_changed_lists']}",
        f"- Full相对+Ib生成变化列表数：{output_changes['full_vs_ib_changed_lists']}",
        f"- 目标SID发生碰撞的列表数：{output_changes['target_sid_collision_lists']}",
        f"- 固定案例覆盖：`{json.dumps(coverage, ensure_ascii=False, sort_keys=True)}`",
        "",
    ])
    for row in selected:
        lines.extend([
            f"## {row['case_id']}",
            "",
            f"- 代表角色：`{row['roles']}`",
            f"- 目标行为：`{row['behavior']}`",
            f"- UTC日编号：`{row['utc_day']}`",
            f"- 历史长度/行为：{row['history_length']} / `{row['history_behavior_counts']}`",
            f"- 目标SID列表：`{row['target_sids']}`",
            f"- 目标SID互异：`{row['target_sids_unique']}`",
            f"- Base：`{row['base']}`",
            f"- +Ib：`{row['ib']}`",
            f"- Full：`{row['full']}`",
            f"- +Ib-Base token accuracy：{row['ib_minus_base_token_accuracy']:+.6f}",
            f"- Full-+Ib token accuracy：{row['full_minus_ib_token_accuracy']:+.6f}",
            f"- Full-Base token accuracy：{row['full_minus_base_token_accuracy']:+.6f}",
            f"- +Ib/Base生成token变化数：{row['ib_vs_base_generated_token_changes']}",
            f"- Full/+Ib生成token变化数：{row['full_vs_ib_generated_token_changes']}",
            "",
            "### 人工Review",
            "",
            "- [ ] 三组目标、历史与行为完全相同",
            "- [ ] 行为指令变化与生成列表变化能明确区分",
            "- [ ] 重复SID未被误写为PrefixTrie非法SID",
            "- [ ] 小样本差值未被写成稳定质量收益",
            "",
        ])
    (output_dir / "v2_pretraining_ablation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return {"coverage": coverage, **output_changes}


def main() -> None:
    args = parse_args()
    positive = (
        args.list_size,
        args.max_history,
        args.max_train_samples,
        args.max_validation_samples,
        args.batch_size,
        args.epochs,
    )
    if min(positive) < 1:
        raise ValueError("sizes, counts and epochs must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if not 0.0 <= args.geometric_decay <= 1.0:
        raise ValueError("geometric-decay must be in [0, 1]")
    device = resolve_device(args.device)
    random.seed(args.seed)
    seed_torch(args.seed, device)

    print("stage=load_events_and_build_paired_daily_cohort")
    events = list(load_retailrocket_events(args.events))
    minimum = min(event.timestamp_ms for event in events)
    maximum = max(event.timestamp_ms for event in events)
    duration = maximum - minimum + 1
    boundaries = TemporalBoundaries(
        train_end_ms=minimum + duration * 8 // 10,
        validation_end_ms=minimum + duration * 9 // 10,
    )
    registry = SIDRegistry.from_json(args.sid_registry)
    events = [event for event in events if event.item_id in registry.item_to_sid]
    samples = build_daily_listwise_samples(
        events,
        boundaries,
        list_size=args.list_size,
        max_history=args.max_history,
        max_samples_per_split={
            Split.TRAIN: args.max_train_samples,
            Split.VALIDATION: args.max_validation_samples,
            Split.TEST: 1,
        },
        sample_seed=args.seed,
    )
    train_samples = [sample for sample in samples if sample.split is Split.TRAIN]
    validation_samples = [
        sample for sample in samples if sample.split is Split.VALIDATION
    ]
    if not train_samples or not validation_samples:
        raise RuntimeError("paired daily cohort has an empty split")
    train_behaviors = Counter(sample.target_behavior.value for sample in train_samples)
    validation_behaviors = Counter(
        sample.target_behavior.value for sample in validation_samples
    )

    print("stage=initialize_three_paired_variants")
    ib_config = model_config(registry, args, use_behavior_instruction=True)
    base_config = model_config(registry, args, use_behavior_instruction=False)
    ib_model = OxygenRECModel(ib_config).to(device)
    full_model = copy.deepcopy(ib_model)
    base_model = OxygenRECModel(base_config).to(device)
    copy_common_initialization(base_model, ib_model, registry.width)
    common_match = common_initialization_matches(base_model, ib_model, registry.width)
    ib_full_match = all(
        torch.equal(left, right)
        for left, right in zip(
            ib_model.state_dict().values(),
            full_model.state_dict().values(),
            strict=True,
        )
    )
    if not common_match or not ib_full_match:
        raise RuntimeError("paired model initialization check failed")
    models = {"base": base_model, "ib": ib_model, "full": full_model}
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        for name, model in models.items()
    }
    trie = PrefixTrie.from_registry(registry)
    before = {
        name: evaluate(
            model, validation_samples, registry, trie, args, device, name
        )[0]
        for name, model in models.items()
    }
    if before["ib"] != before["full"]:
        raise RuntimeError("+Ib and Full must have identical pre-training metrics")

    print("stage=train_three_paired_variants")
    losses = {name: [] for name in VARIANTS}
    for epoch in range(args.epochs):
        order = list(range(len(train_samples)))
        random.Random(args.seed + epoch).shuffle(order)
        ordered_samples = [train_samples[index] for index in order]
        for variant in VARIANTS:
            model = models[variant]
            optimizer = optimizers[variant]
            model.train()
            seed_torch(args.seed + 10_000 + epoch, device)
            total_loss = 0.0
            batches = 0
            for sample_batch in chunks(ordered_samples, args.batch_size):
                batch = make_batch(
                    sample_batch, registry, args, device, variant
                )
                optimizer.zero_grad(set_to_none=True)
                output = model(**model_inputs(batch))
                output.loss.backward()
                optimizer.step()
                total_loss += float(output.loss.detach())
                batches += 1
            losses[variant].append(total_loss / batches)

    after_metrics = {}
    after_rows = {}
    for name, model in models.items():
        after_metrics[name], after_rows[name] = evaluate(
            model, validation_samples, registry, trie, args, device, name
        )
    compared_rows = compare_rows(after_rows)
    summary = {
        "protocol": "base_vs_ib_vs_behavior_weighted_full",
        "public_proxy": "RetailRocket view=click proxy",
        "external_reward_model": False,
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "train_behaviors": dict(sorted(train_behaviors.items())),
        "validation_behaviors": dict(sorted(validation_behaviors.items())),
        "epochs": args.epochs,
        "optimizer_steps_per_variant": math.ceil(
            len(train_samples) / args.batch_size
        ) * args.epochs,
        "same_cohort": True,
        "same_batch_order": True,
        "common_initialization_match": common_match,
        "ib_full_initialization_match": ib_full_match,
        "before_metrics": before,
        "after_metrics": after_metrics,
        "train_loss": losses,
        "after_delta": {
            "ib_minus_base": metric_delta(after_metrics["base"], after_metrics["ib"]),
            "full_minus_ib": metric_delta(after_metrics["ib"], after_metrics["full"]),
            "full_minus_base": metric_delta(after_metrics["base"], after_metrics["full"]),
        },
        "interpretation_boundary": (
            "single-seed public-proxy smoke; no stable quality-gain claim"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        torch.save({
            "variant": name,
            "epoch": args.epochs,
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizers[name].state_dict(),
            "sid_registry_version": registry.version,
            "boundaries": asdict(boundaries),
            "args": vars(args),
        }, args.output_dir / f"{name}-epoch-{args.epochs}.pt")
    report_diagnostics = write_artifacts(
        args.output_dir, summary, compared_rows
    )
    print(
        "OK "
        f"device={device.type} variant=v2_pretraining_ablation "
        f"train={len(train_samples)} validation={len(validation_samples)} "
        f"steps={summary['optimizer_steps_per_variant']} "
        f"common_initialization_match={common_match} "
        f"ib_full_initialization_match={ib_full_match} "
        f"base_token_accuracy={after_metrics['base']['sid_token_accuracy']:.6f} "
        f"ib_token_accuracy={after_metrics['ib']['sid_token_accuracy']:.6f} "
        f"full_token_accuracy={after_metrics['full']['sid_token_accuracy']:.6f} "
        f"base_sid_recall={after_metrics['base']['sid_recall']:.6f} "
        f"ib_sid_recall={after_metrics['ib']['sid_recall']:.6f} "
        f"full_sid_recall={after_metrics['full']['sid_recall']:.6f} "
        f"base_unique={after_metrics['base']['generated_sid_unique_rate']:.6f} "
        f"ib_unique={after_metrics['ib']['generated_sid_unique_rate']:.6f} "
        f"full_unique={after_metrics['full']['generated_sid_unique_rate']:.6f} "
        f"base_all_unique={after_metrics['base']['all_unique_list_rate']:.6f} "
        f"ib_all_unique={after_metrics['ib']['all_unique_list_rate']:.6f} "
        f"full_all_unique={after_metrics['full']['all_unique_list_rate']:.6f} "
        f"base_legal={after_metrics['base']['legal_item_rate']:.6f} "
        f"ib_legal={after_metrics['ib']['legal_item_rate']:.6f} "
        f"full_legal={after_metrics['full']['legal_item_rate']:.6f} "
        f"ib_vs_base_changed_lists="
        f"{report_diagnostics['ib_vs_base_changed_lists']} "
        f"full_vs_ib_changed_lists="
        f"{report_diagnostics['full_vs_ib_changed_lists']} "
        f"review_coverage="
        f"{json.dumps(report_diagnostics['coverage'], sort_keys=True)} "
        f"report={args.output_dir / 'v2_pretraining_ablation.md'} "
        "external_reward_model=False"
    )


if __name__ == "__main__":
    main()