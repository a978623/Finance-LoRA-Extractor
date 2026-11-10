"""Train one Qwen response-only LoRA run and save the adapter plus metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

from .modeling import (
    DEFAULT_MODEL,
    IGNORE_INDEX,
    LORA_TARGETS,
    PROMPT_VERSION,
    collate_sft,
    encode_sft_record,
    load_base,
    load_tokenizer,
    read_jsonl,
    resolve_device,
)


def _permutations(size: int, seed: int):
    """Repeat seeded shuffled epochs; every optimizer update is full-sized."""
    randomizer = random.Random(seed)
    while True:
        order = list(range(size))
        randomizer.shuffle(order)
        yield from order


def _loss_sum(model: Any, batch: dict[str, Any], device: Any) -> Any:
    """Return summed causal loss over the supervised next-token positions."""
    import torch
    from torch.nn import functional as F

    shifted = batch["labels"][:, 1:]
    positions = torch.nonzero(shifted.ne(IGNORE_INDEX).any(dim=0), as_tuple=True)[0]
    targets = shifted.index_select(1, positions).to(device)
    positions = positions.to(device)
    kwargs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "use_cache": False,
    }
    if model.config.model_type == "qwen2":
        # Qwen2 projects only these hidden states into its large vocabulary.
        logits = model(**kwargs, logits_to_keep=positions).logits
    else:
        logits = model(**kwargs).logits.index_select(1, positions)
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    import peft
    import torch
    import transformers
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import get_linear_schedule_with_warmup, set_seed

    output = args.out
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"输出路径已存在且非空：{output}。请选择新的运行目录。")
    records = read_jsonl(args.data)
    if len(records) < args.size:
        raise ValueError(f"数据仅有 {len(records)} 条，无法选择 --size {args.size}。")
    records = records[: args.size]
    prefix_digest = hashlib.sha256(
        json.dumps(
            records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    set_seed(args.seed)
    device, dtype = resolve_device(args.device, training=True)
    print(
        f"读取固定前缀样本 {len(records)} 条；rank={args.rank}；设备={device}；模型精度={dtype}",
        flush=True,
    )
    if device.type == "cpu":
        print("正在使用 CPU，完整训练可能较慢。", flush=True)
    tokenizer = load_tokenizer(args.model, args.revision, args.local_files_only)
    examples = [
        encode_sft_record(tokenizer, record, args.max_length) for record in records
    ]
    print(
        f"分词完成：最长 {max(len(example.input_ids) for example in examples)} token；"
        f"前缀数据监督 {sum(example.response_tokens for example in examples):,} token。",
        flush=True,
    )

    model = load_base(args.model, args.revision, device, dtype, args.local_files_only)
    resolved_revision = getattr(model.config, "_commit_hash", None) or args.revision
    if (
        max(len(example.input_ids) for example in examples)
        > model.config.max_position_embeddings
    ):
        raise ValueError("样本长度超过模型的最大上下文。")
    available_targets = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = set(LORA_TARGETS) - available_targets
    if missing:
        raise ValueError("模型缺少 LoRA 目标层：" + ", ".join(sorted(missing)))
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.rank,
            lora_alpha=2 * args.rank,
            lora_dropout=0.05,
            target_modules=list(LORA_TARGETS),
            bias="none",
            revision=resolved_revision,
        ),
    )
    model.config.use_cache = False
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.train()
    trainable_count, total_count = model.get_nb_trainable_parameters()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    warmup_steps = min(10, args.steps // 10)
    schedule = get_linear_schedule_with_warmup(optimizer, warmup_steps, args.steps)
    sample_order = _permutations(args.size, args.seed)
    effective_batch = args.batch_size * args.gradient_accumulation
    print(
        f"开始 {args.steps} 次参数更新；每次 {effective_batch} 条样本；"
        f"可训练参数 {trainable_count:,} / {total_count:,}。",
        flush=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    response_tokens = total_tokens = 0
    seen_indices: set[int] = set()
    losses = []
    for step in range(1, args.steps + 1):
        indices = list(islice(sample_order, effective_batch))
        seen_indices.update(indices)
        group = [examples[index] for index in indices]
        group_response_tokens = sum(example.response_tokens for example in group)
        group_loss = torch.zeros((), device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0, effective_batch, args.batch_size):
            batch = collate_sft(
                group[offset : offset + args.batch_size], tokenizer.pad_token_id
            )
            summed_loss = _loss_sum(model, batch, device)
            (summed_loss / group_response_tokens).backward()
            group_loss.add_(summed_loss.detach())
        mean_loss = group_loss.item() / group_response_tokens
        if not math.isfinite(mean_loss):
            raise RuntimeError(f"第 {step} 步损失不是有限数，停止保存未完成的适配器。")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        learning_rate = optimizer.param_groups[0]["lr"]
        optimizer.step()
        schedule.step()
        response_tokens += group_response_tokens
        total_tokens += sum(len(example.input_ids) for example in group)
        losses.append(mean_loss)
        if step == 1 or step == args.steps or step % args.log_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[{step:>3}/{args.steps}] loss={mean_loss:.4f}  lr={learning_rate:.2e}  "
                f"监督 token={response_tokens:,}  耗时={elapsed:.1f}s",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    peak_allocated = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else None
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) / 1024**2
        if device.type == "cuda"
        else None
    )
    metadata = {
        "model": args.model,
        "model_revision": resolved_revision,
        "requested_revision": args.revision,
        "prompt_version": PROMPT_VERSION,
        "data_file": str(args.data),
        "sample_prefix_sha256": prefix_digest,
        "rank": args.rank,
        "lora_alpha": 2 * args.rank,
        "lora_dropout": 0.05,
        "target_modules": list(LORA_TARGETS),
        "size": args.size,
        "seed": args.seed,
        "steps": args.steps,
        "warmup_steps": warmup_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": effective_batch,
        "training_example_draws": args.steps * effective_batch,
        "unique_training_examples_seen": len(seen_indices),
        "max_length": args.max_length,
        "max_observed_length": max(len(example.input_ids) for example in examples),
        "learning_rate": args.learning_rate,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "train_seconds": training_seconds,
        "train_tokens": total_tokens,
        "supervised_tokens": response_tokens,
        "peak_gpu_memory_mib": peak_allocated,
        "peak_gpu_reserved_mib": peak_reserved,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        "dtype": str(dtype).removeprefix("torch."),
        "loss_per_update": losses,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True, save_embedding_layers=False)
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"LoRA 适配器已保存：{output}（训练 {training_seconds:.1f}s）", flush=True)
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", required=True, help="固定的模型仓库提交或版本。")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args(argv)
    for option in (
        "rank",
        "size",
        "steps",
        "batch_size",
        "gradient_accumulation",
        "max_length",
        "log_every",
    ):
        if getattr(args, option) < 1:
            parser.error(f"--{option.replace('_', '-')} 必须大于 0。")
    if (
        args.seed < 0
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error("--seed 不得为负数，--learning-rate 必须是正的有限数。")
    train(args)


if __name__ == "__main__":
    main()
