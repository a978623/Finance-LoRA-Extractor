"""Generate raw JSONL predictions with the base model or one LoRA adapter."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .modeling import (
    DEFAULT_MODEL,
    PROMPT_VERSION,
    encode_prompt,
    load_base,
    load_tokenizer,
    read_jsonl,
    resolve_device,
)


def predict(args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch

    if args.out.exists():
        raise ValueError(f"预测输出已存在：{args.out}。请选择新的文件名。")
    records = read_jsonl(args.data)
    device, dtype = resolve_device(args.device, training=False)
    tokenizer = load_tokenizer(args.model, args.revision, args.local_files_only)
    tokenizer.padding_side = "left"
    prompts = []
    for record in records:
        ids = encode_prompt(tokenizer, record["text"])
        if len(ids) > args.max_input_length:
            raise ValueError(
                f"样本 {record['id']!r} 的提示词需要 {len(ids)} token，"
                f"超过 --max-input-length {args.max_input_length}；不会静默截断原文。"
            )
        prompts.append({"input_ids": ids})
    model = load_base(args.model, args.revision, device, dtype, args.local_files_only)
    if (
        max(len(item["input_ids"]) for item in prompts) + args.max_new_tokens
        > model.config.max_position_embeddings
    ):
        raise ValueError("提示词和生成预算超过模型最大上下文。")
    if args.adapter is not None:
        from peft import PeftModel

        if not (args.adapter / "adapter_config.json").is_file():
            raise ValueError(f"找不到适配器配置：{args.adapter}")
        metadata_path = args.adapter / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("prompt_version") != PROMPT_VERSION:
                raise ValueError("适配器使用了不同的提示词版本。")
            if metadata.get("model") != args.model:
                raise ValueError("适配器的基座模型与 --model 不同。")
            revision = getattr(model.config, "_commit_hash", None) or args.revision
            if metadata.get("model_revision") != revision:
                raise ValueError("适配器的基座模型版本与当前模型不同。")
        model = PeftModel.from_pretrained(
            model, args.adapter, is_trainable=False, local_files_only=True
        )
    model.eval()
    generation = copy.deepcopy(model.generation_config)
    generation.do_sample = False
    generation.num_beams = 1
    generation.temperature = None
    generation.top_p = None
    generation.top_k = None
    generation.max_new_tokens = args.max_new_tokens
    generation.pad_token_id = tokenizer.pad_token_id
    if generation.eos_token_id is None:
        generation.eos_token_id = tokenizer.eos_token_id
    generation.use_cache = True
    print(
        f"生成 {len(records)} 条预测；设备={device}；"
        f"模式={str(args.adapter) if args.adapter else 'base'}；贪心解码。",
        flush=True,
    )

    predictions = []
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch = tokenizer.pad(
                prompts[start : start + args.batch_size],
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            ).to(device)
            generated = model.generate(**batch, generation_config=generation)
            continuation = generated[:, batch["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(
                continuation,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for record, output in zip(
                records[start : start + len(texts)], texts, strict=True
            ):
                predictions.append({"id": record["id"], "prediction": output})
            if (
                start == 0
                or len(predictions) == len(records)
                or len(predictions) % (args.batch_size * 4) == 0
            ):
                print(f"已生成 {len(predictions)}/{len(records)} 条", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"预测已保存：{args.out}", flush=True)
    return predictions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", required=True, help="与训练匹配的模型版本。")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=240)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    for option in ("batch_size", "max_new_tokens", "max_input_length"):
        if getattr(args, option) < 1:
            parser.error(f"--{option.replace('_', '-')} 必须大于 0。")
    predict(args)


if __name__ == "__main__":
    main()
