"""Run one fixed-budget baseline, rank and training-size comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from .evaluate import evaluate, read_jsonl
from .modeling import DEFAULT_MODEL, PROMPT_VERSION

MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SPLITS = ("test", "stress")
CONDITIONS = (
    ("base", None, None),
    ("r8-n256", 8, 256),
    ("r4-n256", 4, 256),
    ("r16-n256", 16, 256),
    ("r8-n64", 8, 64),
    ("r8-n512", 8, 512),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    pending.replace(path)


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _run(module: str, args: list[str], timeout: int) -> float:
    command = [sys.executable, "-m", module, *args]
    print("\n> " + subprocess.list2cmdline(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True, timeout=timeout)
    return time.perf_counter() - started


def _adapter_provenance(
    args: argparse.Namespace, training_rows: list[dict]
) -> dict:
    if args.adapters_from is None:
        return {}
    provenance = {}
    for name, rank, size in CONDITIONS:
        if rank is None:
            continue
        folder = args.adapters_from / name
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        prefix = json.dumps(
            training_rows[:size], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        expected = {
            "model": args.model,
            "model_revision": args.revision,
            "prompt_version": PROMPT_VERSION,
            "rank": rank,
            "size": size,
            "steps": args.steps,
            "seed": args.seed,
            "batch_size": 2,
            "gradient_accumulation": 4,
            "learning_rate": 0.0002,
            "lora_alpha": 2 * rank,
            "lora_dropout": 0.05,
            "gradient_checkpointing": not args.no_gradient_checkpointing,
            "sample_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{name} 的既有适配器与实验配置或训练数据不匹配。")
        provenance[name] = {
            key: _sha256(folder / key)
            for key in ("metadata.json", "adapter_config.json", "adapter_model.safetensors")
        }
    return provenance


def run(args: argparse.Namespace) -> None:
    data = {name: args.data_dir / f"{name}.jsonl" for name in ("train", "validation", *SPLITS)}
    gold = {name: read_jsonl(data[name]) for name in SPLITS}
    all_ids = [row["id"] for name in SPLITS for row in gold[name]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("常规与压力测试样本 ID 必须分开。")
    training_rows = read_jsonl(data["train"])
    if len(training_rows) < max(size for _, _, size in CONDITIONS if size):
        raise ValueError("训练集至少需要 512 条样本。")
    provenance = _adapter_provenance(args, training_rows)
    source_files = {
        name: Path(__file__).with_name(name)
        for name in ("experiment.py", "modeling.py", "train.py", "predict.py", "evaluate.py")
    }
    plan = {
        "model": args.model,
        "revision": args.revision,
        "prompt_version": PROMPT_VERSION,
        "conditions": [dict(name=name, rank=rank, size=size) for name, rank, size in CONDITIONS],
        "steps": args.steps,
        "batch_size": 2,
        "gradient_accumulation": 4,
        "learning_rate": 0.0002,
        "training_seed": args.seed,
        "device_request": args.device,
        "prediction_batch": args.prediction_batch,
        "max_new_tokens": args.max_new_tokens,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "data_sha256": {key: _sha256(path) for key, path in data.items()},
        "code_sha256": {name: _sha256(path) for name, path in source_files.items()},
        "reused_adapter_sha256": provenance,
        "runtime": {
            "python": sys.version.split()[0],
            **{name: version(name) for name in ("torch", "transformers", "peft", "accelerate")},
        },
    }
    output = args.out
    plan_path = output / "plan.json"
    if args.resume:
        if not plan_path.is_file() or json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("继续运行时，计划、数据、代码和软件版本必须与原始运行一致。")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _json(plan_path, plan)
    combined = output / "evaluation-input.jsonl"
    _jsonl(
        combined,
        [{"id": row["id"], "text": row["text"]} for split in SPLITS for row in gold[split]],
    )
    prediction_dir = output / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    common = ["--model", args.model, "--revision", args.revision, "--device", args.device]
    if args.local_files_only:
        common.append("--local-files-only")
    summary = {"plan": plan, "conditions": {}}
    for name, rank, size in CONDITIONS:
        print(f"\n=== {name} ===", flush=True)
        adapter = (args.adapters_from or (output / "adapters")) / name
        metadata = adapter / "metadata.json"
        if rank is not None and not metadata.is_file():
            training_args = [
                *common, "--data", str(data["train"]), "--out", str(adapter),
                "--rank", str(rank), "--size", str(size), "--steps", str(args.steps),
                "--seed", str(args.seed), "--batch-size", "2", "--gradient-accumulation", "4",
            ]
            if args.no_gradient_checkpointing:
                training_args.append("--no-gradient-checkpointing")
            _run("finance_lora.train", training_args, args.timeout)
        prediction_path = prediction_dir / f"{name}.jsonl"
        if not prediction_path.is_file():
            # Publish only a complete prediction file, so a retry cannot reuse partial rows.
            pending = prediction_path.with_suffix(".pending")
            pending.unlink(missing_ok=True)
            prediction_args = [
                *common, "--data", str(combined), "--out", str(pending),
                "--batch-size", str(args.prediction_batch), "--max-new-tokens", str(args.max_new_tokens),
            ]
            if rank is not None:
                prediction_args.extend(("--adapter", str(adapter)))
            inference_seconds = _run("finance_lora.predict", prediction_args, args.timeout)
            pending.replace(prediction_path)
            _json(prediction_dir / f"{name}.time.json", {"wall_seconds": inference_seconds})
        if any(_sha256(path) != plan["code_sha256"][key] for key, path in source_files.items()):
            raise ValueError("运行期间代码发生变化。此轮不再计分，请用定稿代码重新运行。")
        if any(_sha256(path) != plan["data_sha256"][key] for key, path in data.items()):
            raise ValueError("运行期间数据发生变化。此轮不再计分。")
        predictions = read_jsonl(prediction_path)
        if len(predictions) != len(all_ids) or {row["id"] for row in predictions} != set(all_ids):
            raise ValueError(f"{name} 的预测 ID 不完整或重复。")
        training = json.loads(metadata.read_text(encoding="utf-8")) if rank is not None else None
        time_path = prediction_dir / f"{name}.time.json"
        condition = {
            "training": training,
            "prediction_sha256": _sha256(prediction_path),
            "inference": json.loads(time_path.read_text(encoding="utf-8")) if time_path.exists() else None,
            "metrics": {},
        }
        for split in SPLITS:
            ids = {row["id"] for row in gold[split]}
            metrics, errors = evaluate(gold[split], [row for row in predictions if row["id"] in ids])
            condition["metrics"][split] = metrics
            _json(output / "metrics" / f"{name}-{split}.json", metrics)
            error_path = output / "errors" / f"{name}-{split}.jsonl"
            error_path.parent.mkdir(exist_ok=True)
            _jsonl(error_path, errors)
            print(
                f"{split} 结构合法率 {metrics['valid_schema_rate']:.1%}  "
                f"整条精确匹配 {metrics['exact_match']:.1%}", flush=True,
            )
        summary["conditions"][name] = condition
        _json(output / "summary.json", summary)
    print(f"\n完成 {len(CONDITIONS)} 个条件。结果：{output / 'summary.json'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic-v1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prediction-batch", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=240)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--timeout", type=int, default=1800, help="单个子进程的秒数上限。")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--adapters-from",
        type=Path,
        help="复核已有适配器的配置和数据后重新推理，不再次训练。",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.prediction_batch, args.max_new_tokens, args.timeout) < 1:
        parser.error("步数、批量大小、生成预算和超时时间都必须为正。")
    try:
        run(args)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"\n实验未完成：{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
