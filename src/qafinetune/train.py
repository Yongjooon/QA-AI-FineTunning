from __future__ import annotations

import argparse
import json
import math
import os
import random
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from qafinetune.io_utils import load_training_records_from_dir, load_training_records_from_zip
from qafinetune.runtime import detect_runtime, ensure_dir, load_json, save_json, setup_logging, suggest_training_preset, utc_timestamp


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class RunPaths:
    run_dir: Path
    extract_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    final_model_dir: Path
    runtime_profile_path: Path
    dataset_profile_path: Path
    run_state_path: Path
    trainer_metrics_path: Path
    latest_run_pointer_path: Path
    chunk_history_path: Path


class RunStateCallback(TrainerCallback):
    def __init__(self, run_state_path: Path, metrics_path: Path, checkpoint_dir: Path) -> None:
        self.run_state_path = run_state_path
        self.metrics_path = metrics_path
        self.checkpoint_dir = checkpoint_dir

    def _write_state(self, payload: dict[str, Any]) -> None:
        existing = load_json(self.run_state_path) or {}
        existing.update(payload)
        save_json(self.run_state_path, existing)

    def on_train_begin(self, args, state, control, **kwargs):
        self._write_state({"status": "running", "global_step": state.global_step, "epoch": state.epoch})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": state.global_step, "epoch": state.epoch, **logs}, ensure_ascii=False) + "\n")

    def on_save(self, args, state, control, **kwargs):
        self._write_state(
            {
                "status": "running",
                "global_step": state.global_step,
                "epoch": state.epoch,
                "latest_checkpoint": str(self.checkpoint_dir),
            }
        )

    def on_train_end(self, args, state, control, **kwargs):
        self._write_state(
            {
                "status": "running",
                "global_step": state.global_step,
                "epoch": state.epoch,
                "latest_checkpoint": str(self.checkpoint_dir),
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Gemma 4 model on QA scenario data.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_source", "--train_zip", dest="train_source", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--resume_mode", choices=["auto", "never", "path"], default="auto")
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=0)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--replay_ratio", type=float, default=0.5)
    parser.add_argument("--gpu_max_memory_gb", type=int, default=5)
    parser.add_argument("--cpu_max_memory_gb", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_run_paths(output_root: str | Path, run_name: str) -> RunPaths:
    base_root = ensure_dir(output_root)
    resolved_run_name = run_name or f"train_{utc_timestamp()}"
    run_dir = ensure_dir(base_root / "train_runs" / resolved_run_name)
    logs_dir = ensure_dir(run_dir / "logs")
    extract_dir = ensure_dir(run_dir / "extracted_data")
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    final_model_dir = ensure_dir(run_dir / "final_model")
    latest_pointer_path = base_root / "train_runs" / "latest_run.json"
    return RunPaths(
        run_dir=run_dir,
        extract_dir=extract_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        final_model_dir=final_model_dir,
        runtime_profile_path=run_dir / "runtime_profile.json",
        dataset_profile_path=run_dir / "dataset_profile.json",
        run_state_path=run_dir / "run_state.json",
        trainer_metrics_path=logs_dir / "trainer_metrics.jsonl",
        latest_run_pointer_path=latest_pointer_path,
        chunk_history_path=logs_dir / "chunk_history.jsonl",
    )


def _typed_text_message(role: str, text: str) -> dict[str, Any]:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def format_messages_for_training(processor, example: dict[str, Any]) -> str:
    messages = example["messages"]
    if not messages:
        messages = [
            _typed_text_message("user", example["prompt"]),
            _typed_text_message("assistant", example["response"]),
        ]
    else:
        messages = [_typed_text_message(item["role"], item["content"]) for item in messages]

    try:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        prompt = example["prompt"].strip()
        response = example["response"].strip()
        return f"User:\n{prompt}\n\nAssistant:\n{response}"


def resolve_text_backend(processor):
    text_backend = getattr(processor, "tokenizer", None) or processor
    if hasattr(text_backend, "padding_side"):
        text_backend.padding_side = "right"
    pad_token = getattr(text_backend, "pad_token", None)
    eos_token = getattr(text_backend, "eos_token", None)
    if pad_token is None and eos_token is not None:
        text_backend.pad_token = eos_token
    return text_backend


def tokenize_records(dataset: Dataset, processor, text_backend, max_seq_length: int) -> Dataset:
    def _tokenize(example: dict[str, Any]) -> dict[str, Any]:
        text = format_messages_for_training(processor, example)
        tokens = text_backend(text, truncation=True, max_length=max_seq_length, padding=False)
        tokens["labels"] = list(tokens["input_ids"])
        return tokens

    return dataset.map(_tokenize, remove_columns=dataset.column_names)


def prepare_model_for_low_vram_lora(model):
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        input_embeddings = model.get_input_embeddings()

        def _make_inputs_require_grad(module, module_input, module_output):
            module_output.requires_grad_(True)

        input_embeddings.register_forward_hook(_make_inputs_require_grad)

    for module_name, module in model.named_modules():
        if "norm" in module_name.lower():
            try:
                module.to(torch.float32)
            except Exception:
                pass

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return model


def load_model_and_processor(
    model_name: str,
    bf16: bool,
    runtime_profile: dict[str, Any],
    offload_dir: Path,
    gpu_max_memory_gb: int,
    cpu_max_memory_gb: int,
):
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    text_backend = resolve_text_backend(processor)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "dtype": compute_dtype,
        "quantization_config": quantization_config,
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }
    if runtime_profile.get("gpu_memory_gb", 0) <= 16:
        ensure_dir(offload_dir)
        load_kwargs["max_memory"] = {0: f"{gpu_max_memory_gb}GiB", "cpu": f"{cpu_max_memory_gb}GiB"}
        load_kwargs["low_cpu_mem_usage"] = True
        load_kwargs["offload_folder"] = str(offload_dir)
        load_kwargs["offload_state_dict"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    model = prepare_model_for_low_vram_lora(model)
    return model, processor, text_backend


def build_chunk_passes(record_count: int, chunk_size: int, num_train_epochs: float) -> list[dict[str, Any]]:
    if chunk_size <= 0 or chunk_size >= record_count:
        return [{"cycle_index": 0, "chunk_index": 0, "start": 0, "end": record_count, "epoch_fraction": num_train_epochs}]

    chunk_bounds = [(start, min(record_count, start + chunk_size)) for start in range(0, record_count, chunk_size)]
    passes: list[dict[str, Any]] = []
    full_cycles = int(math.floor(num_train_epochs))
    fractional_cycle = num_train_epochs - full_cycles

    for cycle_index in range(full_cycles):
        for chunk_index, (start, end) in enumerate(chunk_bounds):
            passes.append(
                {
                    "cycle_index": cycle_index,
                    "chunk_index": chunk_index,
                    "start": start,
                    "end": end,
                    "epoch_fraction": 1.0,
                }
            )

    if fractional_cycle > 0:
        for chunk_index, (start, end) in enumerate(chunk_bounds):
            passes.append(
                {
                    "cycle_index": full_cycles,
                    "chunk_index": chunk_index,
                    "start": start,
                    "end": end,
                    "epoch_fraction": fractional_cycle,
                }
            )

    return passes


def build_chunk_subset(
    shuffled_records: list[dict[str, Any]],
    chunk_pass: dict[str, Any],
    completed_passes: list[dict[str, Any]],
    replay_ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    current_records = shuffled_records[chunk_pass["start"] : chunk_pass["end"]]
    if replay_ratio <= 0 or not completed_passes:
        return list(current_records)

    replay_candidates: list[int] = []
    for previous in completed_passes:
        if previous["cycle_index"] < chunk_pass["cycle_index"] or (
            previous["cycle_index"] == chunk_pass["cycle_index"] and previous["chunk_index"] < chunk_pass["chunk_index"]
        ):
            replay_candidates.extend(range(previous["start"], previous["end"]))

    if not replay_candidates:
        return list(current_records)

    replay_target = max(1, int(math.ceil(len(current_records) * replay_ratio)))
    rng = random.Random(seed + chunk_pass["cycle_index"] * 10_000 + chunk_pass["chunk_index"])
    sampled_indices = rng.sample(replay_candidates, k=min(replay_target, len(replay_candidates)))
    merged_records = [shuffled_records[index] for index in sampled_indices] + list(current_records)
    rng.shuffle(merged_records)
    return merged_records


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    train_source_path = Path(args.train_source)
    if not train_source_path.exists():
        raise FileNotFoundError(f"Training source was not found: {train_source_path}")

    run_paths = resolve_run_paths(args.output_root, args.run_name)
    logger = setup_logging(run_paths.logs_dir / "train.log")
    previous_state = load_json(run_paths.run_state_path) or {}

    runtime_profile = detect_runtime()
    preset = suggest_training_preset(runtime_profile)
    save_json(run_paths.runtime_profile_path, runtime_profile)

    if args.per_device_train_batch_size <= 0:
        args.per_device_train_batch_size = preset["per_device_train_batch_size"]
    if args.gradient_accumulation_steps <= 0:
        args.gradient_accumulation_steps = preset["gradient_accumulation_steps"]
    if args.max_seq_length <= 0:
        args.max_seq_length = preset["max_seq_length"]
    if args.lora_rank <= 0:
        args.lora_rank = preset["lora_rank"]
    if args.lora_alpha <= 0:
        args.lora_alpha = preset["lora_alpha"]

    logger.info("Runtime profile: %s", json.dumps(runtime_profile, ensure_ascii=False))
    logger.info("Selected preset: %s", json.dumps(preset, ensure_ascii=False))

    if train_source_path.is_dir():
        records, dataset_profile = load_training_records_from_dir(train_source_path)
    else:
        records, dataset_profile = load_training_records_from_zip(train_source_path, run_paths.extract_dir)
    if args.max_train_samples > 0:
        records = records[: args.max_train_samples]
        dataset_profile["record_count_after_limit"] = len(records)
    save_json(run_paths.dataset_profile_path, dataset_profile)

    if not records:
        raise RuntimeError("No valid training records were found in the provided training source.")

    logger.info("Loaded %s valid training records", len(records))
    shuffled_records = list(records)
    random.Random(args.seed).shuffle(shuffled_records)
    chunk_passes = build_chunk_passes(len(shuffled_records), args.chunk_size, args.num_train_epochs)
    completed_chunk_passes = int(previous_state.get("completed_chunk_passes", 0))
    completed_pass_meta = chunk_passes[:completed_chunk_passes]

    logger.info(
        "Chunked training plan: %s passes, chunk_size=%s, replay_ratio=%.2f, gpu_max_memory=%sGiB, cpu_max_memory=%sGiB",
        len(chunk_passes),
        args.chunk_size,
        args.replay_ratio,
        args.gpu_max_memory_gb,
        args.cpu_max_memory_gb,
    )

    model, processor, text_backend = load_model_and_processor(
        args.model_name,
        preset["bf16"],
        runtime_profile,
        run_paths.run_dir / "offload",
        args.gpu_max_memory_gb,
        args.cpu_max_memory_gb,
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=DEFAULT_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if completed_chunk_passes > 0 and (run_paths.final_model_dir / "adapter_model.safetensors").exists():
        logger.info("Reloading adapter from %s", run_paths.final_model_dir)
        model = PeftModel.from_pretrained(model, str(run_paths.final_model_dir), is_trainable=True)
    else:
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    run_state_seed = {
        **previous_state,
        "status": "initializing",
        "run_dir": str(run_paths.run_dir),
        "checkpoints_dir": str(run_paths.checkpoints_dir),
        "final_model_dir": str(run_paths.final_model_dir),
        "model_name": args.model_name,
        "train_source": str(train_source_path.resolve()),
        "resume_mode": args.resume_mode,
        "resume_checkpoint": str(run_paths.final_model_dir) if completed_chunk_passes > 0 else None,
        "max_seq_length": args.max_seq_length,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "chunk_size": args.chunk_size,
        "replay_ratio": args.replay_ratio,
        "completed_chunk_passes": completed_chunk_passes,
        "total_chunk_passes": len(chunk_passes),
        "gpu_max_memory_gb": args.gpu_max_memory_gb,
        "cpu_max_memory_gb": args.cpu_max_memory_gb,
    }
    save_json(run_paths.run_state_path, run_state_seed)
    save_json(run_paths.latest_run_pointer_path, run_state_seed | {"run_state_path": str(run_paths.run_state_path)})

    data_collator = DataCollatorForLanguageModeling(tokenizer=text_backend, mlm=False)

    for pass_index, chunk_pass in enumerate(chunk_passes):
        if pass_index < completed_chunk_passes:
            continue

        chunk_records = build_chunk_subset(shuffled_records, chunk_pass, completed_pass_meta, args.replay_ratio, args.seed)
        tokenized_dataset = tokenize_records(Dataset.from_list(chunk_records), processor, text_backend, args.max_seq_length)
        pass_dir = ensure_dir(run_paths.checkpoints_dir / f"pass-{pass_index + 1:03d}")

        logger.info(
            "Starting chunk pass %s/%s (cycle=%s chunk=%s records=%s epoch_fraction=%.3f)",
            pass_index + 1,
            len(chunk_passes),
            chunk_pass["cycle_index"] + 1,
            chunk_pass["chunk_index"] + 1,
            len(chunk_records),
            chunk_pass["epoch_fraction"],
        )

        save_json(
            run_paths.run_state_path,
            {
                **(load_json(run_paths.run_state_path) or {}),
                "status": "running",
                "active_cycle_index": chunk_pass["cycle_index"],
                "active_chunk_index": chunk_pass["chunk_index"],
                "completed_chunk_passes": pass_index,
                "current_chunk_record_count": len(chunk_records),
                "latest_checkpoint": str(pass_dir),
            },
        )

        training_arguments = TrainingArguments(
            output_dir=str(pass_dir),
            overwrite_output_dir=True,
            num_train_epochs=chunk_pass["epoch_fraction"],
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            logging_steps=max(1, min(args.logging_steps, len(tokenized_dataset))),
            save_steps=max(1, len(tokenized_dataset)),
            save_total_limit=1,
            bf16=preset["bf16"],
            fp16=preset["fp16"],
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            report_to=[],
            seed=args.seed + pass_index,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            max_grad_norm=0.3,
            remove_unused_columns=False,
            disable_tqdm=False,
            torch_empty_cache_steps=1,
        )

        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=tokenized_dataset,
            tokenizer=text_backend,
            data_collator=data_collator,
            callbacks=[RunStateCallback(run_paths.run_state_path, run_paths.trainer_metrics_path, pass_dir)],
        )

        try:
            trainer.train()
        except KeyboardInterrupt:
            trainer.save_model(str(run_paths.final_model_dir))
            processor.save_pretrained(str(run_paths.final_model_dir))
            save_json(
                run_paths.run_state_path,
                {
                    **(load_json(run_paths.run_state_path) or {}),
                    "status": "interrupted",
                    "latest_checkpoint": str(pass_dir),
                    "completed_chunk_passes": pass_index,
                },
            )
            logger.warning("Training interrupted during chunk pass %s", pass_index + 1)
            raise

        trainer.save_model(str(pass_dir))
        trainer.save_model(str(run_paths.final_model_dir))
        processor.save_pretrained(str(pass_dir))
        processor.save_pretrained(str(run_paths.final_model_dir))
        completed_pass_meta.append(chunk_pass)

        append_jsonl(
            run_paths.chunk_history_path,
            {
                "pass_index": pass_index + 1,
                "cycle_index": chunk_pass["cycle_index"],
                "chunk_index": chunk_pass["chunk_index"],
                "start": chunk_pass["start"],
                "end": chunk_pass["end"],
                "record_count": len(chunk_records),
                "epoch_fraction": chunk_pass["epoch_fraction"],
            },
        )
        save_json(
            run_paths.run_state_path,
            {
                **(load_json(run_paths.run_state_path) or {}),
                "status": "running",
                "completed_chunk_passes": pass_index + 1,
                "latest_checkpoint": str(pass_dir),
                "final_model_dir": str(run_paths.final_model_dir),
            },
        )
        save_json(
            run_paths.latest_run_pointer_path,
            (load_json(run_paths.run_state_path) or {}) | {"run_state_path": str(run_paths.run_state_path)},
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_state = {
        **(load_json(run_paths.run_state_path) or {}),
        "status": "completed",
        "latest_checkpoint": str(run_paths.checkpoints_dir / f"pass-{len(chunk_passes):03d}") if chunk_passes else None,
        "final_model_dir": str(run_paths.final_model_dir),
        "completed_chunk_passes": len(chunk_passes),
    }
    save_json(run_paths.run_state_path, final_state)
    save_json(run_paths.latest_run_pointer_path, final_state | {"run_state_path": str(run_paths.run_state_path)})
    logger.info("Training completed. Final model saved to %s", run_paths.final_model_dir)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
