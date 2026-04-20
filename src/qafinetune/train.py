from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from qafinetune.io_utils import find_latest_checkpoint, load_training_records_from_dir, load_training_records_from_zip
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


class RunStateCallback(TrainerCallback):
    def __init__(self, run_state_path: Path, metrics_path: Path, checkpoints_dir: Path) -> None:
        self.run_state_path = run_state_path
        self.metrics_path = metrics_path
        self.checkpoints_dir = checkpoints_dir

    def _write_state(self, payload: dict[str, Any]) -> None:
        existing = load_json(self.run_state_path) or {}
        existing.update(payload)
        save_json(self.run_state_path, existing)

    def on_train_begin(self, args, state, control, **kwargs):
        self._write_state(
            {
                "status": "running",
                "global_step": state.global_step,
                "epoch": state.epoch,
            }
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": state.global_step, "epoch": state.epoch, **logs}, ensure_ascii=False) + "\n")

    def on_save(self, args, state, control, **kwargs):
        latest_checkpoint = find_latest_checkpoint(self.checkpoints_dir)
        self._write_state(
            {
                "status": "running",
                "global_step": state.global_step,
                "epoch": state.epoch,
                "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
            }
        )

    def on_train_end(self, args, state, control, **kwargs):
        latest_checkpoint = find_latest_checkpoint(self.checkpoints_dir)
        self._write_state(
            {
                "status": "completed",
                "global_step": state.global_step,
                "epoch": state.epoch,
                "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
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
        tokens = text_backend(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        tokens["labels"] = list(tokens["input_ids"])
        return tokens

    return dataset.map(_tokenize, remove_columns=dataset.column_names)


def load_model_and_processor(model_name: str, bf16: bool):
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    text_backend = resolve_text_backend(processor)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=compute_dtype,
        quantization_config=quantization_config,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    return model, processor, text_backend


def resolve_resume_checkpoint(args: argparse.Namespace, run_paths: RunPaths) -> str | None:
    if args.resume_mode == "never":
        return None
    if args.resume_mode == "path":
        return args.resume_checkpoint or None

    state_payload = load_json(run_paths.run_state_path) or {}
    checkpoint_from_state = state_payload.get("latest_checkpoint")
    if checkpoint_from_state and Path(checkpoint_from_state).exists():
        return checkpoint_from_state

    latest_checkpoint = find_latest_checkpoint(run_paths.checkpoints_dir)
    return str(latest_checkpoint) if latest_checkpoint else None


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

    dataset = Dataset.from_list(records)
    model, processor, text_backend = load_model_and_processor(args.model_name, preset["bf16"])

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=DEFAULT_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized_dataset = tokenize_records(dataset, processor, text_backend, args.max_seq_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer=text_backend, mlm=False)

    training_arguments = TrainingArguments(
        output_dir=str(run_paths.checkpoints_dir),
        overwrite_output_dir=False,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=preset["bf16"],
        fp16=preset["fp16"],
        dataloader_num_workers=2,
        report_to=[],
        seed=args.seed,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        remove_unused_columns=False,
    )

    resume_checkpoint = resolve_resume_checkpoint(args, run_paths)
    if resume_checkpoint:
        logger.info("Resuming from checkpoint: %s", resume_checkpoint)
    else:
        logger.info("Starting a fresh training run")

    run_state_seed = {
        **previous_state,
        "status": "initializing",
        "run_dir": str(run_paths.run_dir),
        "checkpoints_dir": str(run_paths.checkpoints_dir),
        "final_model_dir": str(run_paths.final_model_dir),
        "model_name": args.model_name,
        "train_source": str(train_source_path.resolve()),
        "resume_mode": args.resume_mode,
        "resume_checkpoint": resume_checkpoint,
        "max_seq_length": args.max_seq_length,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
    }
    save_json(run_paths.run_state_path, run_state_seed)
    save_json(run_paths.latest_run_pointer_path, run_state_seed | {"run_state_path": str(run_paths.run_state_path)})

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_dataset,
        tokenizer=text_backend,
        data_collator=data_collator,
        callbacks=[RunStateCallback(run_paths.run_state_path, run_paths.trainer_metrics_path, run_paths.checkpoints_dir)],
    )

    try:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    except KeyboardInterrupt:
        latest_checkpoint = find_latest_checkpoint(run_paths.checkpoints_dir)
        save_json(
            run_paths.run_state_path,
            {
                **(load_json(run_paths.run_state_path) or {}),
                "status": "interrupted",
                "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
            },
        )
        logger.warning("Training was interrupted. Latest checkpoint: %s", latest_checkpoint)
        raise

    trainer.save_model(str(run_paths.final_model_dir))
    processor.save_pretrained(str(run_paths.final_model_dir))

    latest_checkpoint = find_latest_checkpoint(run_paths.checkpoints_dir)
    final_state = {
        **(load_json(run_paths.run_state_path) or {}),
        "status": "completed",
        "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
        "final_model_dir": str(run_paths.final_model_dir),
    }
    save_json(run_paths.run_state_path, final_state)
    save_json(run_paths.latest_run_pointer_path, final_state | {"run_state_path": str(run_paths.run_state_path)})
    logger.info("Training completed. Final model saved to %s", run_paths.final_model_dir)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
