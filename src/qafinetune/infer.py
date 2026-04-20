from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from qafinetune.io_utils import build_generation_prompt_from_zip, extract_tagged_sections
from qafinetune.runtime import detect_runtime, ensure_dir, save_json, setup_logging, utc_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate QA scenarios from a fine-tuned Gemma 4 model.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--input_zip", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--max_new_tokens", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    return parser.parse_args()


def resolve_output_paths(output_root: str | Path, run_name: str) -> dict[str, Path]:
    base_root = ensure_dir(output_root)
    resolved_run_name = run_name or f"generate_{utc_timestamp()}"
    run_dir = ensure_dir(base_root / "generated_runs" / resolved_run_name)
    return {
        "run_dir": run_dir,
        "extract_dir": ensure_dir(run_dir / "extracted_input"),
        "logs_dir": ensure_dir(run_dir / "logs"),
        "raw_output_path": run_dir / "raw_model_output.txt",
        "json_output_path": run_dir / "playwright_scenario.json",
        "summary_output_path": run_dir / "scenario_summary.md",
        "runtime_profile_path": run_dir / "runtime_profile.json",
        "input_profile_path": run_dir / "input_profile.json",
        "generation_meta_path": run_dir / "generation_meta.json",
    }


def load_processor(adapter_path: str, model_name: str):
    try:
        processor = AutoProcessor.from_pretrained(adapter_path, trust_remote_code=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    return processor


def resolve_text_backend(processor):
    text_backend = getattr(processor, "tokenizer", None) or processor
    if hasattr(text_backend, "padding_side"):
        text_backend.padding_side = "left"
    pad_token = getattr(text_backend, "pad_token", None)
    eos_token = getattr(text_backend, "eos_token", None)
    if pad_token is None and eos_token is not None:
        text_backend.pad_token = eos_token
    return text_backend


def load_base_and_adapter(model_name: str, adapter_path: str, bf16: bool):
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    processor = load_processor(adapter_path, model_name)
    text_backend = resolve_text_backend(processor)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    base_model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=compute_dtype,
        quantization_config=quantization_config,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, processor, text_backend


def build_messages(prompt: str) -> list[dict[str, Any]]:
    system_prompt = (
        "You generate QA test scenarios.\n"
        "Always produce both a human-readable scenario summary and a strict Playwright JSON output.\n"
        "Do not omit the required tags."
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


def safe_parse_json(text: str) -> Any:
    if not text:
        return None
    return json.loads(text)


def main() -> None:
    args = parse_args()
    input_zip_path = Path(args.input_zip)
    if not input_zip_path.exists():
        raise FileNotFoundError(f"Input zip was not found: {input_zip_path}")

    output_paths = resolve_output_paths(args.output_root, args.run_name)
    logger = setup_logging(output_paths["logs_dir"] / "generate.log")

    runtime_profile = detect_runtime()
    if not runtime_profile["cuda_available"]:
        raise RuntimeError("CUDA GPU was not detected. Use a GPU-backed Colab runtime.")
    save_json(output_paths["runtime_profile_path"], runtime_profile)

    prompt, input_profile = build_generation_prompt_from_zip(args.input_zip, output_paths["extract_dir"])
    save_json(output_paths["input_profile_path"], input_profile)

    model, processor, text_backend = load_base_and_adapter(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        bf16=runtime_profile["bf16_supported"],
    )

    messages = build_messages(prompt)
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
    except Exception:
        fallback_prompt = (
            "System:\n"
            "Generate a QA scenario summary and strict Playwright JSON.\n\n"
            f"User:\n{prompt}\n\nAssistant:\n"
        )
        inputs = text_backend(fallback_prompt, return_tensors="pt")

    model_device = next(model.parameters()).device
    inputs = {
        key: value.to(model_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    logger.info("Starting generation")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=text_backend.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    generated_text = processor.decode(generated[0][prompt_length:], skip_special_tokens=True)
    output_paths["raw_output_path"].write_text(generated_text, encoding="utf-8")

    scenario_description, json_text = extract_tagged_sections(generated_text)
    if not scenario_description:
        scenario_description = "Scenario summary could not be cleanly extracted. Check raw_model_output.txt."

    parsed_json = None
    json_error = None
    try:
        parsed_json = safe_parse_json(json_text)
    except Exception as exc:
        json_error = str(exc)

    if parsed_json is None:
        fallback_payload = {
            "error": "Model output did not contain valid JSON.",
            "json_error": json_error,
            "raw_json_candidate": json_text,
        }
        output_paths["json_output_path"].write_text(json.dumps(fallback_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        output_paths["json_output_path"].write_text(json.dumps(parsed_json, indent=2, ensure_ascii=False), encoding="utf-8")

    output_paths["summary_output_path"].write_text(scenario_description.strip() + "\n", encoding="utf-8")
    save_json(
        output_paths["generation_meta_path"],
        {
            "model_name": args.model_name,
            "adapter_path": str(Path(args.adapter_path).resolve()),
            "input_zip": str(input_zip_path.resolve()),
            "run_dir": str(output_paths["run_dir"].resolve()),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "json_parse_error": json_error,
        },
    )
    logger.info("Generation completed. Outputs saved to %s", output_paths["run_dir"])


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
