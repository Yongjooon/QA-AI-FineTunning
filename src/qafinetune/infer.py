from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from qafinetune.io_utils import (
    build_generation_jobs_from_dir,
    build_generation_prompt_from_zip,
    extract_tagged_sections,
)
from qafinetune.runtime import detect_runtime, ensure_dir, save_json, setup_logging, utc_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate QA scenarios from a fine-tuned Gemma 4 model.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--input_source", "--input_zip", dest="input_source", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--max_new_tokens", type=int, default=800)
    parser.add_argument("--max_input_tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
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


def load_base_and_adapter(
    model_name: str,
    adapter_path: str,
    bf16: bool,
    runtime_profile: dict[str, Any],
    offload_dir: Path,
):
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    processor = load_processor(adapter_path, model_name)
    text_backend = resolve_text_backend(processor)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
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
        load_kwargs["max_memory"] = {0: "5GiB", "cpu": "64GiB"}
        load_kwargs["low_cpu_mem_usage"] = True
        load_kwargs["offload_folder"] = str(offload_dir)
        load_kwargs["offload_state_dict"] = True

    base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
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


def truncate_inputs(inputs: dict[str, Any], max_input_tokens: int) -> dict[str, Any]:
    input_ids = inputs.get("input_ids")
    if input_ids is None or input_ids.shape[-1] <= max_input_tokens:
        return inputs

    keep_head = min(256, max_input_tokens // 4)
    keep_tail = max_input_tokens - keep_head
    truncated: dict[str, Any] = {}
    for key, value in inputs.items():
        if not hasattr(value, "shape") or value.ndim < 2 or value.shape[-1] != input_ids.shape[-1]:
            truncated[key] = value
            continue
        truncated[key] = torch.cat([value[..., :keep_head], value[..., -keep_tail:]], dim=-1)
    return truncated


def main() -> None:
    args = parse_args()
    input_source_path = Path(args.input_source)
    if not input_source_path.exists():
        raise FileNotFoundError(f"Input source was not found: {input_source_path}")

    output_paths = resolve_output_paths(args.output_root, args.run_name)
    logger = setup_logging(output_paths["logs_dir"] / "generate.log")

    runtime_profile = detect_runtime()
    if not runtime_profile["cuda_available"]:
        raise RuntimeError("CUDA GPU was not detected. Use a GPU-backed Colab runtime.")
    save_json(output_paths["runtime_profile_path"], runtime_profile)

    if input_source_path.is_dir():
        generation_jobs = build_generation_jobs_from_dir(input_source_path)
        input_profile = {
            "mode": "page_split" if len(generation_jobs) > 1 else "single_dir",
            "job_count": len(generation_jobs),
            "jobs": [{"job_name": job["job_name"], "profile": job["profile"]} for job in generation_jobs],
        }
    else:
        prompt, prompt_profile = build_generation_prompt_from_zip(input_source_path, output_paths["extract_dir"])
        generation_jobs = [{"job_name": "input_zip", "prompt": prompt, "profile": prompt_profile}]
        input_profile = {
            "mode": "zip",
            "job_count": 1,
            "jobs": [{"job_name": "input_zip", "profile": prompt_profile}],
        }
    save_json(output_paths["input_profile_path"], input_profile)

    model, processor, text_backend = load_base_and_adapter(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        bf16=runtime_profile["bf16_supported"],
        runtime_profile=runtime_profile,
        offload_dir=output_paths["run_dir"] / "offload",
    )

    page_results: list[dict[str, Any]] = []
    combined_summary_parts: list[str] = []
    pages_dir = ensure_dir(output_paths["run_dir"] / "pages")

    for job_index, job in enumerate(generation_jobs, start=1):
        prompt = job["prompt"]
        job_name = job["job_name"]
        logger.info("Starting generation for job %s/%s: %s", job_index, len(generation_jobs), job_name)
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
            inputs = text_backend(fallback_prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens)

        inputs = truncate_inputs(inputs, args.max_input_tokens)

        model_device = next(model.parameters()).device
        inputs = {
            key: value.to(model_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        prompt_length = inputs["input_ids"].shape[-1]
        logger.info("Job %s prompt token length: %s", job_name, prompt_length)
        torch.cuda.empty_cache()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.temperature > 0,
                pad_token_id=text_backend.eos_token_id,
            )

        generated_text = processor.decode(generated[0][prompt_length:], skip_special_tokens=True)
        page_dir = ensure_dir(pages_dir / job_name)
        (page_dir / "raw_model_output.txt").write_text(generated_text, encoding="utf-8")

        scenario_description, json_text = extract_tagged_sections(generated_text)
        if not scenario_description:
            scenario_description = "Scenario summary could not be cleanly extracted. Check raw_model_output.txt."

        parsed_json = None
        json_error = None
        try:
            parsed_json = safe_parse_json(json_text)
        except Exception as exc:
            json_error = str(exc)

        page_payload = parsed_json
        if page_payload is None:
            page_payload = {
                "error": "Model output did not contain valid JSON.",
                "json_error": json_error,
                "raw_json_candidate": json_text,
            }

        (page_dir / "playwright_scenario.json").write_text(json.dumps(page_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (page_dir / "scenario_summary.md").write_text(scenario_description.strip() + "\n", encoding="utf-8")

        combined_summary_parts.append(f"# {job_name}\n\n{scenario_description.strip()}")
        page_results.append(
            {
                "page_name": job_name,
                "profile": job["profile"],
                "json_parse_error": json_error,
                "playwright_json": page_payload,
            }
        )

    combined_payload = {
        "mode": input_profile["mode"],
        "page_count": len(page_results),
        "page_results": page_results,
    }
    output_paths["raw_output_path"].write_text(
        "\n\n".join(
            (pages_dir / result["page_name"] / "raw_model_output.txt").read_text(encoding="utf-8")
            for result in page_results
        ),
        encoding="utf-8",
    )
    output_paths["json_output_path"].write_text(json.dumps(combined_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    output_paths["summary_output_path"].write_text("\n\n".join(combined_summary_parts).strip() + "\n", encoding="utf-8")
    save_json(
        output_paths["generation_meta_path"],
        {
            "model_name": args.model_name,
            "adapter_path": str(Path(args.adapter_path).resolve()),
            "input_source": str(input_source_path.resolve()),
            "run_dir": str(output_paths["run_dir"].resolve()),
            "max_new_tokens": args.max_new_tokens,
            "max_input_tokens": args.max_input_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "job_count": len(page_results),
        },
    )
    logger.info("Generation completed. Outputs saved to %s", output_paths["run_dir"])


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
