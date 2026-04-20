from __future__ import annotations

import json
import logging
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def setup_logging(log_path: str | Path) -> logging.Logger:
    logger = logging.getLogger("qafinetune")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def detect_runtime() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu_name = None
    gpu_memory_gb = 0.0
    capability = None
    bf16_supported = False
    device_count = 0

    if cuda_available:
        device_count = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_memory_gb = round(props.total_memory / 1024**3, 2)
        capability = f"{props.major}.{props.minor}"
        bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu_count": device_count,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
        "compute_capability": capability,
        "bf16_supported": bf16_supported,
        "cwd": os.getcwd(),
    }


def suggest_training_preset(runtime_profile: dict[str, Any]) -> dict[str, Any]:
    if not runtime_profile["cuda_available"]:
        raise RuntimeError("CUDA GPU was not detected. Use a GPU-backed Colab runtime.")

    memory_gb = runtime_profile["gpu_memory_gb"]
    bf16 = runtime_profile["bf16_supported"]

    if memory_gb >= 39:
        return {
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 3072,
            "lora_rank": 64,
            "lora_alpha": 128,
            "bf16": bf16,
            "fp16": not bf16,
        }
    if memory_gb >= 21:
        return {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 2048,
            "lora_rank": 32,
            "lora_alpha": 64,
            "bf16": bf16,
            "fp16": not bf16,
        }
    return {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "max_seq_length": 1536,
        "lora_rank": 16,
        "lora_alpha": 32,
        "bf16": bf16,
        "fp16": not bf16,
    }
