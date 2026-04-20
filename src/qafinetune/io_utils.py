from __future__ import annotations

import csv
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


PROMPT_FIELDS = [
    "prompt",
    "question",
    "instruction",
    "input",
    "user_input",
    "context",
    "query",
    "request",
]

RESPONSE_FIELDS = [
    "response",
    "answer",
    "output",
    "completion",
    "target",
    "assistant_output",
    "expected_output",
]

JSON_OUTPUT_FIELDS = [
    "playwright_json",
    "scenario_json",
    "json_output",
    "result_json",
]

TEXT_OUTPUT_FIELDS = [
    "scenario_summary",
    "scenario_text",
    "description",
    "narrative",
    "summary",
]

MESSAGE_FIELDS = ["messages", "conversation", "chat", "dialogue", "dialogs"]


def unzip_to_dir(zip_path: str | Path, output_dir: str | Path) -> Path:
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)
    return output_dir


def list_data_files(root_dir: str | Path) -> list[Path]:
    root = Path(root_dir)
    supported = {".json", ".jsonl", ".csv", ".parquet", ".txt", ".md"}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in supported)


def read_records_from_file(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            if isinstance(payload.get("data"), list):
                return [item for item in payload["data"] if isinstance(item, dict)]
            return [payload]
        return []

    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        for line in source.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                records.append(item)
        return records

    if suffix == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    if suffix == ".parquet":
        frame = pd.read_parquet(source)
        return frame.to_dict(orient="records")

    if suffix in {".txt", ".md"}:
        return [{"prompt": source.read_text(encoding="utf-8")}]

    return []


def _first_present(record: dict[str, Any], field_names: list[str]) -> Any:
    for field_name in field_names:
        if field_name in record and record[field_name] not in (None, ""):
            return record[field_name]
    return None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2).strip()


def _normalize_messages(value: Any) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if not isinstance(value, list):
        return normalized

    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", item.get("speaker", "user"))).strip().lower()
        content = item.get("content", item.get("text", item.get("message", "")))
        text = _normalize_text(content)
        if not text:
            continue
        if role not in {"system", "user", "assistant"}:
            role = "user" if role in {"human", "customer"} else "assistant"
        normalized.append({"role": role, "content": text})
    return normalized


def canonicalize_training_record(record: dict[str, Any]) -> dict[str, Any] | None:
    raw_messages = _first_present(record, MESSAGE_FIELDS)
    messages = _normalize_messages(raw_messages)

    if messages:
        assistant_messages = [item for item in messages if item["role"] == "assistant"]
        user_messages = [item for item in messages if item["role"] == "user"]
        if not assistant_messages or not user_messages:
            return None
        prompt_preview = user_messages[-1]["content"]
        response_preview = assistant_messages[-1]["content"]
        return {
            "messages": messages,
            "prompt": prompt_preview,
            "response": response_preview,
            "source_fields": sorted(record.keys()),
        }

    prompt = _normalize_text(_first_present(record, PROMPT_FIELDS))
    response = _normalize_text(_first_present(record, RESPONSE_FIELDS))
    json_output = _normalize_text(_first_present(record, JSON_OUTPUT_FIELDS))
    text_output = _normalize_text(_first_present(record, TEXT_OUTPUT_FIELDS))

    if not response:
        chunks = []
        if json_output:
            chunks.append("### playwright_json")
            chunks.append(json_output)
        if text_output:
            chunks.append("### scenario_description")
            chunks.append(text_output)
        response = "\n\n".join(chunks).strip()

    if not prompt or not response:
        return None

    return {
        "messages": [],
        "prompt": prompt,
        "response": response,
        "source_fields": sorted(record.keys()),
    }


def load_training_records_from_zip(zip_path: str | Path, extract_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    extracted = unzip_to_dir(zip_path, extract_dir)
    records: list[dict[str, Any]] = []
    file_summaries: list[dict[str, Any]] = []
    skipped = 0

    for file_path in list_data_files(extracted):
        raw_records = read_records_from_file(file_path)
        accepted = 0
        for raw_record in raw_records:
            canonical = canonicalize_training_record(raw_record)
            if canonical is None:
                skipped += 1
                continue
            records.append(canonical)
            accepted += 1
        file_summaries.append(
            {
                "file": str(file_path.relative_to(extracted)),
                "rows_read": len(raw_records),
                "rows_accepted": accepted,
            }
        )

    profile = {
        "zip_path": str(Path(zip_path).resolve()),
        "extract_dir": str(extracted.resolve()),
        "record_count": len(records),
        "skipped_records": skipped,
        "files": file_summaries,
    }
    return records, profile


def build_generation_prompt_from_zip(zip_path: str | Path, extract_dir: str | Path, max_chars_per_file: int = 12000) -> tuple[str, dict[str, Any]]:
    extracted = unzip_to_dir(zip_path, extract_dir)
    sections: list[str] = []
    files_summary: list[dict[str, Any]] = []

    for file_path in list_data_files(extracted):
        content = render_file_for_generation_prompt(file_path)
        clipped = content[:max_chars_per_file]
        rel_path = str(file_path.relative_to(extracted))
        sections.append(f"## File: {rel_path}\n{clipped}")
        files_summary.append(
            {
                "file": rel_path,
                "characters_used": len(clipped),
                "characters_total": len(content),
            }
        )

    if not sections:
        raise RuntimeError("No supported input files were found in the provided zip file.")

    bundled_input = "\n\n".join(sections)
    prompt = (
        "You are a QA scenario generation model.\n"
        "Read the bundled project materials below and produce:\n"
        "1. A concise human-readable scenario summary.\n"
        "2. A strict Playwright-ready JSON payload.\n\n"
        "Return your answer in the exact format below:\n"
        "[SCENARIO_DESCRIPTION]\n"
        "...text...\n"
        "[/SCENARIO_DESCRIPTION]\n"
        "[PLAYWRIGHT_JSON]\n"
        "{...json...}\n"
        "[/PLAYWRIGHT_JSON]\n\n"
        "Input bundle:\n\n"
        f"{bundled_input}"
    )

    profile = {
        "zip_path": str(Path(zip_path).resolve()),
        "extract_dir": str(extracted.resolve()),
        "files": files_summary,
    }
    return prompt, profile


def render_file_for_generation_prompt(path: str | Path) -> str:
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix in {".txt", ".md"}:
        return source.read_text(encoding="utf-8", errors="ignore")

    records = read_records_from_file(source)
    if not records:
        return ""

    rendered_rows: list[str] = []
    for index, record in enumerate(records, start=1):
        rendered_rows.append(f"### Record {index}")
        rendered_rows.append(json.dumps(record, ensure_ascii=False, indent=2))
    return "\n\n".join(rendered_rows)


def find_latest_checkpoint(checkpoints_dir: str | Path) -> Path | None:
    root = Path(checkpoints_dir)
    if not root.exists():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir() and re.match(r"checkpoint-\d+", path.name)]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: int(path.name.split("-")[-1]))[-1]


def extract_tagged_sections(text: str) -> tuple[str, str]:
    description_match = re.search(
        r"\[SCENARIO_DESCRIPTION\](.*?)\[/SCENARIO_DESCRIPTION\]",
        text,
        flags=re.DOTALL,
    )
    json_match = re.search(
        r"\[PLAYWRIGHT_JSON\](.*?)\[/PLAYWRIGHT_JSON\]",
        text,
        flags=re.DOTALL,
    )

    description = description_match.group(1).strip() if description_match else ""
    json_text = json_match.group(1).strip() if json_match else extract_first_json_block(text)
    return description, json_text


def extract_first_json_block(text: str) -> str:
    fence_match = re.search(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    start_index = -1
    stack: list[str] = []
    for index, char in enumerate(text):
        if char in "{[":
            if start_index == -1:
                start_index = index
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack:
                continue
            expected = stack.pop()
            if char != expected:
                continue
            if not stack and start_index != -1:
                return text[start_index : index + 1].strip()
    return ""
