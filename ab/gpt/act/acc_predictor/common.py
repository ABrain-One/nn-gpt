"""Shared low-level helpers: safe coercion, JSONL IO, prompt selection, chat templating."""
from __future__ import annotations

import json
from pathlib import Path

from . import config
from .config import *  # noqa: F401,F403




def _active_system_prompt() -> str:
    return COT_SYSTEM_PROMPT if config.USE_CHAIN_OF_THOUGHT else SYSTEM_PROMPT


def _safe_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _apply_chat_template_for_inference(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _stream_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _write_jsonl(path: Path, data: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_messages(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    examples: list[dict] = []
    for row in _stream_jsonl(path):
        if "messages" in row and row.get("architecture_id"):
            examples.append({"messages": row["messages"], "architecture_id": str(row["architecture_id"])})
    if not examples:
        raise ValueError(f"No examples loaded from {path}")
    return examples
