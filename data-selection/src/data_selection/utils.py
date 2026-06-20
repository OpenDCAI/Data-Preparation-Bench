from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from data_selection.dataset import DatasetConfig


def read_jsonl(path: str | Path, dataset_cfg: DatasetConfig) -> list[dict[str, Any]]:
    """Read samples from JSONL and normalize via dataset config."""
    samples: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(dataset_cfg.format(json.loads(line)))
    return samples


def write_jsonl(samples: Sequence[dict[str, Any]], path: str | Path) -> None:
    """Write samples to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def extract_text(sample: Mapping[str, Any]) -> str:
    """Extract concatenated text from normalized messages."""
    messages = sample.get("messages", [])
    if not messages:
        return ""
    return "\n".join(msg.get("content", "") for msg in messages)
