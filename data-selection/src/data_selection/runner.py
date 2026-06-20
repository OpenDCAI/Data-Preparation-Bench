from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from data_selection.dataset import DatasetConfig
from data_selection.utils import read_jsonl, write_jsonl


def run_selection(
    selector: Any,
    input_path: str,
    output_path: str,
    k: int | Sequence[int] | None = None,
    *,
    instruction_key: str = "instruction",
    output_key: str = "output",
    conversations_key: str = "conversations",
) -> None:
    """Load data, run a selector, and write JSONL output.

    For score-based selectors (``_score_based = True``) and multiple ``k``
    values, the selector is run once with ``max(k)`` and the result is
    truncated for each ``k``. For random-based selectors each ``k`` is run
    independently so that seeds remain deterministic per output.

    Output path may contain a ``{k}`` placeholder; otherwise ``_k`` is
    inserted before the extension when multiple ``k`` values are requested.
    """
    dataset_cfg = DatasetConfig(
        path=input_path,
        output=output_path,
        instruction_key=instruction_key,
        output_key=output_key,
        conversations_key=conversations_key,
    )
    samples = read_jsonl(dataset_cfg.path, dataset_cfg)

    if k is None:
        k_values: list[int] = [selector.k]
    elif isinstance(k, int):
        k_values = [k]
    else:
        k_values = list(k)

    is_score_based = getattr(type(selector), "_score_based", False)

    if is_score_based and len(k_values) > 1:
        max_k = max(k_values)
        selector.k = max_k
        all_selected = selector.select(samples)
        for kk in sorted(k_values):
            out = _resolve_output_path(output_path, kk)
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            truncated = all_selected[:kk]
            write_jsonl(truncated, out)
            print(f"Selected {len(truncated)} samples -> {out}")
    else:
        for kk in k_values:
            selector.k = kk
            selected = selector.select(samples)
            out = (
                _resolve_output_path(output_path, kk)
                if len(k_values) > 1
                else output_path
            )
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            write_jsonl(selected, out)
            print(f"Selected {len(selected)} samples -> {out}")


def _resolve_output_path(template: str, k: int) -> str:
    if "{k}" in template:
        return template.replace("{k}", str(k))
    base, ext = os.path.splitext(template)
    return f"{base}_{k}{ext}"
