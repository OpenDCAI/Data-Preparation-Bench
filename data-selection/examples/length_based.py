from __future__ import annotations

from data_selection import LengthBasedSelector
from data_selection.runner import run_selection

if __name__ == "__main__":
    selector = LengthBasedSelector(k=100)
    run_selection(
        selector,
        input_path="data/input.jsonl",
        output_path="data/output_length_based.jsonl",
    )
