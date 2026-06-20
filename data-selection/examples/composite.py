from __future__ import annotations

from data_selection import CompositeSelector, LengthBasedSelector, RandomSelector
from data_selection.runner import run_selection

if __name__ == "__main__":
    selector = CompositeSelector(
        k=50,
        selectors=[
            RandomSelector(k=100, seed=42),
            LengthBasedSelector(k=50),
        ],
    )
    run_selection(
        selector,
        input_path="data/input.jsonl",
        output_path="data/output_composite.jsonl",
    )
