from __future__ import annotations

from data_selection import SourceBalancedRandomSelector
from data_selection.runner import run_selection

if __name__ == "__main__":
    selector = SourceBalancedRandomSelector(
        k=100000,
        source_key="source",
        seed=42,
    )
    run_selection(
        selector,
        input_path="/jizhicfs/linyibo/datasets/dyyyyyyyy/ScaleQuest-Math/scalequest_math.jsonl",
        output_path="data/output_source_balanced_random.jsonl",
    )
