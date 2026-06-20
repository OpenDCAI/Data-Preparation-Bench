from __future__ import annotations

from data_selection import PerplexityBasedSelector
from data_selection.runner import run_selection

if __name__ == "__main__":
    selector = PerplexityBasedSelector(
        k=100000,
        strategy="high",
        text_key="text",
        lang="en",
        model_name="dataflow/operators/eval/GeneralText/models/Kenlm/wikipedia",
    )
    run_selection(
        selector,
        input_path="/jizhicfs/linyibo/datasets/dyyyyyyyy/ScaleQuest-Math/scalequest_math.jsonl",
        output_path="data/output_perplexity_based_high.jsonl",
    )
