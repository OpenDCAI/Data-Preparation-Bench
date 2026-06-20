from __future__ import annotations

from dataflow.operators.eval import FineWebEduScorer, PairQualScorer

from data_selection import QualityScorerSelector
from data_selection.runner import run_selection

if __name__ == "__main__":
    selector = QualityScorerSelector(
        k=100000,
        strategy="composite",
        text_key="text",
        device="cuda",
        model_cache_dir="./dataflow_cache",
        lang="en",
        edu_scorer=FineWebEduScorer(model_cache_dir="./dataflow_cache", device="cuda"),
        pq_scorer=PairQualScorer(
            model_cache_dir="./dataflow_cache",
            device="cuda",
            lang="en",
            max_length=512,
        ),
    )
    run_selection(
        selector,
        input_path="/jizhicfs/linyibo/datasets/dyyyyyyyy/ScaleQuest-Math/scalequest_math.jsonl",
        output_path="data/output_quality_scorer.jsonl",
    )
