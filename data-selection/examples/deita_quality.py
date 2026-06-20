from __future__ import annotations

from dataflow.operators.eval import DeitaComplexityScorer, DeitaQualityScorer

from data_selection import DeitaQualitySelector
from data_selection.runner import run_selection

if __name__ == "__main__":
    selector = DeitaQualitySelector(
        k=100,
        device="cuda",
        quality_scorer=DeitaQualityScorer(device="cuda"),
        complexity_scorer=DeitaComplexityScorer(device="cuda"),
    )
    run_selection(
        selector,
        input_path="data/input.jsonl",
        output_path="data/output_deita_quality.jsonl",
    )
