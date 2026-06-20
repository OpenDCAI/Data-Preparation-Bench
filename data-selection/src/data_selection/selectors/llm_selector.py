from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from dataflow.core import LLMServingABC
from dataflow.operators.eval import MetaScorer

from data_selection.utils import extract_text


class LLMAsSelector:
    """Use an LLM to score and rank samples via DataFlow's MetaScorer."""

    _score_based = True

    def __init__(
        self,
        k: int = 100,
        text_key: str = "text",
        llm_serving: LLMServingABC | None = None,
        dimensions: list[dict[str, Any]] | None = None,
        scorer: MetaScorer | None = None,
    ) -> None:
        self.k = k
        self.text_key = text_key
        if scorer is not None:
            self.scorer: MetaScorer | None = scorer
        elif llm_serving is not None:
            self.scorer = MetaScorer(
                llm_serving=llm_serving,
                dimensions=dimensions or [],
            )
        else:
            self.scorer = None

    def select(self, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if self.k <= 0 or not samples:
            return []

        df = pd.DataFrame(samples)
        if self.text_key not in df.columns:
            df[self.text_key] = [extract_text(s) for s in samples]

        if self.scorer is not None:
            scores_2d = self.scorer.eval(df, input_key=self.text_key)
            avg_scores: list[float] = [float(sum(row)) / len(row) for row in scores_2d]
        else:
            scores_2d = None
            avg_scores = [float(len(extract_text(s).split())) for s in samples]

        paired = list(zip(avg_scores, samples, strict=True))
        paired.sort(key=lambda x: x[0], reverse=True)

        result: list[dict[str, Any]] = []
        for i, (score, s) in enumerate(paired[: min(self.k, len(paired))]):
            orig_idx = samples.index(s)
            dim_scores: list[float] | None = (
                [round(float(v), 4) for v in scores_2d[orig_idx]]
                if scores_2d is not None
                else None
            )
            result.append(
                {
                    **s,
                    "meta": {
                        "selector": "LLMAsSelector",
                        "dimension_scores": dim_scores,
                        "average_score": round(score, 4),
                    },
                }
            )

        return result
