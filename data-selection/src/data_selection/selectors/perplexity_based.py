from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from dataflow.operators.eval import PerplexityScorer

from data_selection.score_cache import ScoreCache
from data_selection.utils import extract_text


class PerplexityBasedSelector:
    """Select samples by perplexity using DataFlow's Kenlm-based scorer."""

    _score_based = True

    def __init__(
        self,
        k: int = 100,
        strategy: str = "low",
        text_key: str = "text",
        lang: str = "en",
        model_name: str = "dataflow/operators/eval/GeneralText/models/Kenlm/wikipedia",
        scorer: PerplexityScorer | None = None,
        scores_cache_path: str | None = None,
        batch_size: int = 1000,
    ) -> None:
        if strategy not in ("low", "high", "mid"):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.k = k
        self.strategy = strategy
        self.text_key = text_key
        self.batch_size = batch_size
        self.scores_cache_path = scores_cache_path
        self.scorer = scorer or PerplexityScorer(lang=lang, model_name=model_name)

    def select(self, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if self.k <= 0 or not samples:
            return []

        cache = ScoreCache(self.scores_cache_path) if self.scores_cache_path else None
        missing = (
            cache.missing_indices(len(samples)) if cache else list(range(len(samples)))
        )

        if missing:
            print(
                f"[PerplexityBasedSelector] Scoring {len(missing)} samples (cached: {len(samples) - len(missing)})"
            )

            for batch_start in range(0, len(missing), self.batch_size):
                batch_indices = missing[batch_start : batch_start + self.batch_size]
                batch_samples = [samples[i] for i in batch_indices]

                df = pd.DataFrame(batch_samples)
                if self.text_key not in df.columns:
                    df[self.text_key] = [extract_text(s) for s in batch_samples]

                ppl_scores = self.scorer.eval(df, input_key=self.text_key)

                entries: list[tuple[int, dict[str, Any]]] = []
                for j, idx in enumerate(batch_indices):
                    entries.append((idx, {"ppl": float(ppl_scores[j])}))

                if cache:
                    cache.put_batch(entries)
                else:
                    if not hasattr(self, "_temp_scores"):
                        self._temp_scores: dict[int, dict[str, Any]] = {}
                    for idx, scores in entries:
                        self._temp_scores[idx] = scores

        # Collect all scores
        all_scores = cache.all_scores() if cache else getattr(self, "_temp_scores", {})

        ordered: list[tuple[float, int]] = []
        for i in range(len(samples)):
            sc = all_scores.get(i)
            if sc is None:
                continue
            ordered.append((sc["ppl"], i))

        if self.strategy == "low":
            ordered.sort(key=lambda x: x[0])
        elif self.strategy == "high":
            ordered.sort(key=lambda x: x[0], reverse=True)
        else:
            mean_ppl = sum(v[0] for v in ordered) / len(ordered)
            ordered.sort(key=lambda x: abs(math.log(x[0]) - math.log(mean_ppl)))

        return [
            {
                **samples[idx],
                "meta": {
                    "selector": "PerplexityBasedSelector",
                    "ppl": ppl,
                    "strategy": self.strategy,
                },
            }
            for ppl, idx in ordered[: min(self.k, len(ordered))]
        ]
