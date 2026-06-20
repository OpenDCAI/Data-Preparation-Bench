from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from dataflow.operators.eval import FineWebEduScorer, PairQualScorer

from data_selection.score_cache import ScoreCache
from data_selection.utils import extract_text


class QualityScorerSelector:
    """Select samples by educational/quality score using DataFlow scorers."""

    _score_based = True

    def __init__(
        self,
        k: int = 100,
        strategy: str = "composite",
        text_key: str = "text",
        device: str = "cuda",
        model_cache_dir: str = "./dataflow_cache",
        lang: str = "en",
        edu_scorer: FineWebEduScorer | None = None,
        pq_scorer: PairQualScorer | None = None,
        scores_cache_path: str | None = None,
        batch_size: int = 256,
    ) -> None:
        if strategy not in ("fineweb_edu", "pairqual", "composite"):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.k = k
        self.strategy = strategy
        self.text_key = text_key
        self.batch_size = batch_size
        self.scores_cache_path = scores_cache_path

        if strategy in ("fineweb_edu", "composite"):
            self.edu_scorer: FineWebEduScorer | None = edu_scorer or FineWebEduScorer(
                model_cache_dir=model_cache_dir, device=device
            )
        else:
            self.edu_scorer = None

        if strategy in ("pairqual", "composite"):
            self.pq_scorer: PairQualScorer | None = pq_scorer or PairQualScorer(
                model_cache_dir=model_cache_dir, device=device, lang=lang
            )
        else:
            self.pq_scorer = None

    def select(self, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if self.k <= 0 or not samples:
            return []

        # Load or create cache
        cache = ScoreCache(self.scores_cache_path) if self.scores_cache_path else None
        missing = (
            cache.missing_indices(len(samples)) if cache else list(range(len(samples)))
        )

        if missing:
            print(
                f"[QualityScorerSelector] Scoring {len(missing)} samples (cached: {len(samples) - len(missing)})"
            )

            for batch_start in range(0, len(missing), self.batch_size):
                batch_indices = missing[batch_start : batch_start + self.batch_size]
                batch_samples = [samples[i] for i in batch_indices]

                df = pd.DataFrame(batch_samples)
                if self.text_key not in df.columns:
                    df[self.text_key] = [extract_text(s) for s in batch_samples]

                edu_scores: list[float] = []
                pq_scores: list[float] = []

                if self.edu_scorer is not None:
                    edu_scores = self.edu_scorer.eval(
                        df, input_key=self.text_key
                    ).tolist()
                if self.pq_scorer is not None:
                    pq_scores = self.pq_scorer.eval(
                        df, input_key=self.text_key
                    ).tolist()

                entries: list[tuple[int, dict[str, Any]]] = []
                for j, idx in enumerate(batch_indices):
                    scores: dict[str, Any] = {}
                    if edu_scores:
                        scores["fineweb_edu"] = float(edu_scores[j])
                    if pq_scores:
                        scores["pairqual"] = float(pq_scores[j])
                    entries.append((idx, scores))

                if cache:
                    cache.put_batch(entries)
                else:
                    if not hasattr(self, "_temp_scores"):
                        self._temp_scores: dict[int, dict[str, Any]] = {}
                    for idx, scores in entries:
                        self._temp_scores[idx] = scores

        # Collect all scores
        all_scores = cache.all_scores() if cache else getattr(self, "_temp_scores", {})

        scored: list[tuple[float, float | None, float | None, Mapping[str, Any]]] = []
        for i, s in enumerate(samples):
            sc = all_scores.get(i)
            if sc is None:
                continue
            e = sc.get("fineweb_edu")
            p = sc.get("pairqual")
            if self.strategy == "fineweb_edu":
                score = e or 0.0
            elif self.strategy == "pairqual":
                score = p or 0.0
            else:
                score = ((e or 0.0) + (p or 0.0)) / 2.0
            scored.append((score, e, p, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                **s,
                "meta": {
                    "selector": "QualityScorerSelector",
                    "strategy": self.strategy,
                    "fineweb_edu_score": round(e, 6) if e is not None else None,
                    "pairqual_score": round(p, 6) if p is not None else None,
                },
            }
            for score, e, p, s in scored[: min(self.k, len(scored))]
        ]
