from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from dataflow.operators.eval import (
    DeitaComplexityScorer,
    DeitaQualityScorer,
)

from data_selection.score_cache import ScoreCache


class DeitaQualitySelector:
    """Select samples by Deita quality x complexity composite score."""

    _score_based = True

    def __init__(
        self,
        k: int = 100,
        device: str = "cuda",
        model_cache_dir: str = "./dataflow_cache",
        max_length: int = 8192,
        quality_scorer: DeitaQualityScorer | None = None,
        complexity_scorer: DeitaComplexityScorer | None = None,
        scores_cache_path: str | None = None,
        batch_size: int = 1000,
    ) -> None:
        self.k = k
        self.scores_cache_path = scores_cache_path
        self.batch_size = batch_size
        self.quality_scorer = quality_scorer or DeitaQualityScorer(
            device=device, model_cache_dir=model_cache_dir, max_length=max_length
        )
        self.complexity_scorer = complexity_scorer or DeitaComplexityScorer(
            device=device, model_cache_dir=model_cache_dir, max_length=max_length
        )

    def select(self, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if self.k <= 0 or not samples:
            return []

        # 可断点续跑的缓存: 每条样本缓存 {"quality": .., "complexity": ..}
        cache = ScoreCache(self.scores_cache_path) if self.scores_cache_path else None
        missing = (
            cache.missing_indices(len(samples)) if cache else list(range(len(samples)))
        )

        if missing:
            print(
                f"[DeitaQualitySelector] Scoring {len(missing)} samples "
                f"(cached: {len(samples) - len(missing)})"
            )

            for batch_start in range(0, len(missing), self.batch_size):
                batch_indices = missing[batch_start : batch_start + self.batch_size]
                batch_samples = [samples[i] for i in batch_indices]

                df = pd.DataFrame(batch_samples)
                df["instruction"] = [_msg_role(s, "user") for s in batch_samples]
                df["output"] = [_msg_role(s, "assistant") for s in batch_samples]

                q_scores = self.quality_scorer.eval(df)
                c_scores = self.complexity_scorer.eval(df)

                entries: list[tuple[int, dict[str, Any]]] = []
                for j, idx in enumerate(batch_indices):
                    entries.append(
                        (
                            idx,
                            {
                                "quality": float(q_scores[j]),
                                "complexity": float(c_scores[j]),
                            },
                        )
                    )

                # 每批立即落盘, 中断后可从此处继续
                if cache:
                    cache.put_batch(entries)
                else:
                    if not hasattr(self, "_temp_scores"):
                        self._temp_scores: dict[int, dict[str, Any]] = {}
                    for idx, sc in entries:
                        self._temp_scores[idx] = sc

        # 汇总所有分数, 计算 composite = quality * complexity 并排序
        all_scores = cache.all_scores() if cache else getattr(self, "_temp_scores", {})

        scored: list[tuple[float, float, float, Mapping[str, Any]]] = []
        for i, s in enumerate(samples):
            sc = all_scores.get(i)
            if sc is None:
                continue
            q = float(sc["quality"])
            c = float(sc["complexity"])
            composite = q * c
            scored.append((composite, q, c, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                **s,
                "meta": {
                    "selector": "DeitaQualitySelector",
                    "quality_score": round(q, 4),
                    "complexity_score": round(c, 4),
                    "composite_score": round(composite, 4),
                },
            }
            for composite, q, c, s in scored[: min(self.k, len(scored))]
        ]


def _msg_role(sample: Mapping[str, Any], role: str) -> str:
    for msg in sample.get("messages", []):
        if msg.get("role") == role:
            return str(msg.get("content", ""))
    return ""
