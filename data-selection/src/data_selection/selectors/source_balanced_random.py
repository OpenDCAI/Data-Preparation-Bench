from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


class SourceBalancedRandomSelector:
    """Sample proportionally from each source dataset to avoid domination."""

    def __init__(
        self, k: int = 100, source_key: str = "source", seed: int | None = None
    ) -> None:
        self.k = k
        self.source_key = source_key
        self.seed = seed

    def select(self, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if self.k <= 0 or not samples:
            return []
        rng = random.Random(self.seed)
        by_source: dict[str, list[dict]] = defaultdict(list)
        for s in samples:
            by_source[s.get(self.source_key, "__unknown__")].append(s)

        n_sources = len(by_source)
        per_source = self.k // n_sources
        remainder = self.k % n_sources

        selected: list[dict[str, Any]] = []
        sources = list(by_source.keys())
        rng.shuffle(sources)
        for i, src in enumerate(sources):
            pool = by_source[src]
            take = per_source + (1 if i < remainder else 0)
            take = min(take, len(pool))
            if take > 0:
                for s in rng.sample(pool, take):
                    selected.append(
                        {
                            **s,
                            "meta": {
                                "selector": "SourceBalancedRandomSelector",
                                "source": src,
                            },
                        }
                    )

        return selected
