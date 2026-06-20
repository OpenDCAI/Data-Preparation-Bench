"""Score caching utilities for resumable selection.

Provides a simple JSONL-based cache for per-sample scores, enabling
expensive scoring operations (GPU models, LLM APIs) to resume from
where they left off after interruption.

Cache format (JSONL):
    {"idx": 0, "scores": {"quality": 3.2, "complexity": 2.1}}
    {"idx": 1, "scores": {"quality": 4.5, "complexity": 3.8}}
    ...
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class ScoreCache:
    """Append-only JSONL cache for per-sample scores."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._cache: dict[int, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load existing cache from disk."""
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self._cache[entry["idx"]] = entry["scores"]

    def get(self, idx: int) -> dict[str, Any] | None:
        """Get cached scores for a sample index, or None if not cached."""
        return self._cache.get(idx)

    def has(self, idx: int) -> bool:
        """Check if scores exist for a sample index."""
        return idx in self._cache

    def put(self, idx: int, scores: dict[str, Any]) -> None:
        """Add scores for a sample and immediately persist to disk."""
        self._cache[idx] = scores
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps({"idx": idx, "scores": scores}, ensure_ascii=False) + "\n"
            )

    def put_batch(self, entries: list[tuple[int, dict[str, Any]]]) -> None:
        """Add multiple score entries and persist in one write."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            for idx, scores in entries:
                self._cache[idx] = scores
                f.write(
                    json.dumps({"idx": idx, "scores": scores}, ensure_ascii=False)
                    + "\n"
                )

    def cached_count(self) -> int:
        """Number of cached entries."""
        return len(self._cache)

    def missing_indices(self, total: int) -> list[int]:
        """Return list of indices in [0, total) that are not yet cached."""
        return [i for i in range(total) if i not in self._cache]

    def all_scores(self) -> dict[int, dict[str, Any]]:
        """Return all cached scores."""
        return dict(self._cache)
