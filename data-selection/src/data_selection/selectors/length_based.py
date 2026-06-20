from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from data_selection.utils import extract_text


class LengthBasedSelector:
    """Select samples by instruction+output or conversations length."""

    def __init__(self, k: int = 100) -> None:
        self.k = k

    def select(self, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if self.k <= 0 or not samples:
            return []

        def _length(s: Mapping[str, Any]) -> int:
            return len(extract_text(s))

        scored = sorted(samples, key=_length, reverse=True)
        return [
            {**s, "meta": {"selector": "LengthBasedSelector", "length": _length(s)}}
            for s in scored[: min(self.k, len(scored))]
        ]
