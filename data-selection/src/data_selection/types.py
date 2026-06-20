from __future__ import annotations

from typing import NotRequired, TypedDict


class Meta(TypedDict, total=False):
    selector: str
    seed: NotRequired[int | None]
    source: NotRequired[str]
    length: NotRequired[int]
    ppl: NotRequired[float]
    strategy: NotRequired[str]
    similarity: NotRequired[float]
    quality_score: NotRequired[float]
    complexity_score: NotRequired[float]
    composite_score: NotRequired[float]
    fineweb_edu_score: NotRequired[float | None]
    pairqual_score: NotRequired[float | None]
    min_distance: NotRequired[float]
    dimension_scores: NotRequired[list[float] | None]
    average_score: NotRequired[float]
