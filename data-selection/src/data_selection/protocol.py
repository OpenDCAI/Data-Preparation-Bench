from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class Selector(Protocol):
    """Protocol for data selectors.

    All strategy-specific parameters (including k) are configured
    at __init__ time. Any class with a compatible `select` method
    satisfies this protocol.
    """

    def select(self, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Select samples from the candidate pool.

        Args:
            samples: Candidate samples as a list of dicts.

        Returns:
            The selected subset, each with a "meta" dict attached.
        """
        ...
