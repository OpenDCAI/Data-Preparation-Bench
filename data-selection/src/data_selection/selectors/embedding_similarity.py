from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from dataflex.offline_selector.offline_near_selector import offline_near_Selector

from data_selection.utils import extract_text


class EmbeddingSimilaritySelector:
    """Near algorithm (DataFlex): select samples most similar to a target set.

    The "target set" (a.k.a. NEAR query) defines the direction we want the
    model to be good at. Candidates are ranked by cosine similarity to the
    centroid of the target-set embeddings, so *all* target samples contribute.

    Target set is resolved in priority order:
      1. ``query_path``: a JSONL file whose ``query_key`` field holds one
         representative target sample per line (recommended, matches NEAR's
         original design of using a validation/target split).
      2. ``domain_proxy_text``: a single proxy sentence (legacy fallback).
      3. otherwise: the candidates themselves (self-similarity).

    Embeddings come from DataFlex's offline_near_Selector backend
    (vLLM / sentence-transformers, already L2-normalized).
    """

    def __init__(
        self,
        k: int = 100,
        domain_proxy_text: str | None = None,
        query_path: str | None = None,
        query_key: str = "query",
        embed_model: str = "Qwen/Qwen3-Embedding-8B",
        embed_method: str = "auto",
        batch_size: int = 32,
        max_chars: int = 2000,
    ) -> None:
        self.k = k
        self.domain_proxy_text = domain_proxy_text
        self.query_path = query_path
        self.query_key = query_key
        self.embed_model = embed_model
        self.embed_method = embed_method
        self.batch_size = batch_size
        # Cap per-sample text length to bound embedding sequence length.
        # SentenceTransformer pads each batch to its longest sequence and
        # attention is O(L^2); a few very long samples (e.g. WizardLM ~12k
        # chars) blow up GPU memory. Truncating keeps L bounded.
        self.max_chars = max_chars

    def _load_query_texts(self) -> list[str]:
        """Read target-set texts from the query JSONL (``query_key`` field)."""
        if not self.query_path:
            return []
        texts: list[str] = []
        with open(self.query_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                q = d.get(self.query_key, "")
                if isinstance(q, str) and q.strip():
                    texts.append(q.strip())
        return texts

    def select(self, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if self.k <= 0 or not samples:
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_path = os.path.join(tmpdir, "candidates.json")
            query_path = os.path.join(tmpdir, "query.json")

            _write_alpaca_json(samples, candidate_path, self.max_chars)

            # Resolve the target set (query).
            query_texts = self._load_query_texts()
            if query_texts:
                _write_alpaca_json(
                    [{"instruction": q, "output": ""} for q in query_texts],
                    query_path,
                    self.max_chars,
                )
            elif self.domain_proxy_text:
                _write_alpaca_json(
                    [{"instruction": self.domain_proxy_text, "output": ""}],
                    query_path,
                    self.max_chars,
                )
            else:
                query_path = candidate_path  # self-similarity fallback

            near = offline_near_Selector(
                candidate_path=candidate_path,
                query_path=query_path,
                embed_model=self.embed_model,
                embed_method=self.embed_method,
                batch_size=self.batch_size,
                save_indices_path=os.path.join(tmpdir, "top_indices.npy"),
                max_K=max(self.k, 64),
            )

            # Pull normalized embeddings directly and aggregate over ALL
            # target samples via cosine similarity to their centroid.
            xb = np.asarray(
                near.candidate_sentence_embedding(), dtype=np.float32
            )  # (N, D)
            xq = np.asarray(near.query_sentence_embedding(), dtype=np.float32)  # (M, D)

            centroid = xq.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 0:
                centroid = centroid / norm
            scores = xb @ centroid  # cosine similarity (xb already normalized)
            order = np.argsort(-scores)[: min(self.k, len(samples))]

        result: list[dict[str, Any]] = []
        for rank, idx in enumerate(order):
            i = int(idx)
            result.append(
                {
                    **samples[i],
                    "meta": {
                        "selector": "EmbeddingSimilaritySelector",
                        "similarity_score": round(float(scores[i]), 6),
                        "neighbor_rank": rank,
                    },
                }
            )
        return result


def _write_alpaca_json(
    samples: Sequence[Mapping[str, Any]], path: str, max_chars: int = 2000
) -> None:
    items: list[dict[str, str]] = []
    for s in samples:
        text = extract_text(s)
        if not text:
            # Fallback for raw {"instruction": ...} proxy dicts without messages.
            text = str(s.get("instruction", ""))
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
        items.append({"instruction": text, "input": "", "output": ""})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
