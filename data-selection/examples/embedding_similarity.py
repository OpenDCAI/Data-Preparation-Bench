from __future__ import annotations

from data_selection import EmbeddingSimilaritySelector
from data_selection.runner import run_selection

if __name__ == "__main__":
    selector = EmbeddingSimilaritySelector(
        k=100000,
        domain_proxy_text="Solve the following math problem step by step.",
        embed_model="/jizhicfs/linyibo/models/Qwen/Qwen3-Embedding-0.6B",
        embed_method="auto",
        batch_size=64,
    )
    run_selection(
        selector,
        input_path="/jizhicfs/linyibo/datasets/dyyyyyyyy/ScaleQuest-Math/scalequest_math.jsonl",
        output_path="data/output_embedding_similarity.jsonl",
    )
