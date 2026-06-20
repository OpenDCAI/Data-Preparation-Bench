# Data Selection

A unified, downstream-grounded data selection framework for LLM supervised fine-tuning. It exposes a single `Selector` protocol and a collection of ready-to-use strategies, so you can switch between random baselines, perplexity filters, embedding-based selectors, quality scorers, diversity algorithms, and LLM-as-a-judge methods without changing the surrounding pipeline.

The framework is designed around two principles:

1. **One interface for every strategy** — every selector implements `select(samples) -> list[dict]` with parameters configured at construction time.
2. **Efficient multi-`k` experiments** — score-based selectors can be scored once at `max(k)` and truncated for smaller budgets, and expensive GPU/API scorers support resumable score caching.

## Installation

This subproject uses [uv](https://docs.astral.sh/uv/) and requires Python 3.12.

```bash
cd data-selection
uv sync
```

To install development tools (pre-commit, pyright, codespell):

```bash
uv sync --group dev
```

Add new dependencies with `uv add <package-name>` instead of editing `pyproject.toml` manually.

## Selector Overview

| Selector | File | Key Dependency | Description |
|----------|------|----------------|-------------|
| `RandomSelector` | `random_selection.py` | — | Random baseline with optional seed. |
| `SourceBalancedRandomSelector` | `source_balanced_random.py` | — | Random selection balanced by the `source` field. |
| `LengthBasedSelector` | `length_based.py` | — | Select shortest or longest samples. |
| `PerplexityBasedSelector` | `perplexity_based.py` | `dataflow PerplexityScorer` | Select by perplexity (`low`, `high`, or `mid`). |
| `EmbeddingSimilaritySelector` | `embedding_similarity.py` | `dataflex offline_near_Selector` | NEAR-style similarity to a target set or domain proxy. |
| `DeitaQualitySelector` | `deita_quality.py` | `dataflow DeitaQuality/ComplexityScorer` | Quality × complexity product. |
| `QualityScorerSelector` | `quality_scorer.py` | `dataflow FineWebEduScorer/PairQualScorer` | FineWeb-Edu / PairQual / composite quality scoring. |
| `DiversityKCenterSelector` | `diversity_kcenter.py` | `dataflex offline_tsds_Selector` | TSDS diversity K-Center selection. |
| `LLMAsSelector` | `llm_selector.py` | `dataflow MetaScorer` | Multi-dimensional LLM-as-a-judge scoring. |
| `CompositeSelector` | `composite.py` | — | Chain multiple selectors sequentially. |

## Input Format

The framework normalizes each input sample to a shared `messages` representation:

```python
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Two raw formats are supported out of the box:

- **Alpaca-style**: `{"instruction": "...", "output": "..."}`
- **Conversations-style**: `{"conversations": [{"role": "...", "content": "..."}]}` (also accepts `from`/`value` and nested `messages` arrays)

You can configure the keys passed to `run_selection` via `instruction_key`, `output_key`, and `conversations_key`.

## Quick Start

### Programmatic API

```python
from data_selection import RandomSelector

samples = [
    {"instruction": "task 1", "output": "result 1", "source": "a"},
    {"instruction": "task 2", "output": "result 2", "source": "b"},
    {"instruction": "task 3", "output": "result 3", "source": "a"},
]

selector = RandomSelector(k=2, seed=42)
selected = selector.select(samples)
```

### JSONL Pipeline

```python
from data_selection import RandomSelector
from data_selection.runner import run_selection

selector = RandomSelector(k=100, seed=42)
run_selection(
    selector,
    input_path="data/input.jsonl",
    output_path="data/output_random.jsonl",
)
```

`run_selection` also accepts a list of `k` values. For score-based selectors, scoring runs once at `max(k)` and results are truncated for each smaller budget:

```python
run_selection(
    selector,
    input_path="data/input.jsonl",
    output_path="data/output_k{k}.jsonl",
    k=[1000, 10000, 100000],
)
```

## Selector Details

### Random and Baseline Selectors

- `RandomSelector(k, seed=None)` — uniform random sample.
- `SourceBalancedRandomSelector(k, source_key="source", seed=None)` — samples evenly across unique `source` values.
- `LengthBasedSelector(k, strategy="shortest")` — `shortest` or `longest` by extracted text length.

### Perplexity-Based Selection

```python
from data_selection import PerplexityBasedSelector

selector = PerplexityBasedSelector(
    k=1000,
    strategy="low",  # "low", "high", or "mid"
    text_key="text",
    lang="en",
    model_name="dataflow/operators/eval/GeneralText/models/Kenlm/wikipedia",
)
```

Set `scores_cache_path="scores.jsonl"` to cache expensive perplexity scores and resume interrupted runs.

### Embedding Similarity (NEAR)

```python
from data_selection import EmbeddingSimilaritySelector

selector = EmbeddingSimilaritySelector(
    k=100000,
    domain_proxy_text="Solve the following math problem step by step.",
    embed_model="Qwen/Qwen3-Embedding-8B",
    embed_method="auto",
    batch_size=64,
)
```

You can also point `query_path` to a JSONL file containing target samples; the selector ranks candidates by cosine similarity to the centroid of the target embeddings.

### Quality Scorers

```python
from dataflow.operators.eval import DeitaQualityScorer, DeitaComplexityScorer
from data_selection import DeitaQualitySelector

selector = DeitaQualitySelector(
    k=100,
    quality_scorer=DeitaQualityScorer(device="cuda"),
    complexity_scorer=DeitaComplexityScorer(device="cuda"),
    scores_cache_path="deita_scores.jsonl",
)
```

For FineWeb-Edu / PairQual scoring, use `QualityScorerSelector`.

### Diversity K-Center (TSDS)

```python
from data_selection import DiversityKCenterSelector

selector = DiversityKCenterSelector(
    k=100000,
    embed_model="Qwen/Qwen3-Embedding-0.6B",
    embed_method="auto",
    batch_size=32,
    sigma=0.75,
    alpha=0.6,
)
```

### LLM-as-a-Judge

```python
from dataflow.operators.eval import MetaScorer
from dataflow.serving.APILLMServing_request import APILLMServing_request
from data_selection import LLMAsSelector

llm_serving = APILLMServing_request(
    api_url="https://api.openai.com/v1/chat/completions",
    key_name_of_api_key="DF_API_KEY",
    model_name="gpt-4o",
    max_workers=10,
)
scorer = MetaScorer(llm_serving=llm_serving, dimensions=[...])
selector = LLMAsSelector(k=1000, text_key="text", scorer=scorer)
```

### Composite Selection

```python
from data_selection import CompositeSelector, RandomSelector, LengthBasedSelector

selector = CompositeSelector([
    LengthBasedSelector(k=10000, strategy="longest"),
    RandomSelector(k=1000, seed=42),
])
```

The composite runs each selector in order, feeding the output of one into the next.

## Score Caching and Multi-`k` Optimization

- Score-based selectors set `_score_based = True`. When `run_selection` receives multiple `k` values, it runs the selector once at `max(k)` and truncates the ranked list for each smaller `k`.
- Set `scores_cache_path` on `PerplexityBasedSelector`, `DeitaQualitySelector`, `QualityScorerSelector`, or `LLMAsSelector` to write per-sample scores to JSONL. On the next run, cached samples are skipped.

## Development

### Run the minimal example

```bash
uv run python main.py
```

### Run an example

```bash
uv run python examples/random.py
```

### Type check

```bash
uv run --with pyright pyright
```

### Pre-commit

```bash
uv run pre-commit run --all-files
```

Pre-commit hooks include `isort`, `black-jupyter`, `pyupgrade --py312-plus`, `autoflake`, `bandit`, `pyright`, and `codespell`. Some hooks rewrite files; if a commit fails, re-stage and retry.

## Adding a New Selector

1. Create a module under `src/data_selection/selectors/`.
2. Implement `select(self, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]`.
3. Configure all strategy parameters (including `k`) in `__init__`.
4. If the selector sorts by a computed score, set `_score_based = True`.
5. Export the selector in `src/data_selection/selectors/__init__.py` and `src/data_selection/__init__.py`.
6. Add an example under `examples/` (optional but recommended).

