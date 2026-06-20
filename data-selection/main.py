from __future__ import annotations

from data_selection import RandomSelector


def main() -> None:
    samples = [
        {"instruction": "task 1", "output": "result 1", "source": "a"},
        {"instruction": "task 2", "output": "result 2", "source": "b"},
        {"instruction": "task 3", "output": "result 3", "source": "a"},
        {"instruction": "task 4", "output": "result 4", "source": "b"},
        {"instruction": "task 5", "output": "result 5", "source": "c"},
    ]
    selector = RandomSelector(k=3, seed=42)
    selected = selector.select(samples)
    for s in selected:
        print(s["instruction"])


if __name__ == "__main__":
    main()
