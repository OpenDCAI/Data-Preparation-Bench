from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DatasetConfig:
    path: str
    output: str = "data/output.jsonl"
    instruction_key: str = "instruction"
    output_key: str = "output"
    conversations_key: str = "conversations"

    def format(self, sample: dict[str, Any]) -> dict[str, Any]:
        messages: list[dict[str, str]] = []

        if self.conversations_key in sample:
            items = sample[self.conversations_key]
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    if "role" in item and "content" in item:
                        messages.append(
                            {"role": str(item["role"]), "content": str(item["content"])}
                        )
                    elif "from" in item and "value" in item:
                        role = _role_map.get(str(item["from"]), str(item["from"]))
                        messages.append({"role": role, "content": str(item["value"])})
                    elif "messages" in item:
                        for msg in item["messages"]:
                            messages.append(
                                {
                                    "role": str(msg.get("role", "user")),
                                    "content": str(msg.get("content", "")),
                                }
                            )
        else:
            messages.append(
                {"role": "user", "content": str(sample.get(self.instruction_key, ""))},
            )
            if sample.get(self.output_key):
                messages.append(
                    {"role": "assistant", "content": str(sample[self.output_key])}
                )

        return {**sample, "messages": messages}


_role_map = {"human": "user", "gpt": "assistant", "system": "system"}
