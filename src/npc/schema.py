"""Dataset record types and JSONL helpers.

Training records are stored as chat-message lists so they map directly onto the
model's chat template. Each record also carries lightweight metadata used by the
filter/eval stages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Iterable, Iterator, Optional

# Prompt categories drive the data mix and the eval breakdown.
CATEGORIES = [
    "lesson",         # curriculum question / numeric problem -> correct teaching
    "student_error",  # wrong work or misconception -> precise correction
    "fourth_wall",    # "you are an AI" / "drop the act" -> holds character, keeps teaching
    "out_of_world",   # real-world / off-subject asks -> oblique deflection back to lesson
    "edge",           # frustration / gibberish / "just give me the answer"
]


@dataclass
class Message:
    role: str          # "system" | "user" | "assistant"
    content: str


@dataclass
class Record:
    """A single supervised example (one student turn -> one Sable turn).

    We keep single-turn for clarity and eval determinism, but the schema supports
    multi-turn by allowing multiple messages.
    """

    messages: list[Message]
    category: str = "lesson"
    source: str = "teacher"      # teacher | seed | manual
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "messages": [asdict(m) for m in self.messages],
            "category": self.category,
            "source": self.source,
            "meta": self.meta,
        }

    @staticmethod
    def from_json(d: dict) -> "Record":
        return Record(
            messages=[Message(**m) for m in d["messages"]],
            category=d.get("category", "lesson"),
            source=d.get("source", "teacher"),
            meta=d.get("meta", {}),
        )

    def user_text(self) -> str:
        for m in self.messages:
            if m.role == "user":
                return m.content
        return ""

    def assistant_text(self) -> str:
        for m in reversed(self.messages):
            if m.role == "assistant":
                return m.content
        return ""


def write_jsonl(path: str, records: Iterable[Record]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> Iterator[Record]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield Record.from_json(json.loads(line))


def read_prompts_jsonl(path: str) -> Iterator[dict]:
    """Read a seed/eval prompt file: {"prompt": str, "category": str, "meta": {...}}."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
