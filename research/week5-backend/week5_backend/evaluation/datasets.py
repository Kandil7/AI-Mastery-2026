from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class EvalExample:
    question: str
    answer: str


def load_dataset(dataset_id: str) -> List[EvalExample]:
    data_path = Path(__file__).parent / f"{dataset_id}.jsonl"
    if not data_path.exists():
        return []
    examples: List[EvalExample] = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        examples.append(EvalExample(question=record["question"], answer=record["answer"]))
    return examples
