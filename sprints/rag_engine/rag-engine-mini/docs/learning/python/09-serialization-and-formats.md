# Python Data Serialization (JSON, CSV, and Schemas)

## Introduction

This guide explains common data serialization formats used in APIs and data pipelines: JSON and CSV. It also covers lightweight schema validation patterns that align with the project.

## Learning Objectives

By the end of this guide, you will be able to:
- Read and write JSON safely
- Work with CSV files
- Validate and normalize data structures
- Choose the right format for a use case
- Avoid common serialization pitfalls

---

## 1. JSON Basics

JSON is the standard format for API payloads.

```python
import json

payload = {"id": "1", "title": "Doc", "tags": ["rag", "ml"]}
text = json.dumps(payload)
restored = json.loads(text)

print(text)
print(restored["title"])
```

---

## 2. Pretty Printing

```python
print(json.dumps(payload, indent=2, sort_keys=True))
```

---

## 3. JSON Files

```python
from pathlib import Path
import json

path = Path("document.json")
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

loaded = json.loads(path.read_text(encoding="utf-8"))
print(loaded)
```

---

## 4. CSV Basics

CSV is useful for tabular exports.

```python
import csv
from pathlib import Path

rows = [
    {"id": "1", "title": "Doc 1"},
    {"id": "2", "title": "Doc 2"},
]

path = Path("docs.csv")
with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["id", "title"])
    writer.writeheader()
    writer.writerows(rows)
```

---

## 5. Schema-Like Validation

Use simple checks or Pydantic at the boundaries.

```python
from typing import Any


def validate_document(data: dict[str, Any]) -> None:
    if "id" not in data or "title" not in data:
        raise ValueError("Missing required fields")
```

---

## 6. Choosing Formats

- JSON: nested data, APIs
- CSV: simple tables, analytics exports

---

## 7. Checklist

Before you commit:
- JSON encoding uses UTF-8
- CSV writes headers and stable field order
- Validation happens at boundaries

---

## Summary

Serialization is about reliable interchange. Use JSON for APIs, CSV for tabular exports, and validate data when it crosses boundaries.

---

## Additional Resources

- json: https://docs.python.org/3/library/json.html
- csv: https://docs.python.org/3/library/csv.html
