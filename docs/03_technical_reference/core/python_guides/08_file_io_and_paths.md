# Python File I/O and Paths

## Introduction

This guide covers reading and writing files safely, working with paths, and common file I/O patterns. These skills are critical for document ingestion, storage, and processing workflows.

## Learning Objectives

By the end of this guide, you will be able to:
- Read and write text and binary files safely
- Use pathlib for cross-platform paths
- Stream large files without loading everything into memory
- Handle file errors gracefully
- Apply safe file I/O patterns in services

---

## 1. Path Handling with pathlib

`pathlib` provides object-oriented path manipulation and is cross-platform.

```python
from pathlib import Path

root = Path("data")
file_path = root / "documents" / "sample.txt"

print(file_path)
print(file_path.suffix)
```

---

## 2. Reading and Writing Text Files

```python
from pathlib import Path

path = Path("notes.txt")
path.write_text("Hello, world!", encoding="utf-8")

content = path.read_text(encoding="utf-8")
print(content)
```

---

## 3. Reading and Writing Binary Files

```python
from pathlib import Path

path = Path("data.bin")
path.write_bytes(b"\x00\x01\x02")

raw = path.read_bytes()
print(raw)
```

---

## 4. Streaming Large Files

Use streaming to avoid large memory usage.

```python
from pathlib import Path
from typing import Iterator


def stream_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield line
```

---

## 5. Handling Errors

```python
from pathlib import Path

try:
    content = Path("missing.txt").read_text(encoding="utf-8")
except FileNotFoundError:
    content = ""
```

---

## 6. Safe File Writing Patterns

Write to a temp file and then move it into place to avoid partial writes.

```python
from pathlib import Path
import tempfile


def safe_write(path: Path, data: str) -> None:
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
        tmp.write(data)
        temp_path = Path(tmp.name)
    temp_path.replace(path)
```

---

## 7. Checklist

Before you commit:
- Use pathlib for new file paths
- Stream large files instead of loading into memory
- Handle missing files explicitly
- Avoid partial writes with temp files

---

## Summary

Safe file I/O is essential for ingestion and storage workflows. Use pathlib, stream data when needed, and protect against partial writes.

---

## Additional Resources

- pathlib: https://docs.python.org/3/library/pathlib.html
- tempfile: https://docs.python.org/3/library/tempfile.html
