# Python CLI Testing and Argument Parsing Patterns

## Introduction

This guide covers testing CLI tools and structuring argument parsing so it is easy to test. It emphasizes separating parsing from execution logic.

## Learning Objectives

By the end of this guide, you will be able to:
- Structure CLI code for testability
- Parse arguments into config objects
- Test CLI logic without invoking subprocesses

---

## 1. Separate Parsing from Logic

```python
import argparse
from dataclasses import dataclass

@dataclass
class Config:
    name: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    return parser


def parse_args(argv: list[str]) -> Config:
    parser = build_parser()
    args = parser.parse_args(argv)
    return Config(name=args.name)
```

---

## 2. Test Without Subprocess

```python

def test_parse_args() -> None:
    cfg = parse_args(["Alice"])
    assert cfg.name == "Alice"
```

---

## 3. Keep Main Thin

```python

def main(argv: list[str]) -> int:
    cfg = parse_args(argv)
    print(f"Hello {cfg.name}")
    return 0
```

---

## Summary

CLI tools are easy to test when parsing and logic are separated. Avoid complex parsing in your core business logic.

---

## Additional Resources

- argparse: https://docs.python.org/3/library/argparse.html
