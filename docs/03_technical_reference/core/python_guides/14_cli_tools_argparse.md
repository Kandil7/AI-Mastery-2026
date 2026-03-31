# Python CLI Tools with argparse

## Introduction

This guide shows how to build simple command-line tools using `argparse`. CLI utilities are useful for scripts, migrations, and batch workflows in this repo.

## Learning Objectives

By the end of this guide, you will be able to:
- Define CLI arguments and options
- Validate and parse input
- Provide helpful usage text
- Structure a simple CLI entry point

---

## 1. Basic CLI with argparse

```python
import argparse

parser = argparse.ArgumentParser(description="Example CLI")
parser.add_argument("name", help="Name to greet")
args = parser.parse_args()

print(f"Hello, {args.name}!")
```

---

## 2. Optional Flags

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=10)
args = parser.parse_args()

print(f"limit={args.limit}")
```

---

## 3. Subcommands

```python
import argparse

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="command", required=True)

hello = sub.add_parser("hello")
hello.add_argument("name")

args = parser.parse_args()
if args.command == "hello":
    print(f"Hello, {args.name}!")
```

---

## 4. Applying to This Repo

Use CLI scripts for:
- Data ingestion
- Batch exports
- Maintenance jobs

Keep logic in modules, and keep CLI thin.

---

## Summary

argparse makes small tools easy to build. Keep CLI parsing separate from business logic for testability.

---

## Additional Resources

- argparse: https://docs.python.org/3/library/argparse.html
