# Python Linting, Formatting, and Type Checking

## Introduction

This guide explains code quality tools commonly used in Python projects: formatting with Black, sorting imports with isort, linting with flake8, and type checking with mypy.

## Learning Objectives

By the end of this guide, you will be able to:
- Understand the role of each quality tool
- Run formatting and linting checks
- Interpret common linter/type errors
- Keep code style consistent across the repo

---

## 1. Formatting with Black

Black enforces a consistent style.

```bash
black --check .
black .
```

---

## 2. Sorting Imports with isort

```bash
isort --check-only .
isort .
```

---

## 3. Linting with flake8

```bash
flake8 src tests
```

Common issues:
- Unused imports
- Line length violations
- Undefined names

---

## 4. Type Checking with mypy

```bash
mypy src
```

Common issues:
- Missing type hints
- Incompatible return types

---

## 5. Makefile Commands

In this repo:

```bash
make lint
make format
```

---

## Summary

Use formatting and linting tools to keep code consistent and safe. Run `make lint` before commits and `make format` when fixing style.

---

## Additional Resources

- Black: https://black.readthedocs.io/
- isort: https://pycqa.github.io/isort/
- flake8: https://flake8.pycqa.org/
- mypy: https://mypy.readthedocs.io/
