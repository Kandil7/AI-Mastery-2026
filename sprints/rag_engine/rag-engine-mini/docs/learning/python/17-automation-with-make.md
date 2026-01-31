# Python Automation with Make and Scripts

## Introduction

This guide explains how to automate common development tasks with Make and scripts. It shows how to keep workflows repeatable for testing, formatting, and running services.

## Learning Objectives

By the end of this guide, you will be able to:
- Understand why automation matters
- Use Makefile targets effectively
- Create small scripts for repeatable workflows
- Avoid duplication in development tasks

---

## 1. Why Use Make

Make provides a simple way to standardize commands for the team.

Example targets used in this repo:

```bash
make install
make test
make lint
make format
make run
```

---

## 2. Simple Makefile Target

```make
.PHONY: test

test:
	pytest tests/ -v
```

---

## 3. Scripts for Repeatability

Small scripts can wrap complex commands.

```bash
#!/usr/bin/env bash
set -e
make format
make test
```

---

## 4. Best Practices

- Keep targets small and focused
- Use `.PHONY` for non-file targets
- Document targets in README

---

## Summary

Automation makes workflows consistent. Use Make targets and small scripts to prevent errors and speed up development.

---

## Additional Resources

- GNU Make: https://www.gnu.org/software/make/
