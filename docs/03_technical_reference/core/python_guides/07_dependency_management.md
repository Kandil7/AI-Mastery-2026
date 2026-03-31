# Python Dependency Management

## Introduction

This guide covers dependency management in Python projects: requirements files, virtual environments, version pinning, and reproducible builds. It maps directly to how this repo manages dependencies.

## Learning Objectives

By the end of this guide, you will be able to:
- Use virtual environments effectively
- Pin dependencies for reproducibility
- Understand direct vs transitive dependencies
- Update dependencies safely
- Avoid common dependency pitfalls

---

## 1. Virtual Environments

Use a virtual environment to isolate project dependencies.

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
```

---

## 2. Requirements Files

Use `requirements.txt` to list dependencies.

```
fastapi==0.111.0
pydantic==2.8.2
celery==5.4.0
```

Pinning exact versions improves reproducibility.

---

## 3. Direct vs Transitive Dependencies

- **Direct**: you explicitly list in `requirements.txt`
- **Transitive**: pulled in automatically by direct deps

Only direct dependencies should be in `requirements.txt`.

---

## 4. Updating Dependencies Safely

Guidelines:
- Update in small batches
- Run tests after each update
- Watch for breaking changes

Example process:

```bash
pip install -U fastapi
pip freeze > requirements.txt
make test
```

---

## 5. Reproducible Builds

Use a consistent lockfile strategy if needed, and ensure `requirements.txt` matches production.

---

## 6. Anti-Patterns

Avoid:
- Unpinned versions in production
- Mixing system Python with project deps
- Updating many dependencies at once without tests

---

## 7. Checklist

Before you commit:
- Dependencies are pinned
- Requirements are minimal and direct
- Tests run after updates

---

## Summary

Dependency management keeps environments stable and predictable. Use virtual environments, pin versions, and update carefully.

---

## Additional Resources

- pip: https://pip.pypa.io/en/stable/
- venv: https://docs.python.org/3/library/venv.html
