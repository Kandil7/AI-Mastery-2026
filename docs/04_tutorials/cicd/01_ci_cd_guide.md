# CI/CD & Automation: Complete Guide

## Table of Contents
1. [What is CI/CD?](#what-is-cicd)
2. [GitHub Actions](#github-actions)
3. [Pre-commit Hooks](#pre-commit-hooks)
4. [Docker Optimization](#docker-optimization)
5. [Security Scanning](#security-scanning)
6. [Release Automation](#release-automation)
7. [Deployment Strategies](#deployment-strategies)

---

## What is CI/CD?

### Definition

**CI/CD** = Continuous Integration + Continuous Deployment

- **Continuous Integration (CI)**: Automatically test code changes
- **Continuous Deployment (CD)**: Automatically deploy tested code

### Benefits

| Benefit | Description |
|---------|-------------|
| **Faster feedback** | Catch errors before merge |
| **Consistent environments** | Same config for dev/prod |
| **Automated testing** | Run tests on every push |
| **Zero-downtime deployment** | Deploy without stopping service |
| **Rollback capability** | Quick revert if issues |

### CI/CD Pipeline

```
Developer Push
    │
    ▼
┌────────────────────────────────────────────────────┐
│ CI Pipeline                                    │
│ ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│ │ Lint     │  │  Test     │  │ Security  │ │
│ │ (Black)  │  │ (Pytest)  │  │ (Trivy)  │ │
│ └──────────┘  └──────────┘  └──────────┘ │
└────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────┐
│ Build & Scan                                   │
│ ┌──────────┐  ┌──────────┐                │
│ │ Docker   │  │  Image    │                │
│ │ Build    │  │  Scan     │                │
│ └──────────┘  └──────────┘                │
└────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────┐
│ CD Pipeline                                    │
│ ┌──────────┐  ┌──────────┐                │
│ │ Deploy   │  │  Smoke    │                │
│ │ (SSH)    │  │  Test     │                │
│ └──────────┘  └──────────┘                │
└────────────────────────────────────────────────────┘
    │
    ▼
Production
```

### Arabic
**الدمج المستمر والنشر المستمر**: اختبار آلي للكود ونشر آلي للتطبيقات

---

## GitHub Actions

### Workflow Structure

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  workflow_dispatch:  # Manual trigger

jobs:
  lint:      # Code quality checks
  test:      # Unit and integration tests
  security:  # Vulnerability scanning
  build:     # Docker image build
  deploy:    # Deploy to environment
```

### Jobs in Sequence

```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
    steps: ...

  test:
    needs: lint  # Runs after lint passes
    runs-on: ubuntu-latest
    steps: ...

  build:
    needs: [lint, test]  # Runs after both pass
    runs-on: ubuntu-latest
    steps: ...

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    environment: production
    steps: ...
```

### Matrix Strategy

```yaml
test:
  runs-on: ubuntu-latest
  strategy:
    matrix:
      python-version: [3.9, 3.10, 3.11]
      database: [postgres:14, postgres:15]

  steps:
    - name: Setup Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Test with ${{ matrix.database }}
      run: pytest
```

### Caching

```yaml
- name: Setup Python with cache
  uses: actions/setup-python@v4
  with:
    python-version: "3.10"
    cache: "pip"  # Cache pip packages

- name: Cache Docker layers
  uses: actions/cache@v3
  with:
    path: /tmp/.buildx-cache
    key: ${{ runner.os }}-buildx-${{ github.sha }}
```

### Artifacts

```yaml
- name: Upload coverage reports
  if: always()  # Even if tests fail
  uses: actions/upload-artifact@v3
  with:
    name: coverage-reports
    path: |
      htmlcov/
      coverage.xml
    retention-days: 30
```

### Secrets Management

```yaml
env:
  DATABASE_URL: ${{ secrets.DATABASE_URL }}
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  SENTRY_DSN: ${{ secrets.SENTRY_DSN }}

steps:
  - name: Use secret
    run: echo "Key has ${{ secrets.OPENAI_API_KEY }} chars"
```

### Services

```yaml
test:
  services:
    postgres:
      image: postgres:15
      env:
        POSTGRES_USER: postgres
        POSTGRES_PASSWORD: postgres
      ports:
        - 5432:5432
      options: >-
        --health-cmd pg_isready
        --health-interval 10s

    redis:
      image: redis:7-alpine
      ports:
        - 6379:6379
```

---

## Pre-commit Hooks

### What are Pre-commit Hooks?

**Pre-commit hooks** run automatically before each commit to ensure code quality.

### Setup

```bash
# Install pre-commit
pip install pre-commit

# Initialize in repository
pre-commit install

# Run hooks on all files
pre-commit run --all-files
```

### Configuration (.pre-commit-config.yaml)

```yaml
repos:
  # Black formatting
  - repo: https://github.com/psf/black
    rev: 24.4.0
    hooks:
      - id: black
        args: [--line-length=100]

  # isort import sorting
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort

  # Flake8 linting
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100']

  # mypy type checking
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
```

### Common Hooks

| Hook | Purpose | Tool |
|-------|----------|-------|
| **trailing-whitespace** | Remove trailing whitespace | pre-commit |
| **end-of-file-fixer** | Add newline at end of file | pre-commit |
| **black** | Format Python code | Black |
| **isort** | Sort imports | isort |
| **flake8** | Lint Python | Flake8 |
| **mypy** | Type checking | mypy |
| **bandit** | Security scanning | Bandit |
| **prettier** | Format YAML/JSON/MD | Prettier |
| **shellcheck** | Lint shell scripts | ShellCheck |
| **hadolint** | Lint Dockerfile | Hadolint |
| **markdownlint** | Lint Markdown | markdownlint |
| **gitleaks** | Detect secrets | Gitleaks |

### Auto-fix Hooks

```yaml
- repo: https://github.com/psf/black
    rev: 24.4.0
    hooks:
      - id: black  # Auto-formats code
```

```bash
# Run with auto-fix
pre-commit run --all-files
```

---

## Docker Optimization

### Multi-stage Builds

```dockerfile
# Stage 1: Builder (contains build tools)
FROM python:3.10-slim as builder

RUN apt-get install gcc g++ make
COPY requirements.txt .
RUN pip install -r requirements.txt

# Stage 2: Runtime (minimal size)
FROM python:3.10-slim

# Copy only what's needed
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/app /opt/app
```

### Layer Caching

```yaml
# GitHub Actions
- name: Build with cache
  uses: docker/build-push-action@v5
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

### Image Size Optimization

```dockerfile
# Use alpine/slim images
FROM python:3.10-slim  # ✅ Smaller
# FROM python:3.10     # ❌ Larger

# Clean up after install
RUN apt-get update && \
    apt-get install -y gcc && \
    rm -rf /var/lib/apt/lists/*  # Clean cache

# Use .dockerignore
node_modules/
.git/
__pycache__/
*.pyc
```

### Production Server

```dockerfile
# Don't use development server
# CMD ["python", "-m", "uvicorn", "src.production.api:app"]  # ❌

# Use production WSGI server
CMD ["gunicorn", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "src.production.api:app"]  # ✅
```

---

## Security Scanning

### Static Application Security Testing (SAST)

```yaml
# GitHub Actions: Bandit
- name: Run Bandit
  run: bandit -r src/ -f json -o bandit-report.json

# GitHub Actions: Trivy
- name: Run Trivy
  uses: aquasecurity/trivy-action@master
  with:
    scan-type: "fs"
    format: "sarif"
```

### Dependency Scanning (SCA)

```yaml
# GitHub Actions: Snyk
- name: Run Snyk
  uses: snyk/actions/python@master
  env:
    SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
  with:
    args: --severity-threshold=high
```

### Container Scanning

```yaml
# Scan built image
- name: Scan image
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: ghcr.io/user/repo:latest
    format: "sarif"
```

### Secret Detection

```yaml
# Gitleaks in pre-commit
- repo: https://github.com/zricethezav/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks
        args: ['--no-banner', '--no-git']
```

### Vulnerability Types

| Type | Description | Tool |
|-------|-------------|-------|
| **SAST** | Static code analysis | Bandit, Semgrep |
| **SCA** | Dependency vulnerabilities | Snyk, Dependabot |
| **Container** | Image vulnerabilities | Trivy, Grype |
| **Secrets** | Hardcoded credentials | Gitleaks, git-secrets |

---

## Release Automation

### Semantic Versioning

```
Major.Minor.Patch
  │    │     │
  │    │     └─ Bug fixes (0.0.1 → 0.0.2)
  │    └─ New features (0.0.1 → 0.1.0)
  └─ Breaking changes (0.1.0 → 1.0.0)
```

### Automated Release Workflow

```yaml
release:
  on:
    push:
      tags: ['v*']

  jobs:
    create-release:
      steps:
        - name: Create GitHub Release
          uses: actions/create-release@v1
          with:
            tag_name: ${{ github.ref_name }}
            release_name: Release ${{ github.ref_name }}
            body: |
              ## What's Changed
              - Feature A
              - Bug fix B
            draft: false
            prerelease: false

        - name: Build and push Docker
          uses: docker/build-push-action@v5
          with:
            tags: |
              ghcr.io/user/repo:latest
              ghcr.io/user/repo:${{ github.ref_name }}
```

### Changelog Generation

```bash
# Install commitizen (conventional commits)
pip install commitizen

# Generate changelog from commits
cz bump --changelog

# Output: CHANGELOG.md with conventional commits
## v1.0.0 (2024-01-30)

### Features
- feat(api): add document upload endpoint
- feat(search): implement hybrid search

### Fixes
- fix(auth): resolve JWT token validation issue
```

---

## Deployment Strategies

### Blue-Green Deployment

```
Current: Blue (v1.0)  Production
New:     Green (v1.1)     Testing
         │
         ├─ Deploy Green
         ├─ Test Green
         ├─ Switch traffic: Blue → Green
         └─ Blue becomes standby
```

### Canary Deployment

```
Step 1: Deploy to 5% of traffic
Step 2: Monitor metrics
Step 3: Deploy to 25% of traffic
Step 4: Monitor metrics
Step 5: Deploy to 100% of traffic
```

### Rolling Deployment

```
Pod 1: v1.0  →  v1.1  (10%)
Pod 2: v1.0  →  v1.1  (10%)
Pod 3: v1.0  →  v1.1  (10%)
...
Pod 10: v1.0 →  v1.1  (10%)

All pods gradually updated
```

### Deployment Strategies Comparison

| Strategy | Downtime | Rollback | Complexity |
|----------|-----------|-----------|-----------|
| **Recreate** | Yes | Fast | Low |
| **Rolling** | No | Slow | Medium |
| **Blue-Green** | No | Instant | High |
| **Canary** | No | Gradual | High |

---

## Summary

| Component | Purpose | Tool |
|-----------|----------|-------|
| **CI** | Automated testing | GitHub Actions |
| **CD** | Automated deployment | GitHub Actions |
| **Pre-commit** | Local quality checks | pre-commit |
| **Docker** | Containerization | Docker BuildKit |
| **Security** | Vulnerability scanning | Trivy, Snyk, Bandit |
| **Release** | Version management | Semantic versioning |

### Key Takeaways

1. **CI/CD automates** testing and deployment
2. **Pre-commit hooks** catch issues locally
3. **Multi-stage Docker** reduces image size
4. **Security scanning** prevents vulnerabilities
5. **Deployment strategies** balance risk and complexity

---

## Further Reading

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Semantic Versioning](https://semver.org/)
- `notebooks/learning/05-cicd/ci-cd-basics.ipynb` - Interactive notebook
