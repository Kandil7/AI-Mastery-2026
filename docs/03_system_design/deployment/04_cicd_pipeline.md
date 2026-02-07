# CI/CD Pipeline Implementation Guide

## Overview

Continuous Integration and Continuous Deployment (CI/CD) automates the software delivery process. This guide teaches you how to build, test, and deploy RAG Engine Mini automatically using GitHub Actions.

## Learning Objectives

By the end of this module, you will:
1. Understand CI/CD concepts and benefits
2. Set up automated testing pipelines
3. Build and scan Docker images automatically
4. Implement security scanning
5. Deploy to multiple environments
6. Use deployment strategies (blue-green, canary)

**Estimated Time:** 4-6 hours

---

## Part 1: CI/CD Fundamentals

### What is CI/CD?

**Continuous Integration (CI):**
- Developers merge code frequently
- Automated builds and tests run
- Issues detected early

**Continuous Deployment (CD):**
- Automatically deploy passing builds
- Multiple environments (staging → production)
- Rollback capability

### The CI/CD Pipeline for RAG Engine

```
Developer Push
      │
      ▼
┌─────────────────────────────────────────┐
│              CI Phase                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │   Lint   │ │   Test   │ │  Build   │ │
│  │   Code   │ │   Code   │ │  Image   │ │
│  └──────────┘ └──────────┘ └──────────┘ │
└─────────────────────────────────────────┘
              │
              ▼ (if all pass)
┌─────────────────────────────────────────┐
│              CD Phase                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │  Stage   │ │  Smoke   │ │   Prod   │ │
│  │  Deploy  │ │   Test   │ │  Deploy  │ │
│  └──────────┘ └──────────┘ └──────────┘ │
└─────────────────────────────────────────┘
```

---

## Part 2: GitHub Actions Pipeline

### Project Workflow Overview

The project uses `.github/workflows/ci-cd.yml` with these stages:

**1. Lint (Code Quality)**
- Black (formatting)
- isort (import sorting)
- Flake8 (linting)
- mypy (type checking)
- Bandit (security scanning)

**2. Test**
- Unit tests (pytest)
- Integration tests
- Coverage reporting
- Service dependencies (Postgres, Redis, Qdrant)

**3. Security Scan**
- Bandit (Python security)
- Dependency vulnerability check
- Docker image scanning

**4. Build**
- Docker image build
- Multi-architecture support
- Push to registry

**5. Deploy**
- Staging deployment
- Production deployment (manual trigger)

### Understanding the Workflow File

```yaml
name: CI/CD Pipeline

# When to run
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  workflow_dispatch:  # Manual trigger

env:
  PYTHON_VERSION: "3.10"
  REGISTRY: ghcr.io
```

**Key Concepts:**

**Triggers (`on`):**
- `push`: Run on every commit to specified branches
- `pull_request`: Run on PRs (for validation)
- `workflow_dispatch`: Allow manual runs from GitHub UI

**Environment Variables (`env`):**
- Shared across all jobs
- Version pinning for consistency

### Stage 1: Lint Job

```yaml
jobs:
  lint:
    name: Code Quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4  # Get code
      
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: "pip"  # Speed up installs
      
      - name: Install tools
        run: |
          pip install black isort flake8 mypy bandit
      
      - name: Run Black
        run: black --check --diff src/ tests/
        # --check: Don't modify, just verify
        # --diff: Show what would change
      
      - name: Run isort
        run: isort --check-only --diff src/ tests/
      
      - name: Run Flake8
        run: |
          flake8 src/ tests/ \
            --max-line-length=100 \
            --ignore=E203,E501,W503
      
      - name: Run mypy
        run: mypy src/ --ignore-missing-imports
      
      - name: Security scan with Bandit
        run: bandit -r src/ -f json -o bandit-report.json || true
        # || true: Don't fail build, just report
      
      - name: Upload report
        uses: actions/upload-artifact@v3
        with:
          name: bandit-report
          path: bandit-report.json
```

**Why These Tools:**

| Tool | Purpose | Failure Impact |
|------|---------|----------------|
| **Black** | Code formatting | Low (cosmetic) |
| **isort** | Import organization | Low (cosmetic) |
| **Flake8** | Code quality | Medium (bugs) |
| **mypy** | Type safety | High (runtime errors) |
| **Bandit** | Security | Critical (vulnerabilities) |

### Stage 2: Test Job

```yaml
test:
  name: Tests
  runs-on: ubuntu-latest
  needs: lint  # Only run if lint passes
  
  services:
    # Service containers for integration tests
    postgres:
      image: postgres:15
      env:
        POSTGRES_USER: postgres
        POSTGRES_PASSWORD: postgres
        POSTGRES_DB: rag_test
      options: >-
        --health-cmd pg_isready
        --health-interval 10s
        --health-timeout 5s
        --health-retries 5
      ports:
        - 5432:5432
    
    redis:
      image: redis:7-alpine
      ports:
        - 6379:6379
    
    qdrant:
      image: qdrant/qdrant:latest
      ports:
        - 6333:6333
  
  steps:
    - uses: actions/checkout@v4
    
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/rag_test
        REDIS_URL: redis://localhost:6379/0
        QDRANT_HOST: localhost
      run: |
        pytest tests/ \
          --cov=src \
          --cov-report=xml \
          --cov-report=html \
          -v
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

**Service Containers:**
- Run alongside tests
- Accessible via localhost
- Health checks ensure readiness

### Stage 3: Build Job

```yaml
build:
  name: Build & Push Docker Image
  runs-on: ubuntu-latest
  needs: [lint, test]
  permissions:
    contents: read
    packages: write
  
  steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Login to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ github.repository }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix=,suffix=,format=short
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha  # GitHub Actions cache
        cache-to: type=gha,mode=max
        platforms: linux/amd64,linux/arm64
```

**Image Tagging Strategy:**
- `latest`: Main branch (production)
- `v1.2.3`: Git tags (releases)
- `sha-abc123`: Specific commits
- `pr-42`: Pull requests

### Stage 4: Deploy Job

```yaml
deploy-staging:
  name: Deploy to Staging
  runs-on: ubuntu-latest
  needs: build
  environment: staging
  if: github.ref == 'refs/heads/develop'
  
  steps:
    - name: Deploy to staging
      run: |
        # Use kubectl or helm
        kubectl set image deployment/rag-engine-api \
          api=${{ env.REGISTRY }}/${{ github.repository }}:sha-${{ github.sha }} \
          -n staging
        
        kubectl rollout status deployment/rag-engine-api -n staging
    
    - name: Run smoke tests
      run: |
        curl -f https://staging-api.example.com/health
        python scripts/smoke_test.py --url https://staging-api.example.com

deploy-production:
  name: Deploy to Production
  runs-on: ubuntu-latest
  needs: deploy-staging
  environment: production
  if: github.ref == 'refs/heads/main'
  
  steps:
    - name: Deploy to production
      run: |
        # Blue-green deployment
        kubectl apply -f k8s/production/
        kubectl rollout status deployment/rag-engine-api -n production
    
    - name: Notify team
      uses: slackapi/slack-github-action@v1
      with:
        payload: |
          {
            "text": "🚀 Production deployment complete!"
          }
```

---

## Part 3: Deployment Strategies

### Strategy 1: Rolling Update (Default)

```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%      # Can exceed desired count by 25%
      maxUnavailable: 0  # Never have unavailable pods
```

**Process:**
1. Create new pod
2. Verify health
3. Delete old pod
4. Repeat

**Pros:** Simple, no extra resources
**Cons:** Slow rollback

### Strategy 2: Blue-Green Deployment

```yaml
# Blue (current)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-engine-api-blue
spec:
  replicas: 3

---
# Green (new)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-engine-api-green
spec:
  replicas: 3
```

**Workflow:**
```bash
# Deploy green
cd green/
kubectl apply -f .

# Test green
kubectl port-forward svc/rag-engine-api-green 8080:80
curl http://localhost:8080/health

# Switch traffic
kubectl patch svc rag-engine-api -p '{"spec":{"selector":{"version":"green"}}}'

# Monitor, rollback if issues
kubectl patch svc rag-engine-api -p '{"spec":{"selector":{"version":"blue"}}}'
```

### Strategy 3: Canary Deployment

```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: rag-engine-api
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-engine-api
  service:
    port: 80
  analysis:
    interval: 30s
    threshold: 5
    maxWeight: 50
    stepWeight: 10
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
    - name: request-duration
      thresholdRange:
        max: 500
```

**Process:**
1. Deploy canary (10% traffic)
2. Monitor metrics
3. Increase to 50%
4. If healthy, 100%
5. If errors, rollback

---

## Part 4: Setting Up Your Pipeline

### Step 1: Enable GitHub Actions

Already enabled by default. The workflow file is at `.github/workflows/ci-cd.yml`.

### Step 2: Configure Secrets

**GitHub → Settings → Secrets and variables → Actions:**

```
PRODUCTION_KUBECONFIG    # Base64 encoded kubeconfig
STAGING_KUBECONFIG       # Base64 encoded kubeconfig
OPENAI_API_KEY          # For tests
DATABASE_URL            # For tests
DOCKER_USERNAME         # For Docker Hub
DOCKER_PASSWORD         # For Docker Hub
SLACK_WEBHOOK_URL       # For notifications
```

### Step 3: Customize for Your Environment

**Modify ci-cd.yml:**
```yaml
# Change registry
env:
  REGISTRY: docker.io  # Instead of ghcr.io

# Add environment-specific variables
- name: Deploy
  env:
    DATABASE_URL: ${{ secrets.PROD_DATABASE_URL }}
```

### Step 4: Test the Pipeline

```bash
# Make a small change
git checkout -b test-ci
echo "# Test" >> README.md
git add . && git commit -m "test: CI pipeline"
git push origin test-ci

# Create PR and watch Actions tab
```

---

## Part 5: Monitoring Your Pipeline

### GitHub Actions UI

**Insights:**
- Build duration trends
- Success/failure rates
- Most frequent errors

**Notifications:**
- Slack integration
- Email alerts
- GitHub notifications

### Pipeline Metrics to Track

| Metric | Target | Action if Failed |
|--------|--------|------------------|
| Build Duration | < 10 min | Optimize dependencies |
| Test Coverage | > 80% | Add tests |
| Security Issues | 0 critical | Fix immediately |
| Deploy Frequency | Daily | Improve automation |
| Lead Time | < 1 day | Optimize process |
| MTTR | < 1 hour | Improve monitoring |

---

## Part 6: Best Practices

### 1. Fast Feedback

**Parallel Jobs:**
```yaml
jobs:
  lint:
    # Fast (2 min)
  test-unit:
    # Medium (5 min)
  test-integration:
    # Slow (10 min)
  # Run lint + unit test quickly
  # Integration can wait
```

**Fail Fast:**
```yaml
steps:
  - name: Quick syntax check
    run: python -m py_compile src/**/*.py
  
  - name: Full test suite
    run: pytest
```

### 2. Security

**Secrets Management:**
- Never commit secrets
- Use GitHub Secrets
- Rotate regularly
- Audit access

**Least Privilege:**
```yaml
permissions:
  contents: read      # Only read code
  packages: write     # Only push images
  id-token: write     # For OIDC
```

### 3. Reliability

**Timeouts:**
```yaml
jobs:
  test:
    timeout-minutes: 15  # Fail if stuck
```

**Retries:**
```yaml
- name: Push to registry
  uses: docker/build-push-action@v5
  with:
    push: true
  retry:
    maxAttempts: 3
```

---

## Next Steps

1. **Review the existing workflow** at `.github/workflows/ci-cd.yml`
2. **Run a test build** by pushing to a branch
3. **Set up required secrets** in GitHub settings
4. **Configure environments** (staging/production)
5. **Add status badges** to README

**Continue to Module 5: Monitoring & Observability!** 📊
