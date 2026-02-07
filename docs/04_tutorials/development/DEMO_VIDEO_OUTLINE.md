# Capstone Demo Video Outline (5 Minutes)

## Video Structure

**Total Duration**: 5 minutes  
**Format**: Screen recording + voiceover  
**Tools**: OBS Studio / Loom / Zoom

---

## Introduction (30 seconds)

**Script**:
> "Hi, I'm [Your Name], and today I'm showcasing my GitHub Issue Classifier — a production-ready ML system that automatically categorizes issues as bugs, features, questions, or documentation requests. This project demonstrates the complete ML lifecycle from data generation to deployment."

**Visuals**:
- Show project title slide
- Quick architecture diagram

---

## Section 1: Problem & Architecture (60 seconds)

**Script**:
> "The problem: GitHub repositories receive hundreds of issues daily. Manual labeling is time-consuming and inconsistent. My solution uses a neural network trained from scratch — no scikit-learn, pure NumPy and custom implementations from my `src/ml/deep_learning.py` module."

**Visuals**:
- Show architecture diagram:
  ```
  Raw Text → TF-IDF → Neural Network → Classification
  ```
- Highlight key components:
  - Data generation (2000+ balanced samples)
  - Text preprocessing (TF-IDF with bigrams)
  - 3-layer neural network (100→64→32→4)

**Key Talking Points**:
- "Built from scratch to demonstrate deep understanding"
- "Achieves >85% test accuracy"
- "Production-ready with <10ms p95 latency"

---

## Section 2: Training Pipeline Demo (90 seconds)

**Script**:
> "Let me walk through the training pipeline. First, I generate synthetic data with realistic GitHub issue patterns..."

**Terminal Demo**:
```bash
# Show command
python scripts/capstone/train_issue_classifier.py

# Fast-forward through output, highlighting:
```

**Highlight in Video** (pause/zoom):
1. **Data Generation**: "Creating 2000 balanced samples across 4 categories"
2. **Training Progress**: "Epoch 50/50 - Accuracy climbing to 87%"
3. **Evaluation Metrics**:
   - Accuracy: 87.3%
   - Precision/Recall/F1 for each class
4. **Visualizations Generated**:
   - Training loss curve
   - Confusion matrix
   - Classification report

**Script Overlay**:
> "Notice the confusion matrix — most misclassifications are between semantically similar categories like 'feature' and 'question', which makes sense. The model achieves 87% accuracy, exceeding our 85% target."

---

## Section 3: API Demo - Live Predictions (90 seconds)

**Script**:
> "Now let's see the API in action. I've deployed a FastAPI service with full observability..."

**Terminal 1 - Start API**:
```bash
uvicorn src.production.issue_classifier_api:app --port 8000
```

**Show**:
- API startup logs
- Health check endpoint response

**Terminal 2 - Test Predictions**:
```bash
# Bug example
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Application crashes when clicking submit button"}'

# Response (highlight):
{
  "label": "bug",
  "confidence": 0.92,
  "all_probabilities": {"bug": 0.92, "feature": 0.04, ...},
  "latency_ms": 8.3
}

# Feature example
curl -X POST http://localhost:8000/classify \
  -d '{"text": "Add dark mode support to the dashboard"}'

# Response:
{
  "label": "feature",
  "confidence": 0.89,
  "latency_ms": 7.1
}
```

**Script**:
> "Notice the sub-10ms latency — this is production-ready. The API also exposes Prometheus metrics at /metrics for monitoring."

**Show Browser** (quick):
- Navigate to `http://localhost:8000/docs`
- Show Swagger UI with all endpoints
- Demonstrate `/metrics` endpoint (Prometheus format)

---

## Section 4: Production Deployment (60 seconds)

**Script**:
> "For deployment, I've containerized everything with Docker..."

**Show Dockerfile.capstone**:
```dockerfile
FROM python:3.10-slim
# ... highlight key steps
COPY src/ /app/src/
COPY models/ /app/models/
HEALTHCHECK --interval=30s ...
```

**Terminal - Docker Demo**:
```bash
# Build
docker build -f Dockerfile.capstone -t issue-classifier .

# Run
docker run -p 8000:8000 issue-classifier

# Show health check passing
docker ps  # Show HEALTHY status
```

**Script**:
> "The container includes health checks, logging, and can scale horizontally behind a load balancer. In production, this would integrate with Kubernetes for auto-scaling based on traffic."

---

## Section 5: Key Learnings & Next Steps (60 seconds)

**Script**:
> "This project taught me three critical lessons:"

**Slide 1 - Technical Depth**:
> "1. **From-scratch implementations** give you true understanding. I can now explain backpropagation, not just use `model.fit()`."

**Slide 2 - Production Engineering**:
> "2. **Production ML isn't just training**. It's APIs, monitoring, Docker, error handling, and performance optimization. My p95 latency is 8ms."

**Slide 3 - Full Lifecycle**:
> "3. **End-to-end ownership** — from synthetic data generation to deployed service with tests and docs."

**Show Quick Metrics Slide**:
```
✓ >85% Accuracy Achieved: 87.3%
✓ <10ms p95 Latency: 8.2ms
✓ Full Test Coverage: 95%
✓ Production Deployment: Docker + FastAPI
✓ Observability: Prometheus metrics
```

**Closing**:
> "Next steps would include A/B testing against sklearn baseline, retraining pipeline with MLflow, and deploying to cloud with CI/CD. Check out the full code and documentation on my GitHub. Thanks for watching!"

**Visuals**:
- Show README with links
- GitHub repo URL: `github.com/yourusername/AI-Mastery-2026`

---

## Production Tips

### Recording Setup
1. **Clean desktop**: Close unnecessary apps
2. **Terminal theme**: Use readable fonts (18pt+), high contrast
3. **Mouse highlighting**: Enable cursor highlight in OBS
4. **Slow mouse movements**: Allow viewers to follow
5. **Test audio**: Clear, no background noise

### Editing Checklist
- [ ] Add intro/outro music (optional, subtle)
- [ ] Speed up long outputs (2-3x)
- [ ] Add captions for key metrics
- [ ] Zoom in on important text
- [ ] Add transitions between sections
- [ ] Include GitHub link in description

### Export Settings
- **Resolution**: 1920x1080 (1080p)
- **Frame rate**: 30fps
- **Format**: MP4 (H.264)
- **Audio**: AAC, 128kbps

---

## Script Word Count
~750 words → ~5 minutes at 150 words/minute speaking pace

---

## Emergency Time Cuts (if over 5 min)
1. Skip Swagger UI demo (just mention it)
2. Reduce Docker section to build+run only
3. Show 2 predictions instead of 3
4. Fast-forward through model training

---

**Ready to record!** 🎥

Save this file for reference during recording.
