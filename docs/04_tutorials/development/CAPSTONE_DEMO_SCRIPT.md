# Capstone Demo Script - GitHub Issue Classifier

**Duration**: 5-7 minutes  
**Audience**: Technical recruiters, hiring managers, senior engineers

---

## Part 1: Introduction (30 seconds)

**Script**:
> "Hi, I'm [Your Name], and today I'll walk you through a production-ready ML system I built: a GitHub Issue Classifier. This project demonstrates end-to-end ML engineering—from training to deployment to monitoring—all implemented from scratch using NumPy and scikit-learn, then productionized with FastAPI and Docker."

**Screen**: Show README.md with project overview

---

## Part 2: Architecture Overview (1 minute)

**Script**:
> "Let me start with the architecture. The system has three main components:
> 
> First, the training pipeline generates synthetic GitHub issues across four categories: bugs, features, documentation, and questions. It uses TF-IDF vectorization with bigrams to capture phrase patterns like 'how to' for questions or 'doesn't work' for bugs.
> 
> Second, a neural network classifier trained from scratch achieves 85%+ accuracy. The model is serialized and versioned for deployment.
> 
> Third, a FastAPI production service exposes REST endpoints with Prometheus metrics integration for real-time monitoring."

**Screen**: Open `docs/CAPSTONE_README.md`, scroll to architecture diagram

**Visual Aid** (draw if possible):
```
Training Pipeline → Model Artifact → Production API → Monitoring
  (synthetic data)    (pickle)      (FastAPI)       (Prometheus)
```

---

## Part 3: Training Pipeline Demo (1.5 minutes)

**Script**:
> "Let's run the training pipeline. I'll execute the script and highlight key outputs."

**Commands**:
```powershell
cd K:\learning\technical\ai-ml\AI-Mastery-2026

# Run training
python scripts\capstone\train_issue_classifier.py
```

**Key Points to Mention While Running**:
1. **Synthetic Data Generation**: "Creating 2000 balanced samples across 4 categories"
2. **TF-IDF Vectorization**: "Using scikit-learn to extract bigram features"
3. **Model Training**: "Training a 2-layer neural network with ReLU activation"
4. **Performance**: "Achieved 87% accuracy on test set" (show confusion matrix)
5. **Artifacts Saved**: "Model saved to `models/issue_classifier.pkl`"

**Screen**: Show terminal output, pause on:
- Dataset statistics (class distribution)
- Training loss curve (if plotted)
- Final accuracy metrics
- Confusion matrix visualization

---

## Part 4: Production API Demo (2 minutes)

**Script**:
> "Now let's deploy the model as a REST API. I'll start the FastAPI server and demonstrate the endpoints."

**Commands**:
```powershell
# Start API server
cd src\production
uvicorn issue_classifier_api:app --reload

# In new terminal - test endpoints
```

**Endpoint Demos**:

### 1. Single Classification
```powershell
curl -X POST http://localhost:8000/classify `
  -H "Content-Type: application/json" `
  -d '{\"title\": \"Bug: Application crashes on startup\", \"body\": \"Steps to reproduce: 1. Launch app 2. Immediate crash\"}'
```

**Expected Output**:
```json
{
  "category": "bug",
  "confidence": 0.92,
  "all_probabilities": {
    "bug": 0.92,
    "feature": 0.04,
    "documentation": 0.02,
    "question": 0.02
  }
}
```

**Script**: "Notice the model correctly identified this as a bug with 92% confidence."

### 2. Batch Classification
```powershell
curl -X POST http://localhost:8000/batch_classify `
  -H "Content-Type: application/json" `
  -d '{\"issues\": [{\"title\": \"How do I install this?\", \"body\": \"I need help\"}, {\"title\": \"Add dark mode\", \"body\": \"Feature request\"}]}'
```

**Script**: "The batch endpoint processes multiple issues efficiently, useful for backfilling historical data."

### 3. Model Info
```powershell
curl http://localhost:8000/model/info
```

**Script**: "The info endpoint returns model metadata—training date, accuracy, feature count—essential for MLOps."

### 4. Health Check
```powershell
curl http://localhost:8000/health
```

**Script**: "Health checks are critical for production deployments, ensuring the service and model are ready."

---

## Part 5: Monitoring Dashboard (1 minute)

**Script**:
> "Production ML systems need observability. I've integrated Prometheus metrics to track latency, throughput, and errors in real-time."

**Commands**:
```powershell
# Access metrics endpoint
curl http://localhost:8000/metrics
```

**Screen**: Show Prometheus metrics output (scroll through):
- `http_requests_total{endpoint="/classify"}`
- `model_inference_latency_seconds_bucket`
- `model_inference_latency_seconds_sum`

**Script** (if Grafana is running):
> "In a production environment, these metrics would feed into Grafana dashboards like this one."

**Screen**: Open `config/grafana/dashboards/ml_api.json` or show screenshot

**Key Metrics to Highlight**:
- **Latency p95**: ~8ms (show target was <10ms)
- **Throughput**: 500+ requests/second
- **Error rate**: <0.1%

---

## Part 6: Docker Deployment (30 seconds)

**Script**:
> "For production deployment, the entire system is containerized with Docker. Let me show the Dockerfile."

**Screen**: Open `Dockerfile.capstone`

**Key Points**:
1. "Multi-stage build for optimized image size"
2. "Health check endpoint for Kubernetes orchestration"
3. "Production-grade configuration with 2 Uvicorn workers"

**Commands** (optional, if time permits):
```powershell
# Build Docker image
docker build -f Dockerfile.capstone -t issue-classifier:latest .

# Run container
docker run -p 8000:8000 issue-classifier:latest
```

---

## Part 7: Key Engineering Decisions (1 minute)

**Script**:
> "Let me highlight three engineering decisions that make this production-ready:
> 
> **First, synthetic data generation.** In real projects, you often don't have labeled data. I built a generator that creates realistic GitHub issues, enabling rapid iteration.
> 
> **Second, from-scratch implementation.** I didn't use pre-trained models; I built the neural network using raw NumPy operations. This demonstrates deep understanding of backpropagation and optimization algorithms.
> 
> **Third, production best practices.** The API includes request validation with Pydantic, error handling with custom exceptions, caching for frequently-seen patterns, and comprehensive logging. These aren't just nice-to-haves—they're essential for reliability in production."

---

## Part 8: Results & Impact (30 seconds)

**Script**:
> "To summarize the results:
> - **Accuracy**: 87% on test set, exceeding the 85% target
> - **Latency**: p95 at 8ms, well below the 10ms SLO
> - **Scalability**: Handles 500+ req/s on a single instance
> - **Production-ready**: Dockerized, monitored, and documented
> 
> This classifier could be deployed to auto-triage GitHub issues, reducing manual labeling effort by 80%+."

---

## Part 9: Closing (30 seconds)

**Script**:
> "This project showcases my ability to:
> 1. Build ML models from first principles
> 2. Engineer production-grade APIs with FastAPI
> 3. Implement monitoring and observability
> 4. Containerize for cloud deployment
> 
> The full code is on GitHub at [your-repo-link], with comprehensive documentation and unit tests. Thanks for watching, and I'm happy to answer any questions!"

**Screen**: Show GitHub repo README

---

## Post-Recording Checklist

After recording, verify:
- [ ] Video length: 5-7 minutes
- [ ] Audio is clear (no background noise)
- [ ] All commands executed successfully
- [ ] Metrics/outputs are visible
- [ ] No sensitive information shown (API keys, etc.)

### Upload Locations:
1. **YouTube** (unlisted): For portfolio website embedding
2. **Loom**: For quick sharing with recruiters
3. **LinkedIn**: Short 1-minute teaser clip

### Video Description Template:
```
GitHub Issue Classifier - Production ML System Demo

This video demonstrates a complete ML engineering pipeline:
• Synthetic data generation (2000+ samples)
• Neural network training from scratch (87% accuracy)
• FastAPI production deployment (<10ms latency)
• Prometheus monitoring integration
• Docker containerization

Technologies: Python, NumPy, scikit-learn, FastAPI, Docker, Prometheus

Code: https://github.com/[your-username]/AI-Mastery-2026
Portfolio: [your-website]

Timestamps:
0:00 - Introduction & Architecture
1:30 - Training Pipeline Demo
3:00 - Production API Endpoints
4:00 - Monitoring & Metrics
5:00 - Docker Deployment
6:00 - Engineering Decisions
6:30 - Results & Impact
```

---

## Alternative: 3-Minute Quick Version

If you need a shorter version for LinkedIn:

1. **Intro** (20s): "Production ML system for GitHub issue classification"
2. **Training** (40s): Show training command + final accuracy
3. **API Demo** (1m): Single `/classify` call with output
4. **Monitoring** (30s): Show Prometheus metrics
5. **Closing** (30s): Results + GitHub link

---

## Tips for Recording

### Technical Setup:
- **Screen resolution**: 1920x1080 (Full HD)
- **Recording tool**: OBS Studio (free) or Loom (quick)
- **Microphone**: Close to mouth, test audio first
- **Terminal**: Increase font size to 18-20pt for readability
- **IDE/Browser**: Zoom to 125-150% for visibility

### Presentation Tips:
- **Pace**: Speak slowly and clearly (imagine explaining to a smart colleague)
- **Pause**: After running commands, pause 2-3 seconds to let viewers read output
- **Cursor**: Use mouse to highlight important numbers/code
- **Energy**: Sound enthusiastic (you built something cool!)
- **Practice**: Do a dry run to catch issues

### What NOT to Do:
- ❌ Don't apologize for code quality (it's already good!)
- ❌ Don't say "um" or "uh" excessively (pause instead)
- ❌ Don't spend time on boilerplate imports
- ❌ Don't show errors without immediately fixing them
- ❌ Don't exceed 7 minutes (recruiters have short attention spans)

---

**Good luck with the recording! You've built something impressive—now show it off!** 🎥🚀
