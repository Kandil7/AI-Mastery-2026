# 📝 AI-Mastery-2026 Assessment Framework

**Comprehensive evaluation system** for measuring learning progress and competency.

---

## 🎯 Assessment Philosophy

We believe in **authentic assessment** that measures real-world skills, not just memorization. Our assessments are:

- ✅ **Practical** - Hands-on coding challenges
- ✅ **Authentic** - Real-world scenarios
- ✅ **Formative** - Help you learn while being assessed
- ✅ **Transparent** - Clear rubrics and expectations
- ✅ **Adaptive** - Different difficulty levels

---

## 📊 Assessment Types

We use 4 types of assessments:

```
1. Knowledge Checks (Quizzes)
   → Test understanding of concepts
   
2. Skills Assessments (Coding Challenges)
   → Test ability to implement solutions
   
3. Project Assessments (Capstones)
   → Test ability to build complete systems
   
4. Certification Exams (Final Exams)
   → Comprehensive evaluation for certification
```

---

## 🎓 Tier 1: Fundamentals Assessments

### Quiz 1.1: Linear Algebra

**Format:** Multiple Choice + Short Answer  
**Duration:** 45 minutes  
**Questions:** 25  
**Pass Score:** 80%

#### Topics Covered

| Topic | Questions | Weight |
|-------|-----------|--------|
| Vectors & Operations | 5 | 20% |
| Matrices & Operations | 5 | 20% |
| Matrix Multiplication | 4 | 16% |
| Determinants | 3 | 12% |
| Eigenvalues & Eigenvectors | 4 | 16% |
| SVD (Singular Value Decomposition) | 4 | 16% |

#### Sample Questions

**Q1.** What is the dot product of vectors [3, 4] and [1, 2]?

A) 5  
B) 11  
C) 7  
D) 10  

**Answer:** B) 11  
**Explanation:** (3×1) + (4×2) = 3 + 8 = 11

---

**Q2.** If matrix A is 3×2 and matrix B is 2×4, what are the dimensions of AB?

A) 3×4  
B) 2×2  
C) 4×3  
D) Cannot multiply  

**Answer:** A) 3×4  
**Explanation:** Inner dimensions match (2), result is outer dimensions (3×4)

---

**Q3.** [Coding] Implement matrix multiplication from scratch:

```python
def matrix_multiply(A, B):
    """
    Multiply two matrices A and B.
    A: m×n matrix (list of lists)
    B: n×p matrix (list of lists)
    Returns: m×p matrix
    """
    # Your implementation here
    pass
```

**Rubric:**
- Correct dimensions check (2 points)
- Correct nested loops (4 points)
- Correct multiplication logic (3 points)
- Returns correct format (1 point)

---

### Quiz 1.2: Probability & Statistics

**Format:** Multiple Choice + Calculation  
**Duration:** 50 minutes  
**Questions:** 30  
**Pass Score:** 80%

#### Topics Covered

| Topic | Questions | Weight |
|-------|-----------|--------|
| Basic Probability | 5 | 17% |
| Conditional Probability | 5 | 17% |
| Bayes' Theorem | 4 | 13% |
| Distributions (Normal, Binomial) | 6 | 20% |
| Mean, Median, Mode | 3 | 10% |
| Variance & Standard Deviation | 4 | 13% |
| Hypothesis Testing | 3 | 10% |

---

### Coding Challenge 1.1: Gradient Descent Implementation

**Duration:** 2 hours  
**Difficulty:** ⭐⭐⭐☆☆  
**Points:** 100

#### Problem Statement

Implement gradient descent from scratch to find the minimum of a function.

#### Requirements

1. **Implement the function** (20 points)
   ```python
   def f(x):
       """Function to minimize: f(x) = x² + 5x + 6"""
       pass
   ```

2. **Implement the derivative** (20 points)
   ```python
   def df(x):
       """Derivative: f'(x) = 2x + 5"""
       pass
   ```

3. **Implement gradient descent** (40 points)
   ```python
   def gradient_descent(start_x, learning_rate, iterations):
       """
       Perform gradient descent.
       
       Args:
           start_x: Starting x value
           learning_rate: Step size (alpha)
           iterations: Number of iterations
       
       Returns:
           final_x: x value at minimum
           history: List of x values during optimization
       """
       pass
   ```

4. **Visualize the optimization** (20 points)
   ```python
   import matplotlib.pyplot as plt
   
   # Plot the function and optimization path
   pass
   ```

#### Test Cases

```python
# Test 1: Basic functionality
final_x, history = gradient_descent(start_x=10, learning_rate=0.1, iterations=100)
assert abs(final_x - (-2.5)) < 0.1  # Minimum is at x = -2.5

# Test 2: Different starting points
for start in [-10, 0, 5, 20]:
    final_x, _ = gradient_descent(start, 0.1, 100)
    assert abs(final_x - (-2.5)) < 0.1

# Test 3: Convergence
assert len(history) > 0
assert history[-1] < history[0]  # Should get closer to minimum
```

#### Rubric

| Criteria | Points | Excellent (90-100%) | Good (70-89%) | Needs Work (<70%) |
|----------|--------|---------------------|---------------|-------------------|
| Correct Implementation | 40 | Works perfectly | Minor bugs | Major issues |
| Code Quality | 20 | Clean, documented | Some issues | Poor structure |
| Visualization | 20 | Clear, informative | Basic plot | Missing/incorrect |
| Testing | 20 | All tests pass | Most pass | Few pass |

---

## 🎓 Tier 2: ML Practitioner Assessments

### Quiz 2.1: Classical Machine Learning

**Format:** Multiple Choice + Code Tracing  
**Duration:** 60 minutes  
**Questions:** 40  
**Pass Score:** 80%

#### Topics Covered

| Algorithm | Questions | Weight |
|-----------|-----------|--------|
| Linear Regression | 6 | 15% |
| Logistic Regression | 6 | 15% |
| Decision Trees | 5 | 12.5% |
| Random Forests | 5 | 12.5% |
| SVM | 5 | 12.5% |
| KNN | 4 | 10% |
| K-Means Clustering | 5 | 12.5% |
| PCA | 4 | 10% |

---

### Coding Challenge 2.1: Build a Complete ML Pipeline

**Duration:** 4 hours  
**Difficulty:** ⭐⭐⭐⭐☆  
**Points:** 200

#### Problem Statement

Build an end-to-end ML pipeline to predict customer churn.

#### Dataset

```python
# Provided dataset (CSV format)
# Columns:
# - customer_id
# - age
# - tenure (months)
# - monthly_charges
# - total_charges
# - contract_type (Month-to-month, One year, Two year)
# - payment_method
# - churn (target: Yes/No)
```

#### Requirements

1. **Data Exploration** (30 points)
   - Load and inspect data
   - Handle missing values
   - Visualize distributions
   - Check class balance

2. **Feature Engineering** (40 points)
   - Encode categorical variables
   - Scale numerical features
   - Create new features (optional)
   - Split train/test

3. **Model Training** (50 points)
   - Train at least 3 different models:
     - Logistic Regression
     - Random Forest
     - Gradient Boosting
   - Use cross-validation

4. **Model Evaluation** (40 points)
   - Accuracy, Precision, Recall, F1
   - ROC-AUC curve
   - Confusion matrix
   - Compare models

5. **Prediction Function** (40 points)
   ```python
   class ChurnPredictor:
       def __init__(self, model_path):
           self.model = load_model(model_path)
       
       def predict(self, customer_data):
           """
           Predict churn for a customer.
           
           Args:
               customer_data: Dict with customer info
           
           Returns:
               prediction: 'Yes' or 'No'
               probability: Churn probability
           """
           pass
   ```

#### Rubric

| Criteria | Points | Excellent | Good | Needs Work |
|----------|--------|-----------|------|------------|
| Data Exploration | 30 | Thorough analysis | Basic analysis | Minimal |
| Feature Engineering | 40 | Creative, effective | Standard approach | Issues |
| Model Training | 50 | 3+ models, CV | 2-3 models | 1 model |
| Model Evaluation | 40 | Comprehensive | Basic metrics | Incomplete |
| Prediction Function | 40 | Production-ready | Works | Bugs |

---

## 🎓 Tier 3: LLM Engineer Assessments

### Quiz 3.1: Transformers & Attention

**Format:** Multiple Choice + Diagram Interpretation  
**Duration:** 75 minutes  
**Questions:** 50  
**Pass Score:** 85%

#### Topics Covered

| Topic | Questions | Weight |
|-------|-----------|--------|
| Self-Attention Mechanism | 10 | 20% |
| Multi-Head Attention | 8 | 16% |
| Positional Encoding | 6 | 12% |
| Transformer Architecture | 10 | 20% |
| Encoder-Decoder | 6 | 12% |
| BERT vs GPT | 6 | 12% |
| Fine-Tuning | 4 | 8% |

---

### Coding Challenge 3.1: Build a RAG System

**Duration:** 8 hours  
**Difficulty:** ⭐⭐⭐⭐⭐  
**Points:** 300

#### Problem Statement

Build a production-ready RAG (Retrieval-Augmented Generation) system for document Q&A.

#### Requirements

1. **Document Processing** (50 points)
   ```python
   class DocumentProcessor:
       def load_documents(self, directory):
           """Load PDF, TXT, MD files"""
           pass
       
       def chunk_documents(self, documents, chunk_size=512, overlap=50):
           """Split into overlapping chunks"""
           pass
       
       def embed_chunks(self, chunks):
           """Generate embeddings"""
           pass
   ```

2. **Vector Store** (50 points)
   ```python
   class VectorStore:
       def __init__(self, dimension=384):
           """Initialize vector store"""
           pass
       
       def add(self, embeddings, metadata):
           """Add embeddings to store"""
           pass
       
       def search(self, query_embedding, k=10):
           """Find k most similar chunks"""
           pass
   ```

3. **Retrieval System** (60 points)
   ```python
   class HybridRetriever:
       def __init__(self, vector_store, keyword_index):
           """Initialize hybrid retriever"""
           pass
       
       def retrieve(self, query, k=10):
           """
           Retrieve using both vector and keyword search.
           Use RRF (Reciprocal Rank Fusion) for combination.
           """
           pass
   ```

4. **Reranking** (40 points)
   ```python
   class CrossEncoderReranker:
       def rerank(self, query, chunks, top_n=5):
           """Rerank chunks using cross-encoder"""
           pass
   ```

5. **Generation** (60 points)
   ```python
   class RAGGenerator:
       def __init__(self, llm, retriever, reranker):
           """Initialize RAG system"""
           pass
       
       def generate(self, query):
           """
           Full RAG pipeline:
           1. Retrieve relevant chunks
           2. Rerank
           3. Generate answer with context
           """
           pass
   ```

6. **Evaluation** (40 points)
   ```python
   class RAGEvaluator:
       def evaluate(self, queries, ground_truth_answers):
           """
           Evaluate RAG system:
           - Answer relevance
           - Context precision
           - Faithfulness (no hallucination)
           """
           pass
   ```

#### Rubric

| Criteria | Points | Excellent | Good | Needs Work |
|----------|--------|-----------|------|------------|
| Document Processing | 50 | Robust, handles errors | Basic implementation | Incomplete |
| Vector Store | 50 | Efficient, scalable | Works | Issues |
| Hybrid Retrieval | 60 | RRF implemented well | Basic combination | Single method |
| Reranking | 40 | Cross-encoder works | Simple reranking | Missing |
| Generation | 60 | Coherent, accurate | Works | Hallucinations |
| Evaluation | 40 | Comprehensive metrics | Basic metrics | Missing |

---

## 🎓 Tier 4: Production Expert Assessments

### Quiz 4.1: MLOps & Production Systems

**Format:** Case Study + Multiple Choice  
**Duration:** 60 minutes  
**Questions:** 35  
**Pass Score:** 85%

#### Topics Covered

| Topic | Questions | Weight |
|-------|-----------|--------|
| CI/CD for ML | 8 | 23% |
| Model Monitoring | 7 | 20% |
| Scaling & Load Balancing | 6 | 17% |
| Security Best Practices | 6 | 17% |
| Cost Optimization | 5 | 14% |
| A/B Testing | 3 | 9% |

---

### Case Study 4.1: Design a Production ML System

**Duration:** 4 hours  
**Difficulty:** ⭐⭐⭐⭐⭐  
**Points:** 250

#### Problem Statement

Design and implement a production-ready ML system for real-time fraud detection.

#### Requirements

1. **System Architecture** (50 points)
   - Draw architecture diagram
   - Justify technology choices
   - Consider scalability
   - Plan for failure

2. **API Design** (50 points)
   ```python
   @app.post("/api/v1/predict")
   @rate_limit("100/minute")
   async def predict_fraud(transaction: TransactionRequest):
       """
       Real-time fraud prediction endpoint.
       Must include:
       - Authentication
       - Rate limiting
       - Input validation
       - Error handling
       - Logging
       """
       pass
   ```

3. **Monitoring Setup** (50 points)
   ```python
   class ModelMonitor:
       def track_prediction_latency(self):
           """Track p50, p95, p99 latency"""
           pass
       
       def detect_data_drift(self, reference_data, current_data):
           """Detect distribution shift"""
           pass
       
       def track_model_accuracy(self, predictions, actuals):
           """Track accuracy over time"""
           pass
   ```

4. **CI/CD Pipeline** (50 points)
   ```yaml
   # GitHub Actions workflow
   name: ML Pipeline
   
   on: [push]
   
   jobs:
     test:
       - Run unit tests
       - Run integration tests
       - Check model performance
   
     deploy:
       - Build Docker image
       - Push to registry
       - Deploy to Kubernetes
       - Run smoke tests
   ```

5. **Security Implementation** (50 points)
   - API authentication
   - Input sanitization
   - Secrets management
   - Audit logging

#### Rubric

| Criteria | Points | Excellent | Good | Needs Work |
|----------|--------|-----------|------|------------|
| Architecture | 50 | Scalable, robust | Functional | Flawed |
| API Design | 50 | Production-ready | Works | Incomplete |
| Monitoring | 50 | Comprehensive | Basic | Missing |
| CI/CD | 50 | Automated, tested | Basic pipeline | Manual |
| Security | 50 | Hardened | Basic security | Vulnerabilities |

---

## 🎓 Tier 5: Capstone Assessments

### Capstone Project Evaluation

**Duration:** 4-6 weeks  
**Difficulty:** ⭐⭐⭐⭐⭐  
**Points:** 500

#### Project Requirements

Students must build a complete AI system that:

1. **Solves a Real Problem** (100 points)
   - Clear problem statement
   - Target users identified
   - Value proposition

2. **Technical Implementation** (150 points)
   - Clean, maintainable code
   - Best practices followed
   - Tests included
   - Documentation

3. **Innovation** (100 points)
   - Novel approach or combination
   - Creative solution
   - Technical challenge

4. **Production Readiness** (100 points)
   - Deployed and accessible
   - Monitoring setup
   - Error handling
   - Performance optimized

5. **Presentation** (50 points)
   - Demo video (5-10 min)
   - Technical report (10-15 pages)
   - Code repository
   - Live presentation (optional)

#### Rubric

| Criteria | Weight | Excellent (90-100%) | Good (70-89%) | Satisfactory (60-69%) | Poor (<60%) |
|----------|--------|---------------------|---------------|----------------------|-------------|
| Problem Definition | 20% | Clear, impactful | Clear | Vague | Unclear |
| Technical Quality | 30% | Excellent code | Good code | Acceptable | Poor code |
| Innovation | 20% | Highly novel | Some novelty | Standard | Derivative |
| Production Quality | 20% | Fully deployed | Mostly ready | Partial | Not deployed |
| Presentation | 10% | Excellent | Good | Adequate | Poor |

---

## 📊 Grading Scale

### Letter Grades

| Percentage | Letter | Description |
|------------|--------|-------------|
| 90-100% | A | Excellent - Exceeds expectations |
| 80-89% | B | Good - Meets expectations |
| 70-79% | C | Satisfactory - Approaching expectations |
| 60-69% | D | Poor - Below expectations |
| 0-59% | F | Fail - Does not meet minimum standards |

### Pass/Fail

- **Pass:** 80% or higher
- **Fail:** Below 80%

---

## 🎓 Certification Requirements

### To Earn Each Certificate

| Certificate | Required Assessments | Minimum Score |
|-------------|---------------------|---------------|
| AI Foundations | Tier 0 Final Exam | 80% |
| ML Fundamentals | Tier 1 Quizzes (avg) + Final Exam | 80% |
| ML Practitioner | Tier 2 Quizzes + Coding Challenge | 80% |
| LLM Engineer | Tier 3 Quizzes + RAG Challenge | 85% |
| Production Expert | Tier 4 Quizzes + Case Study | 85% |
| AI Mastery Capstone | Capstone Project | 80% |

---

## 📝 Academic Integrity

### Honor Code

By submitting assessments, students agree to:

1. **Original Work** - Submit only your own work
2. **Proper Citation** - Cite all sources used
3. **No Cheating** - Do not copy from others
4. **No Plagiarism** - Do not use AI to complete assessments (unless explicitly allowed)

### Violations

| Violation | Consequence |
|-----------|-------------|
| First offense | Warning + resubmit |
| Second offense | Fail assessment |
| Third offense | Fail course |
| Severe violation | Expulsion |

---

## ♿ Accessibility

### Accommodations Available

- Extended time for timed assessments
- Alternative formats for questions
- Screen reader compatibility
- Flexible scheduling

### Request Accommodations

Contact: accommodations@ai-mastery-2026.dev

---

**Last Updated:** April 2, 2026  
**Version:** 1.0  
**Maintained By:** AI-Mastery-2026 Assessment Team

---

[← Back to Course Catalog](../courses/COURSE_CATALOG.md) | [View Quizzes](quizzes/README.md) | [View Coding Challenges](coding-challenges/README.md)
