# Model Evaluation - Interview Questions

A comprehensive collection of model evaluation and metrics questions for AI/ML interviews.

---

## Table of Contents
1. [Classification Metrics](#1-classification-metrics)
2. [Regression Metrics](#2-regression-metrics)
3. [Ranking Metrics](#3-ranking-metrics)
4. [Cross-Validation](#4-cross-validation)
5. [RAG Evaluation](#5-rag-evaluation)

---

## 1. Classification Metrics

### Q1.1: Explain precision, recall, and F1 score.

**Answer:**

**Confusion Matrix:**
```
                Predicted
              Pos    Neg
Actual Pos    TP     FN
       Neg    FP     TN
```

**Metrics:**

| Metric | Formula | Question It Answers |
|--------|---------|---------------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | How often is the model correct? |
| **Precision** | TP/(TP+FP) | Of predicted positives, how many are correct? |
| **Recall** | TP/(TP+FN) | Of actual positives, how many did we find? |
| **F1 Score** | 2·P·R/(P+R) | Harmonic mean of precision and recall |

**Trade-off:**
- High precision → Conservative (few false positives)
- High recall → Aggressive (few false negatives)

**When to Prioritize:**

| Scenario | Prioritize | Reason |
|----------|-----------|--------|
| Spam detection | Precision | Don't lose important emails |
| Disease screening | Recall | Don't miss sick patients |
| Balanced | F1 Score | Need both |

---

### Q1.2: When should you use accuracy vs F1 score?

**Answer:**

**Use Accuracy When:**
- Classes are balanced
- All errors have equal cost
- Simple interpretation needed

**Use F1 Score When:**
- Classes are imbalanced
- False positives and negatives have different costs
- Need balance between precision and recall

**Example - Fraud Detection:**
- 99% legitimate, 1% fraud
- Predicting "all legitimate" → 99% accuracy but useless!
- F1 score properly evaluates fraud detection ability

**Macro vs Micro Averaging (Multi-class):**
- **Macro**: Average F1 across classes (treats all classes equally)
- **Micro**: Global TP, FP, FN (weights by class frequency)

---

### Q1.3: Explain the ROC curve and AUC.

**Answer:**

**ROC Curve:** Plots True Positive Rate vs False Positive Rate at different thresholds.

```
TPR (Recall)
    1│    ╭──────
     │   ╱
     │  ╱
     │ ╱   AUC = Area
     │╱    Under Curve
     └─────────────→ FPR
```

**Interpretation:**

| AUC Value | Model Quality |
|-----------|--------------|
| 0.5 | Random guess |
| 0.7-0.8 | Fair |
| 0.8-0.9 | Good |
| 0.9-1.0 | Excellent |

**Why Use AUC?**
- Threshold-independent evaluation
- Works well for imbalanced classes
- Measures separability between classes

**When NOT to Use AUC:**
- Heavily imbalanced data → Use PR-AUC instead
- When you need a specific threshold
- When interpretability is crucial

---

### Q1.4: What is the PR (Precision-Recall) curve and when do you use it?

**Answer:**

**PR Curve:** Plots Precision vs Recall at different thresholds.

**When to Use PR-AUC:**
- Imbalanced datasets (rare positive class)
- When you care more about positive class
- Medical diagnosis, fraud detection

**Comparison:**

| Aspect | ROC-AUC | PR-AUC |
|--------|---------|--------|
| **Baseline** | 0.5 (random) | Proportion of positives |
| **Imbalanced data** | Can be misleading | More informative |
| **Interpretation** | Easy | Harder |

---

## 2. Regression Metrics

### Q2.1: Compare MSE, MAE, and RMSE.

**Answer:**

| Metric | Formula | Characteristics |
|--------|---------|-----------------|
| **MSE** | (1/n)Σ(y-ŷ)² | Penalizes large errors more |
| **MAE** | (1/n)Σ\|y-ŷ\| | Robust to outliers |
| **RMSE** | √MSE | Same units as target |

**When to Use:**

| Scenario | Metric | Reason |
|----------|--------|--------|
| Large errors are bad | MSE/RMSE | Squares amplify large errors |
| Outliers present | MAE | More robust |
| Need interpretable units | RMSE | Same scale as y |
| Optimization | MSE | Differentiable everywhere |

---

### Q2.2: Explain R² (coefficient of determination).

**Answer:**

**Formula:**
$$R² = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y-\hat{y})²}{\sum(y-\bar{y})²}$$

**Interpretation:**
- R² = 1: Perfect fit
- R² = 0: Model = mean prediction
- R² < 0: Worse than predicting mean

**Key Points:**
- Measures proportion of variance explained
- Can only increase with more features (use Adjusted R²)
- Not affected by scaling

**Adjusted R²:**
$$R²_{adj} = 1 - (1-R²)\frac{n-1}{n-p-1}$$

Penalizes adding useless features.

---

### Q2.3: How do you evaluate models when different error magnitudes matter differently?

**Answer:**

**Weighted Metrics:**
- Assign different weights to samples based on importance
- High-value predictions may need lower relative error

**Percentage Errors:**
- **MAPE**: Mean Absolute Percentage Error = (1/n)Σ|y-ŷ|/|y|
- Problem: Undefined when y=0, asymmetric

**Log-based Metrics:**
- **MSLE**: Mean Squared Log Error
- Good for targets spanning orders of magnitude
- Penalizes underestimation more

**Custom Loss Functions:**
- Design based on business requirements
- Example: Asymmetric loss for over/underestimation

---

## 3. Ranking Metrics

### Q3.1: Explain MRR (Mean Reciprocal Rank).

**Answer:**

**Formula:**
$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$

Where rank_i is the position of the first relevant result.

**Example:**
```
Query 1: Relevant at position 3 → 1/3
Query 2: Relevant at position 1 → 1/1
Query 3: Relevant at position 2 → 1/2

MRR = (1/3 + 1 + 1/2) / 3 = 0.61
```

**Use Cases:**
- Search engines
- Question answering
- Any "find the one right answer" task

---

### Q3.2: Explain nDCG (Normalized Discounted Cumulative Gain).

**Answer:**

**Problem:** Some documents are more relevant than others (graded relevance).

**DCG@k:**
$$DCG@k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

**nDCG@k:**
$$nDCG@k = \frac{DCG@k}{IDCG@k}$$

Where IDCG is DCG of ideal (perfect) ranking.

**Properties:**
- Ranges from 0 to 1
- Accounts for graded relevance
- Discounts lower positions (logarithmic)

**Example:**
```
Ranking: [3, 2, 0, 1]  (relevance scores)
Position weights: 1/log2(2), 1/log2(3), 1/log2(4), 1/log2(5)

DCG = 3*1 + 2*0.63 + 0*0.5 + 1*0.43 = 4.69
Ideal: [3, 2, 1, 0] → IDCG = 5.89
nDCG = 4.69 / 5.89 = 0.80
```

---

### Q3.3: What is Precision@K and Recall@K?

**Answer:**

**Precision@K:**
$$P@K = \frac{|relevant \cap top-k|}{k}$$

Of the top K results, how many are relevant?

**Recall@K:**
$$R@K = \frac{|relevant \cap top-k|}{|relevant|}$$

Of all relevant documents, how many are in top K?

**Example:**
```
Relevant: {doc1, doc3, doc5, doc7}
Top-5 ranked: [doc1, doc2, doc3, doc6, doc5]

P@5 = 3/5 = 0.6  (3 relevant in top 5)
R@5 = 3/4 = 0.75 (3 of 4 relevant found)
```

---

## 4. Cross-Validation

### Q4.1: Explain K-fold cross-validation.

**Answer:**

**Process:**
1. Split data into K equal folds
2. For i = 1 to K:
   - Train on K-1 folds
   - Validate on fold i
3. Average performance across folds

**Benefits:**
- Uses all data for training and validation
- Reduces variance of performance estimate
- Detects overfitting

**When to Use:**
- Limited data
- Need reliable performance estimate
- Model selection

**Common Values:**
- K=5 or K=10 (standard)
- K=n (Leave-One-Out) for very small data

---

### Q4.2: What is stratified sampling and when is it important?

**Answer:**

**Stratified Sampling:** Ensures each fold has the same class distribution as the full dataset.

**Example:**
```
Full data: 90% negative, 10% positive

Without stratification:
  Fold 1: 95% negative, 5% positive  ← Imbalanced!
  
With stratification:
  Fold 1: 90% negative, 10% positive ← Same as full data
```

**When Important:**
- Imbalanced classification
- Multi-class with varying frequencies
- Regression with non-uniform distribution

---

### Q4.3: How do you prevent data leakage in cross-validation?

**Answer:**

**Data Leakage:** Information from validation/test data "leaks" into training.

**Common Mistakes:**

| Mistake | Why It's Wrong | Fix |
|---------|---------------|-----|
| Normalize before split | Uses test data statistics | Fit scaler on train only |
| Feature selection on all data | Uses test data | Select features on train |
| Time series random split | Future leaks to past | Use temporal split |
| Duplicate samples in train/test | Memorizing test | Remove duplicates first |

**Prevention:**
1. **Pipeline everything** - sklearn Pipeline applies transforms correctly
2. **Think about time** - Don't use future to predict past
3. **Check for duplicates** - Same sample in train and test = cheating
4. **Feature engineering on train only**

---

## 5. RAG Evaluation

### Q5.1: How do you evaluate RAG systems?

**Answer:**

**Key Dimensions:**

| Metric | Question | Measurement |
|--------|----------|-------------|
| **Faithfulness** | Is the answer grounded in context? | Check claims against retrieved docs |
| **Relevance** | Does answer address the question? | Semantic similarity to question |
| **Context Precision** | Was retrieved context useful? | % of context used for answer |
| **Context Recall** | Did we find all relevant info? | % of needed info retrieved |

**Evaluation Methods:**

1. **Human Evaluation** - Gold standard but expensive
2. **Reference-based** - Compare to ground truth answers
3. **LLM-as-Judge** - Use GPT-4 to rate quality
4. **Automated metrics** - Embedding similarity, keyword overlap

---

### Q5.2: What is the RAGAS framework?

**Answer:**

**RAGAS** (RAG Assessment) provides automated metrics:

| Metric | Formula/Approach |
|--------|------------------|
| **Faithfulness** | Claims in answer supported by context |
| **Answer Relevance** | How relevant is answer to question |
| **Context Precision** | Are high-ranked contexts relevant |
| **Context Recall** | Is ground truth covered by context |

**Benefits:**
- Automated (no human annotation)
- Comprehensive coverage
- Correlates with human judgment

**Limitations:**
- Relies on LLM for evaluation (circular)
- May miss subtle errors
- Expensive for large evaluations

---

### Q5.3: Explain hallucination detection in LLMs.

**Answer:**

**Hallucination:** LLM generates information not grounded in source/reality.

**Types:**
1. **Intrinsic**: Contradicts the source
2. **Extrinsic**: Adds unsupported information

**Detection Methods:**

| Method | How It Works |
|--------|--------------|
| **Fact checking** | Verify claims against knowledge base |
| **NLI-based** | Does context entail the answer? |
| **Self-consistency** | Sample multiple answers, check agreement |
| **Retrieval grounding** | Check if answer elements in context |

**Metrics:**
- Faithfulness score (% claims grounded)
- Hallucination rate (% ungrounded claims)
- Factual accuracy (against ground truth)

---

## 📚 Summary Table

| Task | Primary Metrics | When Imbalanced |
|------|----------------|-----------------|
| **Binary Classification** | Accuracy, F1 | Precision, Recall, PR-AUC |
| **Multi-class Classification** | Macro F1, Accuracy | Weighted F1 |
| **Regression** | RMSE, MAE, R² | MAPE, custom loss |
| **Ranking** | nDCG, MRR | P@K, R@K |
| **RAG/Generation** | Faithfulness, BLEU | Human eval, LLM-judge |
