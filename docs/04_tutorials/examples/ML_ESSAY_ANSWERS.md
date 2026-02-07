# Essay Answers: ML Foundations

Detailed answers to the essay questions from the ML Study Guide.

---

## Essay 1: Mathematics in Machine Learning

**The Four Pillars of ML Mathematics**

### Linear Algebra
- **PCA**: Uses eigendecomposition of covariance matrix `C = VΛVᵀ` to find principal components
- **SVD**: Netflix decomposes user-movie matrix `R = UΣVᵀ` into latent factors

### Calculus  
- **Gradient Descent**: Update rule `θ := θ - α∇L(θ)` uses partial derivatives
- **Backpropagation**: Chain rule computes gradients through neural network layers

### Probability Theory
- **Naive Bayes**: Google Spam uses `P(spam|words) ∝ P(words|spam)P(spam)`
- **Bayesian Inference**: Combines prior knowledge with data

### Statistics
- **MLE**: Find θ maximizing `P(data|θ)`
- **Hypothesis Testing**: Validate model improvements

---

## Essay 2: MLE vs MAP Estimation

### Maximum Likelihood Estimation (MLE)
```
θ_MLE = argmax P(X|θ)
```
- Only considers data likelihood
- Can overfit with limited data

### Maximum A Posteriori (MAP)
```
θ_MAP = argmax P(X|θ)P(θ)
```
- Adds prior P(θ) as regularization
- Prior encodes beliefs before seeing data

### Overfitting & Regularization

**Overfitting**: Model fits noise, not signal
- Symptoms: High train accuracy, low test accuracy

**MAP as Regularization**:
- Gaussian prior → L2 regularization (Ridge)
- Laplace prior → L1 regularization (Lasso)

```
L2: minimize ||y - Xw||² + λ||w||²
     ↑ data fit         ↑ prior penalty
```

---

## Essay 3: SVD and PCA Relationship

### Mathematical Connection
**PCA** = **SVD** applied to centered data

Given centered data X:
- PCA: eigendecompose `XᵀX = VΛVᵀ`
- SVD: `X = UΣVᵀ`, then `XᵀX = VΣ²Vᵀ`

**Relationship**: `Λ = Σ²/(n-1)`

### Netflix Recommendation System

1. **Problem**: Sparse user-movie rating matrix (99% missing)
2. **SVD Solution**: `R ≈ UΣVᵀ`
   - U: User latent factors (preferences)
   - V: Movie latent factors (genres, mood)
   - Σ: Importance of each factor
3. **Prediction**: `r_um = Σ_k U_uk × σ_k × V_mk`

---

## Essay 4: Supply Chain Evolution

### Traditional → AI-Driven

| Era | Approach | Characteristics |
|-----|----------|-----------------|
| Manual | Spreadsheets | Human intuition |
| OR | Linear programming | Optimal decisions |
| ML | DeepAR forecasting | Uncertainty quantification |
| AI | Humanoid robots | Autonomous operations |

### Amazon's OR Bridge
- MILP for scheduling
- Inventory optimization
- Route planning

### Future: Tesla Optimus
- 24/7 operations
- Hazardous material handling
- Reduced error rates

---

## Essay 5: CNN Architecture

### Components

| Layer | Purpose |
|-------|---------|
| **Convolutional** | Feature extraction via learned filters |
| **Pooling** | Spatial downsampling, translation invariance |
| **Activation** | Non-linearity (ReLU, sigmoid) |
| **Fully Connected** | Classification head |

### Architecture Flow
```
Input → [Conv → ReLU → Pool]×N → FC → Softmax → Classes
```

### Evaluation Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| Accuracy | (TP+TN)/Total | Balanced classes |
| Precision | TP/(TP+FP) | Minimize false positives |
| Recall | TP/(TP+FN) | Minimize false negatives |
| Specificity | TN/(TN+FP) | True negative rate |
| F1 | 2×P×R/(P+R) | Balance P and R |

---

## Quick Summary Table

| Essay | Key Concepts |
|-------|--------------|
| 1 | LinAlg→PCA/SVD, Calc→GD, Prob→Bayes, Stats→MLE |
| 2 | MLE maximizes likelihood, MAP adds prior = regularization |
| 3 | PCA = SVD on centered data, Netflix latent factors |
| 4 | Manual → OR → ML → AI/Robotics evolution |
| 5 | Conv+Pool+FC layers, Accuracy/Precision/Recall metrics |
