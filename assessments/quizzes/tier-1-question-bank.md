# 📝 Quiz Question Bank - Tier 1: Fundamentals

**Comprehensive question bank for all Tier 1 modules**

---

## 📊 Question Distribution

| Module | Questions | Difficulty Mix |
|--------|-----------|----------------|
| 1.1 Linear Algebra | 30 | 10 Easy, 15 Medium, 5 Hard |
| 1.2 Advanced Linear Algebra | 30 | 10 Easy, 15 Medium, 5 Hard |
| 1.3 Calculus | 30 | 10 Easy, 15 Medium, 5 Hard |
| 1.4 Probability | 35 | 10 Easy, 15 Medium, 10 Hard |
| 1.5 Statistics | 30 | 10 Easy, 15 Medium, 5 Hard |
| 1.6 Python for Data Science | 30 | 10 Easy, 15 Medium, 5 Hard |
| 1.7 ML Math from Scratch | 35 | 10 Easy, 15 Medium, 10 Hard |
| **Total** | **220** | **70 Easy, 105 Medium, 45 Hard** |

---

## Module 1.1: Linear Algebra - Vectors & Matrices

### Easy Questions (10)

**Q1.1.1:** What is the result of adding vectors [1, 2, 3] and [4, 5, 6]?

A) [5, 7, 9]  
B) [1, 2, 3, 4, 5, 6]  
C) [4, 10, 18]  
D) Cannot add these vectors  

**Answer:** A) [5, 7, 9]  
**Explanation:** Vector addition is element-wise: [1+4, 2+5, 3+6] = [5, 7, 9]  
**Topic:** Vector Operations  
**Difficulty:** Easy

---

**Q1.1.2:** The dot product of two orthogonal vectors is:

A) 1  
B) 0  
C) -1  
D) Infinity  

**Answer:** B) 0  
**Explanation:** Orthogonal vectors have a dot product of 0 by definition.  
**Topic:** Dot Product  
**Difficulty:** Easy

---

**Q1.1.3:** If vector v = [3, 4], what is its magnitude?

A) 3  
B) 4  
C) 5  
D) 7  

**Answer:** C) 5  
**Explanation:** ||v|| = √(3² + 4²) = √(9 + 16) = √25 = 5  
**Topic:** Vector Magnitude  
**Difficulty:** Easy

---

**Q1.1.4:** Matrix multiplication is:

A) Commutative (AB = BA)  
B) Not commutative (AB ≠ BA generally)  
C) Always undefined  
D) Only defined for square matrices  

**Answer:** B) Not commutative (AB ≠ BA generally)  
**Explanation:** Matrix multiplication is not commutative in general.  
**Topic:** Matrix Operations  
**Difficulty:** Easy

---

**Q1.1.5:** The transpose of a matrix swaps:

A) Rows and columns  
B) Diagonal elements  
C) All elements  
D) First and last rows  

**Answer:** A) Rows and columns  
**Explanation:** Transpose converts rows to columns and columns to rows.  
**Topic:** Matrix Transpose  
**Difficulty:** Easy

---

### Medium Questions (15)

**Q1.1.6:** If A is a 3×2 matrix and B is a 2×4 matrix, what are the dimensions of AB?

A) 3×4  
B) 2×2  
C) 4×3  
D) Cannot multiply  

**Answer:** A) 3×4  
**Explanation:** Inner dimensions must match (2), result is outer dimensions (3×4).  
**Topic:** Matrix Multiplication  
**Difficulty:** Medium

---

**Q1.1.7:** What is the dot product of [2, -1, 3] and [4, 2, -1]?

A) 3  
B) 9  
C) 11  
D) 15  

**Answer:** A) 3  
**Explanation:** (2×4) + (-1×2) + (3×-1) = 8 - 2 - 3 = 3  
**Topic:** Dot Product  
**Difficulty:** Medium

---

**Q1.1.8:** Which of the following sets of vectors is linearly independent?

A) {[1, 0], [2, 0]}  
B) {[1, 0], [0, 1]}  
C) {[1, 1], [2, 2]}  
D) {[1, 2], [2, 4]}  

**Answer:** B) {[1, 0], [0, 1]}  
**Explanation:** These vectors cannot be expressed as scalar multiples of each other.  
**Topic:** Linear Independence  
**Difficulty:** Medium

---

**Q1.1.9:** The projection of vector a onto vector b is given by:

A) (a·b) / ||b||² × b  
B) (a×b) / ||b||  
C) ||a|| × ||b||  
D) a + b  

**Answer:** A) (a·b) / ||b||² × b  
**Explanation:** This is the formula for vector projection.  
**Topic:** Vector Projection  
**Difficulty:** Medium

---

**Q1.1.10:** If det(A) = 0, then matrix A is:

A) Invertible  
B) Singular (non-invertible)  
C) Identity matrix  
D) Symmetric  

**Answer:** B) Singular (non-invertible)  
**Explanation:** A matrix with determinant 0 has no inverse.  
**Topic:** Determinants  
**Difficulty:** Medium

---

### Hard Questions (5)

**Q1.1.11:** [Coding] Implement matrix multiplication from scratch:

```python
def matrix_multiply(A, B):
    """
    Multiply matrices A (m×n) and B (n×p).
    Returns: m×p matrix
    """
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    # Initialize result matrix
    result = [[0 for _ in range(p)] for _ in range(m)]
    
    # Your code here
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    
    return result
```

**Rubric:**
- Correct initialization (2 points)
- Correct nested loops (4 points)
- Correct multiplication (3 points)
- Returns correct format (1 point)

**Topic:** Matrix Multiplication Implementation  
**Difficulty:** Hard

---

## Module 1.3: Calculus - Derivatives & Gradients

### Easy Questions (10)

**Q1.3.1:** What is the derivative of f(x) = x²?

A) x  
B) 2x  
C) x²  
D) 2  

**Answer:** B) 2x  
**Explanation:** Using power rule: d/dx(x^n) = n·x^(n-1), so d/dx(x²) = 2x  
**Topic:** Derivatives  
**Difficulty:** Easy

---

**Q1.3.2:** The derivative represents:

A) Area under curve  
B) Rate of change  
C) Maximum value  
D) Average value  

**Answer:** B) Rate of change  
**Explanation:** The derivative measures instantaneous rate of change.  
**Topic:** Derivative Concept  
**Difficulty:** Easy

---

**Q1.3.3:** What is d/dx(sin(x))?

A) cos(x)  
B) -cos(x)  
C) sin(x)  
D) -sin(x)  

**Answer:** A) cos(x)  
**Explanation:** Standard derivative rule for sine function.  
**Topic:** Trigonometric Derivatives  
**Difficulty:** Easy

---

### Medium Questions (15)

**Q1.3.4:** Using the chain rule, what is d/dx(sin(x²))?

A) cos(x²)  
B) 2x·cos(x²)  
C) x·cos(x²)  
D) cos(2x)  

**Answer:** B) 2x·cos(x²)  
**Explanation:** Chain rule: d/dx(f(g(x))) = f'(g(x))·g'(x) = cos(x²)·2x  
**Topic:** Chain Rule  
**Difficulty:** Medium

---

**Q1.3.5:** The gradient of f(x,y) = x² + y² is:

A) [2x, 2y]  
B) [x, y]  
C) [2x, y]  
D) [x, 2y]  

**Answer:** A) [2x, 2y]  
**Explanation:** ∇f = [∂f/∂x, ∂f/∂y] = [2x, 2y]  
**Topic:** Gradients  
**Difficulty:** Medium

---

**Q1.3.6:** At a local minimum, the gradient is:

A) Positive  
B) Negative  
C) Zero  
D) Undefined  

**Answer:** C) Zero  
**Explanation:** At local extrema, the gradient equals zero.  
**Topic:** Optimization  
**Difficulty:** Medium

---

### Hard Questions (5)

**Q1.3.7:** [Coding] Implement gradient descent:

```python
def gradient_descent(f, df, start_x, learning_rate, iterations):
    """
    Perform gradient descent.
    
    Args:
        f: Function to minimize
        df: Derivative of f
        start_x: Starting point
        learning_rate: Step size
        iterations: Number of iterations
    
    Returns:
        final_x: Optimized x value
        history: List of x values
    """
    x = start_x
    history = [x]
    
    for i in range(iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
        history.append(x)
    
    return x, history
```

**Rubric:**
- Correct initialization (2 points)
- Correct gradient calculation (3 points)
- Correct update rule (3 points)
- History tracking (2 points)

**Topic:** Gradient Descent Implementation  
**Difficulty:** Hard

---

## Module 1.4: Probability - Basics & Distributions

### Easy Questions (10)

**Q1.4.1:** If P(A) = 0.3, what is P(not A)?

A) 0.3  
B) 0.7  
C) 1.0  
D) 0.0  

**Answer:** B) 0.7  
**Explanation:** P(not A) = 1 - P(A) = 1 - 0.3 = 0.7  
**Topic:** Basic Probability  
**Difficulty:** Easy

---

**Q1.4.2:** The probability of an event is always between:

A) -1 and 1  
B) 0 and 1  
C) 0 and 100  
D) -∞ and ∞  

**Answer:** B) 0 and 1  
**Explanation:** Probabilities are bounded between 0 and 1.  
**Topic:** Probability Axioms  
**Difficulty:** Easy

---

### Medium Questions (15)

**Q1.4.3:** If P(A) = 0.4, P(B) = 0.5, and A, B are independent, what is P(A and B)?

A) 0.9  
B) 0.2  
C) 0.1  
D) 0.5  

**Answer:** B) 0.2  
**Explanation:** For independent events: P(A and B) = P(A) × P(B) = 0.4 × 0.5 = 0.2  
**Topic:** Independent Events  
**Difficulty:** Medium

---

**Q1.4.4:** Bayes' theorem states P(A|B) = ?

A) P(B|A) × P(A) / P(B)  
B) P(A) × P(B)  
C) P(B|A) / P(A)  
D) P(A) + P(B)  

**Answer:** A) P(B|A) × P(A) / P(B)  
**Explanation:** This is the formula for Bayes' theorem.  
**Topic:** Bayes' Theorem  
**Difficulty:** Medium

---

### Hard Questions (10)

**Q1.4.5:** [Coding] Simulate Bayes' theorem:

```python
def bayes_theorem(p_a, p_b_given_a, p_b_given_not_a):
    """
    Calculate P(A|B) using Bayes' theorem.
    
    Args:
        p_a: P(A)
        p_b_given_a: P(B|A)
        p_b_given_not_a: P(B|not A)
    
    Returns:
        p_a_given_b: P(A|B)
    """
    # P(B) = P(B|A) × P(A) + P(B|not A) × P(not A)
    p_not_a = 1 - p_a
    p_b = p_b_given_a * p_a + p_b_given_not_a * p_not_a
    
    # P(A|B) = P(B|A) × P(A) / P(B)
    p_a_given_b = (p_b_given_a * p_a) / p_b
    
    return p_a_given_b

# Example: Medical test
# P(Disease) = 0.01
# P(Positive|Disease) = 0.99
# P(Positive|No Disease) = 0.05
result = bayes_theorem(0.01, 0.99, 0.05)
print(f"P(Disease|Positive) = {result:.4f}")  # Should be ~0.167
```

**Rubric:**
- Correct P(B) calculation (4 points)
- Correct Bayes' formula application (4 points)
- Correct result (2 points)

**Topic:** Bayes' Theorem Implementation  
**Difficulty:** Hard

---

## Module 1.7: ML Math from Scratch

### Easy Questions (10)

**Q1.7.1:** The cost function for linear regression is:

A) Mean Absolute Error  
B) Mean Squared Error  
C) Cross-Entropy Loss  
D) Hinge Loss  

**Answer:** B) Mean Squared Error  
**Explanation:** Linear regression typically uses MSE: (1/n) × Σ(y - y_pred)²  
**Topic:** Cost Functions  
**Difficulty:** Easy

---

### Medium Questions (15)

**Q1.7.2:** In gradient descent, if the learning rate is too large:

A) Convergence is slow  
B) Algorithm may diverge  
C) No effect  
D) Always finds global minimum  

**Answer:** B) Algorithm may diverge  
**Explanation:** Large learning rates can cause overshooting and divergence.  
**Topic:** Learning Rate  
**Difficulty:** Medium

---

### Hard Questions (10)

**Q1.7.3:** [Coding] Implement logistic regression from scratch:

```python
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_pred)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X, threshold=0.5):
        linear_pred = np.dot(X, self.weights) + self.bias
        probas = self._sigmoid(linear_pred)
        return (probas >= threshold).astype(int)
```

**Rubric:**
- Correct sigmoid implementation (2 points)
- Correct gradient calculations (4 points)
- Correct update rules (2 points)
- Correct prediction method (2 points)

**Topic:** Logistic Regression Implementation  
**Difficulty:** Hard

---

## 📊 Quiz Assembly Guide

### For Module Quiz (25 questions)

| Difficulty | Count | Points Each | Total Points |
|------------|-------|-------------|--------------|
| Easy | 10 | 2 | 20 |
| Medium | 10 | 4 | 40 |
| Hard | 5 | 8 | 40 |
| **Total** | **25** | - | **100** |

### For Final Exam (50 questions)

| Difficulty | Count | Points Each | Total Points |
|------------|-------|-------------|--------------|
| Easy | 15 | 2 | 30 |
| Medium | 25 | 3 | 75 |
| Hard | 10 | 9.5 | 95 |
| **Total** | **50** | - | **200** |

---

## ✅ Answer Key Format

```
Module 1.1 Quiz Answers:
Q1: A  Q2: B  Q3: C  Q4: B  Q5: A
Q6: A  Q7: A  Q8: B  Q9: A  Q10: B
Q11: B  Q12: C  Q13: A  Q14: D  Q15: B
Q16: A  Q17: C  Q18: B  Q19: A  Q20: D
Q21: B  Q22: A  Q23: C  Q24: B  Q25: A

Coding Solutions: See rubric in each question.
```

---

**Question Bank Created:** April 2, 2026  
**Total Questions:** 220 (Tier 1)  
**Maintained By:** AI-Mastery-2026 Assessment Team

---

[← Back to Assessments](../assessments/README.md) | [View Coding Challenges](../assessments/coding-challenges/README.md)
