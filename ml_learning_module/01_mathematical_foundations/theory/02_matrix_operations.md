# Chapter 2: Matrix Operations and Linear Transformations

> **Learning Duration:** 3 Days  
> **Difficulty:** Intermediate  
> **Prerequisites:** Chapter 1 (Vectors)

---

## 🎯 Learning Objectives

By the end of this chapter, you will:
- Understand matrices as collections of vectors or as linear transformations
- Perform matrix operations (addition, multiplication, transpose, inverse)
- Understand matrix rank and its significance
- Apply matrices to solve linear systems
- Understand how matrices transform vector spaces

---

## 2.1 What Is a Matrix?

### The Intuitive Definition

A **matrix** is a 2D array of numbers arranged in rows and columns. Think of it as:

- A **spreadsheet** of data
- A **collection of vectors** (each column or row is a vector)
- A **linear transformation** (a function that maps vectors to vectors)

### Formal Definition

An $m \times n$ matrix has $m$ rows and $n$ columns:

$$A = \begin{bmatrix} a_{11} & a_{12} & ... & a_{1n} \\ a_{21} & a_{22} & ... & a_{2n} \\ ... & ... & ... & ... \\ a_{m1} & a_{m2} & ... & a_{mn} \end{bmatrix}$$

### Matrices in Machine Learning

| Application | Matrix Represents |
|-------------|-------------------|
| Datasets | $m$ samples × $n$ features |
| Neural Networks | Weights between layers |
| Images | Pixel values (height × width) |
| Text | Document-term matrix |

**Example:** A dataset with 100 samples and 4 features:

```python
import numpy as np

# 100 samples, 4 features each
X = np.random.rand(100, 4)
print(f"Matrix shape: {X.shape}")  # (100, 4)
```

---

## 2.2 Matrix Operations

### 2.2.1 Matrix Addition

**Definition:** Add corresponding elements

$$C = A + B \implies c_{ij} = a_{ij} + b_{ij}$$

**Requirements:**
- Same dimensions ($m \times n$)

**Example:**

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = A + B  # [[6, 8], [10, 12]]
```

**Properties:**
- **Commutative**: $A + B = B + A$
- **Associative**: $(A + B) + C = A + (B + C)$
- **Identity**: $A + 0 = A$

### 2.2.2 Scalar Multiplication

**Definition:** Multiply every element by scalar

$$cA = \begin{bmatrix} ca_{11} & ca_{12} \\ ca_{21} & ca_{22} \end{bmatrix}$$

**Example:**

```python
A = np.array([[1, 2], [3, 4]])
c = 3

result = c * A  # [[3, 6], [9, 12]]
```

### 2.2.3 Matrix Transpose

**Definition:** Swap rows and columns

$$A^T = \begin{bmatrix} a_{11} & a_{21} & a_{31} \\ a_{12} & a_{22} & a_{32} \end{bmatrix} \text{ if } A = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \\ a_{31} & a_{32} \end{bmatrix}$$

**Example:**

```python
A = np.array([[1, 2, 3], [4, 5, 6]])
# Shape: (2, 3)

A_T = A.T
# Shape: (3, 2)
```

**Properties:**
- $(A^T)^T = A$
- $(A + B)^T = A^T + B^T$
- $(cA)^T = cA^T$

### 2.2.4 Matrix Multiplication

**Definition:** The dot product of rows and columns

$$C = AB \implies c_{ij} = \sum_{k=1}^{n} a_{ik} \cdot b_{kj}$$

**Visual Explanation:**

```
         B (n×p)
         ↓
    ┌───┬───┐
 A  │   │   │  → Row i of A dotted with Column j of B
    └───┴───┘
   (m×n)

Result C is (m×p)
```

**Key Point:** The **inner dimensions** must match!

- $A_{m \times n} \times B_{n \times p} = C_{m \times p}$

**Example:**

```python
A = np.array([[1, 2], [3, 4]])  # 2×2
B = np.array([[5, 6], [7, 8]])  # 2×2

C = np.matmul(A, B)
# [[1*5+2*7, 1*6+2*8],    = [[19, 22],
#  [3*5+4*7, 3*6+4*8]]       [43, 50]]
```

**Properties:**
- **Associative**: $(AB)C = A(BC)$
- **Distributive**: $A(B + C) = AB + AC$
- **NOT commutative**: $AB \neq BA$ (in general)

### 2.2.5 Element-wise Operations (Hadamard Product)

**Definition:** Multiply corresponding elements

$$C = A \circ B \implies c_{ij} = a_{ij} \cdot b_{ij}$$

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = A * B  # [[5, 12], [21, 32]]
```

---

## 2.3 Special Matrices

### 2.3.1 Identity Matrix

Has 1s on diagonal, 0s elsewhere:

$$I_n = \begin{bmatrix} 1 & 0 & ... & 0 \\ 0 & 1 & ... & 0 \\ ... & ... & ... & ... \\ 0 & 0 & ... & 1 \end{bmatrix}$$

**Property:** $AI = IA = A$

```python
I = np.eye(3)  # 3×3 identity
```

### 2.3.2 Diagonal Matrix

Non-zero elements only on diagonal:

$$D = \text{diag}(d_1, d_2, ..., d_n) = \begin{bmatrix} d_1 & 0 & ... & 0 \\ 0 & d_2 & ... & 0 \\ ... & ... & ... & ... \\ 0 & 0 & ... & d_n \end{bmatrix}$$

### 2.3.3 Zero Matrix

All elements are 0:

```python
Z = np.zeros((3, 4))  # 3×4 matrix of zeros
```

### 2.3.4 Symmetric Matrix

Equal to its transpose: $A = A^T$

**Example:**

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 5 \\ 3 & 5 & 6 \end{bmatrix}$$

### 2.3.5 Triangular Matrices

- **Upper triangular**: Elements below diagonal are 0
- **Lower triangular**: Elements above diagonal are 0

### 2.3.6 Sparse Matrices

Mostly zeros (common in ML for large datasets)

```python
from scipy import sparse

# Create sparse matrix
S = sparse.csr_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
```

---

## 2.4 Matrix Inverse

### Definition

The **inverse** of a matrix $A$, denoted $A^{-1}$, satisfies:

$$A \cdot A^{-1} = A^{-1} \cdot A = I$$

### When Does It Exist?

Only for **square, non-singular** matrices (determinant ≠ 0)

### Computing the Inverse

```python
A = np.array([[1, 2], [3, 4]])

A_inv = np.linalg.inv(A)
# [[-2. ,  1. ],
#  [ 1.5, -0.5]]

# Verify
np.matmul(A, A_inv)  # Should be close to identity
```

### Properties

- $(A^{-1})^{-1} = A$
- $(AB)^{-1} = B^{-1}A^{-1}$
- $(A^T)^{-1} = (A^{-1})^T$

### Why We Need It

Solving linear systems: $Ax = b$

$$x = A^{-1}b$$

(This is how we solve systems of linear equations!)

---

## 2.5 Matrix Rank

### Definition

The **rank** of a matrix is the number of linearly independent rows (or columns).

- $\text{rank}(A) \leq \min(m, n)$
- Full rank: $\text{rank}(A) = \min(m, n)$

### Computing Rank

```python
A = np.array([[1, 2, 3], [2, 4, 6], [1, 1, 1]])
rank = np.linalg.matrix_rank(A)  # 2 (third row is combination of others)
```

### Types of Matrices by Rank

| Type | Condition | Meaning |
|------|-----------|---------|
| Full Rank | $\text{rank} = \min(m,n)$ | No redundancy |
| Rank Deficient | $\text{rank} < \min(m,n)$ | Redundant information |
| Rank 0 | All zeros | Empty |

### Why Rank Matters in ML

- **Collinearity**: Features that are linear combinations cause rank deficiency
- **Dimensionality Reduction**: Keep only independent components (PCA)
- **Model Complexity**: Full rank needed for unique solutions

---

## 2.6 Determinant

### Definition

The **determinant** is a scalar value computed from square matrices.

For $2 \times 2$:
$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

For larger matrices, it's computed recursively.

### Computing Determinant

```python
A = np.array([[1, 2], [3, 4]])
det = np.linalg.det(A)  # -2.0
```

### Properties

- $\det(AB) = \det(A) \cdot \det(B)$
- $\det(A^T) = \det(A)$
- $\det(A^{-1}) = 1/\det(A)$
- $\det(I) = 1$

### What It Tells Us

- $\det(A) = 0$: Matrix is **singular** (no inverse, rank deficient)
- $\det(A) \neq 0$: Matrix is **invertible** (full rank)

---

## 2.7 Linear Transformations

### The Key Insight

A matrix represents a **linear transformation** - a function that:
1. Preserves vector addition: $T(\vec{u} + \vec{v}) = T(\vec{u}) + T(\vec{v})$
2. Preserves scalar multiplication: $T(c\vec{v}) = cT(\vec{v})$

### Matrix as Transformation

Multiplication by matrix transforms vectors:

$$A\vec{x} = \vec{y}$$

The vector $\vec{x}$ is transformed into $\vec{y}$.

### Common 2D Transformations

| Transformation | Matrix | Effect |
|----------------|--------|--------|
| Identity | $\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ | No change |
| Scaling | $\begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$ | Stretch/shrink |
| Rotation (θ) | $\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ | Rotate |
| Shear | $\begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$ | Skew |
| Reflection | $\begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}$ | Flip |

### Example: Rotation

```python
import matplotlib.pyplot as plt

# Rotation matrix for 45 degrees
theta = np.radians(45)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

# Original vector
v = np.array([1, 0])

# Rotated vector
v_rotated = R @ v
```

---

## 2.8 Matrix Decompositions

### Why Decompose?

Breaking a matrix into parts helps us:
- Solve systems faster
- Understand data structure
- Reduce dimensionality
- Find patterns

### 2.8.1 LU Decomposition

$$A = LU$$

Where:
- $L$ = lower triangular matrix
- $U$ = upper triangular matrix

**Use:** Solving linear systems, computing determinants

```python
from scipy.linalg import lu

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
P, L, U = lu(A)
```

### 2.8.2 Eigenvalue Decomposition

$$A = V \Lambda V^{-1}$$

Where:
- $V$ = eigenvectors (as columns)
- $\Lambda$ = diagonal matrix of eigenvalues

**Use:** PCA, powers of matrices, Markov chains

```python
eigenvalues, eigenvectors = np.linalg.eig(A)
```

### 2.8.3 Singular Value Decomposition (SVD)

$$A = U \Sigma V^T$$

Where:
- $U$, $V$ = orthogonal matrices (rotations)
- $\Sigma$ = diagonal matrix of singular values

**Use:** Dimensionality reduction (PCA), image compression, recommendation systems

```python
U, S, Vt = np.linalg.svd(A)
```

---

## 2.9 Application to Machine Learning

### 2.9.1 Linear Regression (Matrices)

The normal equation: $\theta = (X^T X)^{-1} X^T y$

```python
# Solve linear regression with matrices
X_b = np.c_[np.ones((m, 1)), X]  # Add bias term
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
```

### 2.9.2 Neural Network Weights

Matrices store weights between layers:

```python
# Forward pass through one layer
Z = X @ W + b  # X: (batch, input_dim), W: (input_dim, hidden)
```

### 2.9.3 Principal Component Analysis (PCA)

Uses eigendecomposition:

```python
# PCA using SVD
U, S, Vt = np.linalg.svd(X_centered)
components = Vt[:k]  # Top k principal components
```

### 2.9.4 Data Representation

Many ML algorithms can be expressed as matrix operations, enabling:
- **Vectorization**: Process entire datasets at once
- **GPU Acceleration**: Matrix ops are highly parallelizable
- **Efficient Computation**: Optimized linear algebra libraries

---

## 📝 Summary

### Key Takeaways

1. **Matrices** represent collections of data or linear transformations
2. **Matrix multiplication** requires matching inner dimensions
3. **Matrix inverse** solves linear systems (when it exists)
4. **Rank** tells us about redundancy in data
5. **Matrix decompositions** (LU, Eig, SVD) reveal structure
6. **Transformations** rotate, scale, shear, and project vectors

### Formulas to Remember

| Operation | Formula |
|-----------|---------|
| Matrix Multiply | $c_{ij} = \sum_k a_{ik} b_{kj}$ |
| Determinant (2×2) | $ad - bc$ |
| Inverse exists | $\det(A) \neq 0$ |
| Rank | Max linearly independent rows/cols |
| Normal Eq | $\theta = (X^T X)^{-1} X^T y$ |

---

## 🔄 What's Next

- **Chapter 3:** Eigenvalues and Eigenvectors
- **Chapter 3:** SVD and Matrix Decompositions
- **Implementations:** See `matrix_operations.py`

---

## ❓ Quick Check

1. What is the shape of $A_{3 \times 4} \times B_{4 \times 5}$?
2. What is the determinant of $\begin{bmatrix} 2 & 1 \\ 4 & 2 \end{bmatrix}$?
3. Does $\begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$ have an inverse? Why?
4. What does the rank tell you about linear independence?

*Answers at end of chapter*