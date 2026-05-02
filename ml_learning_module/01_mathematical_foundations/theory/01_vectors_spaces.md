# Chapter 1: Vectors and Vector Spaces

> **Learning Duration:** 3 Days  
> **Difficulty:** Beginner  
> **Prerequisites:** Basic algebra

---

## 🎯 Learning Objectives

By the end of this chapter, you will:
- Understand what vectors are and how they represent data
- Perform vector operations (addition, scalar multiplication, dot product, cross product)
- Compute vector norms and understand their meaning
- Understand vector spaces and subspaces
- Apply vectors to machine learning problems

---

## 1.1 What Is a Vector?

### The Intuitive Definition

A **vector** is a quantity that has both **magnitude** (size) and **direction**. In machine learning, we think of vectors as **ordered lists of numbers** that represent points in multi-dimensional space.

Think of a vector as:
- A point in space (represented by its coordinates)
- A direction with a magnitude
- A list of features describing something

### Mathematical Definition

Formally, a vector is an element of a **vector space**. In ℝⁿ (n-dimensional Euclidean space), a vector is an ordered n-tuple:

$$\vec{v} = (v_1, v_2, v_3, ..., v_n)$$

Where each $v_i$ is a real number.

### Vectors in Machine Learning

In ML, vectors represent:

| Application | Vector Represents |
|-------------|-------------------|
| Text Classification | Document as bag-of-words |
| Image Recognition | Flattened pixel values |
| Recommendation Systems | User/item preferences |
| NLP | Word embeddings |

**Example:** A house price predictor might use a feature vector:

$$\vec{house} = (1500, 3, 2, 1990, 250000)$$

Where:
- $1500$ = square footage
- $3$ = number of bedrooms
- $2$ = number of bathrooms  
- $1990$ = year built
- $250000$ = current value

---

## 1.2 Geometric Interpretation

### Visualizing Vectors

In 2D (and 3D), we can draw vectors as **arrows**:

```
        y
        ↑
        |     ↗ (3, 2)
        |    /
        |   /
        |  /
        | /
        +--------------→ x
        (0,0)
```

The vector $(3, 2)$ starts at the origin $(0, 0)$ and ends at point $(3, 2)$.

### Key Geometric Properties

1. **Magnitude (Length)**: How "long" is the vector?
2. **Direction**: Which way does it point?
3. **Position**: Where is it located in space?

### Vector Addition Geometrically

Adding two vectors means "going from the first vector's tip to the second vector's tip":

```
        y
        ↑
        |     ↗ v1 + v2
        |    ↙
        |   ↗ v1
        |  /
        | /
        +------------→ x
```

This is called the **parallelogram rule**.

### Scalar Multiplication Geometrically

Multiplying by a scalar:
- **Positive**: Stretches or shrinks in same direction
- **Negative**: Reverses direction
- **Zero**: Collapses to origin

```
Original:        Scaled by 2:      Scaled by -0.5:
    ↗                   ↗↗                  ↖
```

---

## 1.3 Vector Operations

### 1.3.1 Addition

**Definition:** Add corresponding components

$$\vec{v} + \vec{w} = (v_1+w_1, v_2+w_2, ..., v_n+w_n)$$

**Example:**

```python
import numpy as np

v = np.array([1, 2, 3])
w = np.array([4, 5, 6])

result = v + w  # [5, 7, 9]
```

**Properties:**
- **Commutative**: $\vec{v} + \vec{w} = \vec{w} + \vec{v}$
- **Associative**: $(\vec{v} + \vec{w}) + \vec{u} = \vec{v} + (\vec{w} + \vec{u})$
- **Identity**: $\vec{v} + \vec{0} = \vec{v}$ (where $\vec{0}$ is the zero vector)

### 1.3.2 Scalar Multiplication

**Definition:** Multiply each component by the scalar

$$c \cdot \vec{v} = (c \cdot v_1, c \cdot v_2, ..., c \cdot v_n)$$

**Example:**

```python
v = np.array([1, 2, 3])
c = 2

result = c * v  # [2, 4, 6]
```

### 1.3.3 Dot Product (Inner Product)

**Definition:** Sum of component-wise products

$$\vec{v} \cdot \vec{w} = \sum_{i=1}^{n} v_i \cdot w_i = v_1w_1 + v_2w_2 + ... + v_nw_n$$

**Example:**

```python
v = np.array([1, 2, 3])
w = np.array([4, 5, 6])

dot = np.dot(v, w)  # 1*4 + 2*5 + 3*6 = 32
```

**Properties:**
- **Commutative**: $\vec{v} \cdot \vec{w} = \vec{w} \cdot \vec{v}$
- **Distributive**: $\vec{u} \cdot (\vec{v} + \vec{w}) = \vec{u} \cdot \vec{v} + \vec{u} \cdot \vec{w}$
- **Relation to Magnitude**: $\vec{v} \cdot \vec{v} = ||\vec{v}||^2$

**Geometric Interpretation:**

The dot product relates to the **angle** between vectors:

$$\vec{v} \cdot \vec{w} = ||\vec{v}|| \cdot ||\vec{w}|| \cdot \cos(\theta)$$

Where $\theta$ is the angle between them.

This means:
- $\vec{v} \cdot \vec{w} > 0$: Vectors point in **similar** directions
- $\vec{v} \cdot \vec{w} = 0$: Vectors are **orthogonal** (perpendicular)
- $\vec{v} \cdot \vec{w} < 0$: Vectors point in **opposite** directions

### 1.3.4 Cross Product (3D Only)

**Definition:** Produces a vector perpendicular to both input vectors

$$\vec{v} \times \vec{w} = \begin{vmatrix} \hat{i} & \hat{j} & \hat{k} \\ v_1 & v_2 & v_3 \\ w_1 & w_2 & w_3 \end{vmatrix}$$

**Example:**

```python
v = np.array([1, 0, 0])  # x-axis
w = np.array([0, 1, 0])  # y-axis

cross = np.cross(v, w)  # [0, 0, 1] - points in z-direction
```

**Properties:**
- **Anti-commutative**: $\vec{v} \times \vec{w} = -(\vec{w} \times \vec{v})$
- **Perpendicular**: Result is perpendicular to both $\vec{v}$ and $\vec{w}$
- **Magnitude**: $||\vec{v} \times \vec{w}|| = ||\vec{v}|| \cdot ||\vec{w}|| \cdot \sin(\theta)$

---

## 1.4 Vector Norms

A **norm** is a function that assigns a positive length to each vector. The most common norm is the **L2 norm** (Euclidean distance).

### 1.4.1 L2 Norm (Euclidean Norm)

$$||\vec{v}||_2 = \sqrt{v_1^2 + v_2^2 + ... + v_n^2} = \sqrt{\sum_{i=1}^{n} v_i^2}$$

**Example:**

```python
v = np.array([3, 4])

l2_norm = np.linalg.norm(v)  # 5.0 (3-4-5 triangle)
```

**Geometric Meaning:** The straight-line distance from origin to point.

### 1.4.2 L1 Norm (Manhattan Norm)

$$||\vec{v}||_1 = |v_1| + |v_2| + ... + |v_n|$$

**Example:**

```python
v = np.array([3, 4])

l1_norm = np.abs(v).sum()  # 7.0
```

**Geometric Meaning:** Distance traveled on a grid (like city blocks).

### 1.4.3 L-infinity Norm (Max Norm)

$$||\vec{v}||_{\infty} = \max(|v_1|, |v_2|, ..., |v_n|)$$

**Example:**

```python
v = np.array([3, 4, -1])

inf_norm = np.abs(v).max()  # 4.0
```

### 1.4.4 Why Do We Need Different Norms?

| Norm | Use Case | Sensitivity |
|------|----------|-------------|
| L2 | General distance, regularization | Sensitive to outliers |
| L1 | Sparse features, robust regression | Less sensitive to outliers |
| L-infinity | Chebyshev distance, worst case | Focuses on largest component |

In ML:
- **L2 regularization** penalizes large weights (prevents overfitting)
- **L1 regularization** encourages sparsity (feature selection)
- **Cosine similarity** uses normalized vectors for direction comparison

---

## 1.5 Vector Spaces

### Definition

A **vector space** is a collection of vectors where:
1. Adding any two vectors gives another vector in the space (closure under addition)
2. Multiplying by any scalar gives another vector in the space (closure under scalar multiplication)

### Standard Vector Spaces

- **ℝⁿ**: n-dimensional real numbers (our primary focus)
- **ℝ**: The real number line (1D vector space)
- **ℝ²**: The 2D plane (coordinate plane)
- **ℝ³**: 3D space

### Subspace

A **subspace** is a subset of a vector space that is itself a vector space. For a subset to be a subspace, it must contain:
1. The zero vector
2. Be closed under addition
3. Be closed under scalar multiplication

**Example:** The xy-plane in ℝ³ is a subspace:
$$\{(x, y, 0) : x, y \in \mathbb{R}\}$$

---

## 1.6 Linear Independence

### Definition

A set of vectors $\{\vec{v_1}, \vec{v_2}, ..., \vec{v_k}\}$ is **linearly independent** if no vector can be written as a combination of the others:

$$c_1\vec{v_1} + c_2\vec{v_2} + ... + c_k\vec{v_k} = \vec{0}$$

Only has the trivial solution $c_1 = c_2 = ... = c_k = 0$.

**Intuitively:** None of the vectors "point in the same direction" as a combination of others.

### Example in 2D

- $(1, 0)$ and $(0, 1)$ are linearly independent (orthogonal axes)
- $(1, 0)$ and $(2, 0)$ are linearly dependent (same direction)

### Span

The **span** of a set of vectors is all possible linear combinations:

$$span(\vec{v_1}, \vec{v_2}) = \{c_1\vec{v_1} + c_2\vec{v_2} : c_1, c_2 \in \mathbb{R}\}$$

**In ML:** The span of feature vectors determines what the model can represent.

---

## 1.7 Basis and Dimension

### Basis

A **basis** is a set of linearly independent vectors that span the entire space. Every vector in the space can be uniquely written as a combination of basis vectors.

**Standard Basis in ℝ³:**
- $\vec{e_1} = (1, 0, 0)$
- $\vec{e_2} = (0, 1, 0)$
- $\vec{e_3} = (0, 0, 1)$

### Dimension

The **dimension** of a vector space is the number of vectors in its basis.

- ℝⁿ has dimension n
- The xy-plane in ℝ³ has dimension 2

---

## 1.8 Application to Machine Learning

### 1.8.1 Feature Vectors

In ML, every data point is a vector in some feature space:

```python
# Each data point becomes a vector
house_features = [1500, 3, 2, 1990]  # 4D feature vector
customer_features = [25, 50000, 3, 1]  # 4D feature vector
```

### 1.8.2 Similarity Measures

Vectors let us measure similarity:

```python
def cosine_similarity(v1, v2):
    """Measure direction similarity between vectors"""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot / (norm1 * norm2)
```

### 1.8.3 Distance-Based Algorithms

Many ML algorithms use vector distances:

- **K-Nearest Neighbors**: Find closest vectors
- **K-Means Clustering**: Minimize distance to centroids
- **Support Vector Machines**: Maximize margin between classes

---

## 📝 Summary

### Key Takeaways

1. **Vectors** are ordered lists of numbers representing points in multi-dimensional space
2. **Operations** (add, scale, dot product) enable computation on vectors
3. **Norms** measure vector magnitude (L1, L2, L-infinity)
4. **Linear independence** determines if vectors provide unique information
5. **Basis** vectors span the space and allow unique representation

### Formulas to Remember

| Operation | Formula |
|-----------|---------|
| Dot Product | $\vec{v} \cdot \vec{w} = \sum v_i w_i$ |
| L2 Norm | $||\vec{v}||_2 = \sqrt{\sum v_i^2}$ |
| L1 Norm | $||\vec{v}||_1 = \sum |v_i|$ |
| Cosine Similarity | $\frac{\vec{v} \cdot \vec{w}}{||\vec{v}|| \cdot ||\vec{w}||}$ |

---

## 🔄 What's Next

Now that you understand vectors, proceed to:
- **Chapter 2:** Matrix Operations - How matrices transform vectors
- **Implementations:** See working code in `01_mathematical_foundations/implementations/vector_operations.py`

---

## ❓ Quick Check

1. What is the dot product of (1, 2, 3) and (4, 5, 6)?
2. Are (1, 0) and (0, 1) linearly independent?
3. What is the L2 norm of (3, 4)?
4. Why is L1 norm useful for sparse features?

*Answers at end of chapter*