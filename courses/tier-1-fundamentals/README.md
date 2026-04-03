# 🎓 Tier 1: Fundamentals

**Build strong mathematical and programming foundations** for machine learning.

---

## 📋 Overview

**Duration:** 8-10 weeks  
**Commitment:** 10-12 hours/week  
**Prerequisites:** Tier 0 or equivalent knowledge  
**Outcome:** Ready for classical ML and deep learning

---

## 🎯 Learning Objectives

By the end of Tier 1, you will be able to:

- ✅ Perform linear algebra operations (vectors, matrices, eigenvalues, SVD)
- ✅ Calculate probabilities and apply statistical inference
- ✅ Use NumPy, Pandas, and Matplotlib for data analysis
- ✅ Implement gradient descent from scratch
- ✅ Understand the mathematical foundations of ML algorithms

---

## 📚 Module List

### Module 1.1: Linear Algebra - Vectors & Matrices

**Duration:** 10 hours  
**Difficulty:** ⭐⭐☆☆☆

#### What You'll Learn

- Vector representation and operations
- Matrix operations (addition, multiplication, transpose)
- Dot product and geometric interpretation
- Vector spaces and subspaces
- Linear combinations and independence

#### Topics Covered

1. **Vectors** (3 hours)
   - Definition and notation
   - Vector addition and scalar multiplication
   - Magnitude and direction
   - Unit vectors
   - Applications in ML (feature vectors)

2. **Matrices** (3 hours)
   - Matrix notation and dimensions
   - Matrix addition and subtraction
   - Scalar multiplication
   - Matrix multiplication
   - Identity and zero matrices
   - Transpose operation

3. **Dot Product** (2 hours)
   - Algebraic definition
   - Geometric interpretation
   - Angle between vectors
   - Orthogonality
   - Projection

4. **Vector Spaces** (2 hours)
   - Definition of vector space
   - Subspaces
   - Span and linear independence
   - Basis and dimension

#### Hands-On Lab

**Exercise:** Build a Vector Operations Library

```python
import numpy as np

class VectorOperations:
    """Implement vector operations from scratch."""
    
    @staticmethod
    def dot_product(v1, v2):
        """
        Calculate dot product of two vectors.
        
        Args:
            v1: First vector (list or np.array)
            v2: Second vector
        
        Returns:
            Scalar value of dot product
        """
        # Your implementation here
        pass
    
    @staticmethod
    def vector_addition(v1, v2):
        """Add two vectors."""
        pass
    
    @staticmethod
    def scalar_multiply(scalar, vector):
        """Multiply vector by scalar."""
        pass
    
    @staticmethod
    def magnitude(vector):
        """Calculate vector magnitude (L2 norm)."""
        pass
    
    @staticmethod
    def normalize(vector):
        """Normalize vector to unit length."""
        pass
    
    @staticmethod
    def angle_between(v1, v2):
        """Calculate angle between two vectors (in radians)."""
        pass
    
    @staticmethod
    def projection(v1, v2):
        """Project v1 onto v2."""
        pass


# Test your implementation
if __name__ == "__main__":
    v1 = np.array([3, 4])
    v2 = np.array([1, 2])
    
    print(f"Dot Product: {VectorOperations.dot_product(v1, v2)}")
    # Expected: 11
    
    print(f"Magnitude of v1: {VectorOperations.magnitude(v1)}")
    # Expected: 5.0
    
    print(f"Normalized v1: {VectorOperations.normalize(v1)}")
    # Expected: [0.6, 0.8]
```

#### Quiz

**Sample Questions:**

**Q1.** What is the dot product of [2, 3, 1] and [4, -1, 2]?

A) 7  
B) 9  
C) 11  
D) 13  

**Answer:** C) 11  
**Explanation:** (2×4) + (3×-1) + (1×2) = 8 - 3 + 2 = 7... wait, let me recalculate: 8 - 3 + 2 = 7. Answer is A) 7

**Q2.** If two vectors are orthogonal, their dot product is:

A) 1  
B) 0  
C) -1  
D) Infinity  

**Answer:** B) 0

**Q3.** [Coding] Implement matrix multiplication without using NumPy:

```python
def matrix_multiply(A, B):
    """
    Multiply matrices A and B.
    A: m×n matrix
    B: n×p matrix
    Returns: m×p matrix
    """
    # Your code here
    pass
```

#### Resources

- **Textbook:** "Linear Algebra and Its Applications" by Gilbert Strang
- **Video:** [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (3Blue1Brown)
- **Interactive:** [Khan Academy Linear Algebra](https://www.khanacademy.org/math/linear-algebra)

---

### Module 1.2: Advanced Linear Algebra - Eigenvalues & SVD

**Duration:** 10 hours  
**Difficulty:** ⭐⭐⭐☆☆

#### What You'll Learn

- Determinants and inverses
- Eigenvalues and eigenvectors
- Diagonalization
- Singular Value Decomposition (SVD)
- Applications in ML (PCA, dimensionality reduction)

#### Topics Covered

1. **Determinants** (2 hours)
   - Definition and properties
   - Calculation for 2×2 and 3×3 matrices
   - Geometric interpretation (volume scaling)
   - Relationship to invertibility

2. **Matrix Inverse** (2 hours)
   - Definition and properties
   - Calculation methods
   - Applications in solving linear systems

3. **Eigenvalues & Eigenvectors** (3 hours)
   - Definition: Av = λv
   - Finding eigenvalues (characteristic equation)
   - Finding eigenvectors
   - Geometric interpretation
   - Diagonalization

4. **Singular Value Decomposition** (3 hours)
   - SVD theorem: A = UΣV^T
   - Computing SVD
   - Geometric interpretation
   - Applications in ML

#### Hands-On Lab

**Exercise:** Implement PCA from Scratch using SVD

```python
import numpy as np

class PCAFromScratch:
    """Implement Principal Component Analysis using SVD."""
    
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
    
    def fit(self, X):
        """
        Fit PCA on data X.
        
        Args:
            X: Data matrix (n_samples, n_features)
        """
        # Step 1: Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Step 2: Compute SVD
        # X = U * S * V^T
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Step 3: Select top n_components
        self.components = Vt[:self.n_components]
        
        # Step 4: Calculate explained variance
        n_samples = X.shape[0]
        self.explained_variance = (S[:self.n_components] ** 2) / (n_samples - 1)
        
        return self
    
    def transform(self, X):
        """
        Transform data to principal components.
        
        Args:
            X: Data matrix (n_samples, n_features)
        
        Returns:
            Transformed data (n_samples, n_components)
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def reconstruction_error(self, X):
        """Calculate reconstruction error."""
        X_centered = X - self.mean
        X_transformed = np.dot(X_centered, self.components.T)
        X_reconstructed = np.dot(X_transformed, self.components)
        error = np.mean((X_centered - X_reconstructed) ** 2)
        return error


# Test on sample data
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    
    # Apply PCA
    pca = PCAFromScratch(n_components=2)
    X_transformed = pca.fit_transform(X)
    
    print(f"Original shape: {X.shape}")
    print(f"Transformed shape: {X_transformed.shape}")
    print(f"Explained variance: {pca.explained_variance}")
    print(f"Total variance explained: {np.sum(pca.explained_variance) / np.var(X) * 100:.2f}%")
```

#### Quiz

**Sample Questions:**

**Q1.** If λ is an eigenvalue of matrix A, then:

A) Av = λ for some vector v  
B) Av = λv for some non-zero vector v  
C) A = λI  
D) det(A) = λ  

**Answer:** B) Av = λv for some non-zero vector v

**Q2.** In SVD, if A is m×n, what are the dimensions of U?

A) m×m  
B) m×n  
C) n×n  
D) n×m  

**Answer:** A) m×m (or m×min(m,n) for reduced SVD)

**Q3.** The number of non-zero eigenvalues equals:

A) The determinant of A  
B) The trace of A  
C) The rank of A  
D) The dimension of A  

**Answer:** C) The rank of A

---

### Module 1.3: Calculus - Derivatives & Gradients

**Duration:** 10 hours  
**Difficulty:** ⭐⭐⭐☆☆

#### What You'll Learn

- Limits and continuity
- Derivatives and rules
- Chain rule
- Partial derivatives
- Gradients and gradient descent
- Applications in optimization

#### Topics Covered

1. **Limits & Continuity** (1.5 hours)
   - Concept of limits
   - Continuity
   - Importance in calculus

2. **Derivatives** (2.5 hours)
   - Definition as rate of change
   - Power rule
   - Product rule
   - Quotient rule
   - Derivatives of common functions

3. **Chain Rule** (2 hours)
   - Composite functions
   - Chain rule formula
   - Applications in backpropagation

4. **Partial Derivatives** (2 hours)
   - Functions of multiple variables
   - Partial derivative notation
   - Computing partial derivatives

5. **Gradients** (2 hours)
   - Gradient vector
   - Geometric interpretation
   - Direction of steepest ascent
   - Gradient descent algorithm

#### Hands-On Lab

**Exercise:** Visualize Gradient Descent

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """Function to minimize: f(x) = x² + 5x + 6"""
    return x**2 + 5*x + 6

def df(x):
    """Derivative: f'(x) = 2x + 5"""
    return 2*x + 5

def gradient_descent(start_x, learning_rate, iterations):
    """
    Perform gradient descent.
    
    Args:
        start_x: Starting x value
        learning_rate: Step size (alpha)
        iterations: Number of iterations
    
    Returns:
        final_x: x value at minimum
        history: List of (x, f(x)) tuples
    """
    x = start_x
    history = [(x, f(x))]
    
    for i in range(iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
        history.append((x, f(x)))
    
    return x, history

# Visualize
if __name__ == "__main__":
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Function and minimum
    x_vals = np.linspace(-10, 5, 100)
    y_vals = f(x_vals)
    
    ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x² + 5x + 6')
    ax1.axvline(x=-2.5, color='r', linestyle='--', label='Minimum at x = -2.5')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Function to Minimize')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Gradient descent path
    final_x, history = gradient_descent(start_x=10, learning_rate=0.1, iterations=50)
    
    history_x = [h[0] for h in history]
    history_y = [h[1] for h in history]
    
    ax2.plot(history_x, history_y, 'go-', label='Gradient Descent Path')
    ax2.plot(x_vals, y_vals, 'b-', alpha=0.3)
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.set_title('Gradient Descent Optimization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gradient_descent_visualization.png', dpi=300)
    print(f"Optimization complete! Final x: {final_x:.4f}, Final f(x): {final_x:.4f}")
    print(f"True minimum: x = -2.5, f(x) = {f(-2.5):.4f}")
```

#### Quiz

**Sample Questions:**

**Q1.** What is the derivative of f(x) = x³?

A) 3x²  
B) x²  
C) 3x  
D) x³/3  

**Answer:** A) 3x²

**Q2.** The gradient points in the direction of:

A) Steepest descent  
B) Steepest ascent  
C) No change  
D) Minimum value  

**Answer:** B) Steepest ascent

**Q3.** [Coding] Implement the chain rule for f(g(x)):

```python
def chain_rule(f, g, df, dg, x):
    """
    Calculate derivative of f(g(x)) at point x.
    
    Args:
        f: Outer function
        g: Inner function
        df: Derivative of f
        dg: Derivative of g
        x: Point to evaluate
    
    Returns:
        Derivative of f(g(x)) at x
    """
    # Your code here
    pass
```

---

### Module 1.4: Probability - Basics & Distributions

**Duration:** 12 hours  
**Difficulty:** ⭐⭐⭐☆☆

#### What You'll Learn

- Basic probability rules
- Conditional probability
- Bayes' theorem
- Random variables
- Common distributions (Normal, Binomial, Poisson)
- Expected value and variance

#### Topics Covered

1. **Basic Probability** (2 hours)
   - Sample space and events
   - Probability axioms
   - Addition rule
   - Multiplication rule

2. **Conditional Probability** (2 hours)
   - Definition and notation
   - Independence
   - Multiplication rule

3. **Bayes' Theorem** (2 hours)
   - Statement and proof
   - Applications
   - Naive Bayes classifier preview

4. **Random Variables** (2 hours)
   - Discrete vs continuous
   - Probability mass/density functions
   - Cumulative distribution

5. **Distributions** (3 hours)
   - Normal (Gaussian) distribution
   - Binomial distribution
   - Poisson distribution
   - Uniform distribution

6. **Expected Value & Variance** (1 hour)
   - Mean and expectation
   - Variance and standard deviation
   - Properties

#### Hands-On Lab

**Exercise:** Simulate Probability Distributions

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class ProbabilitySimulator:
    """Simulate and visualize probability distributions."""
    
    @staticmethod
    def plot_normal_distribution(mu=0, sigma=1, n_samples=10000):
        """Plot normal distribution with histogram."""
        samples = np.random.normal(mu, sigma, n_samples)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        ax1.hist(samples, bins=50, density=True, alpha=0.7, label='Samples')
        
        # Theoretical PDF
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
        pdf = stats.norm.pdf(x, mu, sigma)
        ax1.plot(x, pdf, 'r-', linewidth=2, label=f'Normal(μ={mu}, σ={sigma})')
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('Normal Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(samples, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot')
        
        plt.tight_layout()
        plt.savefig(f'normal_distribution_{mu}_{sigma}.png', dpi=300)
        plt.show()
    
    @staticmethod
    def simulate_bayes_theorem():
        """
        Simulate Bayes' theorem example.
        
        Example: Disease testing
        - P(Disease) = 0.01 (1% prevalence)
        - P(Positive|Disease) = 0.99 (99% sensitivity)
        - P(Positive|No Disease) = 0.05 (5% false positive rate)
        
        What is P(Disease|Positive)?
        """
        # Parameters
        p_disease = 0.01
        p_positive_given_disease = 0.99
        p_positive_given_no_disease = 0.05
        
        # Bayes' theorem
        p_positive = (p_positive_given_disease * p_disease + 
                     p_positive_given_no_disease * (1 - p_disease))
        
        p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive
        
        print(f"P(Disease) = {p_disease}")
        print(f"P(Positive|Disease) = {p_positive_given_disease}")
        print(f"P(Positive|No Disease) = {p_positive_given_no_disease}")
        print(f"\nP(Positive) = {p_positive:.4f}")
        print(f"P(Disease|Positive) = {p_disease_given_positive:.4f}")
        print(f"\nEven with a positive test, probability of disease is only {p_disease_given_positive*100:.2f}%")
        
        return p_disease_given_positive
    
    @staticmethod
    def compare_distributions():
        """Compare different probability distributions."""
        x = np.linspace(-5, 5, 1000)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Normal distribution
        normal_pdf = stats.norm.pdf(x, 0, 1)
        axes[0, 0].plot(x, normal_pdf, 'b-', linewidth=2)
        axes[0, 0].set_title('Normal Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Binomial distribution
        n, p = 20, 0.5
        x_binom = np.arange(0, 21)
        binom_pmf = stats.binom.pmf(x_binom, n, p)
        axes[0, 1].bar(x_binom, binom_pmf, alpha=0.7)
        axes[0, 1].set_title(f'Binomial Distribution (n={n}, p={p})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Poisson distribution
        mu = 3
        x_poisson = np.arange(0, 15)
        poisson_pmf = stats.poisson.pmf(x_poisson, mu)
        axes[1, 0].bar(x_poisson, poisson_pmf, alpha=0.7)
        axes[1, 0].set_title(f'Poisson Distribution (λ={mu})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Uniform distribution
        uniform_pdf = stats.uniform.pdf(x, -2, 4)
        axes[1, 1].plot(x, uniform_pdf, 'g-', linewidth=2)
        axes[1, 1].set_title('Uniform Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('probability_distributions_comparison.png', dpi=300)
        plt.show()


# Run simulations
if __name__ == "__main__":
    # Normal distribution
    ProbabilitySimulator.plot_normal_distribution()
    
    # Bayes' theorem
    ProbabilitySimulator.simulate_bayes_theorem()
    
    # Compare distributions
    ProbabilitySimulator.compare_distributions()
```

#### Quiz

**Sample Questions:**

**Q1.** If P(A) = 0.3 and P(B) = 0.4, and A and B are independent, what is P(A and B)?

A) 0.7  
B) 0.12  
C) 0.1  
D) 0.5  

**Answer:** B) 0.12  
**Explanation:** P(A and B) = P(A) × P(B) = 0.3 × 0.4 = 0.12

**Q2.** Bayes' theorem states:

A) P(A|B) = P(B|A) × P(A) / P(B)  
B) P(A|B) = P(A) × P(B)  
C) P(A|B) = P(B|A) / P(A)  
D) P(A|B) = P(A) + P(B)  

**Answer:** A) P(A|B) = P(B|A) × P(A) / P(B)

**Q3.** The normal distribution is characterized by:

A) Mean only  
B) Variance only  
C) Mean and variance  
D) Mean and median  

**Answer:** C) Mean and variance

---

### Module 1.5: Statistics - Inference & Testing

**Duration:** 10 hours  
**Difficulty:** ⭐⭐⭐☆☆

#### What You'll Learn

- Descriptive statistics
- Sampling distributions
- Confidence intervals
- Hypothesis testing
- t-tests and ANOVA
- Chi-square tests
- Statistical significance vs practical significance

#### Topics Covered

1. **Descriptive Statistics** (2 hours)
   - Mean, median, mode
   - Variance and standard deviation
   - Skewness and kurtosis
   - Five-number summary

2. **Sampling Distributions** (2 hours)
   - Central Limit Theorem
   - Standard error
   - Sampling distribution of mean

3. **Confidence Intervals** (2 hours)
   - Concept and interpretation
   - CI for mean
   - CI for proportion

4. **Hypothesis Testing** (3 hours)
   - Null and alternative hypotheses
   - p-values
   - Type I and Type II errors
   - Statistical significance

5. **Common Tests** (1 hour)
   - t-tests
   - ANOVA
   - Chi-square tests

#### Hands-On Lab

**Exercise:** A/B Testing Analysis

```python
import numpy as np
from scipy import stats

class ABTestAnalyzer:
    """Analyze A/B test results."""
    
    def __init__(self, group_a, group_b):
        """
        Initialize with data from both groups.
        
        Args:
            group_a: List of measurements for control group
            group_b: List of measurements for treatment group
        """
        self.group_a = np.array(group_a)
        self.group_b = np.array(group_b)
    
    def descriptive_stats(self):
        """Calculate descriptive statistics for both groups."""
        stats_dict = {
            'Group A': {
                'mean': np.mean(self.group_a),
                'median': np.median(self.group_a),
                'std': np.std(self.group_a),
                'n': len(self.group_a)
            },
            'Group B': {
                'mean': np.mean(self.group_b),
                'median': np.median(self.group_b),
                'std': np.std(self.group_b),
                'n': len(self.group_b)
            }
        }
        return stats_dict
    
    def two_sample_ttest(self):
        """
        Perform two-sample t-test.
        
        Returns:
            t_statistic: t-value
            p_value: p-value
        """
        t_stat, p_val = stats.ttest_ind(self.group_a, self.group_b)
        return t_stat, p_val
    
    def confidence_interval(self, confidence=0.95):
        """
        Calculate confidence interval for difference in means.
        
        Args:
            confidence: Confidence level (default 0.95)
        
        Returns:
            (lower_bound, upper_bound)
        """
        diff = np.mean(self.group_b) - np.mean(self.group_a)
        
        # Standard error of difference
        se = np.sqrt(np.var(self.group_a)/len(self.group_a) + 
                    np.var(self.group_b)/len(self.group_b))
        
        # Degrees of freedom (Welch's approximation)
        df_num = (np.var(self.group_a)/len(self.group_a) + 
                 np.var(self.group_b)/len(self.group_b))**2
        df_den = ((np.var(self.group_a)/len(self.group_a))**2 / (len(self.group_a)-1) +
                 (np.var(self.group_b)/len(self.group_b))**2 / (len(self.group_b)-1))
        df = df_num / df_den
        
        # Critical value
        t_crit = stats.t.ppf((1 + confidence) / 2, df)
        
        # Confidence interval
        margin = t_crit * se
        lower = diff - margin
        upper = diff + margin
        
        return lower, upper
    
    def analyze(self):
        """Perform complete A/B test analysis."""
        print("=" * 60)
        print("A/B TEST ANALYSIS")
        print("=" * 60)
        
        # Descriptive stats
        desc = self.descriptive_stats()
        print("\nDESCRIPTIVE STATISTICS:")
        for group, stats_dict in desc.items():
            print(f"\n{group}:")
            print(f"  Mean: {stats_dict['mean']:.4f}")
            print(f"  Std Dev: {stats_dict['std']:.4f}")
            print(f"  Sample Size: {stats_dict['n']}")
        
        # T-test
        t_stat, p_val = self.two_sample_ttest()
        print(f"\nT-TEST RESULTS:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_val:.6f}")
        
        if p_val < 0.05:
            print(f"  Result: STATISTICALLY SIGNIFICANT (p < 0.05)")
        else:
            print(f"  Result: NOT statistically significant (p >= 0.05)")
        
        # Confidence interval
        lower, upper = self.confidence_interval()
        print(f"\n95% CONFIDENCE INTERVAL for difference:")
        print(f"  ({lower:.4f}, {upper:.4f})")
        
        if lower > 0:
            print(f"  Interpretation: Group B is significantly BETTER than Group A")
        elif upper < 0:
            print(f"  Interpretation: Group B is significantly WORSE than Group A")
        else:
            print(f"  Interpretation: No significant difference between groups")
        
        print("=" * 60)


# Example usage
if __name__ == "__main__":
    # Simulate A/B test data
    np.random.seed(42)
    
    # Group A (control): conversion times
    group_a = np.random.normal(100, 15, 200)
    
    # Group B (treatment): conversion times with improvement
    group_b = np.random.normal(95, 15, 200)
    
    # Analyze
    analyzer = ABTestAnalyzer(group_a, group_b)
    analyzer.analyze()
```

#### Quiz

**Sample Questions:**

**Q1.** The Central Limit Theorem states that:

A) All distributions are normal  
B) Sample means approach normal distribution as n increases  
C) Variance decreases with sample size  
D) Mean equals median  

**Answer:** B) Sample means approach normal distribution as n increases

**Q2.** A p-value less than 0.05 indicates:

A) The null hypothesis is true  
B) The null hypothesis is false  
C) Strong evidence against the null hypothesis  
D) Weak evidence against the null hypothesis  

**Answer:** C) Strong evidence against the null hypothesis

**Q3.** Type I error occurs when:

A) We reject a true null hypothesis  
B) We fail to reject a false null hypothesis  
C) We accept a true null hypothesis  
D) We reject a false null hypothesis  

**Answer:** A) We reject a true null hypothesis

---

### Module 1.6: Python for Data Science

**Duration:** 15 hours  
**Difficulty:** ⭐⭐☆☆☆

#### What You'll Learn

- NumPy for numerical computing
- Pandas for data manipulation
- Matplotlib and Seaborn for visualization
- Data loading and cleaning
- Exploratory data analysis

#### Topics Covered

1. **NumPy** (4 hours)
   - Arrays and operations
   - Broadcasting
   - Indexing and slicing
   - Linear algebra functions

2. **Pandas** (5 hours)
   - Series and DataFrames
   - Data selection and filtering
   - GroupBy operations
   - Merging and joining
   - Time series

3. **Data Visualization** (4 hours)
   - Matplotlib basics
   - Seaborn statistical plots
   - Customizing plots
   - Best practices

4. **Data Cleaning** (2 hours)
   - Handling missing data
   - Removing duplicates
   - Data type conversion
   - String operations

#### Hands-On Lab

**Exercise:** Exploratory Data Analysis Project

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class EDAProject:
    """Complete exploratory data analysis project."""
    
    def __init__(self, filepath=None):
        """Initialize with dataset."""
        if filepath:
            self.df = pd.read_csv(filepath)
        else:
            # Generate sample dataset
            self.df = self._generate_sample_data()
    
    def _generate_sample_data(self):
        """Generate sample e-commerce dataset."""
        np.random.seed(42)
        n_customers = 1000
        
        data = {
            'customer_id': range(1, n_customers + 1),
            'age': np.random.randint(18, 70, n_customers),
            'income': np.random.normal(50000, 15000, n_customers),
            'spending_score': np.random.randint(1, 100, n_customers),
            'visits': np.random.poisson(5, n_customers),
            'purchases': np.random.poisson(2, n_customers),
            'churned': np.random.choice([0, 1], n_customers, p=[0.8, 0.2])
        }
        
        df = pd.DataFrame(data)
        
        # Add some missing values
        df.loc[np.random.choice(n_customers, 50), 'income'] = np.nan
        
        return df
    
    def initial_exploration(self):
        """Perform initial data exploration."""
        print("=" * 60)
        print("INITIAL DATA EXPLORATION")
        print("=" * 60)
        
        print("\n1. Dataset Shape:")
        print(f"   Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")
        
        print("\n2. First 5 Rows:")
        print(self.df.head())
        
        print("\n3. Data Types:")
        print(self.df.dtypes)
        
        print("\n4. Missing Values:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({'Missing': missing, 'Percent': missing_pct})
        print(missing_df[missing_df['Missing'] > 0])
        
        print("\n5. Statistical Summary:")
        print(self.df.describe())
    
    def visualize_distributions(self):
        """Visualize distributions of numerical variables."""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols[:4]):
            axes[i].hist(self.df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('distributions.png', dpi=300)
        plt.show()
    
    def visualize_relationships(self):
        """Visualize relationships between variables."""
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        corr_matrix = self.df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300)
        plt.show()
        
        # Pairplot for key variables
        sns.pairplot(self.df[['age', 'income', 'spending_score', 'churned']], 
                    hue='churned', diag_kind='hist')
        plt.savefig('pairplot.png', dpi=300)
        plt.show()
    
    def analyze_churn(self):
        """Analyze customer churn."""
        print("\n" + "=" * 60)
        print("CHURN ANALYSIS")
        print("=" * 60)
        
        churn_rate = self.df['churned'].mean() * 100
        print(f"\nOverall Churn Rate: {churn_rate:.2f}%")
        
        # Churn by age group
        self.df['age_group'] = pd.cut(self.df['age'], 
                                      bins=[0, 25, 35, 50, 100],
                                      labels=['18-25', '26-35', '36-50', '50+'])
        
        churn_by_age = self.df.groupby('age_group')['churned'].mean() * 100
        print("\nChurn Rate by Age Group:")
        print(churn_by_age)
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Churn by age
        churn_by_age.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Churn Rate by Age Group')
        axes[0].set_ylabel('Churn Rate (%)')
        axes[0].grid(True, alpha=0.3)
        
        # Income vs Spending Score with churn
        scatter = axes[1].scatter(self.df['income'], self.df['spending_score'],
                                 c=self.df['churned'], alpha=0.5, cmap='coolwarm')
        axes[1].set_xlabel('Income')
        axes[1].set_ylabel('Spending Score')
        axes[1].set_title('Income vs Spending Score (colored by churn)')
        plt.colorbar(scatter, ax=axes[1], label='Churned')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('churn_analysis.png', dpi=300)
        plt.show()
    
    def run_complete_eda(self):
        """Run complete EDA pipeline."""
        self.initial_exploration()
        self.visualize_distributions()
        self.visualize_relationships()
        self.analyze_churn()
        
        print("\n" + "=" * 60)
        print("EDA COMPLETE!")
        print("=" * 60)
        print("\nGenerated Visualizations:")
        print("  - distributions.png")
        print("  - correlation_matrix.png")
        print("  - pairplot.png")
        print("  - churn_analysis.png")


# Run EDA
if __name__ == "__main__":
    eda = EDAProject()
    eda.run_complete_eda()
```

#### Quiz

**Sample Questions:**

**Q1.** In NumPy, broadcasting allows:

A) Sending data to multiple processors  
B) Operating on arrays of different shapes  
C) Converting arrays to lists  
D) Saving arrays to disk  

**Answer:** B) Operating on arrays of different shapes

**Q2.** Which Pandas method is used to handle missing values?

A) dropna() only  
B) fillna() only  
C) Both dropna() and fillna()  
D) None of the above  

**Answer:** C) Both dropna() and fillna()

**Q3.** [Coding] Filter a DataFrame to show only rows where age > 30 and income > 50000:

```python
import pandas as pd

# Given DataFrame df with columns 'age' and 'income'
# Your code here:
filtered_df = ...
```

---

### Module 1.7: ML Math from Scratch - Gradient Descent

**Duration:** 15 hours  
**Difficulty:** ⭐⭐⭐⭐☆

#### What You'll Learn

- Linear regression from scratch
- Gradient descent variants (SGD, Mini-batch)
- Regularization (Ridge, Lasso)
- Logistic regression implementation
- Multi-class classification

#### Topics Covered

1. **Linear Regression** (4 hours)
   - Hypothesis function
   - Cost function (MSE)
   - Gradient calculation
   - Implementation from scratch

2. **Gradient Descent Variants** (4 hours)
   - Batch gradient descent
   - Stochastic gradient descent
   - Mini-batch gradient descent
   - Learning rate scheduling

3. **Regularization** (3 hours)
   - Overfitting problem
   - Ridge (L2) regularization
   - Lasso (L1) regularization
   - Elastic Net

4. **Logistic Regression** (4 hours)
   - Sigmoid function
   - Binary classification
   - Cross-entropy loss
   - Multi-class extension

#### Hands-On Lab

**Exercise:** Build Complete ML Library from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionFromScratch:
    """Linear regression implemented from scratch."""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, reg_lambda=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization  # 'ridge', 'lasso', or None
        self.reg_lambda = reg_lambda
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """
        Fit linear regression model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Calculate cost
            cost = self._calculate_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Add regularization
            if self.regularization == 'ridge':
                dw += (self.reg_lambda / n_samples) * self.weights
            elif self.regularization == 'lasso':
                dw += (self.reg_lambda / n_samples) * np.sign(self.weights)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def _calculate_cost(self, y, y_pred):
        """Calculate cost with optional regularization."""
        n_samples = len(y)
        mse = np.mean((y - y_pred) ** 2)
        
        if self.regularization == 'ridge':
            reg_term = (self.reg_lambda / (2 * n_samples)) * np.sum(self.weights ** 2)
            return mse + reg_term
        elif self.regularization == 'lasso':
            reg_term = (self.reg_lambda / n_samples) * np.sum(np.abs(self.weights))
            return mse + reg_term
        
        return mse
    
    def predict(self, X):
        """Predict using fitted model."""
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class LogisticRegressionFromScratch:
    """Logistic regression implemented from scratch."""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _sigmoid(self, z):
        """Sigmoid activation function."""
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Fit logistic regression model."""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_pred)
            
            # Calculate cost (cross-entropy)
            cost = self._calculate_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def _calculate_cost(self, y, y_pred):
        """Calculate cross-entropy loss."""
        n_samples = len(y)
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cost = -(1 / n_samples) * np.sum(y * np.log(y_pred) + 
                                         (1 - y) * np.log(1 - y_pred))
        return cost
    
    def predict_proba(self, X):
        """Predict probability of class 1."""
        linear_pred = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_pred)
    
    def predict(self, X, threshold=0.5):
        """Predict class labels."""
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def score(self, X, y):
        """Calculate accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# Test implementations
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING ML FROM SCRATCH")
    print("=" * 60)
    
    # Test Linear Regression
    print("\n1. LINEAR REGRESSION TEST")
    np.random.seed(42)
    X_train = np.random.randn(100, 3)
    y_train = 2 * X_train[:, 0] + 3 * X_train[:, 1] - 1 * X_train[:, 2] + np.random.randn(100) * 0.1
    
    lr = LinearRegressionFromScratch(learning_rate=0.1, n_iterations=1000)
    lr.fit(X_train, y_train)
    
    print(f"Weights: {lr.weights}")
    print(f"Bias: {lr.bias:.4f}")
    print(f"R² Score: {lr.score(X_train, y_train):.4f}")
    
    # Plot cost history
    plt.figure(figsize=(10, 5))
    plt.plot(lr.cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Linear Regression - Cost History')
    plt.grid(True, alpha=0.3)
    plt.savefig('linear_regression_cost.png', dpi=300)
    
    # Test Logistic Regression
    print("\n2. LOGISTIC REGRESSION TEST")
    from sklearn.datasets import make_classification
    
    X_binary, y_binary = make_classification(n_samples=200, n_features=5, 
                                             n_informative=3, random_state=42)
    
    log_reg = LogisticRegressionFromScratch(learning_rate=0.1, n_iterations=1000)
    log_reg.fit(X_binary, y_binary)
    
    print(f"Weights: {log_reg.weights}")
    print(f"Bias: {log_reg.bias:.4f}")
    print(f"Accuracy: {log_reg.score(X_binary, y_binary):.4f}")
    
    # Plot cost history
    plt.figure(figsize=(10, 5))
    plt.plot(log_reg.cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Logistic Regression - Cost History')
    plt.grid(True, alpha=0.3)
    plt.savefig('logistic_regression_cost.png', dpi=300)
    
    print("\n" + "=" * 60)
    print("TESTS COMPLETE!")
    print("=" * 60)
```

#### Quiz

**Sample Questions:**

**Q1.** The gradient of the MSE cost function with respect to weights is:

A) (1/n) × X^T × (y_pred - y)  
B) (1/n) × X × (y - y_pred)  
C) X^T × y  
D) (y_pred - y)²  

**Answer:** A) (1/n) × X^T × (y_pred - y)

**Q2.** L2 regularization adds what term to the cost function?

A) λ × sum(|w|)  
B) λ × sum(w²)  
C) λ × max(w)  
D) λ × mean(w)  

**Answer:** B) λ × sum(w²)

**Q3.** [Coding] Implement stochastic gradient descent update:

```python
def sgd_update(self, X_i, y_i, learning_rate):
    """
    Update weights using single sample (SGD).
    
    Args:
        X_i: Single sample
        y_i: Single target
        learning_rate: Learning rate
    """
    # Your code here
    pass
```

---

## 🎓 Tier 1 Final Project

**Build:** Complete ML Analysis Pipeline

### Requirements

Combine everything from Tier 1:

1. **Linear Algebra**
   - Implement matrix operations
   - Use SVD for dimensionality reduction

2. **Calculus**
   - Implement gradient descent
   - Visualize optimization

3. **Probability & Statistics**
   - Perform statistical analysis
   - Conduct hypothesis testing

4. **Python for Data Science**
   - Load and clean data
   - Create visualizations
   - Perform EDA

5. **ML from Scratch**
   - Implement linear/logistic regression
   - Train and evaluate models

### Project Options

Choose one:

1. **House Price Prediction**
   - Dataset: California Housing
   - Implement regression
   - Analyze feature importance

2. **Customer Churn Prediction**
   - Dataset: Telco Customer Churn
   - Implement classification
   - Identify churn factors

3. **Stock Price Analysis**
   - Dataset: Historical stock prices
   - Time series analysis
   - Trend prediction

---

## ✅ Tier 1 Completion Checklist

- [ ] Complete all 7 modules
- [ ] Pass all module quizzes (80%+)
- [ ] Complete all hands-on labs
- [ ] Submit final project
- [ ] Pass Tier 1 Final Exam (80%+)

---

## 📊 Time Commitment

| Module | Theory | Practice | Total |
|--------|--------|----------|-------|
| 1.1 | 4 hours | 6 hours | 10 hours |
| 1.2 | 4 hours | 6 hours | 10 hours |
| 1.3 | 4 hours | 6 hours | 10 hours |
| 1.4 | 5 hours | 7 hours | 12 hours |
| 1.5 | 4 hours | 6 hours | 10 hours |
| 1.6 | 5 hours | 10 hours | 15 hours |
| 1.7 | 6 hours | 9 hours | 15 hours |
| **Final Project** | 2 hours | 18 hours | 20 hours |
| **TOTAL** | **34 hours** | **68 hours** | **102 hours** |

---

## 🎯 What's Next?

After completing Tier 1:

1. **Congratulations!** 🎉 Strong foundation built
2. **Proceed to Tier 2** - ML Practitioner

### Ready for Tier 2?

You should understand:
- ✅ Linear algebra operations
- ✅ Derivatives and gradients
- ✅ Probability and statistics
- ✅ Python data manipulation
- ✅ Gradient descent implementation

If yes → [Start Tier 2: ML Practitioner](../tier-2-ml-practitioner/README.md)

---

**Last Updated:** April 2, 2026  
**Authors:** AI-Mastery-2026 Education Team  
**Version:** 1.0

---

[← Back to Course Catalog](../COURSE_CATALOG.md) | [Start Module 1.1](module-01-linear-algebra/README.md) | [Join Discord](https://discord.gg/aimastery2026)
