# Chapter 4: Probability and Statistics for Machine Learning

> **Learning Duration:** 4 Days  
> **Difficulty:** Intermediate  
> **Prerequisites:** Calculus basics

---

## 🎯 Learning Objectives

By the end of this chapter, you will:
- Understand probability distributions and their properties
- Work with common distributions (Normal, Binomial, Poisson)
- Apply Bayes theorem for inference
- Perform statistical estimation and hypothesis testing
- Apply probabilistic reasoning to ML problems

---

## 4.1 Foundations of Probability

### What Is Probability?

Probability measures the **likelihood** of an event occurring, expressed as a number between 0 and 1.

- $P(A) = 0$: Event A cannot happen
- $P(A) = 1$: Event A is certain
- $0 < P(A) < 1$: Various degrees of likelihood

### Key Definitions

**Sample Space ($\Omega$):** All possible outcomes

**Event ($A$):** A subset of the sample space

**Probability Axioms:**
1. $P(A) \geq 0$ for any event $A$
2. $P(\Omega) = 1$
3. If $A_1, A_2, ...$ are disjoint, then $P(\cup A_i) = \sum P(A_i)$

### Types of Probability

| Type | Formula | When Used |
|------|---------|-----------|
| Marginal | $P(A)$ | Single event |
| Joint | $P(A \cap B)$ | Two events together |
| Conditional | $P(A | B)$ | Given another event occurred |

---

## 4.2 Conditional Probability and Independence

### Conditional Probability

The probability of A given B has occurred:

$$P(A | B) = \frac{P(A \cap B)}{P(B)}$$

**Intuition:** $P(A | B)$ is our updated belief about A after seeing B.

### Example

```python
# Suppose we have:
# P(Disease) = 0.01 (1% of population has disease)
# P(Positive Test | Disease) = 0.99 (test is 99% accurate)
# P(Positive Test | No Disease) = 0.05 (5% false positive)

# What is P(Disease | Positive Test)?
P_disease = 0.01
P_positive_given_disease = 0.99
P_positive_given_no_disease = 0.05

# P(Positive) = P(positive| disease)*P(disease) + P(positive|no disease)*P(no disease)
P_positive = (0.99 * 0.01) + (0.05 * 0.99)

# P(Disease | Positive) = P(Positive | Disease) * P(Disease) / P(Positive)
P_disease_given_positive = (0.99 * 0.01) / P_positive
# ≈ 0.167 (only 16.7%! Even with positive test, likely no disease)
```

### Independence

Two events are **independent** if knowing one doesn't affect the other:

$$P(A | B) = P(A) \quad \text{or} \quad P(A \cap B) = P(A) \cdot P(B)$$

### Bayes Theorem

One of the most important theorems in ML:

$$P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}$$

Or in ML notation (for hypothesis $H$ and evidence $E$):

$$P(H | E) = \frac{P(E | H) \cdot P(H)}{P(E)}$$

| Term | Meaning in ML |
|------|---------------|
| $P(H)$ | Prior (our belief before seeing data) |
| $P(E | H)$ | Likelihood (probability of evidence given hypothesis) |
| $P(E)$ | Evidence (probability of data) |
| $P(H | E)$ | Posterior (updated belief after seeing data) |

---

## 4.3 Random Variables

### Definition

A **random variable** is a function that assigns a number to each outcome in the sample space.

```
Outcome → Random Variable → Number
  coin toss          X          0 or 1
```

### Types

1. **Discrete**: Finite or countable values (integers)
2. **Continuous**: Any value in an interval

### Probability Mass Function (PMF)

For discrete random variables:

$$P(X = x) = p(x)$$

Must satisfy:
- $p(x) \geq 0$
- $\sum_x p(x) = 1$

### Probability Density Function (PDF)

For continuous random variables:

$$P(a \leq X \leq b) = \int_a^b f(x) dx$$

Must satisfy:
- $f(x) \geq 0$
- $\int_{-\infty}^{\infty} f(x) dx = 1$

---

## 4.4 Expectation and Variance

### Expected Value (Mean)

The **mean** of a random variable:

$$E[X] = \sum_x x \cdot p(x) \quad \text{(discrete)}$$
$$E[X] = \int_{-\infty}^{\infty} x \cdot f(x) dx \quad \text{(continuous)}$$

**Properties:**
- $E[cX] = cE[X]$
- $E[X + Y] = E[X] + E[Y]$
- $E[E[X]] = E[X]$ (expectation of expectation)

### Variance

Measures spread around the mean:

$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

**Properties:**
- $\text{Var}(cX) = c^2 \text{Var}(X)$
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ (if independent)

### Standard Deviation

$$\sigma = \sqrt{\text{Var}(X)}$$

---

## 4.5 Common Probability Distributions

### 4.5.1 Uniform Distribution

All outcomes equally likely.

**Discrete:**
$$P(X = x) = \frac{1}{n} \quad \text{for } x \in \{1, 2, ..., n\}$$

**Continuous:**
$$f(x) = \frac{1}{b-a} \quad \text{for } a \leq x \leq b$$

### 4.5.2 Bernoulli Distribution

Single binary trial (success/failure).

$$P(X = 1) = p, \quad P(X = 0) = 1-p$$

- $E[X] = p$
- $\text{Var}(X) = p(1-p)$

### 4.5.3 Binomial Distribution

Number of successes in $n$ independent Bernoulli trials.

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

Where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$

**Example:** Probability of getting 3 heads in 10 coin flips:

```python
from scipy import stats

# n=10, p=0.5, k=3
prob = stats.binom.pmf(3, 10, 0.5)  # 0.117
```

### 4.5.4 Poisson Distribution

Counts of rare events in fixed interval.

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

Where $\lambda$ is the average rate.

**Example:** Number of emails per hour:

```python
# Average of 5 emails per hour, probability of 8?
prob = stats.poisson.pmf(8, 5)  # ~0.065
```

### 4.5.5 Normal (Gaussian) Distribution

Most important distribution in ML!

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

Parameters:
- $\mu$ = mean (center)
- $\sigma^2$ = variance (spread)

**The 68-95-99.7 Rule:**

```
       68%          95%          99.7%
   ←──────→    ←──────────→    ←─────────────→
   |μ-1σ μ+1σ|   |μ-2σ  μ+2σ|   |μ-3σ   μ+3σ|
```

**Standard Normal:** $\mu = 0, \sigma = 1$

```python
# Standard normal
X ~ N(0, 1)

# General normal
X ~ N(μ, σ²)

# Convert to standard: Z = (X - μ) / σ

# Probability of X > 2 for N(0, 1)
prob = 1 - stats.norm.cdf(2)  # ~0.0228
```

### Why Normal Distribution?

1. **Central Limit Theorem**: Sum of many independent random variables approaches normal
2. **Maximum Entropy**: Given mean and variance, normal is the most "random" distribution
3. **Mathematically Convenient**: Easy to work with
4. **Common in Nature**: Many phenomena are approximately normal

---

## 4.6 Joint and Marginal Distributions

### Joint Distribution

For two random variables $X$ and $Y$:

$$P(X = x, Y = y) = p(x, y)$$

### Marginal Distribution

Sum/integrate out the other variable:

$$P(X = x) = \sum_y P(X = x, Y = y)$$

### Independence

$X$ and $Y$ are independent if:

$$P(X = x, Y = y) = P(X = x) \cdot P(Y = y)$$

### Covariance

Measures linear relationship between variables:

$$\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]$$

- Positive: X increases with Y
- Negative: X decreases with Y
- Zero: No linear relationship

### Correlation

Normalized covariance (between -1 and 1):

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}$$

```python
# Compute correlation
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

cov = np.cov(x, y)[0, 1]  # Covariance
corr = np.corrcoef(x, y)[0, 1]  # Correlation: ~0.8
```

---

## 4.7 Maximum Likelihood Estimation (MLE)

### The Idea

Given observed data, find parameters that **maximize** the probability of observing that data.

**Likelihood:** $L(\theta | data) = P(data | \theta)$

**Maximum Likelihood:** $\hat{\theta} = \arg\max_\theta L(\theta | data)$

### Example: Estimating Coin Bias

```python
# Observed: 7 heads out of 10 flips
heads = 7
total = 10

# Likelihood for different p values
def likelihood(p):
    return stats.binom.pmf(heads, total, p)

# Find p that maximizes likelihood
p_values = np.linspace(0, 1, 100)
likelihoods = [likelihood(p) for p in p_values]
best_p = p_values[np.argmax(likelihoods)]

print(f"ML estimate: p = {best_p:.2f}")  # ~0.7
```

### For Normal Distribution

If we have data $x_1, x_2, ..., x_n$ from $N(\mu, \sigma^2)$:

$$\hat{\mu} = \frac{1}{n} \sum_{i=1}^{n} x_i \quad \text{(sample mean)}$$
$$\hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{\mu})^2 \quad \text{(sample variance)}$$

---

## 4.8 Bayesian Inference

### The Bayesian Approach

Instead of point estimates, we compute a **posterior distribution** over parameters.

$$P(\theta | \text{data}) = \frac{P(\text{data} | \theta) \cdot P(\theta)}{P(\text{data})}$$

### Prior, Likelihood, Posterior

| Component | Description |
|-----------|-------------|
| Prior $P(\theta)$ | Our belief before seeing data |
| Likelihood $P(\text{data} | \theta)$ | How likely is the data given parameters? |
| Posterior $P(\theta | \text{data})$ | Updated belief after seeing data |

### Example: Bayesian Coin Estimation

```python
# Prior: Beta(2, 2) - uniform/agnostic prior
# Likelihood: Binomial(10, p) for 7 heads
# Posterior: Beta(2+7, 2+3) = Beta(9, 5)

# Prior parameters
alpha_prior = 2
beta_prior = 2

# Observed data
heads = 7
total = 10

# Posterior parameters
alpha_posterior = alpha_prior + heads
beta_posterior = beta_prior + (total - heads)

# Posterior mean (expected value of p)
posterior_mean = alpha_posterior / (alpha_posterior + beta_posterior)
# = 9/14 ≈ 0.64

# 95% credible interval
lower = stats.beta.ppf(0.025, alpha_posterior, beta_posterior)
upper = stats.beta.ppf(0.975, alpha_posterior, beta_posterior)
# (0.38, 0.87)
```

---

## 4.9 Hypothesis Testing

### The Framework

1. **Null Hypothesis ($H_0$)**: The default assumption (no effect)
2. **Alternative Hypothesis ($H_1$)**: What we're testing for
3. **Test Statistic**: Measure from data
4. **p-value**: Probability of observing data if $H_0$ is true
5. **Significance Level ($\alpha$)**: Threshold (typically 0.05)

### Decision Rule

- If p-value < $\alpha$: Reject $H_0$ (statistically significant)
- If p-value $\geq \alpha$: Fail to reject $H_0$

### Common Tests

| Test | Use Case |
|------|----------|
| t-test | Compare means of two groups |
| ANOVA | Compare means of 3+ groups |
| Chi-square | Test independence of categories |
| Kolmogorov-Smirnov | Test if sample follows distribution |

### Example: A/B Testing

```python
# Group A: 1000 visitors, 50 conversions (5%)
# Group B: 1000 visitors, 70 conversions (7%)

# Is the difference statistically significant?
n_a, p_a = 1000, 0.05
n_b, p_b = 1000, 0.07

# Pooled proportion
p_pool = (n_a * p_a + n_b * p_b) / (n_a + n_b)

# Standard error
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))

# Z-score
z = (p_b - p_a) / se  # ~2.33

# p-value (two-tailed)
p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # ~0.02

# Since p < 0.05, reject null - B is significantly better!
```

---

## 4.10 Application to Machine Learning

### 4.10.1 Naive Bayes Classifier

Uses Bayes theorem with independence assumption:

$$P(y | x) = \frac{P(x | y) \cdot P(y)}{P(x)}$$

```python
def naive_bayes_predict(X, y, x_new):
    """Simplified Naive Bayes for binary features"""
    classes = np.unique(y)
    
    best_class = None
    best_prob = -1
    
    for c in classes:
        # Prior P(y=c)
        prior = np.mean(y == c)
        
        # Likelihood P(x|y=c) - assume independence
        likelihood = 1
        for i, val in enumerate(x_new):
            # P(x_i=val | y=c) from training data
            mask = (y == c)
            prob = np.mean(X[mask, i] == val)
            likelihood *= prob if prob > 0 else 1e-10
            
        # Posterior
        posterior = likelihood * prior
        
        if posterior > best_prob:
            best_prob = posterior
            best_class = c
            
    return best_class
```

### 4.10.2 Maximum A Posteriori (MAP)

Bayesian version of MLE - use prior:

$$\hat{\theta}_{MAP} = \arg\max_\theta P(\theta | \text{data}) = \arg\max_\theta P(\text{data} | \theta) \cdot P(\theta)$$

### 4.10.3 Gaussian Mixture Models

Probability distributions as mixture of Gaussians:

$$P(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$

### 4.10.4 Probabilistic Graphical Models

Bayesian networks and Markov random fields

---

## 📝 Summary

### Key Takeaways

1. **Probability** measures likelihood of events (0 to 1)
2. **Conditional probability** updates beliefs with new evidence
3. **Bayes theorem** is fundamental for inference
4. **Normal distribution** is key in ML (Central Limit Theorem)
5. **MLE** finds parameters that maximize data likelihood
6. **Bayesian inference** gives posterior distributions over parameters
7. **Hypothesis testing** determines statistical significance

### Key Formulas

| Concept | Formula |
|---------|---------|
| Bayes Theorem | $P(A | B) = \frac{P(B | A) P(A)}{P(B)}$ |
| Expectation | $E[X] = \sum x \cdot p(x)$ |
| Variance | $\text{Var}(X) = E[X^2] - (E[X])^2$ |
| Correlation | $\rho = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$ |
| Normal PDF | $\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ |

---

## ❓ Quick Check

1. If P(A) = 0.3, P(B) = 0.4, P(A ∩ B) = 0.1, find P(A | B)
2. What is the 95% confidence interval for N(0,1)?
3. Why is the normal distribution important in ML?
4. What's the difference between MLE and MAP?

*Answers at end of chapter*