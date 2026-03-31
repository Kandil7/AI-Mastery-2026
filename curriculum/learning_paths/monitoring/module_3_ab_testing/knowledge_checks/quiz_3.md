# Module 3 Knowledge Check: A/B Testing

## Question 1: Sample Size Calculation

**Question:**
You want to detect a 5% improvement in user satisfaction (baseline: 3.5/5, std: 1.0) with 80% power and 95% confidence. Using the formula:

n = 2 × ((z_α + z_β) × σ / δ)²

Where:
- z_α = 1.96 (for 95% confidence)
- z_β = 0.84 (for 80% power)
- σ = 1.0 (standard deviation)
- δ = 0.175 (5% of 3.5)

Calculate the required sample size per variant.

**Answer:**
```
n = 2 × ((1.96 + 0.84) × 1.0 / 0.175)²
n = 2 × (2.8 / 0.175)²
n = 2 × (16)²
n = 2 × 256
n = 512

Required sample size: 512 users per variant (1,024 total)
```

---

## Question 2: P-value Interpretation

**Question:**
Your A/B test comparing two LLM models yields:
- Model A satisfaction: 4.2 ± 0.3
- Model B satisfaction: 4.4 ± 0.3
- p-value: 0.03

Your colleague says "There's a 97% chance Model B is better." Is this correct?

**Answer:**
```
No, this is a common misinterpretation.

Correct interpretation:
- p-value of 0.03 means: IF there were no real difference between models,
  there's a 3% chance of observing a difference this large (or larger)
  due to random variation.

- It does NOT mean there's a 97% chance Model B is better.

- The p-value tells us the result is statistically significant at α=0.05,
  but doesn't directly give the probability that one model is better.

For probability statements, Bayesian A/B testing would be more appropriate.
```

---

## Question 3: Multiple Comparisons

**Question:**
You run A/B tests on 20 different metrics. One metric shows p=0.04. 
Should you conclude the treatment had an effect?

**Answer:**
```
No, this is likely a false positive due to multiple comparisons.

With 20 independent tests at α=0.05:
- Expected false positives = 20 × 0.05 = 1

The single p=0.04 result is consistent with random chance.

Solutions:
1. Bonferroni correction: Use α = 0.05/20 = 0.0025
2. Pre-register primary metric before experiment
3. Use False Discovery Rate (FDR) control
4. Require replication before concluding effect

Best practice: Define ONE primary metric before starting the experiment.
```

---

## Question 4: Peeking Problem

**Question:**
You check your A/B test results daily. On day 3, you see p=0.04 and 
consider stopping the experiment. Why is this problematic?

**Answer:**
```
This is the "peeking" or "early stopping" problem.

Problem:
- Each check is a hypothesis test
- Multiple checks inflate false positive rate
- p=0.04 on day 3 might not hold at day 14

Inflated error rate example:
- Single test at α=0.05: 5% false positive rate
- 10 peeking opportunities: ~40% false positive rate

Solutions:
1. Pre-determine sample size and stick to it
2. Use sequential testing with proper corrections
3. Apply alpha spending functions
4. Use Bayesian methods with stopping rules

Best practice: Calculate required sample size upfront and wait until reached.
```

---

## Question 5: Practical vs. Statistical Significance

**Question:**
Your A/B test with 100,000 users shows:
- Control: 4.200 satisfaction
- Treatment: 4.205 satisfaction
- p-value: 0.001

Is this result meaningful? Should you deploy?

**Answer:**
```
Statistically significant: Yes (p=0.001 < 0.05)
Practically significant: Debatable

Analysis:
- Difference: 0.005 on 5-point scale (0.12% improvement)
- With large N, tiny differences become statistically significant
- Question: Is 0.005 improvement worth the cost of change?

Consider:
1. Cost of implementing treatment
2. Risk of unintended consequences
3. User-perceptible difference
4. Business impact

Recommendation:
- Calculate confidence interval for effect size
- Assess practical significance with stakeholders
- Consider running cost-benefit analysis
- May not be worth deploying for such small gain
```

---

*End of Module 3 Knowledge Check*
