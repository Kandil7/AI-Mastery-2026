# Module 2 Knowledge Check: Drift Detection

## Question 1: Drift Types

**Question:** 
Match each scenario to the correct drift type:

A. User queries shift from technical documentation to casual conversation
B. The same query "Who is the president?" now expects a different answer
C. Sentiment classification sees 80% negative queries during a crisis vs. 40% normally
D. Embedding model v2 produces different vectors than v1 for the same text

**Answer:**
```
A → Covariate Drift (input distribution changed)
B → Concept Drift (relationship between query and answer changed)
C → Prior Probability Drift (label distribution changed)
D → Embedding Drift (vector space changed due to model update)
```

---

## Question 2: Statistical Tests

**Question:**
You have the following PSI values for different features:
- Feature A: PSI = 0.05
- Feature B: PSI = 0.15
- Feature C: PSI = 0.35

Interpret each and recommend actions.

**Answer:**
```
Feature A (PSI = 0.05): No significant drift
  Action: Continue normal monitoring

Feature B (PSI = 0.15): Minor drift detected
  Action: Investigate root cause, increase monitoring frequency

Feature C (PSI = 0.35): Significant drift detected
  Action: Immediate investigation, consider model retraining,
          alert team, prepare remediation plan
```

---

## Question 3: KS Test Interpretation

**Question:**
A KS test between training and production data yields:
- KS statistic: 0.25
- p-value: 0.001

What does this tell you? Is drift detected?

**Answer:**
```
KS statistic of 0.25 indicates moderate drift (25% maximum difference
between CDFs).

p-value of 0.001 < 0.05 means the difference is statistically significant
(less than 0.1% chance this occurred by random variation).

Conclusion: Yes, drift is detected with high confidence.
The moderate KS statistic suggests meaningful but not catastrophic drift.
```

---

## Question 4: Embedding Drift

**Question:**
Your embedding drift detector reports:
- Mean shift: 0.3 (normalized)
- Covariance change: 0.5 (relative)
- 15% of dimensions flagged for drift

Is this concerning? What might cause this?

**Answer:**
```
Assessment: Moderate concern

Mean shift of 0.3 is below the 0.5 threshold but notable.
Covariance change of 0.5 indicates significant structural change.
15% of dimensions drifting suggests localized rather than global change.

Possible causes:
1. Partial vocabulary update (new terms added)
2. Domain shift in subset of queries
3. Embedding model fine-tuning on new data
4. Seasonal content changes affecting specific topics

Recommended action: Investigate which dimensions are drifting
to identify affected topics/domains.
```

---

## Question 5: Reference Data Management

**Question:**
Your drift detection system has been running for 6 months. 
The reference data is from initial deployment. 
Performance is degrading but no drift is detected. 
What might be wrong?

**Answer:**
```
Problem: Reference data is stale/outdated

When reference data is too old:
1. Gradual drift accumulates but stays below thresholds
2. Reference no longer represents "normal" behavior
3. System adapts to new normal without detection

Solutions:
1. Implement rolling reference windows (e.g., last 30 days)
2. Periodically update reference with verified "good" data
3. Use multiple reference periods for comparison
4. Add absolute quality metrics alongside relative drift
5. Implement concept drift detection (performance-based)
```

---

*End of Module 2 Knowledge Check*
