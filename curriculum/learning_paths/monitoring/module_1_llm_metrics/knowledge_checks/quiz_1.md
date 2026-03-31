# Module 1 Knowledge Check: LLM-Specific Metrics

## Instructions

Answer the following 5 questions to test your understanding of LLM-specific metrics. Each question includes a detailed answer and explanation.

---

## Question 1: Token Economics

**Question:** 
You are running an LLM service with the following usage in one day:
- GPT-4: 500,000 input tokens, 300,000 output tokens
- GPT-3.5-turbo: 2,000,000 input tokens, 1,500,000 output tokens

Using the pricing below, calculate the total daily cost:
- GPT-4: $0.03/1K input tokens, $0.06/1K output tokens
- GPT-3.5-turbo: $0.0005/1K input tokens, $0.0015/1K output tokens

**Answer:**

```
GPT-4 Cost:
- Input: (500,000 / 1,000) × $0.03 = 500 × $0.03 = $15.00
- Output: (300,000 / 1,000) × $0.06 = 300 × $0.06 = $18.00
- GPT-4 Total: $33.00

GPT-3.5-turbo Cost:
- Input: (2,000,000 / 1,000) × $0.0005 = 2,000 × $0.0005 = $1.00
- Output: (1,500,000 / 1,000) × $0.0015 = 1,500 × $0.0015 = $2.25
- GPT-3.5-turbo Total: $3.25

Total Daily Cost: $33.00 + $3.25 = $36.25
```

**Explanation:**
Token costs are calculated per 1,000 tokens (per 1K). Divide the token count by 1,000, then multiply by the price per 1K. Input and output tokens often have different prices, with output typically being more expensive.

---

## Question 2: Latency Percentiles

**Question:**
You have the following request latencies (in milliseconds) from your LLM service:
[100, 150, 180, 200, 220, 250, 300, 400, 600, 2000]

Calculate the p50, p90, and p95 latencies. What does the p95 tell you that the average doesn't?

**Answer:**

```
Sorted latencies: [100, 150, 180, 200, 220, 250, 300, 400, 600, 2000]
Count: 10

p50 (median): Index = 0.50 × (10-1) = 4.5
  Interpolate between index 4 (220) and index 5 (250)
  p50 = 220 + 0.5 × (250 - 220) = 235ms

p90: Index = 0.90 × (10-1) = 8.1
  Interpolate between index 8 (600) and index 9 (2000)
  p90 = 600 + 0.1 × (2000 - 600) = 740ms

p95: Index = 0.95 × (10-1) = 8.55
  Interpolate between index 8 (600) and index 9 (2000)
  p95 = 600 + 0.55 × (2000 - 600) = 1360ms

Average (mean): (100+150+180+200+220+250+300+400+600+2000) / 10 = 440ms
```

**What p95 tells us that average doesn't:**
The average (440ms) is heavily influenced by the single 2000ms outlier. The p95 (1360ms) tells us that 95% of requests complete within 1360ms, which is more useful for understanding the experience of most users. The average would suggest most requests are around 440ms, which is misleading—only 40% of requests are actually that fast.

---

## Question 3: Prometheus Metric Types

**Question:**
Match each LLM metric to the appropriate Prometheus metric type (Counter, Gauge, Histogram) and explain why:

1. Total tokens processed since service start
2. Current number of active requests
3. Request duration distribution
4. Current tokens per minute rate
5. Total cost in USD

**Answer:**

```
1. Total tokens processed → COUNTER
   Reason: Only increases, never decreases. Cumulative count.

2. Current active requests → GAUGE
   Reason: Can go up and down. Represents current state.

3. Request duration → HISTOGRAM
   Reason: Need to track distribution across buckets for percentiles.

4. Tokens per minute rate → GAUGE
   Reason: Current rate that fluctuates up and down.

5. Total cost → COUNTER
   Reason: Only increases (cumulative spending).
```

**Key distinctions:**
- **Counter:** Monotonically increasing values (totals, counts)
- **Gauge:** Values that can fluctuate (current state, rates)
- **Histogram:** Distribution of values (latencies, sizes)

---

## Question 4: SLO and Error Budget

**Question:**
Your LLM service has an availability SLO of 99.9% over a 30-day window.

a) How many minutes of downtime are allowed in the 30-day window?
b) If you've already had 30 minutes of downtime with 10 days remaining, are you still within your error budget?
c) What is your burn rate if you're consuming error budget 5x faster than allowed?

**Answer:**

```
a) Error budget calculation:
   Total minutes in 30 days = 30 × 24 × 60 = 43,200 minutes
   Allowed downtime = 43,200 × (1 - 0.999) = 43,200 × 0.001 = 43.2 minutes

b) Budget status with 10 days remaining:
   Total budget: 43.2 minutes
   Used: 30 minutes
   Remaining: 13.2 minutes
   
   Pro-rated budget for 20 days elapsed: 43.2 × (20/30) = 28.8 minutes
   You've used 30 minutes but should have only used 28.8 minutes
   Status: OVER BUDGET by 1.2 minutes

c) Burn rate interpretation:
   Burn rate of 5x means you're consuming error budget 5 times faster
   than the allowed rate. At this rate, you'll exhaust your monthly
   budget in 30/5 = 6 days instead of 30 days.
   
   This typically triggers an alert and may require freezing deployments
   until the burn rate returns to normal.
```

---

## Question 5: Hallucination Detection

**Question:**
You implement a self-consistency check for hallucination detection by generating 5 responses to the same prompt and comparing them. The pairwise similarity scores are:

- Response 1-2: 0.92
- Response 1-3: 0.88
- Response 1-4: 0.45
- Response 1-5: 0.90
- Response 2-3: 0.85
- Response 2-4: 0.42
- Response 2-5: 0.88
- Response 3-4: 0.40
- Response 3-5: 0.87
- Response 4-5: 0.43

a) Calculate the overall consistency score.
b) Is this response likely hallucinated (threshold: 0.7)?
c) What might cause low consistency in a non-hallucinated response?

**Answer:**

```
a) Overall consistency score:
   Sum of all pairwise similarities: 
   0.92 + 0.88 + 0.45 + 0.90 + 0.85 + 0.42 + 0.88 + 0.40 + 0.87 + 0.43 = 7.00
   
   Number of pairs: 10 (5 choose 2 = 5!/(2!×3!) = 10)
   
   Average consistency = 7.00 / 10 = 0.70

b) Hallucination assessment:
   Consistency score: 0.70
   Threshold: 0.70
   Result: BORDERLINE (exactly at threshold)
   
   This suggests potential hallucination or high uncertainty.
   Further investigation recommended.

c) Causes of low consistency in non-hallucinated responses:
   1. Open-ended prompts with multiple valid answers
   2. Creative writing tasks (poems, stories)
   3. Opinion-based questions
   4. High temperature settings causing intentional variation
   5. Prompts asking for multiple alternatives/options
   
   Self-consistency works best for factual questions with
   single correct answers.
```

---

## Scoring

| Correct Answers | Score | Status |
|-----------------|-------|--------|
| 5/5 | 100% | Excellent |
| 4/5 | 80% | Good |
| 3/5 | 60% | Needs Review |
| < 3/5 | < 60% | Study Required |

---

*End of Module 1 Knowledge Check*
