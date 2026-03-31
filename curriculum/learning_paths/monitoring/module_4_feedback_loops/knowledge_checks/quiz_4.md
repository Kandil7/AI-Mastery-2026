# Module 4 Knowledge Check: User Feedback Loops

## Question 1: Feedback Types

**Question:**
Classify each feedback type and give an example of how it could be used:

A. User gives a thumbs-up to a response
B. User spends 2 minutes reading a response
C. User edits the generated response before submitting
D. User chooses Response A over Response B in a side-by-side comparison

**Answer:**
```
A. Explicit Feedback (positive)
   Use: Direct signal for reward model training
   Weight: High (intentional signal)

B. Implicit Feedback (engagement)
   Use: Proxy for response quality/relevance
   Weight: Medium (could mean confusion, not just interest)

C. Correction Feedback
   Use: Fine-tuning data, error analysis
   Weight: Very High (shows exactly what was wrong)

D. Preference Feedback
   Use: RLHF reward model, pairwise comparison training
   Weight: High (relative quality signal)
```

---

## Question 2: Feedback Quality

**Question:**
Your feedback data shows:
- 90% of responses receive thumbs-up
- Average rating: 4.8/5
- But user retention is declining

What might explain this discrepancy?

**Answer:**
```
Possible explanations:

1. Feedback Bias
   - Only satisfied users leave feedback (selection bias)
   - Social desirability bias (users rate politely)

2. Feedback Fatigue
   - Users auto-click positive to dismiss prompts
   - Feedback prompt timing is annoying

3. Metric Mismatch
   - Thumbs-up measures immediate satisfaction
   - Retention measures long-term value
   - Users might be satisfied but not finding lasting value

4. Threshold Issues
   - "Good enough" responses get positive feedback
   - But not good enough to return

Solutions:
- Add implicit feedback signals (return rate, session length)
- Reduce feedback friction
- Analyze feedback from power users vs. casual users
- Track feedback trends over time, not just averages
```

---

## Question 3: RLHF Pipeline

**Question:**
Describe the RLHF training pipeline and the role of feedback at each stage.

**Answer:**
```
RLHF Pipeline:

Stage 1: Supervised Fine-Tuning (SFT)
- Feedback role: High-quality human-written examples
- Purpose: Teach model desired behavior format

Stage 2: Reward Model Training
- Feedback role: Human preference comparisons (A vs B)
- Purpose: Train model to score response quality
- Data needed: 10,000-100,000 comparisons

Stage 3: Reinforcement Learning (PPO)
- Feedback role: Reward model provides optimization signal
- Purpose: Fine-tune policy to maximize reward
- Process: Generate → Score → Update → Repeat

Stage 4: Safety Fine-Tuning
- Feedback role: Negative examples, safety ratings
- Purpose: Reduce harmful outputs

Key insight: Human feedback is used directly in Stage 2,
then indirectly via reward model in Stage 3.
```

---

## Question 4: Feedback Bias

**Question:**
Your feedback data is 80% from users in North America, 15% Europe, 5% rest of world.
Your user base is 50% North America, 30% Europe, 20% rest of world.

What problems might this cause? How would you fix it?

**Answer:**
```
Problems:

1. Geographic Bias
   - Model optimized for North American preferences
   - Cultural nuances from other regions ignored

2. Language Bias
   - English responses over-optimized
   - Other languages underrepresented

3. Topic Bias
   - North American topics overrepresented
   - Regional interests underrepresented

Solutions:

1. Reweighting
   - Weight feedback by inverse of overrepresentation
   - NA: 0.5/0.8 = 0.625 weight
   - EU: 0.3/0.15 = 2.0 weight
   - ROW: 0.2/0.05 = 4.0 weight

2. Stratified Sampling
   - Sample feedback to match user distribution
   - Ensure balanced training batches

3. Targeted Collection
   - Prompt underrepresented users for feedback
   - Incentivize feedback from diverse users

4. Separate Models
   - Region-specific fine-tuning
   - Shared base with regional adapters
```

---

## Question 5: Continuous Improvement

**Question:**
Design a feedback-driven continuous improvement loop for an LLM chatbot.
Include feedback collection, analysis, and model update triggers.

**Answer:**
```
Continuous Improvement Loop:

┌─────────────────────────────────────────────────────────────┐
│                    Feedback Loop                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. COLLECT                                                  │
│     • Thumbs up/down on every response                      │
│     • Optional detailed rating (1-5)                        │
│     • Edit tracking (what users changed)                    │
│     • Conversation continuation (implicit quality)          │
│                                                              │
│  2. ANALYZE (Daily)                                          │
│     • Calculate satisfaction rate                           │
│     • Identify low-performing topics                        │
│     • Detect emerging issues (sudden drops)                 │
│     • Cluster negative feedback for patterns                │
│                                                              │
│  3. TRIGGER CONDITIONS                                       │
│     • Satisfaction < 80% for 3 days → Investigate           │
│     • Satisfaction < 70% for 1 day → Alert                  │
│     • Specific topic < 60% → Prioritize fix                 │
│     • 100+ similar complaints → Hotfix                      │
│                                                              │
│  4. IMPROVE                                                  │
│     • Weekly: Fine-tune on new feedback                     │
│     • Monthly: Collect new preference data                  │
│     • Quarterly: Full RLHF cycle                            │
│                                                              │
│  5. VALIDATE                                                 │
│     • A/B test improvements                                 │
│     • Monitor for regressions                               │
│     • Track long-term retention impact                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

*End of Module 4 Knowledge Check*
