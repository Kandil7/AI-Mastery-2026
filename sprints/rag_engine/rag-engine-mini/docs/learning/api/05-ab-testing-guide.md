# A/B Testing Guide
# دليل اختبارات A/B

## Table of Contents / جدول المحتويات

1. [Introduction / مقدمة](#introduction)
2. [Why A/B Testing? / لماذا اختبارات A/B؟](#why-ab-testing)
3. [A/B Testing Architecture / معمارية اختبارات A/B](#ab-testing-architecture)
4. [Designing A/B Tests / تصميم اختبارات A/B](#designing-ab-tests)
5. [GraphQL Integration / التكامل مع GraphQL](#graphql-integration)
6. [Statistical Significance / الدلالة الإحصائية](#statistical-significance)
7. [Best Practices / أفضل الممارسات](#best-practices)
8. [Common Pitfalls / الأخطاء الشائعة](#common-pitfalls)
9. [Summary / الملخص](#summary)

---

## Introduction / مقدمة

### What is A/B Testing? / ما هي اختبارات A/B؟

A/B testing (also known as split testing) is a method of comparing two versions of a webpage, app, or feature to determine which one performs better. Users are randomly assigned to different variants, and their behavior is measured to determine which version is more effective.

**Key characteristics:**
- Controlled experiment with randomized assignment
- Simultaneous comparison of variants
- Data-driven decision making
- Quantitative measurement of performance

**الاختبار A/B** (يُعرف أيضًا باختبار الانقسام) هو طريقة لمقارنة نسختين من صفحة ويب أو تطبيق أو ميزة لتحديد الأداء الأفضل. يتم تعيين المستخدمين عشوائيًا إلى متغيرات مختلفة، ويتم قياس سلوكهم لتحديد النسخة الأكثر فعالية.

**الخصائص الرئيسية:**
- تجربة خاضمة مع تعيين عشوائي
- مقارنة متزامنة للمتغيرات
- صنع قرار قائم على البيانات
- قياس كمي للأداء

---

## Why A/B Testing? / لماذا اختبارات A/B؟

### Benefits / الفوائد

**1. Data-Driven Decisions / قرارات قائمة على البيانات**

Instead of relying on intuition or opinions, A/B testing provides empirical evidence:

```python
# Without A/B testing: Decision based on opinion
design_choice = "blue button"  # "Blue looks better"

# With A/B testing: Decision based on data
results = run_ab_test(variant_a="blue", variant_b="green")
if results["green"]["conversion_rate"] > results["blue"]["conversion_rate"]:
    design_choice = "green button"
```

**بدون اختبارات A/B**: القرارات قائمة على الآراء
**مع اختبارات A/B**: القرارات قائمة على البيانات

**2. Reduced Risk / تقليل المخاطر**

Testing changes on a small percentage of users before full rollout reduces risk:

- Feature A (control): 90% of users
- Feature B (treatment): 10% of users

If B causes issues, only 10% affected.

**2. تقليل المخاطر**

اختبار التغييرات على نسبة صغيرة من المستخدمين قبل الإطلاق الكامل يقلل المخاطر:
- الميزة A (التحكم): 90% من المستخدمين
- الميزة B (العلاج): 10% من المستخدمين

**3. Continuous Improvement / التحسين المستمر**

A/B testing fosters a culture of experimentation and optimization:

```
Hypothesis → Test → Analyze → Learn → Iterate
```

---

## A/B Testing Architecture / معمارية اختبارات A/B

### Components / المكونات

**1. Experiment Configuration / تكوين التجربة**

```python
@dataclass
class Experiment:
    """A/B test experiment configuration."""
    id: str
    name: str
    description: str
    status: ExperimentStatus  # draft, running, paused, completed
    created_at: datetime
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
```

**2. Variant Configuration / تكوين المتغيرات**

```python
@dataclass
class Variant:
    """A/B test variant."""
    id: str
    experiment_id: str
    name: str
    allocation: float  # 0.0 to 1.0 (must sum to 1.0)
    config: Dict  # Feature configuration
```

**Example / مثال**:

```python
experiment = Experiment(
    id="exp_123",
    name="Chat UI Colors",
    description="Test different chat interface colors",
    status=ExperimentStatus.RUNNING,
    created_at=datetime.utcnow(),
    started_at=datetime.utcnow(),
)

variants = [
    Variant(
        id="var_1",
        experiment_id="exp_123",
        name="Blue Theme",
        allocation=0.5,  # 50%
        config={"color": "blue", "chat_bubble": "light"},
    ),
    Variant(
        id="var_2",
        experiment_id="exp_123",
        name="Green Theme",
        allocation=0.5,  # 50%
        config={"color": "green", "chat_bubble": "light"},
    ),
]
```

**3. User Assignment / تعيين المستخدمين**

```python
def assign_variant(experiment_id: str, user_id: str) -> Variant:
    """
    Assign variant to user using deterministic hashing.

    Uses MD5 hash of (user_id:experiment_id) for consistent assignment.
    """
    import hashlib

    # Create hash value (0-65535)
    hash_value = int(
        hashlib.md5(f"{user_id}:{experiment_id}".encode()).hexdigest(),
        16
    )
    normalized_value = hash_value / 65536.0  # 0.0 to 1.0

    # Find allocated variant
    cumulative = 0.0
    for variant in variants:
        cumulative += variant.allocation
        if normalized_value <= cumulative:
            return variant

    return None
```

**Key properties:**
- **Deterministic**: Same user always gets same variant
- **Consistent**: Assignment doesn't change over time
- **Balanced**: Allocations are approximately correct

**الخصائص الرئيسية:**
- **حتمية**: نفس المستخدم يحصل دائمًا على نفس المتغير
- **متسقة**: التعيين لا يتغير بمرور الوقت
- **متوازنة**: التخصيصات تقريبية صحيحة

**4. Metrics Collection / جمع المقاييس**

```python
@dataclass
class Metric:
    """A/B test metric."""
    experiment_id: str
    variant_id: str
    metric_name: str  # e.g., "conversion_rate", "click_rate", "time_on_page"
    metric_value: float
    recorded_at: datetime
```

**Common metrics / مقاييس شائعة:**

| Metric | Description | Example Use Case |
|--------|-------------|------------------|
| `click_rate` | Percentage of users who clicked | Button colors |
| `conversion_rate` | Percentage who completed goal | Form submission |
| `time_to_answer` | Average time to get answer | Chat performance |
| `satisfaction_score` | User satisfaction rating | UX improvements |

### المكونات

**1. تكوين التجربة**

**2. تكوين المتغيرات**

**3. تعيين المستخدمين**

**4. جمع المقاييس**

---

## Designing A/B Tests / تصميم اختبارات A/B

### Step 1: Define Hypothesis / الخطوة 1: تحديد الفرضية

**Good hypothesis / فرضية جيدة**:

```
If we change the chat bubble color from blue to green,
then user satisfaction will increase by 10%,
because green is more calming and readable.
```

**Components / المكونات:**
1. **Change**: What are you testing?
2. **Expected outcome**: What do you think will happen?
3. **Metric**: How will you measure success?
4. **Reason**: Why do you think this will work?

### Step 2: Choose Metrics / الخطوة 2: اختيار المقاييس

**Primary metric / مقياس أساسي**: Main success indicator
- Must be sensitive to change
- Must be directly related to hypothesis

**Secondary metrics / مقاييس ثانوية**: Monitor side effects
- Ensure no negative impact on other areas

**Example / مثال**:

```python
hypothesis = {
    "change": "Change chat answer generation model",
    "expected_outcome": "Increase user satisfaction by 15%",
    "primary_metric": "satisfaction_score",
    "secondary_metrics": [
        "time_to_answer",  # Should not increase significantly
        "error_rate",  # Should not increase
    ],
}
```

### Step 3: Calculate Sample Size / الخطوة 3: حساب حجم العينة

**Formula / الصيغة**:

```python
import math

def calculate_sample_size(
    baseline_rate: float,  # e.g., 0.20 (20%)
    minimum_detectable_effect: float,  # e.g., 0.05 (5%)
    confidence_level: float = 0.95,  # 95%
    power: float = 0.80,  # 80%
) -> int:
    """
    Calculate required sample size per variant.
    """
    from scipy import stats

    # Z-scores
    z_alpha = stats.norm.ppf((1 + confidence_level) / 2)
    z_beta = stats.norm.ppf(power)

    # Pooled proportion
    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect
    p_pooled = (p1 + p2) / 2

    # Sample size formula
    n = (
        (z_alpha * math.sqrt(2 * p_pooled * (1 - p_pooled)) +
        z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2 / (p1 - p2) ** 2

    return int(math.ceil(n))

# Example
sample_size = calculate_sample_size(
    baseline_rate=0.20,  # 20% baseline satisfaction
    minimum_detectable_effect=0.05,  # 5% improvement
)
print(f"Need {sample_size} users per variant")
```

### الخطوة 1: تحديد الفرضية
### الخطوة 2: اختيار المقاييس
### الخطوة 3: حساب حجم العينة

---

## GraphQL Integration / التكامل مع GraphQL

### Queries / الاستعلامات

**1. List Experiments / قائمة التجارب**

```graphql
query ListExperiments($status: ExperimentStatus) {
  experiments(status: $status, limit: 20) {
    id
    name
    description
    status
    created_at
    started_at
    ended_at
    variants {
      id
      name
      allocation
      config
    }
  }
}
```

**2. Get Single Experiment / الحصول على تجربة واحدة**

```graphql
query GetExperiment($experimentId: ID!) {
  experiment(experimentId: $experimentId) {
    id
    name
    description
    status
    variants {
      id
      name
      allocation
      config
    }
  }
}
```

**3. Get Experiment Results / الحصول على نتائج التجربة**

```graphql
query GetResults($experimentId: ID!) {
  experimentResults(experimentId: $experimentId) {
    experiment {
      id
      name
      status
    }
    metrics {
      metricName
      metricValue
      recordedAt
    }
    significant
    summary
  }
}
```

### Mutations / التغييرات

**Create Experiment / إنشاء تجربة**

```graphql
mutation CreateExperiment(
  $name: String!
  $description: String!
  $variants: [String!]!
) {
  createExperiment(
    name: $name
    description: $description
    variantsInput: $variants
  ) {
    id
    name
    description
    status
    variants {
      id
      name
      allocation
      config
    }
  }
}
```

**Example request / مثال الطلب**:

```json
{
  "name": "Chat Answer Model Comparison",
  "description": "Compare GPT-4 vs GPT-3.5 for answer quality",
  "variantsInput": [
    "{\"name\": \"GPT-4\", \"allocation\": 0.5, \"config\": {\"model\": \"gpt-4\"}}",
    "{\"name\": \"GPT-3.5\", \"allocation\": 0.5, \"config\": {\"model\": \"gpt-3.5\"}}"
  ]
}
```

---

## Statistical Significance / الدلالة الإحصائية

### Hypothesis Testing / اختبار الفرضية

**Null Hypothesis (H0) / الفرضية الصفرية**: No difference between variants
**Alternative Hypothesis (H1) / الفرضية البديلة**: Difference exists

### Common Tests / الاختبارات الشائعة

**1. Z-Test (for proportions) / اختبار Z**

```python
from scipy import stats

def z_test_two_proportions(
    conversions_a: int,
    total_a: int,
    conversions_b: int,
    total_b: int,
) -> dict:
    """
    Test if two proportions are significantly different.
    """
    p1 = conversions_a / total_a
    p2 = conversions_b / total_b

    # Pooled proportion
    p_pooled = (conversions_a + conversions_b) / (total_a + total_b)

    # Standard error
    se = math.sqrt(
        p_pooled * (1 - p_pooled) * (1/total_a + 1/total_b)
    )

    # Z-score
    z = (p2 - p1) / se

    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Confidence interval
    ci_lower = (p2 - p1) - 1.96 * se
    ci_upper = (p2 - p1) + 1.96 * se

    return {
        "z_score": z,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "confidence_interval": (ci_lower, ci_upper),
    }
```

**2. T-Test (for continuous metrics) / اختبار T**

```python
def t_test_two_means(
    values_a: List[float],
    values_b: List[float],
) -> dict:
    """
    Test if two means are significantly different.
    """
    from scipy import stats

    t_stat, p_value = stats.ttest_ind(values_a, values_b)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }
```

### Interpreting Results / تفسير النتائج

**P-value / قيمة P**:

| P-value range | Interpretation / التفسير |
|--------------|------------------------|
| < 0.01 | Highly significant / دلالة عالية جدًا |
| 0.01 - 0.05 | Significant / دلالة |
| 0.05 - 0.10 | Borderline / على الحدود |
| > 0.10 | Not significant / لا دلالة |

**Confidence Interval / فترة الثقة**:

```python
# 95% confidence interval means:
# "We are 95% confident the true difference is between X and Y"

if result["significant"]:
    print(f"Variant B is {result['lift']:.2%} better than A (95% CI: {result['confidence_interval']})")
else:
    print(f"No significant difference detected (p={result['p_value']:.3f})")
```

### اختبار الفرضية

### الاختبارات الشائعة

### تفسير النتائج

---

## Best Practices / أفضل الممارسات

### 1. Run One Test at a Time / شغل اختبار واحد في كل مرة

```python
# BAD: Multiple changes at once
variant_a = {"button_color": "blue", "font_size": "12px"}
variant_b = {"button_color": "green", "font_size": "14px"}

# GOOD: One change at a time
variant_a = {"button_color": "blue"}
variant_b = {"button_color": "green"}
```

### 2. Use Sufficient Sample Size / استخدم حجم عينة كافٍ

```python
# BAD: Too small sample
if user_count < 10:
    return winner  # Insufficient data

# GOOD: Calculate required sample
required_sample = calculate_sample_size(baseline, effect_size)
if sample_size < required_sample:
    continue_running_experiment()
else:
    analyze_and_conclude()
```

### 3. Pre-commit to Sample Size / الالتزام مسبقًا بحجم العينة

```python
# Pre-commit: Define stopping criteria
commitment = {
    "minimum_sample": 1000,  # per variant
    "minimum_duration_days": 7,
    "significance_threshold": 0.05,
}

# Check before concluding
if (sample_size < commitment["minimum_sample"] or
    duration_days < commitment["minimum_duration_days"]):
    raise ValueError("Not enough data to conclude")
```

### 4. Monitor Multiple Metrics / راقب مقاييس متعددة

```python
# Check for negative side effects
results = analyze_experiment(experiment_id)

if results["primary_metric"] > baseline:
    if results["secondary_metrics"]["error_rate"] > baseline * 1.2:
        print("WARNING: Primary metric improved but errors increased")
        consider_rollback()
```

### 5. Use Appropriate Statistical Tests / استخدم الاختبارات الإحصائية المناسبة

```python
# For proportions (click rate, conversion rate)
if metric_type == "proportion":
    test_result = z_test_two_proportions(...)
elif metric_type == "continuous":
    # For continuous values (time, score)
    test_result = t_test_two_means(...)
```

### 1. شغل اختبار واحد في كل مرة
### 2. استخدم حجم عينة كافٍ
### 3. الالتزام مسبقًا بحجم العينة
### 4. راقب مقاييس متعددة
### 5. استخدم الاختبارات الإحصائية المناسبة

---

## Common Pitfalls / الأخطاء الشائعة

### 1. Stopping Too Early / الإيقاف مبكرًا جدًا

**Problem / المشكلة**: Checking results constantly and stopping when "ahead".

**Solution / الحل**: Pre-commit to sample size and duration.

### 2. Peeking at Results / النظر في النتائج

**Problem / المشكلة**: Analyzing results before experiment is complete leads to false positives.

**Solution / الحل**: Only analyze after minimum sample and duration.

### 3. Not Accounting for Novelty Effect / عدم مراعاة تأثير الجدة

**Problem / المشكلة**: Users react differently because it's new, not because it's better.

**Solution / الحل**: Run experiment for sufficient time to account for novelty.

### 4. Segmentation Bias / انحياز التجزئة

**Problem / المشكلة**: Different user segments in each variant.

**Solution / الحل**: Use consistent hashing and check segment distribution.

### 5. Multiple Comparisons / المقارنات المتعددة

**Problem / المشكلة**: Testing many variants increases false positive rate.

**Solution / الحل**: Use Bonferroni correction or limit variants.

---

## Summary / الملخص

### Key Takeaways / النقاط الرئيسية

1. **A/B testing is a powerful tool** for data-driven decision making
2. **Components**: Experiments, variants, assignments, metrics
3. **Design process**: Hypothesis → Metrics → Sample size → Run → Analyze
4. **GraphQL integration** provides flexible querying and mutation
5. **Statistical significance** requires proper tests (Z-test, T-test)
6. **Best practices**: One test at a time, sufficient sample, pre-commitment
7. **Common pitfalls**: Early stopping, peeking, novelty effect, segmentation bias

### النقاط الرئيسية

1. **اختبارات A/B أداة قوية** لاتخاذ القرارات القائمة على البيانات
2. **المكونات**: التجارب والمتغيرات والتخصيصات والمقاييس
3. **عملية التصميم**: الفرضية → المقاييس → حجم العينة → التشغيل → التحليل
4. **التكامل مع GraphQL** يوفر استعلامًا مرنًا وتغييرًا
5. **الدلالة الإحصائية** تتطلب اختبارات مناسبة (اختبار Z، اختبار T)
6. **أفضل الممارسات**: اختبار واحد في كل مرة، عينة كافية، التزام مسبق
7. **الأخطاء الشائعة**: الإيقاف المبكر، النظر في النتائج، تأثير الجدة، انحياز التجزئة

---

## Further Reading / قراءة إضافية

- [A/B Testing Best Practices](https://optimizely.com/ab-testing)
- [Statistical Significance Calculator](https://www.optimizely.com/sample-size-calculator)
- [Experiment Design Principles](https://www.amazon.science/working-backwards-experiments)

---

## Arabic Summary / ملخص بالعربية

هذا الدليل يغطي اختبارات A/B في محرك RAG. تشمل المواضيع الرئيسية:

1. **ما هي اختبارات A/B**: طريقة لمقارنة نسختين لتحديد الأداء الأفضل
2. **معمارية اختبارات A/B**: التجارب والمتغيرات والتعيين والمقاييس
3. **تصميم اختبارات A/B**: الفرضية، المقاييس، حجم العينة
4. **التكامل مع GraphQL**: الاستعلامات والتغييرات لإدارة التجارب
5. **الدلالة الإحصائية**: اختبار Z، اختبار T، فترات الثقة
6. **أفضل الممارسات**: اختبار واحد، عينة كافية، مراقبة مقاييس متعددة
7. **الأخطاء الشائعة**: الإيقاف المبكر، تأثير الجدة، انحياز التجزئة

تمثل اختبارات A/B أساسًا لتحسين النظام القائم على البيانات والتجريب المستمر.
