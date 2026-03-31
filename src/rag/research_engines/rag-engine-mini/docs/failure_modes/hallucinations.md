# ❌ Failure Mode: Hallucinations (Making Up Information)

## 🤕 Symptoms
* Model generates plausible-sounding but incorrect information
* Answers contain facts not present in the retrieved context
* Overconfidence in incorrect answers
* Fabrication of citations or sources

## 🔍 Root Causes
1. **Over-reliance on parametric knowledge**: LLM fills gaps with pre-trained knowledge
2. **Poor retrieval quality**: Retrieved chunks don't contain relevant information
3. **Inadequate prompt engineering**: Prompts don't constrain model to use only provided context
4. **Missing grounding checks**: No validation that answers are based on retrieved info
5. **Insufficient guardrails**: No mechanisms to detect and prevent fabrications

## 💡 How This Repository Fixes This
### 1. Grounding Checks
```python
# Ensure answers are grounded in retrieved context
def validate_answer_grounding(answer: str, context: List[str]) -> bool:
    # Check if claims in answer are supported by context
    return check_factual_consistency(answer, context)
```

### 2. Self-Critique Mechanisms
- Built-in validation to assess answer quality
- Confidence scoring for generated responses
- Citation requirements in prompts

### 3. Prompt Engineering
- Explicit instructions to use only provided context
- Structured prompts that require citing sources
- Guardrails to prevent fabrication

## 🔧 How to Trigger/Debug This Issue
1. **Disable Grounding Checks**: Remove validation mechanisms temporarily
2. **Provide Empty Context**: Pass empty context to see if model uses parametric knowledge
3. **Remove Prompt Constraints**: Simplify prompts to allow more freedom
4. **Use Creative LLM Settings**: Increase temperature and creativity settings

## 📊 Expected Impact
Without grounding: ~40% of answers may contain hallucinations
With proper grounding: ~5% of answers contain hallucinations