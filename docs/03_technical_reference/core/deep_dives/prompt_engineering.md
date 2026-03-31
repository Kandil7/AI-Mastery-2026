# 🤖 LLM Guardrails & Prompt Engineering

> Guide to prompt design and guardrails for reliable RAG.

---

## Overview

Prompt engineering is critical for RAG quality. This guide covers:
- System prompt design
- Guardrails against hallucination
- Few-shot examples
- Output format control

---

## System Prompt Design

### Core Principles

1. **Explicit Context Instruction**: Tell the model to ONLY use provided context
2. **Fallback Behavior**: Define what to do when answer isn't in context
3. **Formatting Guidelines**: Specify desired output format
4. **Citation Encouragement**: Ask for source references

### Default RAG Prompt

```python
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

IMPORTANT RULES:
1. Only answer based on the information in the context below.
2. If the answer is not in the context, say "I don't have enough information to answer this question."
3. Be concise and accurate.
4. Cite the relevant parts of the context when possible.
5. Do not make up information that is not in the context.

Context:
{context}
"""
```

---

## Guardrails

### 1. Hallucination Prevention

```python
# Explicit fallback instruction
"If the answer is not in the context, say 'I don't have enough information.'"

# Don't make up facts
"Do not make up information that is not in the context."

# Discourage speculation
"Only state facts that are directly supported by the provided context."
```

### 2. Source Attribution

```python
# Format context with source markers
"[Source 1] {chunk_1_text}"
"[Source 2] {chunk_2_text}"

# Ask for citations
"When answering, cite which source(s) support your answer."
```

### 3. Scope Limitation

```python
# Restrict to document scope
"Only answer questions related to the provided documents."

# Reject off-topic queries
"If the question is unrelated to the context, politely decline."
```

---

## Prompt Templates

### Basic RAG

```python
def build_basic_prompt(question: str, context: str) -> str:
    return f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
```

### Structured Output

```python
def build_structured_prompt(question: str, context: str) -> str:
    return f"""Based on the context below, provide a structured answer.

Context:
{context}

Question: {question}

Respond in this exact JSON format:
{{
  "answer": "Your answer here",
  "confidence": "high|medium|low",
  "sources_used": ["Source 1", "Source 2"],
  "key_points": ["point 1", "point 2"]
}}"""
```

### Multi-Turn Conversation

```python
def build_chat_prompt(history: list, context: str, question: str) -> list:
    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant with access to documents.
            
Current context:
{context}

Rules:
1. Use only the context to answer
2. If unsure, say so
3. Be conversational but accurate"""
        }
    ]
    
    # Add history
    for turn in history:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    
    # Add current question
    messages.append({"role": "user", "content": question})
    
    return messages
```

---

## Advanced Techniques

### Chain-of-Thought Prompting

```python
CHAIN_OF_THOUGHT = """Based on the context, answer step by step:

Context:
{context}

Question: {question}

Let's think through this step by step:
1. First, identify the relevant information in the context
2. Then, analyze how it relates to the question
3. Finally, formulate a clear answer

Step-by-step reasoning:"""
```

### Self-Consistency Check

```python
SELF_CHECK = """Answer the following question. Then verify your answer.

Context:
{context}

Question: {question}

Your answer:
[Provide your answer]

Verification:
[Check if your answer is fully supported by the context. If not, revise.]

Final answer:"""
```

---

## Temperature Settings

| Use Case | Temperature | Rationale |
|----------|-------------|-----------|
| Factual Q&A | 0.0 - 0.2 | Deterministic, accurate |
| Summarization | 0.3 - 0.5 | Slight variation OK |
| Creative writing | 0.7 - 1.0 | More varied output |
| RAG (default) | 0.2 | Balance accuracy/fluency |

---

## Output Format Control

### JSON Mode

```python
# OpenAI
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    response_format={"type": "json_object"},
)
```

### Constrained Output

```python
CONSTRAINED_PROMPT = """Answer in exactly 3 bullet points.
Each bullet should be one sentence.
Do not include any other text.

Context: {context}
Question: {question}

• 
• 
• """
```

---

## Error Handling in Prompts

### Graceful Degradation

```python
FALLBACK_PROMPT = """Try to answer the question. If you cannot:

1. State what information would be needed
2. Suggest related questions you CAN answer from the context

Context: {context}
Question: {question}

Response:"""
```

---

## Prompt Testing

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Faithfulness | Answer matches context | > 0.9 |
| Answer Relevancy | Answers the question | > 0.85 |
| Context Precision | Relevant chunks first | > 0.8 |
| Context Recall | All needed chunks retrieved | > 0.8 |

### Testing Framework

```python
def test_hallucination():
    """Test that model doesn't hallucinate."""
    context = "The sky is blue."
    question = "What color is grass?"
    
    answer = generate_answer(context, question)
    
    # Should decline to answer
    assert "don't have enough information" in answer.lower()
```

---

---

## Stage-Specific Prompts

In RAG Engine Mini, we use specialized prompts for advanced features.

### 1. Document Summarization (Stage 2)
Used during indexing to provide context for chunks.
```python
"Summarize the following document content in 1-2 sentences to provide context for RAG chunks. "
"Output ONLY the summary sentences:\n\n{text}"
```

### 2. Retrieval Relevance Grading (Stage 3)
Used in Self-Corrective RAG to evaluate search results.
```text
"You are a grader evaluating the relevance of retrieved documents to a user question.\n"
"User Question: {question}\n\n"
"Retrieved Documents:\n{context}\n\n"
"If the documents contain information that can answer the question, respond with 'relevant'.\n"
"Otherwise, respond with 'irrelevant'. respond with ONLY one word."
```

### 3. Hallucination Grading (Stage 3)
Verifying if the generated answer is grounded in facts.
```text
"You are a grader evaluating if an answer is grounded in / supported by a set of facts.\n"
"Facts:\n{context}\n\n"
"Question: {question}\n"
"Answer: {answer}\n\n"
"Is the answer fully supported by the facts? Respond with 'grounded'.\n"
"If there is ANY info in the answer not in the facts, respond with 'hallucination'.\n"
"Respond with ONLY one word."
```

### 4. Knowledge Graph Extraction (Stage 3)
Building structured triplets from text.
```text
"Extract key entities and their relationships from the text below.\n"
"Format the output as a JSON list of objects with keys: 'subject', 'relation', 'obj'.\n"
"Keep entities concise (1-3 words).\n\n"
"Text: {text}\n\n"
"JSON Output:"
```

### 5. Vision-to-Text (Stage 4)
Describing images for multi-modal indexing.
```text
"You are a vision assistant. Describe the provided image in detail. "
"Focus on text, charts, or logical diagrams if present. "
"Output your description in 1-2 concise paragraphs."
```

---

## Best Practices Summary

1. ✅ Always include explicit grounding instructions
2. ✅ Define clear fallback behavior
3. ✅ Use source markers in context
4. ✅ Keep temperature low (0.2) for factual tasks
5. ✅ Test with adversarial questions
6. ✅ Monitor for hallucination in production
7. ❌ Don't assume the model will follow instructions perfectly
8. ❌ Don't use high temperatures for factual RAG

