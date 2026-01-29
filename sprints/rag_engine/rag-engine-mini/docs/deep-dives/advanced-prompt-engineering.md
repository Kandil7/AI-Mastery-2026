# 🎯 Deep-Dive: Advanced Prompt Engineering for RAG

Prompt engineering is not just about "asking nicely." In a RAG system, the prompt is the final bridge between retrieved data and the user's answer. This guide covers expert-level patterns used in **RAG Engine Mini**.

---

## 🏗️ The CO-STAR Framework
A robust prompt should follow a structure. We recommend the **CO-STAR** framework:

1.  **Context**: Provide background (e.g., "You are a research assistant...").
2.  **Objective**: Define the goal (e.g., "Answer based on the snippets...").
3.  **Style**: Set the tone (e.g., "Professional and concise").
4.  **Tone**: Emotional quality (e.g., "Helpful but skeptical of hallucinations").
5.  **Audience**: Who is reading (e.g., "For a senior AI engineer").
6.  **Response**: Output format (e.g., "Markdown with citations").

---

## 🛠️ RAG-Specific Patterns

### 1. Context Placement (The "Lost in the Middle" Problem)
LLMs tend to pay more attention to the beginning and end of long prompts.
- **Expert Tip**: Place the **instructions** AFTER the **context snippets**. This ensures the most recent tokens in the model's window are the rules it must follow.

### 2. Guardrails & Grounding
To prevent hallucinations, be explicit:
> "If the provided context does not contain the answer, state: 'I don't have enough information in the provided documents to answer this.' Do not use outside knowledge."

### 3. Citation Enforcement
Make the model "show its work":
> "Every sentence in your answer must include a citation in brackets, e.g., [DocumentA]. Only use the document names provided in the context."

---

## 🧠 Chain-of-Thought (CoT) in RAG
For complex reasoning, ask the model to think step-by-step:
> "First, identify the key entities in the question. Second, look for those entities in the retrieved snippets. Third, synthesize the answer. Finally, verify the answer against the snippets."

---

## 📝 Example Master Prompt Template
This is the pattern used in our `build_rag_prompt` service:

```markdown
### CONTEXT:
{{ snippets }}

### INSTRUCTIONS:
Using ONLY the context above, answer the following question.
Question: {{ question }}

RULES:
1. Cite sources using [ID].
2. If context is empty, say "I don't know".
3. Use professional tone.

ANSWER:
```

---

## 🚀 Mastery Level: Few-Shot Prompting
Sometimes the model needs "examples" to understand a difficult format.
- Provide 2-3 examples of **[Question] -> [Reasoning] -> [Answer]** within the system prompt to stabilize output for complex extraction tasks.
