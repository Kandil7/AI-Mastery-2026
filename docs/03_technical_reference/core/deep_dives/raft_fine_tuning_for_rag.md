# 🧪 RAFT: Retrieval-Augmented Fine-Tuning

## 🎯 The Purpose of RAFT

Standard RAG assumes the LLM can perfectly filter noise from retrieved context. **RAFT (Retrieval-Augmented Fine-Tuning)** is a training paradigm that teaches the model *how* to do RAG better by fine-tuning it on a specialized dataset.

### The Problem with Vanilla LLMs in RAG:
- **Distractor Noise**: Most LLMs get confused when they see 10 chunks where only 2 are relevant.
- **Lost in the Middle**: Models tend to ignore context placed in the middle of a large prompt.
- **No Domain Knowledge**: For niche fields (Law, Medicine, Internal HR), general models lack the nuanced understanding required to interpret retrieved facts.

---

## 🏗️ The RAFT Dataset Structure

RAFT transforms your documents into a training set where each sample contains:
1.  **A Question**.
2.  **Oracle Documents**: The actual documents containing the answer.
3.  **Distractor Documents**: IR-retrieved documents that are irrelevant to the answer.
4.  **Chain of Thought (CoT)**: A step-by-step reasoning path showing how to arrive at the answer using the Oracle docs.

### Learning to Ignore
The model is specifically trained to **ignore distractors**. This is the "secret sauce" of RAFT. It acts like a "Mental Filter" for the model.

---

## 🚀 The Training Loop: SFT meets RAG

Instead of just predicting the next token, RAFT uses **Supervised Fine-Tuning (SFT)** with a specific goal:
- **Conditioning**: The model learns to cite its sources.
- **Inhibition**: The model learns that "Document 4 says X, which is unrelated to the user's question about Y."
- **Focus**: The model learns to prioritize specific semantic markers in your domain's documents.

---

## 🛠️ Implementation in RAG Engine Mini

In this final level, we simulate the **Data Preparation Phase**:
- **Dataset Synthesis**: Converting local chunks into a JSONL format suitable for LoRA (Low-Rank Adaptation) training.
- **Distractor Injection**: Automating the process of picking "fake" documents to challenge the model.
- **Reasoning Generation**: Using a larger model (e.g., GPT-4o) to generate the "Gold" reasoning paths for training a smaller model (e.g., Llama-3-8B).

---

## 🏆 Summary for the Fine-Tuning Specialist

RAFT is the bridge between **Data** and **Model Weights**. By the end of this level, you understand how to not just build a RAG system, but how to **optimize the soul of the model** for your specific data.

---

## 📚 Advanced Reading
- Zhang et al. (UC Berkeley): "RAFT: Adapting Language Models for Retrieval-Augmented Generation"
- Meta AI: "Combining RAG and Fine-Tuning for Specialized Domains"
- Axolotl/Unsloth: Tools for efficient RAFT implementation.
