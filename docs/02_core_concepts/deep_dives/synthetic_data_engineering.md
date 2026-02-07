# 🔄 Deep-Dive: Synthetic Data Engineering (The Flywheel)

The ultimate secret of the world's best AI teams isn't just a better model; it's **better data**. This guide explains how to build an autonomous data flywheel using **RAG Engine Mini**.

---

## 🏗️ The Problem: The Evaluation Bottleneck
In classic software, we write Unit Tests. In RAG, we need a "Ground Truth" (Question + Correct Answer). 
- Manually writing 1,000 test cases takes weeks.
- Documents change, making manual test cases obsolete.

---

## 🚀 The Solution: Synthetic Data Generation (SDG)
We use a high-quality LLM (The Teacher) to look at our raw data and "teach" the RAG system (The Student).

### 1. Mining Ground Truth
The Teacher LLM reads a document chunk and performs **Inverse QA**:
- "Write a question that can only be answered by this chunk."
- "Write the correct answer based on this chunk."

### 2. Diversification
A good testset needs variety:
- **Factoid**: "Who founded X?"
- **Summarization**: "Summarize the three main pillars of Y."
- **Inference**: "Based on Z, what is likely to happen next?"

---

## 🌐 The Evaluation Loop (The Flywheel)
This project implements the full loop:
1.  **Extract**: Read PDFs/Docs.
2.  **Generate**: Use `scripts/generate_synthetic_testset.py` to create a benchmark.
3.  **Benchmark**: Run your RAG against the benchmark using `RAGAS`.
4.  **Analyze**: Find where the model fails (e.g., "It fails on tables!").
5.  **Optimize**: Improve the Table Extraction logic.
6.  **Verify**: Re-run the benchmark to prove the improvement.

---

## 🏁 Mastery: Zero-to-one
By building this flywheel, you have created a self-improving system. You no longer need human experts to spend thousands of hours labeling data. You have scaled human intelligence using AI.

**Welcome to the 1% of AI Engineers.** 💎
