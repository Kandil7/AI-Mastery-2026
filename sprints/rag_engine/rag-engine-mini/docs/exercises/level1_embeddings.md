# 🧪 Exercise 1: Semantic Search & Embeddings

## 🎯 Objective
Understand how text is converted into vectors and how "Distance" equals "Meaning".

## 📝 Task
1.  Open `src/adapters/embeddings/openai.py` (or your local adapter).
2.  Create a strict list of 5 sentences:
    *   "The cat sat on the mat."
    *   "A feline is resting on the rug."
    *   "The dog chases the ball."
    *   "SpaceX launched a rocket."
    *   "Apple released the new iPhone."
3.  Calculate the **Cosine Similarity** between the first sentence ("The cat...") and all others.

## ❓ Questions to Answer
1.  Does "feline/rug" have a higher score than "dog/ball"? Why?
2.  What happens if you use a totally different language (e.g., "Le chat est sur le tapis")?
3.  Modify `config.py` to switch the Embedding Model (e.g., from `text-embedding-3-small` to a local HuggingFace model). How do the scores change?

## 💡 Hints
*   Use `scikit-learn` or `numpy` for cosine calculation: `dot(A, B) / (norm(A) * norm(B))`.
*   Check `src/application/services/embedding_cache.py` to see how we store these vectors to save money.
