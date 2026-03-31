# 🧪 Exercise 1: Understanding Embeddings and Semantic Search

## 🎯 Objective
Understand how text is converted into vectors and how "distance" equals "meaning" in vector space.

## 📋 Prerequisites
- Basic understanding of vectors and dimensions
- Familiarity with Python and Jupyter notebooks
- Completed basic setup of the RAG system

## 🧪 Exercise Tasks

### Task 1: Explore Embedding Generation
1. Locate the embedding service in `src/core/embeddings.py`
2. Run the embedding function on these three sentences:
   - "The cat sat on the mat"
   - "A feline rested on the rug" 
   - "The weather is sunny today"
3. Calculate the cosine similarity between each pair
4. Verify that sentences 1 & 2 are more similar than 1 & 3

### Task 2: Understand Dimensionality
1. Examine the dimension of your chosen embedding model
2. Explain why higher dimensions can better represent meaning
3. Discuss the trade-offs of high-dimensional embeddings

### Task 3: Experiment with Similarity Measures
1. Try different similarity measures (cosine, euclidean, dot product)
2. Compare results and explain differences
3. Determine which measure works best for your use case

## 🛠️ Implementation Hints
```python
from src.core.embeddings import get_embeddings

# Get embeddings for the sentences
sentences = [
    "The cat sat on the mat",
    "A feline rested on the rug",
    "The weather is sunny today"
]

embeddings = [get_embeddings(sentence) for sentence in sentences]

# Calculate similarities
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sim_matrix = cosine_similarity(embeddings)
print("Similarity matrix:")
print(sim_matrix)
```

## 🧠 Reflection Questions
1. Why do semantically similar sentences have similar embeddings?
2. What are the limitations of this approach?
3. How might domain-specific embeddings improve performance?

## 📊 Success Criteria
- Successfully generate embeddings for all sentences
- Correctly identify that sentences 1 & 2 are most similar
- Understand the relationship between semantic similarity and vector distance
- Can articulate the trade-offs of different similarity measures

## 🚀 Challenge Extension
Research and implement a method to visualize these embeddings in 2D space using t-SNE or PCA.