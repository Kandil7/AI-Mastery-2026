# Glossary of AI and LLM Terms

**Last Updated:** March 28, 2026  
**Version:** 1.0.0

---

## 📖 Table of Contents

- [A](#a)
- [B](#b)
- [C](#c)
- [D](#d)
- [E](#e)
- [F](#f)
- [G](#g)
- [H](#h)
- [I](#i)
- [J](#j)
- [K](#k)
- [L](#l)
- [M](#m)
- [N](#n)
- [O](#o)
- [P](#p)
- [Q](#q)
- [R](#r)
- [S](#s)
- [T](#t)
- [U](#u)
- [V](#v)
- [W](#w)
- [X](#x)
- [Y](#y)
- [Z](#z)

---

## A

### **Activation Function**
A function that determines the output of a neural network node. Common examples include ReLU, Sigmoid, and Tanh.

**Example:**
```python
def relu(x):
    return max(0, x)
```

### **Agent**
An AI system that can perceive its environment, make decisions, and take actions to achieve goals.

**See:** [AI Agents](../kb/concepts/agents.md)

### **Attention Mechanism**
A technique that allows models to focus on relevant parts of the input when processing sequences.

**Types:**
- Self-Attention
- Cross-Attention
- Multi-Head Attention

**Formula:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### **API (Application Programming Interface)**
A set of protocols and tools for building software applications. In AI, APIs provide access to models and services.

### **Autoregressive Model**
A model that generates output token by token, where each token depends on previously generated tokens.

**Example:** GPT models are autoregressive.

---

## B

### **Backpropagation**
The algorithm used to train neural networks by computing gradients of the loss function with respect to model parameters.

**Process:**
1. Forward pass: Compute predictions
2. Compute loss
3. Backward pass: Compute gradients
4. Update parameters

### **Batch Size**
The number of training examples processed in one iteration before updating model parameters.

**Trade-offs:**
- Larger batch: Faster training, more memory
- Smaller batch: Better generalization, slower training

### **Beam Search**
A decoding strategy that explores multiple possible sequences simultaneously, keeping the top-k (beam width) candidates at each step.

### **Bias**
1. **Model Bias:** Systematic error in predictions
2. **Parameter:** Learnable offset in neural network layers

### **BPE (Byte Pair Encoding)**
A tokenization algorithm that merges frequent character pairs into tokens.

**Used by:** GPT-2, GPT-3, RoBERTa

---

## C

### **Chain of Thought (CoT)**
A prompting technique that encourages the model to show its reasoning step-by-step.

**Example:**
```
Let's solve this step by step. First, we need to...
```

### **Chunking**
The process of splitting documents into smaller pieces for RAG systems.

**Strategies:**
- Fixed-size chunking
- Semantic chunking
- Recursive chunking

### **Context Window**
The maximum number of tokens a model can process in a single input.

**Examples:**
- GPT-4: 128K tokens
- Llama 2: 4K tokens
- Claude 3: 200K tokens

### **Cosine Similarity**
A metric for measuring similarity between two vectors.

**Formula:**
$$\text{cosine}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

**Range:** -1 (opposite) to 1 (identical)

### **Cross-Encoder**
A model that processes two inputs together (e.g., query and document) for tasks like re-ranking.

**vs. Bi-Encoder:**
- Cross-Encoder: More accurate, slower
- Bi-Encoder: Faster, can pre-compute embeddings

---

## D

### **Dataset**
A collection of data used for training, validation, or testing machine learning models.

**Types:**
- Training set
- Validation set
- Test set

### **Decoder**
The part of a transformer that generates output sequences.

**See:** [Transformer Architecture](../kb/concepts/transformer.md)

### **Deep Learning**
A subset of machine learning using neural networks with multiple layers.

### **Dimensionality**
The number of features or components in a vector.

**Example:** An embedding with 1536 values has dimensionality 1536.

### **Distillation**
The process of training a smaller model (student) to mimic a larger model (teacher).

---

## E

### **Embedding**
A dense vector representation of text that captures semantic meaning.

**Properties:**
- Similar texts have similar vectors
- Can be used for similarity search

**Popular Models:**
- text-embedding-ada-002 (1536 dimensions)
- all-MiniLM-L6-v2 (384 dimensions)

### **Encoder**
The part of a transformer that processes input sequences.

### **Encoder-Decoder**
A transformer architecture with both encoder and decoder components.

**Used by:** T5, BART, mBART

### **Epoch**
One complete pass through the entire training dataset.

### **Evaluation Metrics**
Measures used to assess model performance.

**Common Metrics:**
- Accuracy
- Precision, Recall, F1
- Perplexity
- ROUGE, BLEU
- BERTScore

---

## F

### **Fine-Tuning**
The process of adapting a pre-trained model to a specific task or domain.

**Types:**
- Full fine-tuning (all parameters)
- Parameter-efficient (LoRA, QLoRA)
- Instruction tuning

### **Few-Shot Learning**
Learning from a small number of examples provided in the prompt.

**Example:**
```
Example 1: Input: "Great!" → Sentiment: Positive
Example 2: Input: "Terrible." → Sentiment: Negative
Input: "Not bad." → Sentiment: ?
```

### **FAISS (Facebook AI Similarity Search)**
A library for efficient similarity search and clustering of dense vectors.

### **Forward Pass**
The computation of model output given input, without gradient computation.

---

## G

### **GAN (Generative Adversarial Network)**
A generative model consisting of a generator and discriminator trained adversarially.

### **GPU (Graphics Processing Unit)**
A processor optimized for parallel computations, commonly used for deep learning.

### **Gradient**
The derivative of the loss function with respect to model parameters, used for optimization.

### **Gradient Descent**
An optimization algorithm that minimizes loss by moving parameters in the direction of negative gradient.

**Variants:**
- SGD (Stochastic Gradient Descent)
- Adam
- RMSprop

### **Greedy Decoding**
A simple decoding strategy that always selects the most likely next token.

**vs. Beam Search:**
- Greedy: Fast, may miss better sequences
- Beam Search: Slower, explores more options

---

## H

### **Hallucination**
When an AI model generates false or fabricated information.

**Mitigation:**
- Use RAG systems
- Add fact-checking
- Improve training data

### **Hidden Layer**
A layer in a neural network between input and output layers.

### **HNSW (Hierarchical Navigable Small World)**
An algorithm for approximate nearest neighbor search in vector databases.

### **Hyperparameter**
A configuration parameter set before training (not learned from data).

**Examples:**
- Learning rate
- Batch size
- Number of layers

---

## I

### **In-Context Learning**
The ability of LLMs to learn from examples provided in the prompt without weight updates.

### **Inference**
The process of using a trained model to make predictions.

### **Instruction Tuning**
Fine-tuning a model on instruction-following datasets to improve task completion.

### **IVF (Inverted File Index)**
A vector indexing method that partitions the vector space for faster search.

---

## K

### **K-Nearest Neighbors (KNN)**
An algorithm that finds the k most similar items to a query.

**Used in:** Vector search, recommendation systems

### **Knowledge Distillation**
See [Distillation](#d)

### **KV Cache**
A caching technique that stores key-value pairs from attention layers to speed up inference.

---

## L

### **Latency**
The time taken to process a request and return a response.

### **Layer Normalization**
A technique to normalize activations within a layer, improving training stability.

### **Learning Rate**
A hyperparameter that controls the step size during optimization.

### **LLM (Large Language Model)**
A language model with billions of parameters trained on vast amounts of text.

**Examples:** GPT-4, Claude, Llama, PaLM

### **LoRA (Low-Rank Adaptation)**
A parameter-efficient fine-tuning method that adds low-rank matrices to model weights.

**Benefits:**
- 10,000x fewer parameters to train
- No inference latency
- Easy task switching

### **Loss Function**
A function that measures how well model predictions match the ground truth.

**Common Losses:**
- Cross-Entropy (classification)
- MSE (regression)
- Contrastive Loss (embeddings)

---

## M

### **Masking**
A technique to prevent attention to certain positions (e.g., future tokens in causal attention).

### **Memory**
In AI agents, the component that stores and retrieves information across interactions.

**Types:**
- Short-term memory (context window)
- Long-term memory (vector database)

### **Model**
A mathematical representation learned from data to make predictions.

### **Multi-Head Attention**
An attention mechanism that computes attention multiple times in parallel with different projections.

### **Multi-Modal**
Systems that process multiple types of data (text, images, audio).

---

## N

### **N-gram**
A contiguous sequence of n items (words, characters) from text.

### **Neural Network**
A computational model inspired by biological neurons, consisting of layers of interconnected nodes.

### **NLP (Natural Language Processing)**
The field of AI focused on enabling computers to understand and generate human language.

### **Normalization**
Techniques to scale and stabilize neural network activations.

**Types:**
- Layer Normalization
- Batch Normalization
- RMS Normalization

---

## O

### **Optimization**
The process of adjusting model parameters to minimize loss.

**Optimizers:**
- SGD
- Adam
- AdamW

### **Overfitting**
When a model learns the training data too well, including noise, and performs poorly on new data.

**Prevention:**
- Regularization
- Dropout
- Early stopping
- More data

---

## P

### **Parameter**
A learnable variable in a model (weights and biases).

### **Perplexity**
A metric for language models that measures how well the model predicts a sample.

**Formula:**
$$\text{Perplexity} = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log P(w_i)\right)$$

**Lower is better.**

### **Positional Encoding**
Information added to embeddings to provide sequence order in transformers.

**Types:**
- Sinusoidal (original transformer)
- Learned (BERT)
- RoPE (Rotary Positional Embeddings)

### **Precision**
The fraction of retrieved instances that are relevant.

### **Prompt**
The input text provided to an LLM to elicit a response.

### **Prompt Engineering**
The practice of designing effective prompts to get desired outputs.

### **Pruning**
Removing unnecessary parameters from a model to reduce size and improve efficiency.

---

## Q

### **QLoRA (Quantized LoRA)**
A variant of LoRA that uses quantization for even more efficient fine-tuning.

### **Quantization**
Reducing the numerical precision of model weights to save memory and computation.

**Types:**
- INT8 (8-bit)
- INT4 (4-bit)
- FP16 (16-bit float)

### **Query**
The input provided to a retrieval system or search engine.

---

## R

### **RAG (Retrieval-Augmented Generation)**
A system that combines retrieval of relevant documents with generation of answers.

**Components:**
- Retriever (finds documents)
- Generator (produces answers)

**See:** [RAG Guide](../kb/concepts/rag.md)

### **Recall**
The fraction of relevant instances that are retrieved.

### **Re-ranking**
The process of re-ordering retrieved results using a more accurate (but slower) model.

### **Reinforcement Learning**
A type of learning where agents learn from rewards and penalties.

### **Residual Connection**
A skip connection that adds the input of a layer to its output, helping with gradient flow.

### **Retrieval**
The process of finding relevant documents from a collection.

### **RLHF (Reinforcement Learning from Human Feedback)**
A technique for aligning models with human preferences using reinforcement learning.

### **RoPE (Rotary Positional Embeddings)**
A positional encoding method that uses rotation matrices.

**Used by:** Llama, PaLM

### **RNN (Recurrent Neural Network)**
A neural network architecture for sequential data, predecessor to transformers.

---

## S

### **Self-Attention**
An attention mechanism where a sequence attends to itself.

### **Similarity Search**
Finding items similar to a query, typically using vector embeddings.

### **Softmax**
A function that converts scores to probabilities.

**Formula:**
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

### **Sparse Attention**
Attention mechanisms that only attend to a subset of positions, reducing computation.

### **Step**
One iteration of training (processing one batch and updating parameters).

### **Supervised Learning**
Learning from labeled examples (input-output pairs).

---

## T

### **Temperature**
A parameter that controls randomness in generation.

**Effects:**
- Low (0.1-0.3): More focused, deterministic
- High (0.7-1.0): More creative, random

### **Token**
A unit of text (word or subword) that models process.

**Example:** "Hello world" → ["Hello", " world"]

### **Tokenization**
The process of splitting text into tokens.

**Methods:**
- Word-based
- Subword (BPE, WordPiece)
- Character-based

### **Top-k Sampling**
A decoding strategy that samples from the top k most likely tokens.

### **Top-p Sampling (Nucleus Sampling)**
A decoding strategy that samples from the smallest set of tokens whose cumulative probability exceeds p.

### **Transformer**
A neural network architecture based on self-attention, introduced in "Attention Is All You Need" (2017).

**Components:**
- Self-Attention
- Feed-Forward Networks
- Layer Normalization
- Residual Connections

### **Transfer Learning**
Using knowledge from one task to improve performance on another.

### **TPU (Tensor Processing Unit)**
A specialized processor designed for machine learning workloads.

---

## U

### **Underfitting**
When a model is too simple to capture patterns in the data.

### **Unsupervised Learning**
Learning from unlabeled data, finding patterns without explicit targets.

---

## V

### **Validation Set**
A dataset used to tune hyperparameters and monitor for overfitting.

### **Vector Database**
A database optimized for storing and searching vector embeddings.

**Examples:** Chroma, Pinecone, Weaviate, Qdrant

### **Vector Search**
Finding similar vectors using similarity metrics like cosine similarity.

### **Vocabulary**
The set of all tokens a model can process.

---

## W

### **Weight**
A learnable parameter in a neural network that determines the strength of connections.

### **WordPiece**
A tokenization algorithm used by BERT and related models.

---

## Z

### **Zero-Shot Learning**
Performing a task without any examples, relying only on the model's pre-trained knowledge.

**Example:**
```
Classify the sentiment: "I love this!"
```

### **Zipf's Law**
A linguistic principle stating that word frequency is inversely proportional to rank.

---

## 📚 Related Resources

- [Concepts](../kb/concepts/) - Detailed concept explanations
- [Tutorials](../tutorials/) - Hands-on learning
- [FAQ](../faq/) - Common questions
- [API Reference](../api/) - Technical documentation

---

**Last Updated:** March 28, 2026  
**Terms Defined:** 150+  
**Maintained By:** AI-Mastery-2026 Documentation Team
