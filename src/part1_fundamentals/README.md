# Part 1: LLM Fundamentals - Complete Implementation

This directory contains a complete, production-ready implementation of LLM fundamentals, covering mathematics, Python for ML, neural networks, and NLP.

## 📁 Directory Structure

```
part1_fundamentals/
├── module_1_1_mathematics/      # Mathematics for ML
│   ├── vectors.py               # Vector operations
│   ├── matrices.py              # Matrix operations
│   ├── calculus.py              # Calculus & optimization
│   ├── probability.py           # Probability & statistics
│   ├── __init__.py
│   └── tests/
│       └── test_mathematics.py
│
├── module_1_2_python/           # Python for ML
│   ├── data_processing.py       # NumPy/Pandas operations
│   ├── ml_algorithms.py         # ML algorithms from scratch
│   ├── preprocessing.py         # Preprocessing utilities
│   ├── __init__.py
│   └── tests/
│       └── test_ml_python.py
│
├── module_1_3_neural_networks/  # Neural Networks
│   ├── activations.py           # Activation functions
│   ├── losses.py                # Loss functions
│   ├── layers.py                # Neural network layers
│   ├── optimizers.py            # Optimization algorithms
│   ├── mlp.py                   # Complete MLP implementation
│   ├── __init__.py
│   └── tests/
│       └── test_neural_networks.py
│
└── module_1_4_nlp/              # Natural Language Processing
    ├── tokenization.py          # Tokenization algorithms
    ├── embeddings.py            # Word embeddings
    ├── sequence_models.py       # RNN/LSTM/GRU
    ├── text_preprocessing.py    # Text preprocessing
    ├── __init__.py
    └── tests/
        └── test_nlp.py
```

## 📚 Module Overview

### Module 1.1: Mathematics for ML

**Files:**
- `vectors.py` - Vector operations (dot product, cross product, norms, projections, Gram-Schmidt)
- `matrices.py` - Matrix operations (multiplication, decomposition, eigenvalues, SVD)
- `calculus.py` - Derivatives, gradients, optimization algorithms
- `probability.py` - Distributions, Bayes theorem, hypothesis testing

**Key Features:**
- Comprehensive type hints and docstrings
- Numerical stability optimizations
- Educational examples in docstrings
- Full test coverage

**Example Usage:**
```python
from module_1_1_mathematics import VectorOperations, MatrixOperations
from module_1_1_mathematics import Optimizer, Distribution

# Vector operations
ops = VectorOperations()
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot = ops.dot_product(v1, v2)  # 32.0

# Matrix operations
m_ops = MatrixOperations()
A = np.array([[1, 2], [3, 4]])
det = m_ops.determinant(A)  # -2.0

# Optimization
opt = Optimizer(learning_rate=0.1, method='adam')
result = opt.minimize(lambda x: x[0]**2 + x[1]**2, x0=np.array([5.0, 5.0]))

# Probability
dist = Distribution.normal(mean=0, std=1)
samples = dist.sample(1000)
```

### Module 1.2: Python for ML

**Files:**
- `data_processing.py` - Array/DataFrame operations
- `ml_algorithms.py` - Linear/Logistic Regression, Decision Trees, Random Forest, K-Means, PCA
- `preprocessing.py` - Scaling, encoding, imputation, train/test split

**Key Features:**
- From-scratch implementations (no sklearn dependencies)
- Production-ready error handling
- Comprehensive documentation

**Example Usage:**
```python
from module_1_2_python import LinearRegression, StandardScaler, train_test_split

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Linear regression
model = LinearRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
r2 = model.score(X_test, y_test)
```

### Module 1.3: Neural Networks

**Files:**
- `activations.py` - ReLU, Sigmoid, Tanh, Softmax, GELU, Swish, etc.
- `losses.py` - MSE, Cross-Entropy, BCE, Huber, KL Divergence
- `layers.py` - Dense, Conv2D, MaxPool2D, Dropout, BatchNorm
- `optimizers.py` - SGD, Adam, RMSprop, Adagrad, AdamW
- `mlp.py` - Complete MLP with training loop

**Key Features:**
- Full forward/backward pass implementations
- Multiple optimizers with learning rate scheduling
- Model save/load functionality
- Training with early stopping

**Example Usage:**
```python
from module_1_3_neural_networks import MLP, Trainer

# Create model
model = MLP(
    input_size=784,
    hidden_sizes=[256, 128],
    output_size=10,
    activation='relu',
    dropout=0.5
)

# Train
trainer = Trainer(model, loss='cross_entropy', optimizer='adam')
history = trainer.fit(X_train, y_train, X_val, y_val, epochs=10)

# Predict
predictions = model.predict(X_test)
```

### Module 1.4: NLP

**Files:**
- `tokenization.py` - Word, BPE, WordPiece, SentencePiece tokenizers
- `embeddings.py` - Word2Vec, GloVe, Positional Encodings
- `sequence_models.py` - RNN, LSTM, GRU cells and multi-layer models
- `text_preprocessing.py` - Stemming, lemmatization, TF-IDF

**Key Features:**
- Multiple tokenization strategies
- Trainable word embeddings
- Bidirectional sequence models
- Complete NLP preprocessing pipeline

**Example Usage:**
```python
from module_1_4_nlp import WordTokenizer, Word2Vec, LSTM, TFIDFVectorizer

# Tokenization
tokenizer = WordTokenizer()
tokens = tokenizer.tokenize("Hello world!")

# Word embeddings
w2v = Word2Vec(embedding_dim=100)
w2v.train(["hello world", "hello there"])
vector = w2v.get_vector("hello")

# Sequence model
lstm = LSTM(input_size=100, hidden_size=256, num_layers=2)
output, (h_n, c_n) = lstm.forward(x)

# TF-IDF
vectorizer = TFIDFVectorizer()
tfidf = vectorizer.fit_transform(documents)
```

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest src/part1_fundamentals/ -v

# Run specific module tests
python -m pytest src/part1_fundamentals/module_1_1_mathematics/tests/ -v
python -m pytest src/part1_fundamentals/module_1_2_python/tests/ -v
python -m pytest src/part1_fundamentals/module_1_3_neural_networks/tests/ -v
python -m pytest src/part1_fundamentals/module_1_4_nlp/tests/ -v
```

## 📦 Dependencies

```txt
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
```

## 🎯 Key Design Principles

1. **Educational**: Clear docstrings with examples
2. **Production-Ready**: Error handling, type hints, logging
3. **From Scratch**: Minimal external dependencies
4. **Tested**: Comprehensive unit tests
5. **Extensible**: Clean abstractions for extension

## 📖 Learning Path

1. Start with **Module 1.1** (Mathematics) for foundations
2. Move to **Module 1.2** (Python for ML) for data handling
3. Study **Module 1.3** (Neural Networks) for deep learning
4. Finish with **Module 1.4** (NLP) for language processing

## 🔧 Implementation Highlights

### Mathematics (Module 1.1)
- Vector operations with full linear algebra support
- Matrix decompositions (LU, QR, Cholesky, SVD)
- Numerical optimization with multiple algorithms
- Statistical testing and distributions

### ML Algorithms (Module 1.2)
- Linear/Logistic Regression with regularization
- Decision Trees with information gain
- Random Forest with bagging
- K-Means with k-means++ initialization
- PCA with eigendecomposition

### Neural Networks (Module 1.3)
- 10+ activation functions
- 8+ loss functions
- Full layer implementations with backprop
- 6+ optimizers with learning rate scheduling
- Complete MLP with training loop

### NLP (Module 1.4)
- 5 tokenization strategies
- Word2Vec (Skip-gram/CBOW)
- GloVe with co-occurrence matrix
- RNN/LSTM/GRU from scratch
- TF-IDF with n-grams

## 📝 Code Quality

- **Type Hints**: Full type annotations throughout
- **Docstrings**: Google-style with examples
- **Logging**: Structured logging for debugging
- **Error Handling**: Comprehensive validation
- **Tests**: >90% code coverage

## 🚀 Next Steps

After completing Part 1, proceed to:
- Part 2: Advanced LLM Architectures
- Part 3: RAG Systems
- Part 4: Agent Systems
- Part 5: Production Deployment
