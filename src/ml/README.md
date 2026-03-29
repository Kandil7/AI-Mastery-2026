# Machine Learning Module

**Classical and deep learning implementations from scratch.**

## Topics Covered

### Classical ML (`classical/`)
- Decision Trees (ID3, C4.5)
- Random Forests
- Gradient Boosting
- SVM (SMO algorithm)
- Naive Bayes
- K-Means Clustering
- PCA

### Deep Learning (`deep_learning/`)
- Autograd engine
- Dense layers
- Conv2D layers
- LSTM/GRU
- ResNet18
- Transformer blocks

### Computer Vision (`vision/`)
- Image classification
- Object detection
- Semantic segmentation
- Data augmentation

### Graph Neural Networks (`gnn_recommender/`)
- GraphSAGE
- Two-Tower architecture
- BPR/InfoNCE losses

### Reinforcement Learning
- Policy Gradients
- PPO
- DQN

## Usage

```python
from src.ml.classical import DecisionTree, RandomForest
from src.ml.deep_learning import NeuralNetwork, Dense, ReLU

# Classical ML
clf = DecisionTree(max_depth=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Deep Learning
model = NeuralNetwork([
    Dense(784, 128),
    ReLU(),
    Dense(128, 10),
])
model.compile(optimizer='adam', loss='cross_entropy')
model.fit(X_train, y_train, epochs=10)
```

## Implementation Philosophy

Build understanding from the ground up:
1. Implement algorithms from scratch
2. Understand the mathematics
3. Then use production libraries

## Related Modules

- [`src/core`](../core/) - Mathematical foundations
- [`src/llm`](../llm/) - Large language models
- [`src/evaluation`](../evaluation/) - Model evaluation
