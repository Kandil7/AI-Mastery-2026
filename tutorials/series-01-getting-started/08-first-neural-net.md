# 🧠 Tutorial 8: Your First Neural Network

**Build and train a neural network from scratch in 90 minutes**

---

## 🎯 What You'll Build

By the end of this tutorial, you will have:

- ✅ Understood how neurons work
- ✅ Built a neural network from scratch with NumPy
- ✅ Implemented forward and backward propagation
- ✅ Trained on MNIST digit classification
- ✅ Achieved 90%+ accuracy
- ✅ Visualized training progress

**Time Required:** 90 minutes  
**Difficulty:** ⭐⭐⭐☆☆ (Intermediate)  
**Prerequisites:** Tutorials 1-7, basic calculus, linear algebra

---

## 📋 What You'll Learn

- Biological vs artificial neurons
- Network architecture (layers, neurons, activations)
- Forward propagation
- Backpropagation and chain rule
- Gradient descent for neural networks
- Training loop and epochs
- Evaluating model performance

---

## 🧠 Step 1: Understand Neural Networks (10 minutes)

### Biological Inspiration

```
Biological Neuron:
Dendrites → Cell Body → Axon → Synapses
     (input)   (process)   (output)
```

### Artificial Neuron (Perceptron)

```
Inputs (x₁, x₂, ..., xₙ)
    ↓
Weights (w₁, w₂, ..., wₙ)
    ↓
Sum: z = Σ(wᵢ × xᵢ) + b
    ↓
Activation: a = σ(z)
    ↓
Output
```

### Multi-Layer Network

```
Input Layer → Hidden Layer → Hidden Layer → Output Layer
   (784)          (128)          (64)           (10)
```

For MNIST (28×28 images):
- **Input:** 784 pixels (flattened)
- **Hidden:** 128 neurons (ReLU activation)
- **Hidden:** 64 neurons (ReLU activation)
- **Output:** 10 neurons (Softmax for digits 0-9)

### Activation Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| **Sigmoid** | 1/(1+e⁻ᶻ) | Output (binary) |
| **ReLU** | max(0, z) | Hidden layers |
| **Softmax** | eᶻⁱ/Σeᶻʲ | Output (multi-class) |
| **Tanh** | (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ) | Hidden layers |

---

## 🛠️ Step 2: Setup (5 minutes)

### Install Dependencies

```bash
# Install required libraries
pip install numpy matplotlib
pip install scikit-learn  # For MNIST dataset

# Verify
python -c "import numpy; print('✅ NumPy ready')"
python -c "from sklearn.datasets import fetch_openml; print('✅ sklearn ready')"
```

### Create Notebook

```python
# Create file: tutorial_08_neural_network.ipynb
# Open in VS Code or Jupyter Lab
```

### Import Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

print("✅ All libraries loaded!")
```

---

## 📊 Step 3: Load MNIST Data (10 minutes)

### What is MNIST?

MNIST is a dataset of 70,000 handwritten digits (0-9):
- **60,000 training images**
- **10,000 test images**
- **28×28 pixels** (grayscale)
- **Labels:** 0-9

### Load Dataset

```python
print("📥 Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data  # Images (70000, 784)
y = mnist.target.astype(int)  # Labels (70000,)

print(f"✅ Dataset loaded!")
print(f"   Images: {X.shape[0]}")
print(f"   Pixels per image: {X.shape[1]}")
print(f"   Classes: {np.unique(y)}")
```

### Normalize and Split

```python
# Normalize pixel values to [0, 1]
X = X / 255.0

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42, stratify=y
)

print(f"\n📊 Data Split:")
print(f"   Training: {X_train.shape[0]} samples")
print(f"   Testing: {X_test.shape[0]} samples")
```

### Visualize Sample Images

```python
# Display sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flatten()):
    # Find sample of digit i
    idx = np.where(y_train == i)[0][0]
    image = X_train[idx].reshape(28, 28)
    
    ax.imshow(image, cmap='gray')
    ax.set_title(f'Label: {y_train[idx]}', fontsize=14)
    ax.axis('off')

plt.tight_layout()
plt.savefig('01_mnist_samples.png', dpi=300)
plt.show()

print("✅ Sample images displayed!")
```

---

## 🏗️ Step 4: Build Neural Network (30 minutes)

### Network Architecture

```
Input (784) → Hidden1 (128, ReLU) → Hidden2 (64, ReLU) → Output (10, Softmax)
```

### Implementation

```python
class NeuralNetwork:
    """Neural Network implemented from scratch with NumPy."""
    
    def __init__(self, layer_sizes):
        """
        Initialize neural network.
        
        Args:
            layer_sizes: List of neurons per layer
                        e.g., [784, 128, 64, 10]
        """
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
        
        print(f"✅ Network initialized: {layer_sizes}")
    
    def relu(self, z):
        """ReLU activation function."""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivative of ReLU."""
        return (z > 0).astype(float)
    
    def softmax(self, z):
        """Softmax activation for output layer."""
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward propagation.
        
        Args:
            X: Input data (m, n_features)
        
        Returns:
            activations: List of activations
            z_values: List of pre-activation values
        """
        activations = [X]
        z_values = []
        
        current = X
        
        # Hidden layers (ReLU)
        for i in range(len(self.weights) - 1):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # ReLU activation
            current = self.relu(z)
            activations.append(current)
        
        # Output layer (Softmax)
        z = np.dot(current, self.weights[-1]) + self.biases[-1]
        z_values.append(z)
        output = self.softmax(z)
        activations.append(output)
        
        return activations, z_values
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute cross-entropy loss.
        
        Args:
            y_true: One-hot encoded true labels
            y_pred: Predicted probabilities
        
        Returns:
            loss: Cross-entropy loss
        """
        m = y_true.shape[0]
        # Clip to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss
    
    def backward(self, X, y_true, activations, z_values, learning_rate=0.01):
        """
        Backpropagation.
        
        Args:
            X: Input data
            y_true: One-hot encoded true labels
            activations: From forward pass
            z_values: From forward pass
            learning_rate: Learning rate
        """
        m = X.shape[0]
        
        # Output layer error (softmax + cross-entropy)
        delta = activations[-1] - y_true
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Calculate gradients
            dw = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            # Update weights and biases
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db
            
            # Propagate error to previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(z_values[i-1])
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=100, learning_rate=0.01, batch_size=64):
        """
        Train the neural network.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Mini-batch size
        
        Returns:
            history: Training history
        """
        # One-hot encode labels
        label_binarizer = LabelBinarizer()
        y_train_onehot = label_binarizer.fit_transform(y_train)
        y_val_onehot = label_binarizer.transform(y_val)
        
        m = X_train.shape[0]
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        print(f"\n🏋️ Training Neural Network...")
        print(f"   Epochs: {epochs}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Batch size: {batch_size}")
        print(f"   Training samples: {m}")
        print("-" * 60)
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(m)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_onehot[indices]
            
            # Mini-batch training
            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Forward pass
                activations, z_values = self.forward(X_batch)
                
                # Backward pass
                self.backward(X_batch, y_batch, activations, z_values, learning_rate)
            
            # Compute metrics every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                # Training metrics
                train_activations, _ = self.forward(X_train)
                train_loss = self.compute_loss(y_train_onehot, train_activations[-1])
                train_preds = np.argmax(train_activations[-1], axis=1)
                train_acc = np.mean(train_preds == y_train)
                
                # Validation metrics
                val_activations, _ = self.forward(X_val)
                val_loss = self.compute_loss(y_val_onehot, val_activations[-1])
                val_preds = np.argmax(val_activations[-1], axis=1)
                val_acc = np.mean(val_preds == y_val)
                
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f}")
        
        print("-" * 60)
        print(f"✅ Training complete!")
        print(f"   Final Train Accuracy: {train_acc:.4f}")
        print(f"   Final Val Accuracy: {val_acc:.4f}")
        
        return history
    
    def predict(self, X):
        """Predict class labels."""
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)
    
    def score(self, X, y):
        """Calculate accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
```

---

## 🏋️ Step 5: Train the Network (20 minutes)

### Initialize and Train

```python
# Split training into train/validation
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=10000, random_state=42
)

# Create neural network
nn = NeuralNetwork([784, 128, 64, 10])

# Train
history = nn.train(
    X_train_final, y_train_final,
    X_val, y_val,
    epochs=100,
    learning_rate=0.01,
    batch_size=64
)
```

### Visualize Training

```python
# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
epochs_list = range(1, len(history['train_loss']) + 1)
axes[0].plot(epochs_list, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
axes[0].plot(epochs_list, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=14)
axes[0].set_ylabel('Loss', fontsize=14)
axes[0].set_title('Training & Validation Loss', fontsize=16)
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)

# Accuracy plot
axes[1].plot(epochs_list, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
axes[1].plot(epochs_list, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=14)
axes[1].set_ylabel('Accuracy', fontsize=14)
axes[1].set_title('Training & Validation Accuracy', fontsize=16)
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_training_curves.png', dpi=300)
plt.show()

print("✅ Training curves plotted!")
```

---

## 📊 Step 6: Evaluate Performance (10 minutes)

### Test Set Evaluation

```python
# Evaluate on test set
test_acc = nn.score(X_test, y_test)
print(f"🎯 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# Predictions
y_pred = nn.predict(X_test)
```

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.title('Confusion Matrix - MNIST Test Set', fontsize=16)
plt.tight_layout()
plt.savefig('03_confusion_matrix.png', dpi=300)
plt.show()

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
```

### Visualize Predictions

```python
# Show correct and incorrect predictions
correct = np.where(y_pred == y_test)[0]
incorrect = np.where(y_pred != y_test)[0]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Show 5 correct predictions
for i in range(5):
    idx = correct[np.random.randint(0, len(correct))]
    image = X_test[idx].reshape(28, 28)
    axes[0, i].imshow(image, cmap='gray')
    axes[0, i].set_title(f'✓ Pred: {y_pred[idx]}', fontsize=12, color='green')
    axes[0, i].axis('off')

# Show 5 incorrect predictions
for i in range(5):
    idx = incorrect[np.random.randint(0, len(incorrect))]
    image = X_test[idx].reshape(28, 28)
    axes[1, i].imshow(image, cmap='gray')
    axes[1, i].set_title(f'✗ Pred: {y_pred[idx]}, True: {y_test[idx]}', 
                        fontsize=12, color='red')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('04_predictions.png', dpi=300)
plt.show()

print("✅ Predictions visualized!")
```

---

## 🎯 Step 7: Understanding Check (5 minutes)

### Knowledge Check

**Q1:** What does ReLU stand for?

A) Rectified Linear Unit  
B) Regularized Learning Unit  
C) Random Linear Update  
D) Recurrent Layer Unit  

**Answer:** A) Rectified Linear Unit

---

**Q2:** Why do we use softmax in the output layer?

A) To make all values positive  
B) To convert outputs to probabilities  
C) To speed up training  
D) To prevent overfitting  

**Answer:** B) To convert outputs to probabilities

---

**Q3:** What is backpropagation?

A) Forward pass through the network  
B) Computing gradients using chain rule  
C) Initializing weights  
D) Normalizing inputs  

**Answer:** B) Computing gradients using chain rule

---

## ✅ Tutorial Checklist

- [ ] Understood neural network architecture
- [ ] Loaded and preprocessed MNIST data
- [ ] Implemented forward propagation
- [ ] Implemented backpropagation
- [ ] Trained network for 100 epochs
- [ ] Achieved 90%+ accuracy
- [ ] Visualized training curves
- [ ] Analyzed confusion matrix

---

## 🎓 Key Takeaways

1. **Neurons** - Basic computational units
2. **Layers** - Input, hidden, output
3. **Forward Pass** - Compute predictions
4. **Backward Pass** - Compute gradients
5. **Gradient Descent** - Update weights
6. **Activations** - ReLU for hidden, Softmax for output
7. **Training** - Iterate until convergence

---

## 🚀 Next Steps

1. **Experiment:**
   - Try different architectures
   - Adjust learning rate
   - Add dropout for regularization
   - Train for more epochs

2. **Continue Learning:**
   - Tier 2, Module 2.5: Neural Networks from Scratch
   - Tier 2, Module 2.6: PyTorch Fundamentals

3. **Build Projects:**
   - Image classifier for your own dataset
   - Handwritten digit recognizer app
   - Fashion MNIST classifier

---

## 💡 Challenge (Optional)

**Improve the network!**

1. Add more hidden layers
2. Try different activation functions
3. Implement dropout
4. Add learning rate scheduling
5. Achieve 95%+ accuracy

**Share your best accuracy in Discord!** 🏆

---

**Tutorial Created:** April 2, 2026  
**Last Updated:** April 2, 2026  
**Estimated Time:** 90 minutes  
**Difficulty:** Intermediate

---

[← Back to Tutorials](../README.md) | [Previous: CSV Data](07-csv-data.md) | [Next: Git for AI Projects](09-git-basics.md)
