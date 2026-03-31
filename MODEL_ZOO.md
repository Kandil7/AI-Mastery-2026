# Model Zoo for AI-Mastery-2026
# ==============================
# Pre-trained models and weights

## Available Models

### Classical ML

| Model | Dataset | Accuracy | Size | Download |
|-------|---------|----------|------|----------|
| Decision Tree | Iris | 97% | 2KB | [Link](#) |
| Random Forest | Wine | 98% | 15KB | [Link](#) |
| SVM | Digits | 99% | 125KB | [Link](#) |

### Deep Learning

| Model | Dataset | Accuracy | Size | Download |
|-------|---------|----------|------|----------|
| MLP | MNIST | 98% | 5MB | [Link](#) |
| CNN | CIFAR-10 | 85% | 25MB | [Link](#) |
| ResNet18 | ImageNet | 70% | 45MB | [Link](#) |
| LSTM | Shakespeare | - | 10MB | [Link](#) |

### LLM

| Model | Parameters | Context | Size | Download |
|-------|------------|---------|------|----------|
| TinyBERT | 4M | 128 | 15MB | [Link](#) |
| MiniGPT | 10M | 256 | 40MB | [Link](#) |

## Usage

```python
from src.ml import load_pretrained_model

# Load pre-trained model
model = load_pretrained_model("cnn_cifar10")

# Use for inference
predictions = model.predict(test_data)
```

## Training Your Own

See examples/ for training scripts.

---

**Last Updated:** March 31, 2026
