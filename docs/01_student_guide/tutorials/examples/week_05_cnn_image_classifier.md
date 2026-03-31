# Week 5: CNN Image Classification API

Complete implementation of a Convolutional Neural Network for image classification, deployed as a production API.

## Overview

Build and deploy a CNN-based image classifier for **CIFAR-10** dataset using custom implementations from `src/ml`, then serve it via FastAPI.

**Deliverables**:
1. Custom CNN architecture with ResNet blocks
2. Model training achieving >85% accuracy
3. Production-ready FastAPI endpoint
4. Performance benchmarks (<50ms p95 latency)

---

## Part 1: Enhanced CNN Architecture

### ResNet Block Implementation

Create `src/ml/vision.py`:

```python
import numpy as np
from .deep_learning import Layer

class Conv2D(Layer):
    ""\"Already exists in deep_learning.py - enhanced version""\"
    pass

class ResNetBlock(Layer):
    ""\"
    Residual block: F(x) + x
    Enables training of very deep networks
    ""\"
    def __init__(self, in_channels, out_channels, stride=1):
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2D(out_channels)
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2D(out_channels)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2D(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = Identity()
    
    def forward(self, x):
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection
        out += self.shortcut(x)
        out = relu(out)
        
        return out

class BatchNorm2D(Layer):
    ""\"Batch normalization for 2D convolutions""\"
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # Running stats
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
    def forward(self, x, training=True):
        ""\"
        x: (batch, channels, height, width)
        ""\"
        if training:
            # Compute batch statistics
            batch_mean = x.mean(axis=(0, 2, 3))
            batch_var = x.var(axis=(0, 2, 3))
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        x_norm = (x - mean.reshape(1, -1,  1)) / np.sqrt(var.reshape(1, -1, 1, 1) + self.eps)
        
        # Scale and shift
        out = self.gamma.reshape(1, -1, 1, 1) * x_norm + self.beta.reshape(1, -1, 1, 1)
        
        return out

class GlobalAveragePooling(Layer):
    ""\"Average over spatial dimensions""\"
    def forward(self, x):
        # x: (batch, channels, height, width) -> (batch, channels)
        return x.mean(axis=(2, 3))
```

### Complete CIFAR-10 CNN

```python
from src.ml.vision import Conv2D, ResNetBlock, BatchNorm2D, GlobalAveragePooling
from src.ml.deep_learning import Dense, Activation, Dropout, NeuralNetwork

class CIFAR10CNN(NeuralNetwork):
    def __init__(self):
        super().__init__()
        
        # Initial convolution
        self.add(Conv2D(3, 64, kernel_size=3, padding=1))
        self.add(BatchNorm2D(64))
        self.add(Activation('relu'))
        
        # ResNet blocks
        self.add(ResNetBlock(64, 64))
        self.add(ResNetBlock(64, 128, stride=2))  # Downsample
        self.add(ResNetBlock(128, 128))
        self.add(ResNetBlock(128, 256, stride=2))  # Downsample
        self.add(ResNetBlock(256, 256))
        
        # Global pooling
        self.add(GlobalAveragePooling())
        
        # Classifier
        self.add(Dropout(0.5))
        self.add(Dense(256, 10))
        self.add(Activation('softmax'))
```

---

## Part 2: Training Script

```python
import numpy as np
import torch
from torchvision import datasets, transforms

# Data loading
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Convert to NumPy
X_train = (train_dataset.data / 255.0).astype('float32')
y_train = np.array(train_dataset.targets)

X_test = (test_dataset.data / 255.0).astype('float32')
y_test = np.array(test_dataset.targets)

# Build model
model = CIFAR10CNN()
model.compile(loss=CrossEntropyLoss(), learning_rate=0.001)

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=128,
    validation_data=(X_test, y_test),
    verbose=True
)

# Evaluate
test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Save model
model.save('models/cifar10_cnn.pkl')
```

---

## Part 3: FastAPI Endpoint

```python
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import io

app = FastAPI(title="CIFAR-10 Image Classifier")

# Load model on startup
@app.on_event("startup")
async def load_model():
    global model
    model = CIFAR10CNN()
    model.load('models/cifar10_cnn.pkl')

@app.post("/classify/image")
async def classify_image(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Preprocess
    image = image.resize((32, 32))
    image_array = np.array(image) / 255.0
    image_array = image_array[np.newaxis, ...]  # Add batch dim
    
    # Predict
    with PREDICTION_LATENCY.time():
        proba = model.predict_proba(image_array)[0]
    
    predicted_class = CIFAR10_CLASSES[np.argmax(proba)]
    
    return {
        "class": predicted_class,
        "confidence": float(proba.max()),
        "all_probabilities": {
            CIFAR10_CLASSES[i]: float(proba[i]) 
            for i in range(10)
        }
    }

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
```

---

## Part 4: Benchmarking

```python
import time

# Latency benchmark
latencies = []
for _ in range(1000):
    start = time.time()
    model.predict(sample_image)
    latencies.append((time.time() - start) * 1000)

print(f"Latency p50: {np.percentile(latencies, 50):.2f}ms")
print(f"Latency p95: {np.percentile(latencies, 95):.2f}ms")
print(f"Latency p99: {np.percentile(latencies, 99):.2f}ms")

# Target: p95 < 50ms
```

---

**✅ Week 5 Complete**: CNN-based image classifier API with production deployment!
