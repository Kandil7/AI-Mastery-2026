# Vision Module Usage Examples

Complete guide to using the computer vision module (`src/ml/vision.py`).

---

## Table of Contents
1. [Basic Conv2D Usage](#basic-conv2d-usage)
2. [Building Custom CNNs](#building-custom-cnns)
3. [Using ResNet18](#using-resnet18)
4. [Image Classification Pipeline](#image-classification-pipeline)
5. [Training on CIFAR-10](#training-on-cifar-10)

---

## Basic Conv2D Usage

### Simple Convolution

```python
import numpy as np
from src.ml.vision import Conv2D

# Create a convolutional layer
conv = Conv2D(
    in_channels=3,      # RGB input
    out_channels=64,    # 64 filters
    kernel_size=3,      # 3x3 kernel
    stride=1,
    padding=1
)

# Input: batch of 8 images, 3 channels, 32x32 pixels
input_data = np.random.randn(8, 3, 32, 32)

# Forward pass
output = conv.forward(input_data, training=True)
print(f"Input shape: {input_data.shape}")    # (8, 3, 32, 32)
print(f"Output shape: {output.shape}")       # (8, 64, 32, 32)

# Backward pass (training)
output_gradient = np.random.randn(*output.shape)
input_gradient = conv.backward(output_gradient, learning_rate=0.01)
print(f"Gradient shape: {input_gradient.shape}")  # (8, 3, 32, 32)
```

### MaxPooling

```python
from src.ml.vision import MaxPool2D

# Create pooling layer
pool = MaxPool2D(pool_size=2, stride=2)

# Downsample feature maps
input_data = np.random.randn(8, 64, 32, 32)
output = pool.forward(input_data)
print(f"After pooling: {output.shape}")  # (8, 64, 16, 16)
```

---

## Building Custom CNNs

### Simple CNN for MNIST

```python
from src.ml.vision import Conv2D, MaxPool2D, Flatten
from src.ml.deep_learning import Dense, Activation, CrossEntropyLoss

class SimpleCNN:
    """Simple CNN for 28x28 grayscale images."""
    
    def __init__(self, num_classes=10):
        # Convolutional layers
        self.conv1 = Conv2D(1, 32, kernel_size=3, padding=1)
        self.relu1 = Activation('relu')
        self.pool1 = MaxPool2D(pool_size=2)
        
        self.conv2 = Conv2D(32, 64, kernel_size=3, padding=1)
        self.relu2 = Activation('relu')
        self.pool2 = MaxPool2D(pool_size=2)
        
        # Fully connected layers
        self.flatten = Flatten()
        self.fc1 = Dense(64 * 7 * 7, 128)
        self.relu3 = Activation('relu')
        self.fc2 = Dense(128, num_classes)
    
    def forward(self, x, training=True):
        # Conv block 1
        x = self.conv1.forward(x, training)
        x = self.relu1.forward(x, training)
        x = self.pool1.forward(x, training)
        
        # Conv block 2
        x = self.conv2.forward(x, training)
        x = self.relu2.forward(x, training)
        x = self.pool2.forward(x, training)
        
        # Classifier
        x = self.flatten.forward(x, training)
        x = self.fc1.forward(x, training)
        x = self.relu3.forward(x, training)
        x = self.fc2.forward(x, training)
        
        return x

# Usage
model = SimpleCNN(num_classes=10)
X = np.random.randn(32, 1, 28, 28)  # Batch of 32 MNIST images
logits = model.forward(X)
print(f"Logits shape: {logits.shape}")  # (32, 10)
```

---

## Using ResNet18

### Basic ResNet18 Usage

```python
from src.ml.vision import ResNet18

# Create ResNet18 model
model = ResNet18(
    in_channels=3,     # RGB images
    num_classes=10     # CIFAR-10
)

# Print architecture
model.summary()

# Forward pass
X = np.random.randn(8, 3, 224, 224)  # Batch of 8 images
output = model.forward(X, training=False)
print(f"Output shape: {output.shape}")  # (8, 512, H', W')
```

### ResNet with Custom Input Size

```python
# For CIFAR-10 (32x32 images)
model = ResNet18(in_channels=3, num_classes=10)
X_cifar = np.random.randn(16, 3, 32, 32)
output = model.forward(X_cifar, training=False)
print(f"CIFAR-10 output: {output.shape}")

# For ImageNet (224x224 images)
model_imagenet = ResNet18(in_channels=3, num_classes=1000)
X_imagenet = np.random.randn(4, 3, 224, 224)
output = model_imagenet.forward(X_imagenet, training=False)
print(f"ImageNet output: {output.shape}")
```

---

## Image Classification Pipeline

### Complete End-to-End Pipeline

```python
import numpy as np
from PIL import Image
from src.ml.vision import ResNet18
from src.ml.deep_learning import CrossEntropyLoss

class ImageClassifier:
    """End-to-end image classification pipeline."""
    
    def __init__(self, num_classes=10):
        self.model = ResNet18(in_channels=3, num_classes=num_classes)
        self.loss_fn = CrossEntropyLoss()
        self.class_names = None
    
    def preprocess_image(self, image_path):
        """Load and preprocess an image."""
        # Load image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((32, 32))  # CIFAR-10 size
        
        # To numpy array (HWC format)
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # Convert to CHW format
        img_array = img_array.transpose(2, 0, 1)
        
        # Add batch dimension
        return img_array[np.newaxis, ...]
    
    def predict(self, image_path):
        """Predict class for an image."""
        # Preprocess
        X = self.preprocess_image(image_path)
        
        # Forward pass
        logits = self.model.forward(X, training=False)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Get prediction
        pred_idx = np.argmax(probabilities[0])
        confidence = probabilities[0, pred_idx]
        
        result = {
            'class_idx': int(pred_idx),
            'confidence': float(confidence),
            'all_probabilities': probabilities[0].tolist()
        }
        
        if self.class_names:
            result['class_name'] = self.class_names[pred_idx]
        
        return result
    
    def train_step(self, X_batch, y_batch, learning_rate=0.001):
        """Single training step."""
        # Forward pass
        logits = self.model.forward(X_batch, training=True)
        
        # Compute loss
        loss = self.loss_fn.forward(logits, y_batch)
        
        # Backward pass (simplified - full version would backprop through all layers)
        # grad = self.loss_fn.backward(logits, y_batch)
        # self.model.backward(grad, learning_rate)
        
        # Compute accuracy
        predictions = np.argmax(logits, axis=1)
        accuracy = np.mean(predictions == y_batch)
        
        return {'loss': loss, 'accuracy': accuracy}

# Usage
classifier = ImageClassifier(num_classes=10)
classifier.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                          'dog', 'frog', 'horse', 'ship', 'truck']

# Predict single image
result = classifier.predict('test_image.jpg')
print(f"Predicted: {result['class_name']} ({result['confidence']:.2%})")
```

---

## Training on CIFAR-10

### Complete Training Loop

```python
import numpy as np
from src.ml.vision import ResNet18
from src.ml.deep_learning import CrossEntropyLoss

def train_cifar10(model, X_train, y_train, X_val, y_val, 
                  epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train ResNet18 on CIFAR-10.
    
    Args:
        model: ResNet18 instance
        X_train: Training images (N, 3, 32, 32)
        y_train: Training labels (N,)
        X_val: Validation images
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    
    Returns:
        history: Dictionary with training metrics
    """
    loss_fn = CrossEntropyLoss()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    n_batches = len(X_train) // batch_size
    
    for epoch in range(epochs):
        # Training phase
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        for i in range(n_batches):
            # Get batch
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # Forward pass
            logits = model.forward(X_batch, training=True)
            loss = loss_fn.forward(logits, y_batch)
            
            # Backward pass (simplified)
            # In full implementation, would compute gradients and update weights
            
            # Track metrics
            epoch_loss += loss
            predictions = np.argmax(logits, axis=1)
            epoch_correct += np.sum(predictions == y_batch)
            epoch_total += len(y_batch)
        
        # Compute training metrics
        train_loss = epoch_loss / n_batches
        train_acc = epoch_correct / epoch_total
        
        # Validation phase
        val_logits = model.forward(X_val, training=False)
        val_loss = loss_fn.forward(val_logits, y_val)
        val_predictions = np.argmax(val_logits, axis=1)
        val_acc = np.mean(val_predictions == y_val)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return history

# Example usage
# Load CIFAR-10 data (replace with actual data loading)
X_train = np.random.randn(5000, 3, 32, 32)
y_train = np.random.randint(0, 10, 5000)
X_val = np.random.randn(1000, 3, 32, 32)
y_val = np.random.randint(0, 10, 1000)

# Create and train model
model = ResNet18(in_channels=3, num_classes=10)
history = train_cifar10(model, X_train, y_train, X_val, y_val, epochs=10)

print(f"\nFinal validation accuracy: {history['val_acc'][-1]:.2%}")
```

---

## Advanced Topics

### Data Augmentation

```python
def augment_images(images, training=True):
    """
    Apply data augmentation to images.
    
    Args:
        images: (N, C, H, W) numpy array
        training: Whether to apply augmentations
    
    Returns:
        Augmented images
    """
    if not training:
        return images
    
    augmented = images.copy()
    
    # Random horizontal flip (50% chance)
    flip_mask = np.random.rand(len(images)) > 0.5
    augmented[flip_mask] = augmented[flip_mask, :, :, ::-1]
    
    # Random crop (with padding)
    # (Implementation would pad and randomly crop)
    
    return augmented

# Usage in training loop
X_batch_augmented = augment_images(X_batch, training=True)
logits = model.forward(X_batch_augmented, training=True)
```

### Transfer Learning

```python
# Load pretrained weights (conceptual)
def load_pretrained_resnet(model, weights_path):
    """Load pretrained weights into ResNet model."""
    # In practice, would load from file and map to model parameters
    # model.conv1.filters = pretrained['conv1.weight']
    # etc.
    pass

# Fine-tune on new dataset
model = ResNet18(in_channels=3, num_classes=100)  # 100 new classes
load_pretrained_resnet(model, 'resnet18_imagenet.npz')

# Freeze early layers, only train final layers
# (Would set requires_grad=False for conv layers in PyTorch equivalent)
```

---

## Performance Tips

1. **Batch Size**: Use largest batch size that fits in memory (32-128 typical)
2. **Learning Rate**: Start with 0.001, use cosine annealing or step decay
3. **Data Augmentation**: Essential for small datasets (CIFAR-10)
4. **Mixed Precision**: Convert to PyTorch/TensorFlow for FP16 training
5. **Parallel Processing**: Use DataLoader with multiple workers

---

## See Also

- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [CIFAR-10 Notebook](../notebooks/week_05/resnet_cifar10.ipynb)
- [API Documentation](API_ENHANCEMENTS.md)
