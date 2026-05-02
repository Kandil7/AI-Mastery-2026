# Convolutional Neural Networks (CNNs)

## Introduction

**Convolutional Neural Networks (CNNs)** are specialized neural networks designed for processing grid-structured data, most famously images. They automatically learn spatial hierarchies of features through convolutional layers.

### Why CNNs?

| Traditional ML | CNNs |
|---------------|------|
| Flatten images → lose spatial info | Preserves 2D structure |
| Manual feature engineering | Automatic feature learning |
| Parameters per pixel | Shared weights (parameters) |
| Not invariant to translation | Built-in translation invariance |

---

## 1. Convolution Operation

### 1.1 The Core Idea

Convolution slides a small **kernel** (filter) over the input to produce an output:

```
Convolution Operation:

Input (5x5):          Kernel (3x3):         Output (3x3):
                    
1  1  1  0  0       1  0  1              4  3  4
0  1  1  1  0       0  1  0              3  4  3
0  0  1  1  1  ──►  1  0  1    ──────►    4  3  4
0  0  1  1  0       ╱                   ╱
0  0  0  1  0                          ╱

Step: Multiply corresponding elements and sum
      (1×1 + 1×0 + 1×1) + (0×0 + 1×1 + 1×0) + (1×1 + 0×0 + 1×1) = 4
```

### 1.2 Mathematical Definition

For a 2D convolution:

$$(I * K)_{i,j} = \sum_m \sum_n I_{i+m, j+n} \cdot K_{m,n}$$

Where:
- $I$ = input image
- $K$ = kernel/filter
- $*$ = convolution operator

### 1.3 Key Properties

| Property | Description | Benefit |
|----------|-------------|---------|
| **Local connectivity** | Each output depends on local input region | Reduces parameters |
| **Parameter sharing** | Same kernel used everywhere | Fewer parameters, translation invariant |
| **Translation invariance** | Shift in input → shift in output | Robust to translations |

---

## 2. CNN Architecture

### 2.1 Typical Structure

```
Input Image
     │
     ▼
┌─────────────┐
│  Conv Layer │ ──► Extract features
└─────────────┘
     │
     ▼
┌─────────────┐
│  ReLU       │ ──► Non-linearity
└─────────────┘
     │
     ▼
┌─────────────┐
│  Pooling    │ ──► Downsample
└─────────────┘
     │
     ▼
     ... (repeat)
     │
     ▼
┌─────────────┐
│  FC Layers  │ ──► Classification
└─────────────┘
     │
     ▼
  Predictions
```

### 2.2 Layer Types

#### Convolution Layer
```
Conv Layer:
- Filters: Learnable kernels (e.g., 32 filters of 3x3)
- Activation: ReLU typically
- Output: Feature maps
```

#### Pooling Layer
```
Max Pooling (2x2, stride=2):

Input:              Output:
1  3  2  1         6  8
0  6  5  4    ─►   3  4
2  1  3  1
1  3  2  1

Take max of each 2x2 block
```

#### Fully Connected Layer
```
FC Layer:
- Flatten features → connect to output
- Same as regular neural network layer
- Used at the end for classification
```

---

## 3. Convolutional Layer Details

### 3.1 Hyperparameters

| Parameter | Description | Typical Values |
|-----------|-------------|-----------------|
| **Filters (K)** | Number of kernels | 32, 64, 128, 256 |
| **Kernel size (F)** | Spatial size of filter | 3×3, 5×5, 7×7 |
| **Stride (S)** | Step size of sliding | 1 (common), 2 (downsample) |
| **Padding (P)** | Border of zeros | 0 (valid), 1 (same) |

### 3.2 Output Size Calculation

$$O = \left\lfloor \frac{I - F + 2P}{S} \right\rfloor + 1$$

Where:
- $O$ = output size
- $I$ = input size
- $F$ = kernel size
- $P$ = padding
- $S$ = stride

**Example:**
- Input: 32×32
- Kernel: 3×3
- Padding: 1
- Stride: 1

$$O = \lfloor \frac{32 - 3 + 2 \times 1}{1} \rfloor + 1 = 32$$

### 3.3 Number of Parameters

For a conv layer:
$$\text{Parameters} = (F \times F \times C_{in}) \times C_{out} + C_{out}$$

Example: 3×3 kernel, 3 input channels, 64 filters:
$$(3 \times 3 \times 3) \times 64 + 64 = 1,728 + 64 = 1,792 \text{ parameters}$$

---

## 4. What Do Filters Learn?

### 4.1 Layer-by-Layer Features

```
Layer 1: Edge detection, colors, textures
    ┌───────────────────────┐
    │ ╱╲  │  ╱╲   |  ╱╲  │  │
    │╱  ╲│ │╱  ╲  │╱  ╲ │  │
    └───────────────────────┘

Layer 2: Parts detection (eyes, wheels, patterns)
    ┌───────────────────────┐
    │ (•)   (•)  │ ▫   ▫  │  │
    │   ◯       │  ◯    ◯ │  │
    └───────────────────────┘

Layer 3: Objects (faces, cars, cats)
    ┌───────────────────────┐
    │   🚗     │   🐱    │  │
    │  car    │   cat   │  │
    └───────────────────────┘
```

### 4.2 Feature Visualization

```
Filter Activation Maps:

Input Image         Filter 1          Filter 2          Filter N
   ┌───┐              ┌───┐            ┌───┐            ┌───┐
   │   │ ═══════════► │ █ │ ════════► │   │ ════════► │ ▓ │
   │   │              │ █ │            │   │            │ ▓ │
   └───┘              └───┘            └───┘            └───┘

   Original           Edge              Corner           Texture
   image              detection         detection        patterns
```

---

## 5. Classic CNN Architectures

### 5.1 LeNet-5 (1998)

```
LeNet-5 Architecture:
Input (32×32) → Conv(6, 5×5) → AvgPool(2×2) → Conv(16, 5×5) → 
AvgPool(2×2) → FC(120) → FC(84) → Output(10)

Parameters: ~60,000
```

### 5.2 AlexNet (2012)

```
AlexNet Architecture:
Input (227×227) → Conv(96, 11×11, s=4) → MaxPool(3×3) → 
Conv(256, 5×5) → MaxPool → Conv(384, 3×3) → 
Conv(384, 3×3) → Conv(256, 3×3) → MaxPool → 
FC(4096) → FC(4096) → Output(1000)

Key innovations:
- ReLU activation
- Dropout
- Data augmentation
- GPU training
Parameters: ~60 million
```

### 5.3 VGG-16 (2014)

```
VGG-16:
- 16 weight layers (13 conv + 3 FC)
- All conv: 3×3 kernels
- All pool: 2×2 stride

Pattern: [conv3-64]×2 → pool → [conv3-128]×2 → pool → 
[conv3-256]×3 → pool → [conv3-512]×3 → pool → [conv3-512]×3 → pool → FC

Parameters: ~138 million
```

### 5.4 ResNet (2015)

```
ResNet Key Innovation: Residual Connection

Standard:        With Residual:
                  
x ──┬──► ──┐      x ──┬──► + ──► H(x)
    │      │          │    ↑
    ▼      │          ▼    │
   Conv    │         Conv ──┘
    │      │           │
    ▼      ▼           ▼
   + ─────►          F(x) + x

Allows training of very deep networks (50, 101, 152 layers)
```

---

## 6. Implementation from Scratch

### 6.1 Convolution Operation

```python
import numpy as np

def conv2d(input_data, kernel, stride=1, padding=0):
    """
    2D Convolution operation.
    
    Parameters
    ----------
    input_data : ndarray of shape (H, W) or (C, H, W)
    kernel : ndarray of shape (K, K)
    stride : int, default=1
    padding : int, default=0
    
    Returns
    -------
    output : ndarray
        Convolved feature map
    """
    # Handle 2D or 3D input
    if input_data.ndim == 2:
        input_data = input_data[np.newaxis, :, :]
    
    C_in, H_in, W_in = input_data.shape
    K = kernel.shape[0]
    
    # Apply padding
    if padding > 0:
        padded = np.pad(input_data, ((0, 0), (padding, padding), (padding, padding)))
    else:
        padded = input_data
    
    # Calculate output size
    H_out = (H_in + 2 * padding - K) // stride + 1
    W_out = (W_in + 2 * padding - K) // stride + 1
    
    output = np.zeros((C_in, H_out, W_out))
    
    # Convolution
    for c in range(C_in):
        for i in range(0, H_in + 2 * padding - K + 1, stride):
            for j in range(0, W_in + 2 * padding - K + 1, stride):
                patch = padded[c, i:i+K, j:j+K]
                output[c, i//stride, j//stride] = np.sum(patch * kernel)
    
    return output.squeeze() if output.shape[0] == 1 else output
```

### 6.2 Max Pooling

```python
def max_pool2d(input_data, pool_size=2, stride=2):
    """
    2D Max Pooling operation.
    """
    if input_data.ndim == 2:
        input_data = input_data[np.newaxis, :, :]
    
    C, H, W = input_data.shape
    
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    output = np.zeros((C, H_out, W_out))
    
    for c in range(C):
        for i in range(0, H - pool_size + 1, stride):
            for j in range(0, W - pool_size + 1, stride):
                patch = input_data[c, i:i+pool_size, j:j+pool_size]
                output[c, i//stride, j//stride] = np.max(patch)
    
    return output.squeeze() if output.shape[0] == 1 else output
```

### 6.3 Simple CNN Class

```python
class SimpleCNN:
    """Simple CNN for image classification."""
    
    def __init__(self, n_classes=10):
        self.n_classes = n_classes
        
        # Initialize filters (first conv layer learns edge detection)
        self.filters = [
            np.random.randn(3, 3),  # Edge detector
            np.random.randn(3, 3),  # Color detector
            np.random.randn(3, 3),  # Texture detector
        ]
        
        # FC layer weights
        self.fc_weights = None
        self.fc_bias = None
    
    def forward(self, x):
        """Forward pass through the network."""
        # Store for backprop
        self.cache = [x]
        
        # First conv layer
        for f in self.filters:
            conv_result = conv2d(x, f, stride=1, padding=1)
            if 'conv_out' in locals():
                conv_out = np.maximum(conv_out, conv_result)  # Max filter response
            else:
                conv_out = conv_result
        
        self.cache.append(conv_out)
        
        # ReLU
        relu_out = np.maximum(conv_out, 0)
        self.cache.append(relu_out)
        
        # Pooling
        pooled = max_pool2d(relu_out, pool_size=2, stride=2)
        self.cache.append(pooled)
        
        # Flatten
        flattened = pooled.flatten().reshape(1, -1)
        
        # Simple FC (placeholder for a real network)
        if self.fc_weights is None:
            n_features = flattened.shape[1]
            self.fc_weights = np.random.randn(n_features, self.n_classes) * 0.01
            self.fc_bias = np.zeros((1, self.n_classes))
        
        # Output (softmax)
        scores = flattened @ self.fc_weights + self.fc_bias
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        
        return probs[0]
    
    def predict(self, x):
        """Predict class."""
        probs = self.forward(x)
        return np.argmax(probs)
```

---

## 7. Data Augmentation

### 7.1 Why Augmentation?

- Increases effective training data
- Reduces overfitting
- Makes model robust to variations

### 7.2 Common Augmentations

| Augmentation | Effect |
|--------------|--------|
| **Flip** | Horizontal/Vertical flip |
| **Rotate** | Random rotation |
| **Crop** | Random crop |
| **Color** | Brightness, contrast, saturation |
| **Zoom** | Random zoom in/out |
| **Cutout** | Random patches masked |
| **MixUp** | Blend two images |

### 7.3 Implementation

```python
def augment_image(image):
    """Apply random augmentations."""
    # Random flip
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
    
    # Random rotation (90 degree increments)
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    
    # Random brightness
    brightness = np.random.uniform(0.8, 1.2)
    image = np.clip(image * brightness, 0, 1)
    
    # Random crop
    if np.random.rand() > 0.5:
        h, w = image.shape[:2]
        crop_size = int(min(h, w) * 0.9)
        i = np.random.randint(0, h - crop_size + 1)
        j = np.random.randint(0, w - crop_size + 1)
        image = image[i:i+crop_size, j:j+crop_size]
    
    return image
```

---

## 8. Transfer Learning

### 8.1 Concept

Use pretrained model weights as starting point:

```
Transfer Learning:

1. Take pretrained model (trained on ImageNet)
2. Freeze early layers (learned general features)
3. Fine-tune later layers for your task

Pretrained Features:
Layer 1-3: General (edges, colors, textures) ──► Keep
Layer 4-5: Specific (objects) ──► Fine-tune
FC layers: Task-specific ──► Replace & train
```

### 8.2 When to Use

| Scenario | Approach |
|----------|----------|
| Small dataset (< 10K images) | Freeze all, train only FC |
| Medium dataset | Fine-tune later layers |
| Large dataset | Fine-tune entire network |

---

## 9. Summary

| Concept | Key Points |
|---------|------------|
| **Convolution** | Slides kernel over input, extracts local features |
| **Parameters** | Shared weights, local connectivity |
| **Filters** | Learn edges → textures → parts → objects |
| **Pooling** | Downsampling, reduces spatial size |
| **FC layers** | Final classification |
| **Residual** | Enables very deep networks |

**Key Insight**: CNNs leverage spatial structure - early layers detect local patterns, later layers combine them into global features. This hierarchical feature learning is what makes CNNs so powerful for vision tasks.