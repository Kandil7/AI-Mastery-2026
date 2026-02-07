# Case Study 15: Computer Vision for Quality Control in Manufacturing

## Executive Summary

**Problem**: A manufacturing company producing automotive parts faced 8% defect rate causing recalls and warranty claims worth $3.2M annually.

**Solution**: Implemented a real-time computer vision system using custom CNN to detect defects in automotive brake pads with 99.2% accuracy and <50ms inference time.

**Impact**: Reduced defect rate from 8% to 0.6%, saving $2.8M annually in recalls and warranty claims, while increasing customer satisfaction scores by 23%.

**System design snapshot**:
- SLOs: p99 <50ms inference time; 99.2% accuracy; zero-defect tolerance for safety-critical parts.
- Scale: ~50,000 parts inspected daily across 10 production lines; 24/7 operation.
- Cost guardrails: < $0.005 per inspection; hardware costs amortized over 2 years.
- Data quality gates: image quality checks; automatic retraining triggers on accuracy drift.
- Reliability: redundant cameras and inference systems; fail-safe defaults to human inspection.

---

## Business Context

### Company Profile
- **Industry**: Automotive Parts Manufacturing
- **Production Volume**: 50,000 brake pads daily
- **Defect Rate**: 8% (industry average 2-3%)
- **Problem**: Manual inspection unreliable and costly

### Key Challenges
1. High defect rate leading to recalls and warranty claims
2. Manual inspection inconsistent and slow
3. Need for 100% inspection coverage (not sampling)
4. Safety-critical parts requiring zero-defect tolerance

---

## Technical Approach

### Architecture Overview

```
Conveyor Belt -> Camera System -> Image Preprocessing -> CNN Inference -> Defect Classification
                                                              |
                                                              v
                                                 Pass/Fail Decision -> Human Verification
```

### Data Collection and Preprocessing

**Image Acquisition**:
- High-resolution industrial cameras (5MP) positioned at 4 angles
- LED lighting system for consistent illumination
- Conveyor speed synchronized with camera trigger
- Images captured at 60fps during part movement

**Dataset Creation**:
- 50,000 labeled images (35,000 normal, 15,000 defective)
- Defect categories: surface cracks, dimensional errors, material inconsistencies
- Data augmentation: rotation, scaling, brightness adjustment
- Train/validation/test split: 70/15/15

```python
import cv2
import numpy as np
from PIL import Image
import albumentations as A

def preprocess_image(image_path):
    """Preprocess image for defect detection"""
    img = cv2.imread(image_path)
    
    # Resize to standard dimensions
    img = cv2.resize(img, (224, 224))
    
    # Normalize pixel values
    img = img.astype(np.float32) / 255.0
    
    # Apply augmentations during training
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
    ])
    
    if is_training:
        img = transform(image=img)['image']
    
    return img
```

### Model Architecture

**Custom CNN Implementation**:
```python
from src.ml.vision import Conv2D, MaxPool2D, Dense, Flatten
import numpy as np

class BrakePadDefectCNN:
    def __init__(self):
        self.conv1 = Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)
        
        self.conv2 = Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)
        
        self.conv3 = Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = MaxPool2D(kernel_size=2, stride=2)
        
        self.flatten = Flatten()
        self.fc1 = Dense(128 * 28 * 28, 512)  # Adjusted for 224x224 input
        self.fc2 = Dense(512, 2)  # Binary classification
        
    def forward(self, x):
        x = self.conv1.forward(x)
        x = np.maximum(0, x)  # ReLU
        x = self.pool1.forward(x)
        
        x = self.conv2.forward(x)
        x = np.maximum(0, x)  # ReLU
        x = self.pool2.forward(x)
        
        x = self.conv3.forward(x)
        x = np.maximum(0, x)  # ReLU
        x = self.pool3.forward(x)
        
        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = np.maximum(0, x)  # ReLU
        x = self.fc2.forward(x)
        
        # Softmax for probabilities
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
```

---

## Model Development

### Approach Comparison

| Model | Accuracy | Precision | Recall | F1 | Inference Time | Notes |
|-------|----------|-----------|--------|-----|----------------|-------|
| Traditional CV (Edge Detection) | 0.72 | 0.68 | 0.71 | 0.69 | 25ms | Simple but limited |
| Custom CNN | 0.94 | 0.93 | 0.95 | 0.94 | 35ms | Good balance |
| ResNet-18 | 0.96 | 0.95 | 0.97 | 0.96 | 45ms | Better accuracy |
| **Custom Optimized CNN** | **0.992** | **0.99** | **0.994** | **0.992** | **42ms** | **Selected** |

**Selected Model**: Optimized Custom CNN
- **Reason**: Best balance of accuracy and inference speed for real-time requirements
- **Architecture**: 5 convolutional layers with batch normalization and dropout

### Hyperparameter Tuning

```python
best_params = {
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 100,
    'dropout_rate': 0.3,
    'weight_decay': 1e-4,
    'momentum': 0.9
}
```

### Training Process

```python
def train_model(model, train_loader, val_loader, epochs, learning_rate):
    """Training loop for defect detection model"""
    optimizer = SGD(learning_rate=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Forward pass
            outputs = model.forward(data)
            
            # Compute loss
            loss = cross_entropy_loss(outputs, targets)
            
            # Backward pass
            gradients = compute_gradients(loss, model)
            optimizer.update(model, gradients)
            
            train_loss += loss
            
        # Validation
        val_accuracy = evaluate_model(model, val_loader)
        
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Accuracy: {val_accuracy:.4f}')
```

### Cross-Validation
- **Strategy**: Stratified k-fold (k=5) to maintain defect class balance
- **Validation Accuracy**: 0.991 +/- 0.003
- **Test Accuracy**: 0.992

---

## Production Deployment

### Hardware Infrastructure

**Edge Computing Setup**:
- NVIDIA Jetson AGX Xavier (32GB RAM, 512-core GPU)
- Industrial cameras with Gigabit Ethernet interface
- Real-time conveyor belt control system
- Redundant backup systems

### Software Architecture

```
Camera Feed -> Preprocessing -> Model Inference -> Decision Engine -> Actuator Control
                |                    |                  |                |
                v                    v                  v                v
         Quality Check      Defect Classification  Pass/Fail Logic   Reject/Accept
```

### Real-Time Inference Pipeline

```python
import time
import numpy as np
from threading import Thread
import queue

class RealTimeDefectDetector:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        
    def capture_frames(self):
        """Continuously capture frames from camera"""
        cap = cv2.VideoCapture(0)  # Industrial camera
        
        while True:
            ret, frame = cap.read()
            if ret:
                processed_frame = preprocess_image(frame)
                
                try:
                    self.frame_queue.put_nowait(processed_frame)
                except queue.Full:
                    # Drop oldest frame if queue full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(processed_frame)
                    except queue.Empty:
                        pass
                        
    def process_frames(self):
        """Process frames with model inference"""
        while True:
            try:
                frame = self.frame_queue.get(timeout=1)
                
                # Model inference
                start_time = time.time()
                prediction = self.model.predict(frame[np.newaxis, ...])
                inference_time = time.time() - start_time
                
                # Classification
                defect_probability = prediction[0][1]  # Probability of defect
                is_defective = defect_probability > 0.5
                
                result = {
                    'defective': is_defective,
                    'confidence': defect_probability,
                    'inference_time_ms': inference_time * 1000
                }
                
                # Put result in queue
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    pass  # Skip result if queue full
                    
            except queue.Empty:
                continue
                
    def start_detection(self):
        """Start real-time detection threads"""
        capture_thread = Thread(target=self.capture_frames)
        process_thread = Thread(target=self.process_frames)
        
        capture_thread.start()
        process_thread.start()
        
        return capture_thread, process_thread
```

### Quality Control Integration

**Decision Logic**:
```python
def make_inspection_decision(defect_result):
    """Make pass/fail decision based on model output"""
    
    if defect_result['defective']:
        # High confidence defect - reject immediately
        if defect_result['confidence'] > 0.9:
            return {'action': 'reject', 'reason': 'high_confidence_defect'}
        
        # Medium confidence - human review
        elif defect_result['confidence'] > 0.7:
            return {'action': 'review', 'reason': 'medium_confidence_defect'}
        
        # Low confidence defect - pass with note
        else:
            return {'action': 'pass_with_note', 'reason': 'low_confidence_defect'}
    
    else:
        # High confidence pass - accept
        if defect_result['confidence'] > 0.9:
            return {'action': 'accept', 'reason': 'high_confidence_pass'}
        
        # Low confidence pass - human review
        else:
            return {'action': 'review', 'reason': 'low_confidence_pass'}
```

### Operational SLOs and Runbook
- **Inference Latency**: p99 <50ms; system monitors and alerts if exceeded
- **Accuracy Target**: Maintain >99% accuracy; automatic retraining triggered if below 98.5%
- **Uptime**: 99.9% availability; redundant systems for failover
- **Runbook Highlights**:
  - Camera calibration: monthly, with automated quality checks
  - Model drift: monitor accuracy daily, retrain weekly
  - Hardware failures: automatic failover to backup system within 30 seconds

### Monitoring and Alerting
- **Metrics**: Accuracy, precision, recall, inference time, throughput
- **Alerts**: Page if accuracy drops below 98.5% or inference time exceeds 50ms
- **Drift Detection**: Monitor prediction distribution and trigger retraining if significant shift

---

## Results & Impact

### Model Performance in Production

**Overall Performance**:
- **Accuracy**: 99.2%
- **Precision**: 99.0%
- **Recall**: 99.4%
- **F1 Score**: 99.2%
- **Inference Time**: 42ms (p99)

**Per-Defect Type Performance**:
| Defect Type | Precision | Recall | F1 Score |
|-------------|-----------|--------|----------|
| Surface Cracks | 0.98 | 0.99 | 0.985 |
| Dimensional Errors | 0.99 | 0.98 | 0.985 |
| Material Inconsistencies | 0.97 | 0.99 | 0.980 |

### Business Impact (12 months post-launch)

| Metric | Before ML | After ML | Improvement |
|--------|-----------|----------|-------------|
| **Defect Rate** | 8% | 0.6% | **-92.5%** |
| **Recalls** | 12/year | 1/year | **-91.7%** |
| **Warranty Claims** | $3.2M/year | $0.4M/year | **-87.5%** |
| **Inspection Speed** | 10 parts/min | 100 parts/min | **+900%** |
| **Human Labor** | 10 inspectors | 2 inspectors | **-80%** |
| **Customer Satisfaction** | 3.2/5 | 3.9/5 | **+21.9%** |

### Cost-Benefit Analysis

**Annual Savings**:
- Reduced recalls: $2.4M
- Reduced warranty claims: $0.4M
- Labor cost reduction: $180K
- **Total Annual Benefit**: $2.98M

**Investment**:
- Hardware (cameras, edge computers): $150K
- Software development: $200K
- Installation and integration: $100K
- **Total Investment**: $450K

**ROI**: 564% in first year ($2.98M/$0.45M)

### Feature Importance Analysis

**Visual Features Contributing to Defect Detection**:
1. **Edge discontinuities**: 0.28 (surface cracks detection)
2. **Texture variations**: 0.22 (material inconsistencies)
3. **Color uniformity**: 0.18 (staining, discoloration)
4. **Geometric measurements**: 0.17 (dimensional accuracy)
5. **Surface reflectivity**: 0.15 (finish quality)

**Insights**:
- **Edge analysis** is most important for crack detection
- **Texture analysis** crucial for material defects
- **Geometric measurements** essential for dimensional accuracy

---

## Challenges & Solutions

### Challenge 1: Lighting Variations
- **Problem**: Factory lighting changes throughout day affecting image quality
- **Solution**:
  - Installed consistent LED lighting system
  - Added brightness normalization preprocessing
  - Trained model with various lighting conditions

### Challenge 2: Real-Time Performance
- **Problem**: Need <50ms inference for production line speed
- **Solution**:
  - Optimized model architecture for speed
  - Used NVIDIA Jetson for edge inference
  - Implemented multi-threading for pipeline efficiency

### Challenge 3: Class Imbalance
- **Problem**: Only 8% of parts were defective (before implementation)
- **Solution**:
  - Synthetic defect generation using GANs
  - Focal loss function to address imbalance
  - Oversampling of minority class

### Challenge 4: Safety-Critical Requirements
- **Problem**: Zero tolerance for missing safety-critical defects
- **Solution**:
  - Conservative threshold setting (0.5 instead of 0.7)
  - Human verification for medium-confidence cases
  - Redundant inspection systems

---

## Lessons Learned

### What Worked

1. **Edge Computing Approach**
   - Real-time inference without network latency
   - Reduced bandwidth requirements
   - Improved reliability with offline operation

2. **Comprehensive Data Augmentation**
   - Significantly improved model generalization
   - Better performance on unseen lighting conditions
   - Reduced overfitting to training data

3. **Human-in-the-Loop Design**
   - Maintained safety with human oversight
   - Continuous learning from human corrections
   - Gradual automation increase over time

### What Didn't Work

1. **Initial Simple CV Approach**
   - Edge detection alone insufficient for complex defects
   - Too many false positives/negatives
   - Needed deep learning for feature representation

2. **Cloud-Based Inference**
   - Network latency too high for real-time requirements
   - Connectivity issues caused production delays
   - Switched to edge computing for reliability

---

## Technical Implementation

### Model Training Code

```python
import numpy as np
from src.ml.vision import Conv2D, MaxPool2D, Dense, Flatten
from src.ml.optimizers import SGD
import pickle

class BrakePadCNN:
    def __init__(self):
        # Initialize layers
        self.conv1 = Conv2D(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = BatchNorm2D(32)
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)
        
        self.conv2 = Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2D(64)
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)
        
        self.conv3 = Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = BatchNorm2D(128)
        self.pool3 = MaxPool2D(kernel_size=2, stride=2)
        
        self.dropout = Dropout(0.3)
        self.flatten = Flatten()
        self.fc1 = Dense(128 * 28 * 28, 512)
        self.fc2 = Dense(512, 2)
        
    def forward(self, x, training=True):
        x = self.conv1.forward(x)
        x = self.bn1.forward(x, training)
        x = np.maximum(0, x)  # ReLU
        x = self.pool1.forward(x)
        
        x = self.conv2.forward(x)
        x = self.bn2.forward(x, training)
        x = np.maximum(0, x)  # ReLU
        x = self.pool2.forward(x)
        
        x = self.conv3.forward(x)
        x = self.bn3.forward(x, training)
        x = np.maximum(0, x)  # ReLU
        x = self.pool3.forward(x)
        
        x = self.flatten.forward(x)
        x = self.dropout.forward(x, training)
        x = self.fc1.forward(x)
        x = np.maximum(0, x)  # ReLU
        x = self.fc2.forward(x)
        
        # Softmax
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, dout):
        # Backward pass implementation
        dout = self.fc2.backward(dout)
        dout = self.fc1.backward(dout)
        dout = self.dropout.backward(dout)
        dout = self.flatten.backward(dout)
        # Continue backward through remaining layers...
        
    def save_model(self, filepath):
        """Save model weights to file"""
        model_dict = {}
        for name, layer in self.__dict__.items():
            if hasattr(layer, 'get_weights'):
                model_dict[name] = layer.get_weights()
        with open(filepath, 'wb') as f:
            pickle.dump(model_dict, f)
    
    def load_model(self, filepath):
        """Load model weights from file"""
        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)
        for name, weights in model_dict.items():
            if hasattr(self.__dict__[name], 'set_weights'):
                self.__dict__[name].set_weights(weights)
```

### Data Pipeline

```python
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(data_dir):
    """Load and preprocess image data for training"""
    images = []
    labels = []
    
    # Load normal parts
    normal_dir = os.path.join(data_dir, 'normal')
    for filename in os.listdir(normal_dir):
        img_path = os.path.join(normal_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        images.append(img)
        labels.append(0)  # Normal
    
    # Load defective parts
    defect_dir = os.path.join(data_dir, 'defective')
    for filename in os.listdir(defect_dir):
        img_path = os.path.join(defect_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        images.append(img)
        labels.append(1)  # Defective
    
    return np.array(images), np.array(labels)

def create_data_pipeline(data_dir, test_size=0.2, val_size=0.15):
    """Create train/validation/test splits"""
    X, y = load_and_preprocess_data(data_dir)
    
    # Split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    # Split train+val into train and validation
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, stratify=y_temp, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
```

---

## Next Steps & Future Improvements

### Short-Term (Q1 2026)
- [ ] Expand to detect additional defect types
- [ ] Implement federated learning for multi-factory deployment
- [ ] Add explainability with GradCAM for defect localization

### Medium-Term (Q2-Q3 2026)
- [ ] Extend to other automotive parts (engine components, transmission parts)
- [ ] Implement active learning for continuous improvement
- [ ] Add 3D imaging for geometric defect detection

### Long-Term (2027)
- [ ] Predictive maintenance integration with defect patterns
- [ ] Multi-modal inspection (visual + acoustic + thermal)
- [ ] Autonomous quality assurance with robotic arms

---

## Mathematical Foundations

### Convolution Operation
For a 2D convolution operation:
```
(I * K)(i,j) = Σ(m) Σ(n) I(i+m, j+n) * K(m,n)
```
Where I is the input image, K is the kernel/filter, and the operation slides the kernel across the image.

### Cross-Entropy Loss
For binary classification:
```
L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
```
Where y is the true label and ŷ is the predicted probability.

### Batch Normalization
```
μ_B = (1/m) * Σ x_i
σ²_B = (1/m) * Σ (x_i - μ_B)²
x̂_i = (x_i - μ_B) / √(σ²_B + ε)
y_i = γ * x̂_i + β
```
Where m is the batch size, γ and β are learnable parameters.

---

## Production Considerations

### Scalability
- **Horizontal Scaling**: Deploy additional edge units for new production lines
- **Load Distribution**: Distribute inference across multiple GPUs if needed
- **Resource Management**: Monitor GPU utilization and memory usage

### Security
- **Access Control**: Restrict physical and network access to edge devices
- **Data Protection**: Encrypt model weights and sensitive parameters
- **Audit Logging**: Log all inspection results and system events

### Reliability
- **Redundancy**: Backup cameras and inference systems
- **Failover**: Automatic switchover to manual inspection if system fails
- **Maintenance**: Scheduled downtime windows for updates

### Compliance
- **Quality Standards**: ISO 9001 compliance for quality management
- **Safety Regulations**: Automotive safety standards (ISO 26262)
- **Data Privacy**: Protect any captured employee or customer data

---

## Conclusion

This computer vision quality control system demonstrates advanced ML engineering:
- **Deep Learning**: Custom CNN optimized for defect detection
- **Edge Computing**: Real-time inference at production speeds
- **Integration**: Seamless factory floor integration
- **Impact**: $2.98M annual savings, 92.5% defect reduction

**Key takeaway**: Combining domain expertise with deep learning creates significant business value in manufacturing.

Architecture and ops blueprint: `docs/system_design_solutions/08_cv_quality_control.md`.

---

**Contact**: Implementation details in `src/vision/quality_control.py`.
Notebooks: `notebooks/case_studies/computer_vision_quality_control.ipynb`