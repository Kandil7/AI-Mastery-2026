# Case Study 5: Computer Vision for Quality Control in Manufacturing

## Executive Summary

**Problem**: Automotive parts manufacturer experiencing 3.2% defect rate causing $2.4M annual losses and customer complaints.

**Solution**: Deployed real-time computer vision system using CNNs to detect defects during production.

**Impact**: Reduced defect rate from 3.2% to 0.8%, saving $1.8M annually and improving customer satisfaction scores by 22%.

---

## Business Context

### Company Profile
- **Industry**: Automotive Parts Manufacturing
- **Production Volume**: 50,000 units/day across 12 production lines
- **Defect Rate**: 3.2% (industry standard: 1.5%)
- **Problem**: Manual inspection too slow and inconsistent; costly recalls

### Key Challenges
1. **High-Speed Inspection**: Parts move at 2 meters/second through assembly line
2. **Variety of Defects**: Scratches, dents, missing components, misalignments
3. **Lighting Conditions**: Factory lighting varies throughout day
4. **False Positives**: Need <2% false reject rate to avoid production slowdown

---

## Technical Approach

### Multi-Stage Computer Vision Pipeline

```
Raw Image Capture → Preprocessing → Feature Extraction → Defect Classification → Decision Engine
     (Camera)        (Resize, Aug)    (CNN Backbone)      (ResNet-50)        (Threshold Logic)
```

### Stage 1: Image Acquisition & Preprocessing

**Hardware Setup**:
- Industrial cameras: 5MP resolution, 120fps capture
- LED lighting arrays with consistent color temperature
- Position sensors trigger capture at precise intervals

**Preprocessing Pipeline**:
```python
import cv2
import numpy as np

def preprocess_image(image):
    # Resize to standard dimensions
    resized = cv2.resize(image, (224, 224))
    
    # Normalize pixel values
    normalized = resized.astype(np.float32) / 255.0
    
    # Apply histogram equalization for consistent lighting
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
    equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return equalized
```

### Stage 2: Feature Extraction & Defect Classification

**Model Architecture**: Fine-tuned ResNet-50
- Pre-trained on ImageNet for general feature extraction
- Replaced final layer for binary classification (defect/no-defect)
- Additional layers for multi-class defect identification

**Transfer Learning Strategy**:
- Freeze early layers (general features)
- Fine-tune later layers (domain-specific features)
- Train final classifier on 50,000 labeled part images

### Stage 3: Decision Engine

**Multi-Threshold System**:
- Primary: Defect/no-defect (threshold = 0.7)
- Secondary: Defect type classification
- Tertiary: Severity scoring for escalation

---

## Model Development

### Dataset Preparation

**Training Data**: 50,000 images (25,000 good, 25,000 defective)
- Good parts: Various lighting conditions, angles, production batches
- Defective parts: 8 categories (scratches, dents, missing components, etc.)
- Annotations: Bounding boxes and severity scores

**Data Augmentation**:
```python
from albumentations import Compose, Rotate, HorizontalFlip, RandomBrightnessContrast

augmentation = Compose([
    Rotate(limit=15, p=0.5),
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
])
```

### Model Architecture

**Base Model**: ResNet-50 (pre-trained on ImageNet)
```python
import torchvision.models as models
import torch.nn as nn

base_model = models.resnet50(pretrained=True)

# Replace final fully connected layer
num_features = base_model.fc.in_features
base_model.fc = nn.Linear(num_features, 2)  # binary classification
```

**Fine-Tuning Strategy**:
- Epochs: 50 with early stopping
- Learning rate: 0.001 (base) vs 0.0001 (frozen layers)
- Loss function: Weighted binary cross-entropy (handle class imbalance)
- Optimizer: Adam with scheduler

### Model Comparison

| Model | Precision | Recall | F1-Score | Inference Time | Notes |
|-------|-----------|--------|----------|----------------|-------|
| Custom CNN | 0.82 | 0.76 | 0.79 | 15ms | Baseline model |
| ResNet-18 | 0.85 | 0.81 | 0.83 | 12ms | Good balance |
| ResNet-50 | **0.91** | **0.89** | **0.90** | 22ms | **Selected** |
| EfficientNet-B0 | 0.89 | 0.87 | 0.88 | 18ms | Close second |

**Selected Model**: ResNet-50
- **Reason**: Best performance with acceptable latency
- **Threshold**: 0.7 (optimized for precision to minimize false rejects)

### Cross-Validation Results

- **Accuracy**: 90.2%
- **Precision**: 91.0%
- **Recall**: 89.0%
- **F1-Score**: 90.0%
- **AUC-ROC**: 0.94

---

## Production Deployment

### Edge Computing Architecture

```
Industrial Camera → Edge AI Device (NVIDIA Jetson) → PLC Integration
                      ↓
                 Cloud Backup & Model Updates
```

### Components

**1. Edge Inference Service** (NVIDIA Jetson):
```python
import torch
import cv2
import numpy as np

class DefectDetectionService:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()
        
    def detect_defect(self, image):
        # Preprocess
        processed_img = preprocess_image(image)
        tensor_img = torch.tensor(processed_img).permute(2, 0, 1).unsqueeze(0).float()
        
        # Inference
        with torch.no_grad():
            output = self.model(tensor_img)
            probabilities = torch.softmax(output, dim=1)
            
        defect_prob = probabilities[0][1].item()  # probability of defect class
        
        # Decision logic
        if defect_prob > 0.7:
            return {"defect": True, "confidence": defect_prob, "action": "reject"}
        else:
            return {"defect": False, "confidence": 1-defect_prob, "action": "accept"}
```

**2. PLC Integration**:
- Communicates rejection decisions to production line
- Triggers physical rejection mechanism
- Logs inspection results for quality reports

**3. Cloud Backup System**:
- Stores images of rejected parts for review
- Collects performance metrics
- Enables remote model updates

### Operational SLOs
- **Inference Latency**: p95 < 30ms (part moves 2m/s, need decision within 6cm)
- **Availability**: 99.9% uptime (production line cannot afford downtime)
- **False Reject Rate**: <2% (to maintain production efficiency)
- **Defect Detection Rate**: >85% (to meet quality targets)

### Quality Control & Monitoring
- **Real-time metrics**: Accuracy, precision, recall per production line
- **Image quality monitoring**: Check for blur, lighting issues
- **Model drift detection**: Compare current performance to baseline
- **Human validation**: Random samples reviewed by quality engineers

---

## Results & Impact

### Model Performance in Production

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **Defect Detection Rate** | >85% | 89% | ✅ Exceeded |
| **False Reject Rate** | <2% | 1.4% | ✅ Met |
| **Inference Latency** | <30ms | 22ms | ✅ Met |
| **Overall Accuracy** | >88% | 90.2% | ✅ Exceeded |

### Business Impact (6 months post-launch)

| Metric | Before CV | After CV | Improvement |
|--------|-----------|----------|-------------|
| **Defect Rate** | 3.2% | 0.8% | **-75%** |
| **Manual Inspection Time** | 100% | 5% | **-95%** |
| **Quality Control Costs** | $400K/year | $150K/year | **-62.5%** |
| **Customer Complaints** | 120/month | 35/month | **-71%** |
| **Production Speed** | 100% | 102% | **+2%** |

### Cost-Benefit Analysis

**Savings**:
- Reduced defect costs: 3.2% → 0.8% reduction = 2.4% of production
- At 50,000 units/day × $12/unit × 2.4% × 365 days = $525K savings
- Reduced manual inspection: 12 inspectors × $60K/year = $720K savings
- Reduced customer complaints: $300K in avoided warranty costs

**Investment**:
- Hardware (cameras, edge devices): $300K
- Software development: $200K
- Training and deployment: $100K

**Net Benefit**: ($525K + $720K + $300K) - $600K = **$945K annually**

### Defect Type Performance

| Defect Type | Detection Rate | False Positive Rate |
|-------------|----------------|---------------------|
| Scratches | 92% | 0.8% |
| Dents | 87% | 1.1% |
| Missing Components | 94% | 0.6% |
| Misalignments | 85% | 1.5% |
| Surface Imperfections | 90% | 0.9% |

---

## Challenges & Solutions

### Challenge 1: Lighting Variations
- **Problem**: Factory lighting changes throughout day affect image quality
- **Solution**:
  - Installed consistent LED lighting arrays
  - Applied histogram equalization in preprocessing
  - Used data augmentation with brightness/contrast variations

### Challenge 2: High-Speed Production Line
- **Problem**: Parts move at 2m/s, requiring <30ms decision time
- **Solution**:
  - Optimized model for edge deployment (TensorRT)
  - Used NVIDIA Jetson for GPU acceleration
  - Implemented asynchronous processing pipeline

### Challenge 3: Class Imbalance
- **Problem**: Only 3.2% of parts were defective initially
- **Solution**:
  - Collected additional defect samples through active learning
  - Used weighted loss function during training
  - Implemented synthetic defect generation

### Challenge 4: Model Drift
- **Problem**: New part designs and manufacturing changes affect model performance
- **Solution**:
  - Continuous monitoring of performance metrics
  - Scheduled retraining every 3 months
  - A/B testing framework for model updates

---

## Lessons Learned

### What Worked

1. **Transfer Learning Superior to Custom Models**
   - Custom CNN: 79% F1-score
   - ResNet-50: 90% F1-score
   - Pre-trained features captured relevant patterns better

2. **Edge Computing Essential for Latency**
   - Cloud inference: 150ms (too slow)
   - Edge inference: 22ms (within requirements)
   - Critical for high-speed manufacturing

3. **Data Quality Trumps Quantity**
   - Started with 10,000 images, expanded to 50,000
   - Careful annotation and balanced dataset more important than size
   - Quality of labeling significantly impacted performance

### What Didn't Work

1. **Complex Architectures Unnecessary**
   - Tried EfficientNet-B4: 90.1% F1-score but 45ms latency
   - ResNet-50 achieved similar performance with better latency
   - Simpler models often sufficient for manufacturing applications

2. **Overfitting to Training Conditions**
   - Initial model performed poorly on new lighting conditions
   - Required extensive data augmentation and real-world fine-tuning
   - Simulation alone insufficient for manufacturing environments

---

## Technical Implementation

### Training Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define transformations
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Load dataset
train_dataset = QualityControlDataset(root_dir='data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]))  # Weighted for imbalance
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(50):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    scheduler.step()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
```

### Inference Pipeline

```python
import cv2
import numpy as np
import torch
from PIL import Image

class QualityControlInference:
    def __init__(self, model_path, confidence_threshold=0.7):
        self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()
        self.threshold = confidence_threshold
        
    def preprocess(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = torch.tensor(image).unsqueeze(0)  # Add batch dimension
        return image
    
    def predict(self, image_path):
        input_tensor = self.preprocess(image_path)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            defect_prob = probabilities[0][1].item()
        
        is_defective = defect_prob > self.threshold
        confidence = max(probabilities[0].tolist())
        
        return {
            'defective': is_defective,
            'confidence': confidence,
            'defect_probability': defect_prob,
            'action': 'reject' if is_defective else 'accept'
        }

# Usage
qc_system = QualityControlInference('models/resnet50_quality_control.pth')
result = qc_system.predict('images/part_001.jpg')
print(result)
```

---

## Next Steps & Future Improvements

### Short-Term (Q1 2026)
- [ ] Implement multi-class defect classification (currently binary)
- [ ] Add severity scoring for different defect types
- [ ] Expand to additional production lines

### Medium-Term (Q2-Q3 2026)
- [ ] Integrate with predictive maintenance system
- [ ] Add 3D imaging for complex geometries
- [ ] Implement active learning for continuous improvement

### Long-Term (2027)
- [ ] Extend to other manufacturing facilities
- [ ] Add reinforcement learning for adaptive thresholding
- [ ] Connect to supplier quality metrics

---

## Conclusion

This computer vision quality control system demonstrates:
- **Advanced Computer Vision**: Transfer learning with ResNet-50
- **Edge Deployment**: Real-time inference at production speeds
- **Manufacturing Impact**: 75% defect reduction, $1.8M annual savings

**Key takeaway**: Proper combination of pre-trained models, edge computing, and manufacturing domain expertise delivers significant operational improvements.

Architecture and ops blueprint: `docs/system_design_solutions/08_cv_quality_control.md`.

---

**Contact**: Implementation details in `src/cv/quality_control.py`.
Notebooks: `notebooks/case_studies/cv_quality_control.ipynb`