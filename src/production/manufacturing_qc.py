"""
Manufacturing Quality Control Module
=====================================

Edge AI for real-time defect detection in manufacturing environments.

Implements computer vision-based quality inspection with:
- Real-time inference (<20ms latency target)
- PLC integration for automated reject systems
- Active learning for continuous model improvement
- Shadow deployment for safe model updates

Mathematical Foundations:
- Convolutional feature extraction: F(x) = ReLU(W * x + b)
- Defect classification: softmax(Wx + b)
- Confidence calibration: temperature scaling

Reference: Darwin Edge, Intrinsics Imaging case studies

Author: AI-Mastery-2026
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import logging
import time
import hashlib
import threading
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

class DefectType(Enum):
    """Types of manufacturing defects."""
    NONE = "none"
    SCRATCH = "scratch"
    CRACK = "crack"
    CONTAMINATION = "contamination"
    DEFORMATION = "deformation"
    MISSING_COMPONENT = "missing_component"
    DISCOLORATION = "discoloration"
    SURFACE_DEFECT = "surface_defect"


class InspectionAction(Enum):
    """Actions to take based on inspection result."""
    PASS = "pass"
    REJECT = "reject"
    MANUAL_REVIEW = "manual_review"
    REWORK = "rework"


class PLCProtocol(Enum):
    """Industrial communication protocols."""
    MODBUS_TCP = "modbus_tcp"
    OPC_UA = "opc_ua"
    PROFINET = "profinet"
    ETHERNET_IP = "ethernet_ip"


@dataclass
class InspectionResult:
    """
    Result of a quality inspection.
    
    Attributes:
        inspection_id: Unique identifier
        timestamp: When inspection occurred
        defect_type: Detected defect type
        confidence: Model confidence (0-1)
        latency_ms: Inference latency in milliseconds
        action: Recommended action
        bounding_box: [x, y, width, height] of defect region
        image_hash: Hash of inspected image for traceability
        model_version: Version of model used
    """
    inspection_id: str
    timestamp: datetime
    defect_type: DefectType
    confidence: float
    latency_ms: float
    action: InspectionAction
    bounding_box: Optional[List[float]] = None
    image_hash: str = ""
    model_version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """
    Quality inspection metrics (KPIs).
    
    Implements "Rule of Ten" cost model:
    - Detection at production: 1x cost
    - Detection at assembly: 10x cost
    - Detection at customer: 100x cost
    """
    total_inspections: int = 0
    defects_detected: int = 0
    false_positives: int = 0  # Overkill
    false_negatives: int = 0  # Escapes
    true_positives: int = 0
    true_negatives: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    uptime_seconds: float = 0.0
    
    @property
    def detection_rate(self) -> float:
        """True positive rate / Sensitivity."""
        total_positives = self.true_positives + self.false_negatives
        return self.true_positives / total_positives if total_positives > 0 else 0.0
    
    @property
    def overkill_rate(self) -> float:
        """False positive rate - good products rejected."""
        total_negatives = self.true_negatives + self.false_positives
        return self.false_positives / total_negatives if total_negatives > 0 else 0.0
    
    @property
    def escape_rate(self) -> float:
        """False negative rate - defects that escaped."""
        total_positives = self.true_positives + self.false_negatives
        return self.false_negatives / total_positives if total_positives > 0 else 0.0
    
    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)."""
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        """F1 = 2 * (precision * recall) / (precision + recall)."""
        prec = self.precision
        rec = self.detection_rate
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

class ImagePreprocessor:
    """
    Preprocessing pipeline for industrial images.
    
    Implements:
    - Normalization (0-1 scaling)
    - Contrast enhancement (histogram equalization)
    - ROI extraction (region of interest)
    - Noise reduction (Gaussian blur simulation)
    
    Mathematical basis:
    - Normalization: x' = (x - min) / (max - min)
    - Histogram equalization: CDF-based intensity mapping
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        enhance_contrast: bool = True
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.enhance_contrast = enhance_contrast
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline.
        
        Args:
            image: Raw image (H, W, C) or (H, W)
            
        Returns:
            Preprocessed image ready for inference
        """
        processed = image.copy().astype(np.float32)
        
        # Normalize to 0-1
        if self.normalize:
            processed = self._normalize(processed)
        
        # Enhance contrast
        if self.enhance_contrast:
            processed = self._enhance_contrast(processed)
        
        # Resize to target size
        processed = self._resize(processed)
        
        return processed
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Min-max normalization."""
        min_val = image.min()
        max_val = image.max()
        if max_val - min_val > 0:
            return (image - min_val) / (max_val - min_val)
        return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Simulated histogram equalization.
        
        For each pixel: new_val = CDF(old_val) * (max - min) + min
        """
        if len(image.shape) == 3:
            # Process each channel
            for c in range(image.shape[2]):
                image[:, :, c] = self._equalize_channel(image[:, :, c])
        else:
            image = self._equalize_channel(image)
        return image
    
    def _equalize_channel(self, channel: np.ndarray) -> np.ndarray:
        """Equalize single channel using CDF."""
        # Compute histogram
        flat = channel.flatten()
        hist, bins = np.histogram(flat, bins=256, range=(0, 1))
        
        # Compute CDF
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]
        
        # Map values using CDF
        bin_indices = np.clip((flat * 255).astype(int), 0, 255)
        equalized = cdf_normalized[bin_indices]
        
        return equalized.reshape(channel.shape)
    
    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Simple bilinear resize simulation."""
        h, w = image.shape[:2]
        new_h, new_w = self.target_size
        
        # Create output array
        if len(image.shape) == 3:
            output = np.zeros((new_h, new_w, image.shape[2]))
        else:
            output = np.zeros((new_h, new_w))
        
        # Scale factors
        scale_h = h / new_h
        scale_w = w / new_w
        
        # Bilinear interpolation
        for i in range(new_h):
            for j in range(new_w):
                src_i = i * scale_h
                src_j = j * scale_w
                
                i0 = int(src_i)
                j0 = int(src_j)
                i1 = min(i0 + 1, h - 1)
                j1 = min(j0 + 1, w - 1)
                
                di = src_i - i0
                dj = src_j - j0
                
                if len(image.shape) == 3:
                    for c in range(image.shape[2]):
                        val = (1 - di) * (1 - dj) * image[i0, j0, c] + \
                              di * (1 - dj) * image[i1, j0, c] + \
                              (1 - di) * dj * image[i0, j1, c] + \
                              di * dj * image[i1, j1, c]
                        output[i, j, c] = val
                else:
                    val = (1 - di) * (1 - dj) * image[i0, j0] + \
                          di * (1 - dj) * image[i1, j0] + \
                          (1 - di) * dj * image[i0, j1] + \
                          di * dj * image[i1, j1]
                    output[i, j] = val
        
        return output
    
    def extract_roi(
        self, 
        image: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Extract region of interest.
        
        Args:
            image: Source image
            bbox: (x, y, width, height)
            
        Returns:
            Cropped ROI
        """
        x, y, w, h = bbox
        return image[y:y+h, x:x+w]


# ============================================================
# DEFECT DETECTION MODEL
# ============================================================

class DefectDetector:
    """
    CNN-based defect detector optimized for edge deployment.
    
    Architecture (MobileNet-inspired):
    - Depthwise separable convolutions for efficiency
    - Batch normalization for stable training
    - Global average pooling instead of FC layers
    
    Optimization techniques:
    - INT8 quantization (4x memory reduction)
    - Layer fusion (conv + bn + relu)
    - Inference batching
    
    Target: <20ms inference on edge device
    """
    
    def __init__(
        self,
        num_classes: int = 8,
        input_size: Tuple[int, int] = (224, 224),
        quantized: bool = True,
        temperature: float = 1.0
    ):
        """
        Initialize detector.
        
        Args:
            num_classes: Number of defect classes
            input_size: Expected input dimensions
            quantized: Whether to use INT8 quantization
            temperature: Temperature for confidence calibration
        """
        self.num_classes = num_classes
        self.input_size = input_size
        self.quantized = quantized
        self.temperature = temperature
        
        # Initialize weights (simulated lightweight CNN)
        self._initialize_weights()
        
        # Defect class mapping
        self.class_names = [d.value for d in DefectType]
        
        # Inference statistics
        self.inference_count = 0
        self.total_latency_ms = 0.0
        
        logger.info(f"DefectDetector initialized: {num_classes} classes, "
                   f"input={input_size}, quantized={quantized}")
    
    def _initialize_weights(self):
        """Initialize CNN weights with He initialization."""
        np.random.seed(42)
        
        # Simulated layer weights (lightweight architecture)
        # Conv1: 3x3, 3 -> 32
        self.conv1_w = np.random.randn(32, 3, 3, 3) * np.sqrt(2.0 / (3 * 3 * 3))
        self.conv1_b = np.zeros(32)
        
        # Conv2: 3x3, 32 -> 64 (depthwise separable simulation)
        self.conv2_w = np.random.randn(64, 32, 3, 3) * np.sqrt(2.0 / (32 * 3 * 3))
        self.conv2_b = np.zeros(64)
        
        # Conv3: 3x3, 64 -> 128
        self.conv3_w = np.random.randn(128, 64, 3, 3) * np.sqrt(2.0 / (64 * 3 * 3))
        self.conv3_b = np.zeros(128)
        
        # Classifier: 128 -> num_classes
        self.fc_w = np.random.randn(128, self.num_classes) * np.sqrt(2.0 / 128)
        self.fc_b = np.zeros(self.num_classes)
        
        # Apply quantization if enabled
        if self.quantized:
            self._quantize_weights()
    
    def _quantize_weights(self):
        """
        Apply INT8 quantization simulation.
        
        Quantization formula:
        q = round(x / scale) where scale = max(|x|) / 127
        
        Dequantization:
        x' = q * scale
        """
        for attr in ['conv1_w', 'conv2_w', 'conv3_w', 'fc_w']:
            weights = getattr(self, attr)
            scale = np.max(np.abs(weights)) / 127.0
            quantized = np.round(weights / scale).astype(np.int8)
            # Store dequantized for simulation
            setattr(self, attr, quantized.astype(np.float32) * scale)
            setattr(self, f'{attr}_scale', scale)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation: max(0, x)."""
        return np.maximum(0, x)
    
    def _global_avg_pool(self, x: np.ndarray) -> np.ndarray:
        """Global average pooling over spatial dimensions."""
        # x shape: (batch, channels, H, W)
        return np.mean(x, axis=(2, 3))
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Temperature-scaled softmax.
        
        softmax(x/T) where T is temperature.
        Higher T -> softer probabilities (more uncertainty).
        """
        x_scaled = x / self.temperature
        exp_x = np.exp(x_scaled - np.max(x_scaled, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _conv2d_forward(
        self, 
        x: np.ndarray, 
        w: np.ndarray, 
        b: np.ndarray,
        stride: int = 1,
        padding: int = 1
    ) -> np.ndarray:
        """
        Simplified convolution forward pass.
        
        For production, would use im2col optimization.
        This is a simulation for demonstration.
        """
        batch_size, in_channels, h, w_in = x.shape
        out_channels, _, kh, kw = w.shape
        
        # Output dimensions
        h_out = (h + 2 * padding - kh) // stride + 1
        w_out = (w_in + 2 * padding - kw) // stride + 1
        
        # Pad input
        if padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        else:
            x_padded = x
        
        # Simulate efficient convolution
        # In reality, would use im2col + gemm
        output = np.zeros((batch_size, out_channels, h_out, w_out))
        
        # Simplified: just compute weighted sum at each position
        # This is O(n^2 * k^2 * c_in * c_out) - not optimal but clear
        for n in range(batch_size):
            for oc in range(out_channels):
                for i in range(h_out):
                    for j in range(w_out):
                        # Extract patch
                        i_start = i * stride
                        j_start = j * stride
                        patch = x_padded[n, :, i_start:i_start+kh, j_start:j_start+kw]
                        # Weighted sum
                        output[n, oc, i, j] = np.sum(patch * w[oc]) + b[oc]
        
        return output
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the network.
        
        Args:
            x: Input image(s) (N, C, H, W) format
            
        Returns:
            (class_probabilities, features)
        """
        # Ensure batch dimension
        if len(x.shape) == 3:
            x = x[np.newaxis, ...]
        
        # Conv block 1
        x = self._conv2d_forward(x, self.conv1_w, self.conv1_b)
        x = self._relu(x)
        
        # Simulated max pool (stride 2)
        x = x[:, :, ::2, ::2]
        
        # Conv block 2
        x = self._conv2d_forward(x, self.conv2_w, self.conv2_b)
        x = self._relu(x)
        x = x[:, :, ::2, ::2]
        
        # Conv block 3
        x = self._conv2d_forward(x, self.conv3_w, self.conv3_b)
        features = self._relu(x)
        
        # Global average pooling
        pooled = self._global_avg_pool(features)
        
        # Classification
        logits = pooled @ self.fc_w + self.fc_b
        probabilities = self._softmax(logits)
        
        return probabilities, pooled
    
    def predict(
        self, 
        image: np.ndarray,
        return_features: bool = False
    ) -> Tuple[DefectType, float, Optional[np.ndarray]]:
        """
        Predict defect type for an image.
        
        Args:
            image: Preprocessed image (C, H, W) format
            return_features: Whether to return feature vector
            
        Returns:
            (defect_type, confidence, features)
        """
        start_time = time.perf_counter()
        
        # Ensure correct format
        if len(image.shape) == 3 and image.shape[0] not in [1, 3]:
            # HWC -> CHW
            image = np.transpose(image, (2, 0, 1))
        
        # Forward pass
        probs, features = self.forward(image)
        
        # Get prediction
        class_idx = np.argmax(probs[0])
        confidence = float(probs[0, class_idx])
        defect_type = DefectType(self.class_names[class_idx])
        
        # Update statistics
        latency = (time.perf_counter() - start_time) * 1000
        self.inference_count += 1
        self.total_latency_ms += latency
        
        if return_features:
            return defect_type, confidence, features[0]
        return defect_type, confidence, None
    
    def get_stats(self) -> Dict[str, float]:
        """Get inference statistics."""
        avg_latency = self.total_latency_ms / max(1, self.inference_count)
        return {
            "inference_count": self.inference_count,
            "avg_latency_ms": avg_latency,
            "total_latency_ms": self.total_latency_ms
        }


# ============================================================
# PLC INTERFACE
# ============================================================

class PLCInterface:
    """
    Interface to Programmable Logic Controllers (PLCs).
    
    Supports industrial protocols:
    - Modbus TCP (registers, coils)
    - OPC UA (nodes, subscriptions)
    
    Provides deterministic timing for reject actuators.
    Target: <5ms command latency for real-time control.
    """
    
    def __init__(
        self,
        protocol: PLCProtocol = PLCProtocol.MODBUS_TCP,
        host: str = "192.168.1.100",
        port: int = 502,
        timeout_ms: float = 5.0
    ):
        """
        Initialize PLC connection.
        
        Args:
            protocol: Communication protocol
            host: PLC IP address
            port: Communication port
            timeout_ms: Command timeout
        """
        self.protocol = protocol
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        
        # Connection state
        self.connected = False
        self.last_command_time: Optional[datetime] = None
        
        # Register mapping
        self.registers: Dict[str, int] = {
            "reject_signal": 100,
            "pass_signal": 101,
            "manual_review": 102,
            "system_ready": 103,
            "error_code": 104
        }
        
        # Coil states (simulated)
        self.coil_states: Dict[int, bool] = {}
        
        # Command history for debugging
        self.command_history: List[Dict[str, Any]] = []
        
        logger.info(f"PLCInterface initialized: {protocol.value} @ {host}:{port}")
    
    def connect(self) -> bool:
        """
        Establish connection to PLC.
        
        Returns:
            True if connection successful
        """
        try:
            # Simulated connection
            logger.info(f"Connecting to PLC at {self.host}:{self.port}...")
            time.sleep(0.01)  # Simulate connection overhead
            self.connected = True
            logger.info("PLC connection established")
            return True
        except Exception as e:
            logger.error(f"PLC connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close PLC connection."""
        self.connected = False
        logger.info("PLC connection closed")
    
    def write_coil(self, address: int, value: bool) -> bool:
        """
        Write to a single coil (digital output).
        
        Modbus function code: 0x05
        
        Args:
            address: Coil address
            value: True for ON, False for OFF
            
        Returns:
            True if write successful
        """
        if not self.connected:
            logger.error("Not connected to PLC")
            return False
        
        start_time = time.perf_counter()
        
        try:
            # Simulated coil write
            self.coil_states[address] = value
            
            latency = (time.perf_counter() - start_time) * 1000
            
            self.command_history.append({
                "type": "write_coil",
                "address": address,
                "value": value,
                "latency_ms": latency,
                "timestamp": datetime.now()
            })
            
            self.last_command_time = datetime.now()
            
            if latency > self.timeout_ms:
                logger.warning(f"Coil write exceeded timeout: {latency:.2f}ms")
            
            return True
            
        except Exception as e:
            logger.error(f"Coil write failed: {e}")
            return False
    
    def write_register(self, address: int, value: int) -> bool:
        """
        Write to a holding register.
        
        Modbus function code: 0x06
        
        Args:
            address: Register address
            value: 16-bit integer value
            
        Returns:
            True if write successful
        """
        if not self.connected:
            logger.error("Not connected to PLC")
            return False
        
        start_time = time.perf_counter()
        
        try:
            # Simulated register write
            latency = (time.perf_counter() - start_time) * 1000
            
            self.command_history.append({
                "type": "write_register",
                "address": address,
                "value": value,
                "latency_ms": latency,
                "timestamp": datetime.now()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Register write failed: {e}")
            return False
    
    def trigger_reject(self) -> bool:
        """
        Trigger reject actuator (pneumatic pusher).
        
        Activates coil for configurable pulse duration.
        """
        reject_addr = self.registers["reject_signal"]
        return self.write_coil(reject_addr, True)
    
    def trigger_pass(self) -> bool:
        """Trigger pass signal."""
        pass_addr = self.registers["pass_signal"]
        return self.write_coil(pass_addr, True)
    
    def trigger_manual_review(self) -> bool:
        """Divert to manual review station."""
        review_addr = self.registers["manual_review"]
        return self.write_coil(review_addr, True)
    
    def get_command_latency_stats(self) -> Dict[str, float]:
        """Get command latency statistics."""
        if not self.command_history:
            return {"avg_ms": 0.0, "max_ms": 0.0, "count": 0}
        
        latencies = [cmd["latency_ms"] for cmd in self.command_history]
        return {
            "avg_ms": np.mean(latencies),
            "max_ms": np.max(latencies),
            "min_ms": np.min(latencies),
            "count": len(latencies)
        }


# ============================================================
# QUALITY INSPECTION PIPELINE
# ============================================================

class QualityInspectionPipeline:
    """
    End-to-end quality inspection pipeline.
    
    Components:
    1. Camera ingestion (GigE Vision / USB3)
    2. Image preprocessing
    3. AI inference (DefectDetector)
    4. Decision logic
    5. PLC actuation
    6. Cloud sync (async, metadata only)
    
    Features:
    - Shadow mode for model validation
    - Active learning for edge cases
    - Real-time metrics tracking
    """
    
    def __init__(
        self,
        detector: DefectDetector,
        plc: Optional[PLCInterface] = None,
        confidence_threshold: float = 0.85,
        review_threshold: float = 0.5,
        enable_shadow_mode: bool = False
    ):
        """
        Initialize pipeline.
        
        Args:
            detector: Trained DefectDetector model
            plc: PLC interface for actuation
            confidence_threshold: Threshold for automated decisions
            review_threshold: Below this, send to manual review
            enable_shadow_mode: Run new model in shadow without actuation
        """
        self.detector = detector
        self.plc = plc
        self.preprocessor = ImagePreprocessor()
        
        self.confidence_threshold = confidence_threshold
        self.review_threshold = review_threshold
        self.enable_shadow_mode = enable_shadow_mode
        
        # Shadow model for A/B testing
        self.shadow_detector: Optional[DefectDetector] = None
        
        # Metrics
        self.metrics = QualityMetrics()
        self.start_time = datetime.now()
        
        # Active learning queue
        self.uncertain_samples: deque = deque(maxlen=1000)
        
        # Cloud sync queue (async upload)
        self.cloud_queue: deque = deque(maxlen=10000)
        
        logger.info("QualityInspectionPipeline initialized")
    
    def set_shadow_model(self, shadow_detector: DefectDetector):
        """
        Set shadow model for validation.
        
        Shadow model runs in parallel but doesn't affect actuation.
        Used for safe deployment of new models.
        """
        self.shadow_detector = shadow_detector
        logger.info("Shadow model set for A/B validation")
    
    def inspect(
        self,
        image: np.ndarray,
        ground_truth: Optional[DefectType] = None
    ) -> InspectionResult:
        """
        Run full inspection on an image.
        
        Args:
            image: Raw camera image
            ground_truth: Optional ground truth for metrics
            
        Returns:
            InspectionResult with decision
        """
        inspection_id = hashlib.md5(
            f"{datetime.now().isoformat()}{np.random.random()}".encode()
        ).hexdigest()[:12]
        
        start_time = time.perf_counter()
        
        # 1. Preprocess
        processed = self.preprocessor.preprocess(image)
        
        # 2. Inference
        defect_type, confidence, features = self.detector.predict(
            processed, return_features=True
        )
        
        # 3. Shadow model inference (if enabled)
        if self.shadow_detector is not None:
            shadow_type, shadow_conf, _ = self.shadow_detector.predict(processed)
            # Log discrepancies for analysis
            if shadow_type != defect_type:
                logger.debug(f"Shadow model disagreement: {defect_type} vs {shadow_type}")
        
        # 4. Decision logic
        action = self._determine_action(defect_type, confidence)
        
        # 5. Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # 6. Create result
        result = InspectionResult(
            inspection_id=inspection_id,
            timestamp=datetime.now(),
            defect_type=defect_type,
            confidence=confidence,
            latency_ms=latency_ms,
            action=action,
            image_hash=hashlib.md5(image.tobytes()).hexdigest(),
            model_version=self.detector.__class__.__name__
        )
        
        # 7. Actuate (if PLC connected and not in shadow mode)
        if self.plc is not None and self.plc.connected and not self.enable_shadow_mode:
            self._actuate(action)
        
        # 8. Update metrics
        self._update_metrics(result, ground_truth)
        
        # 9. Active learning: save uncertain samples
        if confidence < self.confidence_threshold and features is not None:
            self.uncertain_samples.append({
                "image": image.copy(),
                "features": features,
                "prediction": defect_type,
                "confidence": confidence,
                "timestamp": datetime.now()
            })
        
        # 10. Queue for cloud sync (metadata only)
        self.cloud_queue.append({
            "inspection_id": result.inspection_id,
            "timestamp": result.timestamp.isoformat(),
            "defect_type": result.defect_type.value,
            "confidence": result.confidence,
            "latency_ms": result.latency_ms,
            "action": result.action.value
        })
        
        return result
    
    def _determine_action(
        self, 
        defect_type: DefectType, 
        confidence: float
    ) -> InspectionAction:
        """
        Determine action based on prediction and confidence.
        
        Decision tree:
        - confidence >= threshold AND defect: REJECT
        - confidence >= threshold AND no defect: PASS
        - confidence < review_threshold: MANUAL_REVIEW
        - otherwise: use defect type to decide
        """
        if confidence < self.review_threshold:
            return InspectionAction.MANUAL_REVIEW
        
        if defect_type == DefectType.NONE:
            return InspectionAction.PASS
        
        if confidence >= self.confidence_threshold:
            return InspectionAction.REJECT
        
        # Medium confidence defect -> manual review
        return InspectionAction.MANUAL_REVIEW
    
    def _actuate(self, action: InspectionAction):
        """Send actuation command to PLC."""
        if action == InspectionAction.REJECT:
            self.plc.trigger_reject()
        elif action == InspectionAction.PASS:
            self.plc.trigger_pass()
        elif action == InspectionAction.MANUAL_REVIEW:
            self.plc.trigger_manual_review()
    
    def _update_metrics(
        self, 
        result: InspectionResult, 
        ground_truth: Optional[DefectType]
    ):
        """Update quality metrics."""
        self.metrics.total_inspections += 1
        
        # Update latency stats
        n = self.metrics.total_inspections
        self.metrics.avg_latency_ms = (
            (self.metrics.avg_latency_ms * (n - 1) + result.latency_ms) / n
        )
        self.metrics.max_latency_ms = max(
            self.metrics.max_latency_ms, result.latency_ms
        )
        
        # Update uptime
        self.metrics.uptime_seconds = (
            datetime.now() - self.start_time
        ).total_seconds()
        
        # If ground truth provided, update accuracy metrics
        if ground_truth is not None:
            predicted_defect = result.defect_type != DefectType.NONE
            actual_defect = ground_truth != DefectType.NONE
            
            if predicted_defect and actual_defect:
                self.metrics.true_positives += 1
                self.metrics.defects_detected += 1
            elif predicted_defect and not actual_defect:
                self.metrics.false_positives += 1
            elif not predicted_defect and actual_defect:
                self.metrics.false_negatives += 1
            else:
                self.metrics.true_negatives += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current quality metrics."""
        return {
            "total_inspections": self.metrics.total_inspections,
            "defects_detected": self.metrics.defects_detected,
            "detection_rate": self.metrics.detection_rate,
            "overkill_rate": self.metrics.overkill_rate,
            "escape_rate": self.metrics.escape_rate,
            "precision": self.metrics.precision,
            "f1_score": self.metrics.f1_score,
            "avg_latency_ms": self.metrics.avg_latency_ms,
            "max_latency_ms": self.metrics.max_latency_ms,
            "uptime_seconds": self.metrics.uptime_seconds,
            "uncertain_samples_queued": len(self.uncertain_samples),
            "cloud_queue_size": len(self.cloud_queue)
        }
    
    def get_active_learning_samples(self, n: int = 100) -> List[Dict]:
        """
        Get uncertain samples for labeling.
        
        These are edge cases that should be manually reviewed and
        added to training data for model improvement.
        """
        samples = []
        while len(samples) < n and self.uncertain_samples:
            samples.append(self.uncertain_samples.popleft())
        return samples
    
    def flush_cloud_queue(self) -> List[Dict]:
        """
        Get all pending cloud sync items.
        
        In production, this would be sent to cloud asynchronously.
        Only metadata is sent - images stay on-premises.
        """
        items = []
        while self.cloud_queue:
            items.append(self.cloud_queue.popleft())
        return items


# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def create_inspection_system(
    connect_plc: bool = True,
    plc_host: str = "192.168.1.100",
    quantized: bool = True
) -> Tuple[QualityInspectionPipeline, DefectDetector, Optional[PLCInterface]]:
    """
    Create a complete inspection system.
    
    Args:
        connect_plc: Whether to connect to PLC
        plc_host: PLC IP address
        quantized: Use quantized model
        
    Returns:
        (pipeline, detector, plc_interface)
    """
    # Create detector
    detector = DefectDetector(
        num_classes=len(DefectType),
        input_size=(224, 224),
        quantized=quantized
    )
    
    # Create PLC interface
    plc = None
    if connect_plc:
        plc = PLCInterface(host=plc_host)
        plc.connect()
    
    # Create pipeline
    pipeline = QualityInspectionPipeline(
        detector=detector,
        plc=plc,
        confidence_threshold=0.85,
        review_threshold=0.5
    )
    
    return pipeline, detector, plc


# ============================================================
# DEMO / TESTING
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Manufacturing Quality Control - Edge AI Demo")
    print("=" * 60)
    
    # Create system
    pipeline, detector, plc = create_inspection_system(connect_plc=True)
    
    # Simulate camera images
    print("\nSimulating 100 inspections...")
    
    for i in range(100):
        # Generate synthetic image (random noise + patterns for defects)
        image = np.random.rand(224, 224, 3).astype(np.float32)
        
        # Randomly add "defect" patterns
        if np.random.random() < 0.1:  # 10% defect rate
            # Add scratch pattern
            x = np.random.randint(50, 150)
            image[x:x+5, 50:200, :] = 0.0
            ground_truth = DefectType.SCRATCH
        else:
            ground_truth = DefectType.NONE
        
        # Run inspection
        result = pipeline.inspect(image, ground_truth=ground_truth)
        
        if i % 20 == 0:
            print(f"  Inspection {i}: {result.defect_type.value} "
                  f"({result.confidence:.2%}) -> {result.action.value} "
                  f"[{result.latency_ms:.1f}ms]")
    
    # Print metrics
    print("\n" + "=" * 60)
    print("Quality Metrics")
    print("=" * 60)
    metrics = pipeline.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Get detector stats
    print("\nDetector Statistics:")
    detector_stats = detector.get_stats()
    for key, value in detector_stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Print PLC stats if connected
    if plc:
        print("\nPLC Command Statistics:")
        plc_stats = plc.get_command_latency_stats()
        for key, value in plc_stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\n✅ Manufacturing QC module demo complete!")
