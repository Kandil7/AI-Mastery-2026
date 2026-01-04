"""
Edge AI Module
===============

Industrial edge deployment for ML models.

Inspired by Siemens Industrial Edge and AWS SageMaker Edge:
- Cloud-to-Edge Lifecycle: Train in cloud, deploy to edge
- Model Compilation: Quantization, pruning for edge hardware
- OTA Updates: Over-the-air model deployment
- Local Inference: Low-latency on-device prediction
- Data Residency: Process sensitive data locally

Key Patterns:
1. Central Training → Edge Deployment
2. Model Optimization for resource-constrained devices
3. Fleet Management for thousands of edge devices
4. Bandwidth Optimization (send metadata, not raw data)

Reference: Siemens Industrial Edge, AWS SageMaker Neo, NVIDIA Jetson
"""

import numpy as np
import logging
import json
import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

class DeviceStatus(Enum):
    """Edge device status."""
    ONLINE = "online"
    OFFLINE = "offline"
    UPDATING = "updating"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ModelStatus(Enum):
    """Deployed model status."""
    DEPLOYED = "deployed"
    PENDING = "pending"
    FAILED = "failed"
    ROLLBACK = "rollback"


class QuantizationType(Enum):
    """Model quantization types."""
    NONE = "none"           # FP32 (full precision)
    FP16 = "fp16"           # Half precision
    INT8 = "int8"           # 8-bit integer
    INT4 = "int4"           # 4-bit integer (aggressive)
    DYNAMIC = "dynamic"     # Dynamic quantization


class OptimizationType(Enum):
    """Model optimization types."""
    NONE = "none"
    PRUNING = "pruning"             # Remove redundant weights
    KNOWLEDGE_DISTILLATION = "kd"   # Train smaller model
    TENSORRT = "tensorrt"           # NVIDIA optimization
    OPENVINO = "openvino"           # Intel optimization
    ONNX = "onnx"                   # Cross-platform


@dataclass
class EdgeDevice:
    """
    Represents an edge device (Siemens IPC pattern).
    
    Attributes:
        device_id: Unique identifier
        name: Human-readable name
        location: Physical location
        capabilities: Hardware capabilities
        status: Current device status
        deployed_models: Currently deployed models
        last_heartbeat: Last communication time
    """
    device_id: str
    name: str
    location: str = ""
    capabilities: Dict[str, Any] = field(default_factory=dict)
    status: DeviceStatus = DeviceStatus.OFFLINE
    deployed_models: Dict[str, str] = field(default_factory=dict)  # model_name -> version
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.capabilities:
            self.capabilities = {
                "cpu_cores": 4,
                "memory_gb": 4,
                "gpu": False,
                "accelerator": None,  # "nvidia_jetson", "intel_ncs", etc.
                "storage_gb": 32
            }
    
    def is_online(self) -> bool:
        """Check if device is online (heartbeat within 5 minutes)."""
        if self.status != DeviceStatus.ONLINE:
            return False
        delta = datetime.now() - self.last_heartbeat
        return delta.total_seconds() < 300  # 5 minutes
    
    def can_deploy(self, model_requirements: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if device can deploy a model.
        
        Args:
            model_requirements: Required resources
            
        Returns:
            (can_deploy, reason)
        """
        required_memory = model_requirements.get("memory_gb", 1)
        requires_gpu = model_requirements.get("requires_gpu", False)
        
        if required_memory > self.capabilities.get("memory_gb", 0):
            return False, "Insufficient memory"
        if requires_gpu and not self.capabilities.get("gpu", False):
            return False, "GPU required but not available"
        if self.status != DeviceStatus.ONLINE:
            return False, f"Device status: {self.status.value}"
            
        return True, "OK"


@dataclass
class CompiledModel:
    """
    A compiled/optimized model for edge deployment.
    
    Attributes:
        model_id: Unique identifier
        name: Model name
        version: Model version
        original_size_mb: Size before optimization
        compiled_size_mb: Size after optimization
        quantization: Quantization applied
        optimization: Optimization applied
        target_device: Target hardware
        accuracy_delta: Accuracy change (negative = degradation)
        latency_improvement: Speedup factor
    """
    model_id: str
    name: str
    version: str
    original_size_mb: float
    compiled_size_mb: float
    quantization: QuantizationType = QuantizationType.NONE
    optimization: OptimizationType = OptimizationType.NONE
    target_device: str = "generic"
    accuracy_delta: float = 0.0  # e.g., -0.02 means 2% accuracy loss
    latency_improvement: float = 1.0  # e.g., 2.0 means 2x faster
    weights: Optional[np.ndarray] = None  # Simulated weights
    checksum: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.checksum and self.weights is not None:
            self.checksum = hashlib.md5(self.weights.tobytes()).hexdigest()
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio."""
        if self.original_size_mb == 0:
            return 1.0
        return self.original_size_mb / self.compiled_size_mb


@dataclass 
class InferenceResult:
    """Result from edge inference."""
    device_id: str
    model_name: str
    model_version: str
    prediction: Any
    confidence: float
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentRecord:
    """Record of a model deployment."""
    deployment_id: str
    device_id: str
    model_id: str
    model_version: str
    status: ModelStatus
    deployed_at: datetime
    updated_at: datetime = field(default_factory=datetime.now)
    error_message: str = ""
    rollback_version: str = ""


# ============================================================
# MODEL COMPILER
# ============================================================

class ModelCompiler:
    """
    Compile models for edge deployment (SageMaker Neo pattern).
    
    Applies quantization and optimization to reduce model size
    and improve inference speed on resource-constrained devices.
    """
    
    def __init__(self):
        self.compiled_models: Dict[str, CompiledModel] = {}
        self.compilation_stats: List[Dict[str, Any]] = []
    
    def compile(
        self,
        model_name: str,
        model_weights: np.ndarray,
        version: str = "1.0.0",
        quantization: QuantizationType = QuantizationType.INT8,
        optimization: OptimizationType = OptimizationType.NONE,
        target_device: str = "generic"
    ) -> CompiledModel:
        """
        Compile a model for edge deployment.
        
        Args:
            model_name: Name of the model
            model_weights: Model weights (simulated)
            version: Model version
            quantization: Quantization type
            optimization: Optimization type
            target_device: Target hardware
            
        Returns:
            CompiledModel ready for deployment
        """
        start_time = time.time()
        
        # Calculate original size
        original_size_mb = model_weights.nbytes / (1024 * 1024)
        
        # Apply quantization (simulated)
        quantized_weights, accuracy_delta = self._apply_quantization(
            model_weights, quantization
        )
        
        # Apply optimization (simulated)
        optimized_weights, latency_improvement = self._apply_optimization(
            quantized_weights, optimization
        )
        
        # Calculate compiled size
        compiled_size_mb = optimized_weights.nbytes / (1024 * 1024)
        
        # Create compiled model
        model_id = f"{model_name}_{version}_{quantization.value}_{target_device}"
        compiled = CompiledModel(
            model_id=model_id,
            name=model_name,
            version=version,
            original_size_mb=original_size_mb,
            compiled_size_mb=compiled_size_mb,
            quantization=quantization,
            optimization=optimization,
            target_device=target_device,
            accuracy_delta=accuracy_delta,
            latency_improvement=latency_improvement,
            weights=optimized_weights
        )
        
        self.compiled_models[model_id] = compiled
        
        # Record stats
        compile_time = time.time() - start_time
        self.compilation_stats.append({
            "model_id": model_id,
            "compile_time_s": compile_time,
            "compression_ratio": compiled.get_compression_ratio(),
            "accuracy_delta": accuracy_delta,
            "latency_improvement": latency_improvement
        })
        
        logger.info(
            f"Compiled {model_name}: {original_size_mb:.2f}MB → "
            f"{compiled_size_mb:.2f}MB ({compiled.get_compression_ratio():.1f}x)"
        )
        
        return compiled
    
    def _apply_quantization(
        self, 
        weights: np.ndarray, 
        quantization: QuantizationType
    ) -> Tuple[np.ndarray, float]:
        """
        Apply quantization to model weights.
        
        Returns:
            (quantized_weights, accuracy_delta)
        """
        if quantization == QuantizationType.NONE:
            return weights, 0.0
        
        # Simulate quantization effects
        quantization_configs = {
            QuantizationType.FP16: (np.float16, -0.001),
            QuantizationType.INT8: (np.int8, -0.02),
            QuantizationType.INT4: (np.int8, -0.05),  # Simulated
            QuantizationType.DYNAMIC: (np.float16, -0.01)
        }
        
        dtype, accuracy_delta = quantization_configs.get(
            quantization, (np.float32, 0.0)
        )
        
        # Scale weights to fit in target dtype range
        if dtype == np.int8:
            scale = 127.0 / (np.abs(weights).max() + 1e-8)
            quantized = (weights * scale).astype(dtype)
        else:
            quantized = weights.astype(dtype)
        
        return quantized, accuracy_delta
    
    def _apply_optimization(
        self,
        weights: np.ndarray,
        optimization: OptimizationType
    ) -> Tuple[np.ndarray, float]:
        """
        Apply optimization to model weights.
        
        Returns:
            (optimized_weights, latency_improvement)
        """
        if optimization == OptimizationType.NONE:
            return weights, 1.0
        
        optimization_effects = {
            OptimizationType.PRUNING: (0.7, 1.5),      # 30% smaller, 1.5x faster
            OptimizationType.TENSORRT: (1.0, 3.0),    # Same size, 3x faster
            OptimizationType.OPENVINO: (1.0, 2.5),    # Same size, 2.5x faster
            OptimizationType.ONNX: (0.95, 1.2),       # Slightly smaller, 1.2x faster
            OptimizationType.KNOWLEDGE_DISTILLATION: (0.5, 2.0)  # 50% smaller, 2x faster
        }
        
        size_factor, latency_improvement = optimization_effects.get(
            optimization, (1.0, 1.0)
        )
        
        # Simulate pruning by zeroing out small weights
        if optimization == OptimizationType.PRUNING:
            threshold = np.percentile(np.abs(weights), 30)
            optimized = np.where(np.abs(weights) < threshold, 0, weights)
            # Convert to sparse representation (simulated)
            optimized = optimized.astype(weights.dtype)
        else:
            optimized = weights
        
        return optimized, latency_improvement
    
    def get_compilation_summary(self) -> Dict[str, Any]:
        """Get summary of all compilations."""
        if not self.compilation_stats:
            return {"total_compilations": 0}
        
        return {
            "total_compilations": len(self.compilation_stats),
            "avg_compression_ratio": np.mean([s["compression_ratio"] for s in self.compilation_stats]),
            "avg_accuracy_delta": np.mean([s["accuracy_delta"] for s in self.compilation_stats]),
            "avg_latency_improvement": np.mean([s["latency_improvement"] for s in self.compilation_stats]),
            "total_compile_time_s": sum([s["compile_time_s"] for s in self.compilation_stats])
        }


# ============================================================
# OTA UPDATE MANAGER
# ============================================================

class OTAUpdateManager:
    """
    Over-the-air model update manager.
    
    Handles deploying models to edge device fleets with:
    - Staged rollouts
    - Automatic rollback on failure
    - Version management
    - Health monitoring
    """
    
    def __init__(self, rollout_percentage: float = 10.0):
        """
        Initialize OTA manager.
        
        Args:
            rollout_percentage: Initial deployment percentage for canary
        """
        self.rollout_percentage = rollout_percentage
        self.deployments: Dict[str, DeploymentRecord] = {}
        self.model_registry: Dict[str, CompiledModel] = {}
        self.deployment_history: List[Dict[str, Any]] = []
    
    def register_model(self, model: CompiledModel) -> None:
        """Register a compiled model for deployment."""
        self.model_registry[model.model_id] = model
        logger.info(f"Registered model: {model.model_id}")
    
    def deploy_to_device(
        self,
        device: EdgeDevice,
        model: CompiledModel,
        force: bool = False
    ) -> DeploymentRecord:
        """
        Deploy a model to a single device.
        
        Args:
            device: Target device
            model: Model to deploy
            force: Force deployment even if incompatible
            
        Returns:
            DeploymentRecord
        """
        # Check compatibility
        requirements = {
            "memory_gb": model.compiled_size_mb / 1024 + 0.5,  # Model + overhead
            "requires_gpu": "tensorrt" in model.optimization.value.lower()
        }
        
        can_deploy, reason = device.can_deploy(requirements)
        
        if not can_deploy and not force:
            record = DeploymentRecord(
                deployment_id=f"deploy_{device.device_id}_{model.model_id}_{int(time.time())}",
                device_id=device.device_id,
                model_id=model.model_id,
                model_version=model.version,
                status=ModelStatus.FAILED,
                deployed_at=datetime.now(),
                error_message=reason
            )
            self.deployments[record.deployment_id] = record
            return record
        
        # Store rollback version
        rollback_version = device.deployed_models.get(model.name, "")
        
        # Simulate deployment
        device.status = DeviceStatus.UPDATING
        time.sleep(0.01)  # Simulate network delay
        
        # Update device
        device.deployed_models[model.name] = model.version
        device.status = DeviceStatus.ONLINE
        device.last_heartbeat = datetime.now()
        
        record = DeploymentRecord(
            deployment_id=f"deploy_{device.device_id}_{model.model_id}_{int(time.time())}",
            device_id=device.device_id,
            model_id=model.model_id,
            model_version=model.version,
            status=ModelStatus.DEPLOYED,
            deployed_at=datetime.now(),
            rollback_version=rollback_version
        )
        
        self.deployments[record.deployment_id] = record
        self.deployment_history.append({
            "deployment_id": record.deployment_id,
            "device_id": device.device_id,
            "model_id": model.model_id,
            "status": record.status.value,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Deployed {model.name}:{model.version} to {device.name}")
        
        return record
    
    def deploy_to_fleet(
        self,
        devices: List[EdgeDevice],
        model: CompiledModel,
        staged: bool = True
    ) -> Dict[str, Any]:
        """
        Deploy model to a fleet of devices.
        
        Args:
            devices: List of target devices
            model: Model to deploy
            staged: Use staged rollout (canary)
            
        Returns:
            Deployment summary
        """
        results = {
            "total_devices": len(devices),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "deployment_ids": []
        }
        
        if staged:
            # Canary deployment: deploy to subset first
            canary_count = max(1, int(len(devices) * self.rollout_percentage / 100))
            canary_devices = devices[:canary_count]
            remaining_devices = devices[canary_count:]
            
            # Deploy to canary
            logger.info(f"Canary deployment to {canary_count} devices...")
            canary_success = 0
            for device in canary_devices:
                record = self.deploy_to_device(device, model)
                results["deployment_ids"].append(record.deployment_id)
                if record.status == ModelStatus.DEPLOYED:
                    canary_success += 1
                    results["successful"] += 1
                else:
                    results["failed"] += 1
            
            # Check canary success rate
            canary_success_rate = canary_success / canary_count if canary_count > 0 else 0
            if canary_success_rate < 0.8:
                logger.warning(
                    f"Canary failed ({canary_success_rate:.0%} success). "
                    f"Skipping remaining {len(remaining_devices)} devices."
                )
                results["skipped"] = len(remaining_devices)
                results["canary_failed"] = True
                return results
            
            devices = remaining_devices
        
        # Full deployment
        for device in devices:
            record = self.deploy_to_device(device, model)
            results["deployment_ids"].append(record.deployment_id)
            if record.status == ModelStatus.DEPLOYED:
                results["successful"] += 1
            else:
                results["failed"] += 1
        
        return results
    
    def rollback_device(
        self,
        device: EdgeDevice,
        model_name: str
    ) -> Optional[DeploymentRecord]:
        """
        Rollback a device to the previous model version.
        
        Args:
            device: Device to rollback
            model_name: Model to rollback
            
        Returns:
            New deployment record or None if no rollback available
        """
        # Find last successful deployment for this device/model
        for record in reversed(self.deployment_history):
            if (record.get("device_id") == device.device_id and 
                model_name in record.get("model_id", "")):
                
                deployment = self.deployments.get(record["deployment_id"])
                if deployment and deployment.rollback_version:
                    # Find the rollback model
                    for model in self.model_registry.values():
                        if model.name == model_name and model.version == deployment.rollback_version:
                            return self.deploy_to_device(device, model, force=True)
        
        return None
    
    def get_fleet_status(
        self,
        devices: List[EdgeDevice],
        model_name: str
    ) -> Dict[str, Any]:
        """Get deployment status across a fleet."""
        status = {
            "total_devices": len(devices),
            "online": 0,
            "offline": 0,
            "deployed": 0,
            "not_deployed": 0,
            "version_distribution": {}
        }
        
        for device in devices:
            if device.is_online():
                status["online"] += 1
            else:
                status["offline"] += 1
            
            if model_name in device.deployed_models:
                status["deployed"] += 1
                version = device.deployed_models[model_name]
                status["version_distribution"][version] = \
                    status["version_distribution"].get(version, 0) + 1
            else:
                status["not_deployed"] += 1
        
        return status


# ============================================================
# EDGE INFERENCE ENGINE
# ============================================================

class EdgeInferenceEngine:
    """
    Local inference on edge devices.
    
    Features:
    - Low-latency on-device prediction
    - Batch inference optimization
    - Data residency compliance (data never leaves device)
    - Telemetry for monitoring (metadata only)
    """
    
    def __init__(
        self,
        device: EdgeDevice,
        send_telemetry: bool = True
    ):
        """
        Initialize inference engine on a device.
        
        Args:
            device: Edge device to run on
            send_telemetry: Whether to collect performance metrics
        """
        self.device = device
        self.send_telemetry = send_telemetry
        self.loaded_models: Dict[str, CompiledModel] = {}
        self.inference_history: List[InferenceResult] = []
        self.telemetry: List[Dict[str, Any]] = []
    
    def load_model(self, model: CompiledModel) -> bool:
        """
        Load a compiled model for inference.
        
        Args:
            model: Compiled model to load
            
        Returns:
            True if loaded successfully
        """
        # Check memory
        required_memory = model.compiled_size_mb / 1024  # GB
        available_memory = self.device.capabilities.get("memory_gb", 0)
        
        # Account for already loaded models
        used_memory = sum(
            m.compiled_size_mb / 1024 for m in self.loaded_models.values()
        )
        
        if required_memory + used_memory > available_memory * 0.8:  # 80% limit
            logger.warning(
                f"Insufficient memory to load {model.name} on {self.device.name}"
            )
            return False
        
        self.loaded_models[model.name] = model
        logger.info(f"Loaded {model.name} on {self.device.name}")
        return True
    
    def predict(
        self,
        model_name: str,
        input_data: np.ndarray,
        return_confidence: bool = True
    ) -> InferenceResult:
        """
        Run inference on the edge device.
        
        Data never leaves the device - only metadata is collected.
        
        Args:
            model_name: Name of model to use
            input_data: Input data
            return_confidence: Include confidence score
            
        Returns:
            InferenceResult
        """
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.loaded_models[model_name]
        start_time = time.time()
        
        # Simulate inference (in real system, this would run the model)
        # Using model weights as simple linear layer simulation
        if model.weights is not None:
            flat_input = input_data.flatten()
            if len(flat_input) > len(model.weights.flatten()):
                flat_input = flat_input[:len(model.weights.flatten())]
            elif len(flat_input) < len(model.weights.flatten()):
                flat_input = np.pad(
                    flat_input, 
                    (0, len(model.weights.flatten()) - len(flat_input))
                )
            
            output = np.dot(flat_input, model.weights.flatten()[:len(flat_input)])
            prediction = float(1 / (1 + np.exp(-output)))  # Sigmoid
        else:
            prediction = 0.5
        
        # Apply latency improvement from optimization
        base_latency = 10.0  # ms
        actual_latency = base_latency / model.latency_improvement
        latency_ms = (time.time() - start_time) * 1000 + actual_latency
        
        # Create result
        confidence = abs(prediction - 0.5) * 2 if return_confidence else 1.0
        result = InferenceResult(
            device_id=self.device.device_id,
            model_name=model_name,
            model_version=model.version,
            prediction=prediction > 0.5,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata={
                "input_shape": list(input_data.shape),
                "quantization": model.quantization.value,
                "optimization": model.optimization.value
            }
        )
        
        self.inference_history.append(result)
        
        # Collect telemetry (metadata only, not the actual data)
        if self.send_telemetry:
            self.telemetry.append({
                "timestamp": datetime.now().isoformat(),
                "device_id": self.device.device_id,
                "model_name": model_name,
                "latency_ms": latency_ms,
                "input_shape": list(input_data.shape),
                # Note: NO actual input data or predictions sent
            })
        
        return result
    
    def batch_predict(
        self,
        model_name: str,
        batch_data: List[np.ndarray]
    ) -> List[InferenceResult]:
        """
        Run batch inference.
        
        Args:
            model_name: Model to use
            batch_data: List of inputs
            
        Returns:
            List of InferenceResults
        """
        return [self.predict(model_name, data) for data in batch_data]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get inference performance statistics."""
        if not self.inference_history:
            return {"total_inferences": 0}
        
        latencies = [r.latency_ms for r in self.inference_history]
        
        return {
            "device_id": self.device.device_id,
            "device_name": self.device.name,
            "total_inferences": len(self.inference_history),
            "avg_latency_ms": np.mean(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "models_loaded": list(self.loaded_models.keys()),
            "telemetry_entries": len(self.telemetry)
        }
    
    def get_telemetry_for_cloud(self) -> List[Dict[str, Any]]:
        """
        Get telemetry to send to cloud (metadata only).
        
        This is the only data that leaves the edge device.
        Actual inference data stays local for data residency.
        """
        return self.telemetry.copy()
    
    def clear_telemetry(self) -> int:
        """Clear telemetry after sending to cloud."""
        count = len(self.telemetry)
        self.telemetry = []
        return count


# ============================================================
# EDGE FLEET MANAGER
# ============================================================

class EdgeFleetManager:
    """
    Manage a fleet of edge devices.
    
    Provides:
    - Device registration and discovery
    - Fleet-wide deployment
    - Health monitoring
    - Telemetry aggregation
    """
    
    def __init__(self, fleet_name: str = "default"):
        """
        Initialize fleet manager.
        
        Args:
            fleet_name: Name of this fleet
        """
        self.fleet_name = fleet_name
        self.devices: Dict[str, EdgeDevice] = {}
        self.compiler = ModelCompiler()
        self.ota_manager = OTAUpdateManager()
        self.engines: Dict[str, EdgeInferenceEngine] = {}
    
    def register_device(self, device: EdgeDevice) -> None:
        """Register a device with the fleet."""
        self.devices[device.device_id] = device
        device.status = DeviceStatus.ONLINE
        device.last_heartbeat = datetime.now()
        logger.info(f"Registered device: {device.name} ({device.device_id})")
    
    def unregister_device(self, device_id: str) -> bool:
        """Unregister a device from the fleet."""
        if device_id in self.devices:
            del self.devices[device_id]
            if device_id in self.engines:
                del self.engines[device_id]
            return True
        return False
    
    def compile_and_deploy(
        self,
        model_name: str,
        model_weights: np.ndarray,
        version: str = "1.0.0",
        quantization: QuantizationType = QuantizationType.INT8,
        optimization: OptimizationType = OptimizationType.NONE,
        staged_rollout: bool = True
    ) -> Dict[str, Any]:
        """
        Compile a model and deploy to all devices.
        
        Args:
            model_name: Model name
            model_weights: Raw model weights
            version: Version string
            quantization: Quantization to apply
            optimization: Optimization to apply
            staged_rollout: Use canary deployment
            
        Returns:
            Deployment summary
        """
        # Compile
        compiled = self.compiler.compile(
            model_name=model_name,
            model_weights=model_weights,
            version=version,
            quantization=quantization,
            optimization=optimization
        )
        
        # Register with OTA manager
        self.ota_manager.register_model(compiled)
        
        # Deploy to fleet
        devices = list(self.devices.values())
        result = self.ota_manager.deploy_to_fleet(
            devices=devices,
            model=compiled,
            staged=staged_rollout
        )
        
        # Create inference engines for successful deployments
        for device in devices:
            if model_name in device.deployed_models:
                if device.device_id not in self.engines:
                    self.engines[device.device_id] = EdgeInferenceEngine(device)
                self.engines[device.device_id].load_model(compiled)
        
        result["compiled_model"] = {
            "model_id": compiled.model_id,
            "original_size_mb": compiled.original_size_mb,
            "compiled_size_mb": compiled.compiled_size_mb,
            "compression_ratio": compiled.get_compression_ratio(),
            "accuracy_delta": compiled.accuracy_delta
        }
        
        return result
    
    def get_fleet_health(self) -> Dict[str, Any]:
        """Get health status of the entire fleet."""
        online = sum(1 for d in self.devices.values() if d.is_online())
        
        return {
            "fleet_name": self.fleet_name,
            "total_devices": len(self.devices),
            "online_devices": online,
            "offline_devices": len(self.devices) - online,
            "health_percentage": (online / len(self.devices) * 100) if self.devices else 0,
            "devices": [
                {
                    "device_id": d.device_id,
                    "name": d.name,
                    "status": d.status.value,
                    "models": list(d.deployed_models.keys()),
                    "last_heartbeat": d.last_heartbeat.isoformat()
                }
                for d in self.devices.values()
            ]
        }
    
    def aggregate_telemetry(self) -> List[Dict[str, Any]]:
        """Aggregate telemetry from all engines."""
        all_telemetry = []
        for engine in self.engines.values():
            telemetry = engine.get_telemetry_for_cloud()
            all_telemetry.extend(telemetry)
            engine.clear_telemetry()
        return all_telemetry
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """Get aggregated inference statistics."""
        if not self.engines:
            return {"total_inferences": 0}
        
        all_stats = [e.get_performance_stats() for e in self.engines.values()]
        total_inferences = sum(s["total_inferences"] for s in all_stats)
        
        if total_inferences == 0:
            return {"total_inferences": 0}
        
        # Weight averages by inference count
        weighted_latency = sum(
            s["avg_latency_ms"] * s["total_inferences"] 
            for s in all_stats if s["total_inferences"] > 0
        ) / total_inferences
        
        return {
            "fleet_name": self.fleet_name,
            "total_devices": len(self.engines),
            "total_inferences": total_inferences,
            "avg_latency_ms": weighted_latency,
            "per_device_stats": all_stats
        }


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_sample_fleet(num_devices: int = 5) -> EdgeFleetManager:
    """
    Create a sample fleet for testing.
    
    Args:
        num_devices: Number of devices to create
        
    Returns:
        Configured EdgeFleetManager
    """
    fleet = EdgeFleetManager(fleet_name="sample_fleet")
    
    device_configs = [
        {"name": "Factory Floor 1", "location": "Building A", "gpu": True, "memory_gb": 8},
        {"name": "Factory Floor 2", "location": "Building A", "gpu": False, "memory_gb": 4},
        {"name": "Warehouse 1", "location": "Building B", "gpu": False, "memory_gb": 4},
        {"name": "Quality Station 1", "location": "Building A", "gpu": True, "memory_gb": 16},
        {"name": "Assembly Line 1", "location": "Building C", "gpu": False, "memory_gb": 2},
    ]
    
    for i in range(min(num_devices, len(device_configs))):
        config = device_configs[i]
        device = EdgeDevice(
            device_id=f"device_{i+1}",
            name=config["name"],
            location=config["location"],
            capabilities={
                "cpu_cores": 4 if not config.get("gpu") else 8,
                "memory_gb": config.get("memory_gb", 4),
                "gpu": config.get("gpu", False),
                "accelerator": "nvidia_jetson" if config.get("gpu") else None,
                "storage_gb": 64
            }
        )
        fleet.register_device(device)
    
    return fleet


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Create a fleet
    fleet = create_sample_fleet(5)
    
    # Create a sample model
    model_weights = np.random.randn(1000, 100).astype(np.float32)
    
    # Compile and deploy
    result = fleet.compile_and_deploy(
        model_name="defect_detector",
        model_weights=model_weights,
        version="1.0.0",
        quantization=QuantizationType.INT8,
        optimization=OptimizationType.PRUNING,
        staged_rollout=True
    )
    
    print(f"Deployment result: {result['successful']}/{result['total_devices']} successful")
    print(f"Compression: {result['compiled_model']['compression_ratio']:.1f}x")
    
    # Run inferences on each device
    for device_id, engine in fleet.engines.items():
        sample_input = np.random.randn(1, 100)
        result = engine.predict("defect_detector", sample_input)
        print(f"Device {device_id}: prediction={result.prediction}, latency={result.latency_ms:.2f}ms")
    
    # Get fleet stats
    print(f"\nFleet health: {fleet.get_fleet_health()['health_percentage']:.0f}%")
    print(f"Total inferences: {fleet.get_inference_stats()['total_inferences']}")
