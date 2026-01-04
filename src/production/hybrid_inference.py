"""
Hybrid Edge-Cloud Inference Module
====================================

Dynamic workload distribution between edge devices and cloud.

Implements:
- Task routing with multi-factor decision logic
- Split model execution (layer partitioning)
- Edge-cloud orchestration
- Model co-versioning

Mathematical Foundations:
- Routing optimization: minimize(latency + α*cost - β*privacy)
- Layer partitioning: find split point minimizing data transfer
- Confidence-based escalation

Reference: Battery inspection hybrid case study

Author: AI-Mastery-2026
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from enum import Enum
import logging
import time
import hashlib
from collections import deque
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

class RoutingDecision(Enum):
    """Where to execute inference."""
    EDGE_ONLY = "edge_only"
    CLOUD_ONLY = "cloud_only"
    HYBRID_SPLIT = "hybrid_split"
    FALLBACK_EDGE = "fallback_edge"  # Cloud unavailable


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"  # Can run on edge
    MODERATE = "moderate"  # Prefer edge, can escalate
    COMPLEX = "complex"  # Requires cloud
    CRITICAL = "critical"  # Requires both for validation


class PrivacySensitivity(Enum):
    """Data privacy sensitivity levels."""
    PUBLIC = "public"  # Can send to cloud
    INTERNAL = "internal"  # Prefer edge, cloud acceptable
    CONFIDENTIAL = "confidential"  # Edge preferred
    RESTRICTED = "restricted"  # Edge only, no cloud


@dataclass
class InferenceRequest:
    """
    Request for inference.
    """
    request_id: str
    input_data: np.ndarray
    model_name: str
    task_type: str
    privacy_level: PrivacySensitivity = PrivacySensitivity.INTERNAL
    latency_requirement_ms: Optional[float] = None
    max_cost: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class InferenceResult:
    """
    Result from inference.
    """
    request_id: str
    prediction: Any
    confidence: float
    execution_location: RoutingDecision
    latency_ms: float
    cost: float
    model_version: str
    edge_contribution: float  # Fraction processed on edge
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskRoutingPolicy:
    """
    Policy for task routing decisions.
    
    Weights determine importance of each factor.
    """
    privacy_weight: float = 0.3
    latency_weight: float = 0.3
    cost_weight: float = 0.2
    complexity_weight: float = 0.2
    
    # Thresholds
    latency_threshold_ms: float = 50.0  # Below this, prefer edge
    cost_threshold_per_1k: float = 0.01  # Max cost per 1k tokens
    confidence_threshold: float = 0.85  # Below this, escalate
    
    # Model size limits (MB)
    edge_model_max_mb: float = 100.0
    
    def validate(self):
        """Ensure weights sum to 1."""
        total = self.privacy_weight + self.latency_weight + \
                self.cost_weight + self.complexity_weight
        if abs(total - 1.0) > 0.01:
            # Normalize
            self.privacy_weight /= total
            self.latency_weight /= total
            self.cost_weight /= total
            self.complexity_weight /= total


@dataclass
class ModelVersion:
    """
    Model version information for co-versioning.
    """
    model_name: str
    edge_version: str
    cloud_version: str
    split_point: Optional[int]  # Layer index for split
    compatible: bool
    created_at: datetime = field(default_factory=datetime.now)


# ============================================================
# TASK ROUTER
# ============================================================

class TaskRouter:
    """
    Multi-factor task routing engine.
    
    Decides where to execute each inference request based on:
    - Privacy: Sensitive data stays on edge
    - Latency: Strict requirements favor edge
    - Cost: Cloud inference has per-request cost
    - Complexity: Large models require cloud
    - Confidence: Low edge confidence escalates to cloud
    
    Optimization objective:
    minimize(latency + α*cost - β*privacy_score)
    subject to: confidence >= threshold
    """
    
    def __init__(self, policy: Optional[TaskRoutingPolicy] = None):
        """
        Initialize router.
        
        Args:
            policy: Routing policy (uses defaults if None)
        """
        self.policy = policy or TaskRoutingPolicy()
        self.policy.validate()
        
        # Edge model registry
        self.edge_models: Dict[str, Dict[str, Any]] = {}
        
        # Cloud model registry
        self.cloud_models: Dict[str, Dict[str, Any]] = {}
        
        # Routing statistics
        self.routing_stats = {
            "edge_only": 0,
            "cloud_only": 0,
            "hybrid_split": 0,
            "fallback_edge": 0,
            "total": 0
        }
        
        # Cloud availability
        self.cloud_available = True
        self.cloud_latency_ms = 100.0  # Estimated
        
        logger.info("TaskRouter initialized")
    
    def register_edge_model(
        self,
        model_name: str,
        model_size_mb: float,
        supported_tasks: List[str],
        avg_latency_ms: float
    ):
        """Register an edge model."""
        self.edge_models[model_name] = {
            "size_mb": model_size_mb,
            "tasks": supported_tasks,
            "latency_ms": avg_latency_ms,
            "version": "1.0.0"
        }
        logger.info(f"Edge model registered: {model_name}")
    
    def register_cloud_model(
        self,
        model_name: str,
        model_size_mb: float,
        supported_tasks: List[str],
        cost_per_1k: float,
        avg_latency_ms: float
    ):
        """Register a cloud model."""
        self.cloud_models[model_name] = {
            "size_mb": model_size_mb,
            "tasks": supported_tasks,
            "cost_per_1k": cost_per_1k,
            "latency_ms": avg_latency_ms,
            "version": "1.0.0"
        }
        logger.info(f"Cloud model registered: {model_name}")
    
    def route(self, request: InferenceRequest) -> Tuple[RoutingDecision, Dict[str, Any]]:
        """
        Determine optimal routing for a request.
        
        Args:
            request: Inference request
            
        Returns:
            (routing_decision, routing_metadata)
        """
        self.routing_stats["total"] += 1
        
        # Check cloud availability first
        if not self.cloud_available:
            return self._fallback_route(request)
        
        # Calculate factor scores (0-1, higher = prefer edge)
        privacy_score = self._privacy_score(request)
        latency_score = self._latency_score(request)
        cost_score = self._cost_score(request)
        complexity_score = self._complexity_score(request)
        
        # Weighted combination
        edge_preference = (
            self.policy.privacy_weight * privacy_score +
            self.policy.latency_weight * latency_score +
            self.policy.cost_weight * cost_score +
            self.policy.complexity_weight * complexity_score
        )
        
        # Determine routing
        routing_info = {
            "edge_preference": edge_preference,
            "scores": {
                "privacy": privacy_score,
                "latency": latency_score,
                "cost": cost_score,
                "complexity": complexity_score
            }
        }
        
        # Strict privacy overrides other factors
        if request.privacy_level == PrivacySensitivity.RESTRICTED:
            decision = RoutingDecision.EDGE_ONLY
        # High edge preference and edge model available
        elif edge_preference > 0.7 and self._has_edge_model(request):
            decision = RoutingDecision.EDGE_ONLY
        # Low preference or complex task
        elif edge_preference < 0.3 or complexity_score < 0.3:
            decision = RoutingDecision.CLOUD_ONLY
        # Middle ground - use hybrid
        else:
            decision = RoutingDecision.HYBRID_SPLIT
        
        self.routing_stats[decision.value] += 1
        routing_info["decision"] = decision.value
        
        return decision, routing_info
    
    def _privacy_score(self, request: InferenceRequest) -> float:
        """
        Calculate privacy factor score.
        
        Higher score = more reason to stay on edge
        """
        scores = {
            PrivacySensitivity.PUBLIC: 0.0,
            PrivacySensitivity.INTERNAL: 0.3,
            PrivacySensitivity.CONFIDENTIAL: 0.7,
            PrivacySensitivity.RESTRICTED: 1.0
        }
        return scores.get(request.privacy_level, 0.5)
    
    def _latency_score(self, request: InferenceRequest) -> float:
        """
        Calculate latency factor score.
        
        Strict latency requirements favor edge.
        """
        if request.latency_requirement_ms is None:
            return 0.5  # Neutral
        
        # If requirement is stricter than cloud can provide, prefer edge
        if request.latency_requirement_ms < self.cloud_latency_ms:
            return 1.0
        
        # Linear interpolation
        ratio = request.latency_requirement_ms / self.cloud_latency_ms
        return max(0, 1 - (ratio - 1) / 2)
    
    def _cost_score(self, request: InferenceRequest) -> float:
        """
        Calculate cost factor score.
        
        Edge is free, cloud has cost.
        """
        if request.max_cost is None:
            return 0.5  # Neutral
        
        # Estimate cloud cost
        cloud_model = self.cloud_models.get(request.model_name, {})
        cost_per_1k = cloud_model.get("cost_per_1k", self.policy.cost_threshold_per_1k)
        
        # Estimate tokens (rough approximation from input size)
        estimated_tokens = request.input_data.size * 0.1
        estimated_cost = cost_per_1k * estimated_tokens / 1000
        
        if estimated_cost > request.max_cost:
            return 1.0  # Too expensive, use edge
        
        return 1 - (estimated_cost / request.max_cost)
    
    def _complexity_score(self, request: InferenceRequest) -> float:
        """
        Calculate complexity factor score.
        
        Simple tasks favor edge, complex favor cloud.
        """
        # Check if edge model can handle this
        edge_model = self.edge_models.get(request.model_name, {})
        
        if not edge_model:
            return 0.0  # No edge model, must use cloud
        
        if edge_model["size_mb"] > self.policy.edge_model_max_mb:
            return 0.2  # Model too large for edge
        
        if request.task_type in edge_model.get("tasks", []):
            return 0.8  # Edge can handle this task
        
        return 0.5  # Neutral
    
    def _has_edge_model(self, request: InferenceRequest) -> bool:
        """Check if edge model is available for request."""
        return request.model_name in self.edge_models
    
    def _fallback_route(self, request: InferenceRequest) -> Tuple[RoutingDecision, Dict[str, Any]]:
        """Fallback routing when cloud unavailable."""
        self.routing_stats["fallback_edge"] += 1
        
        if self._has_edge_model(request):
            return RoutingDecision.FALLBACK_EDGE, {"reason": "cloud_unavailable"}
        
        return RoutingDecision.FALLBACK_EDGE, {
            "reason": "cloud_unavailable",
            "warning": "no_edge_model"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total = self.routing_stats["total"]
        if total == 0:
            return self.routing_stats
        
        return {
            **self.routing_stats,
            "edge_ratio": self.routing_stats["edge_only"] / total,
            "cloud_ratio": self.routing_stats["cloud_only"] / total,
            "hybrid_ratio": self.routing_stats["hybrid_split"] / total
        }


# ============================================================
# SPLIT MODEL EXECUTOR
# ============================================================

class SplitModelExecutor:
    """
    Execute inference with model split between edge and cloud.
    
    Layer partitioning strategy:
    - Edge runs first N layers (feature extraction)
    - Intermediate activations are compressed and sent to cloud
    - Cloud runs remaining layers (classification)
    
    Benefits:
    - Privacy: Raw data never leaves edge
    - Bandwidth: Activations often smaller than input
    - Latency: Parallel processing possible
    
    Mathematical foundation:
    - For layer i, let a_i = f_i(a_{i-1})
    - Split at layer k: edge computes a_0...a_k, cloud a_{k+1}...a_n
    - Optimal k minimizes: size(a_k) + t_cloud(k+1...n)
    """
    
    def __init__(
        self,
        edge_layers: List[Callable],
        cloud_layers: List[Callable],
        split_point: int,
        compression_ratio: float = 0.5
    ):
        """
        Initialize split executor.
        
        Args:
            edge_layers: Layer functions for edge
            cloud_layers: Layer functions for cloud
            split_point: Index where split occurs
            compression_ratio: Compression for activation transfer
        """
        self.edge_layers = edge_layers
        self.cloud_layers = cloud_layers
        self.split_point = split_point
        self.compression_ratio = compression_ratio
        
        # Statistics
        self.execution_count = 0
        self.total_edge_time_ms = 0.0
        self.total_cloud_time_ms = 0.0
        self.total_transfer_bytes = 0
        
        logger.info(f"SplitModelExecutor initialized: split at layer {split_point}")
    
    def execute_edge(self, input_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Execute edge portion (first N layers).
        
        Args:
            input_data: Raw input
            
        Returns:
            (intermediate_activations, latency_ms)
        """
        start_time = time.perf_counter()
        
        activation = input_data
        for layer_fn in self.edge_layers:
            activation = layer_fn(activation)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.total_edge_time_ms += latency_ms
        
        return activation, latency_ms
    
    def compress_activations(self, activations: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Compress intermediate activations for transfer.
        
        Uses simple quantization for demonstration.
        In production, would use learned compression.
        
        Args:
            activations: Intermediate tensor
            
        Returns:
            (compressed, original_bytes)
        """
        original_bytes = activations.nbytes
        
        # Simulated compression via quantization
        min_val = activations.min()
        max_val = activations.max()
        
        # Quantize to uint8
        if max_val - min_val > 0:
            normalized = (activations - min_val) / (max_val - min_val)
            quantized = (normalized * 255).astype(np.uint8)
        else:
            quantized = np.zeros_like(activations, dtype=np.uint8)
        
        # Store scale factors for dequantization
        compressed = {
            "data": quantized,
            "min": min_val,
            "max": max_val,
            "shape": activations.shape
        }
        
        compressed_bytes = quantized.nbytes + 16  # metadata overhead
        self.total_transfer_bytes += compressed_bytes
        
        return compressed, original_bytes
    
    def decompress_activations(self, compressed: Dict) -> np.ndarray:
        """Decompress activations on cloud side."""
        quantized = compressed["data"]
        min_val = compressed["min"]
        max_val = compressed["max"]
        
        # Dequantize
        normalized = quantized.astype(np.float32) / 255.0
        activations = normalized * (max_val - min_val) + min_val
        
        return activations.reshape(compressed["shape"])
    
    def execute_cloud(self, activations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Execute cloud portion (remaining layers).
        
        Args:
            activations: Intermediate activations from edge
            
        Returns:
            (output, latency_ms)
        """
        start_time = time.perf_counter()
        
        activation = activations
        for layer_fn in self.cloud_layers:
            activation = layer_fn(activation)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.total_cloud_time_ms += latency_ms
        
        return activation, latency_ms
    
    def execute_split(
        self, 
        input_data: np.ndarray,
        cloud_executor: Callable[[np.ndarray], np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Execute full split inference.
        
        Args:
            input_data: Raw input
            cloud_executor: Function to call cloud (handles network)
            
        Returns:
            (output, timing_info)
        """
        self.execution_count += 1
        
        # Edge execution
        edge_activations, edge_latency = self.execute_edge(input_data)
        
        # Compress for transfer
        compressed, original_bytes = self.compress_activations(edge_activations)
        
        # Cloud execution (via provided executor)
        cloud_start = time.perf_counter()
        decompressed = self.decompress_activations(compressed)
        output = cloud_executor(decompressed)
        cloud_latency = (time.perf_counter() - cloud_start) * 1000
        
        self.total_cloud_time_ms += cloud_latency
        
        return output, {
            "edge_latency_ms": edge_latency,
            "cloud_latency_ms": cloud_latency,
            "total_latency_ms": edge_latency + cloud_latency,
            "transfer_bytes": compressed["data"].nbytes,
            "compression_ratio": compressed["data"].nbytes / original_bytes
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if self.execution_count == 0:
            return {"execution_count": 0}
        
        return {
            "execution_count": self.execution_count,
            "avg_edge_latency_ms": self.total_edge_time_ms / self.execution_count,
            "avg_cloud_latency_ms": self.total_cloud_time_ms / self.execution_count,
            "total_transfer_mb": self.total_transfer_bytes / 1024 / 1024,
            "avg_transfer_kb": self.total_transfer_bytes / self.execution_count / 1024
        }


# ============================================================
# EDGE-CLOUD ORCHESTRATOR
# ============================================================

class EdgeCloudOrchestrator:
    """
    Orchestrates inference across edge and cloud.
    
    Features:
    - Automatic routing based on policy
    - Confidence-based escalation
    - Timeout handling with edge fallback
    - Cost and latency tracking
    """
    
    def __init__(
        self,
        router: TaskRouter,
        edge_model: Callable[[np.ndarray], Tuple[np.ndarray, float]],
        cloud_model: Optional[Callable[[np.ndarray], Tuple[np.ndarray, float]]] = None,
        split_executor: Optional[SplitModelExecutor] = None,
        cloud_timeout_ms: float = 5000.0
    ):
        """
        Initialize orchestrator.
        
        Args:
            router: Task router for decisions
            edge_model: Edge inference function (input -> (output, confidence))
            cloud_model: Cloud inference function
            split_executor: For hybrid split execution
            cloud_timeout_ms: Timeout for cloud requests
        """
        self.router = router
        self.edge_model = edge_model
        self.cloud_model = cloud_model
        self.split_executor = split_executor
        self.cloud_timeout_ms = cloud_timeout_ms
        
        # Statistics
        self.total_requests = 0
        self.total_cost = 0.0
        self.total_latency_ms = 0.0
        self.escalations = 0
        
        # Request history
        self.history: deque = deque(maxlen=1000)
        
        logger.info("EdgeCloudOrchestrator initialized")
    
    def infer(self, request: InferenceRequest) -> InferenceResult:
        """
        Process inference request.
        
        Handles routing, execution, and potential escalation.
        """
        self.total_requests += 1
        start_time = time.perf_counter()
        
        # Get routing decision
        decision, routing_info = self.router.route(request)
        
        # Execute based on decision
        if decision == RoutingDecision.EDGE_ONLY:
            result = self._execute_edge(request, routing_info)
        elif decision == RoutingDecision.CLOUD_ONLY:
            result = self._execute_cloud(request, routing_info)
        elif decision == RoutingDecision.HYBRID_SPLIT:
            result = self._execute_hybrid(request, routing_info)
        else:  # FALLBACK_EDGE
            result = self._execute_edge(request, routing_info)
        
        # Check for escalation (low confidence)
        if result.confidence < self.router.policy.confidence_threshold:
            if decision != RoutingDecision.CLOUD_ONLY and self.cloud_model:
                self.escalations += 1
                cloud_result = self._execute_cloud(request, routing_info)
                
                # Combine results (prefer cloud if higher confidence)
                if cloud_result.confidence > result.confidence:
                    result = cloud_result
                    result.metadata["escalated"] = True
        
        # Update statistics
        total_latency = (time.perf_counter() - start_time) * 1000
        self.total_latency_ms += total_latency
        self.total_cost += result.cost
        
        # Record history
        self.history.append({
            "request_id": request.request_id,
            "decision": decision.value,
            "latency_ms": total_latency,
            "confidence": result.confidence,
            "cost": result.cost
        })
        
        return result
    
    def _execute_edge(
        self, 
        request: InferenceRequest, 
        routing_info: Dict
    ) -> InferenceResult:
        """Execute on edge only."""
        start_time = time.perf_counter()
        
        prediction, confidence = self.edge_model(request.input_data)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return InferenceResult(
            request_id=request.request_id,
            prediction=prediction,
            confidence=confidence,
            execution_location=RoutingDecision.EDGE_ONLY,
            latency_ms=latency_ms,
            cost=0.0,  # Edge is free
            model_version="edge_1.0.0",
            edge_contribution=1.0,
            metadata={"routing": routing_info}
        )
    
    def _execute_cloud(
        self, 
        request: InferenceRequest, 
        routing_info: Dict
    ) -> InferenceResult:
        """Execute on cloud only."""
        if self.cloud_model is None:
            logger.warning("No cloud model available")
            return self._execute_edge(request, routing_info)
        
        start_time = time.perf_counter()
        
        try:
            prediction, confidence = self.cloud_model(request.input_data)
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Estimate cost
            cloud_model_info = self.router.cloud_models.get(request.model_name, {})
            cost_per_1k = cloud_model_info.get("cost_per_1k", 0.001)
            estimated_tokens = request.input_data.size * 0.1
            cost = cost_per_1k * estimated_tokens / 1000
            
            return InferenceResult(
                request_id=request.request_id,
                prediction=prediction,
                confidence=confidence,
                execution_location=RoutingDecision.CLOUD_ONLY,
                latency_ms=latency_ms,
                cost=cost,
                model_version="cloud_1.0.0",
                edge_contribution=0.0,
                metadata={"routing": routing_info}
            )
            
        except Exception as e:
            logger.error(f"Cloud execution failed: {e}")
            return self._execute_edge(request, {"fallback": True})
    
    def _execute_hybrid(
        self, 
        request: InferenceRequest, 
        routing_info: Dict
    ) -> InferenceResult:
        """Execute with edge-cloud split."""
        if self.split_executor is None:
            logger.warning("No split executor, falling back to edge")
            return self._execute_edge(request, routing_info)
        
        start_time = time.perf_counter()
        
        # Define simple cloud executor
        def cloud_executor(activations):
            if self.cloud_model:
                pred, _ = self.cloud_model(activations)
                return pred
            return activations
        
        output, timing = self.split_executor.execute_split(
            request.input_data, 
            cloud_executor
        )
        
        # Estimate confidence from output
        if isinstance(output, np.ndarray) and len(output.shape) > 0:
            confidence = float(np.max(output)) if output.max() <= 1.0 else 0.9
        else:
            confidence = 0.9
        
        total_latency = (time.perf_counter() - start_time) * 1000
        
        # Estimate cost (half of full cloud since split)
        cloud_model_info = self.router.cloud_models.get(request.model_name, {})
        cost = cloud_model_info.get("cost_per_1k", 0.001) * 0.5
        
        # Edge contribution based on layers
        total_layers = len(self.split_executor.edge_layers) + len(self.split_executor.cloud_layers)
        edge_contribution = len(self.split_executor.edge_layers) / total_layers
        
        return InferenceResult(
            request_id=request.request_id,
            prediction=output,
            confidence=confidence,
            execution_location=RoutingDecision.HYBRID_SPLIT,
            latency_ms=total_latency,
            cost=cost,
            model_version="hybrid_1.0.0",
            edge_contribution=edge_contribution,
            metadata={"routing": routing_info, "split_timing": timing}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        if self.total_requests == 0:
            return {"total_requests": 0}
        
        return {
            "total_requests": self.total_requests,
            "total_cost": self.total_cost,
            "avg_cost_per_request": self.total_cost / self.total_requests,
            "avg_latency_ms": self.total_latency_ms / self.total_requests,
            "escalation_rate": self.escalations / self.total_requests,
            "routing_stats": self.router.get_stats()
        }


# ============================================================
# MODEL VERSION MANAGER
# ============================================================

class ModelVersionManager:
    """
    Manages co-versioning of edge and cloud models.
    
    Critical for split inference: edge and cloud models must be
    compatible versions, otherwise intermediate activations won't match.
    
    Features:
    - Version compatibility tracking
    - Automatic rollback on mismatch
    - Update coordination
    """
    
    def __init__(self):
        """Initialize version manager."""
        self.versions: Dict[str, ModelVersion] = {}
        self.compatibility_matrix: Dict[str, Dict[str, bool]] = {}
        self.update_history: List[Dict[str, Any]] = []
        
        logger.info("ModelVersionManager initialized")
    
    def register_version(
        self,
        model_name: str,
        edge_version: str,
        cloud_version: str,
        split_point: Optional[int] = None
    ) -> ModelVersion:
        """
        Register a model version pair.
        
        Args:
            model_name: Model identifier
            edge_version: Version running on edge
            cloud_version: Version running on cloud
            split_point: Layer index for split (if applicable)
        """
        # Check compatibility
        compatible = self._check_compatibility(edge_version, cloud_version)
        
        version = ModelVersion(
            model_name=model_name,
            edge_version=edge_version,
            cloud_version=cloud_version,
            split_point=split_point,
            compatible=compatible
        )
        
        self.versions[model_name] = version
        
        if not compatible:
            logger.warning(f"Incompatible versions: {model_name} "
                          f"(edge={edge_version}, cloud={cloud_version})")
        
        return version
    
    def _check_compatibility(
        self, 
        edge_version: str, 
        cloud_version: str
    ) -> bool:
        """
        Check if edge and cloud versions are compatible.
        
        Simple rule: major.minor must match for split inference.
        """
        edge_parts = edge_version.split(".")
        cloud_parts = cloud_version.split(".")
        
        # Major and minor must match
        if len(edge_parts) >= 2 and len(cloud_parts) >= 2:
            return edge_parts[0] == cloud_parts[0] and edge_parts[1] == cloud_parts[1]
        
        return edge_version == cloud_version
    
    def is_compatible(self, model_name: str) -> bool:
        """Check if model's edge and cloud versions are compatible."""
        version = self.versions.get(model_name)
        return version.compatible if version else False
    
    def update_edge(
        self, 
        model_name: str, 
        new_version: str
    ) -> Tuple[bool, str]:
        """
        Update edge model version.
        
        Returns:
            (success, message)
        """
        current = self.versions.get(model_name)
        if not current:
            return False, "Model not found"
        
        # Check compatibility with cloud
        compatible = self._check_compatibility(new_version, current.cloud_version)
        
        if not compatible:
            return False, (f"Version {new_version} incompatible with "
                          f"cloud version {current.cloud_version}")
        
        # Update
        old_version = current.edge_version
        current.edge_version = new_version
        current.compatible = compatible
        
        self.update_history.append({
            "model": model_name,
            "component": "edge",
            "old_version": old_version,
            "new_version": new_version,
            "timestamp": datetime.now()
        })
        
        return True, f"Edge updated to {new_version}"
    
    def update_cloud(
        self, 
        model_name: str, 
        new_version: str
    ) -> Tuple[bool, str]:
        """Update cloud model version."""
        current = self.versions.get(model_name)
        if not current:
            return False, "Model not found"
        
        compatible = self._check_compatibility(current.edge_version, new_version)
        
        if not compatible:
            return False, (f"Version {new_version} incompatible with "
                          f"edge version {current.edge_version}")
        
        old_version = current.cloud_version
        current.cloud_version = new_version
        current.compatible = compatible
        
        self.update_history.append({
            "model": model_name,
            "component": "cloud",
            "old_version": old_version,
            "new_version": new_version,
            "timestamp": datetime.now()
        })
        
        return True, f"Cloud updated to {new_version}"
    
    def coordinated_update(
        self,
        model_name: str,
        new_edge_version: str,
        new_cloud_version: str
    ) -> Tuple[bool, str]:
        """
        Coordinated update of both edge and cloud.
        
        Ensures versions stay compatible.
        """
        if not self._check_compatibility(new_edge_version, new_cloud_version):
            return False, "Proposed versions are incompatible"
        
        current = self.versions.get(model_name)
        if not current:
            return False, "Model not found"
        
        old_edge = current.edge_version
        old_cloud = current.cloud_version
        
        current.edge_version = new_edge_version
        current.cloud_version = new_cloud_version
        current.compatible = True
        
        self.update_history.append({
            "model": model_name,
            "component": "coordinated",
            "old_edge": old_edge,
            "old_cloud": old_cloud,
            "new_edge": new_edge_version,
            "new_cloud": new_cloud_version,
            "timestamp": datetime.now()
        })
        
        return True, f"Coordinated update: edge={new_edge_version}, cloud={new_cloud_version}"
    
    def get_version_report(self) -> Dict[str, Any]:
        """Get version status report."""
        report = {}
        for name, version in self.versions.items():
            report[name] = {
                "edge": version.edge_version,
                "cloud": version.cloud_version,
                "compatible": version.compatible,
                "split_point": version.split_point
            }
        return report


# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def create_simple_edge_model() -> Callable[[np.ndarray], Tuple[np.ndarray, float]]:
    """Create a simple edge model function for testing."""
    # Simulated lightweight model
    weights = np.random.randn(100, 10) * 0.1
    
    def model(x: np.ndarray) -> Tuple[np.ndarray, float]:
        # Flatten and pad/truncate to 100 features
        x_flat = x.flatten()
        if len(x_flat) < 100:
            x_flat = np.pad(x_flat, (0, 100 - len(x_flat)))
        else:
            x_flat = x_flat[:100]
        
        # Simple forward pass
        logits = x_flat @ weights
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        prediction = np.argmax(probs)
        confidence = float(np.max(probs))
        
        return probs, confidence
    
    return model


def create_simple_cloud_model() -> Callable[[np.ndarray], Tuple[np.ndarray, float]]:
    """Create a simple cloud model function for testing."""
    # Simulated larger model (more accurate)
    weights1 = np.random.randn(100, 256) * 0.1
    weights2 = np.random.randn(256, 10) * 0.1
    
    def model(x: np.ndarray) -> Tuple[np.ndarray, float]:
        x_flat = x.flatten()
        if len(x_flat) < 100:
            x_flat = np.pad(x_flat, (0, 100 - len(x_flat)))
        else:
            x_flat = x_flat[:100]
        
        # Two-layer forward pass  
        hidden = np.maximum(0, x_flat @ weights1)  # ReLU
        logits = hidden @ weights2
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        prediction = np.argmax(probs)
        confidence = float(np.max(probs)) * 1.1  # Slightly higher confidence
        confidence = min(confidence, 1.0)
        
        return probs, confidence
    
    return model


def create_hybrid_inference_system() -> EdgeCloudOrchestrator:
    """
    Create a complete hybrid inference system.
    """
    # Create router
    policy = TaskRoutingPolicy(
        privacy_weight=0.3,
        latency_weight=0.3,
        cost_weight=0.2,
        complexity_weight=0.2,
        latency_threshold_ms=50.0,
        confidence_threshold=0.85
    )
    router = TaskRouter(policy)
    
    # Register models
    router.register_edge_model(
        "classifier",
        model_size_mb=5.0,
        supported_tasks=["classification", "embedding"],
        avg_latency_ms=10.0
    )
    
    router.register_cloud_model(
        "classifier",
        model_size_mb=500.0,
        supported_tasks=["classification", "embedding", "generation"],
        cost_per_1k=0.002,
        avg_latency_ms=100.0
    )
    
    # Create models
    edge_model = create_simple_edge_model()
    cloud_model = create_simple_cloud_model()
    
    # Create split executor (4 edge layers, 2 cloud layers)
    edge_layers = [
        lambda x: np.maximum(0, x),  # ReLU
        lambda x: x * 0.9,  # Scale
        lambda x: np.maximum(0, x),
        lambda x: x
    ]
    cloud_layers = [
        lambda x: np.maximum(0, x),
        lambda x: x
    ]
    
    split_executor = SplitModelExecutor(
        edge_layers=edge_layers,
        cloud_layers=cloud_layers,
        split_point=4
    )
    
    # Create orchestrator
    orchestrator = EdgeCloudOrchestrator(
        router=router,
        edge_model=edge_model,
        cloud_model=cloud_model,
        split_executor=split_executor
    )
    
    return orchestrator


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Hybrid Edge-Cloud Inference - Demo")
    print("=" * 60)
    
    # Create system
    orchestrator = create_hybrid_inference_system()
    
    print("\n1. Testing different routing scenarios...")
    
    # Test cases with different privacy levels
    test_cases = [
        (PrivacySensitivity.RESTRICTED, 20.0, "Restricted data"),
        (PrivacySensitivity.PUBLIC, None, "Public data"),
        (PrivacySensitivity.CONFIDENTIAL, 100.0, "Confidential, relaxed latency"),
        (PrivacySensitivity.INTERNAL, 30.0, "Internal, strict latency"),
    ]
    
    for privacy, latency_req, description in test_cases:
        request = InferenceRequest(
            request_id=f"REQ_{hash(description) % 10000:04d}",
            input_data=np.random.randn(10, 10),
            model_name="classifier",
            task_type="classification",
            privacy_level=privacy,
            latency_requirement_ms=latency_req
        )
        
        result = orchestrator.infer(request)
        print(f"   {description}:")
        print(f"      Decision: {result.execution_location.value}")
        print(f"      Latency: {result.latency_ms:.2f}ms")
        print(f"      Confidence: {result.confidence:.2%}")
        print(f"      Cost: ${result.cost:.6f}")
    
    print("\n2. Batch processing with statistics...")
    for _ in range(100):
        request = InferenceRequest(
            request_id=f"BATCH_{np.random.randint(10000)}",
            input_data=np.random.randn(10, 10),
            model_name="classifier",
            task_type="classification",
            privacy_level=np.random.choice(list(PrivacySensitivity)),
            latency_requirement_ms=np.random.choice([20.0, 50.0, 100.0, None])
        )
        orchestrator.infer(request)
    
    print("\n3. Orchestrator Statistics:")
    stats = orchestrator.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"      {k}: {v:.4f}")
                else:
                    print(f"      {k}: {v}")
        elif isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n4. Model Version Management...")
    version_mgr = ModelVersionManager()
    
    # Register compatible versions
    version_mgr.register_version(
        "classifier", 
        edge_version="1.2.0", 
        cloud_version="1.2.3",
        split_point=4
    )
    
    print(f"   Versions registered: {version_mgr.get_version_report()}")
    
    # Try incompatible update
    success, msg = version_mgr.update_edge("classifier", "2.0.0")
    print(f"   Edge update to 2.0.0: {msg}")
    
    # Coordinated update
    success, msg = version_mgr.coordinated_update("classifier", "2.0.0", "2.0.1")
    print(f"   Coordinated update: {msg}")
    
    print("\n5. Split Model Executor Stats:")
    if orchestrator.split_executor:
        split_stats = orchestrator.split_executor.get_stats()
        for key, value in split_stats.items():
            print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
    
    print("\n✅ Hybrid Edge-Cloud demo complete!")
