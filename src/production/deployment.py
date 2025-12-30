"""
Production Deployment Module
============================
Deployment utilities for ML models including serialization,
health checks, load balancing, and configuration.

Production Considerations:
- Model versioning and rollback
- Graceful shutdown
- Blue-green deployments
- Canary releases

Author: AI-Mastery-2026
"""

import os
import sys
import json
import time
import signal
import logging
import hashlib
from typing import Any, Dict, Optional, List, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import threading
import pickle

logger = logging.getLogger(__name__)


# ============================================================
# DEPLOYMENT CONFIGURATION
# ============================================================

class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DeploymentConfig:
    """
    Deployment configuration for ML services.
    
    Can be loaded from environment variables or config files.
    """
    # Service settings
    service_name: str = "ml-service"
    service_version: str = "1.0.0"
    environment: Environment = Environment.DEVELOPMENT
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 30
    
    # Model settings
    model_path: str = ""
    model_version: str = "v1"
    
    # Feature flags
    enable_metrics: bool = True
    enable_tracing: bool = False
    enable_caching: bool = True
    
    # Resource limits
    max_batch_size: int = 32
    max_request_size_mb: int = 10
    rate_limit_per_minute: int = 1000
    
    # Health check
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"
    liveness_check_path: str = "/live"
    
    @classmethod
    def from_env(cls) -> 'DeploymentConfig':
        """Load configuration from environment variables."""
        env_str = os.getenv("ENVIRONMENT", "development").lower()
        
        return cls(
            service_name=os.getenv("SERVICE_NAME", "ml-service"),
            service_version=os.getenv("SERVICE_VERSION", "1.0.0"),
            environment=Environment(env_str),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            workers=int(os.getenv("WORKERS", "4")),
            timeout=int(os.getenv("TIMEOUT", "30")),
            model_path=os.getenv("MODEL_PATH", ""),
            model_version=os.getenv("MODEL_VERSION", "v1"),
            enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true",
            enable_tracing=os.getenv("ENABLE_TRACING", "false").lower() == "true",
            enable_caching=os.getenv("ENABLE_CACHING", "true").lower() == "true",
            max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "32")),
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT", "1000")),
        )
    
    @classmethod
    def from_file(cls, path: str) -> 'DeploymentConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Handle environment enum
        if 'environment' in data:
            data['environment'] = Environment(data['environment'])
        
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['environment'] = self.environment.value
        return d
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================
# MODEL SERIALIZATION
# ============================================================

class ModelFormat(Enum):
    """Supported model serialization formats."""
    PICKLE = "pickle"
    JOBLIB = "joblib"
    ONNX = "onnx"
    TORCH = "torch"
    TENSORFLOW = "tensorflow"


@dataclass
class ModelMetadata:
    """Metadata for serialized models."""
    model_name: str
    model_version: str
    format: ModelFormat
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    tags: List[str] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['format'] = self.format.value
        return d


class ModelSerializer:
    """
    Unified model serialization across formats.
    
    Supports pickle, joblib, ONNX, PyTorch, and TensorFlow.
    
    Example:
        >>> serializer = ModelSerializer()
        >>> serializer.save(model, "model.pkl", format=ModelFormat.PICKLE)
        >>> loaded_model = serializer.load("model.pkl")
    """
    
    @staticmethod
    def save(
        model: Any,
        path: str,
        format: ModelFormat = ModelFormat.PICKLE,
        metadata: Optional[ModelMetadata] = None,
        compress: bool = True
    ) -> str:
        """
        Save model to file.
        
        Args:
            model: Model object to save
            path: Output file path
            format: Serialization format
            metadata: Optional model metadata
            compress: Whether to compress (for pickle/joblib)
        
        Returns:
            Path to saved model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == ModelFormat.PICKLE:
            with open(path, 'wb') as f:
                pickle.dump(model, f)
        
        elif format == ModelFormat.JOBLIB:
            import joblib
            joblib.dump(model, path, compress=compress)
        
        elif format == ModelFormat.ONNX:
            # Model should already be in ONNX format
            with open(path, 'wb') as f:
                f.write(model.SerializeToString())
        
        elif format == ModelFormat.TORCH:
            import torch
            torch.save(model.state_dict() if hasattr(model, 'state_dict') else model, path)
        
        elif format == ModelFormat.TENSORFLOW:
            model.save(str(path))
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Save metadata alongside model
        if metadata:
            metadata_path = path.with_suffix('.meta.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
        
        logger.info(f"Saved model to {path}")
        return str(path)
    
    @staticmethod
    def load(
        path: str,
        format: Optional[ModelFormat] = None,
        model_class: Optional[type] = None
    ) -> Any:
        """
        Load model from file.
        
        Args:
            path: Model file path
            format: Format (auto-detected if not specified)
            model_class: For PyTorch, the model class to instantiate
        
        Returns:
            Loaded model
        """
        path = Path(path)
        
        # Auto-detect format from extension
        if format is None:
            ext = path.suffix.lower()
            format_map = {
                '.pkl': ModelFormat.PICKLE,
                '.pickle': ModelFormat.PICKLE,
                '.joblib': ModelFormat.JOBLIB,
                '.onnx': ModelFormat.ONNX,
                '.pt': ModelFormat.TORCH,
                '.pth': ModelFormat.TORCH,
            }
            format = format_map.get(ext, ModelFormat.PICKLE)
        
        if format == ModelFormat.PICKLE:
            with open(path, 'rb') as f:
                return pickle.load(f)
        
        elif format == ModelFormat.JOBLIB:
            import joblib
            return joblib.load(path)
        
        elif format == ModelFormat.ONNX:
            import onnx
            return onnx.load(str(path))
        
        elif format == ModelFormat.TORCH:
            import torch
            state_dict = torch.load(path, map_location='cpu')
            if model_class:
                model = model_class()
                model.load_state_dict(state_dict)
                return model
            return state_dict
        
        elif format == ModelFormat.TENSORFLOW:
            import tensorflow as tf
            return tf.keras.models.load_model(str(path))
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def get_metadata(path: str) -> Optional[ModelMetadata]:
        """Load model metadata if available."""
        metadata_path = Path(path).with_suffix('.meta.json')
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        data['format'] = ModelFormat(data['format'])
        return ModelMetadata(**data)


# ============================================================
# HEALTH CHECKS
# ============================================================

class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """
    Health check system for ML services.
    
    Supports multiple check types:
        - Liveness: Is the service running?
        - Readiness: Can the service handle requests?
        - Dependency: Are external dependencies healthy?
    
    Example:
        >>> checker = HealthChecker()
        >>> checker.add_check("model", lambda: model is not None)
        >>> checker.add_check("redis", check_redis_connection)
        >>> result = checker.run_all()
    """
    
    def __init__(self):
        self.checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._start_time = time.time()
    
    def add_check(
        self,
        name: str,
        check_fn: Callable[[], Union[bool, HealthCheckResult]],
        critical: bool = True
    ):
        """
        Add a health check.
        
        Args:
            name: Check name
            check_fn: Function that returns True/False or HealthCheckResult
            critical: If True, service is unhealthy when check fails
        """
        def wrapper():
            start = time.time()
            try:
                result = check_fn()
                latency = (time.time() - start) * 1000
                
                if isinstance(result, HealthCheckResult):
                    result.latency_ms = latency
                    return result
                
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    latency_ms=latency
                )
            except Exception as e:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                    latency_ms=(time.time() - start) * 1000
                )
        
        self.checks[name] = wrapper
    
    def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific check."""
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check '{name}' not found"
            )
        
        return self.checks[name]()
    
    def run_all(self) -> Dict[str, Any]:
        """
        Run all health checks.
        
        Returns comprehensive health status.
        """
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for name, check_fn in self.checks.items():
            result = check_fn()
            results[name] = {
                "status": result.status.value,
                "message": result.message,
                "latency_ms": result.latency_ms
            }
            
            if result.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        
        return {
            "status": overall_status.value,
            "uptime_seconds": time.time() - self._start_time,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": results
        }
    
    def is_ready(self) -> bool:
        """Quick check if service is ready."""
        for check_fn in self.checks.values():
            result = check_fn()
            if result.status == HealthStatus.UNHEALTHY:
                return False
        return True
    
    def is_alive(self) -> bool:
        """Quick liveness check."""
        return True  # Service is running if we can execute this


# ============================================================
# GRACEFUL SHUTDOWN
# ============================================================

class GracefulShutdown:
    """
    Handles graceful shutdown for ML services.
    
    Features:
        - Catches SIGTERM and SIGINT
        - Waits for in-flight requests
        - Runs cleanup callbacks
    
    Example:
        >>> shutdown = GracefulShutdown()
        >>> shutdown.add_cleanup(lambda: model.unload())
        >>> shutdown.register_signals()
    """
    
    def __init__(self, timeout: int = 30):
        """
        Args:
            timeout: Maximum seconds to wait for shutdown
        """
        self.timeout = timeout
        self._shutdown_requested = threading.Event()
        self._cleanup_callbacks: List[Callable[[], None]] = []
        self._active_requests = 0
        self._lock = threading.Lock()
    
    def register_signals(self):
        """Register signal handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        logger.info("Registered shutdown signal handlers")
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signal."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self._shutdown_requested.set()
        self._do_shutdown()
    
    def add_cleanup(self, callback: Callable[[], None]):
        """Add a cleanup callback to run on shutdown."""
        self._cleanup_callbacks.append(callback)
    
    def request_started(self):
        """Track that a request has started."""
        with self._lock:
            self._active_requests += 1
    
    def request_finished(self):
        """Track that a request has finished."""
        with self._lock:
            self._active_requests -= 1
    
    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested.is_set()
    
    def _do_shutdown(self):
        """Perform graceful shutdown."""
        logger.info("Starting graceful shutdown...")
        
        # Wait for active requests
        start_time = time.time()
        while self._active_requests > 0:
            if time.time() - start_time > self.timeout:
                logger.warning(f"Timeout waiting for {self._active_requests} requests")
                break
            time.sleep(0.1)
        
        # Run cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")
        
        logger.info("Graceful shutdown complete")
        sys.exit(0)


# ============================================================
# MODEL VERSION MANAGEMENT
# ============================================================

@dataclass
class ModelVersion:
    """Represents a model version."""
    version: str
    path: str
    created_at: datetime
    is_active: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)


class ModelVersionManager:
    """
    Manages multiple model versions for safe deployments.
    
    Supports:
        - Blue-green deployments
        - Canary releases
        - Quick rollbacks
    
    Example:
        >>> manager = ModelVersionManager("./models")
        >>> manager.register("v2.0", model, metrics={"accuracy": 0.95})
        >>> manager.activate("v2.0")  # Blue-green switch
        >>> manager.rollback()  # Quick rollback to previous
    """
    
    def __init__(self, base_path: str):
        """
        Args:
            base_path: Directory to store model versions
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.versions: Dict[str, ModelVersion] = {}
        self._active_version: Optional[str] = None
        self._previous_version: Optional[str] = None
        self._load_registry()
    
    def _registry_path(self) -> Path:
        return self.base_path / "versions.json"
    
    def _load_registry(self):
        """Load version registry from disk."""
        registry_path = self._registry_path()
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                data = json.load(f)
            
            for version, info in data.get('versions', {}).items():
                self.versions[version] = ModelVersion(
                    version=version,
                    path=info['path'],
                    created_at=datetime.fromisoformat(info['created_at']),
                    is_active=info.get('is_active', False),
                    metrics=info.get('metrics', {})
                )
                if info.get('is_active'):
                    self._active_version = version
    
    def _save_registry(self):
        """Save version registry to disk."""
        data = {
            'versions': {
                v.version: {
                    'path': v.path,
                    'created_at': v.created_at.isoformat(),
                    'is_active': v.is_active,
                    'metrics': v.metrics
                }
                for v in self.versions.values()
            }
        }
        
        with open(self._registry_path(), 'w') as f:
            json.dump(data, f, indent=2)
    
    def register(
        self,
        version: str,
        model: Any,
        metrics: Optional[Dict[str, float]] = None,
        format: ModelFormat = ModelFormat.JOBLIB
    ) -> str:
        """
        Register a new model version.
        
        Args:
            version: Version string (e.g., "v2.0")
            model: Model object
            metrics: Model performance metrics
            format: Serialization format
        
        Returns:
            Path to saved model
        """
        version_dir = self.base_path / version
        version_dir.mkdir(exist_ok=True)
        
        model_path = version_dir / f"model.{format.value}"
        
        # Save model
        serializer = ModelSerializer()
        serializer.save(model, str(model_path), format=format)
        
        # Register version
        self.versions[version] = ModelVersion(
            version=version,
            path=str(model_path),
            created_at=datetime.utcnow(),
            metrics=metrics or {}
        )
        
        self._save_registry()
        logger.info(f"Registered model version {version}")
        
        return str(model_path)
    
    def activate(self, version: str) -> bool:
        """
        Activate a model version (blue-green switch).
        
        Args:
            version: Version to activate
        
        Returns:
            True if successful
        """
        if version not in self.versions:
            logger.error(f"Version {version} not found")
            return False
        
        # Deactivate current
        if self._active_version:
            self._previous_version = self._active_version
            self.versions[self._active_version].is_active = False
        
        # Activate new
        self.versions[version].is_active = True
        self._active_version = version
        
        self._save_registry()
        logger.info(f"Activated model version {version}")
        
        return True
    
    def rollback(self) -> bool:
        """
        Rollback to previous version.
        
        Returns:
            True if successful
        """
        if not self._previous_version:
            logger.error("No previous version to rollback to")
            return False
        
        return self.activate(self._previous_version)
    
    def get_active(self) -> Optional[ModelVersion]:
        """Get currently active version."""
        if self._active_version:
            return self.versions.get(self._active_version)
        return None
    
    def load_active(self) -> Optional[Any]:
        """Load the currently active model."""
        active = self.get_active()
        if not active:
            return None
        
        return ModelSerializer.load(active.path)
    
    def list_versions(self) -> List[ModelVersion]:
        """List all registered versions."""
        return list(self.versions.values())


# ============================================================
# CANARY DEPLOYMENT
# ============================================================

class CanaryDeployment:
    """
    Canary deployment controller.
    
    Gradually shifts traffic from old to new model version.
    
    Example:
        >>> canary = CanaryDeployment(old_model, new_model, traffic_percent=10)
        >>> result = canary.predict(features)  # Routes to old or new
        >>> canary.increase_traffic(20)  # Increase to 20%
        >>> canary.promote()  # Full switch to new
    """
    
    def __init__(
        self,
        stable_model: Any,
        canary_model: Any,
        traffic_percent: float = 10.0,
        metrics_collector: Optional[Callable[[str, Any], None]] = None
    ):
        """
        Args:
            stable_model: Current production model
            canary_model: New model to test
            traffic_percent: Percentage of traffic to canary (0-100)
            metrics_collector: Optional callback to collect metrics
        """
        self.stable = stable_model
        self.canary = canary_model
        self.traffic_percent = traffic_percent
        self.metrics_collector = metrics_collector
        self._stable_calls = 0
        self._canary_calls = 0
        import random
        self._random = random
    
    def predict(self, *args, **kwargs) -> tuple[Any, str]:
        """
        Route prediction to stable or canary model.
        
        Returns:
            Tuple of (prediction, model_name)
        """
        if self._random.random() * 100 < self.traffic_percent:
            self._canary_calls += 1
            result = self.canary.predict(*args, **kwargs)
            model_name = "canary"
        else:
            self._stable_calls += 1
            result = self.stable.predict(*args, **kwargs)
            model_name = "stable"
        
        if self.metrics_collector:
            self.metrics_collector(model_name, result)
        
        return result, model_name
    
    def increase_traffic(self, percent: float):
        """Increase canary traffic percentage."""
        self.traffic_percent = min(100.0, max(0.0, percent))
        logger.info(f"Canary traffic set to {self.traffic_percent}%")
    
    def promote(self):
        """Promote canary to stable (100% traffic)."""
        self.stable = self.canary
        self.traffic_percent = 0.0
        logger.info("Canary promoted to stable")
    
    def rollback(self):
        """Rollback canary (0% traffic)."""
        self.traffic_percent = 0.0
        logger.info("Canary rolled back")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deployment statistics."""
        total = self._stable_calls + self._canary_calls
        return {
            "traffic_percent": self.traffic_percent,
            "stable_calls": self._stable_calls,
            "canary_calls": self._canary_calls,
            "actual_canary_percent": (self._canary_calls / total * 100) if total > 0 else 0
        }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Configuration
    'Environment', 'DeploymentConfig',
    # Serialization
    'ModelFormat', 'ModelMetadata', 'ModelSerializer',
    # Health checks
    'HealthStatus', 'HealthCheckResult', 'HealthChecker',
    # Shutdown
    'GracefulShutdown',
    # Version management
    'ModelVersion', 'ModelVersionManager',
    # Canary deployment
    'CanaryDeployment',
]
