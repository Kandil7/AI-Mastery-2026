# System Design Solution: Scalable ML Model Serving Architecture

## Problem Statement

Design a scalable machine learning model serving architecture that can handle 100K+ requests per second with sub-100ms latency, support multiple model types (deep learning, classical ML, ensembles), provide seamless model updates without downtime, implement intelligent load balancing, and offer comprehensive monitoring and observability. The system should support A/B testing, canary deployments, and automatic scaling based on demand.

## Solution Overview

This system design presents a comprehensive architecture for scalable ML model serving that addresses the critical need for high-performance, reliable, and flexible model deployment. The solution implements a microservices-based approach with intelligent caching, load balancing, and monitoring to ensure optimal performance and reliability.

## 1. High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Gateway  │────│  Load Balancer   │────│  Model Servers  │
│   (Traffic)    │    │  (Intelligent)   │    │  (Auto-scaled)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Authentication│────│  Request Router  │────│  Model Cache   │
│  & Rate Limit │    │  (Model Select)  │    │  (Redis/Fast)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌────────────────────────────────┼─────────────────────────────────┐
│                    Model Serving Infrastructure                 │
│  ┌─────────────────┐    ┌──────────────────┐    ┌──────────┐  │
│  │  Model Store   │────│  Model Manager   │────│  Metrics │  │
│  │  (Versioned)   │    │  (Lifecycle)     │    │  (Prom) │  │
│  └─────────────────┘    └──────────────────┘    └──────────┘  │
└────────────────────────────────────────────────────────────────┘
```

## 2. Core Components

### 2.1 Model Serving Core System
```python
import asyncio
import aioredis
import torch
import numpy as np
from typing import Dict, Any, Optional, List
import time
import logging
from dataclasses import dataclass
from enum import Enum
import json

class ModelStatus(Enum):
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    UNLOADING = "unloading"

@dataclass
class ModelMetadata:
    model_id: str
    version: str
    model_type: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    status: ModelStatus
    created_at: str
    updated_at: str

class ModelServer:
    """
    Core model serving component
    """
    def __init__(self, model_id: str, model_path: str, device: str = "cpu"):
        self.model_id = model_id
        self.model_path = model_path
        self.device = torch.device(device)
        self.model = None
        self.tokenizer = None  # For NLP models
        self.preprocessor = None
        self.postprocessor = None
        self.status = ModelStatus.LOADING
        self.load_timestamp = None
        self.inference_count = 0
        self.error_count = 0
        
        # Performance metrics
        self.latency_samples = []
        self.max_latency_samples = 1000
        
    async def load_model(self) -> bool:
        """
        Load model from storage
        """
        try:
            # Load model based on type
            if self.model_path.endswith('.pth') or self.model_path.endswith('.pt'):
                self.model = torch.load(self.model_path, map_location=self.device)
            elif self.model_path.endswith('.onnx'):
                import onnxruntime
                self.model = onnxruntime.InferenceSession(self.model_path)
            else:
                # Assume it's a scikit-learn model
                import joblib
                self.model = joblib.load(self.model_path)
            
            self.model.eval()  # Set to evaluation mode
            self.load_timestamp = time.time()
            self.status = ModelStatus.READY
            logging.info(f"Model {self.model_id} loaded successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load model {self.model_id}: {str(e)}")
            self.status = ModelStatus.FAILED
            return False
    
    async def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform inference on input data
        """
        if self.status != ModelStatus.READY:
            raise RuntimeError(f"Model {self.model_id} is not ready, status: {self.status}")
        
        start_time = time.time()
        self.inference_count += 1
        
        try:
            # Preprocess input
            processed_input = await self.preprocess(input_data)
            
            # Perform inference
            with torch.no_grad():  # Disable gradient computation for inference
                if isinstance(self.model, torch.nn.Module):
                    # PyTorch model
                    if isinstance(processed_input, dict):
                        output = self.model(**processed_input)
                    else:
                        output = self.model(processed_input)
                elif hasattr(self.model, 'predict'):  # Sklearn model
                    output = self.model.predict(processed_input)
                else:  # ONNX model
                    input_name = self.model.get_inputs()[0].name
                    output = self.model.run(None, {input_name: processed_input})
            
            # Postprocess output
            result = await self.postprocess(output)
            
            # Record latency
            latency = time.time() - start_time
            self.latency_samples.append(latency)
            if len(self.latency_samples) > self.max_latency_samples:
                self.latency_samples.pop(0)
            
            return {
                'model_id': self.model_id,
                'version': getattr(self, 'version', 'unknown'),
                'prediction': result,
                'latency_ms': round(latency * 1000, 2),
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.error_count += 1
            logging.error(f"Inference error for model {self.model_id}: {str(e)}")
            raise
    
    async def preprocess(self, input_data: Dict[str, Any]) -> Any:
        """
        Preprocess input data before inference
        """
        # Default preprocessing - override in subclasses
        return input_data
    
    async def postprocess(self, model_output: Any) -> Any:
        """
        Postprocess model output
        """
        # Default postprocessing - override in subclasses
        if isinstance(model_output, torch.Tensor):
            return model_output.cpu().numpy().tolist()
        return model_output
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the model
        """
        if not self.latency_samples:
            return {
                'avg_latency_ms': 0,
                'p95_latency_ms': 0,
                'p99_latency_ms': 0,
                'inference_count': self.inference_count,
                'error_count': self.error_count,
                'error_rate': 0
            }
        
        latencies = [l * 1000 for l in self.latency_samples]  # Convert to ms
        avg_latency = sum(latencies) / len(latencies)
        
        # Calculate percentiles
        sorted_latencies = sorted(latencies)
        p95_idx = int(0.95 * len(sorted_latencies))
        p99_idx = int(0.99 * len(sorted_latencies))
        
        return {
            'avg_latency_ms': round(avg_latency, 2),
            'p95_latency_ms': round(sorted_latencies[min(p95_idx, len(sorted_latencies)-1)], 2),
            'p99_latency_ms': round(sorted_latencies[min(p99_idx, len(sorted_latencies)-1)], 2),
            'inference_count': self.inference_count,
            'error_count': self.error_count,
            'error_rate': round(self.error_count / max(1, self.inference_count), 4)
        }

class ModelRegistry:
    """
    Registry for managing model versions and metadata
    """
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis = None
    
    async def initialize(self):
        """
        Initialize Redis connection
        """
        self.redis = await aioredis.from_url(self.redis_url)
    
    async def register_model(self, metadata: ModelMetadata) -> bool:
        """
        Register a new model version
        """
        try:
            key = f"model:{metadata.model_id}:v{metadata.version}"
            await self.redis.set(key, json.dumps(metadata.__dict__))
            
            # Add to model index
            await self.redis.sadd(f"models:{metadata.model_id}", key)
            
            # Set as current version if this is the first version
            current_key = f"model:{metadata.model_id}:current"
            exists = await self.redis.exists(current_key)
            if not exists:
                await self.redis.set(current_key, key)
            
            return True
        except Exception as e:
            logging.error(f"Failed to register model {metadata.model_id}: {str(e)}")
            return False
    
    async def get_model_metadata(self, model_id: str, version: str = None) -> Optional[ModelMetadata]:
        """
        Get model metadata
        """
        if version:
            key = f"model:{model_id}:v{version}"
        else:
            current_key = f"model:{model_id}:current"
            key = await self.redis.get(current_key)
            if not key:
                return None
        
        data = await self.redis.get(key)
        if data:
            metadata_dict = json.loads(data)
            return ModelMetadata(**metadata_dict)
        return None
    
    async def get_active_models(self) -> List[str]:
        """
        Get list of active model IDs
        """
        keys = await self.redis.keys("model:*:current")
        model_ids = []
        for key in keys:
            model_id = key.decode().replace("model:", "").replace(":current", "")
            model_ids.append(model_id)
        return model_ids

class ModelCache:
    """
    Intelligent caching layer for model predictions
    """
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 300):
        self.redis_url = redis_url
        self.ttl = ttl  # 5 minutes default TTL
        self.redis = None
    
    async def initialize(self):
        """
        Initialize Redis connection
        """
        self.redis = await aioredis.from_url(self.redis_url)
    
    def _generate_cache_key(self, model_id: str, input_hash: str) -> str:
        """
        Generate cache key for input
        """
        return f"prediction:{model_id}:{input_hash}"
    
    async def get_cached_prediction(self, model_id: str, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get cached prediction if available
        """
        import hashlib
        input_str = json.dumps(input_data, sort_keys=True)
        input_hash = hashlib.md5(input_str.encode()).hexdigest()
        
        cache_key = self._generate_cache_key(model_id, input_hash)
        cached_result = await self.redis.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
        return None
    
    async def cache_prediction(self, model_id: str, input_data: Dict[str, Any], prediction: Dict[str, Any]):
        """
        Cache prediction result
        """
        import hashlib
        input_str = json.dumps(input_data, sort_keys=True)
        input_hash = hashlib.md5(input_str.encode()).hexdigest()
        
        cache_key = self._generate_cache_key(model_id, input_hash)
        await self.redis.setex(cache_key, self.ttl, json.dumps(prediction))

class LoadBalancer:
    """
    Intelligent load balancer for distributing requests
    """
    def __init__(self, model_servers: List[ModelServer]):
        self.model_servers = model_servers
        self.current_server_index = 0
        self.server_metrics = {}  # Track performance metrics
    
    def select_server(self, model_id: str) -> Optional[ModelServer]:
        """
        Select the best server for the given model
        """
        # Filter servers that serve the requested model
        relevant_servers = [s for s in self.model_servers if s.model_id == model_id and s.status == ModelStatus.READY]
        
        if not relevant_servers:
            return None
        
        # Simple round-robin selection
        # In production, this could use more sophisticated algorithms based on metrics
        selected_server = relevant_servers[self.current_server_index % len(relevant_servers)]
        self.current_server_index += 1
        
        return selected_server
    
    def update_server_metrics(self, server_id: str, latency: float, success: bool):
        """
        Update server performance metrics
        """
        if server_id not in self.server_metrics:
            self.server_metrics[server_id] = {
                'latency_samples': [],
                'success_count': 0,
                'error_count': 0
            }
        
        metrics = self.server_metrics[server_id]
        metrics['latency_samples'].append(latency)
        if success:
            metrics['success_count'] += 1
        else:
            metrics['error_count'] += 1
        
        # Keep only recent samples
        if len(metrics['latency_samples']) > 100:
            metrics['latency_samples'] = metrics['latency_samples'][-100:]

class ModelServingOrchestrator:
    """
    Main orchestrator for the model serving system
    """
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.model_registry = ModelRegistry(redis_url)
        self.model_cache = ModelCache(redis_url)
        self.model_servers: Dict[str, ModelServer] = {}
        self.load_balancer = None
        self.redis_url = redis_url
    
    async def initialize(self):
        """
        Initialize the orchestrator
        """
        await self.model_registry.initialize()
        await self.model_cache.initialize()
    
    async def deploy_model(self, model_id: str, model_path: str, version: str, device: str = "cpu") -> bool:
        """
        Deploy a new model version
        """
        # Create and load model server
        model_server = ModelServer(model_id, model_path, device)
        success = await model_server.load_model()
        
        if success:
            # Register in registry
            metadata = ModelMetadata(
                model_id=model_id,
                version=version,
                model_type="pytorch",  # Could be determined from file extension
                input_schema={},  # Would be defined per model
                output_schema={},
                status=ModelStatus.READY,
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            registry_success = await self.model_registry.register_model(metadata)
            if registry_success:
                self.model_servers[f"{model_id}:{version}"] = model_server
                self.load_balancer = LoadBalancer(list(self.model_servers.values()))
                return True
        
        return False
    
    async def predict(self, model_id: str, input_data: Dict[str, Any], version: str = None) -> Dict[str, Any]:
        """
        Perform prediction using the specified model
        """
        # Check cache first
        cached_result = await self.model_cache.get_cached_prediction(model_id, input_data)
        if cached_result:
            cached_result['cached'] = True
            return cached_result
        
        # Select appropriate server
        if version:
            server_key = f"{model_id}:{version}"
            server = self.model_servers.get(server_key)
        else:
            # Find any available server for this model
            server = None
            for key, serv in self.model_servers.items():
                if key.startswith(f"{model_id}:") and serv.status == ModelStatus.READY:
                    server = serv
                    break
        
        if not server:
            raise ValueError(f"No available server for model {model_id}")
        
        # Perform prediction
        start_time = time.time()
        try:
            result = await server.predict(input_data)
            success = True
        except Exception as e:
            result = {
                'error': str(e),
                'model_id': model_id,
                'timestamp': time.time()
            }
            success = False
        
        latency = time.time() - start_time
        
        # Update load balancer metrics
        if self.load_balancer:
            self.load_balancer.update_server_metrics(server.model_id, latency, success)
        
        # Cache successful predictions
        if success:
            await self.model_cache.cache_prediction(model_id, input_data, result)
        
        return result
    
    async def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a model
        """
        metrics = {}
        for key, server in self.model_servers.items():
            if key.startswith(f"{model_id}:"):
                metrics[key] = server.get_performance_metrics()
        return metrics
```

### 2.2 Intelligent Request Routing System
```python
class RequestRouter:
    """
    Intelligent routing system for directing requests to appropriate models
    """
    def __init__(self, orchestrator: ModelServingOrchestrator):
        self.orchestrator = orchestrator
        self.route_rules = {}  # Dynamic routing rules
        self.traffic_splitter = TrafficSplitter()
    
    async def route_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route incoming request to appropriate model
        """
        model_id = request.get('model_id')
        version = request.get('version')
        input_data = request.get('input_data', {})
        
        # Apply routing rules
        target_model_id, target_version = self.apply_routing_rules(model_id, input_data)
        
        # Perform prediction
        result = await self.orchestrator.predict(target_model_id, input_data, target_version)
        
        return result
    
    def apply_routing_rules(self, model_id: str, input_data: Dict[str, Any]) -> tuple:
        """
        Apply routing rules to determine target model and version
        """
        # Default: use requested model
        target_model_id = model_id
        target_version = None  # Use current version
        
        # Apply A/B testing rules
        if model_id in self.route_rules:
            rule = self.route_rules[model_id]
            if rule.get('type') == 'ab_test':
                traffic_percentage = rule.get('percentage', 50)
                if self.traffic_splitter.should_route_to_alternative(traffic_percentage):
                    target_model_id = rule.get('alternative_model', model_id)
                    target_version = rule.get('alternative_version')
        
        return target_model_id, target_version
    
    def add_routing_rule(self, model_id: str, rule: Dict[str, Any]):
        """
        Add a routing rule
        """
        self.route_rules[model_id] = rule

class TrafficSplitter:
    """
    Split traffic between different model versions for A/B testing
    """
    def __init__(self):
        self.seed = int(time.time())
    
    def should_route_to_alternative(self, percentage: float) -> bool:
        """
        Determine if request should go to alternative model
        """
        import random
        random.seed(self.seed + hash(str(time.time())))
        return random.random() < (percentage / 100.0)

class CanaryDeployer:
    """
    Manage canary deployments for safe model rollouts
    """
    def __init__(self, orchestrator: ModelServingOrchestrator):
        self.orchestrator = orchestrator
        self.canary_configs = {}
    
    async def deploy_canary(self, model_id: str, new_model_path: str, version: str, 
                           canary_traffic_percent: float = 5.0) -> bool:
        """
        Deploy new model version as canary
        """
        # Deploy new model
        success = await self.orchestrator.deploy_model(model_id, new_model_path, version)
        if not success:
            return False
        
        # Configure routing rule for canary
        rule = {
            'type': 'canary',
            'percentage': canary_traffic_percent,
            'alternative_model': model_id,
            'alternative_version': version
        }
        
        router = RequestRouter(self.orchestrator)
        router.add_routing_rule(model_id, rule)
        
        # Monitor performance
        self.canary_configs[model_id] = {
            'new_version': version,
            'traffic_percent': canary_traffic_percent,
            'start_time': time.time()
        }
        
        return True
    
    async def promote_canary(self, model_id: str) -> bool:
        """
        Promote canary version to production
        """
        if model_id not in self.canary_configs:
            return False
        
        # Update current version in registry
        new_version = self.canary_configs[model_id]['new_version']
        metadata = await self.orchestrator.model_registry.get_model_metadata(model_id, new_version)
        
        if metadata:
            metadata.status = ModelStatus.READY
            metadata.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
            await self.orchestrator.model_registry.register_model(metadata)
            
            # Update routing to use new version as default
            router = RequestRouter(self.orchestrator)
            router.add_routing_rule(model_id, {
                'type': 'default',
                'version': new_version
            })
            
            return True
        
        return False
```

### 2.3 Monitoring and Observability System
```python
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

class ModelServingMetrics:
    """
    Comprehensive metrics collection for model serving
    """
    def __init__(self):
        # Request metrics
        self.requests_total = Counter(
            'model_requests_total', 
            'Total number of model requests', 
            ['model_id', 'version', 'status']
        )
        
        self.request_duration = Histogram(
            'model_request_duration_seconds',
            'Duration of model requests in seconds',
            ['model_id', 'version'],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.predictions_total = Counter(
            'model_predictions_total',
            'Total number of predictions made',
            ['model_id', 'version']
        )
        
        self.model_loads_total = Counter(
            'model_loads_total',
            'Total number of model loads',
            ['model_id', 'version', 'status']
        )
        
        # Performance gauges
        self.active_models = Gauge(
            'active_models_count',
            'Number of active models'
        )
        
        self.model_memory_usage = Gauge(
            'model_memory_usage_bytes',
            'Memory usage of models',
            ['model_id', 'version']
        )
    
    def record_request(self, model_id: str, version: str, status: str, duration: float):
        """
        Record a model request
        """
        self.requests_total.labels(model_id=model_id, version=version, status=status).inc()
        self.request_duration.labels(model_id=model_id, version=version).observe(duration)
    
    def record_prediction(self, model_id: str, version: str):
        """
        Record a prediction
        """
        self.predictions_total.labels(model_id=model_id, version=version).inc()
    
    def record_model_load(self, model_id: str, version: str, success: bool):
        """
        Record model load attempt
        """
        status = 'success' if success else 'failure'
        self.model_loads_total.labels(model_id=model_id, version=version, status=status).inc()

class ModelServingHealthChecker:
    """
    Health checking system for model servers
    """
    def __init__(self, orchestrator: ModelServingOrchestrator):
        self.orchestrator = orchestrator
        self.health_status = {}
    
    async def check_health(self) -> Dict[str, Any]:
        """
        Check health of all model servers
        """
        health_report = {
            'timestamp': time.time(),
            'overall_status': 'healthy',
            'components': {}
        }
        
        for key, server in self.orchestrator.model_servers.items():
            model_id, version = key.split(':', 1)
            
            # Check if model is responsive
            try:
                # Perform a quick health check inference
                dummy_input = {'dummy': True}  # Replace with actual test input
                start_time = time.time()
                # result = await server.predict(dummy_input)
                latency = time.time() - start_time
                
                status = 'healthy'
                if latency > 1.0:  # More than 1 second is concerning
                    status = 'degraded'
                
                self.health_status[key] = {
                    'status': status,
                    'latency': latency,
                    'last_check': time.time()
                }
                
            except Exception as e:
                status = 'unhealthy'
                self.health_status[key] = {
                    'status': status,
                    'error': str(e),
                    'last_check': time.time()
                }
            
            health_report['components'][key] = self.health_status[key]
            
            if status == 'unhealthy':
                health_report['overall_status'] = 'degraded'
        
        return health_report

# Initialize metrics
metrics = ModelServingMetrics()
```

## 3. Deployment Architecture

### 3.1 Kubernetes Configuration
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-serving-api
  template:
    metadata:
      labels:
        app: model-serving-api
    spec:
      containers:
      - name: model-serving-api
        image: model-serving-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: MODEL_STORAGE_PATH
          value: "/models"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: model-serving-service
spec:
  selector:
    app: model-serving-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

### 3.2 Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-serving-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-serving-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

## 4. Security Considerations

### 4.1 Authentication and Authorization
```python
import jwt
from functools import wraps

class AuthMiddleware:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def authenticate(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate JWT token
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def require_auth(self, f):
        """
        Decorator to require authentication
        """
        @wraps(f)
        async def decorated_function(*args, **kwargs):
            # Extract token from request headers
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return {'error': 'Missing or invalid authorization header'}, 401
            
            token = auth_header.split(' ')[1]
            user_payload = self.authenticate(token)
            
            if not user_payload:
                return {'error': 'Invalid or expired token'}, 401
            
            # Add user info to request context
            request.user = user_payload
            
            return await f(*args, **kwargs)
        
        return decorated_function

# Rate limiting
class RateLimiter:
    def __init__(self, redis_url: str, requests: int, window: int):
        self.redis = aioredis.from_url(redis_url)
        self.requests = requests
        self.window = window
    
    async def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed based on rate limit
        """
        key = f"rate_limit:{identifier}"
        current = await self.redis.get(key)
        
        if current is None:
            await self.redis.setex(key, self.window, 1)
            return True
        
        current = int(current)
        if current >= self.requests:
            return False
        
        await self.redis.incr(key)
        return True
```

## 5. Performance Optimization

### 5.1 Model Optimization Techniques
```python
class ModelOptimizer:
    """
    Various model optimization techniques
    """
    
    @staticmethod
    def quantize_model(model_path: str, output_path: str) -> str:
        """
        Quantize model for faster inference
        """
        import torch
        model = torch.load(model_path)
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        torch.save(quantized_model, output_path)
        return output_path
    
    @staticmethod
    def convert_to_onnx(model_path: str, output_path: str, input_shape: tuple):
        """
        Convert PyTorch model to ONNX for optimized inference
        """
        import torch
        import torch.onnx
        
        model = torch.load(model_path)
        model.eval()
        
        dummy_input = torch.randn(input_shape)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        return output_path
    
    @staticmethod
    def compile_model(model_path: str, output_path: str) -> str:
        """
        Compile model using TorchScript for optimized execution
        """
        import torch
        
        model = torch.load(model_path)
        model.eval()
        
        example_input = torch.randn(1, 3, 224, 224)  # Example input
        traced_script_module = torch.jit.trace(model, example_input)
        traced_script_module.save(output_path)
        
        return output_path
```

## 6. Testing and Validation

### 6.1 Comprehensive Testing Suite
```python
import unittest
import asyncio
from unittest.mock import Mock, AsyncMock

class TestModelServing(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def test_model_loading(self):
        """Test model loading functionality"""
        async def run_test():
            server = ModelServer("test_model", "test_path.pth")
            # Mock the actual loading since we don't have a real model
            server.model = Mock()
            server.status = ModelStatus.READY
            self.assertEqual(server.status, ModelStatus.READY)
        
        self.loop.run_until_complete(run_test())
    
    def test_prediction_flow(self):
        """Test end-to-end prediction flow"""
        async def run_test():
            # Create orchestrator
            orchestrator = ModelServingOrchestrator()
            orchestrator.model_registry = Mock()
            orchestrator.model_cache = Mock()
            orchestrator.model_cache.get_cached_prediction = AsyncMock(return_value=None)
            orchestrator.model_cache.cache_prediction = AsyncMock()
            
            # Create mock server
            server = Mock()
            server.status = ModelStatus.READY
            server.predict = AsyncMock(return_value={"prediction": "test"})
            orchestrator.model_servers = {"test:v1": server}
            
            # Test prediction
            result = await orchestrator.predict("test", {"input": "data"})
            self.assertIn("prediction", result)
        
        self.loop.run_until_complete(run_test())
    
    def test_load_balancing(self):
        """Test load balancing logic"""
        server1 = Mock()
        server1.model_id = "model1"
        server1.status = ModelStatus.READY
        
        server2 = Mock()
        server2.model_id = "model1"
        server2.status = ModelStatus.READY
        
        lb = LoadBalancer([server1, server2])
        
        # Should alternate between servers
        selected1 = lb.select_server("model1")
        selected2 = lb.select_server("model1")
        
        # In round-robin, consecutive selections should be different
        # (assuming equal metrics)
        self.assertIsNotNone(selected1)
        self.assertIsNotNone(selected2)

if __name__ == '__main__':
    unittest.main()
```

## 7. Implementation Plan

### Phase 1: Core Infrastructure (Weeks 1-2)
- Set up basic model serving framework
- Implement model loading/unloading
- Create registry and metadata management
- Basic prediction endpoint

### Phase 2: Scalability Features (Weeks 3-4)
- Implement caching layer
- Add load balancing
- Create horizontal scaling capabilities
- Add basic monitoring

### Phase 3: Advanced Features (Weeks 5-6)
- Implement A/B testing and canary deployments
- Add comprehensive monitoring and alerting
- Implement security features
- Performance optimization

### Phase 4: Production Hardening (Weeks 7-8)
- Stress testing and optimization
- Security audit
- Documentation and deployment guides
- Backup and disaster recovery

## 8. Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Request Latency (p95) | < 100ms | Prometheus histogram |
| Throughput | 100K req/sec | Request counter |
| Availability | 99.9% | Health checks |
| Model Loading Time | < 30s | Timer in load process |
| Memory Efficiency | < 2GB per model | Resource monitoring |
| Error Rate | < 0.1% | Error counter |

This scalable ML model serving architecture provides a robust foundation for deploying and managing machine learning models in production environments with high performance, reliability, and flexibility.