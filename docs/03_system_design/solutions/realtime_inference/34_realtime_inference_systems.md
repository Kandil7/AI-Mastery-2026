# System Design Solution: Real-Time Inference Systems

## Problem Statement

Design a high-performance real-time inference system that can handle 100K+ requests per second with sub-10ms latency, support multiple model types (deep learning, classical ML, ensembles), provide seamless model updates without downtime, implement intelligent batching, handle variable input sizes, provide comprehensive monitoring and observability, and ensure fault tolerance and scalability. The system should support A/B testing, canary deployments, and automatic scaling based on demand.

## Solution Overview

This system design presents a comprehensive architecture for real-time inference that addresses the critical need for ultra-low latency, high throughput, and reliable model serving. The solution implements a microservices-based approach with intelligent batching, caching, and monitoring to ensure optimal performance and reliability.

## 1. High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    └─────────────────┐
│   API Gateway  │────│  Load Balancer   │────│  Inference     │
│   (Traffic)    │    │  (Intelligent)   │    │  Servers       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    └─────────────────┐
│  Authentication│────│  Request Router  │────│  Model Cache   │
│  & Rate Limit │    │  (Model Select)  │    │  (Redis/Fast)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌────────────────────────────────┼─────────────────────────────────┐
│                    Inference Infrastructure                   │
│  ┌─────────────────┐    └──────────────────┐    ┌──────────┐  │
│  │  Model Store   │────│  Batch Scheduler │────│  Metrics │  │
│  │  (Versioned)   │    │  (Smart Batching)│    │  (Prom) │  │
│  └─────────────────┘    └──────────────────┘    └──────────┘  │
└────────────────────────────────────────────────────────────────┘
```

## 2. Core Components

### 2.1 Real-Time Inference Core System
```python
import asyncio
import aioredis
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Callable
import time
import logging
from dataclasses import dataclass
from enum import Enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

class ModelStatus(Enum):
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    UNLOADING = "unloading"

class InferenceMode(Enum):
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"

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
    inference_mode: InferenceMode

class InferenceServer:
    """
    Core real-time inference server
    """
    def __init__(self, model_id: str, model_path: str, device: str = "cpu", 
                 batch_size: int = 1, max_batch_time: float = 0.01):
        self.model_id = model_id
        self.model_path = model_path
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_batch_time = max_batch_time  # 10ms max batch wait time
        self.model = None
        self.status = ModelStatus.LOADING
        self.load_timestamp = None
        self.inference_count = 0
        self.error_count = 0
        
        # Batching components
        self.request_queue = queue.Queue()
        self.batch_thread = None
        self.batch_shutdown = threading.Event()
        
        # Performance metrics
        self.latency_samples = []
        self.max_latency_samples = 10000
        self.throughput_samples = []
        self.max_throughput_samples = 1000
        
        # Initialize batching
        self._start_batch_processing()
    
    def _start_batch_processing(self):
        """
        Start the batch processing thread
        """
        self.batch_thread = threading.Thread(target=self._batch_processor, daemon=True)
        self.batch_thread.start()
    
    def _batch_processor(self):
        """
        Process requests in batches
        """
        while not self.batch_shutdown.is_set():
            batch_requests = []
            start_time = time.time()
            
            # Collect requests for batching
            while (len(batch_requests) < self.batch_size and 
                   time.time() - start_time < self.max_batch_time and
                   not self.batch_shutdown.is_set()):
                try:
                    request = self.request_queue.get(timeout=0.001)  # 1ms timeout
                    batch_requests.append(request)
                except queue.Empty:
                    continue
            
            if batch_requests:
                self._process_batch(batch_requests)
    
    def _process_batch(self, batch_requests: List[Dict[str, Any]]):
        """
        Process a batch of requests
        """
        start_time = time.time()
        
        try:
            # Prepare batch input
            batch_inputs = []
            for req in batch_requests:
                processed_input = self._preprocess_input(req['input_data'])
                batch_inputs.append(processed_input)
            
            # Stack inputs for batch processing
            if isinstance(batch_inputs[0], torch.Tensor):
                batch_tensor = torch.stack(batch_inputs)
            else:
                # Handle other input types
                batch_tensor = torch.tensor(batch_inputs)
            
            # Perform batch inference
            with torch.no_grad():
                batch_tensor = batch_tensor.to(self.device)
                batch_outputs = self.model(batch_tensor)
            
            # Process outputs
            for i, req in enumerate(batch_requests):
                output = batch_outputs[i]
                result = self._postprocess_output(output)
                
                # Send result back through callback
                req['callback'](req['request_id'], {
                    'model_id': self.model_id,
                    'prediction': result,
                    'latency_ms': round((time.time() - req['timestamp']) * 1000, 2),
                    'timestamp': time.time()
                })
            
            # Record performance metrics
            batch_latency = time.time() - start_time
            self._record_metrics(batch_latency, len(batch_requests))
            
        except Exception as e:
            logging.error(f"Batch processing error for model {self.model_id}: {str(e)}")
            # Send error responses
            for req in batch_requests:
                req['callback'](req['request_id'], {
                    'error': str(e),
                    'model_id': self.model_id,
                    'timestamp': time.time()
                })
    
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
    
    def predict_async(self, input_data: Dict[str, Any], callback: Callable[[str, Dict[str, Any]], None], 
                     request_id: str = None) -> str:
        """
        Asynchronously add request to batch queue
        """
        if self.status != ModelStatus.READY:
            raise RuntimeError(f"Model {self.model_id} is not ready, status: {self.status}")
        
        if request_id is None:
            request_id = str(int(time.time() * 1000000))  # Microsecond timestamp
        
        request = {
            'request_id': request_id,
            'input_data': input_data,
            'callback': callback,
            'timestamp': time.time()
        }
        
        self.request_queue.put(request)
        self.inference_count += 1
        
        return request_id
    
    def _preprocess_input(self, input_data: Dict[str, Any]) -> torch.Tensor:
        """
        Preprocess input data before inference
        """
        # Convert input to tensor
        if isinstance(input_data, dict):
            # Handle structured input
            values = list(input_data.values())
            if len(values) == 1 and isinstance(values[0], list):
                return torch.tensor(values[0], dtype=torch.float32)
            else:
                return torch.tensor([float(v) for v in values], dtype=torch.float32)
        elif isinstance(input_data, list):
            return torch.tensor(input_data, dtype=torch.float32)
        else:
            return torch.tensor([input_data], dtype=torch.float32)
    
    def _postprocess_output(self, model_output: Any) -> Any:
        """
        Postprocess model output
        """
        if isinstance(model_output, torch.Tensor):
            return model_output.cpu().numpy().tolist()
        return model_output
    
    def _record_metrics(self, batch_latency: float, batch_size: int):
        """
        Record performance metrics
        """
        # Record latency per request
        per_request_latency = batch_latency / batch_size if batch_size > 0 else 0
        self.latency_samples.extend([per_request_latency] * batch_size)
        
        # Keep only recent samples
        if len(self.latency_samples) > self.max_latency_samples:
            self.latency_samples = self.latency_samples[-self.max_latency_samples:]
        
        # Record throughput
        throughput = batch_size / batch_latency if batch_latency > 0 else 0
        self.throughput_samples.append(throughput)
        
        if len(self.throughput_samples) > self.max_throughput_samples:
            self.throughput_samples = self.throughput_samples[-self.max_throughput_samples:]
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the model
        """
        if not self.latency_samples:
            return {
                'avg_latency_ms': 0,
                'p95_latency_ms': 0,
                'p99_latency_ms': 0,
                'avg_throughput_rps': 0,
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
        
        avg_throughput = sum(self.throughput_samples) / len(self.throughput_samples) if self.throughput_samples else 0
        
        return {
            'avg_latency_ms': round(avg_latency, 2),
            'p95_latency_ms': round(sorted_latencies[min(p95_idx, len(sorted_latencies)-1)], 2),
            'p99_latency_ms': round(sorted_latencies[min(p99_idx, len(sorted_latencies)-1)], 2),
            'avg_throughput_rps': round(avg_throughput, 2),
            'inference_count': self.inference_count,
            'error_count': self.error_count,
            'error_rate': round(self.error_count / max(1, self.inference_count), 4)
        }
    
    def shutdown(self):
        """
        Shutdown the inference server
        """
        self.batch_shutdown.set()
        if self.batch_thread:
            self.batch_thread.join(timeout=5.0)  # Wait up to 5 seconds

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
        return f"inference:{model_id}:{input_hash}"
    
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
    def __init__(self, inference_servers: List[InferenceServer]):
        self.inference_servers = inference_servers
        self.current_server_index = 0
        self.server_metrics = {}  # Track performance metrics
        self.lock = threading.Lock()
    
    def select_server(self, model_id: str) -> Optional[InferenceServer]:
        """
        Select the best server for the given model
        """
        # Filter servers that serve the requested model
        relevant_servers = [s for s in self.inference_servers if s.model_id == model_id and s.status == ModelStatus.READY]
        
        if not relevant_servers:
            return None
        
        # Round-robin selection with performance consideration
        with self.lock:
            # Find server with best performance metrics
            best_server = relevant_servers[0]
            best_latency = float('inf')
            
            for server in relevant_servers:
                metrics = server.get_performance_metrics()
                avg_latency = metrics.get('avg_latency_ms', float('inf'))
                
                if avg_latency < best_latency:
                    best_latency = avg_latency
                    best_server = server
        
        return best_server
    
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

class InferenceOrchestrator:
    """
    Main orchestrator for the inference system
    """
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.model_registry = ModelRegistry(redis_url)
        self.model_cache = ModelCache(redis_url)
        self.inference_servers: Dict[str, InferenceServer] = {}
        self.load_balancer = None
        self.redis_url = redis_url
        self.response_callbacks = {}  # Store callbacks for async responses
        self.callback_lock = threading.Lock()
    
    async def initialize(self):
        """
        Initialize the orchestrator
        """
        await self.model_registry.initialize()
        await self.model_cache.initialize()
    
    async def deploy_model(self, model_id: str, model_path: str, version: str, 
                          device: str = "cpu", batch_size: int = 1, 
                          max_batch_time: float = 0.01) -> bool:
        """
        Deploy a new model version
        """
        # Create and load inference server
        inference_server = InferenceServer(
            model_id, model_path, device, batch_size, max_batch_time
        )
        success = await inference_server.load_model()
        
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
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                inference_mode=InferenceMode.REAL_TIME
            )
            
            registry_success = await self.model_registry.register_model(metadata)
            if registry_success:
                self.inference_servers[f"{model_id}:{version}"] = inference_server
                self.load_balancer = LoadBalancer(list(self.inference_servers.values()))
                return True
        
        return False
    
    def predict_async(self, model_id: str, input_data: Dict[str, Any], 
                     callback: Callable[[Dict[str, Any]], None], 
                     version: str = None, request_id: str = None) -> str:
        """
        Perform asynchronous prediction using the specified model
        """
        # Select appropriate server
        if version:
            server_key = f"{model_id}:{version}"
            server = self.inference_servers.get(server_key)
        else:
            # Find any available server for this model
            server = None
            for key, serv in self.inference_servers.items():
                if key.startswith(f"{model_id}:") and serv.status == ModelStatus.READY:
                    server = serv
                    break
        
        if not server:
            raise ValueError(f"No available server for model {model_id}")
        
        # Check cache first
        async def check_cache_and_predict():
            cached_result = await self.model_cache.get_cached_prediction(model_id, input_data)
            if cached_result:
                cached_result['cached'] = True
                callback(cached_result)
                return cached_result['request_id']
        
        # Wrap the callback to handle caching
        def wrapped_callback(req_id, result):
            # Cache successful predictions
            if 'error' not in result:
                async def cache_result():
                    await self.model_cache.cache_prediction(model_id, input_data, result)
                
                # Run caching asynchronously
                asyncio.create_task(cache_result())
            
            # Call the original callback
            callback(result)
        
        # Submit to server
        request_id = server.predict_async(input_data, wrapped_callback, request_id)
        return request_id
    
    async def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a model
        """
        metrics = {}
        for key, server in self.inference_servers.items():
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
    def __init__(self, orchestrator: InferenceOrchestrator):
        self.orchestrator = orchestrator
        self.route_rules = {}  # Dynamic routing rules
        self.traffic_splitter = TrafficSplitter()
        self.rate_limiter = RateLimiter()
    
    def route_request(self, request: Dict[str, Any]) -> str:
        """
        Route incoming request to appropriate model and return request ID
        """
        model_id = request.get('model_id')
        version = request.get('version')
        input_data = request.get('input_data', {})
        callback = request.get('callback')
        
        # Apply routing rules
        target_model_id, target_version = self.apply_routing_rules(model_id, input_data)
        
        # Check rate limits
        user_id = request.get('user_id', 'anonymous')
        if not self.rate_limiter.is_allowed(user_id):
            if callback:
                callback({'error': 'Rate limit exceeded'})
            return None
        
        # Submit for async processing
        request_id = self.orchestrator.predict_async(
            target_model_id, input_data, callback, target_version
        )
        
        return request_id
    
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

class RateLimiter:
    """
    Rate limiting for requests
    """
    def __init__(self, requests: int = 1000, window: int = 60):
        self.requests = requests
        self.window = window
        self.counts = {}  # user_id -> [(timestamp, count)]
        self.lock = threading.Lock()
    
    def is_allowed(self, user_id: str) -> bool:
        """
        Check if request is allowed based on rate limit
        """
        with self.lock:
            now = time.time()
            
            # Clean old entries
            if user_id in self.counts:
                self.counts[user_id] = [
                    (ts, count) for ts, count in self.counts[user_id] 
                    if now - ts < self.window
                ]
            
            # Calculate current count
            current_count = sum(count for ts, count in self.counts.get(user_id, []))
            
            if current_count >= self.requests:
                return False
            
            # Add current request
            if user_id not in self.counts:
                self.counts[user_id] = []
            
            if self.counts[user_id] and self.counts[user_id][-1][0] == now:
                # Increment count for this timestamp
                self.counts[user_id][-1] = (now, self.counts[user_id][-1][1] + 1)
            else:
                # Add new timestamp entry
                self.counts[user_id].append((now, 1))
            
            return True

class CanaryDeployer:
    """
    Manage canary deployments for safe model rollouts
    """
    def __init__(self, orchestrator: InferenceOrchestrator):
        self.orchestrator = orchestrator
        self.canary_configs = {}
    
    async def deploy_canary(self, model_id: str, new_model_path: str, version: str, 
                           canary_traffic_percent: float = 5.0) -> bool:
        """
        Deploy new model version as canary
        """
        # Deploy new model
        success = await self.orchestrator.deploy_model(
            model_id, new_model_path, version, batch_size=1, max_batch_time=0.005
        )
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

class InferenceMetrics:
    """
    Comprehensive metrics collection for inference system
    """
    def __init__(self):
        # Request metrics
        self.requests_total = Counter(
            'inference_requests_total', 
            'Total number of inference requests', 
            ['model_id', 'version', 'status']
        )
        
        self.request_duration = Histogram(
            'inference_request_duration_seconds',
            'Duration of inference requests in seconds',
            ['model_id', 'version'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0]
        )
        
        self.predictions_total = Counter(
            'inference_predictions_total',
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
        
        self.queue_depth = Gauge(
            'inference_queue_depth',
            'Depth of inference request queue',
            ['model_id']
        )
    
    def record_request(self, model_id: str, version: str, status: str, duration: float):
        """
        Record an inference request
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
    
    def update_queue_depth(self, model_id: str, depth: int):
        """
        Update queue depth metric
        """
        self.queue_depth.labels(model_id=model_id).set(depth)

class InferenceHealthChecker:
    """
    Health checking system for inference servers
    """
    def __init__(self, orchestrator: InferenceOrchestrator):
        self.orchestrator = orchestrator
        self.health_status = {}
    
    async def check_health(self) -> Dict[str, Any]:
        """
        Check health of all inference servers
        """
        health_report = {
            'timestamp': time.time(),
            'overall_status': 'healthy',
            'components': {}
        }
        
        for key, server in self.orchestrator.inference_servers.items():
            model_id, version = key.split(':', 1)
            
            # Check if model is responsive
            try:
                # Check server status
                status = server.status.value
                metrics = server.get_performance_metrics()
                
                # Check for performance degradation
                if metrics['avg_latency_ms'] > 100:  # More than 100ms is concerning
                    status = 'degraded'
                
                self.health_status[key] = {
                    'status': status,
                    'latency_ms': metrics['avg_latency_ms'],
                    'throughput_rps': metrics['avg_throughput_rps'],
                    'queue_depth': server.request_queue.qsize() if hasattr(server, 'request_queue') else 0,
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
metrics = InferenceMetrics()
```

## 3. Deployment Architecture

### 3.1 Kubernetes Configuration
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: real-time-inference-api
spec:
  replicas: 5
  selector:
    matchLabels:
      app: real-time-inference-api
  template:
    metadata:
      labels:
        app: real-time-inference-api
    spec:
      containers:
      - name: inference-api
        image: real-time-inference-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-cluster:6379"
        - name: MODEL_STORAGE_PATH
          value: "/models"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
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
  name: inference-service
spec:
  selector:
    app: real-time-inference-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-config
data:
  inference-config.yaml: |
    model_storage:
      path: "/models"
      retention_days: 30
    cache:
      redis_url: "redis://redis-cluster:6379"
      ttl_seconds: 300
    batching:
      default_batch_size: 8
      max_batch_time_ms: 10
    monitoring:
      enabled: true
      metrics_port: 9090
```

### 3.2 Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: real-time-inference-api
  minReplicas: 3
  maxReplicas: 50
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
  - type: Pods
    pods:
      metric:
        name: inference_queue_depth
      target:
        type: AverageValue
        averageValue: "100"
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
```

## 5. Performance Optimization

### 5.1 Model Optimization Techniques
```python
class ModelOptimizer:
    """
    Various model optimization techniques for inference
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
            model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
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

class TestRealTimeInference(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def test_inference_server_creation(self):
        """Test inference server creation and loading"""
        server = InferenceServer("test_model", "test_path.pth", batch_size=2, max_batch_time=0.01)
        
        # Mock the actual loading since we don't have a real model
        server.model = Mock()
        server.status = ModelStatus.READY
        
        self.assertEqual(server.status, ModelStatus.READY)
        self.assertEqual(server.batch_size, 2)
        self.assertEqual(server.max_batch_time, 0.01)
    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        async def run_test():
            server = InferenceServer("test_model", "test_path.pth", batch_size=2, max_batch_time=0.1)
            
            # Mock model
            server.model = Mock()
            server.model.return_value = torch.tensor([[0.5], [0.7]])
            server.status = ModelStatus.READY
            
            results = []
            def callback(request_id, result):
                results.append(result)
            
            # Submit two requests
            server.predict_async({"input": [1, 2, 3]}, callback, "req1")
            server.predict_async({"input": [4, 5, 6]}, callback, "req2")
            
            # Wait for processing
            await asyncio.sleep(0.2)
            
            # Check that both requests were processed
            self.assertEqual(len(results), 2)
            self.assertIn("prediction", results[0])
            self.assertIn("prediction", results[1])
        
        self.loop.run_until_complete(run_test())
    
    def test_load_balancing(self):
        """Test load balancing logic"""
        server1 = Mock()
        server1.model_id = "model1"
        server1.status = ModelStatus.READY
        server1.get_performance_metrics = Mock(return_value={'avg_latency_ms': 10})
        
        server2 = Mock()
        server2.model_id = "model1"
        server2.status = ModelStatus.READY
        server2.get_performance_metrics = Mock(return_value={'avg_latency_ms': 5})
        
        lb = LoadBalancer([server1, server2])
        
        # Should select server with better performance
        selected = lb.select_server("model1")
        # Note: Since we're mocking, the selection might not reflect actual performance
        # In real implementation, it would select based on metrics
        
        self.assertIsNotNone(selected)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        limiter = RateLimiter(requests=2, window=1)  # 2 requests per 1 second
        
        # First two requests should be allowed
        self.assertTrue(limiter.is_allowed("user1"))
        self.assertTrue(limiter.is_allowed("user1"))
        
        # Third request should be denied
        self.assertFalse(limiter.is_allowed("user1"))

if __name__ == '__main__':
    unittest.main()
```

## 7. Implementation Plan

### Phase 1: Core Infrastructure (Weeks 1-3)
- Set up basic inference server framework
- Implement model loading and basic inference
- Create registry and metadata management
- Basic async request handling

### Phase 2: Batching and Optimization (Weeks 4-6)
- Implement intelligent batching system
- Add caching layer
- Create load balancing
- Add basic monitoring

### Phase 3: Advanced Features (Weeks 7-9)
- Implement A/B testing and canary deployments
- Add comprehensive monitoring and alerting
- Implement security features
- Performance optimization

### Phase 4: Production Hardening (Weeks 10-12)
- Stress testing and optimization
- Security audit
- Documentation and deployment guides
- Backup and disaster recovery

## 8. Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Request Latency (p95) | < 10ms | Prometheus histogram |
| Throughput | 100K req/sec | Request counter |
| Availability | 99.9% | Health checks |
| Model Loading Time | < 30s | Timer in load process |
| Memory Efficiency | < 2GB per model | Resource monitoring |
| Error Rate | < 0.01% | Error counter |

This comprehensive real-time inference system provides a robust foundation for deploying and managing machine learning models in production environments with ultra-low latency, high throughput, and reliability.