# Monitoring

This section provides comprehensive information about monitoring the Production RAG System, including metrics collection, alerting, and observability practices.

## Monitoring Overview

The Production RAG System implements comprehensive monitoring across all layers of the application stack. The monitoring strategy follows the four golden signals of monitoring: latency, traffic, errors, and saturation.

## Monitoring Architecture

### Observability Stack
The system implements a complete observability stack:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Application   │───▶│  Collection     │───▶│  Visualization  │
│   Instrumentation│    │  Layer         │    │  Layer         │
│                 │    │                │    │                 │
│ • Metrics       │    │ • Prometheus   │    │ • Grafana      │
│ • Logs          │    │ • Jaeger       │    │ • Kibana       │
│ • Traces        │    │ • Fluentd      │    │ • Custom Dash. │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Metrics Collection

### System Metrics

#### CPU and Memory
```python
# src/observability/__init__.py - System metrics
import psutil
from prometheus_client import Gauge

# CPU metrics
cpu_usage_gauge = Gauge('cpu_usage_percentage', 'CPU usage percentage')
cpu_load_1min = Gauge('cpu_load_1min', 'CPU load average 1 minute')
cpu_load_5min = Gauge('cpu_load_5min', 'CPU load average 5 minutes')
cpu_load_15min = Gauge('cpu_load_15min', 'CPU load average 15 minutes')

# Memory metrics
memory_usage_gauge = Gauge('memory_usage_bytes', 'Memory usage in bytes')
memory_available_gauge = Gauge('memory_available_bytes', 'Available memory in bytes')
memory_percent_gauge = Gauge('memory_usage_percentage', 'Memory usage percentage')

def collect_system_metrics():
    """Collect system-level metrics."""
    # CPU metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_usage_gauge.set(cpu_percent)
    
    cpu_loads = psutil.getloadavg()
    cpu_load_1min.set(cpu_loads[0])
    cpu_load_5min.set(cpu_loads[1])
    cpu_load_15min.set(cpu_loads[2])
    
    # Memory metrics
    memory_info = psutil.virtual_memory()
    memory_usage_gauge.set(memory_info.used)
    memory_available_gauge.set(memory_info.available)
    memory_percent_gauge.set(memory_info.percent)
```

#### Disk and Network
```python
# Disk metrics
disk_usage_gauge = Gauge('disk_usage_percentage', 'Disk usage percentage', ['mountpoint'])
disk_read_bytes = Gauge('disk_read_bytes_total', 'Total disk read bytes')
disk_write_bytes = Gauge('disk_write_bytes_total', 'Total disk write bytes')

# Network metrics
network_receive_bytes = Gauge('network_receive_bytes_total', 'Total network receive bytes')
network_transmit_bytes = Gauge('network_transmit_bytes_total', 'Total network transmit bytes')

def collect_disk_network_metrics():
    """Collect disk and network metrics."""
    # Disk metrics
    partitions = psutil.disk_partitions()
    for partition in partitions:
        usage = psutil.disk_usage(partition.mountpoint)
        disk_usage_gauge.labels(mountpoint=partition.mountpoint).set(usage.percent)
    
    # Network metrics
    net_io = psutil.net_io_counters()
    network_receive_bytes.set(net_io.bytes_recv)
    network_transmit_bytes.set(net_io.bytes_sent)
```

### Application Metrics

#### Request Metrics
```python
# src/observability/__init__.py - Request metrics
from prometheus_client import Counter, Histogram

# Request counters
request_count = Counter('request_count_total', 'Total number of requests', ['method', 'endpoint', 'status'])
error_count = Counter('error_count_total', 'Total number of errors', ['method', 'endpoint', 'error_type'])

# Request timing
request_duration_histogram = Histogram(
    'request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0]
)

# API-specific metrics
query_count = Counter('query_count_total', 'Total number of queries')
document_index_count = Counter('document_index_count_total', 'Total number of documents indexed')
```

#### Model Performance Metrics
```python
# Model-specific metrics
model_loading_time = Histogram('model_loading_seconds', 'Time to load models')
embedding_generation_time = Histogram('embedding_generation_seconds', 'Time to generate embeddings')
generation_time = Histogram('generation_seconds', 'Time to generate responses')
token_per_second = Histogram('tokens_per_second', 'Tokens generated per second')

def collect_model_metrics():
    """Collect model performance metrics."""
    # Track model loading times
    start_time = time.time()
    model = load_model()
    loading_time = time.time() - start_time
    model_loading_time.observe(loading_time)
    
    # Track embedding generation
    start_time = time.time()
    embeddings = generate_embeddings(text)
    embedding_time = time.time() - start_time
    embedding_generation_time.observe(embedding_time)
```

### Database Metrics

#### MongoDB Metrics
```python
# Database performance metrics
db_connection_count = Gauge('db_connection_count', 'Number of database connections')
db_query_duration = Histogram('db_query_duration_seconds', 'Database query duration')
db_active_operations = Gauge('db_active_operations', 'Active database operations')

def collect_database_metrics():
    """Collect database performance metrics."""
    # Get MongoDB stats
    try:
        # Get connection stats
        db_stats = get_db_stats()
        db_connection_count.set(db_stats['connections']['current'])
        db_active_operations.set(db_stats['connections']['active'])
    except Exception as e:
        error_count.labels(method='db_monitoring', endpoint='stats', error_type='connection_error').inc()
```

#### Vector Database Metrics
```python
# Vector database metrics
vector_search_duration = Histogram('vector_search_duration_seconds', 'Vector search duration')
vector_index_size = Gauge('vector_index_size', 'Size of vector index')
vector_memory_usage = Gauge('vector_memory_usage_bytes', 'Vector database memory usage')

def collect_vector_db_metrics():
    """Collect vector database metrics."""
    # Track vector search performance
    start_time = time.time()
    results = vector_store.search(query_vector, k=5)
    search_time = time.time() - start_time
    vector_search_duration.observe(search_time)
    
    # Track index size
    index_size = vector_store.get_count()
    vector_index_size.set(index_size)
```

## Logging

### Structured Logging
```python
# src/observability/__init__.py - Structured logging
import logging
import json
from datetime import datetime
from typing import Dict, Any

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create formatter for structured logs
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Add handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _log_structured(self, level: int, message: str, **kwargs):
        """Log structured message with additional context."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': logging.getLevelName(level),
            'message': message,
            'service': 'rag-api',
            'context': kwargs
        }
        
        self.logger.log(level, json.dumps(log_entry))
    
    def info(self, message: str, **kwargs):
        """Log info level message."""
        self._log_structured(logging.INFO, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error level message."""
        self._log_structured(logging.ERROR, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning level message."""
        self._log_structured(logging.WARNING, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug level message."""
        self._log_structured(logging.DEBUG, message, **kwargs)

# Global logger instance
structured_logger = StructuredLogger('rag-system')
```

### Log Context Propagation
```python
# Context propagation for distributed tracing
import contextvars

# Context variables for tracing
trace_id_var = contextvars.ContextVar('trace_id', default=None)
span_id_var = contextvars.ContextVar('span_id', default=None)
user_id_var = contextvars.ContextVar('user_id', default=None)

class ContextAwareLogger(StructuredLogger):
    def _log_structured(self, level: int, message: str, **kwargs):
        """Log structured message with context propagation."""
        context = {
            'trace_id': trace_id_var.get(),
            'span_id': span_id_var.get(),
            'user_id': user_id_var.get()
        }
        
        # Merge context with additional kwargs
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': logging.getLevelName(level),
            'message': message,
            'service': 'rag-api',
            'context': {**context, **kwargs}
        }
        
        self.logger.log(level, json.dumps(log_entry))
```

## Distributed Tracing

### Tracing Implementation
```python
# src/observability/__init__.py - Distributed tracing
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import time

# Initialize tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Set up Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

# Add span processor
span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

class DistributedTracer:
    def __init__(self):
        self.tracer = tracer
    
    def trace_operation(self, operation_name: str, **attributes):
        """Trace an operation with attributes."""
        with self.tracer.start_as_current_span(operation_name) as span:
            # Set attributes
            for key, value in attributes.items():
                span.set_attribute(key, value)
            
            # Add timing information
            start_time = time.time()
            yield span
            duration = time.time() - start_time
            span.set_attribute("duration_ms", duration * 1000)
    
    def trace_query(self, query: str, top_k: int):
        """Trace a query operation."""
        with self.tracer.start_as_current_span("query_operation") as span:
            span.set_attribute("query", query)
            span.set_attribute("top_k", top_k)
            
            start_time = time.time()
            result = yield
            duration = time.time() - start_time
            
            span.set_attribute("duration_ms", duration * 1000)
            span.set_attribute("result_count", len(result.get("retrieved_documents", [])))
```

## Monitoring Endpoints

### Metrics Endpoint
The system provides a `/metrics` endpoint in Prometheus format:

```python
# api.py - Metrics endpoint
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

### Health Check Endpoint
```python
# Enhanced health check with metrics
@app.get("/health")
async def health_check():
    """Comprehensive health check with metrics."""
    global rag_model
    
    # Check model status
    model_status = "initialized" if rag_model else "not_initialized"
    
    # Check database connectivity
    db_connected = True  # Implement actual check
    try:
        # Perform a simple database operation
        await mongo_storage.get_document_count()
    except Exception:
        db_connected = False
    
    # Check vector store connectivity
    vector_store_connected = True  # Implement actual check
    try:
        # Perform a simple vector store operation
        await vector_manager.get_count()
    except Exception:
        vector_store_connected = False
    
    # Calculate system metrics
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    
    return {
        "status": "healthy" if (rag_model and db_connected and vector_store_connected) else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "details": {
            "model_status": model_status,
            "database_connected": db_connected,
            "vector_store_connected": vector_store_connected,
            "cpu_usage": cpu_percent,
            "memory_usage": memory_percent,
            "document_count": await mongo_storage.get_document_count() if db_connected else 0,
            "service": "rag-api"
        }
    }
```

## Alerting

### Alert Rules

#### System-Level Alerts
```yaml
# Prometheus alerting rules
groups:
- name: system_alerts
  rules:
  - alert: HighCPUUsage
    expr: cpu_usage_percentage > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage on {{ $labels.instance }}"
      description: "CPU usage is above 80% for more than 5 minutes (current value: {{ $value }}%)"
  
  - alert: HighMemoryUsage
    expr: memory_usage_percentage > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage on {{ $labels.instance }}"
      description: "Memory usage is above 85% for more than 5 minutes (current value: {{ $value }}%)"
  
  - alert: DiskSpaceLow
    expr: disk_usage_percentage > 90
    for: 10m
    labels:
      severity: critical
    annotations:
      summary: "Disk space low on {{ $labels.mountpoint }}"
      description: "Disk usage is above 90% for more than 10 minutes (current value: {{ $value }}%)"
```

#### Application-Level Alerts
```yaml
- name: application_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(error_count_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is above 10% for more than 2 minutes (current rate: {{ $value }})"
  
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, request_duration_seconds_bucket) > 2
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is above 2 seconds (current value: {{ $value }}s)"
  
  - alert: ServiceDown
    expr: up == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Service is down"
      description: "Service {{ $labels.instance }} has been down for more than 1 minute"
```

#### Model Performance Alerts
```yaml
- name: model_alerts
  rules:
  - alert: SlowModelLoading
    expr: histogram_quantile(0.95, model_loading_seconds_bucket) > 30
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Slow model loading detected"
      description: "95th percentile model loading time is above 30 seconds (current value: {{ $value }}s)"
  
  - alert: SlowVectorSearch
    expr: histogram_quantile(0.95, vector_search_duration_seconds_bucket) > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Slow vector search detected"
      description: "95th percentile vector search time is above 1 second (current value: {{ $value }}s)"
```

## Dashboards

### Grafana Dashboard Configuration

#### System Performance Dashboard
```json
{
  "dashboard": {
    "id": null,
    "title": "RAG System - System Performance",
    "tags": ["rag", "system", "performance"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "cpu_usage_percentage",
            "legendFormat": "{{instance}}",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "label": "Percentage",
            "min": 0,
            "max": 100
          }
        ]
      },
      {
        "id": 2,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "memory_usage_percentage",
            "legendFormat": "{{instance}}",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "label": "Percentage",
            "min": 0,
            "max": 100
          }
        ]
      },
      {
        "id": 3,
        "title": "Disk Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "disk_usage_percentage",
            "legendFormat": "{{mountpoint}}",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "label": "Percentage",
            "min": 0,
            "max": 100
          }
        ]
      }
    ]
  }
}
```

#### Application Performance Dashboard
```json
{
  "dashboard": {
    "id": null,
    "title": "RAG System - Application Performance",
    "tags": ["rag", "application", "performance"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(request_count_total[5m]))",
            "legendFormat": "Total Requests/s",
            "refId": "A"
          }
        ]
      },
      {
        "id": 2,
        "title": "Response Time Percentiles",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.5, request_duration_seconds_bucket)",
            "legendFormat": "P50",
            "refId": "A"
          },
          {
            "expr": "histogram_quantile(0.95, request_duration_seconds_bucket)",
            "legendFormat": "P95",
            "refId": "B"
          },
          {
            "expr": "histogram_quantile(0.99, request_duration_seconds_bucket)",
            "legendFormat": "P99",
            "refId": "C"
          }
        ]
      },
      {
        "id": 3,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(error_count_total[5m])) / sum(rate(request_count_total[5m]))",
            "legendFormat": "Error Rate",
            "refId": "A"
          }
        ]
      },
      {
        "id": 4,
        "title": "Query Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, vector_search_duration_seconds_bucket)",
            "legendFormat": "Vector Search P95",
            "refId": "A"
          },
          {
            "expr": "histogram_quantile(0.95, embedding_generation_seconds_bucket)",
            "legendFormat": "Embedding Gen P95",
            "refId": "B"
          }
        ]
      }
    ]
  }
}
```

## Monitoring Best Practices

### 1. Set Appropriate Thresholds
- Define realistic thresholds based on historical data
- Consider business requirements when setting thresholds
- Regularly review and adjust thresholds

### 2. Implement Meaningful Alerts
- Alert on symptoms rather than causes
- Avoid alert fatigue with proper grouping
- Ensure alerts are actionable

### 3. Monitor the Monitor
- Monitor monitoring system health
- Set up alerts for monitoring system failures
- Regularly test alerting pipelines

### 4. Data Retention Policies
- Implement appropriate data retention policies
- Balance storage costs with monitoring needs
- Archive historical data for long-term analysis

### 5. Security Monitoring
- Monitor for security-related events
- Track authentication and authorization events
- Monitor for unusual access patterns

## Monitoring Tools

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts.yml"

scrape_configs:
  - job_name: 'rag-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
    scrape_timeout: 5s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
```

### Jaeger Configuration
```yaml
# jaeger configuration
version: '3.8'

services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # UI
      - "14268:14268"  # Collector
      - "6831:6831/udp"  # Agent
```

### Alertmanager Configuration
```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alertmanager@example.org'
  smtp_auth_username: 'admin'
  smtp_auth_password: 'password'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default-receiver'

receivers:
  - name: 'default-receiver'
    email_configs:
    - to: 'admin@example.org'
```

## Monitoring Implementation

### Middleware for Request Monitoring
```python
# api.py - Monitoring middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import time

class MonitoringMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        # Record metrics
        request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        duration = time.time() - start_time
        request_duration_histogram.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        return response

# Add middleware to app
app.add_middleware(MonitoringMiddleware)
```

### Custom Metrics Decorator
```python
# src/observability/__init__.py - Metrics decorator
from functools import wraps

def monitor_function(name: str = None):
    """Decorator to monitor function performance."""
    def decorator(func):
        func_name = name or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record success metrics
                function_duration_histogram.labels(
                    function=func_name,
                    status='success'
                ).observe(duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                # Record error metrics
                function_duration_histogram.labels(
                    function=func_name,
                    status='error'
                ).observe(duration)
                
                function_error_count.labels(
                    function=func_name,
                    error_type=type(e).__name__
                ).inc()
                
                raise
        
        return wrapper
    return decorator

# Usage example
@monitor_function("query_processing")
def process_query(query: str):
    """Process a query with monitoring."""
    # Implementation here
    pass
```

## Monitoring Data Analysis

### Performance Analysis
```python
# Analysis script for monitoring data
import pandas as pd
import matplotlib.pyplot as plt
from prometheus_api_client import PrometheusConnect

def analyze_performance():
    """Analyze performance metrics from Prometheus."""
    prom = PrometheusConnect(url="http://localhost:9090")
    
    # Get request duration data
    duration_data = prom.custom_query(query='request_duration_seconds')
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(duration_data)
    
    # Calculate percentiles
    p50 = df['value'].quantile(0.5)
    p95 = df['value'].quantile(0.95)
    p99 = df['value'].quantile(0.99)
    
    print(f"P50: {p50}, P95: {p95}, P99: {p99}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.hist(df['value'], bins=50, alpha=0.7)
    plt.axvline(p95, color='red', linestyle='--', label=f'P95: {p95:.2f}s')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.title('Request Duration Distribution')
    plt.legend()
    plt.show()
```

This monitoring documentation provides a comprehensive guide to implementing and maintaining monitoring for the Production RAG System. Proper monitoring is essential for maintaining system reliability, performance, and availability.