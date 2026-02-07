# Performance Testing & Capacity Planning
## Load Testing, Stress Testing, and Optimization for Cloud Deployments

## Overview

This guide teaches you how to test, measure, and optimize RAG Engine Mini performance in production cloud environments. You'll learn to identify bottlenecks, plan capacity, and ensure your application can handle peak loads.

**Estimated Time:** 3-4 hours  
**Prerequisites:** Cloud deployment completion, understanding of HTTP APIs

**Learning Objectives:**
1. Design comprehensive load tests
2. Implement load testing tools (Locust, k6, Artillery)
3. Identify performance bottlenecks
4. Conduct stress and spike testing
5. Plan capacity for growth
6. Optimize cloud resources
7. Set up performance monitoring

---

## Part 1: Performance Testing Fundamentals

### Types of Performance Tests

**1. Load Testing:**
- Purpose: Validate system handles expected traffic
- Method: Gradually increase load to target levels
- Metrics: Response time, throughput, error rate
- Example: Simulate 1,000 concurrent users

**2. Stress Testing:**
- Purpose: Find breaking point and recovery behavior
- Method: Push beyond normal capacity until failure
- Metrics: Breaking point, recovery time, error handling
- Example: Ramp to 10,000 users until system fails

**3. Spike Testing:**
- Purpose: Test sudden traffic surges
- Method: Instant jump to high load
- Metrics: Response to burst, auto-scaling speed
- Example: 0 → 5,000 users in 10 seconds

**4. Endurance Testing:**
- Purpose: Check for memory leaks and degradation
- Method: Sustained load over hours/days
- Metrics: Memory usage, response time trends
- Example: 8 hours at 80% capacity

**5. Scalability Testing:**
- Purpose: Validate auto-scaling effectiveness
- Method: Increasing load, measure scale events
- Metrics: Scale-up time, cost per request
- Example: Increase load 20% every 5 minutes

### Key Performance Metrics

```
Response Time (Latency):
├── P50 (Median): 50% of requests faster than this
├── P95: 95% of requests faster than this (common SLA)
├── P99: 99% of requests faster than this (tail latency)
└── P99.9: 99.9% of requests (outliers)

Throughput:
├── Requests per second (RPS)
├── Queries per second (QPS)
└── Documents processed per second

Error Rates:
├── HTTP 4xx (Client errors)
├── HTTP 5xx (Server errors)
└── Timeout rate

Resource Utilization:
├── CPU utilization %
├── Memory utilization %
├── Disk I/O
└── Network bandwidth

Business Metrics:
├── Cost per request
├── Cost per user
└── Efficiency (requests per $)
```

### Performance Targets for RAG Engine

```yaml
# performance-targets.yaml
api:
  health_check:
    p95_response_time: 100ms
    p99_response_time: 200ms
    max_error_rate: 0.1%
  
  search_queries:
    p95_response_time: 500ms
    p99_response_time: 1000ms
    max_error_rate: 1%
    min_throughput: 100 RPS
  
  document_upload:
    p95_response_time: 5000ms  # 5 seconds
    p99_response_time: 10000ms # 10 seconds
    max_error_rate: 0.5%
    max_file_size: 100MB

infrastructure:
  cpu_utilization:
    target: 70%
    max: 85%
  
  memory_utilization:
    target: 80%
    max: 90%
  
  auto_scaling:
    scale_up_time: 60 seconds
    scale_down_time: 300 seconds
    min_instances: 3
    max_instances: 20
  
  database:
    max_connections: 80% of limit
    query_time: 95% under 100ms
    replication_lag: under 5 seconds

cost:
  target_cost_per_1k_requests: $0.10
  max_monthly_budget: $500
```

---

## Part 2: Load Testing Tools

### Tool Comparison

| Tool | Language | Best For | Pros | Cons |
|------|----------|----------|------|------|
| **Locust** | Python | Python teams, distributed | Simple code, scalable | Requires Python knowledge |
| **k6** | JavaScript | CI/CD integration | Cloud-native, modern | Steeper learning curve |
| **Artillery** | JavaScript/JSON | Quick tests, scenarios | Easy YAML config | Less flexible |
| **JMeter** | Java | Enterprise, GUI | Mature, GUI | Heavy, complex |
| **Gatling** | Scala | High performance | Fast, reactive | Scala knowledge needed |

### Locust Implementation

**Installation:**
```bash
pip install locust
```

**Basic Test Script:**
```python
# locustfile.py
from locust import HttpUser, task, between
import random

class RAGEngineUser(HttpUser):
    """Simulates a user interacting with RAG Engine"""
    
    # Wait between 1-5 seconds between tasks
    wait_time = between(1, 5)
    
    def on_start(self):
        """Login when user starts"""
        response = self.client.post("/auth/login", json={
            "username": f"test_user_{self.user_id}",
            "password": "test_password"
        })
        if response.status_code == 200:
            self.token = response.json()["access_token"]
        else:
            self.token = None
    
    @task(10)
    def health_check(self):
        """Most frequent: health check"""
        self.client.get("/health")
    
    @task(5)
    def search_documents(self):
        """Search for documents"""
        queries = [
            "machine learning",
            "neural networks",
            "python programming",
            "data structures",
            "cloud architecture"
        ]
        query = random.choice(queries)
        
        with self.client.post(
            "/api/v1/search",
            json={"query": query, "top_k": 5},
            headers={"Authorization": f"Bearer {self.token}"} if self.token else {},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                results = response.json()
                if len(results.get("documents", [])) > 0:
                    response.success()
                else:
                    response.failure("No results returned")
            elif response.status_code == 429:
                response.success()  # Rate limiting is expected
            else:
                response.failure(f"Unexpected status: {response.status_code}")
    
    @task(3)
    def upload_document(self):
        """Upload a document (less frequent)"""
        # Simulate file upload
        files = {
            'file': ('test_document.txt', 'This is a test document content', 'text/plain')
        }
        
        self.client.post(
            "/api/v1/documents/upload",
            files=files,
            headers={"Authorization": f"Bearer {self.token}"} if self.token else {}
        )
    
    @task(2)
    def get_document_status(self):
        """Check document processing status"""
        # In real test, track document IDs from uploads
        doc_id = random.randint(1, 1000)
        self.client.get(
            f"/api/v1/documents/{doc_id}/status",
            headers={"Authorization": f"Bearer {self.token}"} if self.token else {}
        )

class PeakLoadUser(HttpUser):
    """Simulates peak load with minimal wait time"""
    wait_time = between(0.1, 0.5)
    
    @task
    def high_frequency_search(self):
        """High-frequency search requests"""
        self.client.post("/api/v1/search", json={
            "query": "performance test",
            "top_k": 3
        })
```

**Running Locust:**
```bash
# Local testing
locust -f locustfile.py --host http://localhost:8000

# Open http://localhost:8089 to start test

# Distributed testing (multiple workers)
locust -f locustfile.py --master --host http://api.rag-engine.com
locust -f locustfile.py --worker --master-host=192.168.1.100

# Command line (non-interactive)
locust -f locustfile.py \
    --host http://api.rag-engine.com \
    --users 1000 \
    --spawn-rate 100 \
    --run-time 5m \
    --headless \
    --csv=rag-engine-test
```

**Advanced Locust Features:**
```python
# locustfile_advanced.py
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner
import json
import time

# Custom metrics
class Metrics:
    search_times = []
    error_count = 0

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, 
               response, context, exception, **kwargs):
    """Track all requests"""
    if "search" in name:
        Metrics.search_times.append(response_time)
    
    if exception:
        Metrics.error_count += 1

@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Generate report when test ends"""
    if isinstance(environment.runner, MasterRunner):
        # Calculate percentiles
        if Metrics.search_times:
            times = sorted(Metrics.search_times)
            p50 = times[int(len(times) * 0.5)]
            p95 = times[int(len(times) * 0.95)]
            p99 = times[int(len(times) * 0.99)]
            
            print(f"\n=== PERFORMANCE REPORT ===")
            print(f"Total requests: {len(Metrics.search_times)}")
            print(f"P50 response time: {p50:.2f}ms")
            print(f"P95 response time: {p95:.2f}ms")
            print(f"P99 response time: {p99:.2f}ms")
            print(f"Error count: {Metrics.error_count}")
            
            # Write to file
            with open("performance-report.json", "w") as f:
                json.dump({
                    "total_requests": len(Metrics.search_times),
                    "p50": p50,
                    "p95": p95,
                    "p99": p99,
                    "errors": Metrics.error_count,
                    "timestamp": time.time()
                }, f)

class SpikeTest(HttpUser):
    """Test handling of traffic spikes"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.is_spike = False
    
    @task(1)
    def normal_load(self):
        if not self.is_spike:
            self.client.get("/api/v1/search?query=test")
    
    @task(10)
    def spike_load(self):
        if self.is_spike:
            self.client.get("/api/v1/search?query=spike")

# Simulate spike via event
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    def trigger_spike():
        time.sleep(60)  # Wait 1 minute
        print("⚡ SPIKE STARTING - Increasing load 10x")
        for user in environment.runner.user_greenlets:
            if hasattr(user, 'is_spike'):
                user.is_spike = True
        time.sleep(30)  # Spike for 30 seconds
        print("📉 SPIKE ENDING - Returning to normal")
        for user in environment.runner.user_greenlets:
            if hasattr(user, 'is_spike'):
                user.is_spike = False
    
    if isinstance(environment.runner, MasterRunner):
        from gevent import spawn
        spawn(trigger_spike)
```

### k6 Implementation

**Installation:**
```bash
# macOS
brew install k6

# Windows
choco install k6

# Linux
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5A17E747BE80122

# Docker
docker pull grafana/k6
```

**Basic Test Script:**
```javascript
// load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const searchResponseTime = new Trend('search_response_time');
const uploadCounter = new Counter('uploads');

// Test configuration
export const options = {
  // Test stages (ramp up, sustain, ramp down)
  stages: [
    { duration: '2m', target: 100 },    // Ramp up
    { duration: '5m', target: 100 },    // Stay at 100 users
    { duration: '2m', target: 200 },    // Ramp up
    { duration: '5m', target: 200 },    // Stay at 200 users
    { duration: '2m', target: 0 },      // Ramp down
  ],
  
  // Performance thresholds
  thresholds: {
    http_req_duration: ['p(95)<500'],     // 95% under 500ms
    http_req_failed: ['rate<0.1'],        // Less than 0.1% errors
    errors: ['rate<0.1'],
    search_response_time: ['p(95)<500'],
  },
  
  // Tags for cloud execution
  ext: {
    loadimpact: {
      name: 'RAG Engine Load Test',
      projectID: 12345,
    },
  },
};

// Setup: Run once per VU (virtual user)
export function setup() {
  // Login and get token
  const loginRes = http.post('http://api.rag-engine.com/auth/login', {
    username: 'test_user',
    password: 'test_password',
  });
  
  check(loginRes, {
    'login successful': (r) => r.status === 200,
  });
  
  return {
    token: loginRes.json('access_token'),
  };
}

// Main test function: Runs repeatedly
export default function(data) {
  const headers = {
    'Authorization': `Bearer ${data.token}`,
    'Content-Type': 'application/json',
  };
  
  // Health check (most frequent)
  const healthRes = http.get('http://api.rag-engine.com/health');
  check(healthRes, {
    'health check status is 200': (r) => r.status === 200,
    'health check response time < 100ms': (r) => r.timings.duration < 100,
  });
  
  // Search request (common operation)
  const searchRes = http.post('http://api.rag-engine.com/api/v1/search', 
    JSON.stringify({
      query: 'machine learning tutorial',
      top_k: 5,
    }), 
    { headers }
  );
  
  const searchSuccess = check(searchRes, {
    'search status is 200': (r) => r.status === 200,
    'search returns results': (r) => r.json('documents').length > 0,
    'search response time < 500ms': (r) => r.timings.duration < 500,
  });
  
  searchResponseTime.add(searchRes.timings.duration);
  errorRate.add(!searchSuccess);
  
  // Document upload (less frequent, 20% of the time)
  if (Math.random() < 0.2) {
    const file = open('test-document.txt', 'b');
    const uploadRes = http.post('http://api.rag-engine.com/api/v1/documents/upload',
      file,
      {
        headers: {
          'Authorization': `Bearer ${data.token}`,
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    
    check(uploadRes, {
      'upload successful': (r) => r.status === 201,
    });
    
    uploadCounter.add(1);
  }
  
  // Random sleep between 1-5 seconds (think time)
  sleep(Math.random() * 4 + 1);
}

// Teardown: Run once at end
export function teardown(data) {
  console.log('Test completed');
  console.log(`Token used: ${data.token.substring(0, 10)}...`);
}
```

**Running k6:**
```bash
# Local execution
k6 run load-test.js

# Cloud execution (k6 cloud)
k6 cloud run load-test.js

# Output to InfluxDB
k6 run --out influxdb=http://localhost:8086/k6 load-test.js

# With environment variables
k6 run -e API_URL=http://api.rag-engine.com load-test.js

# Multiple outputs
k6 run \
  --out json=results.json \
  --out csv=results.csv \
  --out influxdb=http://localhost:8086/k6 \
  load-test.js
```

**Advanced k6 Scenarios:**
```javascript
// stress-test.js
import http from 'k6/http';
import { check } from 'k6';

export const options = {
  // Stress test: Keep increasing until failure
  stages: [
    { duration: '2m', target: 100 },   // Normal
    { duration: '2m', target: 500 },   // High
    { duration: '2m', target: 1000 },  // Very high
    { duration: '2m', target: 2000 },  // Stress
    { duration: '5m', target: 2000 },  // Sustained stress
    { duration: '2m', target: 0 },     // Recovery
  ],
  
  // Stop test if error rate exceeds 50%
  thresholds: {
    http_req_failed: [
      {
        threshold: 'rate<0.5',
        abortOnFail: true,
        delayAbortEval: '2m',
      },
    ],
  },
};

export default function() {
  const res = http.post('http://api.rag-engine.com/api/v1/search', {
    query: 'stress test',
    top_k: 5,
  });
  
  check(res, {
    'status is 200 or 429': (r) => r.status === 200 || r.status === 429,
  });
}
```

```javascript
// spike-test.js
import http from 'k6/http';
import { check } from 'k6';

export const options = {
  // Spike test: Sudden traffic burst
  stages: [
    { duration: '1m', target: 10 },     // Baseline
    { duration: '10s', target: 5000 },  // SPIKE!
    { duration: '5m', target: 5000 },   // Sustain
    { duration: '2m', target: 10 },     // Recovery
  ],
};

export default function() {
  const res = http.get('http://api.rag-engine.com/health');
  
  check(res, {
    'status is 200': (r) => r.status === 200,
  });
}
```

### Artillery Implementation

**Quick Start:**
```yaml
# artillery-test.yml
config:
  target: "http://api.rag-engine.com"
  phases:
    - duration: 60
      arrivalRate: 10
      name: "Warm up"
    - duration: 120
      arrivalRate: 50
      rampTo: 100
      name: "Ramp up"
    - duration: 300
      arrivalRate: 100
      name: "Sustained load"
  plugins:
    expect: {}
  defaults:
    headers:
      Content-Type: "application/json"

scenarios:
  - name: "Search documents"
    weight: 10
    requests:
      - post:
          url: "/api/v1/search"
          json:
            query: "machine learning"
            top_k: 5
          expect:
            - statusCode: 200
            - contentType: json
            - hasProperty: "documents"
            
  - name: "Upload document"
    weight: 2
    requests:
      - post:
          url: "/api/v1/documents/upload"
          formData:
            file:
              fromFile: "test-document.txt"
          expect:
            - statusCode: 201
            
  - name: "Health check"
    weight: 20
    requests:
      - get:
          url: "/health"
          expect:
            - statusCode: 200
```

**Running Artillery:**
```bash
# Install
npm install -g artillery

# Run test
artillery run artillery-test.yml

# With output
artillery run artillery-test.yml --output results.json

# Generate HTML report
artillery report results.json
```

---

## Part 3: Cloud-Specific Load Testing

### AWS Load Testing

**AWS Distributed Load Testing:**
```bash
# Use AWS Distributed Load Testing solution
# Deployed via CloudFormation

# 1. Deploy the solution
aws cloudformation create-stack \
    --stack-name rag-engine-load-testing \
    --template-url https://s3.amazonaws.com/solutions-reference/distributed-load-testing/latest/distributed-load-testing-on-aws.template \
    --capabilities CAPABILITY_IAM

# 2. Access the console
# https://<api-id>.execute-api.<region>.amazonaws.com/

# 3. Create test with JMeter or Python script
# Upload your locustfile.py

# 4. Configure test parameters
# Target: http://rag-engine-alb-123456.us-west-2.elb.amazonaws.com
# Concurrency: 10,000 users
# Ramp up: 5 minutes
# Duration: 30 minutes

# 5. Run test and view results in real-time
```

**Auto-scaling Validation:**
```bash
#!/bin/bash
# test-aws-autoscaling.sh

TARGET_GROUP_ARN="arn:aws:elasticloadbalancing:us-west-2:123456789:targetgroup/rag-engine-tg/12345678"
ECS_SERVICE="rag-engine-api"
ECS_CLUSTER="rag-engine-prod"

# 1. Get initial state
echo "Initial state:"
aws ecs describe-services \
    --cluster ${ECS_CLUSTER} \
    --services ${ECS_SERVICE} \
    --query 'services[0].runningCount'

# 2. Start load test in background
locust -f locustfile.py \
    --host http://rag-engine-alb-123456.us-west-2.elb.amazonaws.com \
    --users 5000 \
    --spawn-rate 100 \
    --run-time 10m \
    --headless &

LOCUST_PID=$!

# 3. Monitor scaling events
echo "Monitoring auto-scaling..."
for i in {1..30}; do
    sleep 30
    
    # Get current task count
    TASK_COUNT=$(aws ecs describe-services \
        --cluster ${ECS_CLUSTER} \
        --services ${ECS_SERVICE} \
        --query 'services[0].runningCount' \
        --output text)
    
    # Get CloudWatch metrics
    CPU_UTIL=$(aws cloudwatch get-metric-statistics \
        --namespace AWS/ECS \
        --metric-name CPUUtilization \
        --dimensions Name=ServiceName,Value=${ECS_SERVICE} Name=ClusterName,Value=${ECS_CLUSTER} \
        --start-time $(date -u -d '5 minutes ago' +%Y-%m-%dT%H:%M:%S) \
        --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
        --period 300 \
        --statistics Average \
        --query 'Datapoints[0].Average' \
        --output text)
    
    echo "$(date): Tasks: ${TASK_COUNT}, CPU: ${CPU_UTIL}%"
done

# 4. Stop load test
kill ${LOCUST_PID}

# 5. Wait for scale-down
echo "Waiting for scale-down..."
sleep 600

FINAL_COUNT=$(aws ecs describe-services \
    --cluster ${ECS_CLUSTER} \
    --services ${ECS_SERVICE} \
    --query 'services[0].runningCount' \
    --output text)

echo "Final task count: ${FINAL_COUNT}"
echo "Auto-scaling test completed"
```

### GCP Load Testing

**Cloud Run Scaling Test:**
```bash
#!/bin/bash
# test-gcp-scaling.sh

SERVICE_NAME="rag-engine-api"
REGION="us-central1"

# 1. Get initial instance count
echo "Initial state:"
gcloud run services describe ${SERVICE_NAME} \
    --region=${REGION} \
    --format='value(spec.template.spec.containers[0].resources.limits.cpu)'

# 2. Run load test
echo "Starting load test..."
locust -f locustfile.py \
    --host https://${SERVICE_NAME}-${REGION}-uc.a.run.app \
    --users 1000 \
    --spawn-rate 50 \
    --run-time 5m \
    --headless &

LOCUST_PID=$!

# 3. Monitor instance count
echo "Monitoring instance count..."
for i in {1..20}; do
    sleep 15
    
    # Get metrics from Cloud Monitoring
    INSTANCE_COUNT=$(gcloud monitoring metrics list \
        --filter='metric.type="run.googleapis.com/container/instance_count"' \
        --format='value(points[0].value.int64_value)' 2>/dev/null || echo "N/A")
    
    echo "$(date): Instances: ${INSTANCE_COUNT}"
done

# 4. Stop load test
kill ${LOCUST_PID}

# 5. Check cold start times
echo "Checking cold start metrics..."
gcloud monitoring metrics list \
    --filter='metric.type="run.googleapis.com/container/startup_latency"' \
    --format='table(points[0].value.double_value)'

echo "GCP scaling test completed"
```

**GKE Autoscaling Test:**
```bash
#!/bin/bash
# test-gke-autoscaling.sh

NAMESPACE="rag-engine"
DEPLOYMENT="rag-engine-api"

# 1. Get initial state
echo "Initial state:"
kubectl get deployment ${DEPLOYMENT} -n ${NAMESPACE} \
    -o jsonpath='{.status.replicas}'

kubectl get nodes -l cloud.google.com/gke-nodepool=rag-engine-pool

# 2. Apply load
echo "Applying load..."
kubectl run load-generator \
    --image=williamyeh/wrk \
    --rm \
    -it \
    -- \
    -t4 -c100 -d5m \
    http://rag-engine-service.${NAMESPACE}.svc.cluster.local/api/v1/search

# 3. Watch HPA and nodes
echo "Monitoring..."
watch -n 5 "
    echo '=== HPA ===' && \
    kubectl get hpa ${DEPLOYMENT} -n ${NAMESPACE} && \
    echo '' && \
    echo '=== Nodes ===' && \
    kubectl get nodes -l cloud.google.com/gke-nodepool=rag-engine-pool && \
    echo '' && \
    echo '=== Pods ===' && \
    kubectl get pods -n ${NAMESPACE} | grep ${DEPLOYMENT}
"

# 4. Check node pool scaling
gcloud container node-pools describe rag-engine-pool \
    --cluster=rag-engine-cluster \
    --format='value(autoscaling.enabled,autoscaling.minNodeCount,autoscaling.maxNodeCount)'
```

### Azure Load Testing

**Azure Load Testing Service:**
```bash
# Create Azure Load Testing resource
az load create \
    --name rag-engine-load-test \
    --resource-group rag-engine-rg \
    --location eastus

# Create test with JMeter script
az load test create \
    --load-test-resource rag-engine-load-test \
    --test-id rag-engine-api-test \
    --display-name "RAG Engine API Load Test" \
    --description "Testing API performance under load" \

# Run test
az load test run create \
    --load-test-resource rag-engine-load-test \
    --test-id rag-engine-api-test \
    --run-id run-$(date +%s)

# Monitor in Azure Portal
# https://portal.azure.com -> Load Testing -> rag-engine-load-test
```

**Container Apps Scaling Test:**
```bash
#!/bin/bash
# test-azure-scaling.sh

RESOURCE_GROUP="rag-engine-rg"
CONTAINER_APP="rag-engine-api"

# 1. Get initial replica count
echo "Initial state:"
az containerapp show \
    --resource-group ${RESOURCE_GROUP} \
    --name ${CONTAINER_APP} \
    --query properties.configuration.template.scale.minReplicas

# 2. Run load test
echo "Starting load test..."
locust -f locustfile.py \
    --host https://$(az containerapp show \
        --resource-group ${RESOURCE_GROUP} \
        --name ${CONTAINER_APP} \
        --query properties.configuration.ingress.fqdn \
        --output tsv) \
    --users 500 \
    --spawn-rate 50 \
    --run-time 5m \
    --headless &

LOCUST_PID=$!

# 3. Monitor replica count
echo "Monitoring replica count..."
for i in {1..20}; do
    sleep 15
    
    REPLICA_COUNT=$(az containerapp show \
        --resource-group ${RESOURCE_GROUP} \
        --name ${CONTAINER_APP} \
        --query properties.runningCount \
        --output tsv)
    
    echo "$(date): Replicas: ${REPLICA_COUNT}"
done

# 4. Stop load test
kill ${LOCUST_PID}

# 5. Check Application Insights for response times
echo "Response time metrics:"
az monitor app-insights metrics show \
    --app rag-engine-insights \
    --metric "requests/duration" \
    --aggregation Average \
    --interval PT1M \
    --start-time $(date -u -d '10 minutes ago' +%Y-%m-%dT%H:%M:%S) \
    --end-time $(date -u +%Y-%m-%dT%H:%M:%S)

echo "Azure scaling test completed"
```

---

## Part 4: Analyzing Results

### Key Metrics to Analyze

**1. Response Time Distribution:**
```python
# analyze_results.py
import json
import statistics

# Load k6 results
with open('results.json') as f:
    data = json.load(f)

# Extract response times
times = [metric['value'] for metric in data['metrics']['http_req_duration']['values']]

# Calculate percentiles
times_sorted = sorted(times)
p50 = statistics.median(times_sorted)
p95 = times_sorted[int(len(times_sorted) * 0.95)]
p99 = times_sorted[int(len(times_sorted) * 0.99)]

print(f"Response Time Analysis:")
print(f"  P50 (Median): {p50:.2f}ms")
print(f"  P95: {p95:.2f}ms")
print(f"  P99: {p99:.2f}ms")
print(f"  Min: {min(times):.2f}ms")
print(f"  Max: {max(times):.2f}ms")
print(f"  Std Dev: {statistics.stdev(times):.2f}ms")

# Check against SLA
if p95 > 500:
    print("⚠️  WARNING: P95 exceeds 500ms SLA!")
else:
    print("✅ P95 within SLA")
```

**2. Error Analysis:**
```python
# Analyze error patterns
error_metrics = data['metrics']['http_req_failed']
error_rate = error_metrics['values']['rate']

print(f"\nError Analysis:")
print(f"  Error Rate: {error_rate * 100:.2f}%")
print(f"  Total Errors: {error_metrics['values']['count']}")
print(f"  Total Requests: {error_metrics['values']['count'] + data['metrics']['http_reqs']['values']['count']}")

if error_rate > 0.01:
    print("❌ CRITICAL: Error rate exceeds 1%!")
elif error_rate > 0.001:
    print("⚠️  WARNING: Error rate above 0.1%")
else:
    print("✅ Error rate acceptable")
```

**3. Throughput Analysis:**
```python
# Calculate throughput
throughput = data['metrics']['http_reqs']['values']['count'] / data['state']['testRunDuration']

print(f"\nThroughput Analysis:")
print(f"  Requests/Second: {throughput:.2f}")
print(f"  Requests/Minute: {throughput * 60:.2f}")
print(f"  Total Requests: {data['metrics']['http_reqs']['values']['count']}")

# Estimate max capacity
# If current is 200 RPS with 70% CPU, max is ~285 RPS (at 100%)
print(f"  Estimated Max Capacity: {throughput / 0.7:.2f} RPS (at 100% CPU)")
```

### Visualizing Results

**Generate Charts:**
```python
# generate_charts.py
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Load time-series data
df = pd.read_csv('results.csv')

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Response time over time
axes[0].plot(df['timestamp'], df['http_req_duration'], alpha=0.5)
axes[0].axhline(y=500, color='r', linestyle='--', label='SLA (500ms)')
axes[0].set_ylabel('Response Time (ms)')
axes[0].set_title('Response Time Over Time')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# RPS over time
axes[1].plot(df['timestamp'], df['http_reqs_per_sec'])
axes[1].set_ylabel('Requests/Second')
axes[1].set_title('Throughput Over Time')
axes[1].grid(True, alpha=0.3)

# Error rate over time
axes[2].plot(df['timestamp'], df['http_req_failed_rate'] * 100)
axes[2].axhline(y=1, color='r', linestyle='--', label='Max Error Rate (1%)')
axes[2].set_ylabel('Error Rate (%)')
axes[2].set_xlabel('Time')
axes[2].set_title('Error Rate Over Time')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('performance-report.png', dpi=150)
print("✅ Charts saved to performance-report.png")
```

---

## Part 5: Capacity Planning

### Calculating Required Capacity

**Formula:**
```
Required Capacity = (Peak Users × Requests per User per Minute) / Desired Response Time

Example:
- Peak Users: 10,000
- Requests per User per Minute: 10
- Desired Response Time: 500ms = 0.5 seconds
- Required RPS: (10,000 × 10) / 60 = 1,667 RPS

With 200 RPS per instance:
- Required Instances: 1,667 / 200 = 8.3 → 9 instances (round up)
- Add 20% buffer: 9 × 1.2 = 11 instances
```

**Growth Planning:**
```python
# capacity_planner.py

def calculate_capacity(current_metrics, growth_projection):
    """
    Calculate required capacity for growth
    
    current_metrics: {
        'users': 5000,
        'rps': 500,
        'instances': 3,
        'cpu_utilization': 70,
        'response_time_p95': 400
    }
    
    growth_projection: {
        'user_growth_rate': 0.15,  # 15% per month
        'months': 6
    }
    """
    
    # Project future users
    future_users = current_metrics['users'] * \
                   ((1 + growth_projection['user_growth_rate']) ** growth_projection['months'])
    
    # Calculate required RPS (assume same requests per user)
    current_rps_per_user = current_metrics['rps'] / current_metrics['users']
    future_rps = future_users * current_rps_per_user
    
    # Calculate required instances
    current_rps_per_instance = current_metrics['rps'] / current_metrics['instances']
    required_instances = future_rps / current_rps_per_instance
    
    # Add buffer (20%)
    instances_with_buffer = required_instances * 1.2
    
    # Round up
    final_instances = int(instances_with_buffer) + (1 if instances_with_buffer % 1 > 0 else 0)
    
    print("=== CAPACITY PLANNING REPORT ===")
    print(f"Current State:")
    print(f"  Users: {current_metrics['users']:,}")
    print(f"  RPS: {current_metrics['rps']}")
    print(f"  Instances: {current_metrics['instances']}")
    print(f"  CPU: {current_metrics['cpu_utilization']}%")
    print()
    print(f"Growth Projection ({growth_projection['months']} months):")
    print(f"  User Growth Rate: {growth_projection['user_growth_rate'] * 100}%/month")
    print(f"  Future Users: {future_users:,.0f}")
    print(f"  Future RPS: {future_rps:,.0f}")
    print()
    print(f"Required Capacity:")
    print(f"  Minimum Instances: {int(required_instances)}")
    print(f"  With 20% Buffer: {final_instances}")
    print(f"  Instance Growth: +{final_instances - current_metrics['instances']}")
    print()
    print(f"Cost Impact:")
    current_cost = current_metrics['instances'] * 50  # $50/instance/month
    future_cost = final_instances * 50
    print(f"  Current Monthly Cost: ${current_cost:,}")
    print(f"  Future Monthly Cost: ${future_cost:,}")
    print(f"  Cost Increase: ${future_cost - current_cost:,} (+{((future_cost - current_cost) / current_cost) * 100:.1f}%)")
    
    return {
        'future_users': future_users,
        'future_rps': future_rps,
        'required_instances': final_instances,
        'cost_increase': future_cost - current_cost
    }

# Example usage
current = {
    'users': 5000,
    'rps': 500,
    'instances': 3,
    'cpu_utilization': 70,
    'response_time_p95': 400
}

growth = {
    'user_growth_rate': 0.15,
    'months': 6
}

plan = calculate_capacity(current, growth)
```

### Resource Right-Sizing

**Analyze Current Usage:**
```bash
#!/bin/bash
# analyze-resource-usage.sh

# AWS ECS
aws cloudwatch get-metric-statistics \
    --namespace AWS/ECS \
    --metric-name CPUUtilization \
    --dimensions Name=ClusterName,Value=rag-engine-prod Name=ServiceName,Value=rag-engine-api \
    --start-time $(date -u -d '7 days ago' +%Y-%m-%dT%H:%M:%S) \
    --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
    --period 3600 \
    --statistics Average Maximum \
    --output table

# Get memory usage
aws cloudwatch get-metric-statistics \
    --namespace AWS/ECS \
    --metric-name MemoryUtilization \
    --dimensions Name=ClusterName,Value=rag-engine-prod Name=ServiceName,Value=rag-engine-api \
    --start-time $(date -u -d '7 days ago' +%Y-%m-%dT%H:%M:%S) \
    --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
    --period 3600 \
    --statistics Average Maximum
```

**Recommendations:**
```python
# right_sizing_recommendations.py

def generate_recommendations(metrics):
    """
    Generate right-sizing recommendations
    
    metrics: {
        'avg_cpu': 35,
        'max_cpu': 85,
        'avg_memory': 40,
        'max_memory': 70,
        'current_cpu_units': 512,
        'current_memory_mb': 1024
    }
    """
    
    recommendations = []
    
    # CPU Analysis
    if metrics['avg_cpu'] < 30 and metrics['max_cpu'] < 60:
        recommendations.append({
            'resource': 'CPU',
            'current': metrics['current_cpu_units'],
            'recommended': int(metrics['current_cpu_units'] * 0.75),
            'savings': '25%',
            'priority': 'high'
        })
    elif metrics['avg_cpu'] > 70 or metrics['max_cpu'] > 90:
        recommendations.append({
            'resource': 'CPU',
            'current': metrics['current_cpu_units'],
            'recommended': int(metrics['current_cpu_units'] * 1.25),
            'reason': 'High utilization detected',
            'priority': 'high'
        })
    
    # Memory Analysis
    if metrics['avg_memory'] < 30 and metrics['max_memory'] < 60:
        recommendations.append({
            'resource': 'Memory',
            'current': metrics['current_memory_mb'],
            'recommended': int(metrics['current_memory_mb'] * 0.75),
            'savings': '25%',
            'priority': 'medium'
        })
    elif metrics['avg_memory'] > 80:
        recommendations.append({
            'resource': 'Memory',
            'current': metrics['current_memory_mb'],
            'recommended': int(metrics['current_memory_mb'] * 1.5),
            'reason': 'Memory pressure detected',
            'priority': 'high'
        })
    
    return recommendations

# Example
metrics = {
    'avg_cpu': 35,
    'max_cpu': 85,
    'avg_memory': 40,
    'max_memory': 70,
    'current_cpu_units': 512,
    'current_memory_mb': 1024
}

recs = generate_recommendations(metrics)
print("=== RIGHT-SIZING RECOMMENDATIONS ===")
for rec in recs:
    print(f"\n{rec['resource']}:")
    print(f"  Current: {rec['current']}")
    print(f"  Recommended: {rec['recommended']}")
    if 'savings' in rec:
        print(f"  Potential Savings: {rec['savings']}")
    if 'reason' in rec:
        print(f"  Reason: {rec['reason']}")
    print(f"  Priority: {rec['priority']}")
```

---

## Part 6: Performance Optimization

### Database Optimization

**Query Optimization:**
```sql
-- Add indexes for frequent queries
CREATE INDEX CONCURRENTLY idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX CONCURRENTLY idx_documents_created_at ON documents(created_at);
CREATE INDEX CONCURRENTLY idx_documents_user_id ON documents(user_id);

-- Analyze query performance
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT * FROM documents 
WHERE embedding <-> query_embedding < 0.5 
ORDER BY embedding <-> query_embedding 
LIMIT 10;

-- Check slow queries
SELECT query, mean_exec_time, calls 
FROM pg_stat_statements 
ORDER BY mean_exec_time DESC 
LIMIT 10;
```

**Connection Pooling:**
```python
# config.py - Optimized connection pool
DATABASE_CONFIG = {
    'pool_size': 20,              # Fixed pool size
    'max_overflow': 30,           # Extra connections under load
    'pool_timeout': 30,           # Wait for available connection
    'pool_recycle': 3600,         # Recycle connections after 1 hour
    'pool_pre_ping': True,        # Verify connection before use
}
```

### Caching Strategy

**Redis Caching:**
```python
# cache_decorator.py
import functools
import json
from app.cache import redis_client

def cache_result(timeout=300):
    """Cache function results in Redis"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"cache:{func.__name__}:{hash(str(args))}:{hash(str(kwargs))}"
            
            # Try to get from cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            redis_client.setex(
                cache_key,
                timeout,
                json.dumps(result)
            )
            
            return result
        return wrapper
    return decorator

# Usage
@cache_result(timeout=600)  # Cache for 10 minutes
def search_documents(query, top_k=5):
    # Expensive search operation
    return vector_search(query, top_k)
```

### CDN Optimization

**CloudFront/Cloudflare Setup:**
```hcl
# terraform/cdn.tf
resource "aws_cloudfront_distribution" "api" {
  enabled = true
  
  origin {
    domain_name = aws_lb.main.dns_name
    origin_id   = "rag-engine-alb"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }
  
  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD", "OPTIONS"]
    target_origin_id = "rag-engine-alb"
    
    forwarded_values {
      query_string = true
      headers      = ["Authorization", "Origin"]
      
      cookies {
        forward = "none"
      }
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 0
    max_ttl                = 86400
    compress               = true
  }
  
  # Cache health checks
  ordered_cache_behavior {
    path_pattern     = "/health"
    allowed_methods  = ["GET", "HEAD"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "rag-engine-alb"
    
    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
    
    min_ttl     = 5
    default_ttl = 5
    max_ttl     = 5
  }
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    cloudfront_default_certificate = false
    acm_certificate_arn            = aws_acm_certificate.main.arn
    ssl_support_method             = "sni-only"
    minimum_protocol_version       = "TLSv1.2_2021"
  }
}
```

---

## Part 7: Continuous Performance Testing

### CI/CD Integration

**GitHub Actions Workflow:**
```yaml
# .github/workflows/performance-test.yml
name: Performance Tests

on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to test'
        required: true
        default: 'staging'

jobs:
  performance-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup k6
      uses: grafana/setup-k6-action@v1
    
    - name: Run Performance Tests
      run: |
        k6 run \
          --out json=results.json \
          --env API_URL=${{ github.event.inputs.environment == 'production' && 'https://api.rag-engine.com' || 'https://staging-api.rag-engine.com' }} \
          performance-test.js
    
    - name: Analyze Results
      run: |
        python analyze_performance.py results.json
    
    - name: Upload Results
      uses: actions/upload-artifact@v3
      with:
        name: performance-results
        path: |
          results.json
          performance-report.png
    
    - name: Comment PR
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const results = JSON.parse(fs.readFileSync('results.json', 'utf8'));
          
          const p95 = results.metrics.http_req_duration.values['p(95)'];
          const errorRate = results.metrics.http_req_failed.values.rate;
          
          const body = `## Performance Test Results
          - P95 Response Time: ${p95.toFixed(2)}ms
          - Error Rate: ${(errorRate * 100).toFixed(2)}%
          - Status: ${p95 < 500 && errorRate < 0.01 ? '✅ PASSED' : '❌ FAILED'}
          `;
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: body
          });
    
    - name: Fail on Performance Regression
      run: |
        python check_performance_regression.py results.json
```

### Performance Budgets

```javascript
// performance-budget.js
export const options = {
  thresholds: {
    // Response time budgets
    'http_req_duration': [
      {
        threshold: 'p(95)<500',  // 500ms budget
        abortOnFail: true,
        delayAbortEval: '5s',
      },
    ],
    
    // Error rate budget
    'http_req_failed': [
      {
        threshold: 'rate<0.01',  // 1% budget
        abortOnFail: true,
      },
    ],
    
    // Throughput budget (minimum)
    'http_reqs': [
      {
        threshold: 'count>10000',  // Must handle 10k requests
        abortOnFail: true,
      },
    ],
  },
};
```

---

## Summary

You now have comprehensive performance testing capabilities:

✅ **Testing Tools**: Locust, k6, Artillery for various use cases  
✅ **Cloud Testing**: AWS, GCP, Azure specific testing approaches  
✅ **Test Types**: Load, stress, spike, endurance, scalability  
✅ **Analysis**: Metrics calculation, visualization, reporting  
✅ **Capacity Planning**: Growth projection, right-sizing  
✅ **Optimization**: Database, caching, CDN strategies  
✅ **CI/CD Integration**: Automated performance testing  

### Quick Reference

**Start Load Test:**
```bash
locust -f locustfile.py --host http://api.rag-engine.com
```

**Run Stress Test:**
```bash
k6 run stress-test.js
```

**Analyze Results:**
```bash
python analyze_results.py results.json
```

**Plan Capacity:**
```bash
python capacity_planner.py
```

**Remember**: Performance testing is not a one-time activity. Run tests:
- After major deployments
- Weekly (automated)
- Before capacity increases
- During incident investigation

**Happy Testing! 🚀**
