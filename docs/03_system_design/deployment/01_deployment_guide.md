# SDK & Deployment: Complete Guide

## Table of Contents
1. [Python SDK](#python-sdk)
2. [JavaScript SDK](#javascript-sdk)
3. [Deployment to AWS ECS](#deployment-to-aws-ecs)
4. [Deployment to GCP Cloud Run](#deployment-to-gcp-cloud-run)
5. [Deployment to Azure ACI](#deployment-to-azure-aci)
6. [Deployment to Kubernetes](#deployment-to-kubernetes)

---

## Python SDK

### Installation

```bash
# Install from PyPI
pip install rag-engine

# Install from source
git clone https://github.com/user/rag-engine.git
cd rag-engine
pip install -e .
```

### Quick Start

```python
from rag_engine import RAGClient

# Initialize client
client = RAGClient(
    api_key="sk_your_api_key_here",
    base_url="http://localhost:8000",
)

# Ask a question
answer = client.ask("What is RAG?", k=5)
print(f"Answer: {answer.text}")
print(f"Sources: {answer.sources}")

# Upload a document
doc = client.upload_document("test.pdf")
print(f"Document ID: {doc.id}")

# Search documents
results = client.search_documents("vector search")
print(f"Found {len(results)} documents")

# Get query history
history = client.get_query_history()
print(f"History: {len(history)} queries")

# Async usage
import asyncio

async def async_example():
    answer = await client.ask_async("How does RAG work?", k=10)
    print(f"Answer: {answer.text}")

asyncio.run(async_example())
```

### Features

| Feature | Method | Async |
|---------|--------|-------|
| **Ask Question** | `ask()` | ✅ `ask_async()` |
| **Upload Document** | `upload_document()` | ✅ `upload_document_async()` |
| **Search Documents** | `search_documents()` | ❌ |
| **Delete Document** | `delete_document()` | ❌ |
| **Query History** | `get_query_history()` | ❌ |
| **Bulk Upload** | `bulk_upload()` | ✅ `bulk_upload_async()` |

### Error Handling

```python
from rag_engine import RAGClient
from rag_engine.errors import (
    APIError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
)

client = RAGClient(api_key="sk_...")

try:
    answer = client.ask("What is RAG?")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded. Retry later.")
except ValidationError as e:
    print(f"Invalid input: {e.message}")
except APIError as e:
    print(f"API error: {e.status_code} - {e.message}")
```

---

## JavaScript SDK

### Installation

```bash
# Install from npm
npm install @rag-engine/sdk

# Install from source
git clone https://github.com/user/rag-engine.git
cd rag-engine/sdk/javascript
npm install
```

### Quick Start

```javascript
import { RAGClient } from "@rag-engine/sdk";

// Initialize client
const client = new RAGClient({
  apiKey: "sk_your_api_key_here",
  baseUrl: "http://localhost:8000",
});

// Ask a question
const answer = await client.ask("What is RAG?", { k: 5 });
console.log("Answer:", answer.text);
console.log("Sources:", answer.sources);

// Upload a document
const doc = await client.uploadDocument("./test.pdf");
console.log("Document ID:", doc.id);

// Search documents
const results = await client.searchDocuments("vector search");
console.log("Results:", results.length);

// Get query history
const history = await client.getQueryHistory();
console.log("History:", history.length);
```

### React Hook

```javascript
import { RAGClient, useRAGClient } from "@rag-engine/sdk";

function RAGChat() {
  const client = useRAGClient({
    apiKey: "sk_your_api_key_here",
  });

  const [question, setQuestion] = React.useState("");
  const [answer, setAnswer] = React.useState(null);
  const [loading, setLoading] = React.useState(false);

  const handleAsk = async () => {
    setLoading(true);
    try {
      const result = await client.ask(question, { k: 5 });
      setAnswer(result);
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask a question..."
      />
      <button onClick={handleAsk} disabled={loading}>
        {loading ? "Loading..." : "Ask"}
      </button>
      {answer && (
        <div>
          <h2>Answer</h2>
          <p>{answer.text}</p>
          <h3>Sources</h3>
          <ul>
            {answer.sources.map((source) => (
              <li key={source}>{source}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default RAGChat;
```

### TypeScript Support

```typescript
import { RAGClient, Answer, Document } from "@rag-engine/sdk";

// Full TypeScript support
const client = new RAGClient({
  apiKey: "sk_your_api_key_here",
});

async function typedExample() {
  const answer: Answer = await client.ask("What is RAG?");
  const results: Document[] = await client.searchDocuments("query");
  
  console.log("Answer:", answer.text);
  console.log("Sources:", answer.sources);
}
```

---

## Deployment to AWS ECS

### Prerequisites

```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure
# AWS Access Key ID: [your-access-key]
# AWS Secret Access Key: [your-secret-key]
# Default region name: us-east-1
```

### ECR Repository

```bash
# Create repository
aws ecr create-repository --repository-name rag-engine

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

# Tag image
docker tag rag-engine:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/rag-engine:latest

# Push image
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/rag-engine:latest
```

### Task Definition

```json
{
  "family": "rag-engine",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "containerDefinitions": [
    {
      "name": "rag-engine",
      "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/rag-engine:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "valueFrom": {
            "secretKeyRef": {
              "name": "rag-engine-secrets",
              "key": "database-url"
            }
          }
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rag-engine",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Service Definition

```bash
# Create service
aws ecs create-service \
  --cluster rag-engine-cluster \
  --service-name rag-engine \
  --task-definition rag-engine \
  --desired-count 3 \
  --launch-type FARGATE \
  --platform-version LATEST \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-123,subnet-456],securityGroups=[sg-789]}" \
  --load-balancers targetGroupArn=arn:aws:elasticloadbalancing:us-east-1:123456789012:targetgroup/rag-engine/tg-123,containerName=rag-engine,containerPort=8000

# Update service
aws ecs update-service \
  --cluster rag-engine-cluster \
  --service rag-engine \
  --force-new-deployment

# Delete service
aws ecs delete-service \
  --cluster rag-engine-cluster \
  --service rag-engine \
  --force
```

---

## Deployment to GCP Cloud Run

### Prerequisites

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash

# Initialize gcloud
gcloud init

# Authenticate
gcloud auth login
```

### Build and Push

```bash
# Configure Docker for GCP
gcloud auth configure-docker

# Tag image
docker tag rag-engine:latest gcr.io/my-project/rag-engine:latest

# Push image
docker push gcr.io/my-project/rag-engine:latest
```

### Deploy Service

```bash
# Deploy to Cloud Run
gcloud run deploy rag-engine \
  --image gcr.io/my-project/rag-engine:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --max-instances 10 \
  --min-instances 2 \
  --memory 512Mi \
  --cpu 1 \
  --port 8000 \
  --set-env-vars \
    DATABASE_URL=postgresql://..., \
    REDIS_URL=redis://..., \
    OPENAI_API_KEY=... \
  --set-secrets \
    API_KEY=rag-engine-secret:api-key
```

### Update Service

```bash
# Update with new image
gcloud run deploy rag-engine \
  --image gcr.io/my-project/rag-engine:v2.0.0

# Update environment variables
gcloud run services update rag-engine \
  --update-env-vars \
    NEW_VAR=value

# Set secrets
gcloud secrets versions access rag-engine-secret --latest > secret.txt
gcloud run deploy rag-engine \
  --set-secrets-from-file secret.txt
```

### View Logs

```bash
# Stream logs
gcloud logging tail /logs/run.googleapis.com/projects/my-project \
  --filter 'resource.type="cloud_run_revision"'
```

---

## Deployment to Azure ACI

### Prerequisites

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCli | bash

# Login
az login

# Set subscription
az account set --subscription "your-subscription-id"
```

### Create Container Instance

```bash
# Create resource group
az group create --name rag-engine-rg --location eastus

# Create container registry
az acr create --resource-group rag-engine-rg --name ragengineregistry --sku Basic

# Login to ACR
az acr login --name ragengineregistry

# Tag and push
docker tag rag-engine:latest ragengineregistry.azurecr.io/rag-engine:latest
docker push ragengineregistry.azurecr.io/rag-engine:latest

# Deploy to ACI
az container create \
  --resource-group rag-engine-rg \
  --name rag-engine \
  --image ragengineregistry.azurecr.io/rag-engine:latest \
  --dns-name-label rag-engine \
  --ports 8000 \
  --cpu 1 \
  --memory 1 \
  --restart-policy Always \
  --environment-variables \
    DATABASE_URL=postgresql://..., \
    REDIS_URL=redis://..., \
    OPENAI_API_KEY=... \
  --secure-environment-variables \
    API_KEY=...
```

### Scale Instance

```bash
# Update resources
az container update \
  --resource-group rag-engine-rg \
  --name rag-engine \
  --cpu 2 \
  --memory 2

# Stop instance
az container stop --name rag-engine --resource-group rag-engine-rg

# Start instance
az container start --name rag-engine --resource-group rag-engine-rg

# Delete instance
az container delete --name rag-engine --resource-group rag-engine-rg
```

---

## Deployment to Kubernetes

### Create Namespace

```bash
kubectl create namespace rag-engine
```

### Create Secrets

```bash
kubectl create secret generic rag-engine-secrets \
  --from-literal=database-url="postgresql://..." \
  --from-literal=redis-url="redis://..." \
  --from-literal=openai-api-key="..." \
  --namespace=rag-engine
```

### Apply Deployment

```bash
# Apply all resources
kubectl apply -f config/kubernetes/ -n rag-engine

# Apply specific resource
kubectl apply -f config/kubernetes/deployment.yaml -n rag-engine

# Apply ConfigMap
kubectl apply -f config/kubernetes/configmap.yaml -n rag-engine
```

### Scale Deployment

```bash
# Manual scaling
kubectl scale deployment rag-engine --replicas=5 -n rag-engine

# Auto-scaling (HPA)
kubectl apply -f config/kubernetes/hpa.yaml -n rag-engine
```

### Rollout Status

```bash
# Check rollout status
kubectl rollout status deployment/rag-engine -n rag-engine

# Rollback if needed
kubectl rollout undo deployment/rag-engine -n rag-engine

# Rollback to specific revision
kubectl rollout undo deployment/rag-engine --to-revision=3 -n rag-engine
```

### Port Forwarding

```bash
# Forward local port to pod
kubectl port-forward deployment/rag-engine 8000:8000 -n rag-engine

# Access service
kubectl port-forward service/rag-engine-service 8000:80 -n rag-engine
```

---

## Summary

| Platform | Tool | Scale Type |
|---------|------|------------|
| **AWS ECS** | AWS CLI, Docker | Fargate, EC2 |
| **GCP Cloud Run** | gcloud CLI | Managed, autoscaling |
| **Azure ACI** | az CLI | Container instances |
| **Kubernetes** | kubectl, Helm | HPA, manual |

---

## Further Reading

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [GCP Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Azure ACI Documentation](https://docs.microsoft.com/azure/container-instances/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- `sdk/python/rag_engine.py` - Python SDK
- `sdk/javascript/rag-engine.js` - JavaScript SDK
