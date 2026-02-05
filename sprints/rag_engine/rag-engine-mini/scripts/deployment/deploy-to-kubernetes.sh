#!/bin/bash
# deploy-to-kubernetes.sh
# Deploy RAG Engine to Kubernetes cluster
# Usage: ./deploy-to-kubernetes.sh [environment] [namespace]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
ENVIRONMENT=${1:-production}
NAMESPACE=${2:-rag-engine}
APP_NAME="rag-engine"
IMAGE_TAG=${3:-latest}

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   RAG Engine - Kubernetes Deployment                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Step 1: Checking prerequisites...${NC}"

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ kubectl is not installed${NC}"
    exit 1
fi

if ! command -v helm &> /dev/null; then
    echo -e "${YELLOW}⚠️  Helm is not installed (optional but recommended)${NC}"
fi

if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}❌ Cannot connect to Kubernetes cluster${NC}"
    echo "Please ensure:"
    echo "  1. You have a cluster running (minikube, EKS, GKE, etc.)"
    echo "  2. kubectl is configured correctly"
    echo "  3. Run: kubectl config current-context"
    exit 1
fi

echo -e "${GREEN}✅ Connected to Kubernetes cluster${NC}"

# Create namespace
echo -e "${YELLOW}Step 2: Creating namespace...${NC}"
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
echo -e "${GREEN}✅ Namespace ${NAMESPACE} ready${NC}"

# Generate secrets
echo -e "${YELLOW}Step 3: Generating secrets...${NC}"

JWT_SECRET=$(openssl rand -base64 32)
DB_PASSWORD=$(openssl rand -base64 24)
REDIS_PASSWORD=$(openssl rand -base64 24)
API_KEY=$(openssl rand -base64 32)

# Create secrets manifest
cat > /tmp/rag-engine-secrets.yaml <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: ${APP_NAME}-secrets
  namespace: ${NAMESPACE}
type: Opaque
stringData:
  JWT_SECRET: "${JWT_SECRET}"
  DB_PASSWORD: "${DB_PASSWORD}"
  REDIS_PASSWORD: "${REDIS_PASSWORD}"
  API_KEY: "${API_KEY}"
EOF

kubectl apply -f /tmp/rag-engine-secrets.yaml
rm /tmp/rag-engine-secrets.yaml
echo -e "${GREEN}✅ Secrets created${NC}"

# Create configmap
echo -e "${YELLOW}Step 4: Creating ConfigMap...${NC}"
cat > /tmp/rag-engine-config.yaml <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: ${APP_NAME}-config
  namespace: ${NAMESPACE}
data:
  ENVIRONMENT: "${ENVIRONMENT}"
  DEBUG: "false"
  LOG_LEVEL: "INFO"
  DB_HOST: "postgres"
  DB_PORT: "5432"
  DB_NAME: "rag_engine"
  DB_USER: "rag_user"
  REDIS_HOST: "redis"
  REDIS_PORT: "6379"
  QDRANT_HOST: "qdrant"
  QDRANT_PORT: "6333"
  WORKERS: "4"
EOF

kubectl apply -f /tmp/rag-engine-config.yaml
rm /tmp/rag-engine-config.yaml
echo -e "${GREEN}✅ ConfigMap created${NC}"

# Deploy PostgreSQL
echo -e "${YELLOW}Step 5: Deploying PostgreSQL...${NC}"
cat > /tmp/postgres.yaml <<EOF
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: ${NAMESPACE}
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:14-alpine
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          valueFrom:
            configMapKeyRef:
              name: ${APP_NAME}-config
              key: DB_NAME
        - name: POSTGRES_USER
          valueFrom:
            configMapKeyRef:
              name: ${APP_NAME}-config
              key: DB_USER
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: ${APP_NAME}-secrets
              key: DB_PASSWORD
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: ${NAMESPACE}
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
EOF

kubectl apply -f /tmp/postgres.yaml
rm /tmp/postgres.yaml
echo -e "${GREEN}✅ PostgreSQL deployed${NC}"

# Deploy Redis
echo -e "${YELLOW}Step 6: Deploying Redis...${NC}"
cat > /tmp/redis.yaml <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: ${NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        command:
        - redis-server
        - --requirepass
        - \$(REDIS_PASSWORD)
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: ${APP_NAME}-secrets
              key: REDIS_PASSWORD
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        persistentVolumeClaim:
          claimName: redis-data
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-data
  namespace: ${NAMESPACE}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: ${NAMESPACE}
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
EOF

kubectl apply -f /tmp/redis.yaml
rm /tmp/redis.yaml
echo -e "${GREEN}✅ Redis deployed${NC}"

# Deploy Qdrant
echo -e "${YELLOW}Step 7: Deploying Qdrant...${NC}"
cat > /tmp/qdrant.yaml <<EOF
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
  namespace: ${NAMESPACE}
spec:
  serviceName: qdrant
  replicas: 1
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:latest
        ports:
        - containerPort: 6333
        volumeMounts:
        - name: qdrant-data
          mountPath: /qdrant/storage
  volumeClaimTemplates:
  - metadata:
      name: qdrant-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 20Gi
---
apiVersion: v1
kind: Service
metadata:
  name: qdrant
  namespace: ${NAMESPACE}
spec:
  selector:
    app: qdrant
  ports:
  - port: 6333
    targetPort: 6333
EOF

kubectl apply -f /tmp/qdrant.yaml
rm /tmp/qdrant.yaml
echo -e "${GREEN}✅ Qdrant deployed${NC}"

# Deploy RAG Engine API
echo -e "${YELLOW}Step 8: Deploying RAG Engine API...${NC}"
cat > /tmp/api.yaml <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${APP_NAME}-api
  namespace: ${NAMESPACE}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ${APP_NAME}-api
  template:
    metadata:
      labels:
        app: ${APP_NAME}-api
    spec:
      containers:
      - name: api
        image: ${APP_NAME}:${IMAGE_TAG}
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: ${APP_NAME}-config
        - secretRef:
            name: ${APP_NAME}-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
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
  name: ${APP_NAME}-api
  namespace: ${NAMESPACE}
spec:
  type: ClusterIP
  selector:
    app: ${APP_NAME}-api
  ports:
  - port: 80
    targetPort: 8000
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ${APP_NAME}-ingress
  namespace: ${NAMESPACE}
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "false"
spec:
  rules:
  - host: rag-engine.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ${APP_NAME}-api
            port:
              number: 80
EOF

kubectl apply -f /tmp/api.yaml
rm /tmp/api.yaml
echo -e "${GREEN}✅ RAG Engine API deployed${NC}"

# Deploy HPA (Horizontal Pod Autoscaler)
echo -e "${YELLOW}Step 9: Setting up autoscaling...${NC}"
cat > /tmp/hpa.yaml <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ${APP_NAME}-api-hpa
  namespace: ${NAMESPACE}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ${APP_NAME}-api
  minReplicas: 3
  maxReplicas: 10
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
EOF

kubectl apply -f /tmp/hpa.yaml
rm /tmp/hpa.yaml
echo -e "${GREEN}✅ Autoscaling configured${NC}"

# Wait for deployment
echo -e "${YELLOW}Step 10: Waiting for deployment to complete...${NC}"
kubectl rollout status deployment/${APP_NAME}-api -n ${NAMESPACE} --timeout=300s

# Verify deployment
echo -e "${YELLOW}Step 11: Verifying deployment...${NC}"
echo ""
echo -e "${BLUE}Pod Status:${NC}"
kubectl get pods -n ${NAMESPACE}

echo ""
echo -e "${BLUE}Services:${NC}"
kubectl get services -n ${NAMESPACE}

echo ""
echo -e "${Blue}Ingress:${NC}"
kubectl get ingress -n ${NAMESPACE}

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║      🎉 Kubernetes Deployment Successful! 🎉          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Useful commands:${NC}"
echo -e "  View pods: ${YELLOW}kubectl get pods -n ${NAMESPACE}${NC}"
echo -e "  View logs: ${YELLOW}kubectl logs -f deployment/${APP_NAME}-api -n ${NAMESPACE}${NC}"
echo -e "  Port forward: ${YELLOW}kubectl port-forward svc/${APP_NAME}-api 8000:80 -n ${NAMESPACE}${NC}"
echo -e "  Scale: ${YELLOW}kubectl scale deployment ${APP_NAME}-api --replicas=5 -n ${NAMESPACE}${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Configure ingress for your domain"
echo -e "  2. Set up SSL certificates"
echo -e "  3. Configure monitoring (Prometheus/Grafana)"
echo -e "  4. Set up log aggregation"
echo ""
echo -e "${GREEN}Happy deploying! 🚀${NC}"
