# Deployment

This section provides detailed instructions for deploying the Production RAG System in various environments, including local development, containerized deployments, and cloud platforms.

## Prerequisites

Before deploying the Production RAG System, ensure you have the following prerequisites:

### System Requirements
- **Operating System**: Linux, macOS, or Windows
- **Python**: Version 3.10 or higher
- **Docker**: Version 20.10 or higher (for containerized deployments)
- **Docker Compose**: Version 2.0 or higher (for multi-container orchestration)
- **Git**: Version 2.0 or higher

### Hardware Requirements
- **CPU**: Multi-core processor (recommended: 4+ cores)
- **RAM**: Minimum 8GB (recommended: 16GB+ for optimal performance)
- **Storage**: Minimum 10GB free space
- **Network**: Stable internet connection for downloading dependencies

## Local Development Deployment

### 1. Clone the Repository
```bash
git clone https://github.com/your-organization/ai-mastery-2026.git
cd ai-mastery-2026/sprints/week01_rag_production
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root with the following content:
```
ENVIRONMENT=development
DEBUG=true
DATABASE__URL=mongodb://localhost:27017
DATABASE__NAME=rag_db
MODELS__GENERATOR_MODEL=gpt2
MODELS__DENSE_MODEL=all-MiniLM-L6-v2
RETRIEVAL__ALPHA=0.7
RETRIEVAL__FUSION_METHOD=rrf
```

### 5. Start MongoDB (if running locally)
```bash
# Option 1: Using Docker
docker run -d -p 27017:27017 --name rag-mongo mongo:6.0

# Option 2: Using system MongoDB (if installed)
sudo systemctl start mongod
```

### 6. Run the Application
```bash
# Run the API server
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Or run using the Makefile
make run-api
```

### 7. Access the Application
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Containerized Deployment

### 1. Build and Run with Docker Compose
```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up --build -d
```

### 2. Docker Compose Services
The `docker-compose.yml` file defines the following services:

#### `rag-api` Service
- **Image**: Built from the local Dockerfile
- **Ports**: Exposes port 8000
- **Environment Variables**:
  - `ENVIRONMENT=production`
  - `DATABASE__URL=mongodb://rag_user:rag_password@mongo:27017/admin`
  - `DATABASE__NAME=rag_db`
- **Volumes**:
  - `./uploads:/app/uploads` - For file uploads
  - `./data:/app/data` - For persistent data
- **Dependencies**: Depends on the `mongo` service
- **Health Check**: Checks `/health` endpoint every 30 seconds

#### `mongo` Service
- **Image**: `mongo:6.0`
- **Ports**: Exposes port 27017
- **Environment Variables**:
  - `MONGO_INITDB_ROOT_USERNAME=rag_user`
  - `MONGO_INITDB_ROOT_PASSWORD=rag_password`
- **Volumes**: Persists data using the `mongo_data` volume

#### `nginx` Service (Optional)
- **Image**: `nginx:alpine`
- **Ports**: Exposes ports 80 and 443
- **Volumes**:
  - `./nginx.conf:/etc/nginx/nginx.conf`
  - `./ssl:/etc/nginx/ssl`
- **Dependencies**: Depends on the `rag-api` service

### 3. Managing Docker Compose Services
```bash
# View service logs
docker-compose logs -f

# Stop services
docker-compose down

# Scale API service
docker-compose up --scale rag-api=3

# View service status
docker-compose ps
```

## Dockerfile Configuration

The `Dockerfile` is configured as follows:

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy project requirements
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app/

# Create uploads directory
RUN mkdir -p uploads data/chroma data/vector_store
```

## Production Deployment Considerations

### 1. Environment Configuration
For production deployments, use the following environment variables:

```
ENVIRONMENT=production
DEBUG=false
LOGGING__LEVEL=INFO
SECURITY__ENABLE_AUTHENTICATION=true
API__RATE_LIMIT_REQUESTS=1000
API__RATE_LIMIT_WINDOW=60
DATABASE__POOL_SIZE=20
DATABASE__MAX_OVERFLOW=40
MODELS__MAX_NEW_TOKENS=500
RETRIEVAL__TOP_K=5
```

### 2. Security Hardening
- Enable authentication and authorization
- Use HTTPS with valid SSL certificates
- Implement rate limiting
- Validate and sanitize all inputs
- Use secure connection pooling
- Regular security updates

### 3. Resource Management
- Configure appropriate memory limits
- Set CPU quotas based on workload
- Implement graceful degradation
- Monitor resource usage
- Plan for scaling

### 4. Backup and Recovery
- Regular database backups
- Version control for configuration
- Disaster recovery plan
- Automated backup schedules
- Off-site backup storage

## Cloud Platform Deployment

### AWS Deployment
1. **EC2 Instance Setup**:
   - Choose appropriate instance type (e.g., t3.medium or larger)
   - Configure security groups for port access
   - Set up IAM roles for services

2. **Container Orchestration**:
   - Use ECS or EKS for container management
   - Configure load balancers
   - Set up auto-scaling policies

3. **Database**:
   - Use Amazon DocumentDB for MongoDB compatibility
   - Configure backup and maintenance windows
   - Set up read replicas if needed

### Google Cloud Platform
1. **Compute Engine**:
   - Select appropriate machine type
   - Configure firewall rules
   - Set up service accounts

2. **Container Solutions**:
   - Use Google Kubernetes Engine (GKE)
   - Configure Cloud Load Balancing
   - Set up horizontal pod autoscaling

3. **Database**:
   - Use MongoDB Atlas or Cloud SQL
   - Configure backup and recovery options

### Azure Deployment
1. **Virtual Machines**:
   - Choose appropriate VM size
   - Configure network security groups
   - Set up managed identities

2. **Container Services**:
   - Use Azure Kubernetes Service (AKS)
   - Configure Application Gateway
   - Set up auto-scaling rules

3. **Database**:
   - Use Azure Cosmos DB for MongoDB API
   - Configure backup and disaster recovery

## Monitoring and Maintenance

### 1. Health Checks
The system includes health check endpoints:
- `/health` - Detailed health status
- `/metrics` - Prometheus-compatible metrics

### 2. Logging
- Structured logging with correlation IDs
- Log aggregation and analysis
- Alerting for critical issues
- Log retention policies

### 3. Performance Monitoring
- Request latency tracking
- Error rate monitoring
- Resource utilization
- Database performance metrics

### 4. Maintenance Tasks
- Regular dependency updates
- Database optimization
- Log rotation
- Security audits
- Performance tuning

## Troubleshooting Common Issues

### 1. Database Connection Issues
- Verify MongoDB is running and accessible
- Check connection URL and credentials
- Ensure environment variables are set correctly
- Review network connectivity

### 2. Model Loading Issues
- Verify sufficient memory allocation
- Check model names and availability
- Review model download permissions
- Monitor disk space for cached models

### 3. Performance Issues
- Monitor resource utilization
- Check for memory leaks
- Review indexing strategies
- Optimize query patterns

### 4. Container Issues
- Check container logs for errors
- Verify resource limits
- Review network configurations
- Ensure proper volume mounting