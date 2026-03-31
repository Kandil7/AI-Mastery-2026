# Setting Up Your AI/ML Database Development Environment

## Executive Summary

This comprehensive tutorial provides step-by-step instructions for setting up a production-ready development environment for AI/ML database work. Designed for senior AI/ML engineers, this guide covers local, cloud, and hybrid development environments with practical configuration examples.

**Key Features**:
- Complete development environment setup guide
- Local, cloud, and hybrid configuration options
- Integration with popular AI/ML frameworks
- Security and compliance considerations
- Performance optimization for development

## Environment Setup Options

### Option 1: Local Development (Recommended for Learning)

**System Requirements**:
- CPU: 8+ cores (for ML workloads)
- RAM: 32GB+ (64GB recommended for large models)
- Storage: 1TB SSD (NVMe preferred)
- OS: Linux (Ubuntu 22.04) or Windows 11 with WSL2

**Docker-Based Setup**:
```bash
# Install Docker Desktop (Windows/Mac) or Docker Engine (Linux)
# Verify installation
docker --version
docker-compose --version

# Create project directory
mkdir -p ~/ai-db-dev && cd ~/ai-db-dev

# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  # PostgreSQL for relational data
  postgres:
    image: postgres:14
    environment:
      POSTGRES_USER: ai_dev
      POSTGRES_PASSWORD: ai_password
      POSTGRES_DB: ai_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ai_dev"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Milvus for vector storage
  milvus:
    image: milvusdb/milvus:v2.3.0
    ports:
      - "19530:19530"
      - "9091:9091"
    volumes:
      - milvus_data:/var/lib/milvus
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin

  # MinIO for object storage
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data

  # Redis for caching
  redis:
    image: redis:7
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # Jupyter for interactive development
  jupyter:
    image: jupyter/datascience-notebook:latest
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work
      - ./data:/home/jovyan/data
    environment:
      JUPYTER_TOKEN: ai_dev_token
      GRANT_SUDO: "yes"
      NB_UID: "1000"
      NB_GID: "1000"

volumes:
  postgres_data:
  milvus_data:
  minio_data:
  redis_data:
EOF

# Start the environment
docker-compose up -d

# Wait for services to be ready
docker-compose ps
EOF
```

### Option 2: Cloud Development (AWS/GCP/Azure)

**AWS CloudFormation Template**:
```yaml
# aws-dev-environment.yaml
Resources:
  PostgresDB:
    Type: AWS::RDS::DBInstance
    Properties:
      DBInstanceClass: db.m6g.2xlarge
      Engine: postgres
      MasterUserPassword: !Ref DBPassword
      MasterUsername: ai_dev
      AllocatedStorage: 100
      BackupRetentionPeriod: 1
      PubliclyAccessible: false
      VPCSecurityGroups:
        - !Ref DBSecurityGroup

  MilvusCluster:
    Type: Custom::MilvusCluster
    Properties:
      ServiceToken: !GetAtt MilvusLambda.Arn
      InstanceType: m6g.4xlarge
      NodeCount: 3
      StorageSize: 500

  RedisCluster:
    Type: AWS::ElastiCache::ReplicationGroup
    Properties:
      ReplicationGroupDescription: AI/ML Redis Cluster
      NodeType: cache.r6g.2xlarge
      NumNodeGroups: 3
      ReplicasPerNodeGroup: 2
      AutomaticFailoverEnabled: true
```

### Option 3: Hybrid Development (Local + Cloud)

**Local Development with Cloud Backend**:
```python
# config/local_dev.py
DATABASE_URL = "postgresql://ai_dev:ai_password@localhost:5432/ai_db"
VECTOR_DB_URL = "http://localhost:19530"
REDIS_URL = "redis://localhost:6379"
MINIO_URL = "http://localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"

# config/cloud_prod.py  
DATABASE_URL = "postgresql://ai_prod:prod_password@prod-rds.amazonaws.com:5432/ai_prod"
VECTOR_DB_URL = "https://milvus-prod.us-west-2.amazonaws.com:19530"
REDIS_URL = "redis://redis-cluster-prod.us-west-2.amazonaws.com:6379"
MINIO_URL = "https://s3.amazonaws.com"
MINIO_ACCESS_KEY = "cloud_access_key"
MINIO_SECRET_KEY = "cloud_secret_key"

# Environment selection
import os
ENV = os.getenv('APP_ENV', 'local')
if ENV == 'local':
    from config.local_dev import *
else:
    from config.cloud_prod import *
```

## AI/ML Framework Integration

### LangChain Integration
```python
from langchain.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Configure vector store
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vectorstore = Milvus(
    embeddings,
    connection_args={"host": "localhost", "port": "19530"},
    collection_name="ai_documents"
)

# Configure LLM
llm = OpenAI(temperature=0.7, model_name="gpt-4")

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Test the setup
result = qa_chain({"query": "What are the key features of this AI system?"})
print(result["result"])
```

### PyTorch/TensorFlow Integration
```python
import torch
import psycopg2
from torch.utils.data import Dataset, DataLoader

class DatabaseDataset(Dataset):
    def __init__(self, db_config, query):
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()
        self.cursor.execute(query)
        self.data = self.cursor.fetchall()
        self.columns = [desc[0] for desc in self.cursor.description]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        # Convert to tensors
        features = torch.tensor(row[:-1], dtype=torch.float32)
        label = torch.tensor(row[-1], dtype=torch.long)
        return features, label

# Usage
db_config = {
    'host': 'localhost',
    'database': 'ai_db',
    'user': 'ai_dev',
    'password': 'ai_password'
}

dataset = DatabaseDataset(db_config, """
    SELECT feature1, feature2, feature3, target 
    FROM training_data 
    WHERE split = 'train'
""")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Security and Compliance Setup

### Development Security Best Practices
```bash
# 1. Use separate development credentials
# Never use production credentials in development

# 2. Enable TLS for local connections (development only)
# In postgresql.conf:
ssl = on
ssl_cert_file = '/etc/ssl/certs/dev.crt'
ssl_key_file = '/etc/ssl/private/dev.key'

# 3. Set up basic authentication
# Create development role
CREATE ROLE ai_dev WITH LOGIN PASSWORD 'dev_password' NOSUPERUSER;
GRANT CONNECT ON DATABASE ai_db TO ai_dev;
GRANT USAGE ON SCHEMA public TO ai_dev;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ai_dev;

# 4. Enable audit logging for development
ALTER SYSTEM SET log_statement = 'ddl';
ALTER SYSTEM SET log_min_duration_statement = 1000;
```

### Environment Variables Management
```python
# .env file structure
# NEVER commit .env files to version control
DATABASE_URL=postgresql://ai_dev:ai_password@localhost:5432/ai_db
VECTOR_DB_URL=http://localhost:19530
REDIS_URL=redis://localhost:6379
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AWS_ACCESS_KEY_ID=AKIAxxxxxxxxxxxxxxxx
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Python configuration loader
from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    DATABASE_URL = os.getenv('DATABASE_URL')
    VECTOR_DB_URL = os.getenv('VECTOR_DB_URL')
    REDIS_URL = os.getenv('REDIS_URL')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    @staticmethod
    def validate():
        required = ['DATABASE_URL', 'OPENAI_API_KEY']
        missing = [var for var in required if not getattr(Config, var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
```

## Performance Optimization for Development

### Local Performance Tuning
```sql
-- PostgreSQL development tuning (postgresql.conf)
shared_buffers = 4GB
work_mem = 64MB
maintenance_work_mem = 1GB
effective_cache_size = 12GB
random_page_cost = 1.1
effective_io_concurrency = 200
max_worker_processes = 8
max_parallel_workers_per_gather = 4
```

### Resource Management
```bash
# Monitor resource usage
# CPU usage
top -p $(pgrep postgres),$(pgrep milvus),$(pgrep redis)

# Memory usage
free -h
docker stats

# Disk I/O
iostat -x 1

# Network usage
iftop -i eth0
```

## Troubleshooting Common Issues

### Connection Issues
- **Problem**: "Connection refused" to PostgreSQL
- **Solution**: Check if container is running: `docker-compose ps`, verify port mapping

### Performance Issues
- **Problem**: Slow queries in development
- **Solution**: Add indexes, increase work_mem, use EXPLAIN ANALYZE

### Authentication Issues
- **Problem**: "FATAL: password authentication failed"
- **Solution**: Verify credentials in .env file, check PostgreSQL pg_hba.conf

### Dependency Issues
- **Problem**: Module not found errors
- **Solution**: Verify virtual environment, reinstall packages with `pip install --upgrade --force-reinstall`

## Next Steps

1. **Complete the setup**: Follow the instructions above for your preferred environment
2. **Run the hello-world example**: Test your setup with a simple query
3. **Explore the tutorials**: Move to `06_tutorials/02_ai_ml_integration/` for AI/ML-specific tutorials
4. **Join the community**: Participate in the AI-Mastery Discord for help

## Conclusion

This development environment setup guide provides everything you need to get started with AI/ML database development. The key is to start with a simple local setup, get familiar with the tools, and gradually move to more complex configurations as your needs evolve.

Remember: A well-configured development environment saves hours of debugging and allows you to focus on building great AI/ML applications.