# Advanced Database DevOps Automation

This guide provides comprehensive strategies for implementing advanced database DevOps automation in production AI/ML systems. It covers sophisticated CI/CD patterns, Infrastructure as Code (IaC) best practices, GitOps workflows, and comprehensive testing strategies specifically tailored for modern database architectures including relational, vector, and time-series databases.

## Overview

Advanced database DevOps automation goes beyond basic schema migrations to encompass the full lifecycle of database changes in complex AI/ML systems. This includes managing data migrations, ensuring data integrity during deployments, implementing sophisticated testing strategies, and integrating database operations into end-to-end ML pipelines.

## Advanced CI/CD for Databases

### Schema Migration Strategies

#### Multi-Stage Migration Patterns

For production AI systems, implement a three-stage migration approach:

1. **Forward-compatible changes**: Add columns/indexes without breaking existing applications
2. **Application update**: Deploy application code that uses new schema features
3. **Backward-incompatible cleanup**: Remove deprecated schema elements

```sql
-- Stage 1: Add new column (forward-compatible)
ALTER TABLE model_predictions 
ADD COLUMN confidence_score DOUBLE PRECISION DEFAULT 0.0;

-- Stage 2: Application update uses confidence_score
-- (Application code deployed with new logic)

-- Stage 3: Remove old column after verification
ALTER TABLE model_predictions DROP COLUMN old_confidence;
```

#### Zero-Downtime Migrations

Implement zero-downtime migrations using techniques like shadow tables and dual-write patterns:

```sql
-- Shadow table pattern for large table migrations
CREATE TABLE users_new (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    -- New fields for AI features
    embedding_vector VECTOR(768),
    last_inference_at TIMESTAMPTZ
);

-- Dual-write during transition period
CREATE OR REPLACE FUNCTION sync_users_to_new()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO users_new VALUES (NEW.*);
    ELSIF TG_OP = 'UPDATE' THEN
        UPDATE users_new SET 
            name = NEW.name,
            email = NEW.email,
            embedding_vector = NEW.embedding_vector,
            last_inference_at = NEW.last_inference_at
        WHERE id = NEW.id;
    ELSIF TG_OP = 'DELETE' THEN
        DELETE FROM users_new WHERE id = OLD.id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER users_sync_trigger
AFTER INSERT OR UPDATE OR DELETE ON users
FOR EACH ROW EXECUTE FUNCTION sync_users_to_new();
```

### Data Migration Automation

#### Data Migration Framework

Implement a structured data migration framework with validation and rollback capabilities:

```python
# data_migration_framework.py
from typing import Dict, List, Optional, Callable
import logging
from contextlib import contextmanager

class DataMigration:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps: List[Dict] = []
        self.validation_checks: List[Callable] = []
        
    def add_step(self, step_name: str, operation: Callable, rollback: Optional[Callable] = None):
        """Add a migration step with optional rollback"""
        self.steps.append({
            'name': step_name,
            'operation': operation,
            'rollback': rollback,
            'completed': False
        })
    
    def add_validation(self, check_name: str, validator: Callable):
        """Add validation check"""
        self.validation_checks.append(lambda: validator())
    
    def execute(self, dry_run: bool = False):
        """Execute migration with transactional safety"""
        if dry_run:
            logging.info(f"DRY RUN: Migration {self.name}")
            for step in self.steps:
                logging.info(f"  Would execute: {step['name']}")
            return
        
        try:
            # Begin transaction
            with self._transaction():
                # Execute steps
                for i, step in enumerate(self.steps):
                    logging.info(f"Executing step {i+1}: {step['name']}")
                    step['operation']()
                    step['completed'] = True
                
                # Run validation checks
                for check in self.validation_checks:
                    if not check():
                        raise ValueError(f"Validation failed for migration {self.name}")
                
                logging.info(f"Migration {self.name} completed successfully")
                
        except Exception as e:
            # Rollback on failure
            logging.error(f"Migration {self.name} failed: {e}")
            self._rollback()
            raise
    
    @contextmanager
    def _transaction(self):
        # Database transaction context manager
        yield
    
    def _rollback(self):
        """Rollback completed steps in reverse order"""
        for step in reversed(self.steps):
            if step['completed'] and step['rollback']:
                logging.info(f"Rolling back: {step['name']}")
                step['rollback']()

# Example usage
def migrate_user_embeddings():
    migration = DataMigration(
        "user_embedding_migration",
        "Migrate user data to include embedding vectors"
    )
    
    # Step 1: Add embedding column
    migration.add_step(
        "add_embedding_column",
        lambda: execute_sql("ALTER TABLE users ADD COLUMN embedding_vector VECTOR(768)"),
        lambda: execute_sql("ALTER TABLE users DROP COLUMN embedding_vector")
    )
    
    # Step 2: Generate embeddings for existing users
    migration.add_step(
        "generate_embeddings",
        lambda: generate_user_embeddings_batch(),
        lambda: execute_sql("UPDATE users SET embedding_vector = NULL")
    )
    
    # Validation: Check embedding quality
    migration.add_validation(
        "embedding_quality_check",
        lambda: validate_embedding_quality()
    )
    
    migration.execute()
```

#### CI/CD Pipeline Integration

Integrate data migrations into CI/CD pipelines with proper gating:

```yaml
# .github/workflows/database-cicd.yml
name: Advanced Database CI/CD
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup PostgreSQL
        uses: webfactory/ssh-agent@v0.9.0
        with:
          ssh-private-key: ${{ secrets.DB_SSH_KEY }}
      
      - name: Validate schema compatibility
        run: |
          python scripts/schema_compatibility_check.py \
            --current-schema current_schema.sql \
            --proposed-schema proposed_schema.sql
      
      - name: Run data migration dry-run
        run: python scripts/migration_dry_run.py --migration user_embedding_migration
      
      - name: Execute data quality tests
        run: pytest tests/data_quality_tests.py --strict

  deploy:
    needs: validate
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy schema migrations
        run: |
          flyway -url=jdbc:postgresql://${{ secrets.DB_HOST }}:5432/${{ secrets.DB_NAME }} \
                 -user=${{ secrets.DB_USER }} \
                 -password=${{ secrets.DB_PASSWORD }} \
                 migrate
      
      - name: Execute data migrations
        run: |
          python scripts/run_migrations.py \
            --migrations user_embedding_migration,feature_flags_migration \
            --environment production \
            --dry-run false
      
      - name: Post-deployment validation
        run: |
          python scripts/post_deploy_validation.py \
            --check-data-integrity true \
            --check-performance-regression true \
            --threshold-p95-latency 200ms

  rollback:
    runs-on: ubuntu-latest
    if: always() && failure()
    steps:
      - uses: actions/checkout@v3
      
      - name: Rollback migrations
        run: |
          python scripts/rollback_migrations.py \
            --last-successful-commit ${{ github.sha }}
```

## Infrastructure as Code for Database Provisioning

### Terraform Modules for Modern Database Architectures

#### Multi-Database Architecture Module

Create reusable Terraform modules for different database types:

```hcl
# modules/database/aurora-postgresql/main.tf
variable "cluster_identifier" {
  type        = string
  description = "Cluster identifier"
}

variable "engine_version" {
  type        = string
  default     = "14.7"
  description = "PostgreSQL engine version"
}

variable "master_username" {
  type        = string
  description = "Master username"
}

variable "master_user_password" {
  type        = string
  sensitive   = true
  description = "Master user password"
}

variable "instance_class" {
  type        = string
  default     = "db.r5.large"
  description = "Instance class"
}

variable "storage_size" {
  type        = number
  default     = 100
  description = "Storage size in GB"
}

variable "backup_retention_period" {
  type        = number
  default     = 7
  description = "Backup retention period in days"
}

resource "aws_rds_cluster" "postgres" {
  cluster_identifier      = var.cluster_identifier
  engine                  = "aurora-postgresql"
  engine_version          = var.engine_version
  master_username         = var.master_username
  master_user_password    = var.master_user_password
  backup_retention_period = var.backup_retention_period
  preferred_backup_window = "02:00-03:00"
  db_subnet_group_name    = aws_db_subnet_group.main.name
  vpc_security_group_ids  = [aws_security_group.db.id]
  storage_encrypted       = true
  deletion_protection     = true

  lifecycle {
    ignore_changes = [master_user_password]
  }
}

resource "aws_rds_cluster_instance" "postgres" {
  identifier              = "${var.cluster_identifier}-writer"
  cluster_identifier      = aws_rds_cluster.postgres.id
  instance_class          = var.instance_class
  publicly_accessible     = false
  apply_immediately       = true
}

# Read replica for AI inference workloads
resource "aws_rds_cluster_instance" "postgres_reader" {
  count                 = 2
  identifier            = "${var.cluster_identifier}-reader-${count.index + 1}"
  cluster_identifier    = aws_rds_cluster.postgres.id
  instance_class        = var.instance_class
  publicly_accessible   = false
  apply_immediately     = true
  promotion_tier        = 2
}
```

#### Vector Database Provisioning

```hcl
# modules/database/weaviate/main.tf
variable "cluster_name" {
  type        = string
  description = "Weaviate cluster name"
}

variable "instance_count" {
  type        = number
  default     = 3
  description = "Number of instances"
}

variable "instance_type" {
  type        = string
  default     = "c5.4xlarge"
  description = "EC2 instance type"
}

variable "storage_size" {
  type        = number
  default     = 500
  description = "EBS volume size in GB"
}

resource "aws_ec2_instance" "weaviate" {
  count         = var.instance_count
  ami           = data.aws_ami.weaviate.id
  instance_type = var.instance_type
  
  root_block_device {
    volume_size = var.storage_size
    volume_type = "gp3"
  }
  
  vpc_security_group_ids = [aws_security_group.weaviate.id]
  subnet_id              = aws_subnet.private[count.index % length(data.aws_subnets.private.ids)].id
  
  user_data = <<-EOF
    #!/bin/bash
    yum update -y
    yum install -y docker
    systemctl start docker
    systemctl enable docker
    
    # Pull and run Weaviate
    docker run -d \
      --name weaviate \
      -p 8080:8080 \
      -e WEAVIATE_API_ENABLED=true \
      -e WEAVIATE_AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
      -e WEAVIATE_PERSISTENCE_DATA_PATH=/var/lib/weaviate \
      -v /var/lib/weaviate:/var/lib/weaviate \
      semitechnologies/weaviate:1.22.0
  EOF
  
  tags = {
    Name = "${var.cluster_name}-${count.index + 1}"
  }
}

# Auto-scaling group for horizontal scaling
resource "aws_autoscaling_group" "weaviate" {
  name_prefix = "${var.cluster_name}-asg-"
  min_size    = 3
  max_size    = 10
  desired_capacity = 3
  
  launch_template {
    id      = aws_launch_template.weaviate.id
    version = "$Latest"
  }
  
  vpc_zone_identifier = data.aws_subnets.private.ids
  target_group_arns   = [aws_lb_target_group.weaviate.arn]
}
```

#### Time-Series Database Provisioning

```hcl
# modules/database/influxdb/main.tf
variable "cluster_name" {
  type        = string
  description = "InfluxDB cluster name"
}

variable "replica_count" {
  type        = number
  default     = 3
  description = "Number of replicas"
}

resource "kubernetes_deployment" "influxdb" {
  metadata {
    name      = "${var.cluster_name}-influxdb"
    namespace = "monitoring"
  }

  spec {
    replicas = var.replica_count

    selector {
      match_labels = {
        app = "influxdb"
      }
    }

    template {
      metadata {
        labels = {
          app = "influxdb"
        }
      }

      spec {
        container {
          image = "influxdb:2.7"
          name  = "influxdb"

          port {
            container_port = 8086
          }

          env {
            name  = "INFLUXD_HTTP_BIND_ADDRESS"
            value = ":8086"
          }

          env {
            name  = "INFLUXD_REPORTING_DISABLED"
            value = "true"
          }

          volume_mount {
            name       = "influxdb-storage"
            mount_path = "/var/lib/influxdb2"
          }
        }

        volume {
          name = "influxdb-storage"
          persistent_volume_claim {
            claim_name = "influxdb-pvc"
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "influxdb" {
  metadata {
    name      = "${var.cluster_name}-influxdb"
    namespace = "monitoring"
  }

  spec {
    selector = {
      app = "influxdb"
    }

    port {
      port        = 8086
      target_port = 8086
    }

    type = "ClusterIP"
  }
}
```

### GitOps Workflows for Database Changes

#### Argo CD for Database Management

Implement GitOps workflows using Argo CD for database provisioning and configuration:

```yaml
# manifests/database/argocd-app.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ai-database-cluster
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/ai-mastery-2026.git
    targetRevision: HEAD
    path: infrastructure/database/prod
  destination:
    server: https://kubernetes.default.svc
    namespace: database
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
      - ApplyOutOfSyncOnly=true
```

#### Database Configuration as Code

Store database configurations in version-controlled files:

```yaml
# infrastructure/database/config/postgres-config.yaml
database:
  name: ai_mastery_prod
  version: "14.7"
  parameters:
    shared_buffers: "2GB"
    work_mem: "16MB"
    maintenance_work_mem: "256MB"
    effective_cache_size: "6GB"
    random_page_cost: "1.1"
    checkpoint_completion_target: "0.9"
    max_connections: "200"
    autovacuum_max_workers: "6"
    autovacuum_vacuum_scale_factor: "0.1"
    autovacuum_analyze_scale_factor: "0.05"

  extensions:
    - pgvector
    - timescaledb
    - postgis
    - citext

  monitoring:
    prometheus_exporter:
      enabled: true
      port: 9187
      scrape_interval: "15s"

  security:
    ssl_mode: "require"
    password_encryption: "scram-sha-256"
    log_statement: "ddl"
    log_min_duration_statement: "1000"
```

#### Automated Configuration Validation

```python
# scripts/config_validator.py
import yaml
import json
import sys
from typing import Dict, List

def validate_postgres_config(config: Dict) -> List[str]:
    """Validate PostgreSQL configuration against best practices"""
    errors = []
    
    # Memory configuration validation
    shared_buffers = config.get('parameters', {}).get('shared_buffers', '128MB')
    if not shared_buffers.endswith('GB') and not shared_buffers.endswith('MB'):
        errors.append(f"shared_buffers should be in GB/MB format: {shared_buffers}")
    
    # Connection pool validation
    max_connections = config.get('parameters', {}).get('max_connections', 100)
    if max_connections > 300:
        errors.append(f"max_connections too high: {max_connections}. Consider connection pooling.")
    
    # Extension validation
    extensions = config.get('extensions', [])
    required_extensions = ['pgvector', 'timescaledb']
    missing_extensions = [ext for ext in required_extensions if ext not in extensions]
    if missing_extensions:
        errors.append(f"Missing required extensions: {missing_extensions}")
    
    # Security validation
    ssl_mode = config.get('security', {}).get('ssl_mode', 'prefer')
    if ssl_mode != 'require':
        errors.append(f"SSL mode should be 'require' for production: {ssl_mode}")
    
    return errors

def main():
    if len(sys.argv) != 2:
        print("Usage: python config_validator.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        errors = validate_postgres_config(config)
        
        if errors:
            print("Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        else:
            print("Configuration validation passed!")
            
    except Exception as e:
        print(f"Error validating configuration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Database Testing in CI Pipelines

### Comprehensive Testing Strategy

#### Multi-Layer Testing Framework

Implement a four-layer testing strategy:

1. **Unit Tests**: Schema validation, constraint testing
2. **Integration Tests**: End-to-end database interactions
3. **Performance Tests**: Load testing and benchmarking
4. **Chaos Tests**: Failure injection and resilience testing

```python
# tests/database/test_framework.py
import pytest
import asyncio
from typing import List, Dict, Any
from contextlib import asynccontextmanager

class DatabaseTestFramework:
    @asynccontextmanager
    async def test_context(self, test_type: str):
        """Create test context with appropriate isolation"""
        if test_type == "unit":
            # Use transaction rollback for unit tests
            conn = await self._get_connection()
            await conn.execute("BEGIN")
            try:
                yield conn
            finally:
                await conn.execute("ROLLBACK")
                await conn.close()
        elif test_type == "integration":
            # Use separate test database
            test_db = await self._create_test_database()
            try:
                conn = await self._get_connection(db=test_db)
                yield conn
            finally:
                await self._drop_test_database(test_db)
        elif test_type == "performance":
            # Use dedicated performance test environment
            conn = await self._get_connection(env="performance")
            yield conn
            await conn.close()

    async def run_unit_tests(self):
        """Run unit tests for database schema and constraints"""
        tests = [
            self._test_schema_validity,
            self._test_constraints,
            self._test_indexes,
            self._test_stored_procedures
        ]
        
        results = []
        for test in tests:
            try:
                await test()
                results.append({"test": test.__name__, "status": "PASS"})
            except Exception as e:
                results.append({"test": test.__name__, "status": "FAIL", "error": str(e)})
        
        return results

    async def _test_schema_validity(self):
        """Test that schema matches expected definition"""
        # Compare actual schema with expected schema
        actual_schema = await self._get_actual_schema()
        expected_schema = self._load_expected_schema()
        
        assert actual_schema == expected_schema, \
            f"Schema mismatch: {actual_schema} != {expected_schema}"

    async def _test_constraints(self):
        """Test database constraints work correctly"""
        # Test foreign key constraints
        with pytest.raises(Exception):
            await self.conn.execute(
                "INSERT INTO orders (user_id, total) VALUES (999999, 100.0)"
            )
        
        # Test unique constraints
        await self.conn.execute(
            "INSERT INTO users (name, email) VALUES ('Test User', 'test@example.com')"
        )
        with pytest.raises(Exception):
            await self.conn.execute(
                "INSERT INTO users (name, email) VALUES ('Another User', 'test@example.com')"
            )
```

#### Performance Testing Automation

```python
# tests/performance/database_performance_test.py
import asyncio
import time
import statistics
from typing import List, Dict, Tuple
import aiohttp

class DatabasePerformanceTester:
    def __init__(self, db_url: str, concurrency: int = 100):
        self.db_url = db_url
        self.concurrency = concurrency
        self.results: List[Dict] = []
    
    async def run_load_test(self, query: str, iterations: int = 1000) -> Dict:
        """Run load test with specified query"""
        start_time = time.time()
        
        # Create connection pool
        connections = []
        for _ in range(self.concurrency):
            conn = await self._create_connection()
            connections.append(conn)
        
        try:
            # Run concurrent queries
            tasks = []
            for i in range(iterations):
                conn = connections[i % len(connections)]
                task = self._execute_query(conn, query, i)
                tasks.append(task)
            
            # Execute all tasks
            start_exec = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            exec_time = time.time() - start_exec
            
            # Process results
            latencies = []
            successes = 0
            failures = 0
            
            for result in results:
                if isinstance(result, Exception):
                    failures += 属实
                else:
                    latencies.append(result['latency'])
                    successes += 1
            
            # Calculate metrics
            metrics = {
                'total_requests': iterations,
                'success_rate': successes / iterations,
                'failures': failures,
                'avg_latency_ms': statistics.mean(latencies) if latencies else 0,
                'p95_latency_ms': statistics.quantiles(latencies, n=20)[18] if latencies else 0,
                'p99_latency_ms': statistics.quantiles(latencies, n=100)[98] if latencies else 0,
                'throughput_qps': iterations / exec_time,
                'total_time_seconds': time.time() - start_time
            }
            
            return metrics
            
        finally:
            # Cleanup connections
            for conn in connections:
                await conn.close()
    
    async def _execute_query(self, conn, query: str, request_id: int) -> Dict:
        """Execute single query with timing"""
        start_time = time.time()
        
        try:
            # Execute query
            result = await conn.fetch(query)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                'request_id': request_id,
                'latency': latency,
                'rows': len(result),
                'status': 'success'
            }
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return {
                'request_id': request_id,
                'latency': latency,
                'error': str(e),
                'status': 'failure'
            }
    
    async def run_ai_workload_test(self):
        """Test AI-specific workloads"""
        # Test vector similarity search
        vector_search_query = """
        SELECT id, name, 
               vector_distance(embedding_vector, '[0.1, 0.2, 0.3, ...]') as distance
        FROM users 
        ORDER BY distance 
        LIMIT 10
        """
        
        # Test time-series aggregation
        timeseries_query = """
        SELECT 
            time_bucket('1 hour', timestamp) as bucket,
            avg(value) as avg_value,
            count(*) as count
        FROM sensor_data 
        WHERE timestamp > now() - interval '24 hours'
        GROUP BY bucket
        ORDER BY bucket
        """
        
        # Run tests
        vector_results = await self.run_load_test(vector_search_query, 500)
        timeseries_results = await self.run_load_test(timeseries_query, 500)
        
        return {
            'vector_search': vector_results,
            'timeseries_aggregation': timeseries_results
        }

# CI pipeline integration
async def main():
    tester = DatabasePerformanceTester(
        db_url="postgresql://user:pass@localhost:5432/ai_mastery",
        concurrency=50
    )
    
    # Run performance tests
    results = await tester.run_ai_workload_test()
    
    # Check performance thresholds
    if results['vector_search']['p95_latency_ms'] > 200:
        raise Exception(f"Vector search P95 latency too high: {results['vector_search']['p95_latency_ms']}ms")
    
    if results['timeseries_aggregation']['throughput_qps'] < 100:
        raise Exception(f"Timeseries aggregation throughput too low: {results['timeseries_aggregation']['throughput_qps']} QPS")
    
    print("Performance tests passed!")

if __name__ == "__main__":
    asyncio.run(main())
```

### CI Pipeline Integration Examples

#### GitHub Actions Workflow

```yaml
# .github/workflows/advanced-database-tests.yml
name: Advanced Database Testing
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup PostgreSQL
        uses: webfactory/ssh-agent@v0.9.0
        with:
          ssh-private-key: ${{ secrets.DB_SSH_KEY }}
      
      - name: Run unit tests
        run: |
          pip install pytest asyncpg
          pytest tests/database/test_unit.py --verbose

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup test database
        run: |
          psql -U postgres -c "CREATE DATABASE ai_test;"
      
      - name: Run integration tests
        run: |
          pytest tests/database/test_integration.py \
            --db-url postgresql://postgres:@localhost/ai_test \
            --verbose

  performance-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup performance environment
        run: |
          # Configure performance test database with larger resources
          psql -U postgres -c "CREATE DATABASE ai_perf_test;"
      
      - name: Run performance tests
        run: |
          pip install asyncio aiohttp
          python tests/performance/database_performance_test.py \
            --db-url postgresql://postgres:@localhost/ai_perf_test \
            --concurrency 100 \
            --iterations 1000

  chaos-tests:
    runs-on: ubuntu-latest
    needs: performance-tests
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      
      - name: Run chaos tests
        run: |
          pip install chaos-toolkit
          chaos run experiments/database_chaos_experiment.json

  report:
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests, performance-tests, chaos-tests]
    steps:
      - uses: actions/checkout@v3
      
      - name: Generate test report
        run: |
          python scripts/generate_test_report.py \
            --output reports/database_test_report.html \
            --format html
          
      - name: Upload test report
        uses: actions/upload-artifact@v3
        with:
          name: database-test-report
          path: reports/database_test_report.html
```

## Best Practices for Production Database DevOps

### Safety Mechanisms

#### Deployment Gates and Approvals

Implement multi-level deployment gates:

```yaml
# deployment_gates.yaml
deployment_gates:
  pre-deploy:
    - name: "Schema compatibility check"
      command: "python scripts/check_schema_compatibility.py"
      threshold: "success"
    
    - name: "Data integrity validation"
      command: "python scripts/validate_data_integrity.py"
      threshold: "no_errors"
    
    - name: "Performance regression test"
      command: "python scripts/performance_regression_test.py"
      threshold: "p95_latency_increase < 10%"
  
  post-deploy:
    - name: "Smoke tests"
      command: "pytest tests/smoke_tests.py"
      threshold: "all_pass"
    
    - name: "Business metric validation"
      command: "python scripts/validate_business_metrics.py"
      threshold: "error_rate < 0.1%"
  
  auto-rollback:
    conditions:
      - "p95_latency > 500ms"
      - "error_rate > 1%"
      - "connection_errors > 5%"
    action: "execute_rollback_script"
```

#### Canary Deployment Strategy

Implement canary deployments for database changes:

```python
# scripts/canary_deployment.py
import time
import random
from typing import Dict, Optional

class CanaryDeployment:
    def __init__(self, db_connection, canary_percentage: float = 0.1):
        self.db_connection = db_connection
        self.canary_percentage = canary_percentage
        self.canary_active = False
    
    async def activate_canary(self, migration_id: str):
        """Activate canary deployment for specific migration"""
        # Store canary configuration
        await self.db_connection.execute(
            "INSERT INTO canary_deployments (migration_id, percentage, active, created_at) "
            "VALUES ($1, $2, true, NOW())",
            migration_id, self.canary_percentage
        )
        
        self.canary_active = True
        return True
    
    async def route_request(self, user_id: str) -> str:
        """Route request to canary or stable version"""
        if not self.canary_active:
            return "stable"
        
        # Hash user ID to determine routing
        hash_value = abs(hash(user_id)) % 100
        if hash_value < (self.canary_percentage * 100):
            return "canary"
        else:
            return "stable"
    
    async def monitor_canary(self, migration_id: str) -> Dict:
        """Monitor canary performance metrics"""
        # Get canary metrics
        canary_metrics = await self.db_connection.fetch(
            "SELECT "
            "COUNT(*) as total_requests, "
            "SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_count, "
            "AVG(latency_ms) as avg_latency, "
            "PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency "
            "FROM request_logs "
            "WHERE migration_id = $1 AND version = 'canary' "
            "AND created_at > NOW() - INTERVAL '10 minutes'",
            migration_id
        )
        
        # Get stable metrics for comparison
        stable_metrics = await self.db_connection.fetch(
            "SELECT "
            "COUNT(*) as total_requests, "
            "SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_count, "
            "AVG(latency_ms) as avg_latency, "
            "PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency "
            "FROM request_logs "
            "WHERE migration_id = $1 AND version = 'stable' "
            "AND created_at > NOW() - INTERVAL '10 minutes'",
            migration_id
        )
        
        return {
            'canary': dict(canary_metrics[0]) if canary_metrics else {},
            'stable': dict(stable_metrics[0]) if stable_metrics else {}
        }
    
    async def promote_canary(self, migration_id: str):
        """Promote canary to full deployment"""
        # Update all traffic to use canary version
        await self.db_connection.execute(
            "UPDATE canary_deployments "
            "SET percentage = 100, promoted_at = NOW() "
            "WHERE migration_id = $1 AND active = true",
            migration_id
        )
        
        # Clean up old versions
        await self.db_connection.execute(
            "DELETE FROM request_logs "
            "WHERE migration_id = $1 AND version = 'stable' "
            "AND created_at < NOW() - INTERVAL '1 hour'",
            migration_id
        )
```

### Monitoring and Observability Integration

#### Database Metrics Collection

```python
# scripts/metrics_collector.py
import asyncio
import time
from typing import Dict, Any
import prometheus_client

# Custom metrics
DATABASE_QUERIES_TOTAL = prometheus_client.Counter(
    'database_queries_total', 
    'Total number of database queries',
    ['operation', 'status']
)

DATABASE_QUERY_LATENCY = prometheus_client.Histogram(
    'database_query_latency_seconds',
    'Database query latency in seconds',
    ['operation'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

DATABASE_CONNECTIONS = prometheus_client.Gauge(
    'database_connections',
    'Current database connections',
    ['pool']
)

class DatabaseMetricsCollector:
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.start_time = time.time()
    
    async def collect_metrics(self):
        """Collect and expose database metrics"""
        while True:
            try:
                # Get connection pool stats
                pool_stats = self.db_pool.get_stats()
                DATABASE_CONNECTIONS.labels(pool='main').set(pool_stats['connections'])
                
                # Get query statistics
                query_stats = await self._get_query_stats()
                for stat in query_stats:
                    DATABASE_QUERIES_TOTAL.labels(
                        operation=stat['operation'],
                        status=stat['status']
                    ).inc(stat['count'])
                    
                    if stat['latency']:
                        DATABASE_QUERY_LATENCY.labels(
                            operation=stat['operation']
                        ).observe(stat['latency'])
                
                # Collect system metrics
                await self._collect_system_metrics()
                
            except Exception as e:
                print(f"Error collecting metrics: {e}")
            
            await asyncio.sleep(15)  # Collect every 15 seconds
    
    async def _get_query_stats(self) -> list:
        """Get query statistics from database"""
        try:
            async with self.db_pool.acquire() as conn:
                # PostgreSQL query statistics
                stats = await conn.fetch("""
                    SELECT 
                        query,
                        calls,
                        total_exec_time / 1000 as total_latency_sec,
                        mean_exec_time / 1000 as avg_latency_sec,
                        rows
                    FROM pg_stat_statements 
                    WHERE query NOT LIKE 'EXPLAIN%' 
                    AND query NOT LIKE 'SELECT 1'
                    ORDER BY total_exec_time DESC
                    LIMIT 50
                """)
                
                result = []
                for row in stats:
                    # Parse operation type
                    operation = 'unknown'
                    if row['query'].startswith('SELECT'):
                        operation = 'select'
                    elif row['query'].startswith('INSERT'):
                        operation = 'insert'
                    elif row['query'].startswith('UPDATE'):
                        operation = 'update'
                    elif row['query'].startswith('DELETE'):
                        operation = 'delete'
                    
                    result.append({
                        'operation': operation,
                        'status': 'success',
                        'count': row['calls'],
                        'latency': row['avg_latency_sec']
                    })
                
                return result
                
        except Exception as e:
            print(f"Error getting query stats: {e}")
            return []
    
    async def _collect_system_metrics(self):
        """Collect system-level database metrics"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get system metrics
                system_stats = await conn.fetchrow("""
                    SELECT 
                        pg_postmaster_start_time() as start_time,
                        pg_is_in_recovery() as in_recovery,
                        pg_current_wal_lsn() as wal_lsn,
                        pg_last_xact_replay_timestamp() as last_replay
                """)
                
                # Export custom metrics
                prometheus_client.Gauge(
                    'database_uptime_seconds',
                    'Database uptime in seconds'
                ).set(time.time() - system_stats['start_time'].timestamp())
                
                prometheus_client.Gauge(
                    'database_in_recovery',
                    'Database in recovery mode (1=true, 0=false)'
                ).set(1 if system_stats['in_recovery'] else 0)
                
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
```

### Conclusion

Advanced database DevOps automation is essential for production AI/ML systems that require high reliability, performance, and scalability. By implementing the strategies outlined in this guide—sophisticated CI/CD pipelines, Infrastructure as Code for modern database architectures, GitOps workflows, and comprehensive testing—you can ensure your database operations are robust, automated, and aligned with modern DevOps practices.

The key principles to remember:
- **Safety first**: Always implement rollback mechanisms, validation gates, and canary deployments
- **Automation everywhere**: Automate schema changes, data migrations, testing, and monitoring
- **Observability built-in**: Collect comprehensive metrics and integrate with observability platforms
- **AI-specific considerations**: Optimize for vector operations, time-series analytics, and ML workload patterns
- **Continuous improvement**: Regularly review and refine your database DevOps processes

By following these advanced practices, senior AI/ML engineers can build database systems that support rapid iteration while maintaining production-grade reliability and performance.