# Database CI/CD: Continuous Integration and Deployment for Databases

This comprehensive guide covers modern database CI/CD practices, from schema migrations to production deployments, ensuring reliability, safety, and efficiency.

## Table of Contents
1. [Introduction to Database CI/CD]
2. [Schema Migration Strategies]
3. [Testing Strategies for Databases]
4. [Safe Deployment Patterns]
5. [Infrastructure as Code for Databases]
6. [Monitoring and Observability]
7. [Implementation Examples]
8. [Common Anti-Patterns and Solutions]

---

## 1. Introduction to Database CI/CD

Database CI/CD extends continuous integration and deployment practices to database changes, enabling safe, automated, and reliable database evolution.

### Why Database CI/CD Matters
- **Reduced deployment risk**: Automated testing catches issues before production
- **Faster iterations**: Developers can deploy database changes confidently
- **Consistency**: Standardized processes across teams and environments
- **Auditability**: Complete change history and rollback capability
- **Collaboration**: Better coordination between developers and DBAs

### Database CI/CD Maturity Model
| Level | Characteristics | Tools/Techniques |
|-------|----------------|------------------|
| Basic | Manual scripts, no testing | SQL files, basic version control |
| Intermediate | Automated migrations, basic testing | Flyway/Liquibase, unit tests |
| Advanced | Comprehensive testing, canary deployments | Schema validation, integration tests, blue/green |
| Enterprise | GitOps, automated rollbacks, chaos testing | Kubernetes operators, policy-as-code |

### Core Principles
- **Idempotent operations**: Migrations should be safe to run multiple times
- **Atomic changes**: Each migration should be a single transaction
- **Reversible changes**: Ability to rollback safely
- **Environment parity**: Development, staging, production should be similar
- **Automated verification**: Test changes automatically

---

## 2. Schema Migration Strategies

### Migration Tool Comparison

| Tool | Language | Approach | Best For |
|------|----------|----------|----------|
| **Flyway** | Java | Versioned SQL migrations | Simple, SQL-focused teams |
| **Liquibase** | Java | XML/YAML/JSON changelogs | Complex transformations, cross-database |
| **Alembic** | Python | Python-based migrations | Python applications, SQLAlchemy |
| **Prisma Migrate** | TypeScript | Declarative schema | Modern JavaScript/TypeScript apps |
| **Hasura Migrations** | YAML | GraphQL-first | Hasura GraphQL API users |

### Migration Patterns

#### A. Versioned Migrations (Recommended)
```sql
-- V1__initial_schema.sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(254) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- V2__add_name_column.sql
ALTER TABLE users ADD COLUMN name VARCHAR(100) NOT NULL DEFAULT '';

-- V3__create_indexes.sql
CREATE INDEX idx_users_email ON users(email);
```

#### B. State-Based Migrations
```yaml
# schema.yml
tables:
  - name: users
    columns:
      - name: id
        type: uuid
        default: gen_random_uuid()
        primary_key: true
      - name: email
        type: varchar(254)
        unique: true
        not_null: true
      - name: name
        type: varchar(100)
        not_null: true
        default: ''
    indexes:
      - name: idx_users_email
        columns: [email]
```

#### C. Hybrid Approach
Combine versioned and state-based:
- Use versioned migrations for complex logic
- Use state-based for simple schema definitions
- Generate state-based from versioned for consistency checks

### Migration Safety Practices

#### Idempotent Migrations
```sql
-- Safe idempotent migration
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'users' AND column_name = 'name') THEN
        ALTER TABLE users ADD COLUMN name VARCHAR(100) NOT NULL DEFAULT '';
    END IF;
END $$;

-- Or using conditional DDL (PostgreSQL 9.6+)
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS name VARCHAR(100) NOT NULL DEFAULT '';
```

#### Reversible Migrations
```sql
-- Up migration
ALTER TABLE orders ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'pending';

-- Down migration
ALTER TABLE orders DROP COLUMN status;
```

#### Data Migration Safety
```sql
-- Safe data migration pattern
BEGIN;

-- Step 1: Add new column
ALTER TABLE users ADD COLUMN new_field VARCHAR(100);

-- Step 2: Backfill data in batches
DO $$
DECLARE
    batch_size INT := 1000;
    offset INT := 0;
    total_rows INT;
BEGIN
    SELECT COUNT(*) INTO total_rows FROM users WHERE new_field IS NULL;
    
    WHILE offset < total_rows LOOP
        EXECUTE format('
            UPDATE users 
            SET new_field = ''default_value''
            WHERE id IN (
                SELECT id FROM users 
                WHERE new_field IS NULL 
                ORDER BY id 
                LIMIT %s OFFSET %s
            )
        ', batch_size, offset);
        
        offset := offset + batch_size;
        COMMIT; -- Commit after each batch
        PERFORM pg_sleep(0.1); -- Small delay to reduce load
    END LOOP;
END $$;

-- Step 3: Make column non-nullable
ALTER TABLE users ALTER COLUMN new_field SET NOT NULL;

COMMIT;
```

---

## 3. Testing Strategies for Databases

### Testing Pyramid for Databases

```
┌─────────────────────────────────────┐
│  End-to-End Tests (1-5%)           │
│  • Full application integration     │
│  • Realistic data volumes           │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│  Integration Tests (10-20%)        │
│  • Database + application layer1    │
│  • Cross-service interactions1      │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│  Unit Tests (75-85%)               │
│  • Individual migration scripts     │
│  • Stored procedures/functions1     │
│  • Schema validation rules          │
└─────────────────────────────────────┘
```

### Unit Testing Strategies

#### Migration Testing
```python
# pytest example for migration testing
def test_migration_v2_adds_name_column():
    """Test that V2 migration adds name column correctly"""
    # Setup: Apply only migrations up to V1
    db.apply_migrations(up_to='V1')
    
    # Verify initial state
    assert not db.column_exists('users', 'name')
    
    # Apply V2 migration
    db.apply_migration('V2__add_name_column.sql')
    
    # Verify expected state
    assert db.column_exists('users', 'name')
    assert db.column_default('users', 'name') == ''
    
    # Verify data integrity
    user_id = db.insert_user(email='test@example.com')
    user = db.get_user(user_id)
    assert user['name'] == ''
```

#### Schema Validation Tests
```python
def test_schema_constraints():
    """Test that schema constraints work as expected"""
    # Test unique constraint
    db.insert_user(email='test1@example.com')
    with pytest.raises(DatabaseError):
        db.insert_user(email='test1@example.com')  # Should fail due to unique constraint
    
    # Test not null constraint
    with pytest.raises(DatabaseError):
        db.execute("INSERT INTO users (email) VALUES ('test2@example.com')")
        # Should fail due to NOT NULL on name column (if added)
```

#### Performance Tests
```python
def test_query_performance():
    """Test query performance with realistic data volumes"""
    # Insert test data
    for i in range(10000):
        db.insert_user(email=f'user{i}@example.com', name=f'User {i}')
    
    # Test query performance
    start_time = time.time()
    results = db.query_users_by_email('user5000@example.com')
    query_time = time.time() - start_time
    
    # Assert performance requirements
    assert len(results) == 1
    assert query_time < 0.1  # Should be under 100ms
```

### Integration Testing Strategies

#### Test Database Isolation
```python
# Using pytest fixtures for isolated test databases
@pytest.fixture(scope="function")
def test_db():
    """Create isolated test database for each test"""
    db_name = f"test_db_{uuid.uuid4().hex[:8]}"
    
    # Create database
    subprocess.run(['createdb', db_name])
    
    try:
        # Apply all migrations
        db = DatabaseConnection(f"postgresql://localhost/{db_name}")
        db.apply_all_migrations()
        
        yield db
        
    finally:
        # Clean up
        subprocess.run(['dropdb', db_name])
```

#### Contract Testing
```python
def test_database_contract():
    """Test that database contract matches application expectations"""
    # Application expects these columns
    expected_columns = {
        'users': ['id', 'email', 'name', 'created_at'],
        'orders': ['id', 'user_id', 'total', 'status', 'created_at']
    }
    
    # Verify actual columns match expectations
    for table, expected_cols in expected_columns.items():
        actual_cols = db.get_table_columns(table)
        assert set(actual_cols) == set(expected_cols), \
            f"Table {table} columns mismatch: expected {expected_cols}, got {actual_cols}"
    
    # Verify data types
    assert db.get_column_type('users', 'email') == 'varchar(254)'
    assert db.get_column_type('orders', 'total') == 'numeric'
```

---

## 4. Safe Deployment Patterns

### Blue/Green Deployment for Databases

#### Pattern Overview
```
┌─────────────────┐    ┌─────────────────┐
│  Blue Database  │◀──▶│  Green Database │
└─────────────────┘    └─────────────────┘
        ▲                        ▲
        │                        │
┌─────────────────┐    ┌─────────────────┐
│  Application    │◀──▶│  Application    │
│  (Blue)         │    │  (Green)        │
└─────────────────┘    └─────────────────┘
        │                        │
        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│  Traffic Router │◀──▶│  DNS/Load Balancer│
└─────────────────┘    └─────────────────┘
```

#### Implementation Steps
1. **Prepare green database**: Apply migrations to standby database
2. **Sync data**: Replicate data from blue to green
3. **Validate**: Run comprehensive tests on green database
4. **Switch traffic**: Update router to direct traffic to green
5. **Monitor**: Observe metrics and logs
6. **Decommission**: After successful validation, decommission blue

#### Blue/Green with Zero Downtime
```sql
-- Step 1: Create shadow tables
CREATE TABLE users_green AS SELECT * FROM users WHERE 1=0;
ALTER TABLE users_green ADD COLUMN migrated BOOLEAN DEFAULT FALSE;

-- Step 2: Apply schema changes to green table
ALTER TABLE users_green ADD COLUMN new_field VARCHAR(100);

-- Step 3: Backfill data in batches
DO $$
DECLARE
    batch_size INT := 1000;
    offset INT := 0;
    total_rows INT;
BEGIN
    SELECT COUNT(*) INTO total_rows FROM users;
    
    WHILE offset < total_rows LOOP
        EXECUTE format('
            INSERT INTO users_green (id, email, name, created_at, new_field)
            SELECT id, email, name, created_at, ''default_value''
            FROM users 
            ORDER BY id 
            LIMIT %s OFFSET %s
        ', batch_size, offset);
        
        offset := offset + batch_size;
        COMMIT;
    END LOOP;
END $$;

-- Step 4: Switch application to use green table
-- Update application configuration or use view switching
CREATE OR REPLACE VIEW users_current AS SELECT * FROM users_green;
```

### Canary Deployments

#### Database Canary Strategy
1. **Route small percentage of traffic** to new database version
2. **Monitor key metrics**: Error rates, latency, throughput
3. **Gradually increase traffic** based on success criteria
4. **Rollback if thresholds exceeded**

```python
class CanaryDeployer:
    def __init__(self, db_router: DatabaseRouter):
        self.router = db_router
    
    async def deploy_canary(self, migration_id: str, initial_percentage: float = 5.0):
        """Deploy migration to canary subset of traffic"""
        # Set initial canary percentage
        await self.router.set_canary_percentage(migration_id, initial_percentage)
        
        # Monitor for 5 minutes
        await asyncio.sleep(300)
        
        # Check metrics
        metrics = await self._get_deployment_metrics(migration_id)
        
        if metrics['error_rate'] < 0.1 and metrics['latency_p95'] < 200:
            # Success - increase to 25%
            await self.router.set_canary_percentage(migration_id, 25.0)
            await asyncio.sleep(600)  # 10 minutes
            
            metrics = await self._get_deployment_metrics(migration_id)
            if metrics['error_rate'] < 0.05 and metrics['latency_p95'] < 150:
                # Success - increase to 100%
                await self.router.set_canary_percentage(migration_id, 100.0)
                return {'status': 'SUCCESS', 'final_percentage': 100.0}
        
        else:
            # Failure - rollback
            await self.router.rollback_migration(migration_id)
            return {'status': 'FAILED', 'rollback_initiated': True}
```

### Rolling Updates

#### Zero-Downtime Rolling Updates
For read-replica architectures:
1. **Update read replicas** first
2. **Promote one replica** to primary
3. **Update old primary** as new replica
4. **Repeat** until all instances updated

```sql
-- Example rolling update for PostgreSQL
-- Step 1: Update standby replica
SELECT pg_reload_conf(); -- Reload config on standby

-- Step 2: Promote standby to primary
SELECT pg_promote();

-- Step 3: Update old primary as new standby
-- Configure replication from new primary to old primary

-- Step 4: Repeat for remaining replicas
```

---

## 5. Infrastructure as Code for Databases

### IaC Tools Comparison

| Tool | Language | Database Support | Best For |
|------|----------|------------------|----------|
| **Terraform** | HCL | AWS RDS, Azure DB, GCP Cloud SQL | Multi-cloud infrastructure |
| **CloudFormation** | YAML/JSON | AWS services only | AWS-only environments |
| **Pulumi** | Python/TypeScript | Multi-cloud, custom providers | Developer-friendly IaC |
| **CDK** | TypeScript/Python | AWS services | AWS with programming language |
| **DBT** | SQL/YAML | Data warehouses | Analytics and transformation |

### Terraform Database Configuration

```hcl
# main.tf
provider "aws" {
  region = "us-west-2"
}

resource "aws_db_subnet_group" "main" {
  name       = "main"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_db_instance" "primary" {
  identifier           = "production-db"
  engine               = "postgres"
  engine_version       = "14.7"
  instance_class       = "db.r6g.xlarge"
  allocated_storage    = 1000
  storage_type         = "gp3"
  storage_encrypted    = true
  backup_retention_period = 35
  multi_az             = true
  publicly_accessible  = false
  db_subnet_group_name = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.db.id]
  
  # Database parameters
  parameter_group_name = aws_db_parameter_group.default.name
  
  # Monitoring
  enable_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  
  # Tags for cost allocation
  tags = {
    Environment = "production"
    Team        = "data-engineering"
    Owner       = "database-team"
  }
}

resource "aws_db_parameter_group" "default" {
  name        = "postgres14-params"
  family      = "postgres14"
  description = "Custom parameter group for production"

  parameter {
    name         = "max_connections"
    value        = "200"
    apply_method = "pending-reboot"
  }

  parameter {
    name         = "shared_buffers"
    value        = "4GB"
    apply_method = "pending-reboot"
  }

  parameter {
    name         = "work_mem"
    value        = "16MB"
    apply_method = "pending-reboot"
  }
}
```

### Database Configuration as Code

#### Parameter Management
```yaml
# database-config.yaml
production:
  postgresql:
    parameters:
      max_connections: 200
      shared_buffers: "4GB"
      work_mem: "16MB"
      maintenance_work_mem: "1GB"
      effective_cache_size: "12GB"
      checkpoint_completion_target: 0.9
      wal_buffers: "16MB"
      default_statistics_target: 100
    extensions:
      - "pg_stat_statements"
      - "pg_trgm"
      - "uuid-ossp"
      - "postgis"
    security:
      ssl_min_protocol_version: "TLSv1.2"
      password_encryption: "scram-sha-256"
```

#### Environment-Specific Configurations
```python
class DatabaseConfigManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
    
    def get_config(self, environment: str, database_type: str):
        """Get database configuration for specific environment and type"""
        config = self._load_config(self.config_path)
        
        # Get environment-specific config
        env_config = config.get(environment, {})
        
        # Get database-specific config
        db_config = env_config.get(database_type, {})
        
        # Merge with defaults
        defaults = config.get('defaults', {}).get(database_type, {})
        merged_config = {**defaults, **db_config}
        
        return merged_config
    
    def validate_config(self, config: dict) -> list:
        """Validate database configuration against best practices"""
        violations = []
        
        # Check critical parameters
        if config.get('max_connections') > 300:
            violations.append({
                'severity': 'WARNING',
                'rule': 'max_connections_too_high',
                'message': f'max_connections ({config["max_connections"]}) exceeds recommended 200-250',
                'recommendation': 'Reduce to 200-250 for better resource management'
            })
        
        if config.get('shared_buffers') and 'GB' in config['shared_buffers']:
            buffer_size = float(config['shared_buffers'].replace('GB', ''))
            if buffer_size > 0.25 * self._get_system_memory_gb():
                violations.append({
                    'severity': 'WARNING',
                    'rule': 'shared_buffers_too_large',
                    'message': f'shared_buffers ({buffer_size}GB) exceeds 25% of system memory',
                    'recommendation': 'Reduce to 25% of system memory (typically 4-8GB)'
                })
        
        return violations
```

---

## 6. Monitoring and Observability

### Database CI/CD Monitoring Framework

#### Pre-Deployment Monitoring
- **Migration validation**: Verify migration scripts are valid
- **Schema compatibility**: Check backward compatibility
- **Performance impact**: Estimate query plan changes
- **Data integrity**: Validate data consistency

#### Post-Deployment Monitoring
- **Error rates**: Database errors and application errors
- **Latency metrics**: Query latency percentiles
- **Throughput**: Queries per second, transactions per second
- **Resource utilization**: CPU, memory, I/O, connections
- **Replication lag**: For multi-node deployments

### Key Metrics Dashboard

```python
class DatabaseCIMonitor:
    def __init__(self, metrics_client: PrometheusClient):
        self.metrics = metrics_client
    
    def get_deployment_health(self, deployment_id: str):
        """Get health metrics for a database deployment"""
        health = {
            'deployment_id': deployment_id,
            'status': 'HEALTHY',
            'metrics': {},
            'alerts': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Get key metrics
        metrics = {
            'error_rate': self.metrics.get(f'db_error_rate{{deployment_id="{deployment_id}"}}'),
            'latency_p95': self.metrics.get(f'db_latency_seconds_p95{{deployment_id="{deployment_id}"}}'),
            'connection_usage': self.metrics.get(f'db_connections_used{{deployment_id="{deployment_id}"}}'),
            'replication_lag': self.metrics.get(f'db_replication_lag_seconds{{deployment_id="{deployment_id}"}}'),
            'migration_success_rate': self.metrics.get(f'db_migration_success_rate{{deployment_id="{deployment_id}"}}')
        }
        
        health['metrics'] = metrics
        
        # Check for issues
        if metrics['error_rate'] > 0.01:
            health['status'] = 'DEGRADED'
            health['alerts'].append({
                'severity': 'HIGH',
                'message': f'High error rate: {metrics["error_rate"]:.2%}',
                'metric': 'error_rate'
            })
        
        if metrics['latency_p95'] > 500:
            health['status'] = 'DEGRADED'
            health['alerts'].append({
                'severity': 'MEDIUM',
                'message': f'High latency: {metrics["latency_p95"]}ms P95',
                'metric': 'latency_p95'
            })
        
        if metrics['replication_lag'] > 60:
            health['status'] = 'DEGRADED'
            health['alerts'].append({
                'severity': 'MEDIUM',
                'message': f'Replication lag: {metrics["replication_lag"]}s',
                'metric': 'replication_lag'
            })
        
        return health
```

### Observability Integration

#### Distributed Tracing
```python
# OpenTelemetry example for database operations
from opentelemetry import trace
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

# Instrument database connections
Psycopg2Instrumentor().instrument()

# Custom tracing for migrations
def trace_migration(migration_id: str, operation: str):
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span(f"database.migration.{operation}") as span:
        span.set_attribute("migration.id", migration_id)
        span.set_attribute("migration.operation", operation)
        span.set_attribute("service.name", "database-ci-cd")
        
        try:
            # Execute migration
            result = execute_migration(migration_id)
            
            span.set_attribute("migration.success", True)
            span.set_attribute("migration.duration_ms", span.end_time - span.start_time)
            
            return result
            
        except Exception as e:
            span.set_attribute("migration.success", False)
            span.set_attribute("migration.error", str(e))
            span.record_exception(e)
            raise
```

#### Log Correlation
```python
# Structured logging with correlation IDs
def log_database_operation(operation: str, context: dict):
    """Log database operation with correlation ID"""
    correlation_id = context.get('correlation_id', str(uuid.uuid4()))
    
    logger.info(f"Database operation: {operation}",
               extra={
                   'correlation_id': correlation_id,
                   'operation': operation,
                   'service': 'database-ci-cd',
                   'timestamp': datetime.now().isoformat(),
                   'environment': os.getenv('ENVIRONMENT', 'unknown'),
                   'deployment_id': context.get('deployment_id'),
                   'migration_id': context.get('migration_id')
               })
```

---

## 7. Implementation Examples

### Example 1: GitHub Actions Database CI/CD Pipeline
```yaml
# .github/workflows/database-ci.yml
name: Database CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test-migrations:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        ports:
          - 5432:5432
        options: --health-cmd pg_isready --health-interval 10s --health-timeout 5s --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install pytest psycopg2-binary flyway
    
    - name: Run migration tests
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
      run: |
        # Apply migrations up to current version
        flyway -url=$DATABASE_URL -user=postgres -password=postgres migrate
        
        # Run tests
        pytest tests/database_tests.py
    
    - name: Validate schema
      run: |
        python scripts/validate_schema.py
    
  deploy-to-staging:
    needs: test-migrations
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to staging
      env:
        STAGING_DB_URL: ${{ secrets.STAGING_DB_URL }}
      run: |
        flyway -url=$STAGING_DB_URL -user=$STAGING_DB_USER -password=$STAGING_DB_PASSWORD migrate
        
        # Verify deployment
        python scripts/verify_deployment.py staging
    
  deploy-to-production:
    needs: deploy-to-staging
    runs-on: ubuntu-latest
    environment: production
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Blue/Green deployment
      env:
        PROD_DB_URL: ${{ secrets.PROD_DB_URL }}
      run: |
        # Prepare green database
        python scripts/blue_green_prepare.py
        
        # Validate green database
        python scripts/validate_green_db.py
        
        # Switch traffic
        python scripts/switch_traffic.py
        
        # Monitor for 5 minutes
        python scripts/monitor_deployment.py 300
```

### Example 2: Database Change Approval Workflow
```python
class DatabaseChangeApproval:
    def __init__(self, notification_service: NotificationService, db_client: DatabaseClient):
        self.notifications = notification_service
        self.db = db_client
    
    async def request_approval(self, change_request: dict):
        """Request approval for database change"""
        # Create approval request
        approval_id = str(uuid.uuid4())
        
        request = {
            'id': approval_id,
            'change_type': change_request['type'],  # 'migration', 'schema_change', 'data_change'
            'description': change_request['description'],
            'impact_assessment': change_request['impact'],
            'rollback_plan': change_request['rollback'],
            'requested_by': change_request['author'],
            'requested_at': datetime.now(),
            'status': 'PENDING',
            'approvals': []
        }
        
        # Store request
        await self.db.insert('approval_requests', request)
        
        # Notify approvers
        approvers = self._get_approvers(change_request['type'])
        for approver in approvers:
            await self.notifications.send_approval_request(
                approver,
                approval_id,
                change_request['description'],
                change_request['impact']
            )
        
        return approval_id
    
    async def approve_change(self, approval_id: str, approver_id: str, comments: str = ""):
        """Process approval for database change"""
        request = await self.db.get('approval_requests', approval_id)
        
        if request['status'] != 'PENDING':
            raise ValueError(f"Request {approval_id} is not pending")
        
        # Record approval
        approval_record = {
            'approver_id': approver_id,
            'approved_at': datetime.now(),
            'comments': comments,
            'status': 'APPROVED'
        }
        
        request['approvals'].append(approval_record)
        
        # Check if sufficient approvals
        required_approvals = self._get_required_approvals(request['change_type'])
        if len(request['approvals']) >= required_approvals:
            request['status'] = 'APPROVED'
            await self.db.update('approval_requests', approval_id, request)
            
            # Notify requester
            await self.notifications.send_approval_status(
                request['requested_by'],
                approval_id,
                'APPROVED',
                f"Change approved by {len(request['approvals'])}/{required_approvals} reviewers"
            )
            
            return {'status': 'APPROVED', 'can_deploy': True}
        
        else:
            await self.db.update('approval_requests', approval_id, request)
            return {'status': 'PENDING', 'approvals_received': len(request['approvals']), 'required': required_approvals}
    
    def _get_required_approvals(self, change_type: str) -> int:
        """Get required number of approvals based on change type"""
        approval_rules = {
            'minor_migration': 1,
            'major_migration': 2,
            'schema_change': 2,
            'data_change': 3,
            'production_critical': 3
        }
        return approval_rules.get(change_type, 1)
```

### Example 3: Automated Rollback System
```python
class AutomatedRollbackSystem:
    def __init__(self, monitoring: DatabaseCIMonitor, db_client: DatabaseClient):
        self.monitoring = monitoring
        self.db = db_client
    
    async def setup_rollback_monitoring(self, deployment_id: str):
        """Set up automated rollback monitoring for a deployment"""
        # Create monitoring rules
        rules = [
            {
                'name': 'high_error_rate',
                'condition': 'error_rate > 0.05',
                'threshold': 0.05,
                'window': '5m',
                'action': 'rollback'
            },
            {
                'name': 'high_latency',
                'condition': 'latency_p95 > 1000',
                'threshold': 1000,
                'window': '2m',
                'action': 'rollback'
            },
            {
                'name': 'replication_failure',
                'condition': 'replication_lag > 300',
                'threshold': 300,
                'window': '1m',
                'action': 'rollback'
            }
        ]
        
        # Store rules
        await self.db.insert_many('rollback_rules', [
            {**rule, 'deployment_id': deployment_id} for rule in rules
        ])
        
        # Start monitoring task
        asyncio.create_task(self._monitor_deployment(deployment_id))
    
    async def _monitor_deployment(self, deployment_id: str):
        """Monitor deployment and trigger rollback if needed"""
        while True:
            try:
                # Get current health
                health = await self.monitoring.get_deployment_health(deployment_id)
                
                # Check rules
                rules = await self.db.find('rollback_rules', {'deployment_id': deployment_id})
                
                for rule in rules:
                    if rule['action'] == 'rollback':
                        # Evaluate condition
                        current_value = health['metrics'].get(rule['name'].replace('high_', ''), 0)
                        
                        if current_value > rule['threshold']:
                            # Trigger rollback
                            await self._trigger_rollback(deployment_id, rule['name'], current_value)
                            return
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in deployment monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _trigger_rollback(self, deployment_id: str, rule_name: str, current_value: float):
        """Trigger automated rollback"""
        logger.warning(f"Triggering rollback for deployment {deployment_id} due to {rule_name}: {current_value}")
        
        # Get last known good state
        last_good_state = await self.db.get('deployment_states', 
                                          {'deployment_id': deployment_id, 'status': 'GOOD'},
                                          sort=[('timestamp', -1)])
        
        if not last_good_state:
            raise ValueError(f"No good state found for deployment {deployment_id}")
        
        # Execute rollback
        try:
            # Revert to last good state
            await self._revert_to_state(last_good_state)
            
            # Update deployment status
            await self.db.insert('rollbacks', {
                'deployment_id': deployment_id,
                'rollback_reason': f'Auto-rollback: {rule_name} ({current_value})',
                'timestamp': datetime.now(),
                'status': 'SUCCESS',
                'from_state': last_good_state['state_id'],
                'to_state': 'ROLLBACK_COMPLETE'
            })
            
            # Notify team
            await self._notify_rollback(deployment_id, rule_name, current_value)
            
        except Exception as e:
            await self.db.insert('rollbacks', {
                'deployment_id': deployment_id,
                'rollback_reason': f'Auto-rollback failed: {rule_name}',
                'timestamp': datetime.now(),
                'status': 'FAILED',
                'error': str(e)
            })
            raise
    
    async def _revert_to_state(self, state: dict):
        """Revert database to a specific state"""
        # This would implement the actual rollback logic
        # Could involve:
        # - Running reverse migrations
        # - Restoring from backup
        # - Switching to blue/green database
        pass
```

---

## 8. Common Anti-Patterns and Solutions

### Anti-Pattern 1: Direct Production Changes
**Symptom**: Developers making direct changes to production databases
**Root Cause**: Lack of CI/CD process, emergency culture
**Solution**: Enforce CI/CD pipeline, require approvals, automated testing

### Anti-Pattern 2: Un-tested Migrations
**Symptom**: Migrations deployed without testing
**Root Cause**: No testing culture, time pressure
**Solution**: Mandatory testing in CI pipeline, test environments

### Anti-Pattern 3: Non-Idempotent Migrations
**Symptom**: Migrations fail when run multiple times
**Root Cause**: Assuming migrations run only once
**Solution**: Design idempotent migrations, use conditional DDL

### Anti-Pattern 4: No Rollback Strategy
**Symptom**: Unable to recover from failed deployments
**Root Cause**: Focus on forward progress only
**Solution**: Plan rollbacks upfront, automate rollback procedures

### Anti-Pattern 5: Environment Drift
**Symptom**: Development and production behave differently
**Root Cause**: Different configurations, data, versions
**Solution**: Environment parity, infrastructure as code, automated provisioning

---

## Next Steps

1. **Assess current database deployment process**: Identify gaps and risks
2. **Implement basic CI/CD**: Start with automated testing and versioned migrations
3. **Add safety mechanisms**: Approval workflows, canary deployments
4. **Build monitoring**: Comprehensive observability for database changes
5. **Automate rollbacks**: Implement automated recovery for failures

Database CI/CD is essential for modern software development. By implementing these practices, you'll achieve faster, safer, and more reliable database evolution while maintaining data integrity and system stability.