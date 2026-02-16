# Database DevOps and Automation Tutorial for AI/ML Systems

## Overview

This hands-on tutorial teaches senior AI/ML engineers how to implement database DevOps practices and automation for AI applications. We'll cover CI/CD for databases, infrastructure as code, automated operations, and disaster recovery.

## Prerequisites
- Git and GitHub/GitLab
- Docker and Docker Compose
- Terraform or AWS CloudFormation
- PostgreSQL 14+ or MySQL 8+
- Basic understanding of DevOps concepts

## Tutorial Structure
This tutorial is divided into 6 progressive sections:
1. **Database CI/CD** - Continuous integration and deployment for databases
2. **Infrastructure as Code** - Declarative database provisioning
3. **Automated Operations** - Monitoring, alerting, and remediation
4. **Disaster Recovery** - Backup, restore, and failover strategies
5. **AI-Specific Automation** - Feature store and model registry automation
6. **Production Readiness** - Comprehensive automation framework

## Section 1: Database CI/CD

### Step 1: Database migration pipeline
```yaml
# .github/workflows/database-ci.yml
name: Database CI/CD
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  database-ci:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: password
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
        pip install psycopg2 alembic pytest
        
    - name: Run database migrations
      env:
        DATABASE_URL: postgresql://postgres:password@localhost:5432/test_db
      run: |
        alembic upgrade head
        
    - name: Run database tests
      env:
        DATABASE_URL: postgresql://postgres:password@localhost:5432/test_db
      run: |
        pytest tests/database_tests.py
        
    - name: Validate schema
      env:
        DATABASE_URL: postgresql://postgres:password@localhost:5432/test_db
      run: |
        python scripts/validate_schema.py
        
    - name: Generate migration report
      run: |
        python scripts/generate_migration_report.py
```

### Step 2: Migration management with Alembic
```python
# alembic/env.py
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import os

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
from models import Base
target_metadata = Base.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### Step 3: Migration validation script
```python
# scripts/validate_schema.py
import psycopg2
import sys
import os

def validate_database_schema():
    """Validate database schema against expected structure"""
    
    # Connect to database
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'ai_db'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'password')
        )
    except Exception as e:
        print(f"Database connection failed: {e}")
        sys.exit(1)
    
    cursor = conn.cursor()
    
    # Check required tables
    required_tables = [
        'users', 'features', 'models', 'training_data',
        'audit_logs', 'metadata_assets'
    ]
    
    cursor.execute("""
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = 'public' AND tablename IN %s
    """, (tuple(required_tables),))
    
    existing_tables = [row[0] for row in cursor.fetchall()]
    
    missing_tables = set(required_tables) - set(existing_tables)
    
    if missing_tables:
        print(f"ERROR: Missing tables: {missing_tables}")
        conn.close()
        sys.exit(1)
    
    # Check required indexes
    required_indexes = [
        'idx_features_user_id_timestamp',
        'idx_models_status',
        'idx_audit_logs_timestamp'
    ]
    
    cursor.execute("""
        SELECT indexname 
        FROM pg_indexes 
        WHERE schemaname = 'public' AND indexname IN %s
    """, (tuple(required_indexes),))
    
    existing_indexes = [row[0] for row in cursor.fetchall()]
    missing_indexes = set(required_indexes) - set(existing_indexes)
    
    if missing_indexes:
        print(f"WARNING: Missing indexes: {missing_indexes}")
    
    # Check table constraints
    cursor.execute("""
        SELECT conname, conrelid::regclass
        FROM pg_constraint 
        WHERE conrelid::regclass::text IN %s
    """, (tuple(required_tables),))
    
    constraints = cursor.fetchall()
    print(f"Found {len(constraints)} constraints")
    
    conn.close()
    print("Schema validation passed ✅")
    return True

if __name__ == "__main__":
    validate_database_schema()
```

## Section 2: Infrastructure as Code

### Step 1: Terraform for database provisioning
```hcl
# terraform/main.tf
provider "aws" {
  region = var.region
}

# RDS instance for AI workloads
resource "aws_db_instance" "ai_postgres" {
  identifier           = "ai-postgres-${var.environment}"
  engine             = "postgres"
  engine_version     = "14.7"
  instance_class     = "db.m6g.xlarge"
  allocated_storage  = 1000
  storage_type       = "gp3"
  username           = var.db_username
  password           = var.db_password
  db_name            = "ai_db"
  backup_retention_period = 7
  multi_az           = true
  publicly_accessible = false
  skip_final_snapshot = true
  vpc_security_group_ids = [aws_security_group.db_sg.id]
  
  # Performance tuning for AI workloads
  parameter_group_name = aws_db_parameter_group.ai_pg.name
  
  tags = {
    Environment = var.environment
    Project     = "ai-platform"
    Owner       = "data-science-team"
  }
}

# Security group for database
resource "aws_security_group" "db_sg" {
  name        = "ai-db-sg-${var.environment}"
  description = "Security group for AI database"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]  # Internal VPC only
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "ai-db-sg-${var.environment}"
  }
}

# Parameter group for AI workloads
resource "aws_db_parameter_group" "ai_pg" {
  name        = "ai-postgres-pg-${var.environment}"
  family      = "postgres14"
  description = "Parameter group for AI workloads"

  parameter {
    name         = "shared_buffers"
    value        = "8GB"
    apply_method = "pending-reboot"
  }

  parameter {
    name         = "work_mem"
    value        = "64MB"
    apply_method = "pending-reboot"
  }

  parameter {
    name         = "maintenance_work_mem"
    value        = "2GB"
    apply_method = "pending-reboot"
  }

  parameter {
    name         = "effective_cache_size"
    value        = "24GB"
    apply_method = "pending-reboot"
  }

  tags = {
    Environment = var.environment
    Project     = "ai-platform"
  }
}

# Output database endpoint
output "db_endpoint" {
  value = aws_db_instance.ai_postgres.endpoint
}
```

### Step 2: Variable definitions
```hcl
# terraform/variables.tf
variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "vpc_id" {
  description = "VPC ID for database"
  type        = string
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "postgres"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}
```

### Step 3: CI/CD integration for IaC
```yaml
# .github/workflows/terraform.yml
name: Terraform Database Provisioning
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  terraform:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Terraform
      uses: hashicorp/setup-terraform@v2
      with:
        terraform_version: 1.4.6
    
    - name: Terraform Init
      run: terraform init
      working-directory: terraform
    
    - name: Terraform Validate
      run: terraform validate
      working-directory: terraform
    
    - name: Terraform Plan
      id: plan
      run: terraform plan -no-color -out=tfplan
      working-directory: terraform
      continue-on-error: true
    
    - name: Show Plan
      if: steps.plan.outcome == 'failure'
      run: terraform show tfplan
      working-directory: terraform
    
    - name: Terraform Apply
      if: github.ref == 'refs/heads/main' && github.event_name == 'push'
      run: terraform apply -auto-approve tfplan
      working-directory: terraform
      env:
        AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
        AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

## Section 3: Automated Operations

### Step 1: Monitoring and alerting
```python
# monitoring/alert_manager.py
import requests
import json
import time
from datetime import datetime

class DatabaseAlertManager:
    def __init__(self, alert_webhook_url):
        self.webhook_url = alert_webhook_url
        self.alert_rules = []
    
    def add_alert_rule(self, name, condition, severity="warning", description=""):
        """Add an alert rule"""
        self.alert_rules.append({
            'name': name,
            'condition': condition,
            'severity': severity,
            'description': description,
            'last_triggered': None
        })
    
    def check_alerts(self, metrics):
        """Check metrics against alert rules"""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            try:
                # Evaluate condition (simple eval for demo)
                if eval(rule['condition'], {}, metrics):
                    if rule['last_triggered'] is None or \
                       (datetime.now() - rule['last_triggered']).seconds > 300:  # 5 min cooldown
                    
                    alert = {
                        'rule': rule['name'],
                        'severity': rule['severity'],
                        'message': f"{rule['description']} - Current: {metrics}",
                        'timestamp': datetime.now().isoformat(),
                        'metrics': metrics
                    }
                    
                    triggered_alerts.append(alert)
                    rule['last_triggered'] = datetime.now()
                    
            except Exception as e:
                print(f"Error evaluating rule {rule['name']}: {e}")
        
        return triggered_alerts
    
    def send_alert(self, alert):
        """Send alert to webhook"""
        payload = {
            'text': f"[{alert['severity'].upper()}] {alert['rule']}: {alert['message']}",
            'attachments': [{
                'color': 'danger' if alert['severity'] == 'critical' else 'warning' if alert['severity'] == 'warning' else 'good',
                'title': alert['rule'],
                'text': alert['message'],
                'fields': [
                    {'title': 'Timestamp', 'value': alert['timestamp'], 'short': True},
                    {'title': 'Severity', 'value': alert['severity'].upper(), 'short': True}
                ]
            }]
        }
        
        try:
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
            print(f"Alert sent: {alert['rule']}")
        except Exception as e:
            print(f"Failed to send alert: {e}")
    
    def run_monitoring_loop(self, get_metrics_func, interval_seconds=60):
        """Run continuous monitoring loop"""
        print("Starting database monitoring...")
        
        while True:
            try:
                metrics = get_metrics_func()
                alerts = self.check_alerts(metrics)
                
                for alert in alerts:
                    self.send_alert(alert)
                
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                print("Monitoring stopped.")
                break
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(10)

# Usage example
def get_database_metrics():
    """Get current database metrics"""
    # This would query actual database metrics
    return {
        'cpu_usage_percent': 85,
        'memory_usage_percent': 92,
        'connections': 120,
        'slow_queries_per_minute': 15,
        'cache_hit_ratio': 0.85,
        'disk_io_ops_per_second': 1200
    }

alert_manager = DatabaseAlertManager("https://hooks.slack.com/services/XXX")

# Add alert rules
alert_manager.add_alert_rule(
    name="high_cpu_usage",
    condition="cpu_usage_percent > 90",
    severity="critical",
    description="High CPU usage detected"
)

alert_manager.add_alert_rule(
    name="low_cache_hit_ratio",
    condition="cache_hit_ratio < 0.8",
    severity="warning",
    description="Low cache hit ratio"
)

alert_manager.add_alert_rule(
    name="too_many_connections",
    condition="connections > 100",
    severity="warning",
    description="Too many database connections"
)

# Start monitoring
# alert_manager.run_monitoring_loop(get_database_metrics, interval_seconds=30)
```

### Step 2: Automated remediation
```python
# automation/remediation_engine.py
import subprocess
import os
import time
from typing import Dict, List

class RemediationEngine:
    def __init__(self):
        self.remediation_actions = {}
    
    def register_action(self, name, action_func, conditions=None):
        """Register a remediation action"""
        if conditions is None:
            conditions = []
        
        self.remediation_actions[name] = {
            'func': action_func,
            'conditions': conditions,
            'last_executed': None,
            'cooldown_seconds': 300  # 5 minute cooldown
        }
    
    def execute_remediation(self, action_name, context=None):
        """Execute a remediation action"""
        if action_name not in self.remediation_actions:
            raise ValueError(f"Action '{action_name}' not registered")
        
        action = self.remediation_actions[action_name]
        
        # Check cooldown
        if action['last_executed'] and \
           (time.time() - action['last_executed']) < action['cooldown_seconds']:
            print(f"Action '{action_name}' on cooldown")
            return False
        
        try:
            result = action['func'](context)
            action['last_executed'] = time.time()
            print(f"Action '{action_name}' executed successfully")
            return result
        except Exception as e:
            print(f"Action '{action_name}' failed: {e}")
            return False
    
    def check_and_remediate(self, metrics):
        """Check metrics and execute appropriate remediations"""
        executed_actions = []
        
        for name, action in self.remediation_actions.items():
            # Check conditions
            should_execute = True
            for condition in action['conditions']:
                if not condition(metrics):
                    should_execute = False
                    break
            
            if should_execute:
                success = self.execute_remediation(name, metrics)
                if success:
                    executed_actions.append(name)
        
        return executed_actions

# Example remediation actions
def restart_connection_pool(metrics):
    """Restart database connection pool"""
    print("Restarting connection pool...")
    # In production, this would call actual API or run command
    return True

def scale_up_database(metrics):
    """Scale up database instance"""
    print("Scaling up database instance...")
    # This would call cloud provider API
    return True

def clear_query_cache(metrics):
    """Clear query cache"""
    print("Clearing query cache...")
    # This would execute database command
    return True

# Register remediation actions
remediation_engine = RemediationEngine()

remediation_engine.register_action(
    "restart_connection_pool",
    restart_connection_pool,
    conditions=[
        lambda m: m.get('connections', 0) > 150,
        lambda m: m.get('slow_queries_per_minute', 0) > 20
    ]
)

remediation_engine.register_action(
    "scale_up_database",
    scale_up_database,
    conditions=[
        lambda m: m.get('cpu_usage_percent', 0) > 95,
        lambda m: m.get('memory_usage_percent', 0) > 95
    ]
)

remediation_engine.register_action(
    "clear_query_cache",
    clear_query_cache,
    conditions=[
        lambda m: m.get('cache_hit_ratio', 1.0) < 0.7,
        lambda m: m.get('slow_queries_per_minute', 0) > 10
    ]
)

# Usage example
def monitor_and_remediate():
    """Monitor and remediate automatically"""
    print("Starting auto-remediation monitoring...")
    
    while True:
        try:
            metrics = get_database_metrics()
            
            # Check and remediate
            actions = remediation_engine.check_and_remediate(metrics)
            
            if actions:
                print(f"Executed remediation actions: {actions}")
            
            time.sleep(60)  # Check every minute
            
        except KeyboardInterrupt:
            print("Auto-remediation stopped.")
            break
        except Exception as e:
            print(f"Error in auto-remediation: {e}")
            time.sleep(30)
```

## Section 4: Disaster Recovery

### Step 1: Backup strategy
```bash
# scripts/backup_database.sh
#!/bin/bash

# Database backup script for AI workloads

DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_USER=${DB_USER:-postgres}
DB_PASSWORD=${DB_PASSWORD:-password}
DB_NAME=${DB_NAME:-ai_db}
BACKUP_DIR=${BACKUP_DIR:-/backups}
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Full backup
echo "Creating full backup..."
pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -Fc > "$BACKUP_DIR/$DB_NAME-full-$DATE.dump"

# Verify backup
if [ $? -eq 0 ]; then
    echo "Full backup created successfully"
    
    # Upload to S3 (optional)
    if command -v aws &> /dev/null; then
        echo "Uploading to S3..."
        aws s3 cp "$BACKUP_DIR/$DB_NAME-full-$DATE.dump" "s3://ai-backups/$DB_NAME/full/$DATE/"
    fi
    
    # Clean old backups (keep last 30 days)
    find "$BACKUP_DIR" -name "*.dump" -mtime +30 -delete
else
    echo "Backup failed!"
    exit 1
fi

# Incremental backup (WAL archiving)
echo "Setting up WAL archiving..."
# This would configure PostgreSQL for WAL archiving
```

### Step 2: Restore procedure
```python
# scripts/restore_database.py
import subprocess
import os
import sys
from datetime import datetime

def restore_database(backup_file, target_db_name, target_host="localhost"):
    """Restore database from backup file"""
    
    print(f"Restoring database from {backup_file} to {target_db_name}")
    
    # Check backup file exists
    if not os.path.exists(backup_file):
        print(f"Backup file not found: {backup_file}")
        return False
    
    # Create target database
    create_cmd = f"psql -h {target_host} -U postgres -c 'CREATE DATABASE {target_db_name}'"
    try:
        subprocess.run(create_cmd, shell=True, check=True)
        print(f"Database {target_db_name} created")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create database: {e}")
        return False
    
    # Restore backup
    restore_cmd = f"pg_restore -h {target_host} -U postgres -d {target_db_name} {backup_file}"
    
    try:
        print("Starting restore...")
        start_time = datetime.now()
        
        # Use subprocess with real-time output
        process = subprocess.Popen(
            restore_cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Stream output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        rc = process.poll()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if rc == 0:
            print(f"Restore completed successfully in {duration}")
            return True
        else:
            stderr = process.stderr.read()
            print(f"Restore failed: {stderr}")
            return False
            
    except Exception as e:
        print(f"Restore error: {e}")
        return False

def validate_restore(target_db_name, target_host="localhost"):
    """Validate restored database"""
    print(f"Validating database {target_db_name}")
    
    # Check basic connectivity
    try:
        conn = psycopg2.connect(
            host=target_host,
            database=target_db_name,
            user="postgres",
            password="password"
        )
        cursor = conn.cursor()
        
        # Check required tables
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
        table_count = cursor.fetchone()[0]
        
        print(f"Found {table_count} tables")
        
        # Check sample data
        cursor.execute("SELECT COUNT(*) FROM users LIMIT 1")
        user_count = cursor.fetchone()[0]
        
        print(f"Found {user_count} users")
        
        conn.close()
        print("Database validation passed ✅")
        return True
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return False

# Usage example
if __name__ == "__main__":
    backup_file = "/backups/ai_db-full-20240215_103000.dump"
    
    # Restore
    if restore_database(backup_file, "ai_db_restored"):
        # Validate
        if validate_restore("ai_db_restored"):
            print("Disaster recovery successful!")
        else:
            print("Restore validation failed")
    else:
        print("Restore failed")
```

## Section 5: AI-Specific Automation

### Step 1: Feature store automation
```python
# automation/feature_store_automation.py
import schedule
import time
import subprocess
from datetime import datetime, timedelta

class FeatureStoreAutomation:
    def __init__(self):
        self.jobs = []
    
    def schedule_feature_update(self, feature_name, schedule_time, command):
        """Schedule feature update job"""
        job = {
            'feature_name': feature_name,
            'schedule_time': schedule_time,
            'command': command,
            'last_run': None,
            'status': 'scheduled'
        }
        
        self.jobs.append(job)
        return job
    
    def run_scheduled_jobs(self):
        """Run scheduled jobs that are due"""
        now = datetime.now()
        executed_jobs = []
        
        for job in self.jobs:
            # Parse schedule time (simple format: "HH:MM")
            if isinstance(job['schedule_time'], str):
                hour, minute = map(int, job['schedule_time'].split(':'))
                scheduled_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                # If scheduled time is in the past today, consider it for tomorrow
                if scheduled_time < now:
                    scheduled_time = scheduled_time + timedelta(days=1)
                
                if abs((now - scheduled_time).total_seconds()) < 60:  # Within 1 minute
                    print(f"Running feature update: {job['feature_name']}")
                    
                    try:
                        # Execute command
                        result = subprocess.run(job['command'], shell=True, capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            job['last_run'] = now
                            job['status'] = 'success'
                            executed_jobs.append(job['feature_name'])
                            print(f"Feature {job['feature_name']} updated successfully")
                        else:
                            job['status'] = 'failed'
                            print(f"Feature {job['feature_name']} update failed: {result.stderr}")
                            
                    except Exception as e:
                        job['status'] = 'failed'
                        print(f"Feature {job['feature_name']} execution error: {e}")
        
        return executed_jobs
    
    def start_scheduler(self):
        """Start the scheduler loop"""
        print("Starting feature store automation scheduler...")
        
        while True:
            try:
                self.run_scheduled_jobs()
                time.sleep(30)  # Check every 30 seconds
            except KeyboardInterrupt:
                print("Scheduler stopped.")
                break
            except Exception as e:
                print(f"Scheduler error: {e}")
                time.sleep(60)

# Usage example
automation = FeatureStoreAutomation()

# Schedule daily feature updates
automation.schedule_feature_update(
    feature_name="user_engagement_score",
    schedule_time="02:00",  # 2 AM
    command="python scripts/update_user_engagement.py"
)

automation.schedule_feature_update(
    feature_name="real_time_click_features",
    schedule_time="00:05",  # 12:05 AM
    command="python scripts/update_real_time_features.py"
)

automation.schedule_feature_update(
    feature_name="weekly_model_features",
    schedule_time="03:00",  # 3 AM on Sunday
    command="python scripts/update_weekly_model_features.py"
)

# Start scheduler
# automation.start_scheduler()
```

### Step 2: Model registry automation
```python
# automation/model_registry_automation.py
import requests
import json
import time
from datetime import datetime

class ModelRegistryAutomation:
    def __init__(self, registry_url, api_token):
        self.registry_url = registry_url
        self.api_token = api_token
        self.headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }
    
    def promote_model(self, model_id, target_stage):
        """Promote model to target stage"""
        url = f"{self.registry_url}/api/models/{model_id}/promote"
        
        payload = {
            'target_stage': target_stage,
            'reason': f'Automated promotion at {datetime.now().isoformat()}'
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Failed to promote model {model_id}: {e}")
            return None
    
    def evaluate_model_performance(self, model_id):
        """Evaluate model performance metrics"""
        url = f"{self.registry_url}/api/models/{model_id}/metrics"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            metrics = response.json()
            
            # Calculate performance score
            performance_score = 0
            if 'accuracy' in metrics:
                performance_score += metrics['accuracy'] * 0.4
            if 'precision' in metrics:
                performance_score += metrics['precision'] * 0.3
            if 'recall' in metrics:
                performance_score += metrics['recall'] * 0.3
            
            return {
                'model_id': model_id,
                'performance_score': performance_score,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Failed to evaluate model {model_id}: {e}")
            return None
    
    def auto_promote_best_model(self, experiment_id, threshold=0.85):
        """Automatically promote the best model above threshold"""
        url = f"{self.registry_url}/api/experiments/{experiment_id}/models"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            models = response.json()
            
            best_model = None
            best_score = 0
            
            for model in models:
                evaluation = self.evaluate_model_performance(model['id'])
                if evaluation and evaluation['performance_score'] > best_score:
                    best_score = evaluation['performance_score']
                    best_model = model
            
            if best_model and best_score >= threshold:
                print(f"Promoting best model {best_model['id']} with score {best_score:.3f}")
                result = self.promote_model(best_model['id'], 'production')
                return result
            else:
                print(f"No model meets threshold ({threshold}). Best score: {best_score:.3f}")
                return None
                
        except Exception as e:
            print(f"Failed to auto-promote: {e}")
            return None

# Usage example
registry_automation = ModelRegistryAutomation(
    registry_url="https://model-registry.example.com",
    api_token="your-api-token"
)

# Auto-promote best model every hour
def auto_promote_job():
    print("Checking for best model to promote...")
    registry_automation.auto_promote_best_model("experiment-123", threshold=0.85)

# Schedule job
# schedule.every().hour.do(auto_promote_job)
```

## Section 6: Production Readiness Framework

### Step 1: Comprehensive automation framework
```python
# automation/production_readiness.py
from typing import Dict, List, Optional
import json
import time
import subprocess

class ProductionReadinessFramework:
    def __init__(self):
        self.checks = []
        self.automations = []
        self.monitoring = []
    
    def add_readiness_check(self, name, check_func, critical=False):
        """Add a readiness check"""
        self.checks.append({
            'name': name,
            'func': check_func,
            'critical': critical,
            'last_result': None,
            'last_run': None
        })
    
    def add_automation(self, name, automation_func, trigger_conditions=None):
        """Add an automation"""
        if trigger_conditions is None:
            trigger_conditions = []
        
        self.automations.append({
            'name': name,
            'func': automation_func,
            'trigger_conditions': trigger_conditions,
            'last_executed': None
        })
    
    def add_monitoring_metric(self, name, metric_func, alert_threshold=None):
        """Add a monitoring metric"""
        self.monitoring.append({
            'name': name,
            'func': metric_func,
            'alert_threshold': alert_threshold,
            'history': []
        })
    
    def run_readiness_checks(self):
        """Run all readiness checks"""
        results = []
        
        for check in self.checks:
            try:
                result = check['func']()
                check['last_result'] = result
                check['last_run'] = time.time()
                
                status = "PASS" if result.get('passed', False) else "FAIL"
                results.append({
                    'check': check['name'],
                    'status': status,
                    'message': result.get('message', ''),
                    'critical': check['critical']
                })
                
            except Exception as e:
                results.append({
                    'check': check['name'],
                    'status': "ERROR",
                    'message': f"Exception: {e}",
                    'critical': check['critical']
                })
        
        return results
    
    def check_production_readiness(self):
        """Check overall production readiness"""
        checks = self.run_readiness_checks()
        
        critical_failures = [c for c in checks if c['critical'] and c['status'] != 'PASS']
        total_failures = [c for c in checks if c['status'] != 'PASS']
        
        if critical_failures:
            return {
                'ready': False,
                'status': 'CRITICAL_FAILURE',
                'message': f"Critical failures: {len(critical_failures)}",
                'details': checks
            }
        elif total_failures:
            return {
                'ready': False,
                'status': 'FAILURE',
                'message': f"Failures: {len(total_failures)}",
                'details': checks
            }
        else:
            return {
                'ready': True,
                'status': 'READY',
                'message': "All readiness checks passed",
                'details': checks
            }
    
    def run_automations(self, context=None):
        """Run automations based on context"""
        executed = []
        
        for automation in self.automations:
            should_run = True
            for condition in automation['trigger_conditions']:
                if not condition(context):
                    should_run = False
                    break
            
            if should_run:
                try:
                    result = automation['func'](context)
                    automation['last_executed'] = time.time()
                    executed.append({
                        'name': automation['name'],
                        'result': result,
                        'timestamp': time.time()
                    })
                except Exception as e:
                    print(f"Automation {automation['name']} failed: {e}")
        
        return executed

# Example readiness checks
def check_database_connectivity():
    """Check database connectivity"""
    try:
        # Simulate database connection
        time.sleep(0.1)
        return {'passed': True, 'message': 'Database connected'}
    except Exception as e:
        return {'passed': False, 'message': f'Database connection failed: {e}'}

def check_feature_store_health():
    """Check feature store health"""
    try:
        # Simulate feature store health check
        return {'passed': True, 'message': 'Feature store healthy'}
    except Exception as e:
        return {'passed': False, 'message': f'Feature store unhealthy: {e}'}

def check_model_registry():
    """Check model registry"""
    try:
        return {'passed': True, 'message': 'Model registry available'}
    except Exception as e:
        return {'passed': False, 'message': f'Model registry unavailable: {e}'}

# Initialize framework
pr_framework = ProductionReadinessFramework()

# Add readiness checks
pr_framework.add_readiness_check("database_connectivity", check_database_connectivity, critical=True)
pr_framework.add_readiness_check("feature_store_health", check_feature_store_health, critical=True)
pr_framework.add_readiness_check("model_registry", check_model_registry, critical=True)
pr_framework.add_readiness_check("monitoring_system", lambda: {'passed': True, 'message': 'Monitoring system active'}, critical=False)

# Add automations
pr_framework.add_automation(
    "auto_scale_database",
    lambda ctx: print("Auto-scaling database..."),
    trigger_conditions=[
        lambda ctx: ctx.get('cpu_usage_percent', 0) > 90 if ctx else False
    ]
)

pr_framework.add_automation(
    "backup_database",
    lambda ctx: print("Running database backup..."),
    trigger_conditions=[
        lambda ctx: datetime.now().hour == 2 and datetime.now().minute < 5  # 2:00-2:05 AM
    ]
)

# Usage example
def deploy_to_production():
    """Deploy to production with readiness checks"""
    print("Checking production readiness...")
    
    readiness = pr_framework.check_production_readiness()
    
    if readiness['ready']:
        print("✅ Production ready!")
        
        # Run pre-deploy automations
        print("Running pre-deploy automations...")
        pr_framework.run_automations({'cpu_usage_percent': 85})
        
        # Deploy
        print("Deploying to production...")
        # Actual deployment commands here
        
        print("Deployment successful!")
    else:
        print(f"❌ Not ready for production: {readiness['message']}")
        for detail in readiness['details']:
            print(f"  - {detail['check']}: {detail['status']} - {detail['message']}")
        return False

# Run readiness check
# deploy_to_production()
```

## Hands-on Exercises

### Exercise 1: Implement database CI/CD
1. Set up Alembic for your database
2. Create GitHub Actions workflow for database CI/CD
3. Implement schema validation script
4. Test with pull requests

### Exercise 2: Infrastructure as Code
1. Write Terraform configuration for your database
2. Set up CI/CD for Terraform
3. Test with different environments (dev, staging, prod)
4. Implement parameter groups for AI workloads

### Exercise 3: Automated operations
1. Implement alert manager with your monitoring system
2. Create remediation actions for common issues
3. Set up auto-remediation monitoring
4. Test with simulated failures

### Exercise 4: Disaster recovery
1. Implement backup script for your database
2. Create restore procedure
3. Test backup and restore process
4. Set up automated backup scheduling

### Exercise 5: AI-specific automation
1. Implement feature store automation
2. Set up model registry auto-promotion
3. Create comprehensive production readiness framework
4. Integrate with your CI/CD pipeline

## Best Practices Summary

1. **Automate Everything**: CI/CD, IaC, monitoring, remediation
2. **Test Relentlessly**: Test backups, restores, failovers regularly
3. **Monitor Proactively**: Alert before problems become critical
4. **Document Thoroughly**: Clear runbooks for all automation
5. **Review Regularly**: Monthly review of automation effectiveness

This tutorial provides practical, hands-on experience with database DevOps and automation specifically for AI/ML systems. Complete all exercises to master these critical operational skills.