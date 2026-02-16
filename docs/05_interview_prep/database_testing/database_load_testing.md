# Database Performance Testing Guide

## Overview

This comprehensive guide covers database load testing methodologies, benchmark tools configuration, custom benchmark design, and result analysis. It provides practical examples for PostgreSQL (pgbench) and MySQL (sysbench) along with frameworks for custom benchmarks.

## Table of Contents

1. [Load Testing Fundamentals](#load-testing-fundamentals)
2. [pgbench for PostgreSQL](#pgbench-for-postgresql)
3. [sysbench for MySQL](#sysbench-for-mysql)
4. [Custom Benchmark Design](#custom-benchmark-design)
5. [Test Data Generation](#test-data-generation)
6. [Result Analysis and Reporting](#result-analysis-and-reporting)

---

## Load Testing Fundamentals

### What is Database Load Testing?

Database load testing evaluates database performance under various workload conditions, measuring:

- **Throughput**: Maximum sustainable load (transactions/queries per second)
- **Latency**: Response time distribution across different load levels
- **Concurrency**: Performance degradation as user count increases
- **Scalability**: Performance changes with growing data volume
- **Resource Utilization**: CPU, memory, I/O consumption patterns

### Load Testing Methodology

#### 1. Establish Baseline Performance

```markdown
Before optimization or configuration changes:
1. Run standard benchmark at known scale
2. Document all system parameters
3. Record multiple runs for statistical validity
4. Establish performance envelope (min, max, average)
```

#### 2. Define Test Scenarios

| Scenario | Description | Target Metrics |
|----------|-------------|----------------|
| Steady Load | Constant user count | TPS, avg latency |
| Ramp Up | Gradual increase in users | Breaking point |
| Spike Test | Sudden load increase | Recovery time |
| Soak Test | Extended duration | Memory leaks, degradation |
| Peak Load | Maximum expected load | SLA compliance |

#### 3. Test Execution Framework

```python
# Database Load Testing Framework
class DatabaseLoadTest:
    def __init__(self, config):
        self.db_config = config
        self.results = []
        
    def run_scenario(self, scenario):
        """Execute a load test scenario"""
        self.warmup(scenario.warmup_time)
        self.execute_load(scenario)
        self.cooldown(scenario.cooldown_time)
        return self.collect_metrics()
    
    def warmup(self, duration):
        """Warm up caches and connections"""
        # Run low-level load for specified duration
        pass
    
    def execute_load(self, scenario):
        """Execute the actual load test"""
        # Ramp up to target concurrency
        # Maintain load for test duration
        # Record metrics at intervals
        
    def cooldown(self, duration):
        """Cooldown period"""
        # Gradual reduction in load
        pass
```

---

## pgbench for PostgreSQL

### Overview

pgbench is PostgreSQL's built-in benchmarking tool. It provides TPC-B-like testing and supports custom scripts for various workloads.

### Installation and Setup

#### Installation

```bash
# Ubuntu/Debian
sudo apt-get install postgresql-client

# CentOS/RHEL
sudo yum install postgresql

# From source (if needed)
git clone https://github.com/postgres/postgres.git
cd postgres
./configure --prefix=/usr/local/pgsql
make -j$(nproc)
sudo make install
```

#### Verify Installation

```bash
pgbench --version
# Output: pgbench (PostgreSQL) 15.2
```

### Basic pgbench Workflow

#### Step 1: Create Test Database

```bash
# Create a fresh database for testing
createdb -h localhost -U postgres benchmark_db

# Or with connection string
psql "postgres://user:password@localhost:5432/benchmark_db"
```

#### Step 2: Initialize Tables

```bash
# Initialize with default TPC-B-like schema
pgbench -i -s 10 postgres://user:pass@localhost:5432/benchmark_db

# Options:
# -i: Initialize tables
# -s: Scale factor (10 = 10 * 100,000 rows in pgbench_branches)
# -F: Fillfactor (default 100)
# --foreign-keys: Add foreign key constraints
# --index-tablespace: Specify tablespace for indexes
# --tablespace: Specify tablespace for tables
```

#### Step 3: Run Basic Benchmark

```bash
# Simple 60-second benchmark with 10 concurrent clients
pgbench -c 10 -T 60 postgres://user:pass@localhost:5432/benchmark_db

# Options explained:
# -c: Number of concurrent database sessions
# -j: Number of worker threads (for pgbench itself)
# -T: Duration in seconds
# -t: Transactions per client (instead of -T)
# -M: Connection mode (simple, extended, prepared)
# -r: Report latencies
```

#### Step 4: Sample Output

```bash
starting vacuum...end.
transaction type: <builtin: TPC-B (like)>
scaling factor: 10
query mode: simple
number of clients: 10
number of threads: 2
duration: 60 s
number of transactions actually processed: 28450
latency average = 21.084 ms
latency stddev = 15.234 ms
initial connection time = 12.345 ms
tps = 474.166667 (without initial connection time)

# Latency percentiles (with -r flag):
percentile = 50
percentile = 75
percentile = 90
percentile = 95
percentile = 99
```

### pgbench Built-in Scripts

#### Available Built-in Scripts

| Script | Description | Use Case |
|--------|-------------|----------|
| TPC-B (default) | Simple transaction with updates | General throughput |
| TPC-C-like | Order processing simulation | OLTP testing |
| Select-only | Read-only transactions | Read performance |
| Insert-only | Bulk insert testing | Write performance |

#### Using Different Scripts

```bash
# TPC-B (default)
pgbench -c 10 -T 60 -S postgres://user:pass@localhost:5432/benchmark_db

# TPC-C-like (built-in since PostgreSQL 14)
pgbench -c 10 -T 60 -M prepared -T 60 \
    postgres://user:pass@localhost:5432/benchmark_db \
    -N  # Use TPC-C-like script

# Select-only (read benchmark)
pgbench -c 10 -T 60 -S postgres://user:pass@localhost:5432/benchmark_db

# Custom with built-in TPC-C
# First, create the TPC-C schema manually or use custom script
```

### Custom pgbench Scripts

#### Creating Custom Scripts

```sql
-- File: custom_workload.sql
-- Custom workload for your application

-- 1. Simple read query
\set uid random(1, 100000)
SELECT * FROM accounts WHERE id = :uid;

-- 2. Read-write transaction
\set uid random(1, 100000)
\set amount random(1, 1000)
BEGIN;
UPDATE accounts SET balance = balance + :amount WHERE id = :uid;
INSERT INTO transactions (account_id, amount, created_at) 
VALUES (:uid, :amount, NOW());
COMMIT;

-- 3. Complex analytical query
SELECT 
    date_trunc('day', created_at) as day,
    COUNT(*) as count,
    SUM(amount) as total
FROM transactions
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY 1
ORDER BY 1;
```

#### Running Custom Scripts

```bash
# Run with custom script
pgbench -c 10 -T 60 -f custom_workload.sql \
    postgres://user:pass@localhost:5432/benchmark_db

# Multiple scripts with weights
pgbench -c 10 -T 60 \
    -f read_queries.sql=70 \
    -f write_queries.sql=30 \
    postgres://user:pass@localhost:5432/benchmark_db
```

### Advanced pgbench Configuration

#### Comprehensive Benchmark Configuration

```bash
#!/bin/bash
# Advanced pgbench configuration script

DB="postgres://user:pass@localhost:5432/benchmark_db"
RESULTS_DIR="./benchmark_results"
SCALE=100
CLIENTS="1 5 10 25 50 100"

mkdir -p $RESULTS_DIR

# Initialize database
echo "Initializing database at scale $SCALE..."
pgbench -i -s $SCALE $DB

# Run tests at different concurrency levels
for clients in $CLIENTS; do
    echo "Running benchmark with $clients clients..."
    
    # Run with latency reporting
    pgbench -c $clients -j $clients -T 300 -r \
        $DB 2>&1 | tee "$RESULTS_DIR/pgbench_${clients}_clients.log"
    
    # Capture extended statistics
    psql $DB -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 20;" \
        > "$RESULTS_DIR/queries_${clients}_clients.txt" 2>&1
    
    # Wait between runs
    sleep 30
done

echo "Benchmark complete. Results in $RESULTS_DIR"
```

#### pgbench Variables and Randomization

```sql
-- Common pgbench variable functions

-- Random integer in range
\set amount random(1, 10000)

-- Random from list
\set status random_from('pending', 'processing', 'completed')

-- Random with Gaussian distribution
\set id random_gaussian(1, 10000, 2.0)

-- Random within exponential distribution  
\set id random_exponential(1, 10000, 0.5)

-- Set operation
\set id random(1, 100000)
SELECT * FROM accounts WHERE id = :id;
```

### pgbench Performance Tuning

#### Configuration for Better Results

```bash
# Pre-connection setup
export PGCONNECT_TIMEOUT=10
export PGDATABASE=benchmark_db

# Run benchmark multiple times
for i in 1 2 3; do
    echo "Run $i:"
    pgbench -c 10 -T 60 -n $DB
done
```

#### PostgreSQL Configuration for Benchmarking

```sql
-- postgresql.conf settings for benchmarking
-- NOTE: These are for testing only, not production

-- Memory settings (adjust based on available RAM)
shared_buffers = 8GB           -- 25% of RAM
work_mem = 256MB               -- Per-sort/hash operation
maintenance_work_mem = 2GB     -- For VACUUM, CREATE INDEX

-- Checkpoint settings
checkpoint_completion_target = 0.9
wal_buffers = 64MB
max_wal_size = 4GB
min_wal_size = 1GB

-- Query planner
random_page_cost = 1.1        -- For SSD
effective_cache_size = 24GB   -- 75% of RAM

-- Logging (minimal during benchmark)
logging_collector = off
log_destination = 'stderr'

-- Connection settings
max_connections = 200

-- Parallel queries
max_worker_processes = 16
max_parallel_workers_per_gather = 8
```

---

## sysbench for MySQL

### Overview

sysbench is a versatile benchmarking tool that supports multiple database backends including MySQL, PostgreSQL, and Oracle. It excels at OLTP testing.

### Installation

#### Installation Methods

```bash
# Ubuntu/Debian
sudo apt-get install sysbench

# CentOS/RHEL
sudo yum install sysbench

# From source (latest version)
git clone https://github.com/akopytov/sysbench.git
cd sysbench
./autogen.sh
./configure
make -j$(nproc)
sudo make install
```

#### Verify Installation

```bash
sysbench --version
# sysbench 1.0.20
```

### sysbench for MySQL/MariaDB

#### Step 1: Prepare the Test Database

```bash
# Create test database
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS sbtest;"

# Initialize tables with sysbench
sysbench /usr/share/sysbench/oltp_insert.lua \
    --mysql-host=localhost \
    --mysql-port=3306 \
    --mysql-user=root \
    --mysql-password=password \
    --mysql-db=sbtest \
    --table_size=100000 \
    --tables=10 \
    prepare
```

#### Step 2: Run OLTP Benchmark

```bash
# Basic OLTP benchmark
sysbench /usr/share/sysbench/oltp_read_write.lua \
    --mysql-host=localhost \
    --mysql-port=3306 \
    --mysql-user=root \
    --mysql-password=password \
    --mysql-db=sbtest \
    --table-size=100000 \
    --tables=10 \
    --threads=10 \
    --time=60 \
    run

# Options:
# --threads: Number of concurrent threads
# --time: Duration in seconds
# --report-interval: Progress report interval
# --table-size: Rows per table
# --tables: Number of tables
```

#### Step 3: Sample Output

```bash
sysbench 1.0.20 (using system LuaJIT 2.1.0-beta3)

Running the test with following options:
Number of threads: 10
Initializing random number generator from current time


Initializing worker threads...

Threads started!

SQL statistics:
    queries performed:
        read:                            583420
        write:                           166692
        other:                           83346
        total:                           833458
    transactions:                        41672  (694.53 per sec.)
    queries:                              833458 (13890.05 per sec.)
    ignored errors:                      0      (0.00 per sec.)
    reconnects:                           0      (0.00 per sec.)

General statistics:
    total time:                          60.0023s
    total number of events:              41672

Latency (ms):
         min:                                  4.21
         avg:                                 14.38
         max:                                134.56
         95th percentile:                     22.01
         sum:                            599383.37

Threads fairness:
    events (avg/stddev):           4167.2000/45.23
    execution time (avg/stddev):    59.9383/0.01
```

### Available sysbench Test Scripts

#### Built-in Test Scripts

| Script | Description | Use Case |
|--------|-------------|----------|
| oltp_read_only | Read-only transactions | Read performance |
| oltp_read_write | Read-write transactions | Mixed workload |
| oltp_write_only | Write-only transactions | Write performance |
| oltp_insert | Bulk insert testing | Insert throughput |
| select_random_points | Point selects | Key-value access |
| delete_random_ranges | Range deletes | Cleanup operations |

#### Using Different Test Scripts

```bash
# Read-only benchmark
sysbench /usr/share/sysbench/oltp_read_only.lua \
    --threads=20 --time=60 --report-interval=5 \
    --mysql-host=localhost --mysql-db=sbtest \
    prepare
sysbench /usr/share/sysbench/oltp_read_only.lua \
    --threads=20 --time=60 run

# Write-only benchmark  
sysbench /usr/share/sysbench/oltp_write_only.lua \
    --threads=20 --time=60 \
    --mysql-host=localhost --mysql-db=sbtest run

# Point select benchmark (high throughput)
sysbench /usr/share/sysbench/select_random_points.lua \
    --threads=50 --time=60 \
    --mysql-host=localhost --mysql-db=sbtest run
```

### Custom sysbench Scripts

#### Creating a Custom Test

```lua
-- File: custom_oltp.lua
-- Custom sysbench script for your workload

require("sysbench")

-- Define custom commands
sysbench.hooks.before_init = function(thread)
    thread.conn:[[
        CREATE TABLE IF NOT EXISTS custom_test (
            id INT AUTO_INCREMENT PRIMARY KEY,
            data VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ]]
end

-- Custom transaction
local function transaction()
    local rs = thread.conn:query("SELECT COUNT(*) FROM custom_test")
    local id = sysbench.rand.uniform(1, 10000)
    
    thread.conn:query("INSERT INTO custom_test (data) VALUES ('test')")
    thread.conn:query("UPDATE custom_test SET data = 'updated' WHERE id = " .. id)
    thread.conn:query("SELECT * FROM custom_test WHERE id = " .. id)
end

-- Register the test
sysbench.hooks.transaction = function()
    transaction()
end

-- Initialization
sysbench.hooks.init = function()
    -- Prepare statements if needed
end
```

#### Running Custom Script

```bash
sysbench custom_oltp.lua \
    --threads=10 --time=60 \
    --mysql-host=localhost --mysql-db=sbtest \
    --mysql-user=root --mysql-password=password \
    prepare

sysbench custom_oltp.lua \
    --threads=10 --time=60 \
    --mysql-host=localhost --mysql-db=sbtest \
    --mysql-user=root --mysql-password=password \
    run
```

### sysbench for PostgreSQL

```bash
# Install PostgreSQL driver
# sysbench needs postgresql driver support

# Test PostgreSQL
sysbench /usr/share/sysbench/oltp_read_write.lua \
    --db-driver=pgsql \
    --pgsql-host=localhost \
    --pgsql-port=5432 \
    --pgsql-user=postgres \
    --pgsql-password=password \
    --pgsql-db=sbtest \
    prepare

sysbench /usr/share/sysbench/oltp_read_write.lua \
    --db-driver=pgsql \
    --pgsql-host=localhost \
    --pgsql-port=5432 \
    --pgsql-user=postgres \
    --pgsql-db=sbtest \
    --threads=10 --time=60 run
```

---

## Custom Benchmark Design

### When to Create Custom Benchmarks

Custom benchmarks are necessary when:

- Standard benchmarks don't represent your workload
- Testing specific application patterns
- Validating performance after schema changes
- Comparing database configurations
- Reproducing production issues

### Designing Your Benchmark

#### Step 1: Analyze Production Workload

```sql
-- Capture production query patterns
-- PostgreSQL
SELECT query, calls, mean_time, total_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 50;

-- MySQL
SELECT * FROM performance_schema.events_statements_summary_by_digest
ORDER BY SUM_TIMER_WAIT DESC
LIMIT 50;
```

#### Step 2: Define Metrics

```python
# Define your performance requirements
PERFORMANCE_REQUIREMENTS = {
    'throughput': {
        'min_tps': 1000,
        'target_tps': 5000,
    },
    'latency': {
        'p50_max_ms': 50,
        'p95_max_ms': 200,
        'p99_max_ms': 500,
    },
    'availability': {
        'success_rate': 0.999,
    }
}
```

#### Step 3: Create Representative Workload

```python
# Custom benchmark for e-commerce database
class EcommerceBenchmark:
    def __init__(self, db_config):
        self.weights = {
            'product_search': 30,
            'view_product': 25,
            'add_to_cart': 15,
            'checkout': 10,
            'update_inventory': 20,
        }
    
    def product_search(self):
        """30% of traffic"""
        query = """
            SELECT * FROM products 
            WHERE name ILIKE %s 
            AND category_id = %s
            LIMIT 20
        """
        return query, ('%shoe%', self.random_category())
    
    def view_product(self):
        """25% of traffic"""
        query = """
            SELECT p.*, i.quantity as inventory
            FROM products p
            LEFT JOIN inventory i ON p.id = i.product_id
            WHERE p.id = %s
        """
        return query, (self.random_product_id(),)
    
    def checkout(self):
        """10% of traffic - complex transaction"""
        transaction = [
            ("BEGIN", ()),
            ("INSERT INTO orders (customer_id, total) VALUES (%s, %s)", 
             (self.current_customer(), self.cart_total())),
            ("UPDATE inventory SET quantity = quantity - 1 WHERE product_id = %s",
             (self.random_product_id(),)),
            ("COMMIT", ()),
        ]
        return transaction
    
    def run(self, concurrency=10, duration=60):
        """Execute benchmark"""
        # Implementation
        pass
```

### Benchmark Configuration Files

#### YAML Configuration

```yaml
# benchmark_config.yaml
database:
  type: postgresql
  host: localhost
  port: 5432
  name: benchmark_db
  user: benchmark_user
  password: secret
  
workload:
  name: "E-commerce Mixed Workload"
  description: "Simulates typical e-commerce traffic patterns"
  
  # Query weights (must sum to 100)
  weights:
    read_product: 30
    search_products: 25
    add_to_cart: 15
    checkout: 10
    update_inventory: 20
    
  # Test parameters
  parameters:
    scale_factor: 1000  # Number of products
    warmup_seconds: 30
    test_seconds: 300
    cooldown_seconds: 10
    
  # Concurrency levels to test
  concurrency:
    - 5
    - 10
    - 25
    - 50
    - 100
    
thresholds:
  max_p99_latency_ms: 500
  min_success_rate: 0.995
  min_tps: 1000
  
output:
  format: json
  path: ./results/
  save_queries: true
  save_statistics: true
```

#### JSON Configuration

```json
{
  "benchmark": {
    "name": "Order Processing System",
    "version": "1.0"
  },
  "database": {
    "driver": "postgresql",
    "connection_string": "postgresql://user:pass@localhost:5432/orders",
    "pool_size": 20
  },
  "scenarios": [
    {
      "name": "Steady Load",
      "duration_seconds": 300,
      "rampup_seconds": 30,
      "concurrency": 50,
      "queries": [
        {"weight": 40, "query": "get_order_by_id"},
        {"weight": 30, "query": "list_customer_orders"},
        {"weight": 20, "query": "create_order"},
        {"weight": 10, "query": "update_order_status"}
      ]
    },
    {
      "name": "Peak Load",
      "duration_seconds": 60,
      "rampup_seconds": 10,
      "concurrency": 200,
      "queries": [
        {"weight": 50, "query": "get_order_by_id"},
        {"weight": 50, "query": "create_order"}
      ]
    }
  ]
}
```

---

## Test Data Generation

### Data Generation Strategies

#### Using Generated Data

```bash
# pgbench can generate test data
pgbench -i -s 1000 mydb  # Creates ~1M rows

# For larger datasets, use pg_dump or custom scripts
```

#### Custom Data Generation Script

```python
#!/usr/bin/env python3
"""Generate test data for database benchmarking"""

import random
import string
from datetime import datetime, timedelta
import psycopg2
import csv

class TestDataGenerator:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = psycopg2.connect(**db_config)
    
    def generate_customers(self, count=100000):
        """Generate customer data"""
        cursor = self.conn.cursor()
        
        print(f"Generating {count} customers...")
        
        batch_size = 1000
        for i in range(0, count, batch_size):
            batch = []
            for j in range(batch_size):
                customer_id = i + j
                batch.append((
                    f'CUST{customer_id:08d}',
                    f'Customer {customer_id}',
                    random.choice(['ACTIVE', 'INACTIVE', 'SUSPENDED']),
                    random.uniform(100, 10000),
                    datetime.now() - timedelta(days=random.randint(0, 365))
                ))
            
            cursor.executemany("""
                INSERT INTO customers (customer_code, name, status, credit_limit, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """, batch)
            
            if i % 10000 == 0:
                self.conn.commit()
                print(f"  Progress: {i}/{count}")
        
        self.conn.commit()
        print("Customers generated successfully")
    
    def generate_orders(self, count=1000000):
        """Generate order data with realistic distribution"""
        cursor = self.conn.cursor()
        
        print(f"Generating {count} orders...")
        
        # Use COPY for faster bulk insert
        with open('orders_temp.csv', 'w') as f:
            writer = csv.writer(f)
            for i in range(count):
                # Realistic: More recent orders
                days_ago = int(random.expovariate(0.01)) % 365
                order_date = datetime.now() - timedelta(days=days_ago)
                
                writer.writerow([
                    random.randint(1, 100000),
                    order_date,
                    random.choice(['PENDING', 'PROCESSING', 'SHIPPED', 'DELIVERED']),
                    random.uniform(10, 1000)
                ])
                
                if i % 100000 == 0:
                    print(f"  Progress: {i}/{count}")
        
        print("Loading orders to database...")
        with open('orders_temp.csv', 'r') as f:
            cursor.copy_from(f, 'orders', sep=',')
        
        self.conn.commit()
        print("Orders generated successfully")
    
    def generate_skewed_data(self, table, skew_ratio=0.8):
        """Generate data with realistic hot/cold distribution"""
        # 80% of access to 20% of data (hot data)
        # This simulates real-world access patterns
        
        cursor = self.conn.cursor()
        
        # Create hot data (recent, popular items)
        hot_count = int(0.2 * 100000)  # 20% of total
        cold_count = 100000 - hot_count
        
        print(f"Generating {skew_ratio*100}% skewed data...")
        
        # Insert hot data first
        hot_data = [(i, True) for i in range(hot_count)]
        cursor.executemany(f"""
            INSERT INTO {table} (id, is_hot) VALUES (%s, %s)
        """, hot_data)
        
        # Then cold data
        cold_data = [(i + hot_count, False) for i in range(cold_count)]
        cursor.executemany(f"""
            INSERT INTO {table} (id, is_hot) VALUES (%s, %s)
        """, cold_data)
        
        self.conn.commit()

# Usage
if __name__ == '__main__':
    generator = TestDataGenerator({
        'host': 'localhost',
        'database': 'benchmark_db',
        'user': 'postgres',
        'password': 'password'
    })
    
    # Generate test data
    generator.generate_customers(100000)
    generator.generate_orders(1000000)
    generator.generate_skewed_data('products')
```

### Generating Large Datasets

```bash
# Using pg_dump for large datasets
# Export existing production data (anonymized)
pg_dump -h production-db -U user -d production_db \
    --data-only --format=custom \
    --table=orders \
    --where="created_at > '2023-01-01'" | \
    pg_restore -h benchmark-db -d benchmark_db

# Using COPY for bulk data
COPY orders(id, customer_id, total, status, created_at) 
FROM '/path/to/orders.csv' 
WITH (FORMAT csv, HEADER true);
```

---

## Result Analysis and Reporting

### Collecting Metrics

```python
#!/usr/bin/env python3
"""Comprehensive benchmark result collection"""

import json
import time
import psutil
import psycopg2
from datetime import datetime

class BenchmarkResultCollector:
    def __init__(self, db_config):
        self.db_config = db_config
        self.metrics = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': [],
            'database_metrics': [],
            'benchmark_results': {}
        }
    
    def collect_system_metrics(self, duration=60, interval=1):
        """Collect system CPU, memory, I/O metrics"""
        conn = psycopg2.connect(**self.db_config)
        
        print(f"Collecting system metrics for {duration} seconds...")
        end_time = time.time() + duration
        
        while time.time() < end_time:
            metric = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_io': psutil.disk_io_counters()._asdict(),
                'network_io': psutil.net_io_counters()._asdict(),
            }
            
            # Database-specific metrics
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    sum(total_exec_time) as total_query_time,
                    sum(calls) as total_calls
                FROM pg_stat_statements
            """)
            result = cursor.fetchone()
            if result:
                metric['db_query_time'] = result[0]
                metric['db_query_calls'] = result[1]
            
            self.metrics['system_metrics'].append(metric)
            time.sleep(interval)
        
        conn.close()
        return self.metrics['system_metrics']
    
    def collect_database_metrics(self):
        """Collect database-specific metrics"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Connection info
        cursor.execute("""
            SELECT count(*), state 
            FROM pg_stat_activity 
            GROUP BY state
        """)
        self.metrics['connections'] = dict(cursor.fetchall())
        
        # Table statistics
        cursor.execute("""
            SELECT schemaname, relname, seq_scan, idx_scan,
                   n_tup_ins, n_tup_upd, n_tup_del
            FROM pg_stat_user_tables
            ORDER BY seq_scan DESC
            LIMIT 20
        """)
        self.metrics['table_stats'] = [
            dict(zip(['schema', 'table', 'seq_scan', 'idx_scan', 
                      'inserts', 'updates', 'deletes'], row))
            for row in cursor.fetchall()
        ]
        
        # Index usage
        cursor.execute("""
            SELECT schemaname, tablename, indexname, idx_scan
            FROM pg_stat_user_indexes
            ORDER BY idx_scan DESC
            LIMIT 20
        """)
        self.metrics['index_stats'] = [
            dict(zip(['schema', 'table', 'index', 'scans'], row))
            for row in cursor.fetchall()
        ]
        
        conn.close()
        return self.metrics['database_metrics']
    
    def analyze_latency_distribution(self, raw_latencies):
        """Analyze latency distribution"""
        import numpy as np
        
        sorted_latencies = sorted(raw_latencies)
        n = len(sorted_latencies)
        
        return {
            'min': sorted_latencies[0],
            'max': sorted_latencies[-1],
            'mean': np.mean(raw_latencies),
            'median': np.median(raw_latencies),
            'stddev': np.std(raw_latencies),
            'p50': sorted_latencies[int(n * 0.50)],
            'p75': sorted_latencies[int(n * 0.75)],
            'p90': sorted_latencies[int(n * 0.90)],
            'p95': sorted_latencies[int(n * 0.95)],
            'p99': sorted_latencies[int(n * 0.99)],
            'p999': sorted_latencies[int(n * 0.999)] if n > 1000 else sorted_latencies[-1],
        }
    
    def generate_report(self, output_file='benchmark_report.html'):
        """Generate HTML report"""
        html = f"""
        <html>
        <head>
            <title>Benchmark Report - {self.metrics['timestamp']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ margin: 20px 0; padding: 10px; border: 1px solid #ccc; }}
                .chart {{ width: 100%; height: 300px; }}
            </style>
        </head>
        <body>
            <h1>Benchmark Report</h1>
            <p>Generated: {self.metrics['timestamp']}</p>
            
            <div class="metric">
                <h2>Summary</h2>
                <pre>{json.dumps(self.metrics['benchmark_results'], indent=2)}</pre>
            </div>
            
            <div class="metric">
                <h2>System Metrics</h2>
                <p>CPU Average: {sum(m['cpu_percent'] for m in self.metrics['system_metrics']) / len(self.metrics['system_metrics']):.2f}%</p>
                <p>Memory Average: {sum(m['memory_percent'] for m in self.metrics['system_metrics']) / len(self.metrics['system_metrics']):.2f}%</p>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"Report saved to {output_file}")
    
    def save_results(self, filename='benchmark_results.json'):
        """Save results to JSON"""
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        print(f"Results saved to {filename}")
```

### Result Analysis Framework

```python
#!/usr/bin/env python3
"""Analyze benchmark results and detect regressions"""

import json
import numpy as np
from scipy import stats

class BenchmarkAnalyzer:
    def __init__(self):
        self.baseline = None
        self.current = None
    
    def load_results(self, baseline_file, current_file):
        """Load benchmark results from files"""
        with open(baseline_file) as f:
            self.baseline = json.load(f)
        with open(current_file) as f:
            self.current = json.load(f)
    
    def compare_throughput(self):
        """Compare throughput between runs"""
        base_tps = self.baseline['benchmark_results']['tps']
        curr_tps = self.current['benchmark_results']['tps']
        
        change_percent = ((curr_tps - base_tps) / base_tps) * 100
        
        return {
            'baseline_tps': base_tps,
            'current_tps': curr_tps,
            'change_percent': change_percent,
            'improved': curr_tps > base_tps,
            'significant': abs(change_percent) > 5,
        }
    
    def compare_latency(self):
        """Compare latency distributions"""
        base_latency = self.baseline['benchmark_results']['latency']
        curr_latency = self.current['benchmark_results']['latency']
        
        return {
            'baseline_p99': base_latency['p99'],
            'current_p99': curr_latency['p99'],
            'change_percent': ((curr_latency['p99'] - base_latency['p99']) / base_latency['p99']) * 100,
        }
    
    def detect_regression(self, threshold_percent=10):
        """Detect performance regression"""
        throughput_comparison = self.compare_throughput()
        latency_comparison = self.compare_latency()
        
        regression_detected = (
            throughput_comparison['change_percent'] < -threshold_percent or
            latency_comparison['change_percent'] > threshold_percent
        )
        
        return {
            'regression_detected': regression_detected,
            'throughput_change': throughput_comparison['change_percent'],
            'latency_change': latency_comparison['change_percent'],
            'threshold': threshold_percent,
            'recommendation': 'RELEASE BLOCKED' if regression_detected else 'PASS',
        }
    
    def generate_comparison_report(self):
        """Generate comparison report"""
        comparison = {
            'throughput': self.compare_throughput(),
            'latency': self.compare_latency(),
            'regression': self.detect_regression(),
        }
        
        print(json.dumps(comparison, indent=2))
        return comparison
```

### Example Result Output

```json
{
  "benchmark_results": {
    "test_name": "TPC-C Like Benchmark",
    "timestamp": "2024-01-15T10:30:00",
    "configuration": {
      "clients": 10,
      "threads": 2,
      "duration_seconds": 300,
      "scale_factor": 100
    },
    "throughput": {
      "transactions": 125432,
      "tps": 418.107,
      "tps_per_core": 26.13
    },
    "latency": {
      "min_ms": 12.5,
      "avg_ms": 45.2,
      "max_ms": 523.4,
      "p50_ms": 32.1,
      "p75_ms": 45.3,
      "p90_ms": 89.2,
      "p95_ms": 145.6,
      "p99_ms": 298.3,
      "stddev_ms": 52.3
    },
    "errors": {
      "deadlocks": 0,
      "timeouts": 0,
      "connection_errors": 0
    }
  },
  "system_metrics": {
    "cpu_avg_percent": 78.5,
    "memory_avg_percent": 65.2,
    "disk_io_reads_per_sec": 1234,
    "disk_io_writes_per_sec": 567
  }
}
```

---

## Summary

### Key Takeaways

1. **pgbench** is PostgreSQL's built-in tool - excellent for TPC-B/TPC-C testing
2. **sysbench** supports MySQL, PostgreSQL, and other databases - very flexible
3. **Custom benchmarks** are essential when standard benchmarks don't match your workload
4. **Data generation** should reflect real-world data distributions
5. **Comprehensive metrics** collection is crucial for meaningful analysis
6. **Statistical analysis** ensures results are significant and actionable

### Testing Checklist

```markdown
Pre-Test:
[ ] Establish baseline with known configuration
[ ] Document all system parameters
[ ] Ensure database is properly configured
[ ] Generate representative test data
[ ] Warm up caches before measuring

During Test:
[ ] Monitor system resources
[ ] Collect latency percentiles
[ ] Record any errors or timeouts
[ ] Keep environment stable

Post-Test:
[ ] Save complete results
[ ] Analyze for regressions
[ ] Compare with baseline
[ ] Generate detailed report
[ ] Document findings
```

### Common Pitfalls

1. **Not warming up caches** - First run will be slower
2. **Insufficient test duration** - Results won't be stable
3. **Ignoring outliers** - p99 matters for SLA
4. **Testing on shared hardware** - Other processes affect results
5. **Single run only** - Need multiple runs for statistical validity
6. **Forgetting to analyze queries** - The benchmark is only part of the story

---

## Additional Resources

- [PostgreSQL pgbench Documentation](https://www.postgresql.org/docs/current/pgbench.html)
- [sysbench GitHub Repository](https://github.com/akopytov/sysbench)
- [pg_stat_statements Extension](https://www.postgresql.org/docs/current/pgstatstatements.html)
- [MySQL Performance Schema](https://dev.mysql.com/doc/refman/8.0/en/performance-schema.html)
