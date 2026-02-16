# Cloud Database Architecture

## Overview

Cloud database architecture has fundamentally transformed how organizations deploy, manage, and scale their data infrastructure. Modern cloud databases offer capabilities that would require significant engineering effort to implement on-premises, including automatic scaling, built-in high availability, managed backups, and global distribution. This document provides comprehensive coverage of cloud database architectures across major cloud providers, including practical implementation guidance, configuration examples, and decision frameworks for selecting appropriate solutions.

The evolution from traditional database administration to cloud database management represents a paradigm shift in operational responsibilities. Rather than managing hardware, operating systems, and database software, organizations can leverage managed services that handle infrastructure concerns while providing programmatic access to database functionality. This shift allows teams to focus on application-level concerns rather than infrastructure plumbing.

Understanding cloud database architecture requires knowledge of the various service models available, from fully managed database-as-a-service offerings to self-managed databases running on cloud virtual machines. Each approach offers different trade-offs in terms of control, customization, cost, and operational complexity. The content here explores these trade-offs in detail and provides guidance for making informed architectural decisions.

## Cloud-Native Databases

### Amazon Aurora

Amazon Aurora is a MySQL and PostgreSQL-compatible relational database built for the cloud. Aurora achieves up to five times the throughput of standard MySQL and three times the throughput of standard PostgreSQL by using a distributed, fault-tolerant, self-healing storage system that grows automatically as needed.

Aurora's architecture separates the storage layer from the compute layer, allowing independent scaling. The storage layer uses a distributed, multi-tenant design with six-way replication across three Availability Zones. This architecture provides durability guarantees and automatic failover without requiring you to manage replication explicitly.

Creating an Aurora cluster with the AWS CLI:

```bash
# Create Aurora PostgreSQL cluster
aws rds create-db-cluster \
    --db-cluster-identifier aurora-postgres-cluster \
    --engine aurora-postgres \
    --engine-version 15.4 \
    --master-username admin \
    --master-user-password YourSecurePassword123 \
    --db-cluster-instance-class db.r6g.xlarge \
    --allocated-storage 100 \
    --storage-encrypted \
    --backup-retention-period 7 \
    --preferred-backup-window "03:00-04:00" \
    --preferred-maintenance-window "mon:04:00-mon:05:00"

# Add reader instance for read scaling
aws rds create-db-instance \
    --db-instance-identifier aurora-postgres-reader \
    --db-cluster-identifier aurora-postgres-cluster \
    --db-instance-class db.r6g.xlarge \
    --engine aurora-postgres
```

Aurora Serverless provides automatic scaling based on workload:

```bash
# Create Aurora Serverless v2 cluster
aws rds create-db-cluster \
    --db-cluster-identifier aurora-serverless-cluster \
    --engine aurora-postgres \
    --engine-mode serverless-v2 \
    --master-username admin \
    --master-user-password YourSecurePassword123 \
    --scaling-configuration MinCapacity=2,MaxCapacity=64,AutoPause=true,SecondsUntilAutoPause=300
```

Python application code connecting to Aurora:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

# Standard Aurora connection
aurora_engine = create_engine(
    "postgresql+psycopg2://admin:password@aurora-cluster-endpoint.cluster-xxx.us-east-1.rds.amazonaws.com:5432/mydb",
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False
)

# Aurora Serverless connection
aurora_serverless_engine = create_engine(
    "postgresql+psycopg2://admin:password@aurora-serverless-cluster-xxx.serverless.us-east-1.rds.amazonaws.com:5432/mydb",
    pool_size=5,
    pool_recycle=3600
)

# Using Data API for serverless (no connection pooling needed)
import boto3

rds_data = boto3.client('rds-data')

def execute_statementAurora(cluster_arn, secret_arn, sql, params=None):
    response = rds_data.execute_statement(
        resourceArn=cluster_arn,
        secretArn=secret_arn,
        database='mydb',
        sql=sql,
        parameters=params or []
    )
    return response
```

### Google Cloud Spanner

Google Cloud Spanner is a globally distributed relational database that provides strong consistency, horizontal scaling, and automatic sharding. Spanner uses TrueTime, a distributed clock technology, to provide globally consistent transactions across regions without requiring manual conflict resolution.

Spanner is ideal for applications requiring global distribution with strong consistency, such as financial systems, supply chain applications, and globally distributed services. The managed nature of Spanner eliminates much of the operational complexity associated with distributed databases while providing enterprise-grade features.

Creating and configuring a Spanner instance:

```bash
# Create Spanner instance
gcloud spanner instances create production-instance \
    --config=regional-us-central1 \
    --description="Production Spanner Instance" \
    --nodes=3

# Create database and schema
gcloud spanner databases create orders-db \
    --instance=production-instance

# Apply schema
gcloud spanner databases ddl update orders-db \
    --instance=production-instance \
    --ddl='CREATE TABLE Orders (
        OrderId STRING(36) NOT NULL,
        CustomerId STRING(36) NOT NULL,
        TotalAmount FLOAT64 NOT NULL,
        Status STRING(20) NOT NULL,
        CreatedAt TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
        UpdatedAt TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
    ) PRIMARY KEY (OrderId)'
```

Python client for Spanner:

```python
from google.cloud import spanner
from google.cloud.spanner_v1 import Transaction
import os

# Initialize Spanner client
spanner_client = spanner.Client()
instance = spanner_client.instance('production-instance')
database = instance.database('orders-db')

# Read data with strong consistency
def get_order(order_id: str):
    with database.snapshot() as snapshot:
        result = snapshot.execute_sql(
            "SELECT * FROM Orders WHERE OrderId = @order_id",
            params={'order_id': order_id},
            param_types={'order_id': spanner.param_types.STRING}
        )
        return list(result)

# Transaction with read-write
def create_order(order_id: str, customer_id: str, amount: float):
    with database.batch() as batch:
        batch.insert(
            table='Orders',
            columns=['OrderId', 'CustomerId', 'TotalAmount', 'Status', 'CreatedAt', 'UpdatedAt'],
            values=[
                [order_id, customer_id, amount, 'PENDING',
                 spanner.COMMIT_TIMESTAMP, spanner.COMMIT_TIMESTAMP]
            ]
        )

# Interleaved tables for parent-child relationships
ddl = '''
CREATE TABLE Customers (
    CustomerId STRING(36) NOT NULL,
    Name STRING(100) NOT NULL,
    Email STRING(255) NOT NULL,
    CreatedAt TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
) PRIMARY KEY (CustomerId);

CREATE TABLE Orders (
    OrderId STRING(36) NOT NULL,
    CustomerId STRING(36) NOT NULL,
    TotalAmount FLOAT64 NOT NULL,
    Status STRING(20) NOT NULL,
    CreatedAt TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
) PRIMARY KEY (OrderId),
    INTERLEAVE IN PARENT Customers(CustomerId) ON DELETE CASCADE;
'''
```

### Azure Cosmos DB

Azure Cosmos DB is a globally distributed, multi-model database service that offers turnkey global distribution across any number of Azure regions. It supports multiple data models including document, key-value, graph, and column-family, allowing you to choose the model that best fits your application needs.

Cosmos DB provides five consistency models: strong, bounded staleness, session, consistent prefix, and eventual. This flexibility allows you to tune consistency based on your application's requirements, trading off consistency against latency and availability.

Creating and configuring Cosmos DB:

```bash
# Create Cosmos DB account
az cosmosdb create \
    --name my-cosmosdb \
    --resource-group my-resource-group \
    --default-consistency-level session \
    --locations regionName=eastus failoverPriority=0 isZoneRedundant=False \
    --locations regionName=westus failoverPriority=1 isZoneRedundant=False \
    --enable-multiple-write-locations true

# Create database and container
az cosmosdb sql database create \
    --name my-database \
    --resource-group my-resource-group \
    --account-name my-cosmosdb

az cosmosdb sql container create \
    --name my-container \
    --database-name my-database \
    --resource-group my-resource-group \
    --account-name my-cosmosdb \
    --partition-key-path "/customerId" \
    --throughput 1000
```

Python SDK for Cosmos DB:

```python
from azure.cosmos import CosmosClient, PartitionKey
import os

# Initialize Cosmos client
cosmos_client = CosmosClient(
    url=os.environ['COSMOS_ENDPOINT'],
    credential=os.environ['COSMOS_KEY']
)

# Get database and container
database = cosmos_client.get_database_client('my-database')
container = database.get_container_client('orders')

# Create item
order = {
    'id': 'order-123',
    'customerId': 'customer-456',
    'totalAmount': 99.99,
    'status': 'pending',
    'items': [
        {'productId': 'prod-1', 'quantity': 2, 'price': 49.99}
    ]
}
container.create_item(order)

# Query items
query = "SELECT * FROM orders o WHERE o.customerId = @customerId"
items = container.query_items(
    query=query,
    parameters=[{'name': '@customerId', 'value': 'customer-456'}],
    enable_cross_partition_query=True
)

# Change feed processing
def process_changes(changes):
    for item in changes:
        print(f"Processing order: {item['id']}")

# Get change feed iterator
change_feed = container.query_items_change_feed(
    partition_key='customer-456',
    is_start_from_beginning=True
)

# Process changes
for change in iter(change_feed.read_next_page_segments()):
    process_changes(change)
```

## Serverless Databases

### AWS DynamoDB

DynamoDB is a fully managed NoSQL database service that provides single-digit millisecond latency at any scale. DynamoDB automatically partitions data based on throughput requirements, eliminating the need for manual sharding. It offers two capacity modes: provisioned capacity for predictable workloads and on-demand capacity for variable workloads.

DynamoDB's serverless nature makes it particularly attractive for applications with unpredictable traffic patterns or those that want to minimize operational overhead. The pay-per-request pricing model means you only pay for the reads, writes, and storage you actually consume.

Creating and configuring DynamoDB tables:

```bash
# Create DynamoDB table
aws dynamodb create-table \
    --table-name orders \
    --attribute-definitions \
        AttributeName=orderId,AttributeType=S \
        AttributeName=customerId,AttributeType=S \
        AttributeName=createdAt,AttributeType=S \
    --key-schema \
        AttributeName=orderId,KeyType=HASH \
    --global-secondary-indexes \
        '[{"IndexName":"customerId-index","KeySchema":[{"AttributeName":"customerId","KeyType":"HASH"}],"Projection":{"ProjectionType":"ALL"},"ProvisionedThroughput":{"ReadCapacityUnits":10,"WriteCapacityUnits":10}}]' \
    --provisioned-throughput ReadCapacityUnits=20,WriteCapacityUnits=20 \
    --billing-mode PROVISIONED

# Create table with on-demand billing
aws dynamodb create-table \
    --table-name orders-on-demand \
    --attribute-definitions AttributeName=orderId,AttributeType=S \
    --key-schema AttributeName=orderId,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST
```

Python application code for DynamoDB:

```python
import boto3
from boto3.dynamodb.conditions import Key, Attr
from datetime import datetime
import json

# Initialize DynamoDB resource
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('orders')

# Put item with conditional writes
def create_order(order_id: str, customer_id: str, items: list):
    table.put_item(
        Item={
            'orderId': order_id,
            'customerId': customer_id,
            'status': 'pending',
            'items': items,
            'totalAmount': sum(item['price'] * item['quantity'] for item in items),
            'createdAt': datetime.utcnow().isoformat(),
            'updatedAt': datetime.utcnow().isoformat()
        },
        ConditionExpression='attribute_not_exists(orderId)'
    )

# Get item
def get_order(order_id: str):
    response = table.get_item(Key={'orderId': order_id})
    return response.get('Item')

# Query by partition key
def get_customer_orders(customer_id: str):
    response = table.query(
        IndexName='customerId-index',
        KeyConditionExpression=Key('customerId').eq(customer_id)
    )
    return response.get('Items', [])

# Scan with filter (use sparingly)
def get_pending_orders():
    response = table.scan(
        FilterExpression=Attr('status').eq('pending')
    )
    return response.get('Items', [])

# Update with atomic counter
def increment_order_view(order_id: str):
    table.update_item(
        Key={'orderId': order_id},
        UpdateExpression='SET viewCount = if_not_exists(viewCount, :zero) + :inc',
        ExpressionAttributeValues={
            ':inc': 1,
            ':zero': 0
        },
        ReturnValues='ALL_NEW'
    )

# Batch operations
def batch_create_orders(orders: list):
    with table.batch_writer() as batch:
        for order in orders:
            batch.put_item(Item=order)

# DynamoDB Streams for change data capture
def process_order_changes(stream_arn: str):
    import boto3
    kinesis = boto3.client('kinesis')

    # Get records from DynamoDB stream
    response = kinesis.get_shard_iterator(
        StreamArn=stream_arn,
        ShardIteratorType='LATEST',
        ShardIteratorName='orders-shard-iterator'
    )

    iterator = response['ShardIterator']
    while True:
        response = kinesis.get_records(ShardIterator=iterator, Limit=10)
        records = response['Records']

        for record in records:
            # DynamoDB stream records are base64 encoded
            import base64
            event = json.loads(base64.b64decode(record['Data']))
            print(f"Event: {event['eventName']}, Order: {event['dynamodb']['Keys']}")

        if records:
            iterator = response['NextShardIterator']
        else:
            break
```

### PlanetScale

PlanetScale is a MySQL-compatible serverless database platform designed for developers. It provides horizontal scaling through automatic sharding, branching for development workflows, and deploy-friendly schemas that can be applied without blocking operations.

PlanetScale's distinctive feature is its branching capability, which allows developers to create database branches for development and testing, similar to how they work with git branches. These branches can be used to test schema changes and application code before deploying to production.

Working with PlanetScale:

```bash
# Connect to PlanetScale database
pscale database connect production-db --port 3306

# Create branch for development
pscale branch create production-db add-user-preferences

# Deploy schema changes to production
pscale deploy production-db add-user-preferences

# Promote branch to production
pscale branch promote production-db staging
```

Python connection to PlanetScale:

```python
from sqlalchemy import create_engine
import os

# PlanetScale connection using MySQL connector
engine = create_engine(
    "mysql+mysqlconnector://username:password@aws.connect.psdb.cloud/database?ssl_ca=/etc/ssl/certs/ca-certificates.crt",
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)

# Using SSL with PyMySQL
engine_pymysql = create_engine(
    "mysql+pymysql://username:password@aws.connect.psdb.cloud/database?ssl={'ssl_verify_cert': True}",
    pool_pre_ping=True
)

# Connection with PlanetScale branch
def get_branch_connection(branch: str):
    return create_engine(
        f"mysql+pymysql://username:password@aws.connect.psdb.cloud/{branch}?ssl={'ssl_verify_cert': True}"
    )
```

## Multi-Region Database Deployments

### Global Database Architecture

Multi-region database deployments serve applications with global user bases by placing data closer to users, reducing latency and improving user experience. However, global distribution introduces complexity around data consistency, conflict resolution, and operational management.

There are three primary approaches to multi-region databases. Read replicas maintain copies of data in multiple regions for read scaling and disaster recovery. Active-active configurations allow writes in any region, requiring conflict resolution mechanisms. Finally, geographic partitioning routes queries to region-specific data stores based on user location.

AWS Global Accelerator for multi-region routing:

```bash
# Create Application Load Balancer in primary region
aws elbv2 create-load-balancer \
    --name primary-region-alb \
    --subnets subnet-xxx subnet-yyy \
    --security-groups sg-xxx \
    --type application

# Create Global Accelerator
aws globalaccelerator create-accelerator \
    --name my-global-accelerator \
    --enabled

# Add listener
aws globalaccelerator create-listener \
    --accelerator-arn arn:aws:globalaccelerator::123456789012:accelerator/xxx \
    --protocol TCP \
    --port-ranges FromPort=443,ToPort=443

# Add endpoint group for each region
aws globalaccelerator create-endpoint-group \
    --listener-arn arn:aws:globalaccelerator::123456789012:listener/xxx \
    --endpoint-group-region us-east-1 \
    --traffic-dial-percentage 100 \
    --health-check-interval-seconds 10 \
    --health-check-path /health \
    --health-check-protocol HTTPS \
    --threshold-count 3 \
    --endpoint-arn arn:aws:elasticloadbalancing:us-east-1:123456789012:loadbalancer/app/primary/xxx
```

Google Cloud CDN with Cloud Spanner for global distribution:

```python
from google.cloud import spanner
from google.cloud import storage
from google.cloud import compute_v1

# Configure Spanner for multi-region
instance_config = {
    "name": "regional-us-central1",
    "replicas": [
        {"location": "us-central1", "type": "READ_WRITE"},
        {"location": "us-east1", "type": "READ_ONLY"},
        {"location": "us-west1", "type": "READ_ONLY"}
    ]
}

# Create instance with regional configuration
client = spanner.Client()
operation = client.instance('production-instance').update(
    {"config": "regional-us-central1", "nodeCount": 3}
)
operation.result()

# Configure read replicas in multiple regions
def read_from_nearest_region(customer_id: str):
    """Read from the nearest available replica."""
    from google.cloud.spanner_v1 import KeySet

    # Create a bounded staleness timestamp for lower latency reads
    # This provides better latency while guaranteeing bounded staleness
    timestamp = datetime.utcnow() - timedelta(seconds=5)

    with database.snapshot(multi_use_exact_staleness=timestamp) as snapshot:
        result = snapshot.read(
            table='Customers',
            columns=['CustomerId', 'Name', 'Email'],
            keyset=KeySet(all_=True)
        )
        return list(result)
```

Azure Cosmos DB global distribution configuration:

```python
from azure.cosmos import CosmosClient, ConsistencyLevel
import os

# Initialize with multiple locations
cosmos_client = CosmosClient(
    url=os.environ['COSMOS_ENDPOINT'],
    credential=os.environ['COSMOS_KEY'],
    connection_policy={
        'preferred_locations': ['East US', 'West Europe', 'Southeast Asia']
    }
)

# Configure for automatic failover
database = cosmos_client.get_database_client('mydb')
database.create_container(
    'orders',
    partition_key='/customerId',
    default_ttl=86400
)

# Enable multi-region writes
account = cosmos_client.get_account_client()
account.update({
    'locations': [
        {'locationName': 'East US', 'failoverPriority': 0},
        {'locationName': 'West Europe', 'failoverPriority': 1},
        {'locationName': 'Southeast Asia', 'failoverPriority': 2}
    ],
    'write_locations': ['East US'],
    'enable_multiple_writable_locations': True
})

# Read from preferred region
container = database.get_container_client('orders')
# Reads automatically route to nearest region
item = container.read_item('order-123', partition_key='customer-456')
```

### Cross-Region Replication

Cross-region replication keeps data synchronized between geographically distributed databases. The specific implementation depends on your consistency requirements and tolerance for replication lag.

AWS DynamoDB global tables:

```bash
# Create global table (replication across regions)
aws dynamodb create-global-table \
    --global-table-name orders-global \
    --replication-group RegionName=us-east-1 RegionName=us-west-2 RegionName=eu-west-1

# Update table to enable streams for change tracking
aws dynamodb update-table \
    --table-name orders \
    --stream-specification StreamViewType=NEW_AND_OLD_IMAGES
```

PostgreSQL logical replication for cross-region:

```sql
-- On publisher (primary region)
CREATE PUBLICATION orders_pub FOR TABLE orders, order_items;

-- Configure logical replication slot
ALTER SYSTEM SET wal_level = logical;
ALTER SYSTEM SET max_wal_senders = 10;
ALTER SYSTEM SET max_replication_slots = 10;

-- On subscriber (DR region)
CREATE SUBSCRIPTION orders_sub
CONNECTION 'host=primary-db.us-east-1.rds.amazonaws.com port=5432 dbname=mydb user=replicator password=xxx'
PUBLICATION orders_pub;
```

## Cloud Database Cost Optimization

### Cost Management Strategies

Cloud database costs can quickly become a significant portion of cloud spending. Effective cost management requires understanding pricing models, right-sizing resources, and implementing optimization strategies.

The primary cost components for cloud databases typically include compute costs based on instance size and hours running, storage costs for data volumes and backups, I/O costs for read and write operations, and data transfer costs for cross-region and internet data movement.

Right-sizing database instances based on actual usage:

```python
import boto3
from datetime import datetime, timedelta

def analyze_right_sizing_recommendations():
    """
    Analyze CloudWatch metrics to recommend right-sizing.
    """
    cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')

    # Get CPU utilization metrics
    response = cloudwatch.get_metric_statistics(
        Namespace='AWS/RDS',
        MetricName='CPUUtilization',
        Dimensions=[
            {'Name': 'DBInstanceIdentifier', 'Value': 'my-database'}
        ],
        StartTime=datetime.utcnow() - timedelta(days=30),
        EndTime=datetime.utcnow(),
        Period=86400,
        Statistics=['Average', 'Maximum', 'Minimum']
    )

    # Analyze utilization patterns
    avg_cpu = sum([p['Average'] for p in response['Datapoints']]) / len(response['Datapoints'])
    max_cpu = max([p['Maximum'] for p in response['Datapoints']])

    recommendations = {
        'avg_cpu_utilization': avg_cpu,
        'max_cpu_utilization': max_cpu,
        'current_instance': 'db.r6g.xlarge',
        'recommendations': []
    }

    if avg_cpu < 30:
        recommendations['recommendations'].append({
            'action': 'downsize',
            'target_instance': 'db.r6g.large',
            'estimated_savings': 40
        })
    elif avg_cpu < 50:
        recommendations['recommendations'].append({
            'action': 'consider',
            'target_instance': 'db.r6g.xlarge',
            'reason': 'Moderate utilization'
        })

    return recommendations

# Generate cost optimization report
def generate_cost_report(db_identifier: str):
    ce = boto3.client('ce')

    # Get cost and usage by service
    response = ce.get_cost_and_usage(
        TimePeriod={
            'Start': '2024-01-01',
            'End': '2024-01-31'
        },
        Granularity='DAILY',
        Metrics=['UnblendedCost', 'UsageQuantity'],
        GroupBy=[
            {'Type': 'DIMENSION', 'Key': 'SERVICE'}
        ]
    )

    return response
```

Reserved Instance planning:

```bash
# Calculate Reserved Instance requirements based on steady-state usage
# Using AWS Cost Explorer API
aws ce get-cost-and-usage \
    --time-period Start=2024-01-01,End=2024-01-31 \
    --granularity DAILY \
    --metrics "UsageQuantity" \
    --filter "{\"Dimensions\":{\"Key\":\"RDSInstanceType\",\"Values\":[\"db.r6g.xlarge\"]}}" \
    --group-by "[{\"Type\":\"DIMENSION\",\"Key\":\"USAGE_TYPE\"}]"

# Purchase Reserved Instances for predictable workloads
aws rds purchase-reserved-db-instances-offering \
    --reserved-db-instances-offering-id abcdef1234567890 \
    --db-instance-count 1
```

### Storage Optimization

Storage costs can be significant, especially for large datasets. Optimization strategies include:

1. **Compression**: Enable compression where supported (e.g., PostgreSQL TOAST, MySQL InnoDB compression)
2. **Tiered storage**: Move infrequently accessed data to cheaper storage tiers
3. **Data lifecycle policies**: Automatically archive or delete old data
4. **Columnar storage**: Use columnar formats for analytical workloads

```sql
-- PostgreSQL: Enable TOAST compression
ALTER TABLE large_documents SET (toast_tuple_target = 2048);

-- MySQL: Enable InnoDB compression
ALTER TABLE large_table ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=8;

-- AWS S3 lifecycle policy for backups
aws s3api put-bucket-lifecycle-configuration \
    --bucket my-backups \
    --lifecycle-configuration '{
        "Rules": [
            {
                "ID": "MoveToGlacier",
                "Status": "Enabled",
                "Transitions": [
                    {
                        "Days": 30,
                        "StorageClass": "GLACIER"
                    }
                ]
            },
            {
                "ID": "DeleteOldBackups",
                "Status": "Enabled",
                "Expiration": {
                    "Days": 365
                }
            }
        ]
    }'
```

### Performance vs Cost Trade-offs

Optimizing for performance often increases costs, while optimizing for cost may impact performance. Finding the right balance requires understanding your application's requirements:

| Strategy | Performance Impact | Cost Impact | Best For |
|----------|-------------------|-------------|----------|
| Provisioned capacity | High (predictable) | Medium-High | Predictable workloads |
| On-demand capacity | Variable (spikes) | Low-Medium | Unpredictable workloads |
| Read replicas | High read scalability | Medium | Read-heavy applications |
| Sharding | High scalability | High | Very large datasets |
| Caching | High read performance | Low | Frequently accessed data |

---

## Related Resources

- For database fundamentals, see [Database Fundamentals](./database_fundamentals.md)
- For cloud-native patterns, see [Cloud-Native Patterns](../02_intermediate/01_cloud_native_patterns.md)
- For multi-region architectures, see [Multi-Region Architectures](../03_advanced/02_specialized_databases/02_graph_databases.md)
- For cost optimization, see [Cost Optimization](../04_production/04_economics/01_database_economics.md)