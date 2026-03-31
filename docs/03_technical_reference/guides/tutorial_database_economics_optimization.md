# Database Economics and Cost Optimization Tutorial for AI/ML Systems

## Overview

This hands-on tutorial teaches senior AI/ML engineers how to analyze, optimize, and manage database costs for AI applications. We'll cover cost modeling, cloud economics, performance-cost tradeoffs, and optimization strategies.

## Prerequisites
- AWS/GCP/Azure account (for cloud examples)
- PostgreSQL 14+ or MySQL 8+
- Basic understanding of cloud pricing models
- Python for cost analysis scripts

## Tutorial Structure
This tutorial is divided into 5 progressive sections:
1. **Cost Modeling** - Building comprehensive cost models
2. **Cloud Economics** - Understanding cloud pricing and optimization
3. **Performance-Cost Tradeoffs** - Quantitative analysis framework
4. **Resource Optimization** - Optimizing compute, storage, and network
5. **Automated Cost Management** - CI/CD integration and monitoring

## Section 1: Cost Modeling

### Step 1: Build cost model framework
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DatabaseCostModel:
    def __init__(self):
        self.cost_components = {}
        self.scenarios = {}
    
    def add_infrastructure_cost(self, component_name, cost_per_unit, units, period="monthly"):
        """Add infrastructure cost component"""
        self.cost_components[component_name] = {
            'type': 'infrastructure',
            'cost_per_unit': cost_per_unit,
            'units': units,
            'period': period,
            'total': cost_per_unit * units
        }
    
    def add_operational_cost(self, component_name, cost_per_hour, hours_per_month, period="monthly"):
        """Add operational cost component"""
        self.cost_components[component_name] = {
            'type': 'operational',
            'cost_per_hour': cost_per_hour,
            'hours_per_month': hours_per_month,
            'period': period,
            'total': cost_per_hour * hours_per_month
        }
    
    def add_performance_cost(self, component_name, cost_multiplier, baseline_cost, period="monthly"):
        """Add performance-related cost component"""
        self.cost_components[component_name] = {
            'type': 'performance',
            'cost_multiplier': cost_multiplier,
            'baseline_cost': baseline_cost,
            'period': period,
            'total': baseline_cost * cost_multiplier
        }
    
    def calculate_total_cost(self):
        """Calculate total monthly cost"""
        total = sum(comp['total'] for comp in self.cost_components.values())
        return total
    
    def create_cost_scenario(self, name, description, optimizations=None):
        """Create a cost scenario with optimizations"""
        if optimizations is None:
            optimizations = []
        
        scenario = {
            'name': name,
            'description': description,
            'optimizations': optimizations,
            'cost_components': self.cost_components.copy(),
            'total_cost': self.calculate_total_cost()
        }
        
        self.scenarios[name] = scenario
        return scenario
    
    def compare_scenarios(self, scenario_names):
        """Compare multiple scenarios"""
        comparison = []
        for name in scenario_names:
            scenario = self.scenarios[name]
            comparison.append({
                'scenario': name,
                'total_cost': scenario['total_cost'],
                'optimizations': len(scenario['optimizations']),
                'cost_per_query': scenario['total_cost'] / 10000  # Assuming 10K queries
            })
        
        return pd.DataFrame(comparison)

# Usage example
cost_model = DatabaseCostModel()

# Add infrastructure costs
cost_model.add_infrastructure_cost("Compute", 0.115, 720, "monthly")  # db.m6g.xlarge
cost_model.add_infrastructure_cost("Storage", 0.125, 1000, "monthly")  # 1TB SSD
cost_model.add_infrastructure_cost("Network", 0.09, 1000, "monthly")   # 1TB egress

# Add operational costs
cost_model.add_operational_cost("Engineering", 150, 160, "monthly")  # 1 FTE
cost_model.add_operational_cost("Monitoring", 50, 40, "monthly")      # Monitoring tools

# Add performance costs
cost_model.add_performance_cost("Query Optimization", 0.8, 1000, "monthly")  # 20% savings
cost_model.add_performance_cost("Caching", 0.6, 1500, "monthly")           # 40% savings

print(f"Baseline cost: ${cost_model.calculate_total_cost():,.2f}")
```

### Step 2: AI-specific cost modeling
```python
class AICostModel(DatabaseCostModel):
    def __init__(self):
        super().__init__()
        self.ai_components = {}
    
    def add_ai_workload_cost(self, workload_type, queries_per_day, p99_latency_ms, 
                           availability_requirement, region="us-east-1"):
        """Add AI workload cost component"""
        # Base costs
        base_compute_cost = self._calculate_compute_cost(queries_per_day, p99_latency_ms)
        base_storage_cost = self._calculate_storage_cost(queries_per_day)
        base_network_cost = self._calculate_network_cost(queries_per_day)
        
        # AI-specific multipliers
        latency_multiplier = max(1.0, 1.0 + (p99_latency_ms - 100) / 1000)  # >100ms penalty
        availability_multiplier = 1.0 + (1.0 - availability_requirement) * 2.0  # HA premium
        
        total_cost = (base_compute_cost + base_storage_cost + base_network_cost) * \
                    latency_multiplier * availability_multiplier
        
        ai_component_id = f"ai_workload_{len(self.ai_components)+1}"
        self.ai_components[ai_component_id] = {
            'workload_type': workload_type,
            'queries_per_day': queries_per_day,
            'p99_latency_ms': p99_latency_ms,
            'availability_requirement': availability_requirement,
            'region': region,
            'total_cost': total_cost,
            'breakdown': {
                'compute': base_compute_cost,
                'storage': base_storage_cost,
                'network': base_network_cost,
                'latency_premium': base_compute_cost * (latency_multiplier - 1),
                'ha_premium': (base_compute_cost + base_storage_cost) * (availability_multiplier - 1)
            }
        }
        
        return ai_component_id
    
    def _calculate_compute_cost(self, queries_per_day, latency_ms):
        """Calculate compute cost based on workload"""
        # Simplified model: $0.0001 per query for basic workloads
        # Higher for low-latency requirements
        base_cost_per_query = 0.0001
        if latency_ms < 50:
            base_cost_per_query *= 1.5  # Premium for ultra-low latency
        elif latency_ms < 100:
            base_cost_per_query *= 1.2  # Premium for low latency
        
        return base_cost_per_query * queries_per_day * 30  # Monthly
    
    def _calculate_storage_cost(self, queries_per_day):
        """Calculate storage cost based on data volume"""
        # Assume 1KB per query for feature data
        data_volume_gb = (queries_per_day * 1000 * 30) / (1024 * 1024)  # 30 days
        return data_volume_gb * 0.125  # $0.125/GB/month
    
    def _calculate_network_cost(self, queries_per_day):
        """Calculate network cost"""
        # Assume 10KB per query for responses
        data_transfer_tb = (queries_per_day * 10000 * 30) / (1024 * 1024 * 1024)  # 30 days
        return data_transfer_tb * 90  # $90/TB egress

# Usage example
ai_cost_model = AICostModel()

# Add AI workloads
ai_cost_model.add_ai_workload_cost(
    workload_type="real_time_inference",
    queries_per_day=50000,
    p99_latency_ms=85,
    availability_requirement=0.999
)

ai_cost_model.add_ai_workload_cost(
    workload_type="batch_training",
    queries_per_day=10000,
    p99_latency_ms=500,
    availability_requirement=0.995
)

print(f"AI workloads total cost: ${ai_cost_model.calculate_total_cost():,.2f}")

# Create scenarios
baseline_scenario = ai_cost_model.create_cost_scenario(
    "baseline", "Current architecture",
    ["No optimization"]
)

optimized_scenario = ai_cost_model.create_cost_scenario(
    "optimized", "With optimizations",
    ["Index optimization", "Caching layer", "Connection pooling"]
)

comparison = ai_cost_model.compare_scenarios(["baseline", "optimized"])
print(comparison)
```

## Section 2: Cloud Economics

### Step 1: AWS cost analysis
```python
class AWSCostAnalyzer:
    def __init__(self):
        # AWS pricing (as of 2024)
        self.pricing = {
            'rds': {
                'db.m6g.xlarge': {'hourly': 0.115, 'monthly': 82.8},
                'db.r6g.2xlarge': {'hourly': 0.32, 'monthly': 230.4}
            },
            'aurora': {
                'db.r6g.2xlarge': {'hourly': 0.64, 'monthly': 460.8}
            },
            'dynamodb': {
                'read_capacity': {'unit': 0.000001, 'monthly': 0.000001 * 720 * 1000000},  # $0.000001 per 100K reads
                'write_capacity': {'unit': 0.00000125, 'monthly': 0.00000125 * 720 * 1000000}  # $0.00000125 per 100K writes
            },
            'redshift': {
                'dc2.large': {'hourly': 0.25, 'monthly': 180}
            },
            'timestream': {
                'storage': {'monthly': 0.05},  # $0.05/GB/month
                'writes': {'monthly': 0.001}   # $0.001/100K writes
            }
        }
    
    def calculate_rds_cost(self, instance_type, storage_gb, months=1):
        """Calculate RDS cost"""
        instance_cost = self.pricing['rds'][instance_type]['monthly'] * months
        storage_cost = storage_gb * 0.125 * months  # $0.125/GB/month
        return instance_cost + storage_cost
    
    def calculate_dynamodb_cost(self, reads_per_day, writes_per_day, storage_gb, months=1):
        """Calculate DynamoDB cost"""
        read_cost = (reads_per_day * 30 * months) * 0.000001  # $0.000001 per 100K reads
        write_cost = (writes_per_day * 30 * months) * 0.00000125  # $0.00000125 per 100K writes
        storage_cost = storage_gb * 0.25 * months  # $0.25/GB/month
        return read_cost + write_cost + storage_cost
    
    def compare_aws_services(self, workload_type, queries_per_day, storage_gb):
        """Compare AWS services for given workload"""
        results = {}
        
        if workload_type == "oltp":
            # RDS vs Aurora
            rds_cost = self.calculate_rds_cost('db.m6g.xlarge', storage_gb)
            aurora_cost = self.calculate_rds_cost('db.r6g.2xlarge', storage_gb) * 2  # Aurora ~2x RDS
            
            results['rds'] = rds_cost
            results['aurora'] = aurora_cost
        
        elif workload_type == "nosql":
            # DynamoDB cost
            dynamodb_cost = self.calculate_dynamodb_cost(
                reads_per_day=queries_per_day * 0.8,  # 80% reads
                writes_per_day=queries_per_day * 0.2,  # 20% writes
                storage_gb=storage_gb
            )
            results['dynamodb'] = dynamodb_cost
        
        return results

# Usage example
aws_analyzer = AWSCostAnalyzer()

# Compare RDS vs Aurora for OLTP workload
oltp_comparison = aws_analyzer.compare_aws_services(
    workload_type="oltp",
    queries_per_day=50000,
    storage_gb=1000
)
print("OLTP Workload Comparison:")
for service, cost in oltp_comparison.items():
    print(f"{service.upper()}: ${cost:.2f}/month")

# Calculate DynamoDB cost for NoSQL workload
nosql_cost = aws_analyzer.calculate_dynamodb_cost(
    reads_per_day=80000,
    writes_per_day=20000,
    storage_gb=500
)
print(f"DynamoDB cost: ${nosql_cost:.2f}/month")
```

### Step 2: GCP cost analysis
```python
class GPCCostAnalyzer:
    def __init__(self):
        # GCP pricing (as of 2024)
        self.pricing = {
            'cloud_sql': {
                'db-n1-standard-4': {'hourly': 0.128, 'monthly': 92.16},
                'db-n1-standard-8': {'hourly': 0.256, 'monthly': 184.32}
            },
            'bigquery': {
                'analysis': {'per_tb': 5.0},  # $5/TB analyzed
                'storage': {'per_gb': 0.02}   # $0.02/GB/month
            },
            'firestore': {
                'reads': {'per_100k': 0.06},
                'writes': {'per_100k': 0.18},
                'storage': {'per_gb': 0.18}
            }
        }
    
    def calculate_cloud_sql_cost(self, instance_type, storage_gb, months=1):
        """Calculate Cloud SQL cost"""
        instance_cost = self.pricing['cloud_sql'][instance_type]['monthly'] * months
        storage_cost = storage_gb * 0.17 * months  # $0.17/GB/month
        return instance_cost + storage_cost
    
    def calculate_bigquery_cost(self, tb_analyzed_per_month, storage_gb, months=1):
        """Calculate BigQuery cost"""
        analysis_cost = tb_analyzed_per_month * 5.0 * months
        storage_cost = storage_gb * 0.02 * months
        return analysis_cost + storage_cost
    
    def compare_gcp_services(self, workload_type, queries_per_day, storage_gb):
        """Compare GCP services"""
        results = {}
        
        if workload_type == "oltp":
            sql_cost = self.calculate_cloud_sql_cost('db-n1-standard-4', storage_gb)
            results['cloud_sql'] = sql_cost
        
        elif workload_type == "analytics":
            bq_cost = self.calculate_bigquery_cost(
                tb_analyzed_per_month=10,  # 10TB/month
                storage_gb=1000
            )
            results['bigquery'] = bq_cost
        
        return results

# Usage example
gcp_analyzer = GPCCostAnalyzer()

# Compare GCP services
gcp_comparison = gcp_analyzer.compare_gcp_services(
    workload_type="oltp",
    queries_per_day=50000,
    storage_gb=1000
)
print("GCP OLTP Comparison:")
for service, cost in gcp_comparison.items():
    print(f"{service.upper()}: ${cost:.2f}/month")
```

## Section 3: Performance-Cost Tradeoffs

### Step 1: Quantitative tradeoff analysis
```python
class PerformanceCostAnalyzer:
    def __init__(self):
        self.tradeoffs = []
    
    def add_tradeoff(self, strategy, cost_reduction_percent, performance_improvement_percent,
                   implementation_effort=3, maintenance_cost=2):
        """Add a performance-cost tradeoff"""
        roi_score = (cost_reduction_percent * 0.4) + (performance_improvement_percent * 0.6)
        effort_penalty = implementation_effort * 0.2
        maintenance_penalty = maintenance_cost * 0.1
        net_roi = max(0, roi_score - effort_penalty - maintenance_penalty)
        
        tradeoff = {
            'strategy': strategy,
            'cost_reduction_percent': cost_reduction_percent,
            'performance_improvement_percent': performance_improvement_percent,
            'implementation_effort': implementation_effort,
            'maintenance_cost': maintenance_cost,
            'roi_score': roi_score,
            'net_roi': net_roi
        }
        
        self.tradeoffs.append(tradeoff)
        return tradeoff
    
    def get_top_tradeoffs(self, n=5):
        """Get top n tradeoffs by ROI"""
        return sorted(self.tradeoffs, key=lambda x: x['net_roi'], reverse=True)[:n]
    
    def calculate_optimization_plan(self, budget_constraint=None, performance_target=None):
        """Calculate optimal optimization plan"""
        if budget_constraint:
            # Select tradeoffs within budget constraint
            feasible_tradeoffs = [t for t in self.tradeoffs if t['cost_reduction_percent'] >= budget_constraint]
        elif performance_target:
            # Select tradeoffs meeting performance target
            feasible_tradeoffs = [t for t in self.tradeoffs if t['performance_improvement_percent'] >= performance_target]
        else:
            feasible_tradeoffs = self.tradeoffs
        
        return sorted(feasible_tradeoffs, key=lambda x: x['net_roi'], reverse=True)

# Usage example
analyzer = PerformanceCostAnalyzer()

# Add common AI optimization strategies
analyzer.add_tradeoff("Index Optimization", 25, 50, 2, 1)
analyzer.add_tradeoff("Caching Layer", 50, 80, 3, 2)
analyzer.add_tradeoff("Data Compression", 30, 5, 1, 1)
analyzer.add_tradeoff("Connection Pooling", 15, 30, 1, 1)
analyzer.add_tradeoff("Sharding", 40, 200, 5, 4)
analyzer.add_tradeoff("Vector Index Tuning", 35, 70, 3, 2)
analyzer.add_tradeoff("Query Rewriting", 20, 40, 2, 1)
analyzer.add_tradeoff("Read Replicas", 25, 100, 3, 2)

print("Top 5 optimizations by ROI:")
top_5 = analyzer.get_top_tradeoffs(5)
for i, tradeoff in enumerate(top_5, 1):
    print(f"{i}. {tradeoff['strategy']}: ROI={tradeoff['net_roi']:.2f}, "
          f"Cost↓{tradeoff['cost_reduction_percent']}%, Perf↑{tradeoff['performance_improvement_percent']}%")

# Calculate optimization plan
plan = analyzer.calculate_optimization_plan(performance_target=50)
print(f"\nOptimization plan for 50%+ performance improvement:")
for tradeoff in plan:
    print(f"- {tradeoff['strategy']}: {tradeoff['net_roi']:.2f} ROI")
```

## Section 4: Resource Optimization

### Step 1: Compute resource optimization
```sql
-- Resource optimization queries
-- Identify underutilized instances
SELECT 
    datname as database,
    numbackends,
    xact_commit,
    xact_rollback,
    blks_read,
    blks_hit,
    CASE 
        WHEN blks_hit > 0 THEN (blks_hit::FLOAT / (blks_hit + blks_read)) * 100
        ELSE 0
    END as cache_hit_ratio,
    CASE 
        WHEN xact_commit + xact_rollback > 0 THEN (xact_commit::FLOAT / (xact_commit + xact_rollback)) * 100
        ELSE 0
    END as commit_ratio
FROM pg_stat_database 
WHERE datname NOT IN ('template0', 'template1', 'postgres')
ORDER BY cache_hit_ratio ASC;

-- Identify expensive queries
SELECT 
    query,
    total_exec_time,
    calls,
    total_exec_time/calls as avg_time,
    rows,
    rows/calls as rows_per_call
FROM pg_stat_statements 
WHERE total_exec_time > 1000  -- >1 second
ORDER BY total_exec_time DESC
LIMIT 10;
```

### Step 2: Storage optimization techniques
```python
class StorageOptimizer:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def analyze_storage_usage(self):
        """Analyze storage usage patterns"""
        cursor = self.db.cursor()
        
        # Table size analysis
        cursor.execute("""
            SELECT 
                schemaname,
                tablename,
                pg_total_relation_size(quote_ident(schemaname) || '.' || quote_ident(tablename)) as total_size,
                pg_indexes_size(quote_ident(schemaname) || '.' || quote_ident(tablename)) as index_size,
                pg_total_relation_size(quote_ident(schemaname) || '.' || quote_ident(tablename)) - 
                pg_indexes_size(quote_ident(schemaname) || '.' || quote_ident(tablename)) as table_size
            FROM pg_tables 
            WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
            ORDER BY total_size DESC
            LIMIT 20;
        """)
        
        tables = cursor.fetchall()
        
        # Column statistics
        cursor.execute("""
            SELECT 
                table_name,
                column_name,
                data_type,
                count(*) as row_count,
                count(nullif(column_name, NULL)) as non_null_count,
                (count(nullif(column_name, NULL))::FLOAT / count(*)) * 100 as null_percentage
            FROM information_schema.columns c
            LEFT JOIN (
                SELECT table_name, column_name, COUNT(*) as cnt
                FROM information_schema.columns
                GROUP BY table_name, column_name
            ) stats ON c.table_name = stats.table_name AND c.column_name = stats.column_name
            WHERE c.table_schema NOT IN ('information_schema', 'pg_catalog')
            GROUP BY table_name, column_name, data_type
            HAVING (count(nullif(column_name, NULL))::FLOAT / count(*)) * 100 > 50
            ORDER BY null_percentage DESC;
        """)
        
        sparse_columns = cursor.fetchall()
        
        cursor.close()
        
        return {
            'tables': tables,
            'sparse_columns': sparse_columns
        }
    
    def recommend_optimizations(self, analysis_results):
        """Recommend storage optimizations"""
        recommendations = []
        
        # Large tables
        for table in analysis_results['tables'][:5]:
            if table[2] > 1024 * 1024 * 1024:  # >1GB
                recommendations.append({
                    'type': 'table_optimization',
                    'target': table[1],
                    'recommendation': 'Consider partitioning or archiving old data',
                    'potential_savings': f"${table[2]/1024/1024/1024 * 0.125:.2f}/month"
                })
        
        # Sparse columns
        for column in analysis_results['sparse_columns']:
            if column[5] > 80:  # >80% null
                recommendations.append({
                    'type': 'column_optimization',
                    'target': f"{column[0]}.{column[1]}",
                    'recommendation': 'Consider storing in JSON or separate table',
                    'potential_savings': 'Significant storage reduction'
                })
        
        return recommendations

# Usage example
optimizer = StorageOptimizer(db_connection)
analysis = optimizer.analyze_storage_usage()
recommendations = optimizer.recommend_optimizations(analysis)

print("Storage optimization recommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec['target']}: {rec['recommendation']}")
    if 'potential_savings' in rec:
        print(f"   Potential savings: {rec['potential_savings']}")
```

## Section 5: Automated Cost Management

### Step 1: CI/CD integration
```yaml
# .github/workflows/database-cost-analysis.yml
name: Database Cost Analysis
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  cost-analysis:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install pandas numpy psycopg2
        
    - name: Run cost analysis
      env:
        DB_HOST: ${{ secrets.DB_HOST }}
        DB_USER: ${{ secrets.DB_USER }}
        DB_PASSWORD: ${{ secrets.DB_PASSWORD }}
        DB_NAME: ${{ secrets.DB_NAME }}
      run: |
        python scripts/cost_analysis.py
        
    - name: Generate cost report
      run: |
        python scripts/generate_report.py
        
    - name: Upload cost report
      uses: actions/upload-artifact@v3
      with:
        name: cost-report
        path: cost_report.pdf
```

### Step 2: Cost monitoring dashboard
```python
import dash
from dash import dcc, html
import plotly.graph_objs as go
from datetime import datetime, timedelta

def create_cost_dashboard():
    """Create cost monitoring dashboard"""
    
    app = dash.Dash(__name__)
    
    # Get cost data
    def get_cost_data():
        # This would query actual cost data from your systems
        # For demo, return mock data
        dates = [datetime.now() - timedelta(days=i) for i in range(30)]
        baseline_costs = [10000 + i*100 for i in range(30)]  # Increasing baseline
        optimized_costs = [8000 + i*80 for i in range(30)]   # Optimized
        
        return dates, baseline_costs, optimized_costs
    
    dates, baseline, optimized = get_cost_data()
    
    # Dashboard layout
    app.layout = html.Div([
        html.H1("Database Cost Monitoring Dashboard"),
        
        dcc.Graph(
            id='cost-trend',
            figure={
                'data': [
                    go.Scatter(x=dates, y=baseline, mode='lines', name='Baseline'),
                    go.Scatter(x=dates, y=optimized, mode='lines', name='Optimized'),
                ],
                'layout': go.Layout(
                    title='Monthly Database Costs',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Cost ($)'}, 
                    legend={'x': 0, 'y': 1}
                )
            }
        ),
        
        html.Div([
            html.Div([
                html.H3("Current Month Cost"),
                html.Div(id='current-cost', className='metric'),
            ], className='metric-card'),
            
            html.Div([
                html.H3("Monthly Savings"),
                html.Div(id='monthly-savings', className='metric'),
            ], className='metric-card'),
            
            html.Div([
                html.H3("ROI"),
                html.Div(id='roi-score', className='metric'),
            ], className='metric-card'),
        ], className='metrics-row'),
        
        dcc.Interval(
            id='interval-component',
            interval=300*1000,  # Update every 5 minutes
            n_intervals=0
        ),
    ])
    
    @app.callback(
        [dash.Output('current-cost', 'children'),
         dash.Output('monthly-savings', 'children'),
         dash.Output('roi-score', 'children')],
        [dash.Input('interval-component', 'n_intervals')]
    )
    def update_metrics(n):
        # In production, this would fetch real-time cost data
        current_cost = 8500
        baseline_cost = 10500
        savings = baseline_cost - current_cost
        roi = (savings / 2000) * 100  # Assuming $2000 investment
        
        return f"${current_cost:,.2f}", f"${savings:,.2f}", f"{roi:.1f}%"
    
    return app

# Run dashboard
if __name__ == '__main__':
    app = create_cost_dashboard()
    app.run_server(debug=True)
```

## Hands-on Exercises

### Exercise 1: Build cost model
1. Implement the DatabaseCostModel class
2. Add your actual infrastructure costs
3. Create baseline and optimized scenarios
4. Calculate ROI for different optimization strategies

### Exercise 2: Cloud cost analysis
1. Set up AWS/GCP cost analysis
2. Compare different database services for your workload
3. Calculate total cost of ownership
4. Identify cost optimization opportunities

### Exercise 3: Performance-cost tradeoff analysis
1. Profile your current database performance
2. Identify expensive queries and operations
3. Apply the PerformanceCostAnalyzer
4. Prioritize optimizations by ROI

### Exercise 4: Storage optimization
1. Analyze your database storage usage
2. Identify large tables and sparse columns
3. Implement recommended optimizations
4. Measure cost and performance impact

## Best Practices Summary

1. **Model First**: Build cost models before making architectural decisions
2. **Quantify Everything**: Use quantitative metrics for all decisions
3. **Continuous Monitoring**: Set up automated cost monitoring
4. **ROI Focus**: Prioritize optimizations by ROI, not just cost savings
5. **Holistic View**: Consider total cost of ownership, not just infrastructure

This tutorial provides practical, hands-on experience with database economics and cost optimization specifically for AI/ML systems. Complete all exercises to master these critical cost management skills.