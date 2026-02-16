# Database Regression Testing

## Overview

Database regression testing ensures that database changes (schema modifications, configuration updates, code deployments) do not degrade performance. This guide covers performance regression detection, benchmark comparison frameworks, CI/CD integration, and alerting systems.

## Table of Contents

1. [Performance Regression Detection](#performance-regression-detection)
2. [Benchmark Comparison Frameworks](#benchmark-comparison-frameworks)
3. [CI/CD Integration for Performance Tests](#cicd-integration-for-performance-tests)
4. [Alerting on Performance Degradation](#alerting-on-performance-degradation)

---

## Performance Regression Detection

### What is Performance Regression?

Performance regression occurs when database performance degrades compared to a previously established baseline. This can manifest as:

- **Throughput degradation**: Fewer transactions/queries per second
- **Latency increase**: Slower response times
- **Resource inefficiency**: Higher CPU, memory, or I/O usage
- **Query degradation**: Specific queries becoming slower

### Regression Detection Strategies

#### 1. Historical Baseline Comparison

```python
#!/usr/bin/env python3
"""Performance regression detection using historical baselines"""

import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class PerformanceRegressionDetector:
    def __init__(self, db_path: str = "performance_history.db"):
        self.conn = sqlite3.connect(db_path)
        self.setup_database()
    
    def setup_database(self):
        """Create database schema for storing results"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                test_name TEXT NOT NULL,
                branch TEXT,
                commit_hash TEXT,
                config TEXT NOT NULL,
                tps REAL,
                avg_latency_ms REAL,
                p95_latency_ms REAL,
                p99_latency_ms REAL,
                success BOOLEAN,
                notes TEXT
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS regression_thresholds (
                test_name TEXT PRIMARY KEY,
                tps_degradation_threshold REAL DEFAULT 0.10,
                latency_increase_threshold REAL DEFAULT 0.20,
                enabled BOOLEAN DEFAULT 1
            )
        """)
        
        self.conn.commit()
    
    def record_benchmark_run(self, test_name: str, config: Dict,
                            results: Dict, branch: str = None,
                            commit_hash: str = None, notes: str = None):
        """Record a benchmark run for future comparison"""
        self.conn.execute("""
            INSERT INTO benchmark_runs 
            (test_name, branch, commit_hash, config, tps, avg_latency_ms, 
             p95_latency_ms, p99_latency_ms, success, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_name,
            branch,
            commit_hash,
            json.dumps(config),
            results.get('tps'),
            results.get('avg_latency_ms'),
            results.get('p95_latency_ms'),
            results.get('p99_latency_ms'),
            results.get('success', True),
            notes
        ))
        self.conn.commit()
    
    def get_baseline(self, test_name: str, limit: int = 10) -> Optional[Dict]:
        """Get the most recent baseline for comparison"""
        cursor = self.conn.execute("""
            SELECT * FROM benchmark_runs
            WHERE test_name = ? AND success = 1
            ORDER BY timestamp DESC
            LIMIT ?
        """, (test_name, limit))
        
        rows = cursor.fetchall()
        if not rows:
            return None
        
        # Calculate average of recent runs as baseline
        tps_values = [r[5] for r in rows if r[5]]
        latency_p95 = [r[7] for r in rows if r[7]]
        latency_p99 = [r[8] for r in rows if r[8]]
        
        return {
            'timestamp': rows[0][1],
            'test_name': rows[0][2],
            'avg_tps': np.mean(tps_values) if tps_values else None,
            'avg_p95_latency': np.mean(latency_p95) if latency_p95 else None,
            'avg_p99_latency': np.mean(latency_p99) if latency_p99 else None,
            'sample_size': len(rows)
        }
    
    def detect_regression(self, test_name: str, current_results: Dict,
                         threshold_tps: float = 0.10,
                         threshold_latency: float = 0.20) -> Dict:
        """Detect if current results indicate regression"""
        baseline = self.get_baseline(test_name)
        
        if not baseline:
            return {
                'regression_detected': False,
                'has_baseline': False,
                'message': 'No baseline available for comparison'
            }
        
        # Calculate percentage changes
        tps_change = None
        latency_change = None
        
        if baseline['avg_tps'] and current_results.get('tps'):
            tps_change = (current_results['tps'] - baseline['avg_tps']) / baseline['avg_tps']
        
        if baseline['avg_p95_latency'] and current_results.get('p95_latency_ms'):
            latency_change = (current_results['p95_latency_ms'] - 
                             baseline['avg_p95_latency']) / baseline['avg_p95_latency']
        
        # Determine if regression detected
        tps_regression = tps_change is not None and tps_change < -threshold_tps
        latency_regression = latency_change is not None and latency_change > threshold_latency
        
        return {
            'regression_detected': tps_regression or latency_regression,
            'has_baseline': True,
            'baseline': baseline,
            'current': current_results,
            'changes': {
                'tps_change_percent': tps_change * 100 if tps_change else None,
                'latency_change_percent': latency_change * 100 if latency_change else None
            },
            'thresholds': {
                'tps_degradation': threshold_tps * 100,
                'latency_increase': threshold_latency * 100
            },
            'details': {
                'tps_regression': tps_regression,
                'latency_regression': latency_regression
            }
        }
```

#### 2. Statistical Change Detection

```python
import numpy as np
from scipy import stats
from typing import List, Tuple

class StatisticalRegressionDetector:
    """Detect regressions using statistical methods"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
    
    def detect_change_point(self, timeseries: List[float]) -> Tuple[bool, float]:
        """
        Detect if there's a significant change point in the timeseries
        using CUSUM (Cumulative Sum) method
        """
        if len(timeseries) < 10:
            return False, 0.0
        
        timeseries = np.array(timeseries)
        mean = np.mean(timeseries)
        std = np.std(timeseries)
        
        if std == 0:
            return False, 0.0
        
        # Standardized CUSUM
        standardized = (timeseries - mean) / std
        cusum_pos = np.maximum.accumulate(standardized)
        cusum_neg = np.minimum.accumulate(standardized)
        
        cusum = np.maximum(cusum_pos, -cusum_neg)
        
        # Threshold based on confidence level
        n = len(timeseries)
        threshold = stats.norm.ppf(self.confidence_level) * np.sqrt(n)
        
        return np.max(cusum) > threshold, np.max(cusum)
    
    def compare_distributions(self, baseline: List[float], 
                            current: List[float]) -> Dict:
        """
        Compare two distributions to detect significant changes
        Uses Mann-Whitney U test for non-parametric comparison
        """
        if len(baseline) < 5 or len(current) < 5:
            return {'significant': False, 'message': 'Insufficient samples'}
        
        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(baseline, current)
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(current) - np.mean(baseline)
        pooled_std = np.sqrt((np.std(baseline)**2 + np.std(current)**2) / 2)
        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
        
        return {
            'significant': p_value < (1 - self.confidence_level),
            'p_value': p_value,
            'effect_size': effect_size,
            'baseline_mean': np.mean(baseline),
            'current_mean': np.mean(current),
            'change_percent': (mean_diff / np.mean(baseline)) * 100 if np.mean(baseline) else 0,
            'interpretation': self._interpret_effect_size(effect_size)
        }
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        d = abs(d)
        if d < 0.2:
            return 'negligible'
        elif d < 0.5:
            return 'small'
        elif d < 0.8:
            return 'medium'
        else:
            return 'large'
```

### Query-Level Regression Detection

```sql
-- PostgreSQL: Detect query performance regression using pg_stat_statements

-- 1. Capture current query statistics
CREATE TABLE query_stats_baseline AS
SELECT 
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    max_exec_time,
    min_exec_time,
    stddev_exec_time,
    rows
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_statements%';

-- 2. After changes, compare current vs baseline
WITH current_stats AS (
    SELECT 
        query,
        calls,
        total_exec_time,
        mean_exec_time
    FROM pg_stat_statements
    WHERE query NOT LIKE '%pg_stat_statements%'
)
SELECT 
    c.query,
    b.mean_exec_time as baseline_mean,
    c.mean_exec_time as current_mean,
    c.mean_exec_time - b.mean_exec_time as latency_diff,
    CASE 
        WHEN b.mean_exec_time > 0 
        THEN ((c.mean_exec_time - b.mean_exec_time) / b.mean_exec_time) * 100
        ELSE 0 
    END as percent_change,
    c.calls - b.calls as call_diff
FROM current_stats c
LEFT JOIN query_stats_baseline b ON c.query = b.query
WHERE b.mean_exec_time IS NOT NULL
    AND ((c.mean_exec_time - b.mean_exec_time) / NULLIF(b.mean_exec_time, 0)) > 0.5
ORDER BY percent_change DESC
LIMIT 20;
```

---

## Benchmark Comparison Frameworks

### Continuous Benchmarking System

```python
#!/usr/bin/env python3
"""Continuous benchmarking framework for database performance"""

import subprocess
import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys

class ContinuousBenchmarkingFramework:
    """Framework for continuous performance benchmarking"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results_dir = Path(config.get('results_dir', './benchmark_results'))
        self.results_dir.mkdir(exist_ok=True)
        self.baseline_file = self.results_dir / 'baseline.json'
        
    def run_benchmark(self, benchmark_config: Dict) -> Dict:
        """Run a single benchmark and collect results"""
        print(f"Running benchmark: {benchmark_config['name']}")
        
        # Prepare database
        if benchmark_config.get('prepare'):
            self._execute_sql(benchmark_config['prepare'])
        
        # Run the benchmark tool
        start_time = time.time()
        
        result = subprocess.run(
            benchmark_config['command'],
            shell=True,
            capture_output=True,
            text=True
        )
        
        duration = time.time() - start_time
        
        # Parse results
        parsed = self._parse_benchmark_output(
            benchmark_config['parser'],
            result.stdout,
            result.stderr
        )
        
        return {
            'name': benchmark_config['name'],
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'metrics': parsed
        }
    
    def _execute_sql(self, sql: str):
        """Execute SQL statements"""
        # Implementation depends on database
        pass
    
    def _parse_benchmark_output(self, parser: str, stdout: str, 
                                stderr: str) -> Dict:
        """Parse benchmark tool output"""
        if parser == 'pgbench':
            return self._parse_pgbench_output(stdout)
        elif parser == 'sysbench':
            return self._parse_sysbench_output(stdout)
        else:
            return {'raw': stdout}
    
    def _parse_pgbench_output(self, output: str) -> Dict:
        """Parse pgbench output"""
        lines = output.split('\n')
        result = {}
        
        for line in lines:
            if 'tps =' in line:
                result['tps'] = float(line.split('tps =')[1].split()[0])
            elif 'latency average' in line:
                result['avg_latency_ms'] = float(
                    line.split('=')[1].split()[0]
                )
            elif 'latency stddev' in line:
                result['stddev_latency_ms'] = float(
                    line.split('=')[1].split()[0]
                )
        
        return result
    
    def _parse_sysbench_output(self, output: str) -> Dict:
        """Parse sysbench output"""
        lines = output.split('\n')
        result = {}
        
        for line in lines:
            if 'transactions:' in line:
                result['tps'] = float(
                    line.split('(')[1].split('per')[0]
                )
            elif 'avg:' in line and 'ms' in line:
                result['avg_latency_ms'] = float(
                    line.split(':')[1].split('ms')[0].strip()
                )
            elif '95th percentile:' in line:
                result['p95_latency_ms'] = float(
                    line.split(':')[1].split('ms')[0].strip()
                )
        
        return result
    
    def save_results(self, results: Dict, filename: str = None):
        """Save benchmark results to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"results_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {filepath}")
        return filepath
    
    def compare_with_baseline(self, current_results: Dict) -> Dict:
        """Compare current results with stored baseline"""
        if not self.baseline_file.exists():
            return {
                'has_baseline': False,
                'message': 'No baseline available'
            }
        
        with open(self.baseline_file) as f:
            baseline = json.load(f)
        
        comparison = {
            'baseline': baseline.get('metrics', {}),
            'current': current_results.get('metrics', {}),
            'changes': {}
        }
        
        # Calculate changes
        for metric in ['tps', 'avg_latency_ms', 'p95_latency_ms']:
            if metric in baseline.get('metrics', {}) and metric in current_results.get('metrics', {}):
                base_val = baseline['metrics'][metric]
                curr_val = current_results['metrics'][metric]
                
                if base_val and curr_val:
                    if 'latency' in metric:
                        change = ((curr_val - base_val) / base_val) * 100
                    else:
                        change = ((curr_val - base_val) / base_val) * 100
                    
                    comparison['changes'][metric] = change
        
        return comparison
    
    def update_baseline(self, results: Dict):
        """Update the baseline with current results"""
        with open(self.baseline_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Baseline updated")
```

### Multi-Version Comparison

```python
#!/usr/bin/env python3
"""Compare performance across database versions"""

class DatabaseVersionComparator:
    """Compare database performance across versions"""
    
    def __init__(self):
        self.results = {}
    
    def add_version_results(self, version: str, results: Dict):
        """Add results for a specific version"""
        self.results[version] = results
    
    def compare_versions(self, version_a: str, version_b: str) -> Dict:
        """Compare two versions"""
        if version_a not in self.results:
            return {'error': f'Version {version_a} not found'}
        if version_b not in self.results:
            return {'error': f'Version {version_b} not found'}
        
        a = self.results[version_a]
        b = self.results[version_b]
        
        comparison = {
            'versions': {
                'a': version_a,
                'b': version_b
            },
            'metrics': {}
        }
        
        for metric in ['tps', 'avg_latency_ms', 'p95_latency_ms', 'p99_latency_ms']:
            if metric in a and metric in b:
                val_a = a[metric]
                val_b = b[metric]
                
                if val_a and val_b:
                    if 'latency' in metric:
                        improvement = ((val_a - val_b) / val_a) * 100
                    else:
                        improvement = ((val_b - val_a) / val_a) * 100
                    
                    comparison['metrics'][metric] = {
                        'version_a': val_a,
                        'version_b': val_b,
                        'improvement_percent': improvement,
                        'winner': version_b if improvement > 0 else version_a
                    }
        
        return comparison
    
    def generate_report(self) -> str:
        """Generate comparison report"""
        versions = sorted(self.results.keys())
        
        report = ["# Database Version Performance Comparison\n"]
        
        for i in range(len(versions) - 1):
            comparison = self.compare_versions(versions[i], versions[i + 1])
            report.append(f"## {versions[i]} vs {versions[i+1]}\n")
            
            for metric, data in comparison.get('metrics', {}).items():
                report.append(f"### {metric}")
                report.append(f"- {versions[i]}: {data['version_a']}")
                report.append(f"- {versions[i+1]}: {data['version_b']}")
                report.append(f"- Improvement: {data['improvement_percent']:.2f}%")
                report.append(f"- Winner: {data['winner']}\n")
        
        return "\n".join(report)
```

---

## CI/CD Integration for Performance Tests

### GitHub Actions Workflow

```yaml
# .github/workflows/database-benchmark.yml
name: Database Performance Benchmark

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    # Run nightly benchmark
    - cron: '0 2 * * *'

env:
  POSTGRES_HOST: localhost
  POSTGRES_DB: benchmark_db
  POSTGRES_USER: benchmark
  POSTGRES_PASSWORD: ${{ secrets.BENCHMARK_PASSWORD }}

jobs:
  benchmark:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_DB: benchmark_db
          POSTGRES_USER: benchmark
          POSTGRES_PASSWORD: ${{ secrets.BENCHMARK_PASSWORD }}
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install psycopg2-binary pandas numpy scipy
          pip install pgbench  # Or use Docker
      
      - name: Initialize database
        run: |
          PGPASSWORD=${{ secrets.BENCHMARK_PASSWORD }} \
          psql -h localhost -U benchmark -d benchmark_db -c "SELECT version();"
      
      - name: Run baseline benchmark (on main)
        if: github.ref == 'refs/heads/main'
        run: |
          python scripts/run_benchmark.py \
            --baseline \
            --output baseline_results.json
      
      - name: Run comparison benchmark (on PR)
        if: github.event_name == 'pull_request'
        run: |
          # Run benchmark on PR branch
          python scripts/run_benchmark.py \
            --output pr_results.json \
            --compare-with baseline_results.json
      
      - name: Run performance regression test
        run: |
          python scripts/detect_regression.py \
            --baseline baseline.json \
            --current pr_results.json \
            --threshold 0.10
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: |
            *.json
            *.html
      
      - name: Comment on PR
        if: github.event_name == 'pull_request'
        run: |
          python scripts/comment_on_pr.py \
            --results pr_results.json \
            --baseline baseline_results.json
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  # Schedule database maintenance and re-index
  maintain:
    needs: benchmark
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    steps:
      - name: Analyze and maintain database
        run: |
          # Run ANALYZE to update statistics
          # Re-index if necessary
          psql -h localhost -U benchmark -d benchmark_db -c "
            ANALYZE;
            REINDEX DATABASE benchmark_db;
          "
```

### GitLab CI Pipeline

```yaml
# .gitlab-ci.yml
stages:
  - benchmark
  - analyze
  - report

variables:
  POSTGRES_DB: benchmark_db
  POSTGRES_USER: benchmark

benchmark:postgresql:
  image: postgres:15
  stage: benchmark
  services:
    - postgres:15
  variables:
    POSTGRES_DB: benchmark_db
    POSTGRES_USER: benchmark
    POSTGRES_PASSWORD: secure_password
  before_script:
    - apt-get update && apt-get install -y postgresql-client python3-pip
    - pip3 install psycopg2-binary pandas numpy
    - pgbench -i -s 100 -h postgres benchmark_db
  script:
    - python3 scripts/run_benchmark.py --output results.json
  artifacts:
    paths:
      - results.json
    expire_in: 30 days

analyze:regression:
  stage: analyze
  image: python:3.11
  needs:
    - benchmark:postgresql
  script:
    - pip install -r requirements.txt
    - python scripts/detect_regression.py
  allow_failure: false  # Fail pipeline on regression

report:performance:
  stage: report
  image: python:3.11
  needs:
    - analyze:regression
  script:
    - pip install matplotlib pandas
    - python scripts/generate_report.py
  artifacts:
    paths:
      - performance_report.html
    expire_in: 90 days
```

### Jenkins Pipeline

```groovy
// Jenkinsfile
pipeline {
    agent any
    
    environment {
        DB_HOST = 'database-server'
        DB_NAME = 'benchmark_db'
    }
    
    stages {
        stage('Setup') {
            steps {
                sh '''
                    pip install psycopg2-binary pandas numpy scipy
                    docker-compose up -d database
                    sleep 10
                '''
            }
        }
        
        stage('Initialize Database') {
            steps {
                sh '''
                    pgbench -i -s 100 -h $DB_HOST $DB_NAME
                '''
            }
        }
        
        stage('Run Benchmark') {
            steps {
                script {
                    def results = sh(
                        script: 'python run_benchmark.py --output benchmark.json',
                        returnStdout: true
                    )
                    env.BENCHMARK_RESULTS = results
                }
            }
        }
        
        stage('Compare with Baseline') {
            steps {
                script {
                    def comparison = sh(
                        script: 'python compare_baseline.py --current benchmark.json --baseline baseline.json',
                        returnStdout: true
                    )
                    
                    if (comparison.contains('REGRESSION_DETECTED')) {
                        unstable('Performance regression detected')
                    }
                }
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: '*.json,*.html', allowEmptyArchive: true
            emailext (
                subject: "Database Benchmark Results - ${currentBuild.result}",
                body: "Results attached",
                attachLog: true
            )
        }
    }
}
```

### Benchmark Runner Script

```python
#!/usr/bin/env python3
"""Complete benchmark runner for CI/CD integration"""

import subprocess
import json
import sys
import os
from datetime import datetime

class BenchmarkRunner:
    def __init__(self, config_file='benchmark_config.json'):
        with open(config_file) as f:
            self.config = json.load(f)
        
        self.db_config = self.config['database']
        self.benchmark_config = self.config['benchmark']
    
    def run(self):
        """Execute the full benchmark suite"""
        print("=" * 60)
        print("DATABASE PERFORMANCE BENCHMARK")
        print("=" * 60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'tests': []
        }
        
        # Initialize database
        self.initialize_database()
        
        # Run each test scenario
        for test in self.benchmark_config['scenarios']:
            print(f"\nRunning: {test['name']}")
            test_result = self.run_scenario(test)
            results['tests'].append(test_result)
            
            # Check for regression if baseline exists
            if 'baseline' in os.environ:
                regression = self.check_regression(test_result, os.environ['baseline'])
                test_result['regression'] = regression
                
                if regression['detected']:
                    print(f"WARNING: Regression detected in {test['name']}!")
                    print(f"  TPS change: {regression['tps_change']:.2f}%")
                    print(f"  Latency change: {regression['latency_change']:.2f}%")
        
        # Save results
        output_file = self.benchmark_config.get('output', 'benchmark_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to {output_file}")
        
        # Exit with appropriate code
        if any(t.get('regression', {}).get('detected', False) for t in results['tests']):
            print("\nPERFORMANCE REGRESSION DETECTED!")
            sys.exit(1)
        
        sys.exit(0)
    
    def initialize_database(self):
        """Initialize the test database"""
        scale = self.benchmark_config.get('scale_factor', 100)
        
        cmd = [
            'pgbench', '-i', '-s', str(scale),
            '-h', self.db_config['host'],
            '-p', str(self.db_config.get('port', 5432)),
            '-U', self.db_config['user'],
            self.db_config['database']
        ]
        
        print(f"Initializing database with scale factor {scale}...")
        subprocess.run(cmd, check=True, capture_output=True)
    
    def run_scenario(self, scenario: dict) -> dict:
        """Run a single benchmark scenario"""
        cmd = [
            'pgbench',
            '-c', str(scenario.get('clients', 10)),
            '-j', str(scenario.get('threads', 2)),
            '-T', str(scenario.get('duration', 60)),
            '-r',  # Report latencies
            '-h', self.db_config['host'],
            '-p', str(self.db_config.get('port', 5432)),
            '-U', self.db_config['user'],
            self.db_config['database']
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse output
        parsed = self.parse_pgbench_output(result.stdout)
        
        return {
            'name': scenario['name'],
            'configuration': scenario,
            'results': parsed,
            'exit_code': result.returncode
        }
    
    def parse_pgbench_output(self, output: str) -> dict:
        """Parse pgbench output"""
        lines = output.split('\n')
        result = {}
        
        for line in lines:
            if 'tps =' in line:
                result['tps'] = float(line.split('tps =')[1].split()[0])
            elif 'latency average' in line:
                result['avg_latency_ms'] = float(line.split('=')[1].split()[0])
            elif 'latency stddev' in line:
                result['stddev_latency_ms'] = float(line.split('=')[1].split()[0])
        
        return result
    
    def check_regression(self, current: dict, baseline_file: str) -> dict:
        """Check for performance regression"""
        try:
            with open(baseline_file) as f:
                baseline = json.load(f)
        except FileNotFoundError:
            return {'detected': False, 'reason': 'No baseline'}
        
        threshold = self.benchmark_config.get('regression_threshold', 0.10)
        
        # Find matching baseline test
        baseline_test = None
        for test in baseline.get('tests', []):
            if test['name'] == current['name']:
                baseline_test = test
                break
        
        if not baseline_test:
            return {'detected': False, 'reason': 'No matching baseline'}
        
        # Calculate changes
        curr_tps = current['results'].get('tps', 0)
        base_tps = baseline_test['results'].get('tps', 0)
        
        curr_lat = current['results'].get('avg_latency_ms', 0)
        base_lat = baseline_test['results'].get('avg_latency_ms', 0)
        
        tps_change = ((curr_tps - base_tps) / base_tps) if base_tps > 0 else 0
        latency_change = ((curr_lat - base_lat) / base_lat) if base_lat > 0 else 0
        
        return {
            'detected': tps_change < -threshold or latency_change > threshold,
            'tps_change': tps_change * 100,
            'latency_change': latency_change * 100,
            'threshold': threshold * 100
        }

if __name__ == '__main__':
    runner = BenchmarkRunner()
    runner.run()
```

---

## Alerting on Performance Degradation

### Alert Configuration System

```python
#!/usr/bin/env python3
"""Performance alerting system for database benchmarks"""

import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict, Optional
import subprocess

class PerformanceAlertManager:
    """Manage performance alerts and notifications"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.alert_history = []
    
    def check_and_alert(self, benchmark_results: Dict, 
                       baseline: Dict = None) -> List[Dict]:
        """Check results and generate alerts if needed"""
        alerts = []
        
        if not baseline:
            # No baseline to compare, skip alerts
            return alerts
        
        # Check TPS degradation
        tps_threshold = self.config.get('tps_degradation_threshold', 0.10)
        current_tps = benchmark_results.get('tps', 0)
        baseline_tps = baseline.get('tps', 0)
        
        if baseline_tps > 0:
            tps_change = (current_tps - baseline_tps) / baseline_tps
            
            if tps_change < -tps_threshold:
                alerts.append({
                    'severity': 'critical',
                    'type': 'tps_degradation',
                    'message': f"TPS degraded by {abs(tps_change)*100:.1f}%",
                    'current': current_tps,
                    'baseline': baseline_tps,
                    'change_percent': tps_change * 100
                })
        
        # Check latency increase
        latency_threshold = self.config.get('latency_increase_threshold', 0.20)
        current_latency = benchmark_results.get('p95_latency_ms', 0)
        baseline_latency = baseline.get('p95_latency_ms', 0)
        
        if baseline_latency > 0:
            latency_change = (current_latency - baseline_latency) / baseline_latency
            
            if latency_change > latency_threshold:
                alerts.append({
                    'severity': 'critical',
                    'type': 'latency_increase',
                    'message': f"P95 latency increased by {latency_change*100:.1f}%",
                    'current': current_latency,
                    'baseline': baseline_latency,
                    'change_percent': latency_change * 100
                })
        
        # Check error rate
        error_threshold = self.config.get('error_rate_threshold', 0.01)
        errors = benchmark_results.get('errors', 0)
        total_ops = benchmark_results.get('total_operations', 1)
        
        if total_ops > 0:
            error_rate = errors / total_ops
            
            if error_rate > error_threshold:
                alerts.append({
                    'severity': 'warning',
                    'type': 'error_rate',
                    'message': f"Error rate {error_rate*100:.2f}% exceeds threshold",
                    'error_rate': error_rate,
                    'threshold': error_threshold
                })
        
        # Store and process alerts
        self.alert_history.extend(alerts)
        
        for alert in alerts:
            self.send_alert(alert, benchmark_results, baseline)
        
        return alerts
    
    def send_alert(self, alert: Dict, results: Dict, baseline: Dict = None):
        """Send alert notification"""
        alert_methods = self.config.get('alert_methods', ['log'])
        
        if 'email' in alert_methods:
            self.send_email_alert(alert, results)
        
        if 'slack' in alert_methods:
            self.send_slack_alert(alert, results)
        
        if 'log' in alert_methods:
            self.log_alert(alert, results)
        
        if 'webhook' in alert_methods:
            self.send_webhook_alert(alert, results)
    
    def send_email_alert(self, alert: Dict, results: Dict):
        """Send email alert"""
        email_config = self.config.get('email', {})
        
        if not email_config:
            return
        
        msg = MIMEMultipart()
        msg['From'] = email_config.get('from', 'benchmark@example.com')
        msg['To'] = ', '.join(email_config.get('recipients', []))
        msg['Subject'] = f"[{alert['severity'].upper()}] Database Performance Alert"
        
        body = f"""
Database Performance Alert
=========================

Type: {alert['type']}
Severity: {alert['severity']}
Message: {alert['message']}

Details:
- Timestamp: {datetime.now().isoformat()}
- Current Value: {alert.get('current', 'N/A')}
- Baseline Value: {alert.get('baseline', 'N/A')}
- Change: {alert.get('change_percent', 0):.2f}%

Full Results:
{json.dumps(results, indent=2)}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            with smtplib.SMTP(
                email_config.get('smtp_host', 'localhost'),
                email_config.get('smtp_port', 25)
            ) as server:
                server.send_message(msg)
            print(f"Email alert sent: {alert['message']}")
        except Exception as e:
            print(f"Failed to send email: {e}")
    
    def send_slack_alert(self, alert: Dict, results: Dict):
        """Send Slack webhook alert"""
        slack_config = self.config.get('slack', {})
        webhook_url = slack_config.get('webhook_url')
        
        if not webhook_url:
            return
        
        color = 'danger' if alert['severity'] == 'critical' else 'warning'
        
        payload = {
            'attachments': [{
                'color': color,
                'title': f"Database Performance Alert: {alert['type']}",
                'text': alert['message'],
                'fields': [
                    {'title': 'Severity', 'value': alert['severity'], 'short': True},
                    {'title': 'Change', 'value': f"{alert.get('change_percent', 0):.2f}%", 'short': True}
                ]
            }]
        }
        
        import requests
        try:
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            print(f"Slack alert sent: {alert['message']}")
        except Exception as e:
            print(f"Failed to send Slack alert: {e}")
    
    def log_alert(self, alert: Dict, results: Dict):
        """Log alert to file/console"""
        timestamp = datetime.now().isoformat()
        severity = alert['severity'].upper()
        message = alert['message']
        
        log_line = f"[{timestamp}] [{severity}] {message}\n"
        
        print(log_line)
        
        # Also write to alert log file
        log_file = self.config.get('alert_log_file', 'alerts.log')
        with open(log_file, 'a') as f:
            f.write(log_line)
    
    def send_webhook_alert(self, alert: Dict, results: Dict):
        """Send generic webhook alert"""
        webhook_config = self.config.get('webhook', {})
        webhook_url = webhook_config.get('url')
        
        if not webhook_url:
            return
        
        payload = {
            'alert': alert,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        import requests
        try:
            response = requests.post(
                webhook_url,
                json=payload,
                headers=webhook_config.get('headers', {}),
                timeout=10
            )
            response.raise_for_status()
            print(f"Webhook alert sent: {alert['message']}")
        except Exception as e:
            print(f"Failed to send webhook alert: {e}")
```

### Prometheus Alerting Integration

```yaml
# prometheus-alerts.yml
groups:
  - name: database_performance
    interval: 30s
    rules:
      # TPS degradation alert
      - alert: DatabaseTPSRegression
        expr: |
          (
            rate(pg_stat_statements_calls_total[5m]) 
            / rate(pg_stat_statements_calls_total[5m] offset 1h)
          ) < 0.9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Database TPS regression detected"
          description: "Current TPS is {{ $value | humanizePercentage }} of baseline"
      
      # High latency alert
      - alert: DatabaseHighLatency
        expr: |
          histogram_quantile(0.95, 
            rate(pg_stat_statements_exec_time_bucket[5m])
          ) > 500
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Database latency above threshold"
          description: "P95 latency is {{ $value }}ms"
      
      # Connection pool exhaustion
      - alert: DatabaseConnectionPoolExhausted
        expr: |
          pg_stat_activity_count / pg_settings_max_connections > 0.9
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection pool near exhaustion"
          description: "{{ $value | humanizePercentage }} of connections in use"
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Database Performance Benchmark",
    "panels": [
      {
        "title": "Transactions Per Second",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(pg_stat_statements_calls_total[1m])",
            "legendFormat": "{{query}}"
          }
        ],
        "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8}
      },
      {
        "title": "Query Latency (P95)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(pg_stat_statements_exec_time_bucket[5m]))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(pg_stat_statements_exec_time_bucket[5m]))",
            "legendFormat": "P99"
          }
        ],
        "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8}
      },
      {
        "title": "Benchmark Comparison",
        "type": "bargauge",
        "targets": [
          {
            "expr": "benchmark_tps",
            "legendFormat": "Current"
          },
          {
            "expr": "benchmark_tps_baseline",
            "legendFormat": "Baseline"
          }
        ],
        "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8}
      }
    ]
  }
}
```

---

## Complete Example: Full Testing Pipeline

```python
#!/usr/bin/env python3
"""Complete database performance testing pipeline"""

import json
import sys
from datetime import datetime
import argparse
import subprocess
import psycopg2

class DatabasePerformancePipeline:
    """End-to-end performance testing pipeline"""
    
    def __init__(self, args):
        self.args = args
        self.results = {}
    
    def run(self):
        """Execute the full pipeline"""
        print("=" * 60)
        print("DATABASE PERFORMANCE TESTING PIPELINE")
        print(f"Started: {datetime.now().isoformat()}")
        print("=" * 60)
        
        # Step 1: Setup
        self.setup()
        
        # Step 2: Run benchmark
        self.run_benchmark()
        
        # Step 3: Analyze results
        self.analyze_results()
        
        # Step 4: Compare with baseline
        if self.args.compare:
            self.compare_with_baseline()
        
        # Step 5: Generate report
        self.generate_report()
        
        # Step 6: Alert if needed
        self.alert()
        
        print("\nPipeline complete!")
        return 0 if not self.results.get('regression_detected') else 1
    
    def setup(self):
        """Setup test environment"""
        print("\n[1/6] Setting up test environment...")
        
        # Connect to database
        self.db_conn = psycopg2.connect(
            host=self.args.host,
            port=self.args.port,
            database=self.args.database,
            user=self.args.user,
            password=self.args.password
        )
        
        # Initialize with pgbench if requested
        if self.args.initialize:
            cmd = [
                'pgbench', '-i', '-s', str(self.args.scale),
                '-h', self.args.host,
                '-p', str(self.args.port),
                '-U', self.args.user,
                self.args.database
            ]
            subprocess.run(cmd, check=True)
            print(f"  Database initialized with scale factor {self.args.scale}")
    
    def run_benchmark(self):
        """Run the performance benchmark"""
        print("\n[2/6] Running performance benchmark...")
        
        cmd = [
            'pgbench',
            '-c', str(self.args.clients),
            '-j', str(self.args.threads),
            '-T', str(self.args.duration),
            '-r',
            '-h', self.args.host,
            '-p', str(self.args.port),
            '-U', self.args.user,
            self.args.database
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse output
        self.results = self.parse_pgbench(result.stdout)
        self.results['raw_output'] = result.stdout
        
        print(f"  TPS: {self.results.get('tps', 0):.2f}")
        print(f"  Avg Latency: {self.results.get('avg_latency_ms', 0):.2f}ms")
        print(f"  P95 Latency: {self.results.get('p95_latency_ms', 0):.2f}ms")
    
    def parse_pgbench(self, output: str) -> dict:
        """Parse pgbench output"""
        lines = output.split('\n')
        result = {}
        
        for line in lines:
            if 'tps =' in line:
                result['tps'] = float(line.split('tps =')[1].split()[0])
            elif 'latency average' in line:
                result['avg_latency_ms'] = float(line.split('=')[1].split()[0])
            elif 'latency stddev' in line:
                result['stddev_latency_ms'] = float(line.split('=')[1].split()[0])
            elif 'number of transactions' in line:
                result['total_transactions'] = int(
                    line.split(':')[1].strip().split()[0]
                )
        
        return result
    
    def analyze_results(self):
        """Analyze benchmark results"""
        print("\n[3/6] Analyzing results...")
        
        # Add database metrics
        cursor = self.db_conn.cursor()
        
        # Get connection count
        cursor.execute("SELECT count(*) FROM pg_stat_activity")
        self.results['active_connections'] = cursor.fetchone()[0]
        
        # Get database size
        cursor.execute("""
            SELECT pg_size_pretty(pg_database_size(current_database()))
        """)
        self.results['database_size'] = cursor.fetchone()[0]
        
        self.results['timestamp'] = datetime.now().isoformat()
        self.results['config'] = {
            'clients': self.args.clients,
            'threads': self.args.threads,
            'duration': self.args.duration,
            'scale': self.args.scale
        }
        
        print(f"  Active connections: {self.results['active_connections']}")
        print(f"  Database size: {self.results['database_size']}")
    
    def compare_with_baseline(self):
        """Compare with baseline results"""
        print("\n[4/6] Comparing with baseline...")
        
        try:
            with open(self.args.baseline) as f:
                baseline = json.load(f)
        except FileNotFoundError:
            print(f"  Warning: Baseline file not found: {self.args.baseline}")
            return
        
        baseline_metrics = baseline.get('results', {})
        
        # Calculate changes
        tps_change = 0
        if baseline_metrics.get('tps'):
            tps_change = (
                (self.results['tps'] - baseline_metrics['tps']) 
                / baseline_metrics['tps']
            ) * 100
        
        latency_change = 0
        if baseline_metrics.get('avg_latency_ms'):
            latency_change = (
                (self.results['avg_latency_ms'] - baseline_metrics['avg_latency_ms'])
                / baseline_metrics['avg_latency_ms']
            ) * 100
        
        self.results['comparison'] = {
            'baseline_file': self.args.baseline,
            'baseline_tps': baseline_metrics.get('tps'),
            'baseline_latency': baseline_metrics.get('avg_latency_ms'),
            'tps_change_percent': tps_change,
            'latency_change_percent': latency_change
        }
        
        # Determine if regression detected
        regression = tps_change < -10 or latency_change > 20
        self.results['regression_detected'] = regression
        
        print(f"  Baseline TPS: {baseline_metrics.get('tps', 'N/A')}")
        print(f"  Current TPS: {self.results['tps']:.2f}")
        print(f"  Change: {tps_change:+.2f}%")
        
        if regression:
            print("  ⚠️  REGRESSION DETECTED!")
    
    def generate_report(self):
        """Generate results report"""
        print("\n[5/6] Generating report...")
        
        output_file = self.args.output or 'benchmark_results.json'
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"  Report saved to: {output_file}")
    
    def alert(self):
        """Send alerts if configured"""
        print("\n[6/6] Checking alerts...")
        
        if not self.results.get('regression_detected'):
            print("  No alerts - performance acceptable")
            return
        
        # Simple console alert
        print("\n" + "=" * 40)
        print("⚠️  PERFORMANCE ALERT ⚠️")
        print("=" * 40)
        print(f"Regression detected in benchmark!")
        
        comparison = self.results.get('comparison', {})
        print(f"  TPS change: {comparison.get('tps_change_percent', 0):.2f}%")
        print(f"  Latency change: {comparison.get('latency_change_percent', 0):.2f}%")
        print("=" * 40)


def main():
    parser = argparse.ArgumentParser(
        description='Database Performance Testing Pipeline'
    )
    
    # Database connection
    parser.add_argument('--host', default='localhost', help='Database host')
    parser.add_argument('--port', type=int, default=5432, help='Database port')
    parser.add_argument('--database', default='benchmark_db', help='Database name')
    parser.add_argument('--user', default='postgres', help='Database user')
    parser.add_argument('--password', default='', help='Database password')
    
    # Benchmark configuration
    parser.add_argument('--clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--threads', type=int, default=2, help='Number of threads')
    parser.add_argument('--duration', type=int, default=60, help='Test duration (seconds)')
    parser.add_argument('--scale', type=int, default=100, help='Scale factor')
    
    # Options
    parser.add_argument('--initialize', action='store_true', help='Initialize database')
    parser.add_argument('--compare', action='store_true', help='Compare with baseline')
    parser.add_argument('--baseline', help='Baseline results file')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    pipeline = DatabasePerformancePipeline(args)
    sys.exit(pipeline.run())


if __name__ == '__main__':
    main()
```

---

## Summary

### Key Takeaways

1. **Regression detection** is essential for catching performance degradation early
2. **Baseline management** allows meaningful comparisons over time
3. **Statistical methods** help distinguish real regressions from variance
4. **CI/CD integration** ensures performance is tested on every change
5. **Multi-level alerting** (email, Slack, webhooks) keeps teams informed
6. **Query-level detection** identifies specific queries that regressed

### Testing Checklist

```markdown
Baseline Setup:
[ ] Establish initial baseline with stable configuration
[ ] Document system configuration and versions
[ ] Store baseline in version control
[ ] Set up automated baseline refresh (weekly/monthly)

Regression Detection:
[ ] Define regression thresholds (TPS, latency)
[ ] Implement statistical significance testing
[ ] Set up query-level regression detection
[ ] Configure alerting for different severity levels

CI/CD Integration:
[ ] Add benchmark step to pipeline
[ ] Configure pass/fail criteria
[ ] Set up result storage and visualization
[ ] Add PR comments with results

Monitoring:
[ ] Set up Grafana dashboards
[ ] Configure Prometheus alerts
[ ] Create runbooks for responding to alerts
[ ] Schedule regular benchmark runs (nightly/weekly)
```

### Best Practices

1. **Always warm up** the database before benchmarking
2. **Run multiple times** to ensure statistical validity
3. **Isolate environment** from other workloads during testing
4. **Track configuration** - document everything that might affect results
5. **Automate everything** - manual testing is not reproducible
6. **Fail the build** on significant regressions
7. **Review trends** over time, not just individual runs

---

## Additional Resources

- [pg_stat_statements](https://www.postgresql.org/docs/current/pgstatstatements.html)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Prometheus Alerting](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)
- [PostgreSQL Monitoring](https://www.postgresql.org/docs/current/monitoring.html)
