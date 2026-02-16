# 数据库调试完全指南

## 概述

数据库调试是解决数据库性能问题、故障排除和数据恢复的关键技能。本文档提供全面的数据库调试策略，涵盖从常见问题诊断到复杂故障恢复的完整体系。

---

## 目录

1. [常见数据库问题与解决方案](#1-常见数据库问题与解决方案)
2. [查询性能调试](#2-查询性能调试)
3. [锁与死锁调试](#3-锁与死锁调试)
4. [内存与资源调试](#4-内存与资源调试)
5. [网络与连接调试](#5-网络与连接调试)
6. [恢复程序](#6-恢复程序)

---

## 1. 常见数据库问题与解决方案

### 1.1 问题分类与诊断流程

数据库问题可以按照影响范围和严重程度进行分类。理解问题的性质是快速定位和解决的关键。以下是常见问题类型的诊断流程图：

```
┌─────────────────────────────────────────────────────────────┐
│                     数据库问题诊断流程                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   问题现象观察   │
                    │ (错误信息/性能) │
                    └────────┬────────┘
                             │
             ┌───────────────┼───────────────┐
             │               │               │
             ▼               ▼               ▼
      ┌────────────┐  ┌────────────┐  ┌────────────┐
      │ 连接问题   │  │ 性能问题   │  │ 数据问题   │
      │ (超时/拒绝)│  │ (慢查询)   │  │ (丢失/损坏)│
      └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
            │                │                │
            ▼                ▼                ▼
      ┌───────────┐   ┌───────────┐    ┌───────────┐
      │检查网络   │   │分析执行   │    │检查日志   │
      │检查认证   │   │计划/索引  │    │验证备份   │
      └───────────┘   └───────────┘    └───────────┘
```

### 1.1.1 连接问题诊断

连接问题是数据库最常见的问题之一，可能由多种原因引起。以下是一个系统化的诊断方法：

#### 网络连通性检查

```python
import socket
import time
from contextlib import contextmanager

class DatabaseConnectivity诊断:
    """数据库连接诊断工具"""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
    
    def check_port_open(self, timeout: float = 5.0) -> dict:
        """检查端口是否开放"""
        result = {
            'host': self.host,
            'port': self.port,
            'open': False,
            'response_time_ms': None,
            'error': None
        }
        
        start_time = time.time()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            connect_result = sock.connect_ex((self.host, self.port))
            sock.close()
            
            result['open'] = (connect_result == 0)
            result['response_time_ms'] = (time.time() - start_time) * 1000
            
        except socket.gaierror as e:
            result['error'] = f"DNS解析失败: {str(e)}"
        except socket.timeout:
            result['error'] = "连接超时"
        except Exception as e:
            result['error'] = f"未知错误: {str(e)}"
        
        return result
    
    def check_dns_resolution(self) -> dict:
        """检查DNS解析"""
        result = {
            'hostname': self.host,
            'resolved': False,
            'ip_addresses': [],
            'error': None
        }
        
        try:
            ip_list = socket.getaddrinfo(self.host, self.port)
            result['resolved'] = True
            result['ip_addresses'] = list(set(item[4][0] for item in ip_list))
        except socket.gaierror as e:
            result['error'] = f"DNS解析失败: {str(e)}"
        
        return result
    
    def check_firewall_rules(self) -> dict:
        """检查防火墙规则（Linux示例）"""
        import subprocess
        
        result = {
            'firewall_checked': False,
            'iptables_rules': [],
            'likely_blocked': False
        }
        
        try:
            # 检查iptables规则
            cmd = f"iptables -L -n | grep {self.port}"
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True
            )
            result['iptables_rules'] = proc.stdout.strip().split('\n')
            result['firewall_checked'] = True
            
            # 如果有DROP规则针对该端口，可能被阻止
            for rule in result['iptables_rules']:
                if 'DROP' in rule and str(self.port) in rule:
                    result['likely_blocked'] = True
                    
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def diagnose_connection_issues(self) -> dict:
        """综合诊断连接问题"""
        diagnosis = {
            'timestamp': time.time(),
            'checks': {}
        }
        
        # 1. DNS解析检查
        diagnosis['checks']['dns'] = self.check_dns_resolution()
        
        # 2. 端口开放检查
        diagnosis['checks']['port'] = self.check_port_open()
        
        # 3. 防火墙检查
        diagnosis['checks']['firewall'] = self.check_firewall_rules()
        
        # 4. 综合结论
        if not diagnosis['checks']['dns']['resolved']:
            diagnosis['conclusion'] = 'DNS解析失败，请检查主机名'
        elif not diagnosis['checks']['port']['open']:
            diagnosis['conclusion'] = '端口不可达，可能是防火墙阻止或服务未启动'
        elif diagnosis['checks']['firewall'].get('likely_blocked'):
            diagnosis['conclusion'] = '可能被防火墙阻止'
        else:
            diagnosis['conclusion'] = '网络连接正常，可能是认证或配置问题'
        
        return diagnosis


# 使用示例
def diagnose_database_connection(host: str, port: int):
    """诊断数据库连接问题"""
    diagnostic = DatabaseConnectivity诊断(host, port)
    result = diagnostic.diagnose_connection_issues()
    
    print(f"诊断时间: {result['timestamp']}")
    print(f"DNS解析: {result['checks']['dns']}")
    print(f"端口状态: {result['checks']['port']}")
    print(f"防火墙: {result['checks']['firewall']}")
    print(f"结论: {result['conclusion']}")
    
    return result
```

#### 认证问题诊断

```python
import hashlib
import hmac
from datetime import datetime

class DatabaseAuthentication诊断:
    """数据库认证问题诊断"""
    
    def __init__(self, db_type: str):
        self.db_type = db_type.lower()
    
    def check_user_exists(self, connection, username: str) -> dict:
        """检查用户是否存在"""
        result = {
            'user_exists': False,
            'user_details': None,
            'error': None
        }
        
        try:
            if self.db_type == 'postgresql':
                query = """
                    SELECT usename, usecreatedb, usesuper, passwd 
                    FROM pg_user 
                    WHERE usename = %s
                """
            elif self.db_type == 'mysql':
                query = """
                    SELECT User, Host, plugin, authentication_string 
                    FROM mysql.user 
                    WHERE User = %s
                """
            
            cursor = connection.cursor()
            cursor.execute(query, (username,))
            user = cursor.fetchone()
            
            result['user_exists'] = (user is not None)
            if user:
                if self.db_type == 'postgresql':
                    result['user_details'] = {
                        'username': user[0],
                        'can_create_db': user[1],
                        'is_superuser': user[2],
                        'password_hash': user[3][:20] + '...' if user[3] else None
                    }
                elif self.db_type == 'mysql':
                    result['user_details'] = {
                        'username': user[0],
                        'host': user[1],
                        'auth_plugin': user[2],
                        'password_set': bool(user[3])
                    }
                    
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def verify_password(self, connection, username: str, password: str) -> dict:
        """验证密码"""
        result = {
            'password_correct': False,
            'method': 'unknown',
            'error': None
        }
        
        try:
            if self.db_type == 'postgresql':
                # PostgreSQL使用MD5或SCRAM认证
                query = "SELECT passwd FROM pg_user WHERE usename = %s"
                cursor = connection.cursor()
                cursor.execute(query, (username,))
                user = cursor.fetchone()
                
                if user and user[0]:
                    stored_hash = user[0]
                    # 验证MD5哈希
                    password_hash = 'md5' + hashlib.md5(
                        (password + username).encode()
                    ).hexdigest()
                    result['password_correct'] = (password_hash == stored_hash)
                    result['method'] = 'MD5'
                    
            elif self.db_type == 'mysql':
                # MySQL使用 caching_sha2_password 或 mysql_native_password
                query = "SELECT authentication_string FROM mysql.user WHERE User = %s"
                cursor = connection.cursor()
                cursor.execute(query, (username,))
                user = cursor.fetchone()
                
                if user and user[0]:
                    # 简化的验证（实际需要完整的认证协议）
                    result['password_correct'] = True  # 需要完整实现
                    result['method'] = 'caching_sha2_password'
                    
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def diagnose_auth_failure(self, connection, username: str, password: str) -> dict:
        """诊断认证失败原因"""
        diagnosis = {
            'timestamp': datetime.now().isoformat(),
            'username': username,
            'diagnosis': {}
        }
        
        # 检查用户是否存在
        user_check = self.check_user_exists(connection, username)
        diagnosis['diagnosis']['user_exists'] = user_check
        
        if not user_check['user_exists']:
            diagnosis['conclusion'] = '用户不存在'
            return diagnosis
        
        # 检查密码
        if password:
            password_check = self.verify_password(connection, username, password)
            diagnosis['diagnosis']['password'] = password_check
            
            if not password_check['password_correct']:
                diagnosis['conclusion'] = '密码错误'
            else:
                diagnosis['conclusion'] = '密码正确，可能是权限问题'
        else:
            diagnosis['conclusion'] = '未提供密码'
        
        # 检查用户权限
        try:
            if self.db_type == 'postgresql':
                query = """
                    SELECT privilege_type, table_schema, table_name 
                    FROM information_schema.usage_privileges 
                    WHERE grantee = %s
                """
            elif self.db_type == 'mysql':
                query = """
                    SELECT Privilege, Db, User, Host 
                    FROM mysql.db 
                    WHERE User = %s
                """
            
            cursor = connection.cursor()
            cursor.execute(query, (username,))
            privileges = cursor.fetchall()
            diagnosis['diagnosis']['privileges'] = list(privileges)
            
        except Exception as e:
            diagnosis['diagnosis']['privileges_error'] = str(e)
        
        return diagnosis
```

### 1.2 性能问题快速定位

```python
import time
import psutil
from datetime import datetime
from typing import List, Dict, Optional

class DatabasePerformance诊断:
    """数据库性能问题诊断"""
    
    def __init__(self, connection):
        self.connection = connection
    
    def get_system_metrics(self) -> Dict:
        """获取系统级指标"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else None,
            'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else None,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_database_connections(self) -> List[Dict]:
        """获取当前数据库连接"""
        try:
            # PostgreSQL
            query = """
                SELECT 
                    pid, usename, application_name, client_addr,
                    backend_start, query_start, state, query,
                    wait_event_type, wait_event
                FROM pg_stat_activity
                WHERE datname = current_database()
                ORDER BY query_start DESC
            """
            cursor = self.connection.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            
            connections = []
            for row in cursor.fetchall():
                connections.append(dict(zip(columns, row)))
            
            return connections
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def identify_blocking_queries(self) -> List[Dict]:
        """识别阻塞查询"""
        try:
            # PostgreSQL阻塞查询
            query = """
                SELECT 
                    blocked_locks.pid AS blocked_pid,
                    blocked_activity.usename AS blocked_user,
                    blocking_locks.pid AS blocking_pid,
                    blocking_activity.usename AS blocking_user,
                    blocked_activity.query AS blocked_query,
                    blocking_activity.query AS blocking_query,
                    blocked_activity.application_name AS blocked_application,
                    blocking_activity.application_name AS blocking_application
                FROM pg_catalog.pg_locks blocked_locks
                JOIN pg_catalog.pg_stat_activity blocked_activity 
                    ON blocked_activity.pid = blocked_locks.pid
                JOIN pg_catalog.pg_locks blocking_locks 
                    ON blocking_locks.locktype = blocked_locks.locktype
                    AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
                    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
                    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
                    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
                    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
                    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
                    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
                    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
                    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
                    AND blocking_locks.pid != blocked_locks.pid
                JOIN pg_catalog.pg_stat_activity blocking_activity 
                    ON blocking_activity.pid = blocking_locks.pid
                WHERE NOT blocked_locks.granted
            """
            
            cursor = self.connection.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            
            blocking = []
            for row in cursor.fetchall():
                blocking.append(dict(zip(columns, row)))
            
            return blocking
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def get_slow_queries(self, min_duration_seconds: float = 1.0) -> List[Dict]:
        """获取慢查询"""
        try:
            query = """
                SELECT 
                    pid, now() - query_start AS duration,
                    usename, query, state
                FROM pg_stat_activity
                WHERE state != 'idle'
                    AND query_start < now() - interval '%s seconds'
                ORDER BY query_start
            """
            
            cursor = self.connection.cursor()
            cursor.execute(query, (min_duration_seconds,))
            columns = [desc[0] for desc in cursor.description]
            
            slow_queries = []
            for row in cursor.fetchall():
                slow_queries.append(dict(zip(columns, row)))
            
            return slow_queries
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def get_index_usage_stats(self) -> List[Dict]:
        """获取索引使用统计"""
        try:
            query = """
                SELECT 
                    schemaname, tablename, indexname,
                    idx_scan, idx_tup_read, idx_tup_fetch,
                    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
                FROM pg_stat_user_indexes
                WHERE idx_scan = 0
                    AND indexrelid NOT IN (SELECT conindid FROM pg_constraint)
                ORDER BY pg_relation_size(indexrelid) DESC
            """
            
            cursor = self.connection.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            
            unused_indexes = []
            for row in cursor.fetchall():
                unused_indexes.append(dict(zip(columns, row)))
            
            return unused_indexes
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def generate_performance_report(self) -> Dict:
        """生成性能诊断报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': self.get_system_metrics(),
            'active_connections': self.get_database_connections(),
            'blocking_queries': self.identify_blocking_queries(),
            'slow_queries': self.get_slow_queries(),
            'unused_indexes': self.get_index_usage_stats(),
            'recommendations': []
        }
        
        # 生成建议
        if len(report['blocking_queries']) > 0:
            report['recommendations'].append(
                "检测到阻塞查询，建议检查长时间运行的事务"
            )
        
        if len(report['slow_queries']) > 5:
            report['recommendations'].append(
                "存在较多慢查询，建议优化查询或增加索引"
            )
        
        if len(report['unused_indexes']) > 0:
            report['recommendations'].append(
                f"发现{len(report['unused_indexes'])}个未使用的索引，考虑删除以提高写入性能"
            )
        
        return report
```

---

## 2. 查询性能调试

### 2.1 执行计划分析

查询执行计划是理解数据库如何执行查询的关键。通过分析执行计划，可以识别性能瓶颈并优化查询。

```python
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class JoinType(Enum):
    """连接类型"""
    NESTED_LOOP = "Nested Loop"
    HASH_JOIN = "Hash Join"
    MERGE_JOIN = "Merge Join"

class ScanType(Enum):
    """扫描类型"""
    SEQ_SCAN = "Seq Scan"
    INDEX_SCAN = "Index Scan"
    INDEX_ONLY_SCAN = "Index Only Scan"
    BITMAP_SCAN = "Bitmap Heap Scan"

@dataclass
class ExecutionPlanNode:
    """执行计划节点"""
    operation: str
    relation: Optional[str]
    startup_cost: float
    total_cost: float
    plan_rows: int
    plan_width: int
    actual_rows: int
    actual_time: float
    children: List['ExecutionPlanNode']
    details: Dict

class QueryPlanAnalyzer:
    """查询计划分析器"""
    
    def __init__(self, connection):
        self.connection = connection
    
    def explain_query(self, query: str, analyze: bool = True, 
                     format_json: bool = True) -> Dict:
        """执行EXPLAIN分析查询"""
        try:
            cursor = self.connection.cursor()
            
            if format_json:
                if analyze:
                    cursor.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}")
                else:
                    cursor.execute(f"EXPLAIN (FORMAT JSON) {query}")
            else:
                if analyze:
                    cursor.execute(f"EXPLAIN (ANALYZE, BUFFERS) {query}")
                else:
                    cursor.execute(f"EXPLAIN {query}")
            
            if format_json:
                result = cursor.fetchone()[0]
                return json.loads(json.dumps(result))
            else:
                return {'text': cursor.fetchall()}
                
        except Exception as e:
            return {'error': str(e)}
    
    def parse_plan(self, plan: Dict) -> List[ExecutionPlanNode]:
        """解析执行计划"""
        nodes = []
        
        def parse_node(node_dict: Dict) -> ExecutionPlanNode:
            return ExecutionPlanNode(
                operation=node_dict.get('Operation', node_dict.get('Node Type', 'Unknown')),
                relation=node_dict.get('Relation Name'),
                startup_cost=node_dict.get('Startup Cost', 0),
                total_cost=node_dict.get('Total Cost', 0),
                plan_rows=node_dict.get('Plan Rows', 0),
                plan_width=node_dict.get('Plan Width', 0),
                actual_rows=node_dict.get('Actual Rows', 0),
                actual_time=node_dict.get('Actual Total Time', 0),
                children=[parse_node(child) for child in node_dict.get('Plans', [])],
                details=node_dict
            )
        
        if 'Plan' in plan:
            return [parse_node(plan['Plan'])]
        
        return nodes
    
    def identify_performance_issues(self, query: str) -> Dict:
        """识别性能问题"""
        plan = self.explain_query(query, analyze=True, format_json=True)
        
        if 'error' in plan:
            return {'error': plan['error']}
        
        issues = {
            'high_cost_operations': [],
            'missing_indexes': [],
            'inefficient_joins': [],
            'sequential_scans': [],
            'recommendations': []
        }
        
        def analyze_node(node: Dict, depth: int = 0):
            node_type = node.get('Node Type', '')
            
            # 检查全表扫描
            if 'Seq Scan' in node_type:
                relation = node.get('Relation Name', 'Unknown')
                issues['sequential_scans'].append({
                    'relation': relation,
                    'rows': node.get('Plan Rows', 0),
                    'cost': node.get('Total Cost', 0)
                })
                
                # 检查是否有WHERE条件但没有索引
                if 'Filter' in node and 'Index Cond' not in node:
                    issues['recommendations'].append(
                        f"表 {relation} 建议添加索引以优化查询"
                    )
            
            # 检查高成本操作
            if node.get('Total Cost', 0) > 1000:
                issues['high_cost_operations'].append({
                    'operation': node_type,
                    'cost': node.get('Total Cost', 0),
                    'relation': node.get('Relation Name')
                })
            
            # 递归分析子节点
            for child in node.get('Plans', []):
                analyze_node(child, depth + 1)
        
        if 'Plan' in plan:
            analyze_node(plan['Plan'])
        
        return issues
    
    def compare_plans(self, query1: str, query2: str) -> Dict:
        """比较两个查询的执行计划"""
        plan1 = self.explain_query(query1, analyze=True)
        plan2 = self.explain_query(query2, analyze=True)
        
        if 'error' in plan1 or 'error' in plan2:
            return {'error': '无法获取查询计划'}
        
        return {
            'query1_plan': plan1,
            'query2_plan': plan2,
            'query1_cost': plan1.get('Plan', {}).get('Total Cost', 0),
            'query2_cost': plan2.get('Plan', {}).get('Total Cost', 0),
            'cost_difference': (
                plan1.get('Plan', {}).get('Total Cost', 0) - 
                plan2.get('Plan', {}).get('Total Cost', 0)
            )
        }
```

### 2.2 索引优化调试

```python
class IndexOptimizer:
    """索引优化工具"""
    
    def __init__(self, connection):
        self.connection = connection
    
    def analyze_missing_indexes(self) -> List[Dict]:
        """分析缺失的索引"""
        query = """
            SELECT
                schemaname,
                tablename,
                indexname,
                idx_scan,
                idx_tup_read,
                idx_tup_fetch,
                pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
            FROM pg_stat_user_indexes
            WHERE indexrelid NOT IN (
                SELECT conindid FROM pg_constraint
            )
            ORDER BY idx_scan ASC, pg_relation_size(indexrelid) DESC
        """
        
        cursor = self.connection.cursor()
        cursor.execute(query)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'schema': row[0],
                'table': row[1],
                'index': row[2],
                'scans': row[3],
                'tuples_read': row[4],
                'tuples_fetched': row[5],
                'size': row[6]
            })
        
        return results
    
    def suggest_indexes_from_queries(self, slow_queries: List[str]) -> List[Dict]:
        """基于慢查询建议索引"""
        suggestions = []
        
        # PostgreSQL的pg_stat_statements需要先启用
        # CREATE EXTENSION IF NOT EXISTS pg_stat_statements
        
        for query in slow_queries:
            # 提取WHERE子句中的列
            import re
            
            # 简单的WHERE条件提取
            where_pattern = r'WHERE\s+(\w+)\s*=\s*'
            matches = re.findall(where_pattern, query, re.IGNORECASE)
            
            for column in matches:
                suggestions.append({
                    'column': column,
                    'reason': 'Used in WHERE clause',
                    'sample_query': query[:100]
                })
        
        return suggestions
    
    def analyze_index_efficiency(self, table_name: str) -> Dict:
        """分析索引效率"""
        cursor = self.connection.cursor()
        
        # 获取表的所有索引
        cursor.execute("""
            SELECT 
                i.relname AS index_name,
                a.attname AS column_name,
                idx_scan,
                idx_tup_read,
                idx_tup_fetch,
                pg_relation_size(i.relid) AS index_size
            FROM pg_class t
            JOIN pg_index ix ON t.oid = ix.indrelid
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
            JOIN pg_stat_user_indexes s ON s.indexrelid = i.oid
            WHERE t.relname = %s
            ORDER BY s.idx_scan DESC
        """, (table_name,))
        
        indexes = []
        for row in cursor.fetchall():
            indexes.append({
                'name': row[0],
                'column': row[1],
                'scans': row[2],
                'tuples_read': row[3],
                'tuples_fetched': row[4],
                'size': row[5]
            })
        
        # 计算总体统计
        total_scans = sum(idx['scans'] for idx in indexes)
        total_size = sum(idx['size'] for idx in indexes)
        
        return {
            'table': table_name,
            'indexes': indexes,
            'total_scans': total_scans,
            'total_size': total_size,
            'recommendations': self._generate_index_recommendations(indexes)
        }
    
    def _generate_index_recommendations(self, indexes: List[Dict]) -> List[str]:
        """生成索引建议"""
        recommendations = []
        
        unused_indexes = [idx for idx in indexes if idx['scans'] == 0]
        if unused_indexes:
            recommendations.append(
                f"建议删除{len(unused_indexes)}个未使用的索引以提高写入性能"
            )
        
        large_indexes = [idx for idx in indexes if idx['size'] > 100_000_000]  # > 100MB
        if large_indexes:
            recommendations.append(
                f"存在{len(large_indexes)}个大型索引，考虑使用索引覆盖或分区"
            )
        
        return recommendations
    
    def create_index_safely(self, table: str, columns: List[str],
                           index_name: str = None, 
                           concurrent: bool = True) -> str:
        """安全创建索引"""
        if index_name is None:
            index_name = f"idx_{table}_{'_'.join(columns)}"
        
        if concurrent:
            return f"""
                CREATE INDEX CONCURRENTLY {index_name}
                ON {table} ({', '.join(columns)})
            """
        else:
            return f"""
                CREATE INDEX {index_name}
                ON {table} ({', '.join(columns)})
            """
```

---

## 3. 锁与死锁调试

### 3.1 锁监控与诊断

```python
import threading
import time
from datetime import datetime
from typing import List, Dict, Optional

class LockMonitor:
    """数据库锁监控器"""
    
    def __init__(self, connection):
        self.connection = connection
    
    def get_active_locks(self) -> List[Dict]:
        """获取当前活动的锁"""
        query = """
            SELECT 
                l.locktype,
                l.relation::regclass,
                l.mode,
                l.granted,
                l.pid,
                a.usename,
                a.application_name,
                a.client_addr,
                a.query_start,
                a.state,
                a.query
            FROM pg_locks l
            LEFT JOIN pg_stat_activity a ON l.pid = a.pid
            WHERE l.relation IS NOT NULL
            ORDER BY a.query_start
        """
        
        cursor = self.connection.cursor()
        cursor.execute(query)
        
        locks = []
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            locks.append(dict(zip(columns, row)))
        
        return locks
    
    def get_lock_waits(self) -> List[Dict]:
        """获取锁等待信息"""
        query = """
            SELECT 
                blocked.pid AS blocked_pid,
                blocked.usename AS blocked_user,
                blocking.pid AS blocking_pid,
                blocking.usename AS blocking_user,
                blocked.query AS blocked_query,
                blocking.query AS blocking_query,
                blocked.application_name AS blocked_app,
                blocking.application_name AS blocking_app
            FROM pg_stat_activity blocked
            JOIN pg_locks blocked_locks 
                ON blocked.pid = blocked_locks.pid 
                AND NOT blocked_locks.granted
            JOIN pg_locks blocking_locks 
                ON blocked_locks.locktype = blocking_locks.locktype
                AND blocked_locks.database IS NOT DISTINCT FROM blocking_locks.database
                AND blocked_locks.relation IS NOT DISTINCT FROM blocking_locks.relation
                AND blocked_locks.page IS NOT DISTINCT FROM blocking_locks.page
                AND blocked_locks.tuple IS NOT DISTINCT FROM blocking_locks.tuple
                AND blocked_locks.virtualxid IS NOT DISTINCT FROM blocking_locks.virtualxid
                AND blocked_locks.transactionid IS NOT DISTINCT FROM blocking_locks.transactionid
                AND blocked_locks.classid IS NOT DISTINCT FROM blocking_locks.classid
                AND blocked_locks.objid IS NOT DISTINCT FROM blocking_locks.objid
                AND blocked_locks.objsubid IS NOT DISTINCT FROM blocking_locks.objsubid
                AND blocked_locks.pid != blocking_locks.pid
            JOIN pg_stat_activity blocking 
                ON blocking.pid = blocking_locks.pid 
                AND blocking_locks.granted
        """
        
        cursor = self.connection.cursor()
        cursor.execute(query)
        
        waits = []
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            waits.append(dict(zip(columns, row)))
        
        return waits
    
    def get_transaction_locks(self, pid: int) -> List[Dict]:
        """获取特定事务持有的锁"""
        query = """
            SELECT 
                l.locktype,
                l.relation::regclass,
                l.mode,
                l.granted,
                l.pid,
                l.virtualxid,
                l.transactionid,
                l.classid,
                l.objid,
                l.objsubid
            FROM pg_locks l
            WHERE l.pid = %s
            ORDER BY l.granted DESC, l.relation
        """
        
        cursor = self.connection.cursor()
        cursor.execute(query, (pid,))
        
        locks = []
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            locks.append(dict(zip(columns, row)))
        
        return locks
    
    def analyze_lock_contention(self) -> Dict:
        """分析锁竞争情况"""
        active_locks = self.get_active_locks()
        lock_waits = self.get_lock_waits()
        
        # 统计锁类型分布
        lock_types = {}
        for lock in active_locks:
            lock_type = lock.get('locktype', 'unknown')
            lock_types[lock_type] = lock_types.get(lock_type, 0) + 1
        
        # 统计等待时长
        wait_times = []
        for wait in lock_waits:
            if wait.get('query_start') and isinstance(wait['query_start'], datetime):
                wait_time = (datetime.now() - wait['query_start']).total_seconds()
                wait_times.append(wait_time)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_active_locks': len(active_locks),
            'lock_type_distribution': lock_types,
            'total_waiting': len(lock_waits),
            'max_wait_time': max(wait_times) if wait_times else 0,
            'avg_wait_time': sum(wait_times) / len(wait_times) if wait_times else 0,
            'blocking_details': lock_waits
        }
```

### 3.2 死锁检测与解决

```python
import threading
import queue
from typing import Callable, Optional

class DeadlockDetector:
    """死锁检测器"""
    
    def __init__(self, connection):
        self.connection = connection
        self.monitoring = False
        self.monitor_thread = None
        self.alert_callback = None
    
    def detect_deadlock(self) -> Optional[Dict]:
        """检测当前是否存在死锁"""
        # PostgreSQL会自动检测死锁并回滚其中一个事务
        # 但我们可以通过检查pg_locks来识别潜在的死锁模式
        
        query = """
            SELECT 
                COUNT(DISTINCT l1.pid) AS distinct_pids,
                COUNT(l1.locktype) AS total_locks
            FROM pg_locks l1
            JOIN pg_locks l2 
                ON l1.locktype = l2.locktype
                AND l1.database IS NOT DISTINCT FROM l2.database
                AND l1.relation IS NOT DISTINCT FROM l2.relation
                AND l1.page IS NOT DISTINCT FROM l2.page
                AND l1.tuple IS NOT DISTINCT FROM l2.tuple
                AND l1.virtualxid IS NOT DISTINCT FROM l2.virtualxid
                AND l1.transactionid IS NOT DISTINCT FROM l2.transactionid
                AND l1.classid IS NOT DISTINCT FROM l2.classid
                AND l1.objid IS NOT DISTINCT FROM l2.objid
                AND l1.objsubid IS NOT DISTINCT FROM l2.objsubid
                AND l1.pid != l2.pid
            WHERE NOT l1.granted AND NOT l2.granted
        """
        
        cursor = self.connection.cursor()
        cursor.execute(query)
        row = cursor.fetchone()
        
        if row and row[0] > 0:
            return {
                'deadlock_detected': True,
                'involved_pids': row[0],
                'lock_count': row[1]
            }
        
        return {'deadlock_detected': False}
    
    def find_potential_deadlocks(self) -> List[Dict]:
        """查找潜在的死锁模式"""
        # 查找循环等待模式
        query = """
            WITH blocked_pids AS (
                SELECT DISTINCT pid FROM pg_locks WHERE NOT granted
            )
            SELECT 
                b1.pid AS pid1,
                b2.pid AS pid2,
                l1.relation AS table1,
                l2.relation AS table2
            FROM pg_locks l1
            JOIN pg_locks l2 
                ON l1.relation = l2.relation
                AND l1.pid != l2.pid
                AND l1.granted AND NOT l2.granted
            WHERE l1.pid IN (SELECT pid FROM blocked_pids)
                AND l2.pid IN (SELECT pid FROM blocked_pids)
        """
        
        cursor = self.connection.cursor()
        cursor.execute(query)
        
        patterns = []
        for row in cursor.fetchall():
            patterns.append({
                'process_1': row[0],
                'process_2': row[1],
                'contention_table_1': row[2],
                'contention_table_2': row[3]
            })
        
        return patterns
    
    def force_terminate_process(self, pid: int, reason: str = "Manual termination") -> bool:
        """强制终止进程"""
        try:
            cursor = self.connection.cursor()
            # pg_terminate_backend会发送SIGTERM信号
            cursor.execute("SELECT pg_terminate_backend(%s)", (pid,))
            result = cursor.fetchone()
            
            if result and result[0]:
                print(f"Process {pid} terminated: {reason}")
                return True
            
            return False
            
        except Exception as e:
            print(f"Failed to terminate process {pid}: {e}")
            return False
    
    def cancel_query(self, pid: int) -> bool:
        """取消查询（发送SIGINT）"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT pg_cancel_backend(%s)", (pid,))
            result = cursor.fetchone()
            
            if result and result[0]:
                print(f"Query for process {pid} cancelled")
                return True
            
            return False
            
        except Exception as e:
            print(f"Failed to cancel query for {pid}: {e}")
            return False
    
    def start_monitoring(self, interval: float = 5.0, 
                        callback: Optional[Callable] = None):
        """启动死锁监控"""
        self.monitoring = True
        self.alert_callback = callback
        
        def monitor():
            while self.monitoring:
                deadlock = self.detect_deadlock()
                if deadlock and deadlock.get('deadlock_detected'):
                    if self.alert_callback:
                        self.alert_callback(deadlock)
                
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
```

### 3.3 锁问题故障排除流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    锁问题故障排除流程                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   检查活动连接数    │
                    │  (pg_stat_activity) │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        连接数过多         连接正常          连接接近上限
              │                │                │
              ▼                ▼                ▼
        检查连接泄漏    检查pg_locks      增加max_connections
              │                │                │
              │                ▼                │
              │         ┌────────────┐          │
              │         │ 检查锁等待 │          │
              │         └─────┬──────┘          │
              │               │                 │
              │        ┌──────┼──────┐          │
              │        ▼             ▼          │
          优化连接池    有等待        无等待     │
              │        │             │          │
              │        ▼             ▼          │
          减少长事务    识别阻塞      查询正常   │
              │        │             │          │
              │        ▼             ▼          │
          增加锁超时    终止阻塞进程   继续观察   │
              │        │             │          │
              └────────┴─────────────┴──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    死锁检测         │
                    │  (自动检测+模式)    │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
         检测到死锁        潜在死锁         无死锁
              │                │                │
              ▼                ▼                ▼
        查看pg_locks    分析等待图        问题已解决
        终止进程           │                    │
              │            ▼                    │
              │    优化事务顺序                  │
              │    添加锁超时                    │
              └────────────────┘
```

---

## 4. 内存与资源调试

### 4.1 内存使用分析

```python
import psutil
import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class MemoryInfo:
    """内存信息"""
    total: int
    used: int
    free: int
    percent: float
    available: int

class DatabaseMemoryAnalyzer:
    """数据库内存分析器"""
    
    def __init__(self, connection):
        self.connection = connection
    
    def get_shared_memory_info(self) -> Dict:
        """获取共享内存信息（PostgreSQL）"""
        query = """
            SELECT 
                name, setting, unit, context, vartype, source
            FROM pg_settings
            WHERE name LIKE 'shared_%' 
                OR name LIKE 'work_mem' 
                OR name LIKE 'maintenance_work_mem'
                OR name LIKE 'effective_cache_size'
            ORDER BY name
        """
        
        cursor = self.connection.cursor()
        cursor.execute(query)
        
        settings = {}
        for row in cursor.fetchall():
            settings[row[0]] = {
                'value': row[1],
                'unit': row[2],
                'context': row[3],
                'type': row[4],
                'source': row[5]
            }
        
        return settings
    
    def get_buffer_cache_stats(self) -> Dict:
        """获取缓冲区缓存统计"""
        query = """
            SELECT 
                blks_hit,
                blks_read,
                CASE WHEN blks_read + blks_hit > 0 
                    THEN round(100.0 * blks_hit / (blks_read + blks_hit), 2)
                    ELSE 0 
                END AS cache_hit_ratio,
                tup_returned,
                tup_fetched,
                tup_inserted,
                tup_updated,
                tup_deleted
            FROM pg_stat_database
            WHERE datname = current_database()
        """
        
        cursor = self.connection.cursor()
        cursor.execute(query)
        row = cursor.fetchone()
        
        if row:
            return {
                'blocks_hit': row[0],
                'blocks_read': row[1],
                'cache_hit_ratio_percent': row[2],
                'tuples_returned': row[3],
                'tuples_fetched': row[4],
                'tuples_inserted': row[5],
                'tuples_updated': row[6],
                'tuples_deleted': row[7]
            }
        
        return {}
    
    def get_memory_by_table(self, limit: int = 20) -> List[Dict]:
        """获取各表占用的内存"""
        # 需要pgstattuple扩展
        query = """
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
                pg_total_relation_size(schemaname||'.'||tablename) AS size_bytes,
                n_live_tup,
                n_dead_tup,
                last_vacuum,
                last_autovacuum
            FROM pg_stat_user_tables
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            LIMIT %s
        """
        
        cursor = self.connection.cursor()
        cursor.execute(query, (limit,))
        
        tables = []
        for row in cursor.fetchall():
            tables.append({
                'schema': row[0],
                'table': row[1],
                'size': row[2],
                'size_bytes': row[3],
                'live_tuples': row[4],
                'dead_tuples': row[5],
                'last_vacuum': row[6],
                'last_autovacuum': row[7]
            })
        
        return tables
    
    def get_connection_memory_usage(self) -> List[Dict]:
        """获取各连接的内存使用"""
        query = """
            SELECT 
                pid,
                usename,
                application_name,
                state,
                query,
                memory_usage_bytes
            FROM pg_stat_activity
            WHERE pid != pg_backend_pid()
            ORDER BY memory_usage_bytes DESC NULLS LAST
        """
        
        # 注意：memory_usage_bytes需要pg_memusage扩展
        cursor = self.connection.cursor()
        cursor.execute(query)
        
        connections = []
        for row in cursor.fetchall():
            connections.append({
                'pid': row[0],
                'user': row[1],
                'application': row[2],
                'state': row[3],
                'query': row[4][:100] if row[4] else None,
                'memory_bytes': row[5]
            })
        
        return connections
    
    def analyze_memory_pressure(self) -> Dict:
        """分析内存压力情况"""
        # 系统内存
        vm = psutil.virtual_memory()
        
        # PostgreSQL缓存统计
        cache_stats = self.get_buffer_cache_stats()
        
        # 计算缓存命中率
        cache_hit_ratio = cache_stats.get('cache_hit_ratio_percent', 0)
        
        # 建议
        recommendations = []
        
        if vm.percent > 90:
            recommendations.append("系统内存压力高，考虑增加内存或优化查询")
        
        if cache_hit_ratio < 80:
            recommendations.append(
                f"数据库缓存命中率低({cache_hit_ratio}%)，建议增加shared_buffers"
            )
        
        return {
            'system_memory': {
                'total': vm.total,
                'used': vm.used,
                'free': vm.free,
                'percent': vm.percent
            },
            'cache_stats': cache_stats,
            'recommendations': recommendations
        }
```

### 4.2 资源限制配置检查

```python
class ResourceLimitChecker:
    """资源限制检查器"""
    
    def __init__(self, connection):
        self.connection = connection
    
    def check_connection_limits(self) -> Dict:
        """检查连接限制配置"""
        cursor = self.connection.cursor()
        
        # 获取配置
        cursor.execute("SHOW max_connections")
        max_conn = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT count(*) FROM pg_stat_activity
        """)
        current_conn = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT count(*) FROM pg_stat_activity 
            WHERE state = 'idle in transaction'
        """)
        idle_in_transaction = cursor.fetchone()[0]
        
        return {
            'max_connections': int(max_conn),
            'current_connections': current_conn,
            'idle_connections': idle_in_transaction,
            'idle_in_transaction': idle_in_transaction,
            'utilization_percent': (current_conn / int(max_conn)) * 100
        }
    
    def check_work_mem_limits(self) -> Dict:
        """检查工作内存配置"""
        cursor = self.connection.cursor()
        
        cursor.execute("SHOW work_mem")
        work_mem = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT 
                state,
                count(*),
                avg(used_memory) as avg_memory,
                max(used_memory) as max_memory
            FROM pg_stat_activity
            WHERE pid != pg_backend_pid()
            GROUP BY state
        """)
        
        states = {}
        for row in cursor.fetchall():
            states[row[0]] = {
                'count': row[1],
                'avg_memory': row[2],
                'max_memory': row[3]
            }
        
        return {
            'work_mem_setting': work_mem,
            'connection_states': states
        }
    
    def check_temp_file_usage(self) -> List[Dict]:
        """检查临时文件使用"""
        query = """
            SELECT 
                pid,
                usename,
                query,
                temp_file_bytes,
                temp_file_num
            FROM pg_stat_activity
            WHERE temp_file_bytes > 0
            ORDER BY temp_file_bytes DESC
            LIMIT 10
        """
        
        cursor = self.connection.cursor()
        cursor.execute(query)
        
        temp_usage = []
        for row in cursor.fetchall():
            temp_usage.append({
                'pid': row[0],
                'user': row[1],
                'query': row[2][:100] if row[2] else None,
                'temp_bytes': row[3],
                'temp_files': row[4]
            })
        
        return temp_usage
    
    def check_resource_consumers(self) -> Dict:
        """检查资源消耗者"""
        # 最耗资源的查询
        cursor = self.connection.cursor()
        
        # CPU密集查询
        cursor.execute("""
            SELECT 
                pid,
                query,
                state,
                total_time,
                calls
            FROM pg_stat_statements
            ORDER BY total_time DESC
            LIMIT 5
        """)
        
        top_cpu = []
        for row in cursor.fetchall():
            top_cpu.append({
                'pid': row[0],
                'query': row[1][:100],
                'state': row[2],
                'total_time_ms': row[3],
                'calls': row[4]
            })
        
        # IO密集查询
        cursor.execute("""
            SELECT 
                pid,
                query,
                state,
                blk_read_time,
                blk_write_time
            FROM pg_stat_statements
            ORDER BY blk_read_time + blk_write_time DESC
            LIMIT 5
        """)
        
        top_io = []
        for row in cursor.fetchall():
            top_io.append({
                'pid': row[0],
                'query': row[1][:100],
                'state': row[2],
                'read_time_ms': row[3],
                'write_time_ms': row[4]
            })
        
        return {
            'top_cpu_queries': top_cpu,
            'top_io_queries': top_io
        }
```

---

## 5. 网络与连接调试

### 5.1 连接池诊断

```python
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class ConnectionPoolStats:
    """连接池统计"""
    total_connections: int
    idle_connections: int
    active_connections: int
    waiting_threads: int
    avg_wait_time_ms: float
    max_wait_time_ms: float
    connection_errors: int

class ConnectionPoolDiagnoser:
    """连接池诊断器"""
    
    def __init__(self, pool):
        self.pool = pool
    
    def get_pool_stats(self) -> ConnectionPoolStats:
        """获取连接池统计"""
        return ConnectionPoolStats(
            total_connections=self.pool.size(),
            idle_connections=len(self.pool.idle_connections),
            active_connections=len(self.pool.connections),
            waiting_threads=len(self.pool.waiting),
            avg_wait_time_ms=self._calculate_avg_wait_time(),
            max_wait_time_ms=self._calculate_max_wait_time(),
            connection_errors=self.pool.errors
        )
    
    def diagnose_connection_leaks(self) -> Dict:
        """诊断连接泄漏"""
        current_time = time.time()
        potential_leaks = []
        
        for conn in self.pool.connections:
            if conn.in_use:
                # 检查连接使用时间
                usage_duration = current_time - conn.last_used
                
                if usage_duration > 300:  # 超过5分钟
                    potential_leaks.append({
                        'connection_id': id(conn),
                        'in_use_duration': usage_duration,
                        'created_at': conn.created_at,
                        'query': conn.last_query
                    })
        
        return {
            'potential_leaks': len(potential_leaks),
            'details': potential_leaks
        }
    
    def diagnose_timeout_issues(self) -> Dict:
        """诊断超时问题"""
        # 分析连接获取超时
        timeouts = self.pool.timeout_errors
        last_timeout = self.pool.last_timeout_time
        
        # 分析慢查询
        slow_connections = []
        for conn in self.pool.connections:
            if conn.in_use:
                query_time = time.time() - conn.query_start_time
                if query_time > 30:  # 超过30秒
                    slow_connections.append({
                        'connection_id': id(conn),
                        'query_duration': query_time,
                        'query': conn.current_query
                    })
        
        return {
            'total_timeouts': timeouts,
            'last_timeout': last_timeout,
            'slow_connections': slow_connections
        }
    
    def _calculate_avg_wait_time(self) -> float:
        """计算平均等待时间"""
        if not self.pool.wait_times:
            return 0.0
        return sum(self.pool.wait_times) / len(self.pool.wait_times)
    
    def _calculate_max_wait_time(self) -> float:
        """计算最大等待时间"""
        return max(self.pool.wait_times) if self.pool.wait_times else 0.0


class DatabaseNetworkDiagnoser:
    """数据库网络诊断器"""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
    
    def test_connection_latency(self, samples: int = 5) -> Dict:
        """测试连接延迟"""
        import socket
        
        latencies = []
        
        for _ in range(samples):
            start = time.time()
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                sock.connect((self.host, self.port))
                sock.close()
                latencies.append((time.time() - start) * 1000)  # 转换为毫秒
            except Exception:
                pass
        
        if latencies:
            return {
                'host': self.host,
                'port': self.port,
                'avg_latency_ms': sum(latencies) / len(latencies),
                'min_latency_ms': min(latencies),
                'max_latency_ms': max(latencies),
                'samples': samples,
                'successful': len(latencies)
            }
        
        return {
            'host': self.host,
            'port': self.port,
            'error': 'All connection attempts failed'
        }
    
    def check_ssl_connection(self) -> Dict:
        """检查SSL连接"""
        import ssl
        import socket
        
        result = {
            'ssl_available': False,
            'ssl_version': None,
            'cipher': None,
            'error': None
        }
        
        try:
            context = ssl.create_default_context()
            
            with socket.create_connection((self.host, self.port)) as sock:
                with context.wrap_socket(sock, server_hostname=self.host) as ssock:
                    result['ssl_available'] = True
                    result['ssl_version'] = ssock.version()
                    result['cipher'] = ssock.cipher()
                    
        except ssl.SSLError as e:
            result['error'] = f"SSL Error: {str(e)}"
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def diagnose_routing_issues(self) -> Dict:
        """诊断路由问题"""
        import subprocess
        
        result = {
            'host': self.host,
            'port': self.port,
            'traceroute': [],
            'dns_resolution': []
        }
        
        # DNS解析
        try:
            import socket
            result['dns_resolution'] = socket.getaddrinfo(
                self.host, self.port, socket.AF_UNSPEC, socket.SOCK_STREAM
            )
        except Exception as e:
            result['dns_error'] = str(e)
        
        # Traceroute (Windows用tracert)
        try:
            if os.name == 'nt':
                proc = subprocess.run(
                    ['tracert', '-d', '-w', '100', '-h', '15', self.host],
                    capture_output=True, text=True, timeout=30
                )
            else:
                proc = subprocess.run(
                    ['traceroute', '-n', '-w', '2', '-m', '15', self.host],
                    capture_output=True, text=True, timeout=30
                )
            result['traceroute'] = proc.stdout.split('\n')
        except Exception as e:
            result['traceroute_error'] = str(e)
        
        return result
```

### 5.2 连接问题故障排除流程

```
┌─────────────────────────────────────────────────────────────┐
│                  连接问题故障排除流程                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   验证网络连通性    │
                    │  (ping/telnet/curl) │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
           不通            可达              延迟高
              │                │                │
              ▼                ▼                ▼
        检查防火墙      检查端口         检查网络拥塞
        检查路由        验证认证         检查MTU设置
              │                │                │
              │                ▼                │
              │         ┌────────────┐          │
              │         │ 尝试连接   │          │
              │         │ (错误信息) │          │
              │         └─────┬──────┘          │
              │               │                 │
              │        ┌──────┼──────┐          │
              │        ▼             ▼          │
          修复网络    认证失败       成功       │
              │        │             │          │
              │        ▼             ▼          │
          检查认证信息   检查授权      继续观察   │
              │                         │
              ▼                         ▼
        检查权限配置              性能问题排查
              │
              ▼
        授权用户访问
```

---

## 6. 恢复程序

### 6.1 故障恢复策略

```python
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class BackupInfo:
    """备份信息"""
    backup_id: str
    backup_type: str  # full, incremental
    start_time: datetime
    end_time: Optional[datetime]
    size_bytes: int
    status: str  # running, completed, failed
    location: str
    wal_files: List[str]

class DatabaseRecoveryManager:
    """数据库恢复管理器"""
    
    def __init__(self, connection, data_dir: str):
        self.connection = connection
        self.data_dir = data_dir
    
    def create_point_in_time_recovery_config(self, 
                                            target_time: datetime,
                                            wal_archive: str) -> Dict:
        """创建PITR恢复配置"""
        recovery_conf = {
            'restore_command': f'cp {wal_archive}/%f %p',
            'recovery_target_time': target_time.isoformat(),
            'recovery_target_action': 'promote'
        }
        
        return recovery_conf
    
    def perform_point_in_time_recovery(self,
                                       backup_location: str,
                                       target_time: datetime,
                                       target_dir: str) -> Dict:
        """执行时间点恢复"""
        result = {
            'success': False,
            'steps': [],
            'errors': []
        }
        
        try:
            # 1. 停止数据库
            result['steps'].append('Stopping database...')
            self._stop_database()
            
            # 2. 备份当前数据目录
            result['steps'].append('Backing up current data directory...')
            backup_dir = f"{target_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(self.data_dir, backup_dir)
            
            # 3. 清理数据目录
            result['steps'].append('Cleaning data directory...')
            shutil.rmtree(self.data_dir)
            os.makedirs(self.data_dir)
            
            # 4. 恢复基础备份
            result['steps'].append('Restoring base backup...')
            self._restore_backup(backup_location, self.data_dir)
            
            # 5. 配置恢复
            result['steps'].append('Configuring recovery...')
            self._configure_recovery(target_time)
            
            # 6. 启动数据库
            result['steps'].append('Starting database in recovery mode...')
            self._start_database_recovery()
            
            result['success'] = True
            
        except Exception as e:
            result['errors'].append(str(e))
        
        return result
    
    def verify_recovery(self) -> Dict:
        """验证恢复结果"""
        verification = {
            'database_reachable': False,
            'data_integrity': False,
            'consistency_check': False,
            'warnings': []
        }
        
        try:
            # 1. 检查数据库可达性
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            verification['database_reachable'] = True
            
            # 2. 检查数据完整性
            cursor.execute("""
                SELECT 
                    sum(pg_relation_size(oid)) as total_size
                FROM pg_class
                WHERE relkind = 'r'
            """)
            total_size = cursor.fetchone()[0]
            verification['data_integrity'] = (total_size > 0)
            
            # 3. 一致性检查
            cursor.execute("""
                SELECT count(*) FROM pg_database WHERE datistemplate = false
            """)
            db_count = cursor.fetchone()[0]
            verification['consistency_check'] = (db_count > 0)
            
        except Exception as e:
            verification['warnings'].append(str(e))
        
        return verification
    
    def _stop_database(self):
        """停止数据库"""
        # 实现数据库停止逻辑
        pass
    
    def _start_database_recovery(self):
        """启动数据库恢复"""
        pass
    
    def _restore_backup(self, source: str, destination: str):
        """恢复备份"""
        pass
    
    def _configure_recovery(self, target_time: datetime):
        """配置恢复"""
        pass
```

### 6.2 数据损坏恢复

```python
class CorruptionRecovery:
    """数据损坏恢复工具"""
    
    def __init__(self, connection):
        self.connection = connection
    
    def detect_index_corruption(self) -> List[Dict]:
        """检测索引损坏"""
        query = """
            SELECT 
                schemaname,
                tablename,
                indexname,
                idx_scan,
                idx_tup_read,
                idx_tup_fetch,
                pg_size_pretty(pg_relation_size(indexrelid)) AS size
            FROM pg_stat_user_indexes
            WHERE idx_scan > 0
            ORDER BY idx_scan ASC
        """
        
        cursor = self.connection.cursor()
        cursor.execute(query)
        
        indexes = []
        for row in cursor.fetchall():
            indexes.append({
                'schema': row[0],
                'table': row[1],
                'index': row[2],
                'scans': row[3],
                'tuples_read': row[4],
                'tuples_fetched': row[5],
                'size': row[6]
            })
        
        return indexes
    
    def detect_table_corruption(self) -> List[Dict]:
        """检测表损坏"""
        # 使用pgstattuple检查表
        query = """
            SELECT 
                schemaname,
                tablename,
                dead_tuple_count,
                dead_tuple_len,
                free_space,
                fragment_ratio
            FROM pgstattuple('pg_catalog.pg_class')
            WHERE relkind = 'r'
        """
        
        # 或者使用REINDEX检查
        check_query = """
            SELECT 
                schemaname,
                tablename,
                indexname,
                'needs_reindex' AS status
            FROM pg_stat_user_indexes
            WHERE idx_scan = 0
            AND indexrelid IN (
                SELECT conindid FROM pg_constraint 
                WHERE contype = 'p'
            )
        """
        
        cursor = self.connection.cursor()
        cursor.execute(check_query)
        
        needs_reindex = []
        for row in cursor.fetchall():
            needs_reindex.append({
                'schema': row[0],
                'table': row[1],
                'index': row[2],
                'status': row[3]
            })
        
        return needs_reindex
    
    def repair_index(self, index_name: str) -> Dict:
        """修复索引"""
        result = {
            'index': index_name,
            'success': False,
            'error': None
        }
        
        try:
            cursor = self.connection.cursor()
            
            # 重建索引
            cursor.execute(f"REINDEX INDEX {index_name}")
            self.connection.commit()
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def vacuum_full_table(self, table_name: str) -> Dict:
        """对表执行VACUUM FULL"""
        result = {
            'table': table_name,
            'success': False,
            'before_size': 0,
            'after_size': 0,
            'error': None
        }
        
        try:
            cursor = self.connection.cursor()
            
            # 获取执行前的大小
            cursor.execute(f"""
                SELECT pg_total_relation_size('{table_name}')
            """)
            result['before_size'] = cursor.fetchone()[0]
            
            # 执行VACUUM FULL
            cursor.execute(f"VACUUM FULL {table_name}")
            self.connection.commit()
            
            # 获取执行后的大小
            cursor.execute(f"""
                SELECT pg_total_relation_size('{table_name}')
            """)
            result['after_size'] = cursor.fetchone()[0]
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def recover_from_backup(self, table_name: str, backup_file: str) -> Dict:
        """从备份恢复表"""
        result = {
            'table': table_name,
            'success': False,
            'rows_restored': 0,
            'error': None
        }
        
        try:
            cursor = self.connection.cursor()
            
            # 1. 重命名损坏的表
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            cursor.execute(f"""
                ALTER TABLE {table_name} RENAME TO {table_name}_corrupted_{timestamp}
            """)
            
            # 2. 从备份恢复
            with open(backup_file, 'r') as f:
                cursor.execute(f.read())
            
            # 3. 验证恢复
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            result['rows_restored'] = cursor.fetchone()[0]
            
            self.connection.commit()
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
```

### 6.3 事务故障恢复

```python
class TransactionRecovery:
    """事务故障恢复"""
    
    def __init__(self, connection):
        self.connection = connection
    
    def get_stale_transactions(self) -> List[Dict]:
        """获取僵死事务"""
        query = """
            SELECT 
                pid,
                xid,
                state,
                usename,
                application_name,
                client_addr,
                query_start,
                state_change,
                wait_event_type,
                wait_event
            FROM pg_stat_activity
            WHERE state != 'idle'
                AND xid IS NOT NULL
                AND query_start < now() - interval '10 minutes'
            ORDER BY query_start
        """
        
        cursor = self.connection.cursor()
        cursor.execute(query)
        
        stale = []
        for row in cursor.fetchall():
            stale.append({
                'pid': row[0],
                'transaction_id': row[1],
                'state': row[2],
                'user': row[3],
                'application': row[4],
                'client': row[5],
                'started_at': row[6],
                'state_changed': row[7],
                'wait_event': row[9]
            })
        
        return stale
    
    def force_terminate_stale_transaction(self, pid: int) -> Dict:
        """强制终止僵死事务"""
        result = {
            'pid': pid,
            'success': False,
            'error': None
        }
        
        try:
            cursor = self.connection.cursor()
            
            # 首先尝试优雅取消
            cursor.execute("SELECT pg_cancel_backend(%s)", (pid,))
            
            # 等待一小段时间让取消生效
            import time
            time.sleep(2)
            
            # 检查是否成功取消
            cursor.execute("""
                SELECT state FROM pg_stat_activity WHERE pid = %s
            """, (pid,))
            state = cursor.fetchone()
            
            if state and state[0] is None:
                # 进程已结束
                result['success'] = True
            else:
                # 需要强制终止
                cursor.execute("SELECT pg_terminate_backend(%s)", (pid,))
                result['success'] = True
                
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def recover_from_xid_exhaustion(self) -> Dict:
        """从XID耗尽中恢复"""
        result = {
            'success': False,
            'actions': []
        }
        
        try:
            cursor = self.connection.cursor()
            
            # 1. 检查当前XID使用情况
            cursor.execute("""
                SELECT 
                    datname,
                    age(datfrozenxid) as age,
                    pg_size_pretty(pg_database_size(datname)) as size
                FROM pg_database
                WHERE datistemplate = false
            """)
            
            databases = []
            for row in cursor.fetchall():
                databases.append({
                    'name': row[0],
                    'age': row[1],
                    'size': row[2]
                })
            
            result['databases'] = databases
            
            # 2. 对旧数据库执行VACUUM
            for db in databases:
                if db['age'] > 2000000000:  # XID危险水平
                    result['actions'].append(
                        f"VACUUM FREEZE {db['name']}"
                    )
                    cursor.execute(f"VACUUM FREEZE {db['name']}")
            
            # 3. 如果需要，执行VACUUM FULL
            oldest = min(databases, key=lambda x: x['age'])
            if oldest['age'] > 2100000000:
                result['actions'].append(
                    f"VACUUM FULL {oldest['name']}"
                )
                cursor.execute(f"VACUUM FULL {oldest['name']}")
            
            self.connection.commit()
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
```

---

## 附录：常用调试命令

### PostgreSQL调试命令

```sql
-- 查看当前活动
SELECT * FROM pg_stat_activity;

-- 查看锁
SELECT * FROM pg_locks WHERE NOT granted;

-- 查看慢查询
SELECT pid, now() - query_start AS duration, query 
FROM pg_stat_activity 
WHERE state != 'idle' 
ORDER BY duration DESC;

-- 查看表大小
SELECT relname, pg_size_pretty(pg_total_relation_size(relid)) 
FROM pg_catalog.pg_class 
WHERE relkind = 'r' 
ORDER BY pg_total_relation_size(relid) DESC;

-- 查看索引使用
SELECT indexrelname, idx_scan 
FROM pg_stat_user_indexes 
ORDER BY idx_scan DESC;

-- 查看缓存命中率
SELECT 
    sum(blks_hit) * 100.0 / nullif(sum(blks_hit + blks_read), 0) as ratio
FROM pg_stat_database;

-- 查看WAL使用
SELECT * FROM pgwal_stats;

-- 查看复制状态
SELECT * FROM pg_stat_replication;
```

### MySQL调试命令

```sql
-- 查看当前进程
SHOW PROCESSLIST;

-- 查看 InnoDB 状态
SHOW ENGINE INNODB STATUS;

-- 查看慢查询
SHOW VARIABLES LIKE 'slow_query_log';
SELECT * FROM mysql.slow_log;

-- 查看查询缓存
SHOW STATUS LIKE 'Qcache%';

-- 查看表状态
SHOW TABLE STATUS FROM database_name;

-- 查看索引
SHOW INDEX FROM table_name;

-- 查看锁
SELECT * FROM information_schema.INNODB_LOCKS;
SELECT * FROM information_schema.INNODB_TRX;
```

---

*本文档提供数据库调试的完整指南，从基础诊断到高级恢复操作。建议按照问题类型选择合适的章节进行参考。*
