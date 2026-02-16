# Database Threat Modeling

## Table of Contents

1. [Introduction to Database Threat Modeling](#introduction-to-database-threat-modeling)
2. [Common Database Threats and Vulnerabilities](#common-database-threats-and-vulnerabilities)
3. [Attack Vectors and Mitigation Strategies](#attack-vectors-and-mitigation-strategies)
4. [Zero-Trust Database Architecture](#zero-trust-database-architecture)
5. [Penetration Testing for Databases](#penetration-testing-for-databases)
6. [Incident Response Procedures](#incident-response-procedures)
7. [Security Automation and Tooling](#security-automation-and-tooling)

---

## Introduction to Database Threat Modeling

Threat modeling is a systematic approach to identifying, quantifying, and addressing security risks in database systems. This process enables security teams to understand potential attack paths, prioritize mitigation efforts, and allocate resources effectively. A comprehensive threat model considers the database infrastructure, application integrations, data flows, and user interactions.

The threat modeling process typically involves several key activities. First, the database system is decomposed into components including the database engine, storage systems, network interfaces, and application integrations. Second, threat agents are identified, encompassing external attackers, malicious insiders, compromised applications, and accidental data exposure. Third, attack vectors are mapped, documenting how threats could exploit vulnerabilities to achieve their objectives. Fourth, vulnerabilities are assessed based on likelihood and potential impact. Finally, mitigation strategies are developed and prioritized.

Database threat modeling should be performed during system design, after significant architecture changes, and periodically for existing systems. The STRIDE methodology provides a useful framework for categorizing threats: Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, and Elevation of Privilege.

---

## Common Database Threats and Vulnerabilities

### Injection Attacks

SQL injection remains one of the most prevalent and dangerous database vulnerabilities. Attackers exploit insufficient input validation to inject malicious SQL code that executes within the database context. Successful injection attacks can bypass authentication, access unauthorized data, modify or delete data, execute system commands, or compromise the underlying server.

Injection vulnerabilities arise when applications concatenate user input directly into SQL queries rather than using parameterized statements. Even ORMs can introduce vulnerabilities if used incorrectly. Second-order injection attacks store malicious input in the database for later execution when different code paths process the data.

```python
# VULNERABLE: String concatenation in SQL queries
def get_user_by_username(username):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()

# SECURE: Parameterized query
def get_user_by_username_secure(username):
    query = "SELECT * FROM users WHERE username = %s"
    cursor.execute(query, (username,))
    return cursor.fetchone()

# Attack example:
# Input: ' OR '1'='1
# Resulting query: SELECT * FROM users WHERE username = '' OR '1'='1'
# This returns all users!
```

### Authentication and Authorization Bypass

Weak authentication mechanisms enable unauthorized access to databases through credential theft, brute force attacks, or authentication bypass. Default credentials, weak password policies, and missing multi-factor authentication create entry points for attackers. Once authenticated, privilege escalation exploits allow attackers to gain higher access levels than originally granted.

```sql
-- PostgreSQL: Check for common authentication vulnerabilities

-- Check for users with default or weak passwords
SELECT rolname, rolvaliduntil
FROM pg_authid
WHERE rolpassword = 'md5' || md5(rolname || rolpassword)
   OR rolpassword LIKE 'password%'
   OR rolpassword LIKE '123%';

-- Check for users that never expire
SELECT rolname, rolvaliduntil
FROM pg_authid
WHERE rolvaliduntil IS NULL
  AND rolcanlogin = true;

-- Check for superuser accounts
SELECT rolname, rolsuper
FROM pg_roles
WHERE rolsuper = true;

-- Check for publicly accessible tables
SELECT schemaname, tablename, tableowner
FROM pg_tables
WHERE schemaname = 'public'
  AND tableowner != 'postgres';
```

### Data Exfiltration

Data exfiltration involves unauthorized transfer of data from a database. Attackers may exploit vulnerabilities to extract sensitive information, use legitimate access channels to access unauthorized data, or leverage compromised application accounts. Large-scale exfiltration can occur through query-based attacks or database replication features.

```sql
-- PostgreSQL: Monitor for data exfiltration patterns

-- Detect unusually large SELECT queries
SELECT 
    datname,
    pid,
    usename,
    application_name,
    query,
    bytes / 1024 / 1024 AS mb_returned,
    duration
FROM pg_stat_activity
WHERE state = 'active'
  AND query LIKE 'SELECT%'
  AND bytes > 100000000;  -- Over 100MB

-- Detect mass data export attempts
SELECT 
    usename,
    COUNT(*) AS query_count,
    SUM(rows) AS total_rows,
    MAX(query_start) AS last_query
FROM pg_stat_statements s
JOIN pg_user u ON s.userid = u.usesysid
WHERE s.query LIKE 'SELECT%'
  AND s.calls > 1000
GROUP BY usename
ORDER BY total_rows DESC;

-- Monitor network connections for data transfer
SELECT 
    client_addr,
    COUNT(*) AS connections,
    SUM(pg_stat_activity.query::text::bytea) / 1024 / 1024 AS mb_transferred
FROM pg_stat_activity
WHERE state = 'active'
GROUP BY client_addr
HAVING SUM(pg_stat_activity.query::text::bytea) > 100000000;
```

### Denial of Service

Denial of service attacks render databases unavailable to legitimate users. Attackers may exploit resource-intensive queries, exhaust connections, trigger locks, or exploit software vulnerabilities to crash the database. Even legitimate operations can cause DoS if not properly controlled through connection limits, query timeouts, and resource quotas.

```sql
-- PostgreSQL: Detect and prevent DoS patterns

-- Identify long-running queries
SELECT 
    pid,
    usename,
    application_name,
    state,
    query,
    duration,
    wait_event_type
FROM pg_stat_activity
WHERE state != 'idle'
  AND query_start < NOW() - INTERVAL '5 minutes'
ORDER BY duration DESC;

-- Check for query floods from single source
SELECT 
    client_addr,
    COUNT(*) AS queries_per_minute
FROM pg_stat_activity
WHERE query_start > NOW() - INTERVAL '1 minute'
GROUP BY client_addr
HAVING COUNT(*) > 1000;

-- Monitor for lock contention
SELECT 
    d.datname,
    l.relation::regclass,
    l.transactionid,
    l.mode,
    l.granted,
    l.pid,
    a.usename,
    a.query
FROM pg_locks l
JOIN pg_database d ON l.database = d.oid
LEFT JOIN pg_stat_activity a ON l.pid = a.pid
WHERE NOT l.granted
ORDER BY l.pid;
```

### Insider Threats

Insider threats originate from within the organization, including malicious employees, compromised internal accounts, and careless users. Insiders have legitimate access to systems and data, making detection challenging. They may steal data for financial gain, sell access to external parties, or accidentally expose sensitive information.

```sql
-- PostgreSQL: Detect insider threat patterns

-- Monitor access to sensitive tables outside business hours
SELECT 
    usename,
    tablename,
    call_count,
    total_time
FROM pg_stat_statements s
JOIN pg_user u ON s.userid = u.usesysid
WHERE s.query LIKE '%sensitive_table%'
  AND s.query_start < current_date
  AND extract(hour FROM s.query_start) NOT BETWEEN 9 AND 17;

-- Detect privilege escalation attempts
SELECT 
    username,
    admin_access,
    access_time,
    source_ip
FROM audit_log
WHERE event_type = 'role_change'
  AND new_value LIKE '%superuser%';

-- Monitor for data downloads by administrators
SELECT 
    admin_username,
    table_accessed,
    records_exported,
    export_time,
    destination
FROM admin_audit
WHERE action = 'data_export'
  AND export_time > NOW() - INTERVAL '24 hours';
```

---

## Attack Vectors and Mitigation Strategies

### Attack Vector Analysis

Understanding attack vectors enables targeted defensive investments. Each vector represents a path through which threats can reach database assets. This section documents common attack vectors and specific mitigations for database systems.

| Attack Vector | Description | Primary Mitigation |
|---------------|-------------|-------------------|
| Web Application | SQL injection through web interfaces | Input validation, parameterized queries |
| Direct Network | Direct database connection attacks | Network segmentation, firewall rules |
| Backup Theft | Stolen database backups | Encryption at rest, secure storage |
| Privileged User | Malicious admin access | Separation of duties, audit logging |
| Application Layer | Compromised application server | Application hardening, API security |
| Supply Chain | Compromised dependencies | Dependency scanning, code signing |
| Social Engineering | Phishing for credentials | Training, MFA |
| Zero-Day | Unknown vulnerabilities | Defense in depth, rapid patching |

### Web Application Attack Prevention

Web applications provide the most common attack surface for database compromise. SQL injection attacks through web interfaces can bypass authentication, extract data, or modify database contents. Defenses must be implemented at multiple layers.

```python
# Secure database access implementation

import psycopg2
from psycopg2 import sql
from typing import Any, List, Tuple

class SecureDatabaseConnection:
    """
    Secure database connection with injection prevention
    and comprehensive security controls.
    """
    
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.conn = None
    
    def connect(self):
        """Establish encrypted connection."""
        self.conn = psycopg2.connect(
            self.dsn,
            sslmode='require',  # Enforce TLS
            sslrootcert='ca.pem',
            sslcert='client-cert.pem',
            sslkey='client-key.pem'
        )
        return self.conn
    
    def execute_query(self, query: str, params: tuple = None) -> List[Tuple]:
        """
        Execute query with parameterization to prevent injection.
        Never use string formatting or f-strings with user input.
        """
        cursor = self.conn.cursor()
        
        # Validate query type - only allow SELECT
        query_type = query.strip().split()[0].upper()
        if query_type not in ('SELECT', 'WITH'):
            raise ValueError("Only read queries allowed")
        
        # Use parameterized queries exclusively
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        return cursor.fetchall()
    
    def execute_safe_update(self, table: str, data: dict, where: dict) -> int:
        """
        Safe UPDATE using SQL composition with whitelist.
        """
        # Whitelist allowed tables
        allowed_tables = {'users', 'orders', 'products', 'customers'}
        if table not in allowed_tables:
            raise ValueError(f"Table {table} not allowed")
        
        # Build query using SQL composition
        set_clause = ', '.join(
            sql.Identifier(k).as_string() + ' = %s' 
            for k in data.keys()
        )
        where_clause = ' AND '.join(
            sql.Identifier(k).as_string() + ' = %s'
            for k in where.keys()
        )
        
        query = sql.SQL("UPDATE {} SET {} WHERE {}").format(
            sql.Identifier(table),
            sql.SQL(set_clause),
            sql.SQL(where_clause)
        )
        
        cursor = self.conn.cursor()
        cursor.execute(query, list(data.values()) + list(where.values()))
        
        return cursor.rowcount

# Usage with proper error handling
def get_user_safe(user_id):
    try:
        db = SecureDatabaseConnection("postgresql://user:pass@host/db")
        db.connect()
        
        # This is SAFE - parameterized query
        result = db.execute_query(
            "SELECT id, name, email FROM users WHERE id = %s",
            (user_id,)
        )
        
        return result[0] if result else None
    
    except psycopg2.Error as e:
        # Log error securely without exposing details
        logging.error(f"Database error: {type(e).__name__}")
        raise ValueError("Database operation failed")
```

### Network-Based Attack Prevention

Network attacks target database network interfaces, including connection interception, man-in-the-middle attacks, and direct port scanning. Network-level defenses provide the first line of defense against external attackers.

```yaml
# Kubernetes NetworkPolicy for database protection

apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: database-network-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: postgres
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow from application tier only
    - from:
        - podSelector:
            matchLabels:
              tier: application
      ports:
        - protocol: TCP
          port: 5432
    # Allow from monitoring
    - from:
        - podSelector:
            matchLabels:
              app: prometheus
      ports:
        - protocol: TCP
          port: 9187
    # Deny all other ingress
  egress:
    # Allow DNS
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
    # Allow internal communication
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - protocol: TCP
          port: 5432
    # Deny all other egress
```

### Privilege Escalation Prevention

Privilege escalation attacks exploit database features or vulnerabilities to gain higher access levels than originally granted. Preventing escalation requires implementing least privilege, monitoring privilege changes, and securing administrative functions.

```sql
-- PostgreSQL: Implement defense against privilege escalation

-- 1. Separate administrative functions
CREATE ROLE db_admin WITH LOGIN;
CREATE ROLE security_admin WITH LOGIN;
CREATE ROLE audit_reader WITH LOGIN;

-- 2. Grant specific privileges, not superuser
GRANT pg_read_all_settings TO security_admin;
GRANT pg_read_all_settings TO audit_reader;
GRANT pg_execute_server_program TO db_admin;

-- 3. Create custom roles with limited capabilities
CREATE ROLE readonly_analytics;
GRANT CONNECT ON DATABASE analytics_db TO readonly_analytics;
GRANT USAGE ON SCHEMA analytics TO readonly_analytics;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO readonly_analytics;

-- 4. Monitor privilege changes
CREATE TABLE privilege_change_log (
    log_id BIGSERIAL PRIMARY KEY,
    change_time TIMESTAMPTZ DEFAULT NOW(),
    changed_by VARCHAR(100),
    target_role VARCHAR(100),
    privilege_added VARCHAR(100),
    privilege_removed VARCHAR(100),
    reason TEXT
);

-- 5. Create trigger to log privilege changes
CREATE OR REPLACE FUNCTION log_privilege_changes()
RETURNS EVENT TRIGGER AS $$
BEGIN
    IF TG_EVENT = 'GRANT' THEN
        INSERT INTO privilege_change_log (
            changed_by, target_role, privilege_added
        ) VALUES (
            current_user,
            (TG_ARGV[0]),
            (TG_ARGV[1])
        );
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- 6. Create alerting for suspicious role changes
CREATE OR REPLACE FUNCTION check_privilege_escalation()
RETURNS TABLE (suspicious_activity TEXT) AS $$
BEGIN
    RETURN QUERY
    SELECT 'Superuser role granted to: ' || rolname::TEXT AS suspicious_activity
    FROM pg_roles
    WHERE rolsuper = true
      AND rolname NOT IN ('postgres', 'system_admin');
END;
$$ LANGUAGE plpgsql;
```

---

## Zero-Trust Database Architecture

### Zero-Trust Principles

Zero-trust architecture operates on the principle of "never trust, always verify." Every access request must be authenticated, authorized, and encrypted regardless of source location. Traditional perimeter-based security assumes internal networks are trustworthy, but modern architectures recognize that threats can originate from anywhere, including inside the network.

Zero-trust database implementations enforce several key principles. First, all connections must authenticate and authorize using strong mechanisms. Second, access is granted based on the principle of least privilege. Third, all communications are encrypted. Fourth, continuous monitoring detects anomalies and enforces policies. Fifth, security policies are dynamically adjusted based on risk signals.

### Implementing Zero-Trust Database Access

```yaml
# Zero-trust database proxy configuration

services:
  db-proxy:
    image: pgbouncer:latest
    environment:
      # Authentication
      AUTH_TYPE: scram-sha-256
      AUTH_FILE: /etc/pgbouncer/userlist
      
      # TLS enforcement
      client_tls_sslmode: require
      server_tls_sslmode: require
      client_tls_ciphers: HIGH:!aNULL:!MD5:!RC4
      server_tls_ciphers: HIGH:!aNULL:!MD5:!RC4
      
      # Connection management
      POOL_MODE: transaction
      DEFAULT_POOL_SIZE: 25
      MIN_POOL_SIZE: 10
      MAX_CLIENT_CONN: 1000
      
      # Timeouts
      SERVER_IDLE_TIMEOUT: 60s
      SERVER_CONNECT_TIMEOUT: 10s
      SERVER_LOGIN_RETRY: 3
      
      # Logging and audit
      LOG_CONNECTIONS: 1
      LOG_DISCONNECTIONS: 1
      LOG_POOLER_ERRORS: 1
```

### Microsegmentation for Databases

Microsegmentation divides the database infrastructure into isolated segments, each with its own security controls. This limits lateral movement if an attacker compromises a single segment and provides fine-grained control over data access.

```sql
-- PostgreSQL: Implement microsegmentation with separate schemas

-- Create segmented schemas for different data classifications
CREATE SCHEMA restricted;     -- Highly sensitive data
CREATE SCHEMA confidential;   -- Internal confidential data
CREATE SCHEMA internal;       -- Internal business data
CREATE SCHEMA public;          -- Publicly available data

-- Assign ownership
ALTER SCHEMA restricted OWNER TO data_classification_officer;
ALTER SCHEMA confidential OWNER TO security_officer;
ALTER SCHEMA internal OWNER TO department_lead;
ALTER SCHEMA public OWNER TO content_manager;

-- Create role hierarchy
CREATE ROLE classification_manager;
CREATE ROLE restricted_user;
CREATE ROLE confidential_user;
CREATE ROLE internal_user;
CREATE ROLE public_user;

-- Grant schema access based on classification level
GRANT USAGE ON SCHEMA restricted TO restricted_user;
GRANT ALL ON SCHEMA restricted TO classification_manager;

GRANT USAGE ON SCHEMA confidential TO confidential_user;
GRANT ALL ON SCHEMA confidential TO classification_manager, security_officer;

GRANT USAGE ON SCHEMA internal TO internal_user;
GRANT ALL ON SCHEMA internal TO classification_manager, department_lead;

GRANT USAGE ON SCHEMA public TO public_user;
GRANT ALL ON SCHEMA public TO classification_manager, content_manager;

-- Create cross-schema access policies
CREATE POLICY restrict_cross_classification
ON restricted.sensitive_table
FOR ALL
USING (
    current_user IN ('classification_manager', 'restricted_user')
);
```

### Continuous Authentication and Authorization

Zero-trust requires continuous verification rather than one-time authentication. Database access should be re-evaluated based on context changes, including user behavior, device status, location, and access patterns.

```python
# Continuous authorization implementation

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import jwt

@dataclass
class AccessContext:
    """Context for continuous authorization decisions."""
    user_id: str
    session_id: str
    ip_address: str
    user_agent: str
    device_fingerprint: str
    location: str
    access_time: datetime
    risk_score: float = 0.0

class ContinuousAuthZ:
    """
    Continuous authorization engine for zero-trust database access.
    Evaluates access requests against dynamic policies.
    """
    
    def __init__(self, policy_engine, risk_engine):
        self.policy_engine = policy_engine
        self.risk_engine = risk_engine
    
    def evaluate_access(
        self,
        context: AccessContext,
        requested_resource: str,
        operation: str
    ) -> bool:
        """
        Evaluate access request against policies.
        Returns True if access should be granted.
        """
        
        # Check basic authentication
        if not self._verify_authentication(context):
            return False
        
        # Calculate dynamic risk score
        risk_score = self.risk_engine.calculate_risk(context)
        context.risk_score = risk_score
        
        # Evaluate against policies
        policies = self.policy_engine.get_policies(
            context.user_id,
            requested_resource,
            operation
        )
        
        for policy in policies:
            if not self._evaluate_policy(context, policy):
                return False
        
        # Additional verification for high-risk requests
        if risk_score > 0.7:
            return self._require_additional_verification(context)
        
        return True
    
    def _verify_authentication(self, context: AccessContext) -> bool:
        """Verify user authentication is still valid."""
        # Check session validity, token expiration, etc.
        return True
    
    def _evaluate_policy(self, context: AccessContext, policy) -> bool:
        """Evaluate single policy against context."""
        # Policy evaluation logic
        return True
    
    def _require_additional_verification(
        self, 
        context: AccessContext
    ) -> bool:
        """Request additional verification for high-risk access."""
        # Could trigger MFA, security questions, etc.
        pass
```

---

## Penetration Testing for Databases

### Database Penetration Testing Methodology

Penetration testing simulates real attacks to identify vulnerabilities before malicious actors can exploit them. Database penetration tests follow a structured methodology including reconnaissance, vulnerability identification, exploitation, and reporting.

The reconnaissance phase gathers information about the database infrastructure, including version numbers, configurations, user accounts, and network accessibility. Vulnerability identification uses automated tools and manual techniques to discover security weaknesses. Exploitation attempts to leverage vulnerabilities to achieve unauthorized access or data extraction. Reporting documents findings, risk ratings, and remediation recommendations.

### SQL Injection Testing

```python
# SQL injection testing toolkit

import requests
from typing import List, Dict
import time

class SQLInjectionTester:
    """
    Automated SQL injection testing for database-backed applications.
    Use only with authorization!
    """
    
    def __init__(self, target_url: str):
        self.target_url = target_url
        self.findings = []
        
        # SQL injection payloads
        self.payloads = [
            "' OR '1'='1",
            "' OR '1'='1' --",
            "' OR '1'='1' /*",
            "admin' --",
            "admin' #",
            "' UNION SELECT NULL--",
            "' UNION SELECT NULL,NULL--",
            "1' AND '1'='1",
            "1' AND '1'='2",
            "'; DROP TABLE users;--",
        ]
    
    def test_parameter(self, param_name: str, test_values: List[str]) -> Dict:
        """
        Test a single parameter for SQL injection.
        """
        results = {
            'parameter': param_name,
            'vulnerable': False,
            'findings': []
        }
        
        for payload in test_values:
            try:
                response = self._send_request({param_name: payload})
                
                # Check for SQL error indicators
                if self._detect_sql_error(response):
                    results['vulnerable'] = True
                    results['findings'].append({
                        'payload': payload,
                        'error_detected': True,
                        'evidence': self._extract_error(response)
                    })
                
                # Check for boolean-based blind injection
                if self._detect_blind_injection(response):
                    results['vulnerable'] = True
                    results['findings'].append({
                        'payload': payload,
                        'blind_injection': True
                    })
            
            except Exception as e:
                # Connection errors might indicate successful injection
                if 'timeout' in str(e).lower():
                    results['vulnerable'] = True
                    results['findings'].append({
                        'payload': payload,
                        'dos_detected': True
                    })
        
        if results['vulnerable']:
            self.findings.append(results)
        
        return results
    
    def _send_request(self, data: dict) -> requests.Response:
        """Send test request to target."""
        return requests.post(
            self.target_url,
            data=data,
            timeout=10
        )
    
    def _detect_sql_error(self, response: requests.Response) -> bool:
        """Detect SQL error messages in response."""
        error_patterns = [
            'SQL syntax',
            'MySQL',
            'PostgreSQL',
            'ORA-',
            'Microsoft SQL',
            'sqlite3',
            'unterminated',
            'warning:',
        ]
        
        response_text = response.text.lower()
        return any(pattern.lower() in response_text for pattern in error_patterns)
    
    def _detect_blind_injection(self, response: requests.Response) -> bool:
        """Detect evidence of blind SQL injection."""
        # Compare responses to true/false conditions
        # Implementation depends on application behavior
        return False
    
    def _extract_error(self, response: requests.Response) -> str:
        """Extract SQL error message from response."""
        # Extract relevant error information
        return response.text[:500]
```

### Database Configuration Testing

```bash
#!/bin/bash
# PostgreSQL configuration security test script
# Use only on systems you own or have explicit authorization to test

echo "=== PostgreSQL Security Configuration Audit ==="

# Test 1: Check SSL configuration
echo "1. Checking SSL configuration..."
psql -c "SHOW ssl;" 2>/dev/null

# Test 2: Check authentication method
echo "2. Checking authentication method..."
grep -E "host.*all.*all.*md5|host.*all.*all.*trust" pg_hba.conf

# Test 3: Check superuser accounts
echo "3. Checking superuser accounts..."
psql -c "SELECT rolname FROM pg_roles WHERE rolsuper;"

# Test 4: Check publicly accessible tables
echo "4. Checking public schema access..."
psql -c "SELECT tablename FROM pg_tables WHERE schemaname='public';"

# Test 5: Check log settings
echo "5. Checking audit logging..."
psql -c "SHOW logging_collector;"

# Test 6: Check role permissions
echo "6. Checking role permissions..."
psql -c "SELECT grantee, privilege_type FROM information_schema.table_privileges WHERE table_schema='public';"

# Test 7: Check RLS status
echo "7. Checking Row-Level Security..."
psql -c "SELECT relname, relrowsecurity FROM pg_class WHERE relrowsecurity = true;"

# Test 8: Check password encryption
echo "8. Checking password encryption..."
psql -c "SHOW password_encryption;"

echo "=== Audit Complete ==="
```

### Automated Vulnerability Scanning

```yaml
# OpenVAS vulnerability scan configuration for databases

openvas_task:
  name: "Database Security Assessment"
  scan_config: "Full and fast"
  target:
    hosts:
      - 10.0.1.50  # Database server IP
    ports:
      - 5432       # PostgreSQL
      - 3306       # MySQL
      - 1433      # SQL Server
    credentials:
      database:
        username: "security_scanner"
        password: "{{ vault_db_scanner_password }}"
  
  schedule:
    frequency: monthly
    start_time: "02:00"
  
  alerts:
    - email: security@example.com
    - webhook: https://siem.example.com/webhook
  
  nvt_selection:
    # Database-specific vulnerability families
    family:
      - "Databases"
      - "PostgreSQL"
      - "MySQL"
      - "Oracle"
      - "Microsoft SQL Server"
    severity:
      - "High"
      - "Critical"

# Nmap database enumeration script
# nmap --script=mysql-info,mysql-enum,mysql-brute -p 3306 target
# nmap --script=pgsql-info,pgsql-brute -p 5432 target
```

---

## Incident Response Procedures

### Database Security Incident Classification

Database security incidents require rapid, coordinated response. Incidents are classified by severity to determine appropriate response levels and resource allocation.

| Severity | Description | Response Time | Examples |
|----------|-------------|---------------|----------|
| Critical | Active breach, data exfiltration | Immediate | Active SQL injection attack, ransomware |
| High | Confirmed vulnerability, potential exposure | 1 hour | Missing encryption, privilege escalation |
| Medium | Configuration weakness | 24 hours | Weak passwords, excessive privileges |
| Low | Minor policy violation | 72 hours | Logged access anomaly, audit finding |

### Incident Response Workflow

```python
# Database security incident response system

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional
import logging

class IncidentSeverity(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

class IncidentStatus(Enum):
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    ERADICATED = "eradicated"
    RECOVERED = "recovered"
    CLOSED = "closed"

@dataclass
class SecurityIncident:
    """Database security incident record."""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    detected_at: datetime
    affected_systems: List[str]
    affected_data: List[str]
    assigned_to: Optional[str]
    remediation_steps: List[str]
    timeline: List[dict]

class DatabaseIncidentResponse:
    """
    Incident response system for database security events.
    """
    
    def __init__(self, db_connection, notification_service):
        self.db = db_connection
        self.notifications = notification_service
        self.logger = logging.getLogger('incident_response')
    
    def handle_suspicious_activity(self, alert: dict):
        """
        Handle detected suspicious database activity.
        """
        # Create incident record
        incident = SecurityIncident(
            incident_id=self._generate_incident_id(),
            title=alert['title'],
            description=alert['description'],
            severity=self._classify_severity(alert),
            status=IncidentStatus.DETECTED,
            detected_at=datetime.now(),
            affected_systems=alert.get('affected_systems', []),
            affected_data=alert.get('affected_data', []),
            assigned_to=None,
            remediation_steps=[],
            timeline=[{
                'timestamp': datetime.now(),
                'action': 'Incident detected',
                'actor': 'Automated system'
            }]
        )
        
        # Store incident
        self._store_incident(incident)
        
        # Initiate response based on severity
        if incident.severity == IncidentSeverity.CRITICAL:
            self._respond_critical(incident, alert)
        elif incident.severity == IncidentSeverity.HIGH:
            self._respond_high(incident, alert)
        else:
            self._respond_standard(incident, alert)
        
        return incident
    
    def _respond_critical(self, incident: SecurityIncident, alert: dict):
        """Critical incident response - immediate action required."""
        # 1. Notify security team immediately
        self.notifications.send_critical_alert(
            f"Critical database security incident: {incident.title}",
            incident.description
        )
        
        # 2. Isolate affected systems
        self._isolate_system(alert['affected_systems'])
        
        # 3. Disable compromised accounts
        if 'compromised_accounts' in alert:
            self._disable_accounts(alert['compromised_accounts'])
        
        # 4. Preserve evidence
        self._preserve_evidence(incident, alert)
        
        # 5. Escalate
        incident.status = IncidentStatus.INVESTIGATING
        self._escalate_incident(incident)
    
    def _isolate_system(self, systems: List[str]):
        """Isolate affected database systems."""
        # Implement system isolation logic
        self.logger.critical(f"Isolating systems: {systems}")
    
    def _disable_accounts(self, accounts: List[str]):
        """Disable compromised database accounts."""
        for account in accounts:
            try:
                self.db.execute(
                    f"ALTER USER {account} WITH LOGIN NOCREATEROLE NOSUPERUSER"
                )
                self.logger.info(f"Disabled account: {account}")
            except Exception as e:
                self.logger.error(f"Failed to disable account {account}: {e}")
    
    def _preserve_evidence(self, incident: SecurityIncident, alert: dict):
        """Preserve evidence for forensic analysis."""
        # Create forensic snapshot
        evidence = {
            'incident_id': incident.incident_id,
            'timestamp': datetime.now(),
            'system_state': self._capture_system_state(),
            'query_log': self._capture_query_log(alert),
            'network_connections': self._capture_connections(),
            'active_sessions': self._capture_sessions()
        }
        
        # Store evidence securely
        self._store_evidence(evidence)
    
    def _capture_system_state(self) -> dict:
        """Capture current database system state."""
        return {
            'connections': self.db.get_connection_count(),
            'active_queries': self.db.get_active_queries(),
            'locks': self.db.get_locks(),
            'configuration': self.db.get_configuration()
        }
```

### Containment and Recovery Procedures

```sql
-- PostgreSQL: Emergency containment procedures

-- 1. Terminate suspicious sessions
SELECT 
    pg_terminate_backend(pid) AS terminated,
    pid,
    usename,
    client_addr,
    query_start,
    state
FROM pg_stat_activity
WHERE state = 'active'
  AND query_start < NOW() - INTERVAL '5 minutes'
  AND usename IN ('suspicious_user1', 'suspicious_user2');

-- 2. Revoke access from compromised accounts
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM compromised_user;
REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public FROM compromised_user;
REVOKE USAGE ON SCHEMA public FROM compromised_user;
ALTER USER compromised_user WITH NOLOGIN;

-- 3. Check for unauthorized modifications
SELECT 
    schemaname,
    tablename,
    tupl ins,       -- Rows inserted
    tupd_upd,       -- Rows updated
    tupd_del        -- Rows deleted
FROM pg_stat_user_tables
WHERE schemaname IN ('public', 'sensitive')
  AND (tupd_ins > 0 OR tupd_upd > 0 OR tupd_del > 0)
  AND last_autoanalyze > NOW() - INTERVAL '1 hour';

-- 4. Identify recently created objects (potential backdoors)
SELECT 
    proname,
    pronamespace::regnamespace,
    prokind,
    prosrc
FROM pg_proc
WHERE pronamespace::regnamespace::text = 'public'
  AND created_at > NOW() - INTERVAL '24 hours';

-- 5. Verify data integrity
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE idx_scan = 0
  AND indexname NOT LIKE '%_pkey';

-- 6. Document incident in audit log
INSERT INTO security_incidents (
    incident_type,
    severity,
    description,
    actions_taken,
    detected_at,
    resolved_at
) VALUES (
    'unauthorized_access',
    'critical',
    'Suspicious activity detected from compromised account',
    'Account disabled, sessions terminated, forensic snapshot created',
    NOW(),
    NULL
);
```

---

## Security Automation and Tooling

### Database Security Monitoring

Automated security monitoring provides continuous surveillance for threats, enabling rapid detection and response. Modern monitoring systems combine multiple data sources including database logs, network traffic, and system metrics.

```python
# Database security monitoring implementation

import logging
from datetime import datetime, timedelta
from typing import List, Dict
import psycopg2
from psycopg2.extras import RealDictCursor

class DatabaseSecurityMonitor:
    """
    Automated security monitoring for database systems.
    Detects anomalies and potential security threats.
    """
    
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.logger = logging.getLogger('db_security_monitor')
        self.baseline = self._load_baseline()
    
    def run_security_checks(self) -> List[Dict]:
        """
        Run comprehensive security checks.
        """
        results = []
        
        results.append(self._check_failed_authentication())
        results.append(self._check_privilege_escalation())
        results.append(self._check_unusual_query_patterns())
        results.append(self._check_excessive_permissions())
        results.append(self._check_configuration_changes())
        results.append(self._check_data_exfiltration())
        
        return results
    
    def _check_failed_authentication(self) -> Dict:
        """Check for brute force authentication attempts."""
        query = """
            SELECT 
                pg_get_authid(rolname) AS username,
                COUNT(*) AS failed_attempts,
                MAX(auth.event_time) AS last_attempt
            FROM pg_authid rolname
            JOIN pg_logicaldecode.connections('') AS auth
                ON auth.rolname = rolname.rolname
            WHERE auth.auth_status = 'failed'
              AND auth.event_time > NOW() - INTERVAL '15 minutes'
            GROUP BY rolname
            HAVING COUNT(*) > 5
        """
        
        # Simplified check using pg_stat_activity
        check_query = """
            SELECT 
                usename,
                COUNT(*) AS failed_count,
                MAX(backend_start) AS last_attempt
            FROM pg_stat_activity
            WHERE state = 'idle'
              AND backend_start > NOW() - INTERVAL '15 minutes'
              AND wait_event IS NOT NULL
            GROUP BY usename
            HAVING COUNT(*) > 10
        """
        
        return {
            'check_name': 'failed_authentication',
            'results': self._execute_query(check_query),
            'alert_threshold': 10
        }
    
    def _check_privilege_escalation(self) -> Dict:
        """Check for privilege escalation attempts."""
        query = """
            SELECT 
                rolname AS role_name,
                rolsuper AS is_superuser,
                rolcanlogin AS can_login
            FROM pg_roles
            WHERE rolsuper = true
              AND rolname NOT IN ('postgres', 'pg_monitor')
        """
        
        return {
            'check_name': 'privilege_escalation',
            'results': self._execute_query(query),
            'alert_if_found': True
        }
    
    def _check_unusual_query_patterns(self) -> Dict:
        """Detect unusual query patterns indicating potential attacks."""
        query = """
            SELECT 
                query,
                calls,
                total_exec_time,
                rows,
                mean_exec_time
            FROM pg_stat_statements
            WHERE query LIKE '%union%select%'
               OR query LIKE '%information_schema%'
               OR query LIKE '%concat%'
            ORDER BY total_exec_time DESC
            LIMIT 10
        """
        
        return {
            'check_name': 'unusual_queries',
            'results': self._execute_query(query)
        }
    
    def _execute_query(self, query: str) -> List[Dict]:
        """Execute query and return results."""
        with psycopg2.connect(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query)
                return cursor.fetchall()
    
    def _load_baseline(self) -> Dict:
        """Load baseline metrics for comparison."""
        # Load from stored configuration
        return {}
```

### Security Orchestration and Automation

Security orchestration automates response to detected threats, reducing response times and ensuring consistent handling. Integration with SOAR platforms enables automated playbooks for common security scenarios.

```yaml
# Security automation playbook example

apiVersion: automation.example.com/v1
kind: SecurityPlaybook
metadata:
  name: database-threat-response
spec:
  name: "Database Threat Response"
  trigger:
    condition: "security_alert"
    severity: "high"
    source: "database_monitoring"
  
  steps:
    - name: "alert_security_team"
      action: notification.send
      params:
        channel: "security-alerts"
        message: "High severity database security alert detected"
    
    - name: "capture_evidence"
      action: database.snapshot
      params:
        snapshot_type: "forensic"
        retention_days: 90
    
    - name: "isolate_affected_system"
      action: network.isolate
      params:
        target: "{{ alert.affected_system }}"
        reason: "Security investigation"
    
    - name: "disable_compromised_accounts"
      action: database.revoke_access
      params:
        accounts: "{{ alert.compromised_accounts }}"
    
    - name: "enable_increased_monitoring"
      action: monitoring.increase
      params:
        target: "{{ alert.affected_system }}"
        level: "paranoid"
        duration_minutes: 60
    
    - name: "create_incident_ticket"
      action: ticketing.create
      params:
        title: "Database Security Incident: {{ alert.title }}"
        severity: "{{ alert.severity }}"
        assignee: "security-team"
    
    - name: "run_forensic_analysis"
      action: forensics.analyze
      params:
        evidence_id: "{{ steps.capture_evidence.evidence_id }}"
        analysis_type: "full"
    
    - name: "generate_incident_report"
      action: reporting.generate
      params:
        template: "security-incident"
        include: 
          - "alert_details"
          - "evidence_snapshot"
          - "timeline"
          - "recommendations"
```

### Vulnerability Scanning Integration

```python
# Integrate vulnerability scanning with database security

from typing import Dict, List
import subprocess

class DatabaseVulnerabilityScanner:
    """
    Automated vulnerability scanning for database systems.
    """
    
    def __init__(self, config: Dict):
        self.config = config
    
    def scan_postgresql(self, host: str, port: int = 5432) -> List[Dict]:
        """
        Scan PostgreSQL database for vulnerabilities.
        """
        vulnerabilities = []
        
        # Run nmap vulnerability scripts
        result = subprocess.run([
            'nmap',
            '--script', 'pgsql-info,pgsql-brute,pgsql-vacuumlo',
            '-p', str(port),
            host
        ], capture_output=True, text=True)
        
        # Parse results
        vulnerabilities.extend(self._parse_nmap_results(result.stdout))
        
        # Run CVE checks
        vulnerabilities.extend(self._check_cve_announcements(host))
        
        # Configuration audits
        vulnerabilities.extend(self._audit_configuration(host))
        
        return vulnerabilities
    
    def _parse_nmap_results(self, output: str) -> List[Dict]:
        """Parse nmap script output for vulnerabilities."""
        # Implementation depends on output format
        return []
    
    def _check_cve_announcements(self, host: str) -> List[Dict]:
        """Check for relevant CVE announcements."""
        # Query CVE database for PostgreSQL vulnerabilities
        return []
    
    def _audit_configuration(self, host: str) -> List[Dict]:
        """Audit database configuration for security issues."""
        issues = []
        
        config_checks = [
            ('ssl', 'should be enabled'),
            ('password_encryption', 'should be scram-sha-256 or better'),
            ('log_connections', 'should be on'),
            ('log_disconnections', 'should be on'),
        ]
        
        # Add implementation to check configuration
        return issues
    
    def generate_report(self, vulnerabilities: List[Dict]) -> str:
        """Generate vulnerability assessment report."""
        report = "# Database Vulnerability Assessment\n\n"
        
        critical = [v for v in vulnerabilities if v.get('severity') == 'critical']
        high = [v for v in vulnerabilities if v.get('severity') == 'high']
        medium = [v for v in vulnerabilities if v.get('severity') == 'medium']
        
        report += f"## Summary\n"
        report += f"- Critical: {len(critical)}\n"
        report += f"- High: {len(high)}\n"
        report += f"- Medium: {len(medium)}\n\n"
        
        for vuln in critical:
            report += f"### {vuln['title']}\n"
            report += f"**Severity:** {vuln['severity']}\n\n"
            report += f"{vuln['description']}\n\n"
            report += f"**Recommendation:** {vuln.get('recommendation', 'N/A')}\n\n"
        
        return report
```

---

## Conclusion

Database threat modeling provides a systematic approach to identifying, prioritizing, and mitigating security risks. This guide has covered common database threats including injection attacks, authentication bypass, data exfiltration, denial of service, and insider threats. Each threat category includes practical examples and mitigation strategies applicable to modern database systems.

Zero-trust database architecture implements the principle of never trusting always verifying, providing robust protection against modern attack vectors. Organizations should implement defense-in-depth strategies combining network segmentation, encryption, access controls, and continuous monitoring.

Penetration testing validates security controls by simulating real attacks, identifying vulnerabilities before they can be exploited. Regular testing should be performed by qualified security professionals with appropriate authorization.

Incident response procedures ensure rapid, coordinated action when security events occur. Well-defined procedures minimize damage, preserve evidence, and enable quick recovery. Automation enhances response consistency and reduces reaction times.

Security automation and tooling provide continuous protection for database systems. Automated monitoring, vulnerability scanning, and orchestration enable organizations to maintain strong security postures at scale.

---

## References

- OWASP Top 10
- NIST SP 800-53 Security and Privacy Controls
- CIS Benchmarks for Databases
- PCI DSS Requirements
- MITRE ATT&CK Framework
- Zero Trust Architecture (NIST SP 800-207)
