# Database Security Best Practices

## Table of Contents

1. [Introduction](#introduction)
2. [Authentication and Authorization Mechanisms](#authentication-and-authorization-mechanisms)
3. [Role-Based Access Control (RBAC)](#role-based-access-control-rbac)
4. [Encryption at Rest and in Transit](#encryption-at-rest-and-in-transit)
5. [Row-Level Security and Data Masking](#row-level-security-and-data-masking)
6. [SQL Injection Prevention](#sql-injection-prevention)
7. [Network Security and Firewall Rules](#network-security-and-firewall-rules)
8. [Audit Logging and Monitoring](#audit-logging-and-monitoring)
9. [Security Checklist](#security-checklist)

---

## Introduction

Database security is a critical component of any enterprise data architecture. This guide provides comprehensive best practices for securing databases across multiple dimensions: authentication, authorization, encryption, data masking, network security, and audit logging. The recommendations herein align with industry standards including OWASP, NIST, and various compliance frameworks such as GDPR, HIPAA, SOC 2, and PCI DSS.

Security must be implemented as a defense-in-depth strategy, layering multiple security controls throughout the database infrastructure. No single control provides complete protection; rather, the combination of multiple security measures creates a robust security posture that can withstand various attack vectors.

The practices outlined in this document apply to relational databases such as PostgreSQL, MySQL, and SQL Server, as well as NoSQL databases like MongoDB and Cassandra. Cloud-native databases including Amazon Aurora, Google Cloud Spanner, and Azure SQL Database provide built-in security features that complement these best practices.

---

## Authentication and Authorization Mechanisms

### Strong Authentication Protocols

Authentication serves as the first line of defense in database security, verifying the identity of users, applications, and systems attempting to access database resources. Implementing strong authentication mechanisms prevents unauthorized access and ensures that only legitimate entities can connect to the database.

Modern database systems support multiple authentication methods, each with varying levels of security. Password-based authentication remains common but should be implemented with strict policies regarding password complexity, expiration, and history. Certificate-based authentication provides stronger security by eliminating password-related vulnerabilities such as brute force attacks and credential theft. Integration with enterprise identity systems through LDAP or Active Directory enables centralized user management and simplifies compliance with security policies.

Multi-factor authentication (MFA) adds additional security layers by requiring users to present multiple forms of identification. This significantly reduces the risk of unauthorized access even if credentials are compromised. For databases containing sensitive data, MFA should be considered mandatory rather than optional.

```sql
-- PostgreSQL: Configure certificate-based authentication
-- In pg_hba.conf:
hostssl all all 0.0.0.0/0 cert clientcert=verify-full

-- Create user with certificate authentication
CREATE ROLE app_user WITH LOGIN;
ALTER ROLE app_user SET cert_oidc_issuer = 'CN=Enterprise CA,O=Company Inc';

-- MySQL: Enable caching_sha2_password authentication
ALTER USER 'app_user'@'%'
IDENTIFIED WITH caching_sha2_password
REQUIRE SSL;

-- SQL Server: Configure Windows Authentication with encryption
-- In connection string:
Server=myserver;Database=mydb;Integrated Security=SSPI;Encrypt=true;
```

### Connection Security

Database connections must be encrypted to prevent eavesdropping and man-in-the-middle attacks. Transport Layer Security (TLS) protocols encrypt data in transit between clients and database servers. Configuration should enforce TLS 1.2 or higher, disable older protocols, and use strong cipher suites.

Connection pooling introduces additional security considerations. While connection pools improve performance, they can also create security vulnerabilities if not properly managed. Connection pool configurations should enforce connection validation, implement proper timeout settings, and ensure that connections are properly closed and returned to the pool.

```ini
# PostgreSQL connection with TLS configuration
# postgresql.conf
ssl = on
ssl_cert_file = '/etc/ssl/certs/server.crt'
ssl_key_file = '/etc/ssl/private/server.key'
ssl_prefer_server_ciphers = on
ssl_ciphers = 'HIGH:!aNULL:!MD5:!RC4'
min_protocol_version = 'TLSv1.2'

# MySQL connection security
# my.cnf
[mysqld]
require_secure_transport = ON
tls_version = TLSv1.2,TLSv1.3
ssl_ca = /etc/ssl/certs/ca.pem
ssl_cert = /etc/ssl/certs/server-cert.pem
ssl_key = /etc/ssl/private/server-key.pem
```

### Service Account Management

Applications require database accounts to function, but these service accounts present significant security risks if not properly managed. Service accounts should follow the principle of least privilege, receiving only the permissions necessary to perform their designated functions. Each application should have its own dedicated database account rather than sharing credentials across multiple applications.

Credential management for service accounts requires careful attention. Credentials should never be hardcoded in application source code or configuration files. Instead, use secure secrets management systems such as HashiCorp Vault, AWS Secrets Manager, or Azure Key Vault. These systems provide encrypted storage, access controls, and audit logging for sensitive credentials.

```python
# Example: Using a secrets management system for database credentials
import boto3
import json

def get_database_credentials(secret_name: str) -> dict:
    """
    Retrieve database credentials from AWS Secrets Manager.
    Credentials are retrieved at runtime and never stored in code.
    """
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Application code uses retrieved credentials
def connect_to_database():
    credentials = get_database_credentials('prod/database/credentials')
    connection = create_engine(
        f"postgresql://{credentials['username']}:{credentials['password']}"
        f"@database.example.com:5432/proddb"
    )
    return connection
```

---

## Role-Based Access Control (RBAC)

### Designing the RBAC Framework

Role-based access control provides a structured approach to managing database permissions by assigning privileges to roles rather than individual users. This approach simplifies permission management, improves auditability, and supports the principle of least privilege. A well-designed RBAC framework should reflect the organization's structure and the job functions of database users.

The RBAC hierarchy typically includes multiple levels of access, from administrative roles with full database control to read-only roles for reporting and analytics. Defining clear role boundaries prevents privilege creep, where users accumulate unnecessary permissions over time. Regular access reviews ensure that users maintain only the permissions required for their current job functions.

Creating distinct roles for different functions improves security by limiting the blast radius of compromised accounts. An analyst with read-only access cannot inadvertently or maliciously modify data. An application service account with write access to specific tables cannot access unrelated data.

```sql
-- PostgreSQL: Comprehensive RBAC implementation

-- Create application roles
CREATE ROLE app_readonly;
CREATE ROLE app_readwrite;
CREATE ROLE app_admin;

-- Create user roles
CREATE ROLE analyst_readonly;
CREATE ROLE analyst_analytics;
CREATE ROLE developer;
CREATE ROLE dba_admin;
CREATE ROLE security_audit;

-- Grant table-level permissions to application roles
GRANT CONNECT ON DATABASE production_db TO app_readonly;
GRANT USAGE ON SCHEMA public TO app_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO app_readonly;

GRANT CONNECT ON DATABASE production_db TO app_readwrite;
GRANT USAGE ON SCHEMA public TO app_readwrite;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_readwrite;

-- Grant schema-level permissions to developer role
GRANT USAGE, CREATE ON SCHEMA dev TO developer;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA dev TO developer;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA dev TO developer;

-- DBA administrative role
GRANT dba_admin TO dba_user;
GRANT pg_read_all_settings TO dba_admin;
GRANT pg_execute_server_program TO dba_admin;

-- Set default privileges for new objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT SELECT ON TABLES TO app_readonly;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO app_readwrite;
```

### Role Hierarchy and Inheritance

Role inheritance allows organizations to create hierarchical role structures where higher-level roles automatically acquire permissions from lower-level roles. This reduces redundancy in permission management and ensures consistent access controls across the organization. However, inheritance chains should be carefully designed to prevent unintended permission accumulation.

```sql
-- Create role hierarchy

-- Base roles with specific permissions
CREATE ROLE data_consumer;
GRANT SELECT ON ALL TABLES IN SCHEMA reporting TO data_consumer;

CREATE ROLE data_editor;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA reporting TO data_editor;
GRANT data_consumer TO data_editor;  -- Inherit data_consumer permissions

CREATE ROLE data_manager;
GRANT DELETE ON ALL TABLES IN SCHEMA reporting TO data_manager;
GRANT data_editor TO data_manager;  -- Inherit all inherited permissions

CREATE ROLE reporting_admin;
GRANT ALL PRIVILEGES ON SCHEMA reporting TO reporting_admin;
GRANT data_manager TO reporting_admin;

-- Verify role permissions
SELECT
    r.rolname AS role_name,
    r.rolsuper AS is_superuser,
    r.rolcanlogin AS can_login
FROM pg_roles r
WHERE r.rolname IN ('data_consumer', 'data_editor', 'data_manager', 'reporting_admin')
ORDER BY r.rolname;
```

### Implementing Least Privilege

The principle of least privilege requires that users and applications receive only the minimum permissions necessary to perform their functions. This limits the potential impact of compromised credentials and reduces the attack surface of the database. Implementing least privilege requires careful analysis of actual permission requirements and ongoing vigilance against permission creep.

```sql
-- PostgreSQL: Implementing least privilege for table access

-- Instead of granting broad access, grant specific table permissions
GRANT SELECT ON orders TO app_service_account;
GRANT SELECT ON customers TO app_service_account;
GRANT SELECT, INSERT, UPDATE ON order_items TO app_service_account;
GRANT SELECT, UPDATE ON inventory TO app_service_account;

-- For stored procedures, grant EXECUTE permission
GRANT EXECUTE ON FUNCTION calculate_order_total(uuid) TO app_service_account;
GRANT EXECUTE ON FUNCTION process_payment(uuid, numeric) TO payment_service;

-- Column-level security for sensitive data
GRANT SELECT (id, name, email, status) ON users TO analyst_role;
GRANT SELECT (id, name, email, status, created_at) ON users TO senior_analyst_role;
-- Password hash and security questions only for admin role
GRANT SELECT (id, name, email, status, created_at, password_hash, security_questions)
ON users TO admin_role;

-- Verify effective permissions for a role
SELECT
    grantee,
    table_schema,
    table_name,
    privilege_type
FROM information_schema.table_privileges
WHERE grantee = 'app_service_account'
ORDER BY table_schema, table_name, privilege_type;
```

---

## Encryption at Rest and in Transit

### Encryption at Rest

Data at rest encryption protects stored data from unauthorized access in case of physical storage compromise, media theft, or unauthorized access to storage systems. Modern database systems support multiple encryption approaches, including transparent data encryption (TDE), application-level encryption, and full-disk encryption.

Transparent data encryption provides the strongest security because it encrypts data without requiring application changes. The database engine handles encryption and decryption automatically, ensuring that data is never stored in plaintext. Encryption keys are managed separately from data, typically through hardware security modules (HSMs) or key management services.

```sql
-- PostgreSQL: Tablespace encryption (PostgreSQL 10+)
CREATE TABLESPACE encrypted_tablespace
LOCATION '/data/encrypted'
WITH (encrypted = true);

-- PostgreSQL: Column-level encryption using pgcrypto extension
CREATE EXTENSION pgcrypto;

-- Encrypt sensitive columns at application level
-- Using AES-256 encryption with secure key management
ALTER TABLE customers
ADD COLUMN ssn_encrypted BYTEA;

-- Encryption function with proper key management
CREATE OR REPLACE FUNCTION encrypt_ssn(p_ssn TEXT, p_key BYTEA)
RETURNS BYTEA AS $$
BEGIN
    RETURN pgp_sym_encrypt(p_ssn, p_key, 'cipher=aes-256-cbc, compress-algo=1');
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION decrypt_ssn(p_encrypted BYTEA, p_key BYTEA)
RETURNS TEXT AS $$
BEGIN
    RETURN pgp_sym_decrypt(p_encrypted::BYTEA, p_key, 'cipher=aes-256-cbc');
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- SQL Server: Transparent Data Encryption
ALTER DATABASE [ProductionDB]
SET ENCRYPTION ON;

-- Create encryption key and certificate
USE ProductionDB;
CREATE MASTER KEY ENCRYPTION BY PASSWORD = 'ComplexPassword123!';
CREATE CERTIFICATE DatabaseCert WITH SUBJECT = 'Database Encryption';
CREATE DATABASE ENCRYPTION KEY
WITH ALGORITHM = AES_256
ENCRYPTION BY SERVER CERTIFICATE DatabaseCert;

-- MySQL: InnoDB tablespace encryption
ALTER TABLE customers ENCRYPTION = 'Y';
```

### Encryption in Transit

All database communications should be encrypted using TLS to prevent eavesdropping, man-in-the-middle attacks, and data interception. Even internal network traffic between application servers and databases requires encryption because internal networks can be compromised through insider threats or lateral movement by attackers.

```ini
# PostgreSQL: Enforce SSL connections
# postgresql.conf
ssl = on
ssl_cert_file = '/etc/ssl/certs/server.crt'
ssl_key_file = '/etc/ssl/private/server.key'
ssl_ca_file = '/etc/ssl/certs/ca.crt'

# Require SSL for all connections
# pg_hba.conf
hostssl all all 0.0.0.0/0 md5
hostssl all all ::0/0 md5

# MySQL: Force secure transport
# my.cnf
[mysqld]
require_secure_transport = ON
ssl_ca = /etc/ssl/certs/ca.pem
ssl_cert = /etc/ssl/certs/server-cert.pem
ssl_key = /etc/ssl/private/server-key.pem

# Client configuration
[client]
ssl_ca = /etc/ssl/certs/ca.pem
ssl_cert = /etc/ssl/certs/client-cert.pem
ssl_key = /etc/ssl/private/client-key.pem
```

### Key Management

Encryption is only as strong as key management practices. Encryption keys must be stored separately from encrypted data, rotated regularly, and protected with strong access controls. Key compromise renders all encrypted data vulnerable, making key management a critical security function.

```python
# Example: Key management best practices with HashiCorp Vault
import hvac
import os

class DatabaseKeyManager:
    """
    Manages database encryption keys using HashiCorp Vault.
    Keys are stored separately from data and rotated regularly.
    """

    def __init__(self, vault_addr: str, mount_point: str = 'database-keys'):
        self.client = hvac.Client(url=vault_addr)
        self.mount_point = mount_point
        self._ensure_secrets_engine()

    def _ensure_secrets_engine(self):
        """Ensure the KV secrets engine is enabled."""
        if not self.client.sys.is_mounted(self.mount_point):
            self.client.sys.enable_secrets_engine(
                'kv',
                path=self.mount_point,
                options={'version': 2}
            )

    def store_key(self, key_name: str, key_value: str):
        """Store an encryption key securely."""
        self.client.secrets.kv.v2.create_or_update_secret(
            path=key_name,
            secret={'encryption_key': key_value},
            mount_point=self.mount_point
        )

    def retrieve_key(self, key_name: str) -> str:
        """Retrieve an encryption key."""
        response = self.client.secrets.kv.v2.read_secret_version(
            path=key_name,
            mount_point=self.mount_point
        )
        return response['data']['data']['encryption_key']

    def rotate_key(self, key_name: str) -> str:
        """
        Rotate an encryption key. This creates a new key version
        while preserving access to data encrypted with old keys.
        """
        import secrets
        new_key = secrets.token_hex(32)  # 256-bit key
        self.store_key(key_name, new_key)

        # Log key rotation for audit purposes
        print(f"Key {key_name} rotated at {datetime.now()}")

        return new_key
```

---

## Row-Level Security and Data Masking

### Row-Level Security (RLS)

Row-level security enables fine-grained access control by filtering query results based on user context. This security feature is particularly valuable in multi-tenant applications where different customers should only see their own data. RLS policies are defined at the table level and automatically applied to all queries, ensuring consistent enforcement without requiring application code changes.

PostgreSQL provides robust RLS support through policy definitions that can reference session variables, current user attributes, or arbitrary expressions. This allows implementation of complex access patterns such as tenant isolation, department-based filtering, or hierarchical data access.

```sql
-- PostgreSQL: Enable Row-Level Security
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE customers ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Create policies for multi-tenant application
-- Each tenant sees only their own data
CREATE POLICY tenant_isolation_orders ON orders
    FOR SELECT
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_customers ON customers
    FOR ALL
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- Set tenant context for application connections
ALTER ROLE app_user SET app.tenant_id = '550e8400-e29b-41d4-a716-446655440000';

-- More complex policy: Users can see their own records
-- plus records from their department
CREATE POLICY department_data_access ON documents
    FOR SELECT
    USING (
        owner_id = current_user
        OR department_id = current_setting('app.department_id')::uuid
        OR is_public = true
    );

-- Policy for hierarchical access
CREATE POLICY manager_view ON employee_records
    FOR SELECT
    USING (
        id = current_user::uuid
        OR manager_id = current_user::uuid
        OR department_id IN (
            SELECT department_id
            FROM user_departments
            WHERE user_id = current_user::uuid
        )
    );
```

### Dynamic Data Masking

Data masking protects sensitive information by obscuring it during display or transmission while preserving the original data for processing. Unlike encryption, which protects data at rest, masking protects data in use and in transit. Dynamic data masking transforms data in real-time based on the querying user's role.

```sql
-- PostgreSQL: Implement data masking using views and column permissions

-- Create a masking function
CREATE OR REPLACE FUNCTION mask_email(email TEXT)
RETURNS TEXT AS $$
BEGIN
    IF email IS NULL THEN
        RETURN NULL;
    END IF;
    -- Show only first character and domain
    RETURN LEFT(email, 1) || '***@' || RIGHT(email, LENGTH(email) - POSITION('@' IN email));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create masked view for customer service representatives
CREATE VIEW customers_masked AS
SELECT
    customer_id,
    first_name,
    last_name,
    mask_email(email) AS email,
    mask_phone(phone) AS phone,
    mask_ssn(ssn) AS ssn,
    address_line_1,
    city,
    state,
    postal_code
FROM customers;

-- Column-level security with conditional access
GRANT SELECT (customer_id, first_name, last_name) ON customers TO csr_role;
GRANT SELECT (email, phone) ON customers TO csr_extended_role;
GRANT SELECT (ssn) ON customers TO hr_role;

-- SQL Server: Dynamic Data Masking
ALTER TABLE customers
ADD MASKED WITH (FUNCTION = 'email()') FOR email;

ALTER TABLE customers
ADD MASKED WITH (FUNCTION = 'partial(4, "XXXX", 0)') FOR phone;

ALTER TABLE customers
ADD MASKED WITH (FUNCTION = 'default()') FOR ssn;

-- Grant unmasked access to specific roles
GRANT UNMASK TO hr_analyst;
GRANT SELECT ON customers TO hr_analyst;
```

### Implementing Column-Level Security

Column-level security restricts access to specific columns within tables, providing finer granularity than row-level security. This is essential for protecting sensitive columns such as Social Security numbers, credit card information, or salary data that should only be accessible to authorized users.

```sql
-- PostgreSQL: Column-level security implementation

-- Revoke default access to sensitive columns
REVOKE SELECT (salary, ssn, date_of_birth) ON employees FROM PUBLIC;

-- Grant selective access
GRANT SELECT (employee_id, first_name, last_name, department, title) ON employees TO all_employees;
GRANT SELECT (salary) ON employees TO manager_role;
GRANT SELECT (salary, ssn, date_of_birth) ON employees TO hr_role;

-- Create secure view for different access levels
CREATE VIEW employee_directory AS
SELECT
    employee_id,
    first_name,
    last_name,
    email,
    department,
    title,
    hire_date
FROM employees;

CREATE VIEW employee_salaries AS
SELECT
    e.employee_id,
    e.first_name,
    e.last_name,
    e.department,
    s.salary,
    s.bonus
FROM employees e
JOIN salaries s ON e.employee_id = s.employee_id;

-- MySQL: Column-level privileges
GRANT SELECT (id, name, email) ON application.users TO app_readonly;
GRANT SELECT (id, name, email, phone, address) ON application.users TO app_support;
GRANT SELECT, INSERT, UPDATE (id, name, email, phone, address) ON application.users TO app_write;
```

---

## SQL Injection Prevention

### Understanding SQL Injection

SQL injection attacks exploit insufficient input validation to inject malicious SQL code into queries. Successful attacks can exfiltrate sensitive data, modify or delete data, execute administrative operations, or compromise the underlying server. Preventing SQL injection requires treating all user input as untrusted and using defensive coding practices.

The most effective prevention method is parameterized queries, which separate SQL code from data values. This ensures that user input cannot be interpreted as SQL code, eliminating the injection attack surface entirely. ORMs and query builders typically use parameterized queries under the hood, but developers must verify their usage.

```python
# SECURE: Parameterized queries using different database drivers

# PostgreSQL with psycopg2
import psycopg2

def get_user_by_username(username):
    conn = psycopg2.connect(database="users")
    cursor = conn.cursor()

    # Parameterized query - safe from SQL injection
    query = "SELECT * FROM users WHERE username = %s"
    cursor.execute(query, (username,))  # Tuple, not string interpolation

    return cursor.fetchone()

# MySQL with mysql-connector
import mysql.connector

def get_user_by_email(email):
    conn = mysql.connector.connect(database="users")
    cursor = conn.cursor()

    # Parameterized query
    query = "SELECT * FROM users WHERE email = %s"
    cursor.execute(query, (email,))

    return cursor.fetchone()

# SQL Server with pyodbc
import pyodbc

def get_order_details(order_id):
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()

    # Parameterized query
    query = "SELECT * FROM orders WHERE order_id = ?"
    cursor.execute(query, (order_id,))

    return cursor.fetchone()
```

### ORM Security Best Practices

Object-Relational Mapping (ORM) frameworks provide some protection against SQL injection by abstracting away raw SQL queries. However, improper ORM usage can still introduce vulnerabilities. Understanding how ORM query methods translate to SQL helps developers avoid common pitfalls.

```python
# SECURE: ORM queries with proper parameterization

# SQLAlchemy (Python)
from sqlalchemy.orm import Session

def get_user_by_id(session: Session, user_id: int):
    # Using ORM query - automatically parameterized
    return session.query(User).filter(User.id == user_id).first()

def search_users(session: Session, search_term: str):
    # Using filter() - safe from injection
    return session.query(User).filter(
        User.name.ilike(f"%{search_term}%")
    ).all()

# Django ORM
from django.db import models

def get_user_by_username(username):
    # Safe - Django ORM uses parameterized queries
    return User.objects.filter(username=username).first()

def search_users(search_term):
    # Safe - Django ORM handles parameterization
    return User.objects.filter(name__icontains=search_term)
```

---

## Network Security and Firewall Rules

### Network Segmentation

Network segmentation isolates database servers from other network segments, reducing the attack surface and preventing lateral movement in case of compromise. Databases should be placed in dedicated network segments with strict firewall rules that only allow necessary traffic.

```bash
# Example: AWS Security Group for database
# Inbound rules:
# - Port 5432 (PostgreSQL) from application servers only
# - Port 22 (SSH) from management bastion host only
# - No internet access for database instances

# Example: Linux iptables rules
iptables -A INPUT -p tcp --dport 5432 -s 10.0.1.0/24 -j ACCEPT
iptables -A INPUT -p tcp --dport 5432 -j DROP
iptables -A INPUT -p tcp --dport 22 -s 10.0.0.10/32 -j ACCEPT
iptables -A INPUT -p tcp --dport 22 -j DROP
```

### Database-Specific Network Security

Database systems provide additional network security features beyond basic firewall rules. PostgreSQL's `pg_hba.conf` file controls client authentication and connection types, allowing fine-grained control over who can connect and how. MySQL's `mysql.user` table and host-based privileges provide similar functionality.

```ini
# PostgreSQL: pg_hba.conf security configuration
# Local connections
local   all             all                                     peer
# SSL connections from application servers
hostssl all             all             10.0.1.0/24         md5
# Replication connections
hostssl replication     all             10.0.2.0/24         md5
# Admin connections from management network
hostssl all             all             10.0.0.0/24         cert clientcert=verify-full
# Deny all other connections
host    all             all             0.0.0.0/0           reject
```

### Secure Communication Channels

For databases that must communicate across networks, secure communication channels are essential. This includes using VPNs for cross-cloud or hybrid deployments, and implementing mutual TLS (mTLS) for service-to-service communication.

```yaml
# Kubernetes NetworkPolicy for database security
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: database-network-policy
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

---

## Audit Logging and Monitoring

### Comprehensive Audit Logging

Audit logs provide a record of all database activities, enabling forensic analysis, compliance verification, and security incident response. Comprehensive audit logging should capture connection attempts, authentication events, data access, modifications, and administrative operations.

```sql
-- PostgreSQL: Enable audit logging
-- In postgresql.conf
log_statement = 'all'  -- Log all statements
log_connections = on
log_disconnections = on
log_duration = on
log_min_duration_statement = 1000  -- Log statements taking >1s
log_checkpoints = on
log_lock_waits = on

-- Create audit log table
CREATE TABLE audit_log (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_name TEXT NOT NULL,
    application_name TEXT,
    client_address INET,
    query TEXT,
    query_duration_ms INT,
    rows_affected INT,
    event_type TEXT NOT NULL,
    details JSONB
);

-- Trigger to capture DML operations
CREATE OR REPLACE FUNCTION log_dml_operation()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (user_name, application_name, client_address, query, event_type, details)
        VALUES (current_user, current_setting('application_name'), inet_client_addr(), TG_TABLE_NAME, 'INSERT', row_to_json(NEW));
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (user_name, application_name, client_address, query, event_type, details)
        VALUES (current_user, current_setting('application_name'), inet_client_addr(), TG_TABLE_NAME, 'UPDATE', json_build_object('old', row_to_json(OLD), 'new', row_to_json(NEW)));
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (user_name, application_name, client_address, query, event_type, details)
        VALUES (current_user, current_setting('application_name'), inet_client_addr(), TG_TABLE_NAME, 'DELETE', row_to_json(OLD));
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to tables
CREATE TRIGGER audit_triggers
AFTER INSERT OR UPDATE OR DELETE ON orders
FOR EACH ROW EXECUTE FUNCTION log_dml_operation();
```

### Real-Time Monitoring and Alerting

Real-time monitoring detects security incidents as they occur, enabling rapid response. Monitoring should track unusual activity patterns such as high-volume queries, access from unexpected locations, or privilege escalation attempts.

```python
# Example: Real-time security monitoring with Prometheus and Alertmanager
import prometheus_client
from prometheus_client import Counter, Histogram

# Define metrics
db_queries_total = Counter('db_queries_total', 'Total number of database queries', ['user', 'operation'])
db_query_duration_seconds = Histogram('db_query_duration_seconds', 'Query duration in seconds', ['user', 'operation'])
db_failed_attempts = Counter('db_failed_attempts_total', 'Failed authentication attempts', ['user'])

# Monitor suspicious activity
def monitor_security_events():
    # Check for high-frequency queries
    high_frequency_threshold = 100  # queries per minute
    recent_queries = get_recent_queries(time_window=60)
    
    for user, count in recent_queries.items():
        if count > high_frequency_threshold:
            alert(f"Suspicious query frequency: {count} queries/min from user {user}")
    
    # Check for privilege escalation
    recent_privilege_changes = get_privilege_changes(last_hour=True)
    for change in recent_privilege_changes:
        if change['new_privilege'] == 'superuser':
            alert(f"Privilege escalation detected: {change['user']} granted superuser access")

# Prometheus exporter for database metrics
@app.route('/metrics')
def metrics():
    return generate_latest()
```

### Compliance Reporting

Compliance requirements such as GDPR, HIPAA, and PCI DSS mandate specific security controls and reporting. Database security implementations should include automated compliance reporting capabilities that generate evidence for audits.

```sql
-- Generate compliance report for GDPR right-to-be-forgotten requests
CREATE OR REPLACE FUNCTION generate_gdpr_compliance_report(user_id UUID)
RETURNS TABLE (
    table_name TEXT,
    row_count INT,
    data_types TEXT[],
    retention_period_days INT
) AS $$
DECLARE
    result RECORD;
BEGIN
    FOR result IN
        SELECT
            tablename AS table_name,
            COUNT(*) AS row_count,
            ARRAY_AGG(DISTINCT column_name) AS data_types,
            MAX(retention_days) AS retention_period_days
        FROM information_schema.columns c
        JOIN (
            SELECT 'users' AS tablename, 365 AS retention_days UNION ALL
            SELECT 'orders', 730 UNION ALL
            SELECT 'logs', 90
        ) t ON c.table_name = t.tablename
        WHERE c.column_name ILIKE '%email%' OR c.column_name ILIKE '%name%' OR c.column_name ILIKE '%address%'
        GROUP BY tablename
    LOOP
        RETURN NEXT result;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

---

## Security Checklist

### Pre-Deployment Security Checklist
- [ ] All database connections use TLS 1.2+
- [ ] Strong authentication configured (MFA for admin accounts)
- [ ] Principle of least privilege implemented
- [ ] Sensitive data encrypted at rest
- [ ] Row-level security enabled for multi-tenant applications
- [ ] SQL injection prevention measures in place
- [ ] Comprehensive audit logging configured
- [ ] Network segmentation implemented
- [ ] Regular security patching schedule established
- [ ] Backup encryption verified
- [ ] Disaster recovery plan tested

### Ongoing Security Maintenance
- [ ] Monthly access reviews performed
- [ ] Quarterly penetration testing conducted
- [ ] Annual security audits completed
- [ ] Key rotation scheduled and automated
- [ ] Security incident response plan updated
- [ ] Compliance requirements reviewed and updated
- [ ] Security training provided to database administrators
- [ ] Vulnerability scanning performed regularly
- [ ] Threat intelligence integrated into security operations
- [ ] Security metrics monitored and reported

### Critical Security Controls
- [ ] Multi-factor authentication for administrative access
- [ ] Encryption of sensitive data at rest and in transit
- [ ] Row-level security for multi-tenant isolation
- [ ] Parameterized queries for all application code
- [ ] Comprehensive audit logging with retention
- [ ] Network segmentation and firewall rules
- [ ] Regular security patching and updates
- [ ] Secure credential management for service accounts
- [ ] Automated vulnerability scanning
- [ ] Incident response capability established

By following these best practices, organizations can establish a robust database security posture that protects sensitive data, ensures regulatory compliance, and maintains business continuity in the face of evolving security threats.