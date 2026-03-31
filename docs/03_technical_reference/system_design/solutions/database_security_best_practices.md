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
GRANT SELECT, INSERT ON order_items TO app_service_account;
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
        User.name.ilike(f"%{search_term}%}")
    ).all()

# Django ORM (Python)
from django.db import models

class User(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()

# Using Django's ORM - automatically parameterized
def get_user_by_email(email):
    return User.objects.get(email=email)  # Parameterized internally

def search_users(name):
    return User.objects.filter(name__icontains=name)  # Safe filtering

# HIBERNATE (Java)
@Entity
public class User {
    @Id
    private Long id;
    private String name;
    
    // JPQL - uses parameterized queries
    public User findByName(String name) {
        TypedQuery<User> query = entityManager.createQuery(
            "SELECT u FROM User u WHERE u.name = :name", 
            User.class
        );
        query.setParameter("name", name);
        return query.getSingleResult();
    }
}
```

### Input Validation and Sanitization

While parameterized queries provide the primary defense against SQL injection, input validation adds an additional security layer. Validation ensures that data conforms to expected formats, lengths, and character sets before processing. This defense-in-depth approach provides protection even if parameterized queries are not used in some code paths.

```python
# Input validation example with multiple validation layers

import re
from typing import Optional

class InputValidator:
    """
    Validates and sanitizes user input before database operations.
    Provides defense-in-depth against injection attacks.
    """
    
    @staticmethod
    def validate_username(username: str) -> Optional[str]:
        """Validate username format and sanitize."""
        if not username:
            return None
        
        # Length check
        if len(username) < 3 or len(username) > 50:
            return None
        
        # Pattern check - alphanumeric and underscore only
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            return None
        
        return username
    
    @staticmethod
    def validate_email(email: str) -> Optional[str]:
        """Validate email format."""
        if not email:
            return None
        
        # RFC 5322 simplified email regex
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            return None
        
        return email.lower()
    
    @staticmethod
    def validate_numeric_id(value: str) -> Optional[int]:
        """Validate numeric ID input."""
        try:
            id_value = int(value)
            if id_value <= 0 or id_value > 2147483647:
                return None
            return id_value
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def sanitize_search_term(term: str) -> str:
        """Sanitize search term to prevent injection."""
        if not term:
            return ""
        
        # Remove or escape special SQL characters
        dangerous_chars = [';', '--', '/*', '*/', 'xp_', 'sp_', 'EXEC', 'UNION']
        sanitized = term
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limit length
        return sanitized[:200]
```

---

## Network Security and Firewall Rules

### Database Network Segmentation

Network segmentation isolates databases from untrusted network segments, reducing the attack surface and limiting lateral movement in case of a breach. Databases should reside in private network segments inaccessible from public networks. All database access should occur through controlled entry points such as application servers or bastion hosts.

Virtual private clouds (VPCs) provide network isolation in cloud environments. Database servers should be placed in private subnets with no direct internet access. Security groups and network ACLs control which resources can communicate with database instances.

```yaml
# AWS VPC Security Groups for database isolation

# Security group for PostgreSQL database
Resources:
  DatabaseSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: "Security group for PostgreSQL database"
      VpcId: !Ref DatabaseVPC
      SecurityGroupIngress:
        # Allow PostgreSQL from application tier only
        - IpProtocol: tcp
          FromPort: 5432
          ToPort: 5432
          SourceSecurityGroupId: !Ref AppTierSecurityGroup
        # Allow PostgreSQL from bastion host for admin access
        - IpProtocol: tcp
          FromPort: 5432
          ToPort: 5432
          SourceSecurityGroupId: !Ref BastionSecurityGroup
        # Explicitly deny all other inbound traffic
        - IpProtocol: -1
          FromPort: -1
          ToPort: -1
          CidrIp: 0.0.0.0/0
          Action: deny
  
  # Application tier security group
  AppTierSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: "Security group for application servers"
      VpcId: !Ref DatabaseVPC
      SecurityGroupIngress:
        # Allow HTTPS from ALB
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: 0.0.0.0/0
        # Allow SSH from bastion only
        - IpProtocol: tcp
          FromPort: 22
          ToPort: 22
          SourceSecurityGroupId: !Ref BastionSecurityGroup
```

### Firewall Configuration

Database firewall rules should follow the principle of least privilege, allowing only necessary network paths. Default deny policies block all traffic not explicitly permitted. Rules should specify source IP addresses or security groups, destination ports, and protocols.

```ini
# PostgreSQL: Configure connection listening and restrictions
# postgresql.conf
listen_addresses = 'localhost'  # Only listen on localhost
# For multi-instance setup, listen on internal network interface only
# listen_addresses = '10.0.1.10,10.0.1.11'

# Enable TCP/IP filtering if available
# Not all systems support this; use at OS level if possible

# MySQL: Restrict network binding
# my.cnf
[mysqld]
bind-address = 10.0.1.10  # Listen on specific internal IP only
skip-networking = 0      # Allow network connections
```

### Database Proxy for Connection Management

Database proxies sit between applications and database servers, providing an additional security layer. Proxies can enforce connection limits, filter queries, implement authentication, and provide centralized logging. They also enable database failover without application changes.

```yaml
# PgBouncer connection pooler configuration for security

[databases]
production_db = host=postgres-primary port=5432 dbname=proddb
replica_db = host=postgres-replica port=5432 dbname=proddb

[pgbouncer]
listen_port = 6432
listen_addr = 127.0.0.1  # Only listen locally
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
server_reset_query = DISCARD ALL
server_check_delay = 30s
max_client_conn = 1000
default_pool_size = 25
min_pool_size = 10
reserve_pool_size = 5
reserve_pool_timeout = 5

# Security: Log connections and disconnections
log_connections = 1
log_disconnections = 1
log_pooler_errors = 1

# Query timeout to prevent long-running queries
query_timeout = 60s
server_login_retry = 3
```

---

## Audit Logging and Monitoring

### Comprehensive Audit Logging

Audit logging records all database activities essential for security analysis, compliance reporting, and incident investigation. A comprehensive audit strategy captures authentication events, data access, schema changes, administrative operations, and failed operations. Log retention policies must meet regulatory requirements, often requiring multi-year retention for sensitive data.

```sql
-- PostgreSQL: Create comprehensive audit trigger system

-- Create audit log table
CREATE TABLE audit_log (
    audit_id BIGSERIAL PRIMARY KEY,
    audit_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    audit_user TEXT NOT NULL,
    audit_operation TEXT NOT NULL,
    audit_table TEXT NOT NULL,
    audit_schema TEXT NOT NULL,
    audit_record_id TEXT,
    old_data JSONB,
    new_data JSONB,
    application_name TEXT,
    client_addr INET,
    query_text TEXT
);

-- Create indexes for efficient audit log querying
CREATE INDEX idx_audit_log_timestamp ON audit_log(audit_timestamp);
CREATE INDEX idx_audit_log_user ON audit_log(audit_user);
CREATE INDEX idx_audit_log_table ON audit_log(audit_table);
CREATE INDEX idx_audit_log_operation ON audit_log(audit_operation);

-- Create audit function
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (
            audit_user,
            audit_operation,
            audit_table,
            audit_schema,
            audit_record_id,
            new_data,
            application_name,
            client_addr,
            query_text
        ) VALUES (
            current_user,
            'INSERT',
            TG_TABLE_NAME,
            TG_TABLE_SCHEMA,
            row_to_json(NEW)->>id::text,
            to_jsonb(NEW),
            current_setting('application_name', true),
            inet_client_addr(),
            current_query()
        );
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (
            audit_user,
            audit_operation,
            audit_table,
            audit_schema,
            audit_record_id,
            old_data,
            new_data,
            application_name,
            client_addr,
            query_text
        ) VALUES (
            current_user,
            'UPDATE',
            TG_TABLE_NAME,
            TG_TABLE_SCHEMA,
            row_to_json(NEW)->>id::text,
            to_jsonb(OLD),
            to_jsonb(NEW),
            current_setting('application_name', true),
            inet_client_addr(),
            current_query()
        );
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (
            audit_user,
            audit_operation,
            audit_table,
            audit_schema,
            audit_record_id,
            old_data,
            application_name,
            client_addr,
            query_text
        ) VALUES (
            current_user,
            'DELETE',
            TG_TABLE_NAME,
            TG_TABLE_SCHEMA,
            row_to_json(OLD)->>id::text,
            to_jsonb(OLD),
            current_setting('application_name', true),
            inet_client_addr(),
            current_query()
        );
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Apply audit trigger to sensitive tables
CREATE TRIGGER audit_customers
AFTER INSERT OR UPDATE OR DELETE ON customers
FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_orders
AFTER INSERT OR UPDATE OR DELETE ON orders
FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_payments
AFTER INSERT OR UPDATE OR DELETE ON payments
FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
```

### Security Monitoring and Alerting

Real-time monitoring detects security threats as they occur, enabling rapid response. Security monitoring should track failed authentication attempts, unusual access patterns, privilege escalations, and data exfiltration attempts. Integration with security information and event management (SIEM) systems enables correlation with other security events.

```python
# Security monitoring implementation with database audit analysis

import logging
from datetime import datetime, timedelta
from typing import List, Dict

class DatabaseSecurityMonitor:
    """
    Monitors database audit logs for security threats.
    Generates alerts for suspicious activities.
    """
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.logger = logging.getLogger('db_security')
        self.alert_threshold = {
            'failed_logins': 5,
            'failed_queries': 10,
            'bulk_select': 1000,
            'privilege_escalation': 1
        }
    
    def check_failed_authentication(self) -> List[Dict]:
        """Detect brute force authentication attacks."""
        query = """
            SELECT 
                username,
                COUNT(*) AS failed_attempts,
                MAX(timestamp) AS last_attempt,
                MIN(timestamp) AS first_attempt
            FROM authentication_log
            WHERE success = false
                AND timestamp > %s
            GROUP BY username
            HAVING COUNT(*) > %s
        """
        
        threshold_time = datetime.now() - timedelta(minutes=15)
        cursor = self.db.cursor()
        cursor.execute(query, (threshold_time, self.alert_threshold['failed_logins']))
        
        results = cursor.fetchall()
        
        for row in results:
            self.logger.warning(
                f"Failed authentication alert: {row[0]} "
                f"had {row[1]} failed attempts"
            )
        
        return results
    
    def detect_privilege_escalation(self) -> List[Dict]:
        """Detect privilege escalation attempts."""
        query = """
            SELECT 
                audit_timestamp,
                audit_user,
                audit_operation,
                new_data
            FROM audit_log
            WHERE audit_table = 'pg_authid'
                AND audit_timestamp > %s
        """
        
        threshold_time = datetime.now() - timedelta(hours=1)
        cursor = self.db.cursor()
        cursor.execute(query, (threshold_time,))
        
        results = cursor.fetchall()
        
        for row in results:
            self.logger.critical(
                f"Privilege escalation detected: {row[1]} "
                f"performed {row[2]} at {row[0]}"
            )
        
        return results
    
    def detect_bulk_data_access(self) -> List[Dict]:
        """Detect unusual bulk data access patterns."""
        query = """
            SELECT 
                audit_user,
                audit_table,
                COUNT(*) AS record_count,
                MAX(audit_timestamp) AS last_access
            FROM audit_log
            WHERE audit_operation = 'SELECT'
                AND audit_timestamp > %s
            GROUP BY audit_user, audit_table
            HAVING COUNT(*) > %s
        """
        
        threshold_time = datetime.now() - timedelta(minutes=30)
        cursor = self.db.cursor()
        cursor.execute(query, (threshold_time, self.alert_threshold['bulk_select']))
        
        results = cursor.fetchall()
        
        for row in results:
            self.logger.warning(
                f"Bulk data access detected: {row[0]} accessed "
                f"{row[2]} records from {row[1]}"
            )
        
        return results
    
    def generate_security_report(self) -> Dict:
        """Generate comprehensive security status report."""
        return {
            'timestamp': datetime.now(),
            'failed_authentications': self.check_failed_authentication(),
            'privilege_escalations': self.detect_privilege_escalation(),
            'bulk_access': self.detect_bulk_data_access()
        }
```

### Database Activity Monitoring with pgAudit

PostgreSQL's pgAudit extension provides detailed session and object-level auditing. This extension captures query text, timing information, and affected objects, enabling comprehensive security analysis and compliance reporting.

```sql
-- PostgreSQL: Configure pgAudit for comprehensive logging

-- Enable pgAudit in postgresql.conf
shared_preload_libraries = 'pg_stat_statements,pg_audit'
pg_audit.log = 'write, ddl, misc_set, function'
pg_audit.log_directory = 'pg_audit'
pg_audit.log_rotation_age = 1d
pg_audit.log_rotation_size = 100MB
pg_audit.log_connections = on
pg_audit.log_disconnections = on
pg_audit.log_parameter = on
pg_audit.log_statement = 'ddl'

-- Set audit role
CREATE ROLE audit_user WITH LOGIN;
GRANT pg_read_all_settings TO audit_user;

-- Audit specific tables
ALTER TABLE customers SET (pg_audit.log = 'write, delete');
ALTER TABLE orders SET (pg_audit.log = 'write');
ALTER TABLE payments SET (pg_audit.log = 'all');

-- Query audit logs
SELECT 
    timestamp,
    class,
    command,
    object_type,
    object_name,
    statement,
    user_name
FROM pg_audit_log
WHERE timestamp > NOW() - INTERVAL '1 hour'
ORDER BY timestamp DESC;
```

---

## Security Checklist

### Pre-Deployment Security Checklist

Before deploying any database to production, verify the following security controls are in place. This checklist covers critical security configurations that should be validated during deployment review.

| Security Control | Description | Status |
|-----------------|-------------|--------|
| Strong Authentication | Configure certificate-based or MFA authentication | [ ] |
| TLS Encryption | Enable TLS 1.2+ for all connections | [ ] |
| Encryption at Rest | Enable transparent data encryption | [ ] |
| RBAC Implementation | Create and assign roles based on job functions | [ ] |
| Row-Level Security | Enable RLS for multi-tenant or sensitive data | [ ] |
| SQL Injection Prevention | Use parameterized queries exclusively | [ ] |
| Network Segmentation | Place database in private network segment | [ ] |
| Firewall Rules | Configure strict inbound access rules | [ ] |
| Audit Logging | Enable comprehensive audit logging | [ ] |
| Security Monitoring | Configure real-time security alerts | [ ] |
| Credential Management | Use secrets management for credentials | [ ] |
| Backup Encryption | Encrypt database backups | [ ] |
| Patch Management | Apply latest security patches | [ ] |
| Vulnerability Scanning | Run database vulnerability scans | [ ] |

### Ongoing Security Maintenance

Security requires continuous attention. Establish regular security maintenance procedures to ensure ongoing protection against evolving threats.

- **Weekly**: Review failed authentication logs for brute force attempts
- **Monthly**: Conduct access reviews and remove unnecessary permissions
- **Quarterly**: Rotate encryption keys and update security policies
- **Annually**: Conduct penetration testing and security assessments

---

## Conclusion

Database security requires a comprehensive approach combining multiple defensive layers. Authentication mechanisms verify identity, authorization controls limit access, encryption protects data at rest and in transit, row-level security provides fine-grained access control, input validation prevents injection attacks, network segmentation isolates databases, and audit logging enables detection and response to security events.

Implement these best practices as part of a defense-in-depth strategy, and regularly review and update security controls to address evolving threats. Remember that security is not a one-time configuration but an ongoing process requiring continuous monitoring, assessment, and improvement.
