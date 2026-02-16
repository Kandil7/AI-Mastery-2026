# Database Security

Database security is critical for protecting sensitive data and ensuring compliance in AI/ML systems. This document covers comprehensive security practices for production database systems.

## Overview

Database security encompasses multiple layers: authentication, authorization, encryption, auditing, and vulnerability management. For senior AI/ML engineers, understanding database security is essential for building trustworthy AI applications that handle sensitive data.

## Core Security Principles

### Defense in Depth
- **Network layer**: Firewalls, VPCs, network segmentation
- **Application layer**: Input validation, parameterized queries
- **Database layer**: Authentication, authorization, encryption
- **Data layer**: Field-level encryption, tokenization
- **Physical layer**: Hardware security, secure data centers

### Zero Trust Architecture
- **Verify explicitly**: Never trust, always verify
- **Least privilege**: Grant minimum necessary permissions
- **Micro-segmentation**: Isolate database components
- **Continuous monitoring**: Real-time threat detection

## Authentication and Authorization

### Authentication Methods
- **Password-based**: Strong password policies, multi-factor authentication
- **Certificate-based**: TLS client certificates for mutual TLS
- **Token-based**: JWT, OAuth tokens for API access
- **IAM integration**: Cloud provider identity management

### Role-Based Access Control (RBAC)
```sql
-- PostgreSQL example
CREATE ROLE readonly_user WITH LOGIN PASSWORD 'secure_password';
CREATE ROLE write_user WITH LOGIN PASSWORD 'secure_password';
CREATE ROLE admin_user WITH LOGIN PASSWORD 'secure_password';

-- Grant permissions
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO write_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO admin_user;

-- Row-level security (RLS)
CREATE POLICY user_data_isolation ON users
FOR SELECT USING (current_user = owner_id);
```

### Attribute-Based Access Control (ABAC)
- **Dynamic policies**: Based on user attributes, resource attributes, environment
- **Example**: "Users can read data only if department matches"
- **Implementation**: Policy engines like Open Policy Agent (OPA)

## Data Encryption

### At-Rest Encryption
- **Transparent Data Encryption (TDE)**: Automatic encryption of data files
- **Filesystem encryption**: Encrypted volumes, LUKS
- **Cloud provider encryption**: AWS KMS, Azure Key Vault, GCP Cloud KMS
- **Key management**: HSMs, key rotation policies

### In-Transit Encryption
- **TLS 1.3**: Modern encryption for network traffic
- **Mutual TLS**: Both client and server authenticate
- **Certificate pinning**: Prevent man-in-the-middle attacks
- **Connection pooling**: Ensure encrypted connections

### Application-Level Encryption
- **Field-level encryption**: Encrypt sensitive fields individually
- **Client-side encryption**: Encrypt before sending to database
- **Homomorphic encryption**: Compute on encrypted data (emerging)
- **Tokenization**: Replace sensitive data with non-sensitive tokens

```python
# Application-level encryption example
from cryptography.fernet import Fernet
import os

class DataEncryptor:
    def __init__(self):
        # Store key securely (not in code!)
        self.key = os.getenv('ENCRYPTION_KEY')
        self.cipher_suite = Fernet(self.key)
    
    def encrypt_field(self, data):
        return self.cipher_suite.encrypt(data.encode())
    
    def decrypt_field(self, encrypted_data):
        return self.cipher_suite.decrypt(encrypted_data).decode()
```

## Vulnerability Management

### Common Database Vulnerabilities
- **SQL injection**: Unsanitized user input in queries
- **NoSQL injection**: Similar issues in NoSQL databases
- **Insecure defaults**: Default credentials, open ports
- **Misconfigurations**: Excessive privileges, weak encryption
- **Outdated software**: Unpatched vulnerabilities

### Prevention Strategies
- **Parameterized queries**: Always use prepared statements
- **Input validation**: Strict validation of all inputs
- **Regular patching**: Automated patch management
- **Security scanning**: Static and dynamic analysis
- **Penetration testing**: Regular security assessments

```sql
-- Safe query patterns
-- ❌ Dangerous: "SELECT * FROM users WHERE name = '" + user_input + "'"
-- ✅ Safe: 
PREPARE stmt FROM 'SELECT * FROM users WHERE name = ?';
EXECUTE stmt USING @user_input;

-- OR using ORM
User.query.filter(User.name == user_input).all();
```

## Auditing and Monitoring

### Audit Logging
- **Access logs**: Who accessed what and when
- **Query logs**: What queries were executed
- **Change logs**: What data was modified
- **Authentication logs**: Login attempts and failures

```sql
-- PostgreSQL audit logging configuration
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_connections = on;
ALTER SYSTEM SET log_disconnections = on;
ALTER SYSTEM SET log_duration = on;

-- Create audit table
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_id VARCHAR(255),
    action VARCHAR(50),
    table_name VARCHAR(255),
    record_id VARCHAR(255),
    details JSONB
);
```

### Real-time Monitoring
- **Anomaly detection**: Unusual query patterns or access times
- **Performance metrics**: Query latency, connection counts
- **Security events**: Failed logins, privilege escalation
- **Compliance monitoring**: GDPR, HIPAA, SOC 2 requirements

## AI/ML Specific Security Considerations

### Model Data Security
- **Training data protection**: Sensitive training data encryption
- **Model parameter security**: Protect model weights and architecture
- **Inference data security**: Secure input/output data in real-time
- **Prompt security**: Prevent prompt injection attacks

### Federated Learning Security
- **Secure aggregation**: Privacy-preserving model updates
- **Differential privacy**: Add noise to protect individual data
- **Homomorphic encryption**: Compute on encrypted model updates
- **Trusted execution environments**: SGX, TrustZone for secure computation

### AI-Specific Threats
- **Model stealing**: Protect against model extraction attacks
- **Adversarial attacks**: Defend against input manipulation
- **Data poisoning**: Detect and prevent malicious training data
- **Prompt injection**: Secure LLM interfaces from malicious prompts

## Compliance Requirements

### GDPR (General Data Protection Regulation)
- **Right to be forgotten**: Delete user data upon request
- **Data minimization**: Collect only necessary data
- **Consent management**: Track and manage user consent
- **Breach notification**: Report breaches within 72 hours

### HIPAA (Health Insurance Portability and Accountability Act)
- **Protected health information (PHI)**: Special handling requirements
- **Access controls**: Strict role-based access
- **Audit trails**: Comprehensive logging
- **Business associate agreements**: Third-party vendor requirements

### SOC 2 (Service Organization Control 2)
- **Security**: Logical and physical access controls
- **Availability**: System availability and disaster recovery
- **Processing integrity**: Data processing accuracy
- **Confidentiality**: Protection of confidential information
- **Privacy**: Collection, use, retention of personal data

## Best Practices

1. **Implement defense in depth**: Multiple layers of security
2. **Follow least privilege principle**: Minimum necessary permissions
3. **Encrypt sensitive data**: At rest, in transit, and in use
4. **Regular security audits**: Quarterly reviews and penetration testing
5. **Automate security**: CI/CD security checks, automated patching
6. **Train developers**: Security awareness and secure coding practices

## Related Resources

- [Database Operations] - Operational security practices
- [AI/ML Security] - AI-specific security considerations
- [Compliance Frameworks] - Detailed compliance requirements
- [Zero Trust Architecture] - Comprehensive zero trust implementation