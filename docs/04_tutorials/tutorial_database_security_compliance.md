# Database Security and Compliance Tutorial for AI/ML Systems

## Overview

This hands-on tutorial teaches senior AI/ML engineers how to implement enterprise-grade security and compliance for database systems in AI applications. We'll cover encryption, access control, auditing, and regulatory compliance.

## Prerequisites
- PostgreSQL 14+ or MySQL 8+
- Redis 7+
- AWS/GCP/Azure account (for cloud examples)
- Basic understanding of security concepts

## Tutorial Structure
This tutorial is divided into 5 progressive sections:
1. **Encryption Implementation** - Data at rest, in transit, field-level
2. **Access Control** - RBAC, ABAC, tenant isolation
3. **Auditing and Monitoring** - Comprehensive logging and alerting
4. **Compliance Frameworks** - GDPR, HIPAA, SOC 2 implementation
5. **Vulnerability Assessment** - Security scanning and remediation

## Section 1: Encryption Implementation

### Step 1: Enable encryption at rest
```sql
-- PostgreSQL TDE (Transparent Data Encryption) with pgcrypto
-- First, create encryption key management
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Create encrypted table
CREATE TABLE encrypted_users (
    id UUID PRIMARY KEY,
    email TEXT,
    encrypted_data BYTEA,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert encrypted data
INSERT INTO encrypted_users (id, email, encrypted_data)
VALUES (
    gen_random_uuid(),
    'user@example.com',
    pgp_sym_encrypt(
        json_build_object(
            'name', 'John Doe',
            'ssn', '123-45-6789',
            'credit_card', '4111-1111-1111-1111'
        )::TEXT,
        'your-strong-password-here',
        'compress-algo=1, cipher-algo=aes256'
    )
);

-- Query encrypted data
SELECT 
    id,
    email,
    pgp_sym_decrypt(encrypted_data, 'your-strong-password-here')::JSON AS decrypted_data
FROM encrypted_users;
```

### Step 2: Field-level encryption for sensitive AI data
```python
import os
from cryptography.fernet import Fernet
import base64

class FieldLevelEncryption:
    def __init__(self, key=None):
        if key is None:
            key = Fernet.generate_key()
        self.cipher_suite = Fernet(key)
    
    def encrypt_field(self, data, field_name):
        """Encrypt specific fields based on sensitivity"""
        if field_name in ['ssn', 'credit_card', 'health_data']:
            return self.cipher_suite.encrypt(data.encode()).decode()
        return data
    
    def decrypt_field(self, encrypted_data, field_name):
        """Decrypt specific fields"""
        if field_name in ['ssn', 'credit_card', 'health_data']:
            return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
        return encrypted_data

# Usage example
fle = FieldLevelEncryption()

# Encrypt sensitive AI training data
training_data = {
    'user_id': '123',
    'ssn': '123-45-6789',
    'health_data': '{"diagnosis": "diabetes", "medications": ["insulin"]}',
    'features': {'age': 45, 'engagement_score': 0.85}
}

encrypted_data = {
    'user_id': training_data['user_id'],
    'ssn': fle.encrypt_field(training_data['ssn'], 'ssn'),
    'health_data': fle.encrypt_field(training_data['health_data'], 'health_data'),
    'features': training_data['features']
}

print("Encrypted:", encrypted_data)
```

### Step 3: TLS/SSL for data in transit
```sql
-- PostgreSQL SSL configuration
-- In postgresql.conf:
ssl = on
ssl_cert_file = '/path/to/server.crt'
ssl_key_file = '/path/to/server.key'
ssl_ca_file = '/path/to/ca.crt'

-- In pg_hba.conf:
hostssl all all 0.0.0.0/0 md5

-- Python connection with SSL
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="ai_db",
    user="postgres",
    password="password",
    sslmode="verify-full",
    sslrootcert="/path/to/ca.crt"
)
```

## Section 2: Access Control

### Step 1: Role-Based Access Control (RBAC)
```sql
-- Create roles for AI system
CREATE ROLE ai_analyst WITH LOGIN PASSWORD 'secure_password';
CREATE ROLE ai_engineer WITH LOGIN PASSWORD 'secure_password';
CREATE ROLE ai_admin WITH LOGIN PASSWORD 'secure_password';

-- Grant privileges
GRANT CONNECT ON DATABASE ai_db TO ai_analyst;
GRANT USAGE ON SCHEMA public TO ai_analyst;
GRANT SELECT ON TABLE features TO ai_analyst;

GRANT CONNECT ON DATABASE ai_db TO ai_engineer;
GRANT USAGE ON SCHEMA public TO ai_engineer;
GRANT SELECT, INSERT, UPDATE ON TABLE features TO ai_engineer;
GRANT SELECT ON TABLE models TO ai_engineer;

GRANT ALL PRIVILEGES ON DATABASE ai_db TO ai_admin;
```

### Step 2: Attribute-Based Access Control (ABAC)
```sql
-- PostgreSQL row-level security
ALTER TABLE features ENABLE ROW LEVEL SECURITY;

-- Policy for tenant isolation
CREATE POLICY tenant_isolation_policy ON features
USING (tenant_id = current_setting('app.tenant_id')::UUID);

-- Policy for feature access based on user role
CREATE POLICY feature_access_policy ON features
USING (
    current_setting('app.user_role') = 'admin' OR
    (current_setting('app.user_role') = 'engineer' AND is_active = true)
);

-- Set session variables
SET LOCAL app.tenant_id = '123e4567-e89b-12d3-a456-426614174000';
SET LOCAL app.user_role = 'engineer';
```

### Step 3: Multi-tenant access patterns
```python
class MultiTenantAccessControl:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def set_tenant_context(self, tenant_id, user_role):
        """Set tenant context for database operations"""
        cursor = self.db.cursor()
        cursor.execute("SET LOCAL app.tenant_id = %s", (str(tenant_id),))
        cursor.execute("SET LOCAL app.user_role = %s", (user_role,))
        cursor.close()
    
    def execute_query(self, query, params=None):
        """Execute query with tenant context"""
        cursor = self.db.cursor()
        try:
            cursor.execute(query, params or ())
            return cursor.fetchall()
        finally:
            cursor.close()
```

## Section 3: Auditing and Monitoring

### Step 1: Comprehensive audit logging
```sql
-- PostgreSQL audit extension
CREATE EXTENSION IF NOT EXISTS pgaudit;

-- Configure pgaudit
ALTER SYSTEM SET pgaudit.log = 'write, ddl';
ALTER SYSTEM SET pgaudit.log_catalog = ON;
ALTER SYSTEM SET pgaudit.role = 'audit_role';

-- Create audit log table
CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_id UUID,
    tenant_id UUID,
    operation VARCHAR(50),
    table_name VARCHAR(100),
    record_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT
);

-- Trigger for auditing
CREATE OR REPLACE FUNCTION audit_trigger()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_logs (user_id, tenant_id, operation, table_name, record_id, new_values, ip_address)
        VALUES (
            current_setting('app.user_id', TRUE)::UUID,
            current_setting('app.tenant_id', TRUE)::UUID,
            TG_OP,
            TG_TABLE_NAME,
            NEW.id,
            row_to_json(NEW),
            inet_client_addr()
        );
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_logs (user_id, tenant_id, operation, table_name, record_id, old_values, new_values, ip_address)
        VALUES (
            current_setting('app.user_id', TRUE)::UUID,
            current_setting('app.tenant_id', TRUE)::UUID,
            TG_OP,
            TG_TABLE_NAME,
            NEW.id,
            row_to_json(OLD),
            row_to_json(NEW),
            inet_client_addr()
        );
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_logs (user_id, tenant_id, operation, table_name, record_id, old_values, ip_address)
        VALUES (
            current_setting('app.user_id', TRUE)::UUID,
            current_setting('app.tenant_id', TRUE)::UUID,
            TG_OP,
            TG_TABLE_NAME,
            OLD.id,
            row_to_json(OLD),
            inet_client_addr()
        );
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to tables
CREATE TRIGGER audit_features
AFTER INSERT OR UPDATE OR DELETE ON features
FOR EACH ROW EXECUTE FUNCTION audit_trigger();
```

### Step 2: Real-time monitoring dashboard
```python
import dash
from dash import dcc, html
import plotly.graph_objs as go
import psycopg2
from datetime import datetime, timedelta

def create_security_dashboard():
    """Create real-time security monitoring dashboard"""
    
    app = dash.Dash(__name__)
    
    # Get security metrics
    def get_security_metrics():
        conn = psycopg2.connect(
            host="localhost",
            database="ai_db",
            user="postgres",
            password="password"
        )
        
        cursor = conn.cursor()
        
        # Failed login attempts
        cursor.execute("""
            SELECT COUNT(*) FROM audit_logs 
            WHERE operation = 'LOGIN' AND new_values->>'success' = 'false'
            AND timestamp > NOW() - INTERVAL '1 hour'
        """)
        failed_logins = cursor.fetchone()[0]
        
        # Sensitive data access
        cursor.execute("""
            SELECT COUNT(*) FROM audit_logs 
            WHERE table_name IN ('users', 'health_data', 'financial_data')
            AND timestamp > NOW() - INTERVAL '1 hour'
        """)
        sensitive_access = cursor.fetchone()[0]
        
        # Anomalous activity
        cursor.execute("""
            SELECT COUNT(*) FROM audit_logs 
            WHERE timestamp > NOW() - INTERVAL '1 hour'
            AND (new_values->>'anomaly_score' IS NOT NULL 
                 AND (new_values->>'anomaly_score')::FLOAT > 0.8)
        """)
        anomalies = cursor.fetchone()[0]
        
        conn.close()
        
        return failed_logins, sensitive_access, anomalies
    
    # Dashboard layout
    app.layout = html.Div([
        html.H1("AI System Security Dashboard"),
        
        dcc.Interval(
            id='interval-component',
            interval=60*1000,  # Update every minute
            n_intervals=0
        ),
        
        html.Div([
            html.Div([
                html.H3("Failed Login Attempts"),
                html.Div(id='failed-logins', className='metric'),
            ], className='metric-card'),
            
            html.Div([
                html.H3("Sensitive Data Access"),
                html.Div(id='sensitive-access', className='metric'),
            ], className='metric-card'),
            
            html.Div([
                html.H3("Anomalous Activity"),
                html.Div(id='anomalies', className='metric'),
            ], className='metric-card'),
        ], className='metrics-row'),
        
        dcc.Graph(id='security-timeline'),
    ])
    
    @app.callback(
        [dash.Output('failed-logins', 'children'),
         dash.Output('sensitive-access', 'children'),
         dash.Output('anomalies', 'children'),
         dash.Output('security-timeline', 'figure')],
        [dash.Input('interval-component', 'n_intervals')]
    )
    def update_metrics(n):
        failed, sensitive, anomalies = get_security_metrics()
        
        # Create timeline chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[datetime.now() - timedelta(minutes=i) for i in range(60)],
            y=[failed + i%5 for i in range(60)],
            mode='lines',
            name='Failed Logins'
        ))
        
        return str(failed), str(sensitive), str(anomalies), fig
    
    return app

# Run dashboard
if __name__ == '__main__':
    app = create_security_dashboard()
    app.run_server(debug=True)
```

## Section 4: Compliance Frameworks

### Step 1: GDPR Implementation
```sql
-- Right to be forgotten implementation
CREATE OR REPLACE FUNCTION gdpr_delete_user(user_id UUID)
RETURNS VOID AS $$
BEGIN
    -- Delete user data
    DELETE FROM users WHERE id = user_id;
    DELETE FROM features WHERE user_id = user_id;
    DELETE FROM training_data WHERE user_id = user_id;
    DELETE FROM model_predictions WHERE user_id = user_id;
    
    -- Log GDPR request
    INSERT INTO gdpr_requests (user_id, request_type, status, processed_at)
    VALUES (user_id, 'deletion', 'completed', NOW());
    
    -- Anonymize remaining references
    UPDATE audit_logs 
    SET user_id = NULL, new_values = jsonb_set(new_values, '{user_id}', '"ANONYMIZED"')
    WHERE user_id = user_id;
    
    COMMIT;
END;
$$ LANGUAGE plpgsql;

-- Data portability implementation
CREATE OR REPLACE FUNCTION gdpr_export_user_data(user_id UUID)
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'user', (SELECT row_to_json(u) FROM users u WHERE u.id = user_id),
        'features', (SELECT json_agg(row_to_json(f)) FROM features f WHERE f.user_id = user_id),
        'training_data', (SELECT json_agg(row_to_json(t)) FROM training_data t WHERE t.user_id = user_id)
    ) INTO result;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;
```

### Step 2: HIPAA Compliance for Health AI
```sql
-- HIPAA-specific encryption
CREATE TABLE health_records (
    id UUID PRIMARY KEY,
    patient_id UUID NOT NULL,
    encrypted_record BYTEA,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Audit trail for HIPAA
CREATE TABLE hipaa_audit_trail (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_id UUID,
    patient_id UUID,
    action VARCHAR(50),
    access_reason VARCHAR(200),
    ip_address INET,
    success BOOLEAN
);

-- Trigger for HIPAA audit
CREATE OR REPLACE FUNCTION hipaa_audit_trigger()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO hipaa_audit_trail (user_id, patient_id, action, ip_address, success)
    VALUES (
        current_setting('app.user_id', TRUE)::UUID,
        NEW.patient_id,
        TG_OP,
        inet_client_addr(),
        TRUE
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER hipaa_audit_health_records
AFTER INSERT OR UPDATE OR DELETE ON health_records
FOR EACH ROW EXECUTE FUNCTION hipaa_audit_trigger();
```

## Section 5: Vulnerability Assessment

### Step 1: Automated security scanning
```python
import requests
import json
from datetime import datetime

class DatabaseSecurityScanner:
    def __init__(self, db_url, api_key):
        self.db_url = db_url
        self.api_key = api_key
        self.headers = {'Authorization': f'Bearer {api_key}'}
    
    def scan_database(self):
        """Run comprehensive security scan"""
        payload = {
            'scan_type': 'comprehensive',
            'targets': ['users', 'features', 'training_data'],
            'checks': [
                'sql_injection',
                'insecure_configurations',
                'missing_encryption',
                'excessive_privileges',
                'audit_gaps'
            ]
        }
        
        response = requests.post(
            f"{self.db_url}/api/v1/scans",
            headers=self.headers,
            json=payload
        )
        
        return response.json()
    
    def generate_compliance_report(self, scan_results):
        """Generate compliance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'compliance_status': 'PASS' if scan_results['critical_issues'] == 0 else 'FAIL',
            'summary': {
                'total_issues': scan_results['total_issues'],
                'critical': scan_results['critical_issues'],
                'high': scan_results['high_issues'],
                'medium': scan_results['medium_issues'],
                'low': scan_results['low_issues']
            },
            'recommendations': []
        }
        
        # Generate recommendations based on findings
        if scan_results['critical_issues'] > 0:
            report['recommendations'].append({
                'priority': 'CRITICAL',
                'description': 'Critical security issues found',
                'action': 'Immediate remediation required'
            })
        
        return report

# Usage example
scanner = DatabaseSecurityScanner('https://security-api.example.com', 'your-api-key')
results = scanner.scan_database()
report = scanner.generate_compliance_report(results)
print(json.dumps(report, indent=2))
```

## Hands-on Exercises

### Exercise 1: Implement field-level encryption
1. Create a table with sensitive AI training data
2. Implement field-level encryption for PII fields
3. Test encryption/decryption functionality
4. Measure performance impact

### Exercise 2: Build RBAC system
1. Create roles for different AI team members
2. Implement row-level security policies
3. Test access control with different user contexts
4. Verify tenant isolation

### Exercise 3: Create audit logging
1. Set up PostgreSQL audit extension
2. Create audit trigger for critical tables
3. Test audit logging with various operations
4. Build simple monitoring dashboard

### Exercise 4: GDPR compliance implementation
1. Implement right-to-be-forgotten function
2. Create data export function
3. Test with sample user data
4. Verify compliance requirements

## Best Practices Summary

1. **Defense in Depth**: Multiple layers of security
2. **Least Privilege**: Grant minimum necessary permissions
3. **Continuous Monitoring**: Real-time security monitoring
4. **Regular Audits**: Monthly security assessments
5. **Compliance by Design**: Build compliance into architecture

This tutorial provides practical, hands-on experience with database security and compliance specifically for AI/ML systems. Complete all exercises to master these critical enterprise-grade security skills.