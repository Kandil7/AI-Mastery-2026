# Database Compliance Guide

## Table of Contents

1. [Introduction to Database Compliance](#introduction-to-database-compliance)
2. [GDPR Compliance for Databases](#gdpr-compliance-for-databases)
3. [HIPAA Compliance for Healthcare Data](#hipaa-compliance-for-healthcare-data)
4. [SOC 2 Compliance Requirements](#soc-2-compliance-requirements)
5. [PCI DSS for Payment Data](#pci-dss-for-payment-data)
6. [Data Retention and Deletion Policies](#data-retention-and-deletion-policies)
7. [Audit Trail Requirements](#audit-trail-requirements)
8. [Privacy-Preserving Techniques](#privacy-preserving-techniques)
9. [Compliance Checklist](#compliance-checklist)

---

## Introduction to Database Compliance

Database compliance encompasses the legal, regulatory, and organizational requirements governing how organizations collect, store, process, and protect data. Different types of data are subject to different compliance frameworks, and organizations handling multiple data types must implement controls satisfying all applicable requirements.

Compliance is not merely a technical implementation but involves processes, policies, training, and ongoing monitoring. Regulatory frameworks impose specific requirements for data protection, access controls, audit logging, incident response, and data subject rights. Understanding these requirements enables the design of database architectures that support compliance while maintaining operational efficiency.

Non-compliance can result in significant financial penalties, reputational damage, and legal liability. The European Union's General Data Protection Regulation (GDPR) imposes fines up to 4% of global annual revenue. HIPAA violations can reach $1.5 million per violation category. PCI DSS non-compliance can result in fines ranging from $5,000 to $100,000 monthly and loss of payment processing privileges.

This guide provides technical implementation details for major compliance frameworks, enabling database administrators and developers to build compliant database systems. Each section includes specific SQL scripts, configuration examples, and procedural requirements necessary for compliance.

---

## GDPR Compliance for Databases

### Overview of GDPR Requirements

The General Data Protection Regulation (GDPR) applies to organizations processing personal data of EU residents, regardless of where the organization is located. GDPR establishes requirements for data minimization, purpose limitation, storage limitation, accuracy, integrity and confidentiality, accountability, and data subject rights.

The regulation defines personal data broadly, encompassing any information relating to an identifiable natural person. This includes names, identification numbers, location data, online identifiers, and factors specific to physical, physiological, genetic, mental, economic, cultural, or social identity.

GDPR introduces specific database-related requirements including the right to access, right to rectification, right to erasure (right to be forgotten), data portability, and restrictions on automated decision-making. Database systems must support these rights through technical implementations.

### Data Minimization and Purpose Limitation

GDPR requires that organizations collect only personal data that is adequate, relevant, and limited to what is necessary for the purposes for which it is processed. Database schemas should be designed to enforce data minimization, storing only essential fields and avoiding unnecessary data collection.

```sql
-- PostgreSQL: Implement data minimization with column-level security

-- Create customer table with only necessary fields
CREATE TABLE customers (
    customer_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    -- Store minimal customer identification
    customer_reference VARCHAR(50) UNIQUE NOT NULL,
    -- Avoid storing excessive personal details
    communication_preference VARCHAR(20) DEFAULT 'email',
    account_status VARCHAR(20) DEFAULT 'active'
);

-- Separate sensitive data with restricted access
CREATE TABLE customer_identifiers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID NOT NULL REFERENCES customers(customer_id),
    -- Encrypted identifier storage
    identifier_type VARCHAR(20) NOT NULL,
    identifier_encrypted BYTEA NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    verified_at TIMESTAMPTZ
);

-- Purpose tracking table
CREATE TABLE data_processing_purposes (
    purpose_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    purpose_name VARCHAR(100) NOT NULL,
    purpose_description TEXT,
    legal_basis VARCHAR(50) NOT NULL,
    data_categories JSONB NOT NULL,
    retention_period INTERVAL NOT NULL
);

-- Link data to specific purposes
CREATE TABLE customer_data_purposes (
    customer_id UUID NOT NULL REFERENCES customers(customer_id),
    purpose_id UUID NOT NULL REFERENCES data_processing_purposes(purpose_id),
    consent_given_at TIMESTAMPTZ,
    consent_withdrawn_at TIMESTAMPTZ,
    PRIMARY KEY (customer_id, purpose_id)
);
```

### Right to Access Implementation

Data subjects have the right to obtain confirmation as to whether personal data concerning them is being processed, where, and for what purpose. They can also receive a copy of their personal data in a commonly used electronic format.

```sql
-- PostgreSQL: Implement right to access functionality

-- Create function to export all data for a customer
CREATE OR REPLACE FUNCTION export_customer_data(p_customer_id UUID)
RETURNS JSONB AS $$
DECLARE
    customer_data JSONB;
    transaction_data JSONB;
    consent_data JSONB;
    purpose_data JSONB;
BEGIN
    -- Collect customer profile data
    SELECT to_jsonb(c) INTO customer_data
    FROM customers c
    WHERE c.customer_id = p_customer_id;
    
    -- Collect transaction history (with access control)
    SELECT jsonb_agg(to_jsonb(t)) INTO transaction_data
    FROM transactions t
    WHERE t.customer_id = p_customer_id
      AND t.created_at > NOW() - INTERVAL '7 days';  -- Limit to recent data
    
    -- Collect consent records
    SELECT jsonb_agg(to_jsonb(cp)) INTO consent_data
    FROM customer_data_purposes cp
    WHERE cp.customer_id = p_customer_id;
    
    -- Collect processing purposes
    SELECT jsonb_agg(
        jsonb_build_object(
            'purpose', dp.purpose_name,
            'legal_basis', dp.legal_basis,
            'consent_given', cdp.consent_given_at,
            'consent_withdrawn', cdp.consent_withdrawn_at
        )
    ) INTO purpose_data
    FROM customer_data_purposes cdp
    JOIN data_processing_purposes dp ON cdp.purpose_id = dp.purpose_id
    WHERE cdp.customer_id = p_customer_id;
    
    RETURN jsonb_build_object(
        'customer_id', p_customer_id,
        'exported_at', NOW(),
        'customer_profile', customer_data,
        'recent_transactions', transaction_data,
        'consents', consent_data,
        'processing_purposes', purpose_data
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permission to authenticated customers
GRANT EXECUTE ON FUNCTION export_customer_data(UUID) TO authenticated_user;

-- MySQL equivalent: Export customer data procedure
DELIMITER //

CREATE PROCEDURE export_customer_data(IN p_customer_id INT)
BEGIN
    SELECT 
        c.*,
        (SELECT JSON_ARRAYAGG(JSON_OBJECT(
            'transaction_id', t.transaction_id,
            'amount', t.amount,
            'date', t.created_at
        ))
        FROM transactions t
        WHERE t.customer_id = p_customer_id
          AND t.created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)) AS transactions,
        (SELECT JSON_ARRAYAGG(JSON_OBJECT(
            'purpose', dp.purpose_name,
            'consent_given', cdp.consent_given_at
        ))
        FROM customer_data_purposes cdp
        JOIN data_processing_purposes dp ON cdp.purpose_id = dp.purpose_id
        WHERE cdp.customer_id = p_customer_id) AS purposes
    FROM customers c
    WHERE c.customer_id = p_customer_id;
END //

DELIMITER ;
```

### Right to Erasure (Right to Be Forgotten)

GDPR establishes the right to erasure, requiring organizations to delete personal data when the data subject withdraws consent, the data is no longer necessary for its original purpose, or the data subject objects to processing. Database implementations must support complete data deletion while respecting legal retention requirements.

```sql
-- PostgreSQL: Implement right to erasure with cascading deletion

-- Create erasure request tracking table
CREATE TABLE erasure_requests (
    request_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID NOT NULL REFERENCES customers(customer_id),
    request_date TIMESTAMPTZ DEFAULT NOW(),
    completion_date TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'pending',
    reason TEXT
);

-- Create function to process erasure requests
CREATE OR REPLACE FUNCTION process_erasure_request(p_request_id UUID)
RETURNS VOID AS $$
DECLARE
    v_customer_id UUID;
    v_request RECORD;
BEGIN
    -- Get request details
    SELECT * INTO v_request
    FROM erasure_requests
    WHERE request_id = p_request_id;
    
    v_customer_id := v_request.customer_id;
    
    -- Delete from all related tables (respecting legal holds)
    -- Delete transactions without legal hold
    DELETE FROM transactions 
    WHERE customer_id = v_customer_id
      AND legal_hold = false;
    
    -- Delete consent records
    DELETE FROM customer_data_purposes 
    WHERE customer_id = v_customer_id;
    
    -- Delete encrypted identifiers
    DELETE FROM customer_identifiers 
    WHERE customer_id = v_customer_id;
    
    -- Anonymize customer record (preserve for analytics)
    UPDATE customers 
    SET 
        customer_reference = 'DELETED_' || customer_id::text,
        communication_preference = 'none',
        account_status = 'erased',
        erased_at = NOW()
    WHERE customer_id = v_customer_id;
    
    -- Update request status
    UPDATE erasure_requests 
    SET status = 'completed', completion_date = NOW()
    WHERE request_id = p_request_id;
    
    -- Log erasure for compliance
    INSERT INTO compliance_log (event_type, entity_type, entity_id, details)
    VALUES ('data_erasure', 'customer', v_customer_id, 
            jsonb_build_object('request_id', p_request_id));
END;
$$ LANGUAGE plpgsql;

-- Set up automatic erasure processing
CREATE OR REPLACE FUNCTION handle_erasure_requests()
RETURNS VOID AS $$
DECLARE
    v_request RECORD;
BEGIN
    FOR v_request IN
        SELECT request_id, customer_id
        FROM erasure_requests
        WHERE status = 'pending'
          AND request_date < NOW() - INTERVAL '30 days'
    LOOP
        PERFORM process_erasure_request(v_request.request_id);
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

### GDPR Compliance Configuration

```ini
# PostgreSQL: GDPR compliance configuration

# postgresql.conf

# Enable logging for compliance
logging_collector = on
log_directory = 'log'
log_filename = 'postgresql-%Y-%m-%d.log'
log_rotation_age = 1d
log_rotation_size = 100MB

# Log connections and queries for audit
log_connections = on
log_disconnections = on
log_statement = 'ddl'

# Security settings
password_encryption = 'scram-sha-256'
ssl = on
```

---

## HIPAA Compliance for Healthcare Data

### HIPAA Overview

The Health Insurance Portability and Accountability Act (HIPAA) establishes national standards for protecting Protected Health Information (PHI). HIPAA applies to covered entities (healthcare providers, health plans, healthcare clearinghouses) and their business associates. The Privacy Rule, Security Rule, and Breach Notification Rule define specific requirements for PHI protection.

The HIPAA Security Rule requires administrative, physical, and technical safeguards for electronic PHI (ePHI). Database systems storing ePHI must implement access controls, audit controls, integrity controls, and transmission security. The Security Rule specifies both required and addressable safeguards, with organizations required to implement all required safeguards and addressable safeguards based on risk assessment.

### ePHI Access Controls

HIPAA requires access controls limiting PHI access to authorized persons. Database implementations must support user authentication, automatic logoff, and emergency access procedures.

```sql
-- PostgreSQL: Implement HIPAA-compliant access controls

-- Create HIPAA-compliant user roles
CREATE ROLE hipaa_patient;
CREATE ROLE hipaa_physician;
CREATE ROLE hipaa_nurse;
CREATE ROLE hipaa_billing;
CREATE ROLE hipaa_auditor;
CREATE ROLE hipaa_admin;

-- Grant minimum necessary access
-- Patient can view their own records
GRANT SELECT ON patient_records TO hipaa_patient;
GRANT SELECT ON patient_appointments TO hipaa_patient;

-- Physician can view patients in their care
CREATE POLICY physician_patient_access ON patient_records
    FOR SELECT
    USING (
        physician_id IN (
            SELECT physician_id 
            FROM physician_patient_assignments 
            WHERE patient_id = patient_records.patient_id
              AND physician_id = current_user
        )
    );

-- Nurse has broader but limited access
GRANT SELECT ON patient_records TO hipaa_nurse;
GRANT SELECT ON patient_vitals TO hipaa_nurse;
GRANT UPDATE (notes, vital_signs) ON patient_records TO hipaa_nurse;

-- Billing has access to billing-relevant fields only
GRANT SELECT (patient_id, billing_code, amount, service_date) 
ON medical_claims TO hipaa_billing;

-- Auditor has read-only access to all PHI for compliance audits
GRANT SELECT ON ALL TABLES IN SCHEMA medical TO hipaa_auditor;

-- Enable row-level security
ALTER TABLE patient_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE patient_vitals ENABLE ROW LEVEL SECURITY;
ALTER TABLE medical_claims ENABLE ROW LEVEL SECURITY;

-- Automatic session timeout (application-level implementation required)
-- Application should implement:
-- - Automatic logoff after 15 minutes of inactivity
-- - Emergency access procedures with break-glass accounts
-- - Unique user identification
```

### Audit Controls for HIPAA

HIPAA requires audit controls that record and examine activity in systems containing ePHI. Database audit trails must record access, modifications, and deletions of PHI.

```sql
-- PostgreSQL: HIPAA-compliant audit logging

-- Create HIPAA audit log with required fields
CREATE TABLE hipaa_audit_log (
    audit_id BIGSERIAL PRIMARY KEY,
    audit_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id VARCHAR(100) NOT NULL,
    user_role VARCHAR(50),
    action_type VARCHAR(50) NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    record_id TEXT,
    phi_accessed BOOLEAN DEFAULT true,
    ip_address INET,
    session_id VARCHAR(100),
    application_name VARCHAR(100),
    description TEXT
);

-- Create indexes for efficient HIPAA audit queries
CREATE INDEX idx_hipaa_audit_timestamp ON hipaa_audit_log(audit_timestamp);
CREATE INDEX idx_hipaa_audit_user ON hipaa_audit_log(user_id);
CREATE INDEX idx_hipaa_audit_table ON hipaa_audit_log(table_name);
CREATE INDEX idx_hipaa_audit_phi ON hipaa_audit_log(phi_accessed);

-- HIPAA audit trigger function
CREATE OR REPLACE FUNCTION hipaa_audit_trigger()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO hipaa_audit_log (
        user_id,
        user_role,
        action_type,
        table_name,
        record_id,
        ip_address,
        session_id,
        application_name
    ) VALUES (
        current_user,
        current_setting('app.user_role', true),
        TG_OP,
        TG_TABLE_NAME,
        row_to_json(NEW)->>id::text,
        inet_client_addr(),
        current_setting('app.session_id', true),
        current_setting('application_name', true)
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to all PHI tables
CREATE TRIGGER hipaa_audit_patient_records
AFTER INSERT OR UPDATE OR DELETE ON patient_records
FOR EACH ROW EXECUTE FUNCTION hipaa_audit_trigger();

CREATE TRIGGER hipaa_audit_patient_vitals
AFTER INSERT OR UPDATE OR DELETE ON patient_vitals
FOR EACH ROW EXECUTE FUNCTION hipaa_audit_trigger();

CREATE TRIGGER hipaa_audit_prescriptions
AFTER INSERT OR UPDATE OR DELETE ON prescriptions
FOR EACH ROW EXECUTE FUNCTION hipaa_audit_trigger();
```

### PHI Encryption Requirements

HIPAA requires encryption of ePHI at rest and in transit. Organizations must implement technical safeguards for data confidentiality during storage and transmission.

```sql
-- PostgreSQL: PHI encryption implementation

-- Enable tablespace encryption
CREATE TABLESPACE phi_encrypted 
LOCATION '/data/phi_encrypted' 
WITH (encrypted = true);

-- Move PHI tables to encrypted tablespace
ALTER TABLE patient_records 
SET TABLESPACE phi_encrypted;

ALTER TABLE prescriptions 
SET TABLESPACE phi_encrypted;

-- Column-level encryption for highly sensitive PHI
CREATE EXTENSION pgcrypto;

-- Encrypt specific PHI columns
ALTER TABLE patient_records 
ADD COLUMN diagnosis_encrypted BYTEA,
ADD COLUMN ssn_encrypted BYTEA,
ADD COLUMN payment_info_encrypted BYTEA;

-- Encryption key management (use HSM in production)
CREATE OR REPLACE FUNCTION encrypt_phi(p_data TEXT, p_key BYTEA)
RETURNS BYTEA AS $$
BEGIN
    RETURN pgp_sym_encrypt(p_data, p_key, 
        'compress-algo=1, cipher-algo=aes-256-cbc');
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION decrypt_phi(p_encrypted BYTEA, p_key BYTEA)
RETURNS TEXT AS $$
BEGIN
    RETURN pgp_sym_decrypt(p_encrypted::BYTEA, p_key,
        'compress-algo=1, cipher-algo=aes-256-cbc');
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Restrict access to encryption functions
REVOKE ALL ON FUNCTION encrypt_phi(TEXT, BYTEA) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION encrypt_phi(TEXT, BYTEA) TO hipaa_physician;
GRANT EXECUTE ON FUNCTION decrypt_phi(BYTEA, BYTEA) TO hipaa_physician;
```

---

## SOC 2 Compliance Requirements

### SOC 2 Overview

SOC 2 (Service Organization Control 2) is a compliance framework developed by the American Institute of Certified Public Accountants (AICPA). SOC 2 defines criteria for managing customer data based on five trust service criteria: security, availability, processing integrity, confidentiality, and privacy. SOC 2 reports provide assurance about an organization's controls relevant to security and privacy.

Type II SOC 2 reports examine control effectiveness over a period (typically 6-12 months), while Type I reports evaluate controls at a specific point in time. Database implementations must support evidence collection for all five trust service criteria.

### Security Controls for SOC 2

SOC 2 security criteria require controls protecting against unauthorized access. Database implementations must demonstrate access control, encryption, logging, and monitoring.

```sql
-- PostgreSQL: SOC 2 compliance security controls

-- 1. Access Control - Unique user identification
CREATE TABLE db_users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    employee_id VARCHAR(20) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) NOT NULL,
    department VARCHAR(50),
    job_title VARCHAR(100),
    access_level VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    terminated_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT true
);

-- Track user access assignments
CREATE TABLE user_access (
    access_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES db_users(user_id),
    database_name VARCHAR(50) NOT NULL,
    role_name VARCHAR(50) NOT NULL,
    granted_by VARCHAR(50) NOT NULL,
    granted_at TIMESTAMPTZ DEFAULT NOW(),
    expiration_date TIMESTAMPTZ,
    business_justification TEXT
);

-- 2. Password policy enforcement
ALTER ROLE ALL VALID UNTIL '90 days';
ALTER ROLE all WITH PASSWORD MINLENGTH 12;

-- Set password complexity requirements
-- (Implementation varies by database system)
-- PostgreSQL: Use passwordcheck extension if available

-- 3. Session management
-- Application should implement:
-- - Session timeout after 30 minutes of inactivity
-- - Concurrent session limits
-- - Session token rotation
```

### Availability and Processing Integrity

SOC 2 availability criteria require systems operate as committed and data remains available. Processing integrity requires accurate and complete data processing.

```sql
-- PostgreSQL: High availability configuration for SOC 2

-- Configure synchronous replication for high availability
ALTER SYSTEM SET synchronous_commit = on;
ALTER SYSTEM SET synchronous_standby_names = 'standby1,standby2';

-- Create availability monitoring function
CREATE OR REPLACE FUNCTION check_database_availability()
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
    v_replication_lag NUMERIC;
    v_connection_count INT;
BEGIN
    -- Check if database is accepting connections
    PERFORM 1 FROM pg_database WHERE datname = current_database();
    
    -- Get replication lag
    SELECT COALESCE(MAX(pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn)), 0)
    INTO v_replication_lag
    FROM pg_stat_replication;
    
    -- Get connection count
    SELECT COUNT(*) INTO v_connection_count
    FROM pg_stat_activity
    WHERE state = 'active';
    
    v_result := jsonb_build_object(
        'timestamp', NOW(),
        'database_available', true,
        'replication_lag_bytes', v_replication_lag,
        'active_connections', v_connection_count,
        'status', CASE 
            WHEN v_replication_lag > 100000000 THEN 'degraded'
            ELSE 'healthy'
        END
    );
    
    RETURN v_result;
END;
$$ LANGUAGE plpgsql;

-- Integrity checking
CREATE OR REPLACE FUNCTION verify_data_integrity(p_table_name TEXT)
RETURNS TABLE (check_name TEXT, status TEXT, details TEXT) AS $$
BEGIN
    RETURN QUERY
    EXECUTE format(
        'SELECT ''null_check''::TEXT, 
                CASE WHEN COUNT(*) > 0 THEN ''fail'' ELSE ''pass'' END,
                COUNT(*)::TEXT
         FROM %I WHERE %s IS NULL',
        p_table_name,
        (SELECT column_name 
         FROM information_schema.columns 
         WHERE table_name = p_table_name 
         AND is_nullable = 'NO' LIMIT 1)
    );
END;
$$ LANGUAGE plpgsql;
```

---

## PCI DSS for Payment Data

### PCI DSS Overview

The Payment Card Industry Data Security Standard (PCI DSS) applies to organizations handling payment card data. PCI DSS establishes requirements for securing cardholder data, including network security, access controls, encryption, vulnerability management, and monitoring. Database systems storing cardholder data must comply with PCI DSS requirements.

PCI DSS defines 12 main requirements organized into six control objectives: build and maintain secure networks, protect cardholder data, maintain vulnerability management programs, implement strong access controls, regularly monitor networks, and maintain information security policies. Non-compliance can result in significant fines and loss of payment processing capabilities.

### Cardholder Data Protection

PCI DSS requires protection of stored cardholder data through encryption, truncation, masking, and access controls. Databases must never store full primary account numbers (PAN) unless encrypted.

```sql
-- PostgreSQL: PCI DSS-compliant cardholder data storage

-- Create cardholder data table with proper segmentation
CREATE TABLE cardholder_data (
    card_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID NOT NULL REFERENCES customers(customer_id),
    -- Tokenized PAN - never store actual card number
    pan_token VARCHAR(50) UNIQUE NOT NULL,
    -- Last 4 digits for display purposes only
    last_four VARCHAR(4) NOT NULL,
    -- Card brand for processing
    card_brand VARCHAR(20) NOT NULL,
    -- Expiry date (not sensitive)
    expiry_month INT NOT NULL,
    expiry_year INT NOT NULL,
    -- Token from payment processor
    payment_token VARCHAR(255) NOT NULL,
    -- Cardholder name at time of transaction
    cardholder_name VARCHAR(100),
    -- Tokenization method used
    tokenization_method VARCHAR(20) DEFAULT 'processor',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    token_expires_at TIMESTAMPTZ
);

-- Encryption at rest for entire table
ALTER TABLE cardholder_data SET TABLESPACE pci_compliant_storage;

-- Access control - restrict PAN token access
REVOKE ALL ON cardholder_data FROM PUBLIC;
GRANT SELECT (card_id, customer_id, last_four, card_brand, expiry_month, 
              expiry_year) TO payment_service;
GRANT SELECT (card_id, customer_id, last_four, card_brand, 
              expiry_month, expiry_year, cardholder_name) 
TO customer_service;
GRANT ALL ON cardholder_data TO pci_admin;

-- PCI DSS 3.3: Mask PAN when displayed
CREATE VIEW cardholder_data_masked AS
SELECT 
    card_id,
    customer_id,
    '****-****-****-' || last_four AS masked_pan,
    card_brand,
    expiry_month,
    expiry_year,
    CASE 
        WHEN cardholder_name IS NOT NULL 
        THEN SUBSTRING(cardholder_name, 1, 1) || '***'
        ELSE NULL 
    END AS masked_name
FROM cardholder_data;

GRANT SELECT ON cardholder_data_masked TO customer_service;

-- Transaction logging (PCI DSS 10.2)
CREATE TABLE payment_transactions (
    transaction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    card_id UUID REFERENCES cardholder_data(card_id),
    transaction_type VARCHAR(20) NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    authorization_code VARCHAR(50),
    response_code VARCHAR(10),
    processed_by VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT
);

-- PCI DSS 10.1: Log all access to system components
CREATE TABLE pci_audit_log (
    audit_id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_id VARCHAR(100),
    action VARCHAR(50),
    table_name VARCHAR(100),
    record_id TEXT,
    ip_address INET,
    success BOOLEAN DEFAULT true
);
```

### PCI DSS Access Control Requirements

PCI DSS requires restriction of cardholder data access to authorized personnel, unique user IDs, and automatic session timeout.

```sql
-- PostgreSQL: PCI DSS access control implementation

-- Create PCI-specific roles
CREATE ROLE pci_payment_processor;
CREATE ROLE pci_refund_processor;
CREATE ROLE pci_reporting;
CREATE ROLE pci_auditor;
CREATE ROLE pci_admin;

-- Principle of least privilege access
-- PCI DSS 7.1: Restrict access to cardholder data
GRANT SELECT, INSERT ON payment_transactions TO pci_payment_processor;
GRANT SELECT ON payment_transactions TO pci_refund_processor;
GRANT SELECT (transaction_id, card_id, amount, created_at) 
ON payment_transactions TO pci_reporting;
GRANT SELECT ON ALL TABLES IN SCHEMA payments TO pci_auditor;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA payments TO pci_admin;

-- PCI DSS 8.1: Unique user identification
-- (Implemented at application level with database integration)
CREATE TABLE pci_user_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(100) NOT NULL,
    login_time TIMESTAMPTZ DEFAULT NOW(),
    last_activity TIMESTAMPTZ DEFAULT NOW(),
    ip_address INET,
    is_active BOOLEAN DEFAULT true,
    -- Automatic logout after 15 minutes of inactivity
    CONSTRAINT session_timeout 
    CHECK (last_activity > NOW() - INTERVAL '15 minutes')
);

-- PCI DSS 8.2: Password requirements
-- Database-level password policy
ALTER SYSTEM SET password_encryption = 'scram-sha-256';

-- Grant administrative access by individual user
-- PCI DSS 8.5: Unique IDs for each user
CREATE ROLE individual_admin WITH LOGIN;
GRANT pci_admin TO individual_admin;
```

---

## Data Retention and Deletion Policies

### Establishing Retention Policies

Compliance frameworks require organizations to retain data only as long as necessary for the purposes for which it was collected. Data retention policies must specify retention periods for different data categories, legal holds that suspend deletion, and secure destruction procedures.

```sql
-- PostgreSQL: Data retention management

-- Create retention policy table
CREATE TABLE data_retention_policies (
    policy_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name VARCHAR(100) NOT NULL,
    data_category VARCHAR(50) NOT NULL,
    retention_period INTERVAL NOT NULL,
    legal_hold_period INTERVAL,
    deletion_method VARCHAR(20) DEFAULT 'permanent_delete',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(table_name)
);

-- Define retention policies
INSERT INTO data_retention_policies (table_name, data_category, retention_period)
VALUES 
    ('login_logs', 'authentication', '1 year'),
    ('audit_logs', 'compliance', '7 years'),
    ('transaction_logs', 'financial', '7 years'),
    ('customer_data', 'personal', '5 years'),
    ('session_data', 'operational', '30 days'),
    ('temporary_files', 'temp', '7 days');

-- Create legal hold table
CREATE TABLE legal_holds (
    hold_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(50) NOT NULL,
    entity_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    hold_start_date DATE NOT NULL,
    hold_end_date DATE,
    legal_case VARCHAR(100),
    placed_by VARCHAR(100) NOT NULL,
    placed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create retention enforcement function
CREATE OR REPLACE FUNCTION enforce_retention_policies()
RETURNS TABLE (table_name TEXT, records_deleted BIGINT) AS $$
DECLARE
    v_policy RECORD;
    v_deleted BIGINT := 0;
BEGIN
    FOR v_policy IN
        SELECT table_name, retention_period
        FROM data_retention_policies
    LOOP
        EXECUTE format(
            'DELETE FROM %I 
             WHERE created_at < NOW() - $1
               AND id NOT IN (
                   SELECT entity_id FROM legal_holds 
                   WHERE entity_type = $2
               )',
            v_policy.table_name
        ) 
        USING v_policy.retention_period, v_policy.table_name;
        
        GET DIAGNOSTICS v_deleted = ROW_COUNT;
        
        IF v_deleted > 0 THEN
            RETURN NEXT TABLE (v_policy.table_name, v_deleted);
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Schedule retention enforcement (run daily via cron)
-- 0 2 * * * psql -c "SELECT enforce_retention_policies()"
```

### Secure Data Deletion

When data reaches the end of its retention period, it must be securely deleted. Simple deletion may leave recoverable data on disk. Secure deletion overwrites data or uses database features for permanent removal.

```sql
-- PostgreSQL: Secure deletion implementation

-- Create deletion log for compliance
CREATE TABLE deletion_log (
    deletion_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name VARCHAR(100) NOT NULL,
    record_id TEXT NOT NULL,
    deletion_timestamp TIMESTAMPTZ DEFAULT NOW(),
    deletion_method VARCHAR(50),
    deleted_by VARCHAR(100),
    retention_period_satisfied BOOLEAN,
    legal_hold_applied BOOLEAN,
    notes TEXT
);

-- Secure deletion function
CREATE OR REPLACE FUNCTION secure_delete(
    p_table_name TEXT,
    p_record_id UUID,
    p_deletion_method VARCHAR(50) DEFAULT 'permanent'
)
RETURNS VOID AS $$
DECLARE
    v_record_exists BOOLEAN;
    v_legal_hold BOOLEAN;
    v_sql TEXT;
BEGIN
    -- Check for legal hold
    SELECT EXISTS (
        SELECT 1 FROM legal_holds 
        WHERE entity_type = p_table_name 
          AND entity_id = p_record_id::text
          AND (hold_end_date IS NULL OR hold_end_date > CURRENT_DATE)
    ) INTO v_legal_hold;
    
    IF v_legal_hold THEN
        RAISE EXCEPTION 'Cannot delete record under legal hold: %', p_record_id;
    END IF;
    
    -- Log deletion before executing
    INSERT INTO deletion_log (
        table_name,
        record_id,
        deletion_method,
        deleted_by,
        retention_period_satisfied,
        legal_hold_applied
    ) VALUES (
        p_table_name,
        p_record_id,
        p_deletion_method,
        current_user,
        true,
        false
    );
    
    -- Execute deletion
    v_sql := format(
        'DELETE FROM %I WHERE id = $1',
        p_table_name
    );
    EXECUTE v_sql USING p_record_id;
END;
$$ LANGUAGE plpgsql;

-- For encrypted data, also securely erase encryption keys
CREATE TABLE encryption_key_archive (
    key_id UUID PRIMARY KEY,
    key_hash VARCHAR(255) NOT NULL,
    encrypted_data_reference TEXT,
    key_created_at TIMESTAMPTZ,
    key_archived_at TIMESTAMPTZ DEFAULT NOW(),
    key_destroyed_at TIMESTAMPTZ
);

-- Cryptographic erasure function
CREATE OR REPLACE FUNCTION cryptographic_erase(p_key_id UUID)
RETURNS VOID AS $$
BEGIN
    -- Overwrite key reference
    UPDATE encryption_key_archive
    SET key_hash = 'DESTROYED_' || key_hash,
        key_destroyed_at = NOW()
    WHERE key_id = p_key_id;
    
    -- Note: In production, also delete from HSM
END;
$$ LANGUAGE plpgsql;
```

---

## Audit Trail Requirements

### Comprehensive Audit Logging

Compliance frameworks require audit trails capturing who accessed data, what actions were performed, when actions occurred, and what data was affected. Audit logs must be tamper-proof and retained according to regulatory requirements.

```sql
-- PostgreSQL: Comprehensive compliance audit trail

-- Create unified audit log table
CREATE TABLE compliance_audit_log (
    audit_id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_timestamp TIMESTAMPTZ,
    event_type VARCHAR(50) NOT NULL,
    event_category VARCHAR(30),
    user_id VARCHAR(100),
    user_role VARCHAR(50),
    session_id VARCHAR(100),
    client_ip INET,
    client_port INT,
    database_name VARCHAR(50),
    schema_name VARCHAR(50),
    table_name VARCHAR(100),
    column_name VARCHAR(100),
    record_id TEXT,
    operation VARCHAR(10),
    old_value JSONB,
    new_value JSONB,
    sql_statement TEXT,
    application_name VARCHAR(100),
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    -- Tamper-proofing
    hash_value VARCHAR(64)
);

-- Create indexes for efficient querying
CREATE INDEX idx_audit_timestamp ON compliance_audit_log(timestamp);
CREATE INDEX idx_audit_user ON compliance_audit_log(user_id);
CREATE INDEX idx_audit_table ON compliance_audit_log(table_name);
CREATE INDEX idx_audit_event_type ON compliance_audit_log(event_type);

-- Create audit hash chain for tamper detection
CREATE OR REPLACE FUNCTION calculate_audit_hash(
    p_audit_id BIGINT,
    p_timestamp TIMESTAMPTZ,
    p_event_type VARCHAR,
    p_user_id VARCHAR,
    p_table_name VARCHAR,
    p_record_id TEXT
)
RETURNS TEXT AS $$
BEGIN
    RETURN encode(
        sha256(
            concat(
                p_audit_id,
                p_timestamp,
                p_event_type,
                p_user_id,
                p_table_name,
                p_record_id
            )::bytea
        ),
        'hex'
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create function to verify audit log integrity
CREATE OR REPLACE FUNCTION verify_audit_integrity(
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ
)
RETURNS TABLE (is_valid BOOLEAN, invalid_records BIGINT) AS $$
DECLARE
    v_invalid_count BIGINT := 0;
    v_record RECORD;
BEGIN
    FOR v_record IN
        SELECT audit_id, hash_value
        FROM compliance_audit_log
        WHERE timestamp BETWEEN p_start_time AND p_end_time
    LOOP
        -- Verify each record's hash
        -- (Simplified - in production use cryptographic signatures)
        IF v_record.hash_value IS NULL THEN
            v_invalid_count := v_invalid_count + 1;
        END IF;
    END LOOP;
    
    RETURN QUERY SELECT (v_invalid_count = 0), v_invalid_count;
END;
$$ LANGUAGE plpgsql;

-- Configure log retention
-- postgresql.conf
-- log_filename = 'compliance-audit-%Y-%m-%d.log'
-- log_rotation_age = 1d
-- log_statement = 'all'
```

### Log Protection and Retention

Audit logs themselves must be protected from unauthorized access or modification. Retention periods vary by regulation but typically span 3-7 years for compliance data.

```sql
-- PostgreSQL: Secure audit log management

-- Create dedicated audit schema
CREATE SCHEMA audit;
ALTER SCHEMA audit OWNER TO audit_admin;

-- Move audit tables to separate tablespace
CREATE TABLESPACE audit_tb LOCATION '/data/audit';

-- Create protected audit table
CREATE TABLE audit.audit_events (
    event_id BIGSERIAL,
    event_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_data JSONB NOT NULL,
    -- Immutability marker
    created_at TIMESTAMPTZ DEFAULT NOW()
) TABLESPACE audit_tb;

-- Prevent modification of audit records
CREATE OR REPLACE FUNCTION prevent_audit_modification()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Audit records cannot be modified or deleted';
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER protect_audit_events
BEFORE UPDATE OR DELETE ON audit.audit_events
FOR EACH ROW EXECUTE FUNCTION prevent_audit_modification();

-- Grant minimal access
REVOKE ALL ON audit.audit_events FROM PUBLIC;
GRANT INSERT ON audit.audit_events TO database_system;
GRANT SELECT ON audit.audit_events TO compliance_auditor;
```

---

## Privacy-Preserving Techniques

### Differential Privacy

Differential privacy adds statistical noise to query results, enabling aggregate analysis while protecting individual records. This technique is valuable for analytics and machine learning applications involving sensitive data.

```sql
-- PostgreSQL: Implement differential privacy for analytics

-- Create function to add Laplace noise for differential privacy
CREATE OR REPLACE FUNCTION add_laplace_noise(
    p_value NUMERIC,
    p_sensitivity NUMERIC,
    p_epsilon NUMERIC DEFAULT 1.0
)
RETURNS NUMERIC AS $$
DECLARE
    v_scale NUMERIC;
    v_noise NUMERIC;
BEGIN
    -- Scale parameter for Laplace distribution
    v_scale := p_sensitivity / p_epsilon;
    
    -- Generate Laplace noise using inverse CDF method
    v_noise := v_scale * ln(1 - 2 * random()) * 
               CASE WHEN random() < 0.5 THEN -1 ELSE 1 END;
    
    RETURN p_value + v_noise;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Privacy-preserving count function
CREATE OR REPLACE FUNCTION dp_count(
    p_query TEXT,
    p_epsilon NUMERIC DEFAULT 1.0
)
RETURNS NUMERIC AS $$
DECLARE
    v_count NUMERIC;
    v_result NUMERIC;
BEGIN
    -- Execute query to get exact count
    EXECUTE p_query INTO v_count;
    
    -- Add noise based on sensitivity (1 for count)
    v_result := add_laplace_noise(v_count::NUMERIC, 1.0, p_epsilon);
    
    -- Ensure non-negative result
    RETURN GREATEST(0, ROUND(v_result));
END;
$$ LANGUAGE plpgsql;

-- Example: Count users by region with privacy
CREATE VIEW region_user_counts_dp AS
SELECT 
    region,
    dp_count(
        format(
            'SELECT COUNT(*) FROM users WHERE region = ''%s''',
            region
        ),
        1.0  -- Epsilon parameter controls privacy/accuracy trade-off
    ) AS estimated_count
FROM (SELECT DISTINCT region FROM users) AS regions;
```

### Data Anonymization for Testing

Test and development environments must not contain real production data. Anonymization techniques transform sensitive data while preserving data characteristics necessary for testing.

```sql
-- PostgreSQL: Data anonymization functions

CREATE EXTENSION faker;

-- Create anonymization functions for sensitive data
CREATE OR REPLACE FUNCTION anonymize_email(p_email TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN 'user_' || 
           md5(p_email::bytea)::varchar(8) || 
           '@example.com';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION anonymize_name(p_name TEXT)
RETURNS TEXT AS $$
DECLARE
    v_first_names TEXT[] := ARRAY['John', 'Jane', 'Alice', 'Bob', 'Charlie'];
    v_last_names TEXT[] := ARRAY['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'];
BEGIN
    RETURN v_first_names[1 + floor(random() * 5)] || ' ' ||
           v_last_names[1 + floor(random() * 5)];
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION anonymize_ssn(p_ssn TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN 'XXX-XX-' || 
           LPAD(floor(random() * 10000)::INT::TEXT, 4, '0');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION anonymize_phone(p_phone TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN '(555) ' || 
           LPAD(floor(random() * 900 + 100)::INT::TEXT, 3, '0') || '-' ||
           LPAD(floor(random() * 9000)::INT::TEXT, 4, '0');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create anonymized test dataset
CREATE TABLE customers_test AS
SELECT 
    customer_id,
    anonymize_name(first_name) AS first_name,
    anonymize_name(last_name) AS last_name,
    anonymize_email(email) AS email,
    anonymize_phone(phone) AS phone,
    anonymize_ssn(ssn) AS ssn,
    city,
    state,
    zip_code
FROM customers
WHERE 1=1;

-- Anonymize transaction amounts while preserving distributions
CREATE TABLE transactions_test AS
SELECT 
    transaction_id,
    customer_id,
    -- Add noise to amounts while preserving statistical properties
    ROUND(amount * (0.9 + random() * 0.2), 2) AS amount,
    transaction_date,
    transaction_type
FROM transactions;
```

---

## Compliance Checklist

### GDPR Compliance Checklist

| Requirement | Description | Implementation Status |
|-------------|-------------|----------------------|
| Lawful basis for processing | Document legal basis for each data processing activity | [ ] |
| Consent management | Implement consent collection and management | [ ] |
| Data minimization | Collect only necessary personal data | [ ] |
| Right to access | Enable data subject data export | [ ] |
| Right to erasure | Implement data deletion functionality | [ ] |
| Data portability | Provide data in machine-readable format | [ ] |
| Privacy by design | Implement data protection by default | [ ] |
| Data protection impact assessment | Conduct DPIA for high-risk processing | [ ] |
| Breach notification | Establish 72-hour breach notification process | [ ] |
| Data protection officer | Appoint DPO if required | [ ] |

### HIPAA Compliance Checklist

| Requirement | Description | Implementation Status |
|-------------|-------------|----------------------|
| Risk analysis | Conduct comprehensive risk analysis | [ ] |
| Access controls | Implement unique user identification | [ ] |
| Audit controls | Enable comprehensive audit logging | [ ] |
| Integrity controls | Protect ePHI from unauthorized modification | [ ] |
| Transmission security | Encrypt ePHI in transit | [ ] |
| Encryption at rest | Encrypt ePHI at rest | [ ] |
| Emergency access | Establish break-glass procedures | [ ] |
| Workforce training | Train workforce on HIPAA requirements | [ ] |
| Business associate agreements | Execute BAAs with vendors | [ ] |
| Breach notification | Establish breach notification procedures | [ ] |

### PCI DSS Compliance Checklist

| Requirement | Description | Implementation Status |
|-------------|-------------|----------------------|
| Network security | Install and maintain firewall configuration | [ ] |
| Cardholder data protection | Protect stored cardholder data | [ ] |
| Vulnerability management | Maintain secure systems and applications | [ ] |
| Access control | Restrict access to cardholder data | [ ] |
| Monitoring and testing | Track and monitor network access | [ ] |
| Information security | Maintain information security policy | [ ] |
| Cardholder data encryption | Encrypt transmission of cardholder data | [ ] |
| Unique IDs | Assign unique IDs to each user | [ ] |
| Log retention | Maintain audit logs | [ ] |
| Regular testing | Conduct regular security testing | [ ] |

---

## Conclusion

Database compliance requires careful implementation of technical controls aligned with specific regulatory requirements. This guide has covered the major compliance frameworks including GDPR, HIPAA, SOC 2, and PCI DSS, providing SQL implementations, configuration examples, and procedural requirements necessary for building compliant database systems.

Key implementation patterns include encryption for data at rest and in transit, comprehensive audit logging for accountability, role-based access controls enforcing least privilege, data minimization and retention policies, and privacy-preserving techniques for analytics and testing.

Compliance is an ongoing process requiring regular audits, policy reviews, and updates as regulations evolve. Organizations should establish compliance monitoring programs, conduct periodic risk assessments, and maintain documentation demonstrating compliance with applicable frameworks.

---

## References

- GDPR: Regulation (EU) 2016/679
- HIPAA: 45 CFR Part 164
- PCI DSS v4.0
- SOC 2 Type II Examination Criteria
- NIST SP 800-53 Security and Privacy Controls
- OWASP Top 10
