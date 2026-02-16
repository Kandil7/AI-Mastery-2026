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

-- Transaction logging (PCI DSS requirements)
CREATE TABLE card_transaction_log (
    log_id BIGSERIAL PRIMARY KEY,
    card_id UUID NOT NULL REFERENCES cardholder_data(card_id),
    transaction_id VARCHAR(50) NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    merchant_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    action_type VARCHAR(20) NOT NULL,
    user_id VARCHAR(100),
    ip_address INET,
    success BOOLEAN NOT NULL
);

-- Index for PCI DSS audit requirements
CREATE INDEX idx_card_transaction_log_timestamp ON card_transaction_log(timestamp);
CREATE INDEX idx_card_transaction_log_card_id ON card_transaction_log(card_id);
CREATE INDEX idx_card_transaction_log_merchant_id ON card_transaction_log(merchant_id);
```

## Data Retention and Deletion Policies

### Legal Retention Requirements

Different jurisdictions and industries have varying data retention requirements. Database systems must support configurable retention policies that comply with legal requirements.

```sql
-- PostgreSQL: Data retention policy implementation

-- Create retention policy table
CREATE TABLE data_retention_policies (
    policy_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    data_category VARCHAR(50) NOT NULL,
    retention_period INTERVAL NOT NULL,
    legal_basis TEXT NOT NULL,
    auto_delete BOOLEAN DEFAULT true,
    last_reviewed TIMESTAMPTZ DEFAULT NOW()
);

-- Example policies
INSERT INTO data_retention_policies (data_category, retention_period, legal_basis, auto_delete)
VALUES
('customer_personal_data', INTERVAL '7 years', 'GDPR Article 6(1)(c)', true),
('financial_transactions', INTERVAL '10 years', 'IRS Regulations', true),
('medical_records', INTERVAL '21 years', 'HIPAA', true),
('system_logs', INTERVAL '90 days', 'Internal Policy', true);

-- Automated data deletion function
CREATE OR REPLACE FUNCTION apply_retention_policy()
RETURNS VOID AS $$
DECLARE
    v_policy RECORD;
    v_deletion_count INT;
BEGIN
    FOR v_policy IN
        SELECT * FROM data_retention_policies
        WHERE auto_delete = true
    LOOP
        -- Execute deletion for each policy
        EXECUTE format(
            'DELETE FROM %I WHERE created_at < NOW() - %L',
            v_policy.data_category,
            v_policy.retention_period
        ) INTO v_deletion_count;

        -- Log deletion for compliance
        INSERT INTO compliance_log (event_type, entity_type, details)
        VALUES ('retention_deletion', v_policy.data_category,
                format('Deleted %s records older than %s', v_deletion_count, v_policy.retention_period));
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Schedule daily retention policy execution
-- CREATE EXTENSION IF NOT EXISTS pg_cron;
-- SELECT cron.schedule('0 2 * * *', $$SELECT apply_retention_policy()$$);
```

## Audit Trail Requirements

### Comprehensive Audit Logging

Regulatory frameworks require comprehensive audit trails that capture all significant database activities. Audit logs must be immutable, tamper-evident, and retained for specified periods.

```sql
-- PostgreSQL: Comprehensive audit trail implementation

-- Create immutable audit log table
CREATE TABLE compliance_audit_log (
    audit_id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id TEXT NOT NULL,
    user_id VARCHAR(100) NOT NULL,
    user_role VARCHAR(50),
    ip_address INET,
    session_id VARCHAR(100),
    application_name VARCHAR(100),
    action_details JSONB,
    -- Immutable hash for tamper detection
    previous_hash BYTEA,
    current_hash BYTEA
);

-- Create trigger for hash chain
CREATE OR REPLACE FUNCTION audit_log_hash_trigger()
RETURNS TRIGGER AS $$
DECLARE
    v_previous_hash BYTEA;
BEGIN
    -- Get previous hash
    SELECT current_hash INTO v_previous_hash
    FROM compliance_audit_log
    ORDER BY audit_id DESC
    LIMIT 1;

    -- Calculate new hash
    NEW.previous_hash = v_previous_hash;
    NEW.current_hash = digest(
        CONCAT(
            NEW.event_type, '|',
            NEW.entity_type, '|',
            NEW.entity_id, '|',
            NEW.user_id, '|',
            NEW.created_at::text, '|',
            COALESCE(NEW.action_details::text, '')
        ),
        'sha256'
    );

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER audit_log_hash
BEFORE INSERT ON compliance_audit_log
FOR EACH ROW EXECUTE FUNCTION audit_log_hash_trigger();

-- Create audit views for different compliance requirements
CREATE VIEW gdpr_audit_view AS
SELECT
    audit_id,
    created_at,
    event_type,
    entity_type,
    entity_id,
    user_id,
    action_details->>'subject_id' AS subject_id,
    action_details->>'purpose' AS purpose
FROM compliance_audit_log
WHERE event_type IN ('data_access', 'data_modification', 'data_deletion');

CREATE VIEW hipaa_audit_view AS
SELECT
    audit_id,
    created_at,
    event_type,
    entity_type,
    entity_id,
    user_id,
    user_role,
    action_details->>'phi_accessed' AS phi_accessed,
    action_details->>'access_reason' AS access_reason
FROM compliance_audit_log
WHERE entity_type LIKE '%phi%' OR event_type LIKE '%phi%';
```

## Privacy-Preserving Techniques

### Differential Privacy

Differential privacy provides mathematical guarantees for privacy preservation in data analysis. Database systems can implement differential privacy for analytical queries.

```sql
-- PostgreSQL: Differential privacy implementation

-- Create function for differentially private aggregation
CREATE OR REPLACE FUNCTION dp_sum(p_column TEXT, p_epsilon NUMERIC, p_sensitivity NUMERIC)
RETURNS NUMERIC AS $$
DECLARE
    v_raw_sum NUMERIC;
    v_noise NUMERIC;
BEGIN
    -- Calculate raw sum
    EXECUTE format('SELECT SUM(%I) FROM analytics_table', p_column)
    INTO v_raw_sum;

    -- Add Laplace noise for differential privacy
    -- Laplace distribution: scale = sensitivity / epsilon
    v_noise = random() * (2 * p_sensitivity / p_epsilon) - (p_sensitivity / p_epsilon);

    RETURN v_raw_sum + v_noise;
END;
$$ LANGUAGE plpgsql;

-- Example usage
-- SELECT dp_sum('revenue', 1.0, 1000.0) AS dp_revenue;
```

### Homomorphic Encryption

Homomorphic encryption allows computation on encrypted data without decryption. While computationally expensive, it provides strong privacy guarantees.

```sql
-- PostgreSQL: Homomorphic encryption example (conceptual)
-- Note: Actual implementation would use specialized libraries

CREATE OR REPLACE FUNCTION he_add(p_encrypted1 BYTEA, p_encrypted2 BYTEA)
RETURNS BYTEA AS $$
BEGIN
    -- This would call external homomorphic encryption library
    -- For example: using Microsoft SEAL or IBM HElib
    RETURN homomorphic_add(p_encrypted1, p_encrypted2);
END;
$$ LANGUAGE plpgsql;
```

## Compliance Checklist

### GDPR Compliance Checklist
- [ ] Data minimization implemented in schema design
- [ ] Right to access functionality available
- [ ] Right to erasure (right to be forgotten) implemented
- [ ] Data portability supported
- [ ] Consent management system in place
- [ ] Data protection officer appointed
- [ ] Data breach notification procedures established
- [ ] Privacy impact assessments conducted

### HIPAA Compliance Checklist
- [ ] Access controls implemented (role-based, least privilege)
- [ ] Audit controls in place (comprehensive logging)
- [ ] Encryption at rest and in transit
- [ ] Emergency access procedures established
- [ ] Physical safeguards for servers
- [ ] Business associate agreements in place
- [ ] Workforce training completed
- [ ] Risk analysis performed

### SOC 2 Compliance Checklist
- [ ] Security controls documented and implemented
- [ ] Availability monitoring in place
- [ ] Processing integrity verification
- [ ] Confidentiality controls for sensitive data
- [ ] Privacy controls for personal data
- [ ] Change management procedures
- [ ] Incident response plan
- [ ] Vendor management controls

### PCI DSS Compliance Checklist
- [ ] Network segmentation for cardholder data environment
- [ ] Strong access controls for cardholder data
- [ ] Encryption of stored cardholder data
- [ ] Secure wireless networks
- [ ] Regular vulnerability scanning
- [ ] Penetration testing performed
- [ ] Monitoring and logging of all access
- [ ] Information security policy maintained

> 💡 **Pro Tip**: Compliance is an ongoing process, not a one-time project. Establish continuous compliance monitoring and regular reviews. The biggest compliance risks often come from configuration drift and unpatched vulnerabilities rather than fundamental architectural flaws.