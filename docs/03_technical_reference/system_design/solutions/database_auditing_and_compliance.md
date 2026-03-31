# Database Auditing and Compliance for AI/ML Systems

## Overview

Compliance and auditing are critical for AI/ML systems handling sensitive data. This document covers patterns for implementing robust auditing and compliance frameworks aligned with major regulatory requirements.

## GDPR Compliance Patterns

### Core Requirements
- **Right to Erasure**: Ability to delete personal data upon request
- **Data Minimization**: Store only necessary data for AI/ML purposes
- **Purpose Limitation**: Clear documentation of data usage purposes
- **Consent Management**: Track and manage user consent for data processing

### Implementation Strategies
```sql
-- GDPR-compliant data model with audit trails
CREATE TABLE user_data (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,
    data_version INT DEFAULT 1,
    consent_status JSONB,
    processing_purposes TEXT[]
);

-- Audit log table for GDPR compliance
CREATE TABLE gdpr_audit_log (
    id UUID PRIMARY KEY,
    user_id UUID,
    action TEXT NOT NULL, -- 'create', 'read', 'update', 'delete', 'export'
    data_type TEXT,
    fields_affected TEXT[],
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    request_id TEXT,
    metadata JSONB
);

-- Trigger for automatic audit logging
CREATE OR REPLACE FUNCTION log_gdpr_action()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO gdpr_audit_log (user_id, action, data_type, fields_affected, ip_address, timestamp)
    VALUES (
        CURRENT_SETTING('app.user_id', TRUE)::UUID,
        TG_OP,
        TG_TABLE_NAME,
        ARRAY_AGG(NEW.*),
        inet_client_addr(),
        NOW()
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_gdpr_audit
AFTER INSERT OR UPDATE OR DELETE ON user_data
FOR EACH ROW EXECUTE FUNCTION log_gdpr_action();
```

### Data Subject Request Automation
- **Automated Deletion Pipeline**: Batch jobs that identify and delete user data across all systems
- **Data Export Tools**: Generate comprehensive data exports in standard formats (JSON, CSV)
- **Consent Verification**: Automated verification of consent status before data processing

## HIPAA Compliance Patterns

### Key Requirements
- **Protected Health Information (PHI)**: Strict controls on medical data
- **Access Controls**: Role-based access with audit trails
- **Encryption**: Both at rest and in transit
- **Business Associate Agreements**: Proper contractual obligations

### Implementation Examples
```python
# HIPAA-compliant data access pattern
class HIPAADataAccess:
    def __init__(self, audit_logger, rbac_system):
        self.audit_logger = audit_logger
        self.rbac_system = rbac_system
    
    def get_patient_data(self, patient_id, user_id, purpose):
        """Get patient data with HIPAA-compliant access control"""
        # Verify user has appropriate role for this purpose
        if not self.rbac_system.has_permission(user_id, 'access_patient_data', purpose):
            self.audit_logger.log_access_denied(user_id, patient_id, purpose)
            raise PermissionError("Insufficient privileges for requested purpose")
        
        # Log authorized access
        self.audit_logger.log_access_granted(user_id, patient_id, purpose, 
                                            self._get_user_role(user_id))
        
        # Retrieve data with PHI masking based on user role
        data = self._retrieve_patient_data(patient_id)
        return self._mask_phi_based_on_role(data, self.rbac_system.get_role(user_id))
```

### Audit Trail Requirements
- **Minimum Retention**: 6 years for healthcare data
- **Immutable Logs**: Write-once storage for audit records
- **Real-time Monitoring**: Alerting on suspicious access patterns
- **Separation of Duties**: Different teams manage data vs. audit logs

## SOC 2 Compliance Patterns

### Trust Services Criteria
- **Security**: Protection against unauthorized access
- **Availability**: System availability and performance
- **Processing Integrity**: Complete and accurate processing
- **Confidentiality**: Protection of confidential information
- **Privacy**: Collection, use, retention of personal information

### Database-Specific Controls
```sql
-- SOC 2 compliant database monitoring
CREATE TABLE system_monitoring (
    id UUID PRIMARY KEY,
    metric_name TEXT NOT NULL,
    value NUMERIC,
    threshold_warning NUMERIC,
    threshold_critical NUMERIC,
    status TEXT CHECK (status IN ('normal', 'warning', 'critical')),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    source TEXT
);

-- Automated alerting for SOC 2 controls
CREATE OR REPLACE FUNCTION check_soc2_controls()
RETURNS VOID AS $$
DECLARE
    high_latency RECORD;
BEGIN
    -- Check for high query latency (availability control)
    SELECT * INTO high_latency FROM system_monitoring 
    WHERE metric_name = 'query_latency_ms' 
    AND value > threshold_critical 
    AND timestamp > NOW() - INTERVAL '5 minutes';
    
    IF FOUND THEN
        PERFORM pg_notify('soc2_alerts', 
            json_build_object(
                'metric', 'query_latency',
                'value', high_latency.value,
                'threshold', high_latency.threshold_critical,
                'timestamp', high_latency.timestamp
            )::TEXT);
    END IF;
END;
$$ LANGUAGE plpgsql;
```

## Comprehensive Auditing Framework

### Multi-Layer Auditing
1. **Application Layer**: Business logic audits and consent tracking
2. **Database Layer**: Query-level auditing and transaction logs
3. **Infrastructure Layer**: Network traffic and system access logs
4. **ML Layer**: Model training and inference audit trails

### AI/ML Specific Auditing
- **Model Training Audits**: Track data sources, preprocessing steps, hyperparameters
- **Inference Audits**: Log input data, model version, output, and confidence scores
- **Bias Detection**: Audit for demographic parity and fairness metrics
- **Data Provenance**: Track lineage from raw data to final predictions

```python
class MLAuditLogger:
    def __init__(self, database_connection):
        self.db = database_connection
    
    def log_training_run(self, run_id, config, data_sources, metrics):
        """Log complete ML training run with compliance metadata"""
        audit_record = {
            'run_id': run_id,
            'timestamp': datetime.utcnow(),
            'user_id': get_current_user_id(),
            'model_type': config['model_type'],
            'data_sources': data_sources,
            'preprocessing_steps': config.get('preprocessing', []),
            'hyperparameters': config.get('hyperparameters', {}),
            'performance_metrics': metrics,
            'compliance_tags': self._determine_compliance_tags(data_sources, config),
            'data_retention_period': self._calculate_retention_period(data_sources)
        }
        
        self.db.execute("""
            INSERT INTO ml_training_audits 
            (run_id, timestamp, user_id, model_type, data_sources, 
             preprocessing_steps, hyperparameters, performance_metrics,
             compliance_tags, data_retention_period)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, tuple(audit_record.values()))
```

## Real-World Implementation Examples

### Healthcare AI Platform
- **GDPR + HIPAA Hybrid**: Dual compliance framework for EU and US operations
- **Automated Consent Management**: Dynamic consent forms with granular permissions
- **Audit Dashboard**: Real-time visualization of compliance metrics
- **Quarterly Audits**: Automated reports for regulatory submissions

### Financial AI System
- **SOC 2 Type II Certified**: Comprehensive controls for financial data
- **Data Lineage Tracking**: End-to-end tracing from transaction data to model outputs
- **Incident Response**: Automated workflows for data breaches and security incidents
- **Third-Party Validation**: Regular external audits and penetration testing

## Performance and Scalability Considerations

| Audit Level | Storage Overhead | Query Impact | Recommended Use |
|-------------|------------------|--------------|-----------------|
| Basic Access Logs | 5-10% | <1% | Minimum compliance |
| Full Query Logging | 20-50% | 5-15% | High-risk systems |
| ML-Specific Auditing | 15-30% | 3-10% | AI/ML production systems |
| Real-time Anomaly Detection | 30-70% | 10-25% | Critical infrastructure |

## Best Practices

1. **Immutable Audit Logs**: Use write-once storage or blockchain-based solutions
2. **Separation of Duties**: Different teams manage data vs. audit logs
3. **Automated Compliance Checks**: Continuous validation against regulatory requirements
4. **Retention Policies**: Automated data lifecycle management based on regulations
5. **Cross-System Correlation**: Unified audit IDs across microservices and databases
6. **Anonymized Analytics**: Perform compliance analytics on anonymized audit data

## References
- GDPR Article 30: Records of Processing Activities
- HIPAA Security Rule: Technical Safeguards (45 CFR § 164.312)
- SOC 2 Trust Services Criteria
- NIST SP 800-53: Security and Privacy Controls
- ISO/IEC 27001: Information Security Management
- AWS Compliance Programs for AI/ML Workloads