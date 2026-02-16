# Database Governance and Data Lineage Tutorial for AI/ML Systems

## Overview

This hands-on tutorial teaches senior AI/ML engineers how to implement comprehensive database governance and data lineage tracking for AI applications. We'll cover metadata management, lineage tracking, quality governance, and compliance frameworks.

## Prerequisites
- PostgreSQL 14+ or MySQL 8+
- Neo4j or Amazon Neptune (for lineage graphs)
- Apache Atlas or DataHub (optional for advanced governance)
- Basic understanding of data governance concepts

## Tutorial Structure
This tutorial is divided into 6 progressive sections:
1. **Metadata Management** - Structured metadata capture
2. **Data Lineage Tracking** - End-to-end lineage implementation
3. **Quality Governance** - Data quality monitoring and remediation
4. **Compliance Frameworks** - Regulatory compliance implementation
5. **Governance Workflows** - Automated governance processes
6. **AI-Specific Governance** - Feature store and model governance

## Section 1: Metadata Management

### Step 1: Design metadata schema
```sql
-- Core metadata tables
CREATE TABLE data_assets (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL, -- 'table', 'view', 'feature', 'model'
    description TEXT,
    owner VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'active' -- 'active', 'deprecated', 'archived'
);

CREATE TABLE metadata_fields (
    id UUID PRIMARY KEY,
    asset_id UUID REFERENCES data_assets(id),
    field_name VARCHAR(255),
    data_type VARCHAR(50),
    description TEXT,
    is_required BOOLEAN DEFAULT FALSE,
    is_sensitive BOOLEAN DEFAULT FALSE,
    validation_rules JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE data_quality_metrics (
    id UUID PRIMARY KEY,
    asset_id UUID REFERENCES data_assets(id),
    metric_name VARCHAR(100),
    value FLOAT,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    threshold FLOAT,
    status VARCHAR(20) -- 'pass', 'warn', 'fail'
);
```

### Step 2: Implement metadata capture
```python
import json
from datetime import datetime
import uuid

class MetadataManager:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def register_asset(self, name, asset_type, description="", owner=""):
        """Register a new data asset"""
        asset_id = str(uuid.uuid4())
        
        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO data_assets (id, name, type, description, owner, status)
            VALUES (%s, %s, %s, %s, %s, 'active')
        """, (asset_id, name, asset_type, description, owner))
        
        self.db.commit()
        cursor.close()
        
        return asset_id
    
    def add_field_metadata(self, asset_id, field_name, data_type, 
                          description="", is_required=False, is_sensitive=False):
        """Add metadata for a specific field"""
        field_id = str(uuid.uuid4())
        
        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO metadata_fields (id, asset_id, field_name, data_type, 
                                       description, is_required, is_sensitive)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (field_id, asset_id, field_name, data_type, description, 
              is_required, is_sensitive))
        
        self.db.commit()
        cursor.close()
        
        return field_id
    
    def update_quality_metric(self, asset_id, metric_name, value, threshold=None):
        """Update data quality metric"""
        status = 'pass' if value >= (threshold or 0) else 'fail'
        
        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO data_quality_metrics (id, asset_id, metric_name, value, 
                                            timestamp, threshold, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (str(uuid.uuid4()), asset_id, metric_name, value, 
              datetime.now(), threshold, status))
        
        self.db.commit()
        cursor.close()

# Usage example
metadata_manager = MetadataManager(db_connection)

# Register feature store
feature_store_id = metadata_manager.register_asset(
    "user_features", "feature", "User engagement features", "data_science_team"
)

# Add field metadata
metadata_manager.add_field_metadata(
    feature_store_id, "user_id", "UUID", "Unique user identifier", True, False
)
metadata_manager.add_field_metadata(
    feature_store_id, "engagement_score", "FLOAT", "User engagement score", True, False
)
metadata_manager.add_field_metadata(
    feature_store_id, "health_data", "JSON", "Health metrics", False, True
)

# Update quality metrics
metadata_manager.update_quality_metric(feature_store_id, "completeness", 98.5, 95.0)
metadata_manager.update_quality_metric(feature_store_id, "accuracy", 99.2, 98.0)
```

## Section 2: Data Lineage Tracking

### Step 1: Design lineage schema
```sql
-- Lineage tables
CREATE TABLE lineage_operations (
    id UUID PRIMARY KEY,
    operation_type VARCHAR(50) NOT NULL, -- 'ETL', 'FEATURE_COMPUTATION', 'MODEL_TRAINING'
    source_asset_id UUID REFERENCES data_assets(id),
    target_asset_id UUID REFERENCES data_assets(id),
    transformation_code_hash VARCHAR(64),
    parameters JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    executed_by VARCHAR(255),
    status VARCHAR(20) DEFAULT 'completed'
);

CREATE TABLE lineage_dependencies (
    id UUID PRIMARY KEY,
    operation_id UUID REFERENCES lineage_operations(id),
    dependent_asset_id UUID REFERENCES data_assets(id),
    dependency_type VARCHAR(50), -- 'input', 'output', 'parameter'
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast lineage queries
CREATE INDEX idx_lineage_source ON lineage_operations (source_asset_id);
CREATE INDEX idx_lineage_target ON lineage_operations (target_asset_id);
CREATE INDEX idx_lineage_operation ON lineage_operations (operation_type);
```

### Step 2: Implement lineage capture
```python
class LineageTracker:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def track_operation(self, operation_type, source_asset_id, target_asset_id,
                       transformation_code, parameters=None, executed_by="system"):
        """Track a data operation in lineage"""
        operation_id = str(uuid.uuid4())
        code_hash = self._hash_code(transformation_code)
        
        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO lineage_operations (id, operation_type, source_asset_id, 
                                          target_asset_id, transformation_code_hash,
                                          parameters, executed_by, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'completed')
        """, (operation_id, operation_type, source_asset_id, target_asset_id,
              code_hash, json.dumps(parameters or {}), executed_by))
        
        # Track dependencies
        if source_asset_id:
            cursor.execute("""
                INSERT INTO lineage_dependencies (id, operation_id, dependent_asset_id, 
                                                dependency_type, created_at)
                VALUES (%s, %s, %s, 'input', %s)
            """, (str(uuid.uuid4()), operation_id, source_asset_id, datetime.now()))
        
        if target_asset_id:
            cursor.execute("""
                INSERT INTO lineage_dependencies (id, operation_id, dependent_asset_id, 
                                                dependency_type, created_at)
                VALUES (%s, %s, %s, 'output', %s)
            """, (str(uuid.uuid4()), operation_id, target_asset_id, datetime.now()))
        
        self.db.commit()
        cursor.close()
        
        return operation_id
    
    def _hash_code(self, code):
        """Hash transformation code for lineage tracking"""
        import hashlib
        return hashlib.sha256(code.encode()).hexdigest()[:64]
    
    def get_lineage_path(self, asset_id):
        """Get full lineage path for an asset"""
        cursor = self.db.cursor()
        
        # Get forward lineage (what this asset feeds into)
        cursor.execute("""
            WITH RECURSIVE lineage AS (
                SELECT 
                    lo.id,
                    lo.operation_type,
                    lo.source_asset_id,
                    lo.target_asset_id,
                    lo.created_at,
                    1 as depth
                FROM lineage_operations lo
                WHERE lo.source_asset_id = %s
                
                UNION ALL
                
                SELECT 
                    lo.id,
                    lo.operation_type,
                    lo.source_asset_id,
                    lo.target_asset_id,
                    lo.created_at,
                    l.depth + 1
                FROM lineage_operations lo
                INNER JOIN lineage l ON lo.source_asset_id = l.target_asset_id
                WHERE l.depth < 10
            )
            SELECT * FROM lineage ORDER BY depth, created_at;
        """, (asset_id,))
        
        forward_lineage = cursor.fetchall()
        
        # Get backward lineage (what feeds into this asset)
        cursor.execute("""
            WITH RECURSIVE lineage AS (
                SELECT 
                    lo.id,
                    lo.operation_type,
                    lo.source_asset_id,
                    lo.target_asset_id,
                    lo.created_at,
                    1 as depth
                FROM lineage_operations lo
                WHERE lo.target_asset_id = %s
                
                UNION ALL
                
                SELECT 
                    lo.id,
                    lo.operation_type,
                    lo.source_asset_id,
                    lo.target_asset_id,
                    lo.created_at,
                    l.depth + 1
                FROM lineage_operations lo
                INNER JOIN lineage l ON lo.target_asset_id = l.source_asset_id
                WHERE l.depth < 10
            )
            SELECT * FROM lineage ORDER BY depth, created_at;
        """, (asset_id,))
        
        backward_lineage = cursor.fetchall()
        
        cursor.close()
        
        return {
            'forward': forward_lineage,
            'backward': backward_lineage
        }

# Usage example
lineage_tracker = LineageTracker(db_connection)

# Track feature computation
feature_asset_id = metadata_manager.register_asset("user_engagement_feature", "feature")
source_asset_id = metadata_manager.register_asset("raw_user_events", "table")

lineage_tracker.track_operation(
    operation_type="FEATURE_COMPUTATION",
    source_asset_id=source_asset_id,
    target_asset_id=feature_asset_id,
    transformation_code="""
        SELECT 
            user_id,
            AVG(clicks) as engagement_score,
            COUNT(*) as session_count
        FROM raw_user_events
        GROUP BY user_id
    """,
    parameters={
        "window_size": 30,
        "aggregation_method": "average"
    },
    executed_by="feature_pipeline_v2"
)
```

## Section 3: Quality Governance

### Step 1: Implement quality rules engine
```sql
-- Quality rules table
CREATE TABLE quality_rules (
    id UUID PRIMARY KEY,
    asset_id UUID REFERENCES data_assets(id),
    rule_name VARCHAR(255),
    rule_type VARCHAR(50), -- 'completeness', 'accuracy', 'consistency', 'timeliness'
    threshold FLOAT,
    severity VARCHAR(20), -- 'low', 'medium', 'high', 'critical'
    description TEXT,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Quality rule violations
CREATE TABLE quality_violations (
    id UUID PRIMARY KEY,
    rule_id UUID REFERENCES quality_rules(id),
    asset_id UUID REFERENCES data_assets(id),
    violation_value FLOAT,
    threshold FLOAT,
    severity VARCHAR(20),
    status VARCHAR(20), -- 'open', 'in_progress', 'resolved'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(255)
);
```

### Step 2: Quality monitoring workflow
```python
class QualityGovernance:
    def __init__(self, db_connection):
        self.db = db_connection
        self.metadata_manager = MetadataManager(db_connection)
        self.lineage_tracker = LineageTracker(db_connection)
    
    def define_quality_rule(self, asset_id, rule_name, rule_type, threshold, severity, description=""):
        """Define a quality rule for an asset"""
        rule_id = str(uuid.uuid4())
        
        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO quality_rules (id, asset_id, rule_name, rule_type, 
                                     threshold, severity, description, enabled)
            VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE)
        """, (rule_id, asset_id, rule_name, rule_type, threshold, severity, description))
        
        self.db.commit()
        cursor.close()
        
        return rule_id
    
    def evaluate_quality_rules(self, asset_id):
        """Evaluate all quality rules for an asset"""
        cursor = self.db.cursor()
        
        # Get active rules for the asset
        cursor.execute("""
            SELECT id, rule_name, rule_type, threshold, severity
            FROM quality_rules 
            WHERE asset_id = %s AND enabled = TRUE
        """, (asset_id,))
        
        rules = cursor.fetchall()
        
        violations = []
        
        for rule_id, rule_name, rule_type, threshold, severity in rules:
            # Evaluate based on rule type
            if rule_type == 'completeness':
                completeness = self._get_completeness(asset_id)
                if completeness < threshold:
                    violations.append({
                        'rule_id': rule_id,
                        'asset_id': asset_id,
                        'violation_value': completeness,
                        'threshold': threshold,
                        'severity': severity,
                        'status': 'open'
                    })
            
            elif rule_type == 'accuracy':
                accuracy = self._get_accuracy(asset_id)
                if accuracy < threshold:
                    violations.append({
                        'rule_id': rule_id,
                        'asset_id': asset_id,
                        'violation_value': accuracy,
                        'threshold': threshold,
                        'severity': severity,
                        'status': 'open'
                    })
        
        # Insert violations
        for violation in violations:
            cursor.execute("""
                INSERT INTO quality_violations (id, rule_id, asset_id, violation_value,
                                              threshold, severity, status, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                str(uuid.uuid4()),
                violation['rule_id'],
                violation['asset_id'],
                violation['violation_value'],
                violation['threshold'],
                violation['severity'],
                violation['status'],
                datetime.now()
            ))
        
        self.db.commit()
        cursor.close()
        
        return violations
    
    def _get_completeness(self, asset_id):
        """Get completeness metric for an asset"""
        # This would query actual data completeness
        # For demo, return mock value
        return 98.5
    
    def _get_accuracy(self, asset_id):
        """Get accuracy metric for an asset"""
        # This would query actual data accuracy
        # For demo, return mock value
        return 99.2

# Usage example
quality_governance = QualityGovernance(db_connection)

# Define quality rules for feature store
feature_asset_id = metadata_manager.register_asset("user_features", "feature")

quality_governance.define_quality_rule(
    feature_asset_id, "completeness_check", "completeness", 95.0, "high",
    "Ensure at least 95% of records have complete data"
)
quality_governance.define_quality_rule(
    feature_asset_id, "accuracy_check", "accuracy", 98.0, "critical",
    "Ensure data accuracy is at least 98%"
)

# Evaluate rules
violations = quality_governance.evaluate_quality_rules(feature_asset_id)
print(f"Found {len(violations)} quality violations")
```

## Section 4: Compliance Frameworks

### Step 1: GDPR compliance implementation
```sql
-- GDPR-specific metadata
ALTER TABLE data_assets ADD COLUMN gdpr_category VARCHAR(50);
ALTER TABLE metadata_fields ADD COLUMN gdpr_sensitivity VARCHAR(20); -- 'low', 'medium', 'high'

-- GDPR consent management
CREATE TABLE gdpr_consent (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    asset_id UUID REFERENCES data_assets(id),
    consent_type VARCHAR(50), -- 'processing', 'storage', 'sharing'
    granted BOOLEAN NOT NULL,
    granted_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    ip_address INET,
    user_agent TEXT
);

-- GDPR audit trail
CREATE TABLE gdpr_audit_trail (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_id UUID,
    action VARCHAR(50),
    asset_id UUID,
    details JSONB,
    ip_address INET
);
```

### Step 2: Automated compliance reporting
```python
class ComplianceReporter:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def generate_gdpr_report(self, start_date=None, end_date=None):
        """Generate GDPR compliance report"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        cursor = self.db.cursor()
        
        # Get consent summary
        cursor.execute("""
            SELECT 
                consent_type,
                COUNT(*) as total,
                COUNT(CASE WHEN granted = TRUE THEN 1 END) as granted,
                COUNT(CASE WHEN granted = FALSE THEN 1 END) as revoked
            FROM gdpr_consent
            WHERE granted_at BETWEEN %s AND %s
            GROUP BY consent_type
        """, (start_date, end_date))
        
        consent_summary = cursor.fetchall()
        
        # Get data subject requests
        cursor.execute("""
            SELECT 
                request_type,
                COUNT(*) as count,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed
            FROM gdpr_requests
            WHERE created_at BETWEEN %s AND %s
            GROUP BY request_type
        """, (start_date, end_date))
        
        dsr_summary = cursor.fetchall()
        
        # Get quality compliance
        cursor.execute("""
            SELECT 
                severity,
                COUNT(*) as total_violations,
                COUNT(CASE WHEN status = 'open' THEN 1 END) as open_violations
            FROM quality_violations
            WHERE created_at BETWEEN %s AND %s
            GROUP BY severity
        """, (start_date, end_date))
        
        quality_summary = cursor.fetchall()
        
        cursor.close()
        
        return {
            'period': f"{start_date.date()} to {end_date.date()}",
            'consent_summary': consent_summary,
            'dsr_summary': dsr_summary,
            'quality_summary': quality_summary,
            'overall_compliance': self._calculate_overall_compliance(consent_summary, dsr_summary, quality_summary)
        }
    
    def _calculate_overall_compliance(self, consent, dsr, quality):
        """Calculate overall compliance score"""
        # Simple weighted calculation
        consent_score = sum(c[2]/c[1] if c[1] > 0 else 0 for c in consent) / len(consent) if consent else 1.0
        dsr_score = sum(d[2]/d[1] if d[1] > 0 else 0 for d in dsr) / len(dsr) if dsr else 1.0
        quality_score = 1.0 - (sum(q[2]/q[1] if q[1] > 0 else 0 for q in quality) / len(quality) if quality else 0.0)
        
        return (consent_score * 0.4) + (dsr_score * 0.3) + (quality_score * 0.3)

# Usage example
compliance_reporter = ComplianceReporter(db_connection)
report = compliance_reporter.generate_gdpr_report()
print(json.dumps(report, indent=2))
```

## Section 5: Governance Workflows

### Step 1: Automated governance workflows
```python
class GovernanceWorkflow:
    def __init__(self, db_connection):
        self.db = db_connection
        self.quality_governance = QualityGovernance(db_connection)
    
    def run_governance_cycle(self):
        """Run automated governance cycle"""
        print("Starting governance cycle...")
        
        # 1. Refresh metadata
        self._refresh_metadata()
        
        # 2. Evaluate quality rules
        print("Evaluating quality rules...")
        self._evaluate_quality_rules()
        
        # 3. Check lineage completeness
        print("Checking lineage completeness...")
        self._check_lineage_completeness()
        
        # 4. Generate compliance reports
        print("Generating compliance reports...")
        self._generate_compliance_reports()
        
        # 5. Send alerts for critical issues
        print("Sending alerts for critical issues...")
        self._send_critical_alerts()
        
        print("Governance cycle completed.")
    
    def _refresh_metadata(self):
        """Refresh metadata from data sources"""
        # This would connect to various data sources and update metadata
        pass
    
    def _evaluate_quality_rules(self):
        """Evaluate quality rules for all assets"""
        cursor = self.db.cursor()
        cursor.execute("SELECT id FROM data_assets WHERE status = 'active'")
        assets = cursor.fetchall()
        
        for asset_id, in assets:
            try:
                self.quality_governance.evaluate_quality_rules(asset_id)
            except Exception as e:
                print(f"Error evaluating rules for asset {asset_id}: {e}")
    
    def _check_lineage_completeness(self):
        """Check if lineage is complete for critical assets"""
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT id, name FROM data_assets 
            WHERE type IN ('model', 'feature', 'training_data') 
            AND status = 'active'
        """)
        critical_assets = cursor.fetchall()
        
        for asset_id, name in critical_assets:
            lineage = self.lineage_tracker.get_lineage_path(asset_id)
            if not lineage['backward'] and not lineage['forward']:
                print(f"Warning: No lineage found for critical asset '{name}' (ID: {asset_id})")
    
    def _generate_compliance_reports(self):
        """Generate and store compliance reports"""
        reporter = ComplianceReporter(self.db)
        report = reporter.generate_gdpr_report()
        
        # Store report
        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO compliance_reports (id, report_type, content, generated_at, status)
            VALUES (%s, 'gdpr_monthly', %s, %s, 'generated')
        """, (str(uuid.uuid4()), json.dumps(report), datetime.now()))
        self.db.commit()
        cursor.close()
    
    def _send_critical_alerts(self):
        """Send alerts for critical governance issues"""
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT 
                qv.id, 
                da.name as asset_name,
                qr.rule_name,
                qv.violation_value,
                qv.threshold,
                qv.severity
            FROM quality_violations qv
            JOIN quality_rules qr ON qv.rule_id = qr.id
            JOIN data_assets da ON qv.asset_id = da.id
            WHERE qv.status = 'open' AND qv.severity = 'critical'
        """)
        
        critical_violations = cursor.fetchall()
        
        for violation_id, asset_name, rule_name, violation_value, threshold, severity in critical_violations:
            print(f"CRITICAL ALERT: {severity.upper()} violation in '{asset_name}' - {rule_name}")
            print(f"  Value: {violation_value}, Threshold: {threshold}")
            # In production, this would send email/SMS alerts
        
        cursor.close()

# Usage example
workflow = GovernanceWorkflow(db_connection)
workflow.run_governance_cycle()
```

## Section 6: AI-Specific Governance

### Step 1: Feature store governance
```sql
-- Feature store specific governance
CREATE TABLE feature_definitions (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    version VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE feature_versions (
    id UUID PRIMARY KEY,
    feature_id UUID REFERENCES feature_definitions(id),
    version VARCHAR(20),
    definition_hash VARCHAR(64),
    training_data_version VARCHAR(255),
    performance_metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE feature_usage (
    id UUID PRIMARY KEY,
    feature_id UUID REFERENCES feature_definitions(id),
    model_id UUID,
    pipeline_id VARCHAR(255),
    usage_count INTEGER DEFAULT 0,
    last_used TIMESTAMPTZ DEFAULT NOW()
);
```

### Step 2: Model registry governance
```sql
-- Model registry with governance
CREATE TABLE models (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    version VARCHAR(20),
    status VARCHAR(20), -- 'development', 'testing', 'production', 'deprecated'
    created_by VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    governance_status VARCHAR(20) -- 'pending_review', 'approved', 'rejected'
);

CREATE TABLE model_artifacts (
    id UUID PRIMARY KEY,
    model_id UUID REFERENCES models(id),
    artifact_type VARCHAR(50), -- 'weights', 'config', 'metrics', 'predictions'
    storage_path TEXT,
    checksum VARCHAR(64),
    size_bytes BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE model_reviews (
    id UUID PRIMARY KEY,
    model_id UUID REFERENCES models(id),
    reviewer_id VARCHAR(255),
    review_date TIMESTAMPTZ,
    status VARCHAR(20), -- 'approved', 'rejected', 'needs_changes'
    comments TEXT,
    checklist JSONB
);
```

## Hands-on Exercises

### Exercise 1: Implement metadata capture
1. Create metadata tables for your AI data assets
2. Implement the MetadataManager class
3. Register sample feature stores and training datasets
4. Add field metadata for key columns

### Exercise 2: Build lineage tracking
1. Implement the LineageTracker class
2. Track a simple ETL process
3. Query lineage for a specific asset
4. Visualize the lineage graph

### Exercise 3: Quality governance system
1. Define quality rules for your data assets
2. Implement quality evaluation
3. Create violation tracking
4. Build simple alerting system

### Exercise 4: AI-specific governance
1. Implement feature store governance tables
2. Track feature versions and usage
3. Build model registry with governance workflow
4. Test with sample AI pipeline

## Best Practices Summary

1. **Start Small**: Begin with critical assets and expand gradually
2. **Automate**: Build automated governance workflows
3. **Integrate**: Integrate governance into CI/CD pipelines
4. **Monitor**: Continuous monitoring of governance metrics
5. **Improve**: Regular review and improvement of governance processes

This tutorial provides practical, hands-on experience with database governance and data lineage specifically for AI/ML systems. Complete all exercises to master these critical governance skills.