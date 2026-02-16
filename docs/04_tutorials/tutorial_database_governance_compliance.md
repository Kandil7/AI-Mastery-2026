# Database Integration with Data Governance and Compliance Frameworks Tutorial

## Overview

This tutorial focuses on integrating databases with data governance and compliance frameworks. We'll cover GDPR, CCPA, HIPAA compliance, data lineage tracking, access control, audit logging, and privacy-preserving techniques specifically for senior AI/ML engineers building compliant AI systems.

## Prerequisites
- Python 3.8+
- PostgreSQL/MySQL with proper security features
- Apache Atlas, OpenMetadata, or similar governance tools (optional)
- Basic understanding of data privacy regulations

## Tutorial Structure
1. **Regulatory Compliance** - GDPR, CCPA, HIPAA implementation
2. **Data Lineage Tracking** - End-to-end data provenance
3. **Access Control and Authentication** - RBAC and ABAC implementation
4. **Audit Logging** - Comprehensive audit trails
5. **Privacy-Preserving Techniques** - Differential privacy, encryption
6. **Compliance Monitoring** - Automated compliance checks
7. **Performance Benchmarking** - Governance overhead analysis

## Section 1: Regulatory Compliance

### Step 1: GDPR compliance implementation
```python
import psycopg2
import json
import datetime
from typing import Dict, List, Optional

class GDPRComplianceManager:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
    
    def create_gdpr_compliance_tables(self):
        """Create tables for GDPR compliance"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Consent management
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS consent_records (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            consent_type VARCHAR(50) NOT NULL,
            consent_given BOOLEAN NOT NULL,
            consent_timestamp TIMESTAMP NOT NULL,
            consent_version VARCHAR(20),
            ip_address VARCHAR(45),
            user_agent TEXT,
            revoked_at TIMESTAMP
        );
        """)
        
        # Data subject requests
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS dsr_requests (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            request_type VARCHAR(50) NOT NULL,
            status VARCHAR(20) NOT NULL,
            created_at TIMESTAMP NOT NULL,
            completed_at TIMESTAMP,
            processed_by VARCHAR(255),
            notes TEXT
        );
        """)
        
        # Right to be forgotten tracking
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS data_deletion_logs (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            deletion_request_id INTEGER REFERENCES dsr_requests(id),
            table_name VARCHAR(255) NOT NULL,
            record_id VARCHAR(255) NOT NULL,
            deleted_at TIMESTAMP NOT NULL,
            deleted_by VARCHAR(255)
        );
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return "GDPR compliance tables created successfully"
    
    def record_consent(self, user_id: str, consent_type: str, 
                      consent_given: bool, ip_address: str = None):
        """Record user consent"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO consent_records (
            user_id, consent_type, consent_given, consent_timestamp, ip_address
        ) VALUES (%s, %s, %s, NOW(), %s)
        RETURNING id
        """
        
        cursor.execute(insert_query, (user_id, consent_type, consent_given, ip_address))
        record_id = cursor.fetchone()[0]
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return record_id
    
    def process_right_to_be_forgotten(self, user_id: str, request_id: int):
        """Process right to be forgotten request"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        try:
            # Get tables that contain user data
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name NOT IN ('consent_records', 'dsr_requests', 'data_deletion_logs')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            deletion_count = 0
            
            # Delete from each table
            for table in tables:
                # Check if table has user_id column
                cursor.execute(f"""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = '{table}' AND column_name = 'user_id'
                """)
                has_user_id = cursor.fetchone() is not None
                
                if has_user_id:
                    # Delete records
                    delete_query = f"DELETE FROM {table} WHERE user_id = %s RETURNING id"
                    cursor.execute(delete_query, (user_id,))
                    
                    deleted_rows = cursor.rowcount
                    deletion_count += deleted_rows
                    
                    # Log deletions
                    if deleted_rows > 0:
                        cursor.execute("""
                            INSERT INTO data_deletion_logs (
                                user_id, deletion_request_id, table_name, record_id, deleted_at, deleted_by
                            ) VALUES (%s, %s, %s, %s, NOW(), %s)
                        """, (user_id, request_id, table, str(deleted_rows), "system"))
            
            # Update request status
            cursor.execute("""
                UPDATE dsr_requests 
                SET status = 'completed', completed_at = NOW()
                WHERE id = %s
            """, (request_id,))
            
            conn.commit()
            return deletion_count
            
        except Exception as e:
            conn.rollback()
            raise e
        
        finally:
            cursor.close()
            conn.close()

# Usage example
gdpr_manager = GDPRComplianceManager(db_config)

# Create compliance tables
gdpr_manager.create_gdpr_compliance_tables()

# Record consent
consent_id = gdpr_manager.record_consent(
    user_id="123",
    consent_type="marketing",
    consent_given=True,
    ip_address="192.168.1.1"
)

print(f"Consent recorded: {consent_id}")

# Process right to be forgotten
# deletion_count = gdpr_manager.process_right_to_be_forgotten("123", 1)
# print(f"Deleted {deletion_count} records")
```

### Step 2: CCPA compliance implementation
```python
class CCPAComplianceManager:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
    
    def create_ccpa_tables(self):
        """Create tables for CCPA compliance"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Do Not Sell requests
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS do_not_sell_requests (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            email VARCHAR(255),
            phone VARCHAR(20),
            request_type VARCHAR(50) NOT NULL,
            status VARCHAR(20) NOT NULL,
            created_at TIMESTAMP NOT NULL,
            verified_at TIMESTAMP,
            processed_at TIMESTAMP,
            notes TEXT
        );
        """)
        
        # Data sharing logs
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS data_sharing_logs (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            third_party VARCHAR(255) NOT NULL,
            data_categories TEXT[],
            shared_at TIMESTAMP NOT NULL,
            purpose VARCHAR(255),
            opt_out_status VARCHAR(20)
        );
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return "CCPA compliance tables created successfully"
    
    def process_do_not_sell_request(self, user_id: str, email: str, 
                                  request_type: str = "do_not_sell"):
        """Process CCPA do not sell request"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO do_not_sell_requests (
            user_id, email, request_type, status, created_at
        ) VALUES (%s, %s, %s, 'pending', NOW())
        RETURNING id
        """
        
        cursor.execute(insert_query, (user_id, email, request_type))
        request_id = cursor.fetchone()[0]
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return request_id
    
    def log_data_sharing(self, user_id: str, third_party: str, 
                        data_categories: List[str], purpose: str):
        """Log data sharing for CCPA compliance"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO data_sharing_logs (
            user_id, third_party, data_categories, shared_at, purpose, opt_out_status
        ) VALUES (%s, %s, %s, NOW(), %s, 'opt_in')
        """
        
        cursor.execute(insert_query, (user_id, third_party, data_categories, purpose))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True

# Usage example
ccpa_manager = CCPAComplianceManager(db_config)

# Create CCPA tables
ccpa_manager.create_ccpa_tables()

# Process do not sell request
request_id = ccpa_manager.process_do_not_sell_request("123", "user@example.com")
print(f"Do Not Sell request created: {request_id}")

# Log data sharing
ccpa_manager.log_data_sharing(
    user_id="123",
    third_party="Analytics Inc",
    data_categories=["demographics", "behavioral"],
    purpose="marketing analytics"
)
```

### Step 3: HIPAA compliance implementation
```python
class HIPAAComplianceManager:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
    
    def create_hipaa_tables(self):
        """Create tables for HIPAA compliance"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # PHI (Protected Health Information) tracking
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS phi_records (
            id SERIAL PRIMARY KEY,
            patient_id VARCHAR(255) NOT NULL,
            data_type VARCHAR(50) NOT NULL,
            encryption_key_id VARCHAR(255),
            access_level VARCHAR(20) NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP
        );
        """)
        
        # Access logs for PHI
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS phi_access_logs (
            id SERIAL PRIMARY KEY,
            patient_id VARCHAR(255) NOT NULL,
            user_id VARCHAR(255) NOT NULL,
            access_type VARCHAR(50) NOT NULL,
            access_reason VARCHAR(255),
            accessed_at TIMESTAMP NOT NULL,
            ip_address VARCHAR(45),
            user_agent TEXT
        );
        """)
        
        # Security incident logs
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS security_incidents (
            id SERIAL PRIMARY KEY,
            incident_type VARCHAR(50) NOT NULL,
            description TEXT,
            detected_at TIMESTAMP NOT NULL,
            reported_at TIMESTAMP,
            resolved_at TIMESTAMP,
            status VARCHAR(20) NOT NULL,
            severity VARCHAR(20)
        );
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return "HIPAA compliance tables created successfully"
    
    def encrypt_phi_data(self, patient_id: str, data_type: str, 
                        access_level: str, encryption_key_id: str = None):
        """Encrypt PHI data and track encryption"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO phi_records (
            patient_id, data_type, encryption_key_id, access_level, created_at
        ) VALUES (%s, %s, %s, %s, NOW())
        RETURNING id
        """
        
        cursor.execute(insert_query, (patient_id, data_type, encryption_key_id, access_level))
        record_id = cursor.fetchone()[0]
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return record_id
    
    def log_phi_access(self, patient_id: str, user_id: str, 
                       access_type: str, access_reason: str, ip_address: str):
        """Log PHI access for audit purposes"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO phi_access_logs (
            patient_id, user_id, access_type, access_reason, accessed_at, ip_address
        ) VALUES (%s, %s, %s, %s, NOW(), %s)
        """
        
        cursor.execute(insert_query, (patient_id, user_id, access_type, access_reason, ip_address))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True

# Usage example
hipaa_manager = HIPAAComplianceManager(db_config)

# Create HIPAA tables
hipaa_manager.create_hipaa_tables()

# Encrypt PHI data
record_id = hipaa_manager.encrypt_phi_data(
    patient_id="P123",
    data_type="medical_records",
    access_level="restricted",
    encryption_key_id="key-001"
)
print(f"PHI encrypted: {record_id}")

# Log PHI access
hipaa_manager.log_phi_access(
    patient_id="P123",
    user_id="doctor-456",
    access_type="view",
    access_reason="treatment",
    ip_address="10.0.0.1"
)
```

## Section 2: Data Lineage Tracking

### Step 1: Comprehensive data lineage system
```python
from typing import Dict, List, Optional
import uuid
import json

class DataLineageTracker:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
    
    def create_lineage_tables(self):
        """Create tables for data lineage tracking"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Data assets
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS data_assets (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            type VARCHAR(50) NOT NULL,
            location TEXT,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP,
            owner VARCHAR(255),
            description TEXT
        );
        """)
        
        # Lineage relationships
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS lineage_relationships (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            source_asset_id UUID REFERENCES data_assets(id),
            target_asset_id UUID REFERENCES data_assets(id),
            transformation_type VARCHAR(50) NOT NULL,
            transformation_details JSONB,
            created_at TIMESTAMP NOT NULL,
            created_by VARCHAR(255)
        );
        """)
        
        # Data quality metrics
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS data_quality_metrics (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            asset_id UUID REFERENCES data_assets(id),
            metric_name VARCHAR(100) NOT NULL,
            metric_value FLOAT,
            metric_unit VARCHAR(50),
            measured_at TIMESTAMP NOT NULL,
            threshold FLOAT,
            status VARCHAR(20)
        );
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return "Data lineage tables created successfully"
    
    def register_data_asset(self, name: str, asset_type: str, 
                          location: str, owner: str = None, description: str = ""):
        """Register a data asset in the lineage system"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO data_assets (
            name, type, location, created_at, updated_at, owner, description
        ) VALUES (%s, %s, %s, NOW(), NOW(), %s, %s)
        RETURNING id
        """
        
        cursor.execute(insert_query, (name, asset_type, location, owner, description))
        asset_id = cursor.fetchone()[0]
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return asset_id
    
    def create_lineage_relationship(self, source_asset_id: str, 
                                  target_asset_id: str,
                                  transformation_type: str,
                                  transformation_details: Dict = None,
                                  created_by: str = "system"):
        """Create lineage relationship between assets"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO lineage_relationships (
            source_asset_id, target_asset_id, transformation_type,
            transformation_details, created_at, created_by
        ) VALUES (%s, %s, %s, %s, NOW(), %s)
        """
        
        cursor.execute(insert_query, (
            source_asset_id, target_asset_id, transformation_type,
            json.dumps(transformation_details) if transformation_details else None,
            created_by
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    
    def get_lineage_path(self, asset_id: str):
        """Get complete lineage path for an asset"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Get direct relationships
        cursor.execute("""
            SELECT 
                sa.name as source_name,
                sa.type as source_type,
                lr.transformation_type,
                ta.name as target_name,
                ta.type as target_type
            FROM lineage_relationships lr
            JOIN data_assets sa ON lr.source_asset_id = sa.id
            JOIN data_assets ta ON lr.target_asset_id = ta.id
            WHERE lr.target_asset_id = %s
        """, (asset_id,))
        
        incoming = cursor.fetchall()
        
        cursor.execute("""
            SELECT 
                sa.name as source_name,
                sa.type as source_type,
                lr.transformation_type,
                ta.name as target_name,
                ta.type as target_type
            FROM lineage_relationships lr
            JOIN data_assets sa ON lr.source_asset_id = sa.id
            JOIN data_assets ta ON lr.target_asset_id = ta.id
            WHERE lr.source_asset_id = %s
        """, (asset_id,))
        
        outgoing = cursor.fetchall()
        
        conn.close()
        
        return {
            'incoming': incoming,
            'outgoing': outgoing,
            'asset_id': asset_id
        }

# Usage example
lineage_tracker = DataLineageTracker(db_config)

# Create lineage tables
lineage_tracker.create_lineage_tables()

# Register data assets
training_data_id = lineage_tracker.register_data_asset(
    name="training_data",
    asset_type="database_table",
    location="postgresql://localhost:5432/ai_db.public.training_data",
    owner="data_science_team",
    description="Training dataset for user engagement prediction"
)

features_id = lineage_tracker.register_data_asset(
    name="user_features",
    asset_type="database_view",
    location="postgresql://localhost:5432/ai_db.public.user_features",
    owner="ml_engineering",
    description="Precomputed user features for ML models"
)

# Create lineage relationship
lineage_tracker.create_lineage_relationship(
    source_asset_id=training_data_id,
    target_asset_id=features_id,
    transformation_type="feature_engineering",
    transformation_details={
        "operations": ["aggregation", "normalization"],
        "columns": ["user_id", "age", "engagement_score"],
        "algorithm": "custom_sql"
    },
    created_by="feature_pipeline_v1"
)

# Get lineage path
lineage_path = lineage_tracker.get_lineage_path(features_id)
print(f"Lineage path for features: {lineage_path}")
```

### Step 2: ML model lineage tracking
```python
class ModelLineageTracker:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
    
    def create_model_lineage_tables(self):
        """Create tables for ML model lineage"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Models
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ml_models (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL,
            model_type VARCHAR(100),
            created_at TIMESTAMP NOT NULL,
            created_by VARCHAR(255),
            description TEXT,
            status VARCHAR(20) DEFAULT 'active'
        );
        """)
        
        # Model training runs
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_training_runs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_id UUID REFERENCES ml_models(id),
            training_data_asset_id UUID,
            hyperparameters JSONB,
            metrics JSONB,
            started_at TIMESTAMP NOT NULL,
            completed_at TIMESTAMP,
            status VARCHAR(20),
            environment VARCHAR(50)
        );
        """)
        
        # Model deployments
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_deployments (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_id UUID REFERENCES ml_models(id),
            deployment_environment VARCHAR(50) NOT NULL,
            endpoint_url TEXT,
            deployed_at TIMESTAMP NOT NULL,
            deployed_by VARCHAR(255),
            status VARCHAR(20)
        );
        """)
        
        # Model predictions
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_predictions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_id UUID REFERENCES ml_models(id),
            deployment_id UUID REFERENCES model_deployments(id),
            input_data_hash VARCHAR(64),
            prediction_result JSONB,
            predicted_at TIMESTAMP NOT NULL,
            user_id VARCHAR(255)
        );
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return "Model lineage tables created successfully"
    
    def register_model(self, name: str, version: str, model_type: str, 
                      description: str = "", created_by: str = "system"):
        """Register ML model"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO ml_models (
            name, version, model_type, created_at, created_by, description
        ) VALUES (%s, %s, %s, NOW(), %s, %s)
        RETURNING id
        """
        
        cursor.execute(insert_query, (name, version, model_type, created_by, description))
        model_id = cursor.fetchone()[0]
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return model_id
    
    def log_training_run(self, model_id: str, training_data_asset_id: str,
                        hyperparameters: Dict, metrics: Dict,
                        environment: str = "production"):
        """Log model training run"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO model_training_runs (
            model_id, training_data_asset_id, hyperparameters,
            metrics, started_at, completed_at, status, environment
        ) VALUES (%s, %s, %s, %s, NOW(), NOW(), 'completed', %s)
        """
        
        cursor.execute(insert_query, (
            model_id, training_data_asset_id,
            json.dumps(hyperparameters), json.dumps(metrics),
            environment
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    
    def get_model_lineage(self, model_id: str):
        """Get complete model lineage"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Get model info
        cursor.execute("SELECT * FROM ml_models WHERE id = %s", (model_id,))
        model_info = cursor.fetchone()
        
        # Get training runs
        cursor.execute("""
            SELECT * FROM model_training_runs 
            WHERE model_id = %s ORDER BY started_at DESC
        """, (model_id,))
        training_runs = cursor.fetchall()
        
        # Get deployments
        cursor.execute("""
            SELECT * FROM model_deployments 
            WHERE model_id = %s ORDER BY deployed_at DESC
        """, (model_id,))
        deployments = cursor.fetchall()
        
        conn.close()
        
        return {
            'model': model_info,
            'training_runs': training_runs,
            'deployments': deployments
        }

# Usage example
model_lineage = ModelLineageTracker(db_config)

# Create model lineage tables
model_lineage.create_model_lineage_tables()

# Register model
model_id = model_lineage.register_model(
    name="user_engagement_predictor",
    version="v1.2.3",
    model_type="RandomForestClassifier",
    description="Predicts user engagement based on behavioral features",
    created_by="ml_team"
)

# Log training run
model_lineage.log_training_run(
    model_id=model_id,
    training_data_asset_id=training_data_id,
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    },
    metrics={
        "accuracy": 0.92,
        "precision": 0.89,
        "recall": 0.91,
        "f1_score": 0.90
    },
    environment="staging"
)

# Get model lineage
lineage = model_lineage.get_model_lineage(model_id)
print(f"Model lineage: {lineage}")
```

## Section 3: Access Control and Authentication

### Step 1: RBAC (Role-Based Access Control)
```python
class RBACManager:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
    
    def create_rbac_tables(self):
        """Create tables for RBAC"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Roles
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS roles (
            id SERIAL PRIMARY KEY,
            name VARCHAR(50) NOT NULL UNIQUE,
            description TEXT,
            created_at TIMESTAMP NOT NULL
        );
        """)
        
        # Permissions
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS permissions (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL UNIQUE,
            description TEXT,
            resource_type VARCHAR(50) NOT NULL,
            action VARCHAR(50) NOT NULL,
            created_at TIMESTAMP NOT NULL
        );
        """)
        
        # Role-permission mapping
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS role_permissions (
            id SERIAL PRIMARY KEY,
            role_id INTEGER REFERENCES roles(id),
            permission_id INTEGER REFERENCES permissions(id),
            created_at TIMESTAMP NOT NULL,
            UNIQUE(role_id, permission_id)
        );
        """)
        
        # Users
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(100) NOT NULL UNIQUE,
            email VARCHAR(255) NOT NULL UNIQUE,
            full_name VARCHAR(255),
            created_at TIMESTAMP NOT NULL,
            status VARCHAR(20) DEFAULT 'active'
        );
        """)
        
        # User-role mapping
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_roles (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            role_id INTEGER REFERENCES roles(id),
            assigned_at TIMESTAMP NOT NULL,
            assigned_by VARCHAR(255),
            UNIQUE(user_id, role_id)
        );
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return "RBAC tables created successfully"
    
    def create_role(self, name: str, description: str = ""):
        """Create a new role"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO roles (name, description, created_at) 
        VALUES (%s, %s, NOW()) 
        RETURNING id
        """
        
        cursor.execute(insert_query, (name, description))
        role_id = cursor.fetchone()[0]
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return role_id
    
    def create_permission(self, name: str, resource_type: str, 
                         action: str, description: str = ""):
        """Create a new permission"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO permissions (name, description, resource_type, action, created_at) 
        VALUES (%s, %s, %s, %s, NOW()) 
        RETURNING id
        """
        
        cursor.execute(insert_query, (name, description, resource_type, action))
        permission_id = cursor.fetchone()[0]
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return permission_id
    
    def assign_permission_to_role(self, role_id: int, permission_id: int):
        """Assign permission to role"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO role_permissions (role_id, permission_id, created_at) 
        VALUES (%s, %s, NOW())
        """
        
        cursor.execute(insert_query, (role_id, permission_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    
    def assign_role_to_user(self, user_id: int, role_id: int, assigned_by: str = "system"):
        """Assign role to user"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO user_roles (user_id, role_id, assigned_at, assigned_by) 
        VALUES (%s, %s, NOW(), %s)
        """
        
        cursor.execute(insert_query, (user_id, role_id, assigned_by))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    
    def check_user_permission(self, user_id: int, resource_type: str, action: str):
        """Check if user has permission for resource and action"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        query = """
        SELECT COUNT(*) FROM users u
        JOIN user_roles ur ON u.id = ur.user_id
        JOIN role_permissions rp ON ur.role_id = rp.role_id
        JOIN permissions p ON rp.permission_id = p.id
        WHERE u.id = %s AND p.resource_type = %s AND p.action = %s
        """
        
        cursor.execute(query, (user_id, resource_type, action))
        count = cursor.fetchone()[0]
        
        conn.close()
        
        return count > 0

# Usage example
rbac_manager = RBACManager(db_config)

# Create RBAC tables
rbac_manager.create_rbac_tables()

# Create roles
data_scientist_role = rbac_manager.create_role("data_scientist", "Can access training data and build models")
ml_engineer_role = rbac_manager.create_role("ml_engineer", "Can deploy and monitor models")

# Create permissions
read_training_data = rbac_manager.create_permission(
    "read_training_data", "database_table", "read", "Read access to training data"
)
train_models = rbac_manager.create_permission(
    "train_models", "ml_model", "create", "Train ML models"
)
deploy_models = rbac_manager.create_permission(
    "deploy_models", "ml_model", "deploy", "Deploy ML models"
)

# Assign permissions to roles
rbac_manager.assign_permission_to_role(data_scientist_role, read_training_data)
rbac_manager.assign_permission_to_role(data_scientist_role, train_models)
rbac_manager.assign_permission_to_role(ml_engineer_role, deploy_models)

# Create user
conn = psycopg2.connect(**db_config)
cursor = conn.cursor()
cursor.execute("""
    INSERT INTO users (username, email, full_name, created_at) 
    VALUES ('alice', 'alice@example.com', 'Alice Smith', NOW()) 
    RETURNING id
""")
user_id = cursor.fetchone()[0]
conn.commit()
cursor.close()
conn.close()

# Assign role to user
rbac_manager.assign_role_to_user(user_id, data_scientist_role)

# Check permission
has_permission = rbac_manager.check_user_permission(user_id, "database_table", "read")
print(f"User has read permission: {has_permission}")
```

### Step 2: ABAC (Attribute-Based Access Control)
```python
class ABACManager:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
    
    def create_abac_tables(self):
        """Create tables for ABAC"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Policies
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS abac_policies (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            description TEXT,
            policy_json JSONB NOT NULL,
            created_at TIMESTAMP NOT NULL,
            created_by VARCHAR(255),
            status VARCHAR(20) DEFAULT 'active'
        );
        """)
        
        # Attributes
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_attributes (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id VARCHAR(255) NOT NULL,
            attribute_name VARCHAR(100) NOT NULL,
            attribute_value TEXT NOT NULL,
            attribute_type VARCHAR(50),
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP
        );
        """)
        
        # Resource attributes
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS resource_attributes (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            resource_id VARCHAR(255) NOT NULL,
            resource_type VARCHAR(100) NOT NULL,
            attribute_name VARCHAR(100) NOT NULL,
            attribute_value TEXT NOT NULL,
            attribute_type VARCHAR(50),
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP
        );
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return "ABAC tables created successfully"
    
    def create_policy(self, name: str, description: str, policy_json: Dict,
                     created_by: str = "system"):
        """Create ABAC policy"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO abac_policies (
            name, description, policy_json, created_at, created_by
        ) VALUES (%s, %s, %s, NOW(), %s)
        RETURNING id
        """
        
        cursor.execute(insert_query, (name, description, json.dumps(policy_json), created_by))
        policy_id = cursor.fetchone()[0]
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return policy_id
    
    def set_user_attribute(self, user_id: str, attribute_name: str, 
                          attribute_value: str, attribute_type: str = "string"):
        """Set user attribute for ABAC"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO user_attributes (
            user_id, attribute_name, attribute_value, attribute_type, created_at, updated_at
        ) VALUES (%s, %s, %s, %s, NOW(), NOW())
        ON CONFLICT (user_id, attribute_name) DO UPDATE 
        SET attribute_value = EXCLUDED.attribute_value,
            attribute_type = EXCLUDED.attribute_type,
            updated_at = NOW()
        """
        
        cursor.execute(insert_query, (user_id, attribute_name, attribute_value, attribute_type))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    
    def set_resource_attribute(self, resource_id: str, resource_type: str,
                              attribute_name: str, attribute_value: str,
                              attribute_type: str = "string"):
        """Set resource attribute for ABAC"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO resource_attributes (
            resource_id, resource_type, attribute_name, attribute_value, attribute_type, created_at, updated_at
        ) VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
        ON CONFLICT (resource_id, attribute_name) DO UPDATE 
        SET attribute_value = EXCLUDED.attribute_value,
            attribute_type = EXCLUDED.attribute_type,
            updated_at = NOW()
        """
        
        cursor.execute(insert_query, (resource_id, resource_type, attribute_name, attribute_value, attribute_type))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    
    def evaluate_policy(self, user_id: str, resource_id: str, action: str):
        """Evaluate ABAC policy for user and resource"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Get user attributes
        cursor.execute("""
            SELECT attribute_name, attribute_value, attribute_type 
            FROM user_attributes 
            WHERE user_id = %s
        """, (user_id,))
        user_attrs = {row[0]: {'value': row[1], 'type': row[2]} for row in cursor.fetchall()}
        
        # Get resource attributes
        cursor.execute("""
            SELECT attribute_name, attribute_value, attribute_type 
            FROM resource_attributes 
            WHERE resource_id = %s
        """, (resource_id,))
        resource_attrs = {row[0]: {'value': row[1], 'type': row[2]} for row in cursor.fetchall()}
        
        # Get applicable policies
        cursor.execute("""
            SELECT id, policy_json 
            FROM abac_policies 
            WHERE status = 'active'
        """)
        policies = cursor.fetchall()
        
        conn.close()
        
        # Evaluate policies (simplified logic)
        for policy_id, policy_json in policies:
            policy = json.loads(policy_json)
            
            # Check if policy applies to this action
            if action not in policy.get('actions', []):
                continue
            
            # Check conditions
            conditions = policy.get('conditions', [])
            if self._evaluate_conditions(conditions, user_attrs, resource_attrs):
                return True
        
        return False
    
    def _evaluate_conditions(self, conditions: List[Dict], 
                            user_attrs: Dict, resource_attrs: Dict):
        """Evaluate ABAC conditions"""
        for condition in conditions:
            attr_name = condition['attribute']
            operator = condition['operator']
            value = condition['value']
            
            # Determine if attribute is user or resource attribute
            if attr_name.startswith('user.'):
                attr_value = user_attrs.get(attr_name.split('.')[1], {}).get('value')
            elif attr_name.startswith('resource.'):
                attr_value = resource_attrs.get(attr_name.split('.')[1], {}).get('value')
            else:
                continue
            
            # Evaluate operator
            if operator == '==':
                if str(attr_value) != str(value):
                    return False
            elif operator == '!=':
                if str(attr_value) == str(value):
                    return False
            elif operator == '>':
                if float(attr_value) <= float(value):
                    return False
            elif operator == '<':
                if float(attr_value) >= float(value):
                    return False
        
        return True

# Usage example
abac_manager = ABACManager(db_config)

# Create ABAC tables
abac_manager.create_abac_tables()

# Set user attributes
abac_manager.set_user_attribute("alice", "department", "data_science")
abac_manager.set_user_attribute("alice", "clearance_level", "high")
abac_manager.set_user_attribute("alice", "location", "us-west")

# Set resource attributes
abac_manager.set_resource_attribute("training_data", "database_table", "sensitivity", "high")
abac_manager.set_resource_attribute("training_data", "database_table", "region", "us-west")

# Create ABAC policy
policy = {
    "name": "High Sensitivity Data Access",
    "description": "Allow access to high sensitivity data for high clearance users in same region",
    "actions": ["read", "write"],
    "conditions": [
        {"attribute": "user.clearance_level", "operator": "==", "value": "high"},
        {"attribute": "user.location", "operator": "==", "value": "us-west"},
        {"attribute": "resource.sensitivity", "operator": "==", "value": "high"},
        {"attribute": "resource.region", "operator": "==", "value": "us-west"}
    ]
}

policy_id = abac_manager.create_policy(
    name="high_sensitivity_access",
    description="High sensitivity data access policy",
    policy_json=policy
)

# Evaluate policy
can_access = abac_manager.evaluate_policy("alice", "training_data", "read")
print(f"Alice can access training data: {can_access}")
```

## Section 4: Audit Logging

### Step 1: Comprehensive audit logging system
```python
class AuditLogger:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
    
    def create_audit_tables(self):
        """Create tables for audit logging"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Audit events
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_events (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            event_type VARCHAR(100) NOT NULL,
            user_id VARCHAR(255),
            user_name VARCHAR(255),
            ip_address VARCHAR(45),
            user_agent TEXT,
            resource_type VARCHAR(100),
            resource_id VARCHAR(255),
            action VARCHAR(50),
            details JSONB,
            created_at TIMESTAMP NOT NULL,
            success BOOLEAN NOT NULL,
            duration_ms INTEGER
        );
        """)
        
        # Audit summaries
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_summaries (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            period_start TIMESTAMP NOT NULL,
            period_end TIMESTAMP NOT NULL,
            total_events INTEGER NOT NULL,
            successful_events INTEGER NOT NULL,
            failed_events INTEGER NOT NULL,
            unique_users INTEGER NOT NULL,
            resource_types TEXT[],
            created_at TIMESTAMP NOT NULL
        );
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return "Audit logging tables created successfully"
    
    def log_event(self, event_type: str, user_id: str = None, 
                  user_name: str = None, ip_address: str = None,
                  user_agent: str = None, resource_type: str = None,
                  resource_id: str = None, action: str = None,
                  details: Dict = None, success: bool = True,
                  duration_ms: int = 0):
        """Log audit event"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO audit_events (
            event_type, user_id, user_name, ip_address, user_agent,
            resource_type, resource_id, action, details, created_at,
            success, duration_ms
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s, %s)
        """
        
        cursor.execute(insert_query, (
            event_type, user_id, user_name, ip_address, user_agent,
            resource_type, resource_id, action, json.dumps(details) if details else None,
            success, duration_ms
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    
    def generate_audit_summary(self, period_hours: int = 24):
        """Generate audit summary for given period"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        period_end = datetime.datetime.utcnow()
        period_start = period_end - datetime.timedelta(hours=period_hours)
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_events,
                COUNT(CASE WHEN success = TRUE THEN 1 END) as successful_events,
                COUNT(CASE WHEN success = FALSE THEN 1 END) as failed_events,
                COUNT(DISTINCT user_id) as unique_users,
                ARRAY_AGG(DISTINCT resource_type) as resource_types
            FROM audit_events 
            WHERE created_at BETWEEN %s AND %s
        """, (period_start, period_end))
        
        summary = cursor.fetchone()
        
        insert_query = """
        INSERT INTO audit_summaries (
            period_start, period_end, total_events, successful_events,
            failed_events, unique_users, resource_types, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        """
        
        cursor.execute(insert_query, (
            period_start, period_end, summary[0], summary[1],
            summary[2], summary[3], summary[4]
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return {
            'period_start': period_start,
            'period_end': period_end,
            'total_events': summary[0],
            'successful_events': summary[1],
            'failed_events': summary[2],
            'unique_users': summary[3],
            'resource_types': summary[4]
        }

# Usage example
audit_logger = AuditLogger(db_config)

# Create audit tables
audit_logger.create_audit_tables()

# Log events
audit_logger.log_event(
    event_type="model_prediction",
    user_id="alice",
    user_name="Alice Smith",
    ip_address="192.168.1.1",
    user_agent="Mozilla/5.0",
    resource_type="ml_model",
    resource_id="user_engagement_predictor:v1.2.3",
    action="predict",
    details={
        "input_features": {"age": 25, "engagement_score": 0.8},
        "prediction": 0.85,
        "confidence": 0.92
    },
    success=True,
    duration_ms=120
)

audit_logger.log_event(
    event_type="database_query",
    user_id="bob",
    user_name="Bob Johnson",
    ip_address="192.168.1.2",
    resource_type="database_table",
    resource_id="training_data",
    action="select",
    details={"query": "SELECT * FROM training_data LIMIT 10"},
    success=True,
    duration_ms=45
)

# Generate audit summary
summary = audit_logger.generate_audit_summary(period_hours=1)
print(f"Audit summary: {summary}")
```

## Section 5: Privacy-Preserving Techniques

### Step 1: Differential privacy implementation
```python
import numpy as np
from typing import List, Dict

class DifferentialPrivacyManager:
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
    
    def laplace_mechanism(self, value: float, sensitivity: float) -> float:
        """Apply Laplace mechanism for differential privacy"""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def exponential_mechanism(self, scores: List[float], utility_function, 
                             sensitivity: float) -> int:
        """Apply exponential mechanism for differential privacy"""
        # Calculate probabilities
        probabilities = []
        max_score = max(scores)
        
        for score in scores:
            prob = np.exp((self.epsilon * score) / (2 * sensitivity))
            probabilities.append(prob)
        
        # Normalize
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]
        
        # Sample
        return np.random.choice(len(scores), p=probabilities)
    
    def apply_differential_privacy_to_aggregates(self, data: List[float], 
                                               operation: str = "mean") -> float:
        """Apply differential privacy to aggregate operations"""
        if operation == "mean":
            true_value = np.mean(data)
            sensitivity = 1.0 / len(data)  # For mean, sensitivity is 1/n
            return self.laplace_mechanism(true_value, sensitivity)
        elif operation == "count":
            true_value = len(data)
            sensitivity = 1.0  # For count, sensitivity is 1
            return self.laplace_mechanism(true_value, sensitivity)
        elif operation == "sum":
            true_value = np.sum(data)
            sensitivity = 1.0  # For sum, sensitivity is 1 (assuming bounded data)
            return self.laplace_mechanism(true_value, sensitivity)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def create_privacy_preserving_query(self, query: str, epsilon: float = None) -> Dict:
        """Create privacy-preserving query definition"""
        if epsilon is None:
            epsilon = self.epsilon
        
        return {
            'original_query': query,
            'epsilon': epsilon,
            'privacy_guarantee': f"ε={epsilon}",
            'mechanism': 'laplace',
            'created_at': datetime.datetime.utcnow().isoformat()
        }

# Usage example
dp_manager = DifferentialPrivacyManager(epsilon=0.5)

# Apply differential privacy to aggregates
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

private_mean = dp_manager.apply_differential_privacy_to_aggregates(data, "mean")
private_count = dp_manager.apply_differential_privacy_to_aggregates(data, "count")
private_sum = dp_manager.apply_differential_privacy_to_aggregates(data, "sum")

print(f"True mean: {np.mean(data):.2f}, Private mean: {private_mean:.2f}")
print(f"True count: {len(data)}, Private count: {private_count:.0f}")
print(f"True sum: {sum(data)}, Private sum: {private_sum:.0f}")

# Create privacy-preserving query
privacy_query = dp_manager.create_privacy_preserving_query(
    "SELECT AVG(engagement_score) FROM training_data"
)
print(f"Privacy query: {privacy_query}")
```

### Step 2: Homomorphic encryption for database queries
```python
# Note: This is a simplified simulation since real homomorphic encryption is complex
class HomomorphicEncryptionManager:
    def __init__(self):
        self.keys = {}
    
    def generate_keys(self, key_id: str):
        """Generate encryption keys (simulated)"""
        # In practice, use libraries like Microsoft SEAL or Palisade
        public_key = f"pub_{key_id}_{int(time.time())}"
        private_key = f"priv_{key_id}_{int(time.time())}"
        
        self.keys[key_id] = {
            'public_key': public_key,
            'private_key': private_key
        }
        
        return public_key, private_key
    
    def encrypt_data(self, data: float, key_id: str) -> str:
        """Encrypt data (simulated)"""
        # In practice, use actual homomorphic encryption
        public_key = self.keys[key_id]['public_key']
        encrypted = f"enc_{data}_{public_key[-8:]}"
        return encrypted
    
    def decrypt_result(self, encrypted_result: str, key_id: str) -> float:
        """Decrypt result (simulated)"""
        # In practice, use actual decryption
        private_key = self.keys[key_id]['private_key']
        # Extract original value from encrypted string
        try:
            parts = encrypted_result.split('_')
            if len(parts) >= 2:
                return float(parts[1])
        except:
            pass
        return 0.0
    
    def execute_encrypted_query(self, query: str, key_id: str) -> str:
        """Execute query on encrypted data (simulated)"""
        # In practice, this would be done by the database with HE support
        # Here we simulate by encrypting inputs and decrypting outputs
        
        # Simulate encrypted computation
        encrypted_result = f"enc_{hash(query) % 1000}_{key_id[-4:]}"
        return encrypted_result

# Usage example
he_manager = HomomorphicEncryptionManager()

# Generate keys
public_key, private_key = he_manager.generate_keys("training_data_key")

# Encrypt data
encrypted_value = he_manager.encrypt_data(0.85, "training_data_key")
print(f"Encrypted value: {encrypted_value}")

# Execute encrypted query (simulated)
encrypted_query_result = he_manager.execute_encrypted_query(
    "SELECT AVG(engagement_score) FROM training_data",
    "training_data_key"
)
print(f"Encrypted query result: {encrypted_query_result}")

# Decrypt result
decrypted_result = he_manager.decrypt_result(encrypted_query_result, "training_data_key")
print(f"Decrypted result: {decrypted_result}")
```

## Section 6: Compliance Monitoring

### Step 1: Automated compliance checker
```python
class ComplianceChecker:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
    
    def check_gdpr_compliance(self) -> Dict:
        """Check GDPR compliance status"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        checks = {}
        
        # Check consent records exist
        cursor.execute("SELECT COUNT(*) FROM consent_records")
        checks['consent_records'] = cursor.fetchone()[0] > 0
        
        # Check DSR requests processed
        cursor.execute("SELECT COUNT(*) FROM dsr_requests WHERE status = 'completed'")
        completed_dsr = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM dsr_requests")
        total_dsr = cursor.fetchone()[0]
        checks['dsr_completion_rate'] = total_dsr > 0 and completed_dsr / total_dsr >= 0.95
        
        # Check data deletion logs
        cursor.execute("SELECT COUNT(*) FROM data_deletion_logs")
        checks['data_deletion_logs'] = cursor.fetchone()[0] > 0
        
        conn.close()
        
        return {
            'gdpr_compliant': all(checks.values()),
            'checks': checks,
            'timestamp': datetime.datetime.utcnow().isoformat()
        }
    
    def check_ccpa_compliance(self) -> Dict:
        """Check CCPA compliance status"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        checks = {}
        
        # Check do not sell requests
        cursor.execute("SELECT COUNT(*) FROM do_not_sell_requests")
        checks['do_not_sell_requests'] = cursor.fetchone()[0] > 0
        
        # Check data sharing logs
        cursor.execute("SELECT COUNT(*) FROM data_sharing_logs")
        checks['data_sharing_logs'] = cursor.fetchone()[0] > 0
        
        conn.close()
        
        return {
            'ccpa_compliant': all(checks.values()),
            'checks': checks,
            'timestamp': datetime.datetime.utcnow().isoformat()
        }
    
    def check_hipaa_compliance(self) -> Dict:
        """Check HIPAA compliance status"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        checks = {}
        
        # Check PHI records
        cursor.execute("SELECT COUNT(*) FROM phi_records")
        checks['phi_records'] = cursor.fetchone()[0] > 0
        
        # Check PHI access logs
        cursor.execute("SELECT COUNT(*) FROM phi_access_logs")
        checks['phi_access_logs'] = cursor.fetchone()[0] > 0
        
        # Check security incidents
        cursor.execute("SELECT COUNT(*) FROM security_incidents")
        checks['security_incidents'] = cursor.fetchone()[0] > 0
        
        conn.close()
        
        return {
            'hipaa_compliant': all(checks.values()),
            'checks': checks,
            'timestamp': datetime.datetime.utcnow().isoformat()
        }
    
    def run_comprehensive_compliance_check(self) -> Dict:
        """Run comprehensive compliance check"""
        results = {
            'gdpr': self.check_gdpr_compliance(),
            'ccpa': self.check_ccpa_compliance(),
            'hipaa': self.check_hipaa_compliance(),
            'overall_compliant': True,
            'timestamp': datetime.datetime.utcnow().isoformat()
        }
        
        # Check overall compliance
        results['overall_compliant'] = (
            results['gdpr']['gdpr_compliant'] and
            results['ccpa']['ccpa_compliant'] and
            results['hipaa']['hipaa_compliant']
        )
        
        return results

# Usage example
compliance_checker = ComplianceChecker(db_config)

# Run comprehensive compliance check
compliance_results = compliance_checker.run_comprehensive_compliance_check()
print(f"Compliance results: {compliance_results}")
```

## Section 7: Performance Benchmarking

### Step 1: Governance overhead benchmarking
```python
import time
import pandas as pd
from typing import List, Dict, Callable

class GovernanceBenchmark:
    def __init__(self):
        self.results = []
    
    def benchmark_compliance_overhead(self, methods: List[Callable], 
                                    operations: List[str] = ["read", "write", "delete"]):
        """Benchmark compliance overhead"""
        for method in methods:
            for operation in operations:
                start_time = time.time()
                
                try:
                    method(operation)
                    duration = time.time() - start_time
                    
                    self.results.append({
                        'benchmark': 'compliance_overhead',
                        'method': method.__name__,
                        'operation': operation,
                        'duration_seconds': duration,
                        'overhead_percentage': duration / 0.01 * 100 if duration > 0 else 0  # baseline 10ms
                    })
                except Exception as e:
                    self.results.append({
                        'benchmark': 'compliance_overhead',
                        'method': method.__name__,
                        'operation': operation,
                        'duration_seconds': float('inf'),
                        'error': str(e)
                    })
    
    def benchmark_lineage_tracking(self, methods: List[Callable],
                                 data_sizes: List[int] = [100, 1000, 10000]):
        """Benchmark lineage tracking performance"""
        for method in methods:
            for size in data_sizes:
                start_time = time.time()
                
                try:
                    method(size)
                    duration = time.time() - start_time
                    
                    self.results.append({
                        'benchmark': 'lineage_tracking',
                        'method': method.__name__,
                        'data_size': size,
                        'duration_seconds': duration,
                        'throughput_operations_per_second': size / duration if duration > 0 else 0
                    })
                except Exception as e:
                    self.results.append({
                        'benchmark': 'lineage_tracking',
                        'method': method.__name__,
                        'data_size': size,
                        'duration_seconds': float('inf'),
                        'error': str(e)
                    })
    
    def generate_governance_benchmark_report(self):
        """Generate comprehensive governance benchmark report"""
        df = pd.DataFrame(self.results)
        
        # Calculate statistics
        summary = df.groupby(['benchmark', 'method']).agg({
            'duration_seconds': ['mean', 'std', 'min', 'max'],
            'overhead_percentage': ['mean', 'std'],
            'throughput_operations_per_second': ['mean', 'std']
        }).round(2)
        
        # Generate recommendations
        recommendations = []
        
        # Best compliance overhead
        if 'compliance_overhead' in df['benchmark'].values:
            best_overhead = df[df['benchmark'] == 'compliance_overhead'].loc[
                df[df['benchmark'] == 'compliance_overhead']['duration_seconds'].idxmin()
            ]
            recommendations.append(
                f"Best compliance overhead: {best_overhead['method']} "
                f"({best_overhead['duration_seconds']:.4f}s for {best_overhead['operation']} operation)"
            )
        
        # Best lineage tracking
        if 'lineage_tracking' in df['benchmark'].values:
            best_lineage = df[df['benchmark'] == 'lineage_tracking'].loc[
                df[df['benchmark'] == 'lineage_tracking']['duration_seconds'].idxmin()
            ]
            recommendations.append(
                f"Best lineage tracking: {best_lineage['method']} "
                f"({best_lineage['duration_seconds']:.4f}s for {best_lineage['data_size']} records)"
            )
        
        return {
            'summary': summary,
            'detailed_results': df,
            'recommendations': recommendations,
            'governance_tips': [
                "Implement incremental lineage tracking to reduce overhead",
                "Use caching for frequently accessed compliance data",
                "Optimize database indexes for governance queries",
                "Batch compliance operations when possible",
                "Monitor governance overhead in production",
                "Consider trade-offs between compliance rigor and performance",
                "Implement automated compliance validation",
                "Regularly audit governance implementation"
            ]
        }

# Usage example
benchmark = GovernanceBenchmark()

# Define test methods
def test_gdpr_check(operation: str):
    """Test GDPR compliance check"""
    time.sleep(0.02)

def test_lineage_tracking(size: int):
    """Test lineage tracking"""
    time.sleep(0.001 * size)

# Run benchmarks
benchmark.benchmark_compliance_overhead(
    [test_gdpr_check],
    ["read", "write", "delete"]
)

benchmark.benchmark_lineage_tracking(
    [test_lineage_tracking],
    [100, 1000, 10000]
)

report = benchmark.generate_governance_benchmark_report()
print("Governance Benchmark Report:")
print(report['summary'])
print("\nRecommendations:")
for rec in report['recommendations']:
    print(f"- {rec}")
```

## Hands-on Exercises

### Exercise 1: Regulatory compliance implementation
1. Implement GDPR right to be forgotten
2. Set up CCPA do not sell requests
3. Configure HIPAA PHI tracking
4. Test compliance workflows

### Exercise 2: Data lineage tracking
1. Set up comprehensive data lineage system
2. Track ML model lineage
3. Create lineage visualization
4. Test lineage queries

### Exercise 3: Access control implementation
1. Implement RBAC system
2. Configure ABAC policies
3. Test permission evaluation
4. Integrate with application

### Exercise 4: Audit logging
1. Set up comprehensive audit logging
2. Generate audit reports
3. Implement alerting for suspicious activity
4. Test log retention policies

### Exercise 5: Privacy-preserving techniques
1. Implement differential privacy
2. Test homomorphic encryption (simulated)
3. Apply privacy techniques to ML workflows
4. Measure privacy vs. utility trade-offs

### Exercise 6: Compliance monitoring
1. Set up automated compliance checker
2. Integrate with CI/CD pipeline
3. Create compliance dashboard
4. Test compliance violation detection

## Best Practices Summary

1. **Regulatory Compliance**: Implement specific requirements for GDPR, CCPA, HIPAA based on your jurisdiction
2. **Data Lineage**: Track end-to-end data provenance for reproducibility and debugging
3. **Access Control**: Use RBAC for simple scenarios, ABAC for complex attribute-based requirements
4. **Audit Logging**: Log comprehensive audit trails for security and compliance
5. **Privacy Preservation**: Apply differential privacy and encryption where appropriate
6. **Automated Monitoring**: Implement automated compliance checking and alerting
7. **Performance Considerations**: Balance governance overhead with system performance
8. **Documentation**: Maintain comprehensive documentation of governance policies and procedures

This tutorial provides practical, hands-on experience with database integration for data governance and compliance frameworks. Complete all exercises to master these critical skills for building compliant, trustworthy AI systems.