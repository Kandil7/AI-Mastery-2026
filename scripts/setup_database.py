"""
Database Setup Script

This script initializes the PostgreSQL database schema for the AI-Mastery-2026 platform.
Includes tables for prediction logging, experiment tracking, and model versioning.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection settings
DATABASE_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432'),
    'database': os.environ.get('DB_NAME', 'ai_mastery'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', 'postgres'),
}


def get_connection_string() -> str:
    """Get PostgreSQL connection string."""
    return (
        f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}"
        f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
    )


# SQL Schema Definitions
SCHEMA_SQL = """
-- ============================================
-- AI-Mastery-2026 Database Schema
-- ============================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- Models Table
-- ============================================
CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    model_type VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    description TEXT,
    file_path VARCHAR(500),
    n_features INTEGER,
    n_classes INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);
CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type);
CREATE INDEX IF NOT EXISTS idx_models_active ON models(is_active);

-- ============================================
-- Predictions Table
-- ============================================
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES models(id) ON DELETE SET NULL,
    model_name VARCHAR(255) NOT NULL,
    input_features JSONB NOT NULL,
    prediction JSONB NOT NULL,
    probabilities JSONB,
    processing_time_ms FLOAT,
    request_id VARCHAR(100),
    client_ip VARCHAR(45),
    user_agent VARCHAR(500),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model_name ON predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_request_id ON predictions(request_id);

-- ============================================
-- Experiments Table
-- ============================================
CREATE TABLE IF NOT EXISTS experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'created',
    parameters JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    artifacts JSONB DEFAULT '[]',
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(name);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);

-- ============================================
-- Training Runs Table
-- ============================================
CREATE TABLE IF NOT EXISTS training_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID REFERENCES experiments(id) ON DELETE CASCADE,
    model_name VARCHAR(255) NOT NULL,
    hyperparameters JSONB DEFAULT '{}',
    train_metrics JSONB DEFAULT '{}',
    val_metrics JSONB DEFAULT '{}',
    test_metrics JSONB DEFAULT '{}',
    duration_seconds FLOAT,
    status VARCHAR(50) DEFAULT 'running',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_training_runs_experiment ON training_runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);

-- ============================================
-- API Metrics Table
-- ============================================
CREATE TABLE IF NOT EXISTS api_metrics (
    id SERIAL PRIMARY KEY,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms FLOAT NOT NULL,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    client_ip VARCHAR(45),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_api_metrics_endpoint ON api_metrics(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_metrics_created ON api_metrics(created_at);

-- ============================================
-- System Events Table
-- ============================================
CREATE TABLE IF NOT EXISTS system_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    event_source VARCHAR(100) NOT NULL,
    severity VARCHAR(20) DEFAULT 'info',
    message TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(event_type);
CREATE INDEX IF NOT EXISTS idx_system_events_severity ON system_events(severity);
CREATE INDEX IF NOT EXISTS idx_system_events_created ON system_events(created_at);

-- ============================================
-- Views for Analytics
-- ============================================
CREATE OR REPLACE VIEW prediction_stats AS
SELECT 
    model_name,
    DATE(created_at) as date,
    COUNT(*) as prediction_count,
    AVG(processing_time_ms) as avg_processing_time_ms,
    MIN(processing_time_ms) as min_processing_time_ms,
    MAX(processing_time_ms) as max_processing_time_ms
FROM predictions
GROUP BY model_name, DATE(created_at);

CREATE OR REPLACE VIEW api_health AS
SELECT 
    endpoint,
    DATE(created_at) as date,
    COUNT(*) as request_count,
    AVG(response_time_ms) as avg_response_time_ms,
    COUNT(CASE WHEN status_code < 400 THEN 1 END) as success_count,
    COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_count,
    ROUND(COUNT(CASE WHEN status_code < 400 THEN 1 END)::decimal / COUNT(*) * 100, 2) as success_rate
FROM api_metrics
GROUP BY endpoint, DATE(created_at);

-- ============================================
-- Functions
-- ============================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for auto-updating timestamps
DROP TRIGGER IF EXISTS update_models_updated_at ON models;
CREATE TRIGGER update_models_updated_at
    BEFORE UPDATE ON models
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS update_experiments_updated_at ON experiments;
CREATE TRIGGER update_experiments_updated_at
    BEFORE UPDATE ON experiments
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- ============================================
-- Insert default data
-- ============================================
INSERT INTO models (name, model_type, version, description, n_features)
VALUES 
    ('classification_model', 'RandomForestClassifier', '1.0.0', 'Binary classification model', 10),
    ('regression_model', 'GradientBoostingRegressor', '1.0.0', 'Regression model', 10),
    ('logistic_model', 'LogisticRegression', '1.0.0', 'Logistic regression classifier', 5)
ON CONFLICT (name) DO NOTHING;

-- Log schema creation
INSERT INTO system_events (event_type, event_source, severity, message)
VALUES ('schema_created', 'setup_database.py', 'info', 'Database schema initialized successfully');

SELECT 'Database schema initialized successfully!' as status;
"""


def setup_database_with_psycopg2():
    """Set up database using psycopg2."""
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        
        logger.info("Connecting to PostgreSQL...")
        
        # First, try to create the database if it doesn't exist
        conn = psycopg2.connect(
            host=DATABASE_CONFIG['host'],
            port=DATABASE_CONFIG['port'],
            user=DATABASE_CONFIG['user'],
            password=DATABASE_CONFIG['password'],
            database='postgres'
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
            (DATABASE_CONFIG['database'],)
        )
        exists = cursor.fetchone()
        
        if not exists:
            logger.info(f"Creating database '{DATABASE_CONFIG['database']}'...")
            cursor.execute(f"CREATE DATABASE {DATABASE_CONFIG['database']}")
            logger.info("Database created successfully")
        else:
            logger.info(f"Database '{DATABASE_CONFIG['database']}' already exists")
        
        cursor.close()
        conn.close()
        
        # Now connect to the actual database and run schema
        logger.info("Connecting to target database...")
        conn = psycopg2.connect(
            host=DATABASE_CONFIG['host'],
            port=DATABASE_CONFIG['port'],
            user=DATABASE_CONFIG['user'],
            password=DATABASE_CONFIG['password'],
            database=DATABASE_CONFIG['database']
        )
        
        cursor = conn.cursor()
        
        logger.info("Executing schema SQL...")
        cursor.execute(SCHEMA_SQL)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("Database setup completed successfully!")
        return True
        
    except ImportError:
        logger.warning("psycopg2 not installed. Install with: pip install psycopg2-binary")
        return False
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False


def setup_database_with_sqlalchemy():
    """Set up database using SQLAlchemy."""
    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.exc import ProgrammingError
        
        logger.info("Setting up database with SQLAlchemy...")
        
        # Create engine
        engine = create_engine(get_connection_string())
        
        # Execute schema
        with engine.connect() as conn:
            conn.execute(text(SCHEMA_SQL))
            conn.commit()
        
        logger.info("Database setup completed successfully!")
        return True
        
    except ImportError:
        logger.warning("SQLAlchemy not installed. Install with: pip install sqlalchemy")
        return False
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False


def export_schema_to_file(output_path: str = "schema.sql"):
    """Export schema to a SQL file."""
    with open(output_path, 'w') as f:
        f.write(f"-- AI-Mastery-2026 Database Schema\n")
        f.write(f"-- Generated: {datetime.now().isoformat()}\n\n")
        f.write(SCHEMA_SQL)
    logger.info(f"Schema exported to: {output_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Set up PostgreSQL database')
    parser.add_argument('--export-only', action='store_true',
                       help='Only export schema to file, do not execute')
    parser.add_argument('--output', type=str, default='schema.sql',
                       help='Output file for schema export')
    parser.add_argument('--driver', type=str, choices=['psycopg2', 'sqlalchemy'],
                       default='psycopg2', help='Database driver to use')
    
    args = parser.parse_args()
    
    if args.export_only:
        export_schema_to_file(args.output)
        return
    
    logger.info("=" * 50)
    logger.info("AI-Mastery-2026 Database Setup")
    logger.info("=" * 50)
    logger.info(f"Host: {DATABASE_CONFIG['host']}")
    logger.info(f"Port: {DATABASE_CONFIG['port']}")
    logger.info(f"Database: {DATABASE_CONFIG['database']}")
    logger.info(f"User: {DATABASE_CONFIG['user']}")
    logger.info("=" * 50)
    
    if args.driver == 'psycopg2':
        success = setup_database_with_psycopg2()
    else:
        success = setup_database_with_sqlalchemy()
    
    if not success:
        logger.info("Falling back to exporting schema file...")
        export_schema_to_file(args.output)
        logger.info(f"Run the following to set up the database manually:")
        logger.info(f"  psql -h {DATABASE_CONFIG['host']} -U {DATABASE_CONFIG['user']} -d {DATABASE_CONFIG['database']} -f {args.output}")
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
