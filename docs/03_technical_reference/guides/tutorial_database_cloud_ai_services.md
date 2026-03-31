# Database Integration with Cloud AI Services Tutorial

## Overview

This tutorial focuses on integrating databases with major cloud AI services: AWS SageMaker, Google Vertex AI, and Azure Machine Learning. We'll cover cloud-native database integration patterns, serverless architectures, and production deployment strategies specifically for senior AI/ML engineers building scalable AI systems.

## Prerequisites
- AWS/GCP/Azure account with appropriate permissions
- Python 3.8+
- Boto3 (AWS), google-cloud-aiplatform (GCP), azureml-sdk (Azure)
- PostgreSQL/MySQL/Aurora/Cloud SQL/SQL Database
- Basic understanding of cloud AI services

## Tutorial Structure
1. **AWS SageMaker Integration** - Database-backed ML workflows
2. **Google Vertex AI Integration** - BigQuery and AI platform integration
3. **Azure Machine Learning Integration** - SQL Database and AI services
4. **Cross-Cloud Patterns** - Unified integration strategies
5. **Serverless Architecture** - Lambda, Cloud Functions, Azure Functions
6. **Performance Benchmarking** - Cloud AI service comparison

## Section 1: AWS SageMaker Integration

### Step 1: SageMaker with RDS/Aurora integration
```python
import boto3
import pandas as pd
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.estimator import SKLearn
import os

class AWSSageMakerDatabaseIntegration:
    def __init__(self, region_name: str = "us-west-2"):
        self.region_name = region_name
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.rds_client = boto3.client('rds', region_name=region_name)
        self.sagemaker_session = sagemaker.Session()
        self.role = get_execution_role()
    
    def create_sagemaker_data_pipeline(self, rds_instance_identifier: str,
                                     database_name: str, 
                                     table_name: str,
                                     s3_bucket: str,
                                     s3_prefix: str):
        """Create data pipeline from RDS to S3 for SageMaker"""
        # Get RDS endpoint
        response = self.rds_client.describe_db_instances(
            DBInstanceIdentifier=rds_instance_identifier
        )
        endpoint = response['DBInstances'][0]['Endpoint']['Address']
        port = response['DBInstances'][0]['Endpoint']['Port']
        
        # Create temporary credentials
        temp_credentials = self.rds_client.generate_db_auth_token(
            DBHostname=endpoint,
            Port=port,
            DBUsername='admin',
            Region=self.region_name
        )
        
        # Generate SQL query for data extraction
        query = f"SELECT * FROM {table_name} WHERE split = 'train'"
        
        # Use AWS Glue or direct extraction (simplified)
        # In practice, use AWS Glue for ETL or Lambda for custom extraction
        
        # Save to S3
        s3_key = f"{s3_prefix}/training_data.csv"
        self.s3_client.put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            Body="id,feature1,feature2,label\n1,0.5,0.7,1\n2,0.3,0.9,0"
        )
        
        return f"Data pipeline created: s3://{s3_bucket}/{s3_key}"
    
    def train_sagemaker_model_from_database(self, 
                                          rds_instance_identifier: str,
                                          database_name: str,
                                          table_name: str,
                                          s3_bucket: str,
                                          entry_point: str = "train.py"):
        """Train SageMaker model using database data"""
        # Create data pipeline first
        data_pipeline = self.create_sagemaker_data_pipeline(
            rds_instance_identifier, database_name, table_name,
            s3_bucket, "sagemaker-input"
        )
        
        # Define SageMaker estimator
        sklearn = SKLearn(
            entry_point=entry_point,
            role=self.role,
            instance_count=1,
            instance_type='ml.m5.xlarge',
            framework_version='1.0-1',
            py_version='py3',
            base_job_name='database-training',
            sagemaker_session=self.sagemaker_session
        )
        
        # Train
        train_data = f"s3://{s3_bucket}/sagemaker-input/"
        sklearn.fit({'train': train_data})
        
        return sklearn
    
    def deploy_sagemaker_endpoint(self, model_name: str, instance_type: str = 'ml.m5.large'):
        """Deploy SageMaker endpoint"""
        # Get model from training job
        model = self.sagemaker_session.create_model(
            name=model_name,
            role=self.role,
            image_uri=f"763104351884.dkr.ecr.{self.region_name}.amazonaws.com/pytorch-inference:2.0.0-cpu-py310"
        )
        
        # Create endpoint config
        endpoint_config = self.sagemaker_session.create_endpoint_config(
            name=f"{model_name}-config",
            models=[{'ModelName': model_name}],
            initial_instance_count=1,
            instance_type=instance_type
        )
        
        # Create endpoint
        endpoint = self.sagemaker_session.create_endpoint(
            endpoint_name=f"{model_name}-endpoint",
            config_name=f"{model_name}-config"
        )
        
        return endpoint

# Usage example
sagemaker_integration = AWSSageMakerDatabaseIntegration("us-west-2")

# Train model
# estimator = sagemaker_integration.train_sagemaker_model_from_database(
#     rds_instance_identifier="my-rds-instance",
#     database_name="ai_db",
#     table_name="training_data",
#     s3_bucket="my-sagemaker-bucket"
# )
```

### Step 2: SageMaker Feature Store integration
```python
from sagemaker.feature_store.feature_group import FeatureGroup
import pandas as pd

class SageMakerFeatureStoreIntegration:
    def __init__(self, region_name: str = "us-west-2"):
        self.region_name = region_name
        self.sagemaker_session = sagemaker.Session()
        self.feature_store = boto3.client('sagemaker-featurestore-runtime', region_name=region_name)
    
    def create_feature_group(self, feature_group_name: str,
                           record_identifier_name: str,
                           event_time_feature_name: str,
                           feature_definitions: List[Dict]):
        """Create SageMaker Feature Group"""
        feature_group = FeatureGroup(
            name=feature_group_name,
            record_identifier_name=record_identifier_name,
            event_time_feature_name=event_time_feature_name,
            sagemaker_session=self.sagemaker_session,
            role_arn=get_execution_role()
        )
        
        # Create feature group
        feature_group.create(
            feature_definitions=feature_definitions,
            online=True,
            enable_online=True,
            offline=True
        )
        
        return feature_group
    
    def ingest_data_from_database(self, feature_group_name: str,
                                db_config: Dict,
                                query: str):
        """Ingest data from database to Feature Store"""
        # Connect to database
        conn = psycopg2.connect(**db_config)
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Convert to records
        records = []
        for _, row in df.iterrows():
            record = []
            for col in df.columns:
                record.append({
                    'feature_name': col,
                    'value_as_string': str(row[col])
                })
            records.append(record)
        
        # Ingest to Feature Store
        feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=self.sagemaker_session)
        feature_group.put_records(records=records)
        
        return f"Ingested {len(records)} records to {feature_group_name}"

# Usage example
feature_store_integration = SageMakerFeatureStoreIntegration()

# Define features
features = [
    {'FeatureName': 'user_id', 'ValueType': 'Integral'},
    {'FeatureName': 'age', 'ValueType': 'Integral'},
    {'FeatureName': 'engagement_score', 'ValueType': 'Fractional'},
    {'FeatureName': 'session_count', 'ValueType': 'Integral'},
    {'FeatureName': 'label', 'ValueType': 'Integral'}
]

# Create feature group
feature_group = feature_store_integration.create_feature_group(
    feature_group_name="user_engagement_features",
    record_identifier_name="user_id",
    event_time_feature_name="event_time",
    feature_definitions=features
)

# Ingest data
db_config = {
    'host': 'my-rds-instance.cluster-xyz.us-west-2.rds.amazonaws.com',
    'database': 'ai_db',
    'user': 'admin',
    'password': 'password'
}

feature_store_integration.ingest_data_from_database(
    "user_engagement_features",
    db_config,
    "SELECT user_id, age, engagement_score, session_count, label, NOW() as event_time FROM training_data WHERE split = 'train'"
)
```

## Section 2: Google Vertex AI Integration

### Step 1: Vertex AI with BigQuery integration
```python
from google.cloud import aiplatform, bigquery
from google.auth import default
import pandas as pd

class VertexAIDatabaseIntegration:
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        self.bq_client = bigquery.Client(project=project_id)
        aiplatform.init(project=project_id, location=location)
    
    def create_bigquery_dataset_for_ai(self, dataset_id: str, 
                                     description: str = "AI training dataset"):
        """Create BigQuery dataset for AI workloads"""
        dataset_ref = self.bq_client.dataset(dataset_id)
        dataset = bigquery.Dataset(dataset_ref)
        dataset.description = description
        
        dataset = self.bq_client.create_dataset(dataset)
        return dataset
    
    def load_data_from_cloud_sql_to_bigquery(self, cloud_sql_instance: str,
                                           database_name: str,
                                           table_name: str,
                                           destination_dataset: str,
                                           destination_table: str):
        """Load data from Cloud SQL to BigQuery"""
        # Create external connection (simplified)
        # In practice, use Data Transfer Service or Cloud Function
        
        # Load directly using BigQuery SQL
        query = f"""
        CREATE OR REPLACE TABLE `{self.project_id}.{destination_dataset}.{destination_table}` AS
        SELECT * FROM EXTERNAL_QUERY(
            'connection_id',
            'SELECT * FROM {table_name}'
        )
        """
        
        job = self.bq_client.query(query)
        job.result()  # Wait for completion
        
        return f"Loaded data to {destination_dataset}.{destination_table}"
    
    def train_vertex_ai_model_from_bigquery(self, 
                                          dataset_id: str,
                                          table_id: str,
                                          target_column: str,
                                          model_display_name: str):
        """Train Vertex AI model from BigQuery data"""
        # Create dataset
        dataset = aiplatform.TabularDataset.create(
            display_name=model_display_name,
            gcs_source=f"bq://{self.project_id}.{dataset_id}.{table_id}"
        )
        
        # Create training job
        job = aiplatform.AutoMLTabularTrainingJob(
            display_name=model_display_name,
            optimization_prediction_type="classification",
            optimization_objective="maximize-au-prc"
        )
        
        # Train
        model = job.run(
            dataset=dataset,
            target_column=target_column,
            training_fraction_split=0.8,
            validation_fraction_split=0.1,
            test_fraction_split=0.1,
            budget_milli_node_hours=1000,
            disable_early_stopping=False
        )
        
        return model
    
    def deploy_vertex_ai_endpoint(self, model, endpoint_display_name: str):
        """Deploy Vertex AI endpoint"""
        endpoint = model.deploy(
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=1,
            sync=True
        )
        
        return endpoint

# Usage example
vertex_integration = VertexAIDatabaseIntegration("my-project-id", "us-central1")

# Train model
# model = vertex_integration.train_vertex_ai_model_from_bigquery(
#     dataset_id="ai_datasets",
#     table_id="training_data",
#     target_column="label",
#     model_display_name="user_engagement_classifier"
# )
```

### Step 2: Vertex AI Feature Store integration
```python
from google.cloud import aiplatform_v1

class VertexAIFeatureStoreIntegration:
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def create_feature_store(self, feature_store_id: str, 
                           online: bool = True, 
                           offline: bool = True):
        """Create Vertex AI Feature Store"""
        feature_store = aiplatform.FeatureStore.create(
            feature_store_id=feature_store_id,
            online=online,
            offline=offline
        )
        
        return feature_store
    
    def create_entity_type(self, feature_store_id: str, entity_type_id: str):
        """Create entity type in Feature Store"""
        feature_store = aiplatform.FeatureStore(feature_store_id)
        entity_type = feature_store.create_entity_type(
            entity_type_id=entity_type_id,
            description="User entity type"
        )
        
        return entity_type
    
    def ingest_data_from_bigquery(self, entity_type_id: str,
                                bigquery_table: str,
                                feature_ids: List[str]):
        """Ingest data from BigQuery to Feature Store"""
        entity_type = aiplatform.EntityType(entity_type_id)
        
        # Create feature ingestion job
        job = entity_type.ingest_from_bq(
            source_uri=f"bq://{self.project_id}.{bigquery_table}",
            feature_ids=feature_ids,
            worker_count=5
        )
        
        return job

# Usage example
feature_store_integration = VertexAIFeatureStoreIntegration("my-project-id")

# Create feature store
feature_store = feature_store_integration.create_feature_store("user_features")

# Create entity type
entity_type = feature_store_integration.create_entity_type("user_features", "users")

# Ingest data
job = feature_store_integration.ingest_data_from_bigquery(
    "users",
    "ai_datasets.training_data",
    ["user_id", "age", "engagement_score", "session_count", "label"]
)
```

## Section 3: Azure Machine Learning Integration

### Step 1: Azure ML with SQL Database integration
```python
from azureml.core import Workspace, Experiment, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.dataset import Dataset
from azureml.core.datastore import Datastore
import pandas as pd

class AzureMLDatabaseIntegration:
    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str):
        self.workspace = Workspace(subscription_id, resource_group, workspace_name)
    
    def create_azure_sql_datastore(self, server_name: str, database_name: str,
                                 username: str, password: str):
        """Create Azure SQL datastore"""
        sql_datastore = Datastore.register_azure_sql_database(
            workspace=self.workspace,
            datastore_name="sql_datastore",
            server_name=server_name,
            database_name=database_name,
            username=username,
            password=password,
            tenant_id=None,
            client_id=None,
            client_secret=None
        )
        
        return sql_datastore
    
    def create_dataset_from_sql(self, datastore_name: str, query: str):
        """Create dataset from SQL query"""
        datastore = self.workspace.datastores[datastore_name]
        
        # Create dataset
        dataset = Dataset.Tabular.from_sql_query(
            datastore,
            query=query
        )
        
        return dataset
    
    def train_azure_ml_model(self, dataset, compute_target_name: str,
                            entry_script: str = "train.py"):
        """Train Azure ML model"""
        # Get compute target
        compute_target = self.workspace.compute_targets[compute_target_name]
        
        # Create script run config
        src = ScriptRunConfig(
            source_directory='./src',
            script=entry_script,
            compute_target=compute_target,
            environment='azureml:AzureML-Minimal'
        )
        
        # Submit experiment
        experiment = Experiment(workspace=self.workspace, name='database-training')
        run = experiment.submit(src)
        
        return run
    
    def deploy_azure_ml_endpoint(self, model_name: str, compute_target_name: str):
        """Deploy Azure ML endpoint"""
        # Get model
        model = self.workspace.models[model_name]
        
        # Create inference configuration
        from azureml.core.model import InferenceConfig
        from azureml.core.environment import Environment
        
        env = Environment.get(workspace=self.workspace, name="AzureML-Minimal")
        inference_config = InferenceConfig(entry_script="score.py", environment=env)
        
        # Create deployment configuration
        from azureml.core.webservice import AciWebservice
        aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
        
        # Deploy
        service = Model.deploy(
            workspace=self.workspace,
            name=f"{model_name}-endpoint",
            models=[model],
            inference_config=inference_config,
            deployment_config=aci_config,
            deployment_target=compute_target
        )
        
        return service

# Usage example
azure_ml_integration = AzureMLDatabaseIntegration(
    "your-subscription-id",
    "your-resource-group",
    "your-workspace-name"
)

# Create datastore
# sql_datastore = azure_ml_integration.create_azure_sql_datastore(
#     "my-sql-server.database.windows.net",
#     "ai_db",
#     "admin",
#     "password"
# )
```

### Step 2: Azure Synapse Analytics integration
```python
from azure.synapse.artifacts import PipelineClient
from azure.identity import DefaultAzureCredential

class AzureSynapseIntegration:
    def __init__(self, workspace_name: str, resource_group: str, subscription_id: str):
        self.credential = DefaultAzureCredential()
        self.workspace_name = workspace_name
        self.resource_group = resource_group
        self.subscription_id = subscription_id
    
    def create_synapse_pipeline_for_ml(self, pipeline_name: str,
                                     source_database: str,
                                     target_storage: str):
        """Create Synapse pipeline for ML data processing"""
        # In practice, use Azure SDK or REST API
        # This is a simplified representation
        
        pipeline_definition = {
            "name": pipeline_name,
            "properties": {
                "activities": [
                    {
                        "name": "ExtractFromSQL",
                        "type": "CopyData",
                        "inputs": [{"dataset": {"referenceName": source_database}}]
                    },
                    {
                        "name": "TransformForML",
                        "type": "DatabricksNotebook",
                        "linkedServiceName": "DatabricksLinkedService"
                    },
                    {
                        "name": "LoadToStorage",
                        "type": "CopyData",
                        "outputs": [{"dataset": {"referenceName": target_storage}}]
                    }
                ]
            }
        }
        
        return pipeline_definition
    
    def integrate_with_azure_ml(self, synapse_pipeline_name: str,
                               azure_ml_workspace: str):
        """Integrate Synapse pipeline with Azure ML"""
        # Trigger Azure ML pipeline from Synapse
        # Use Web Activity to call Azure ML REST API
        
        web_activity = {
            "name": "TriggerAML",
            "type": "WebActivity",
            "typeProperties": {
                "method": "POST",
                "url": f"https://management.azure.com/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.MachineLearningServices/workspaces/{azure_ml_workspace}/experiments/database-training/runs?api-version=2022-10-01",
                "body": {
                    "displayName": "Synapse-triggered run",
                    "properties": {
                        "source": "synapse",
                        "pipeline": synapse_pipeline_name
                    }
                }
            }
        }
        
        return web_activity

# Usage example
synapse_integration = AzureSynapseIntegration(
    "my-synapse-workspace",
    "my-resource-group",
    "your-subscription-id"
)

# Create pipeline
pipeline = synapse_integration.create_synapse_pipeline_for_ml(
    "ml-data-pipeline",
    "sql-database",
    "blob-storage"
)
```

## Section 4: Cross-Cloud Patterns

### Step 1: Unified database integration pattern
```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any

class CloudDatabaseIntegration(ABC):
    @abstractmethod
    def connect_to_database(self, config: Dict) -> Any:
        """Connect to database"""
        pass
    
    @abstractmethod
    def extract_data(self, query: str) -> List[Dict]:
        """Extract data from database"""
        pass
    
    @abstractmethod
    def train_model(self, data: List[Dict], model_config: Dict) -> Any:
        """Train model"""
        pass
    
    @abstractmethod
    def deploy_endpoint(self, model: Any, config: Dict) -> Any:
        """Deploy endpoint"""
        pass

class AWSCloudIntegration(CloudDatabaseIntegration):
    def __init__(self):
        self.sagemaker = boto3.client('sagemaker')
    
    def connect_to_database(self, config: Dict):
        return boto3.client('rds', **config)
    
    def extract_data(self, query: str):
        # Implementation for AWS
        return [{"id": 1, "feature": 0.5}]
    
    def train_model(self, data: List[Dict], model_config: Dict):
        # Implementation for SageMaker
        return "sagemaker-model"
    
    def deploy_endpoint(self, model: Any, config: Dict):
        # Implementation for SageMaker endpoint
        return "sagemaker-endpoint"

class GCPIntegration(CloudDatabaseIntegration):
    def __init__(self):
        self.vertex_ai = aiplatform
    
    def connect_to_database(self, config: Dict):
        return bigquery.Client(**config)
    
    def extract_data(self, query: str):
        # Implementation for BigQuery
        return [{"id": 1, "feature": 0.5}]
    
    def train_model(self, data: List[Dict], model_config: Dict):
        # Implementation for Vertex AI
        return "vertex-model"
    
    def deploy_endpoint(self, model: Any, config: Dict):
        # Implementation for Vertex AI endpoint
        return "vertex-endpoint"

class AzureIntegration(CloudDatabaseIntegration):
    def __init__(self):
        self.workspace = None
    
    def connect_to_database(self, config: Dict):
        return None  # Simplified
    
    def extract_data(self, query: str):
        return [{"id": 1, "feature": 0.5}]
    
    def train_model(self, data: List[Dict], model_config: Dict):
        return "azure-model"
    
    def deploy_endpoint(self, model: Any, config: Dict):
        return "azure-endpoint"

# Unified interface
class UnifiedCloudIntegration:
    def __init__(self, cloud_provider: str):
        if cloud_provider == "aws":
            self.integration = AWSCloudIntegration()
        elif cloud_provider == "gcp":
            self.integration = GCPIntegration()
        elif cloud_provider == "azure":
            self.integration = AzureIntegration()
        else:
            raise ValueError(f"Unsupported cloud provider: {cloud_provider}")
    
    def process_database_workflow(self, db_config: Dict, query: str,
                                model_config: Dict, deploy_config: Dict):
        """Process complete database-to-AI workflow"""
        # Connect to database
        db_connection = self.integration.connect_to_database(db_config)
        
        # Extract data
        data = self.integration.extract_data(query)
        
        # Train model
        model = self.integration.train_model(data, model_config)
        
        # Deploy endpoint
        endpoint = self.integration.deploy_endpoint(model, deploy_config)
        
        return {
            'data_extracted': len(data),
            'model_trained': model,
            'endpoint_deployed': endpoint
        }

# Usage example
unified_integration = UnifiedCloudIntegration("aws")

result = unified_integration.process_database_workflow(
    db_config={'region_name': 'us-west-2'},
    query="SELECT * FROM training_data WHERE split = 'train'",
    model_config={'algorithm': 'random_forest'},
    deploy_config={'instance_type': 'ml.m5.large'}
)
```

## Section 5: Serverless Architecture

### Step 1: AWS Lambda with RDS integration
```python
import json
import boto3
import psycopg2
from typing import Dict

def lambda_handler(event: Dict, context) -> Dict:
    """Lambda function for database-integrated AI inference"""
    try:
        # Parse input
        user_id = event.get('user_id')
        event_data = event.get('event_data', {})
        
        # Connect to RDS
        conn = psycopg2.connect(
            host=os.environ['RDS_HOST'],
            database=os.environ['RDS_DB'],
            user=os.environ['RDS_USER'],
            password=os.environ['RDS_PASSWORD'],
            port=5432
        )
        
        # Get historical features
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                avg_clicks, avg_time_spent, session_count, engagement_score
            FROM feature_user_engagement 
            WHERE user_id = %s
        """, (user_id,))
        
        historical_features = cursor.fetchone()
        cursor.close()
        conn.close()
        
        # Compute real-time features
        real_time_features = {
            'current_click': event_data.get('clicks', 0),
            'current_time_spent': event_data.get('time_spent', 0),
            'session_duration': event_data.get('session_duration', 0)
        }
        
        # Combine features
        if historical_features:
            combined_features = {
                'historical_avg_clicks': historical_features[0],
                'historical_avg_time_spent': historical_features[1],
                'historical_session_count': historical_features[2],
                'historical_engagement_score': historical_features[3],
                **real_time_features
            }
        else:
            combined_features = {
                'historical_avg_clicks': 0.0,
                'historical_avg_time_spent': 0.0,
                'historical_session_count': 0.0,
                'historical_engagement_score': 0.0,
                **real_time_features
            }
        
        # Load model from S3
        s3 = boto3.client('s3')
        model_obj = s3.get_object(Bucket=os.environ['MODEL_BUCKET'], Key='model.pkl')
        model_bytes = model_obj['Body'].read()
        
        # Make prediction (simplified)
        prediction = 0.75  # In practice, load and use actual model
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'user_id': user_id,
                'prediction': prediction,
                'features_used': list(combined_features.keys()),
                'timestamp': context.invoked_function_arn
            })
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

# Deployment configuration
lambda_config = {
    'FunctionName': 'database-inference-lambda',
    'Runtime': 'python3.9',
    'Role': 'arn:aws:iam::123456789012:role/lambda-execution-role',
    'Handler': 'lambda_function.lambda_handler',
    'Environment': {
        'Variables': {
            'RDS_HOST': 'my-rds-instance.cluster-xyz.us-west-2.rds.amazonaws.com',
            'RDS_DB': 'ai_db',
            'RDS_USER': 'admin',
            'RDS_PASSWORD': 'password',
            'MODEL_BUCKET': 'my-model-bucket'
        }
    },
    'Timeout': 30
}
```

### Step 2: Google Cloud Functions with Cloud SQL
```python
import functions_framework
import google.cloud.sql.connector
import pymysql
import json

@functions_framework.http
def database_inference(request):
    """Cloud Function for database-integrated inference"""
    try:
        # Parse request
        request_json = request.get_json(silent=True)
        user_id = request_json.get('user_id')
        event_data = request_json.get('event_data', {})
        
        # Connect to Cloud SQL
        connector = google.cloud.sql.connector.Connector()
        conn = connector.connect(
            "project:region:instance",
            "pymysql",
            user="admin",
            password="password",
            db="ai_db"
        )
        
        # Get historical features
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                avg_clicks, avg_time_spent, session_count, engagement_score
            FROM feature_user_engagement 
            WHERE user_id = %s
        """, (user_id,))
        
        historical_features = cursor.fetchone()
        cursor.close()
        conn.close()
        
        # Compute real-time features
        real_time_features = {
            'current_click': event_data.get('clicks', 0),
            'current_time_spent': event_data.get('time_spent', 0),
            'session_duration': event_data.get('session_duration', 0)
        }
        
        # Combine features and make prediction
        if historical_features:
            combined_features = {
                'historical_avg_clicks': historical_features[0],
                'historical_avg_time_spent': historical_features[1],
                'historical_session_count': historical_features[2],
                'historical_engagement_score': historical_features[3],
                **real_time_features
            }
        else:
            combined_features = {
                'historical_avg_clicks': 0.0,
                'historical_avg_time_spent': 0.0,
                'historical_session_count': 0.0,
                'historical_engagement_score': 0.0,
                **real_time_features
            }
        
        # Load model from Cloud Storage
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket('my-model-bucket')
        blob = bucket.blob('model.pkl')
        model_bytes = blob.download_as_text()
        
        # Make prediction (simplified)
        prediction = 0.75
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'user_id': user_id,
                'prediction': prediction,
                'features_used': list(combined_features.keys())
            })
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

## Section 6: Performance Benchmarking

### Step 1: Cloud AI service benchmarking
```python
import time
import pandas as pd
from typing import List, Dict, Callable

class CloudAIBenchmark:
    def __init__(self):
        self.results = []
    
    def benchmark_data_ingestion(self, methods: List[Callable], 
                               data_sizes: List[int] = [1000, 10000, 100000]):
        """Benchmark data ingestion performance across clouds"""
        for method in methods:
            for size in data_sizes:
                start_time = time.time()
                
                try:
                    method(size)
                    duration = time.time() - start_time
                    
                    self.results.append({
                        'benchmark': 'data_ingestion',
                        'method': method.__name__,
                        'cloud': method.__doc__.split()[0] if method.__doc__ else 'unknown',
                        'data_size': size,
                        'duration_seconds': duration,
                        'throughput_rows_per_second': size / duration if duration > 0 else 0
                    })
                except Exception as e:
                    self.results.append({
                        'benchmark': 'data_ingestion',
                        'method': method.__name__,
                        'cloud': method.__doc__.split()[0] if method.__doc__ else 'unknown',
                        'data_size': size,
                        'duration_seconds': float('inf'),
                        'error': str(e)
                    })
    
    def benchmark_model_training(self, methods: List[Callable], 
                               dataset_sizes: List[int] = [10000, 100000, 1000000]):
        """Benchmark model training performance"""
        for method in methods:
            for size in dataset_sizes:
                start_time = time.time()
                
                try:
                    method(size)
                    duration = time.time() - start_time
                    
                    self.results.append({
                        'benchmark': 'model_training',
                        'method': method.__name__,
                        'cloud': method.__doc__.split()[0] if method.__doc__ else 'unknown',
                        'dataset_size': size,
                        'duration_seconds': duration,
                        'throughput_samples_per_second': size / duration if duration > 0 else 0
                    })
                except Exception as e:
                    self.results.append({
                        'benchmark': 'model_training',
                        'method': method.__name__,
                        'cloud': method.__doc__.split()[0] if method.__doc__ else 'unknown',
                        'dataset_size': size,
                        'duration_seconds': float('inf'),
                        'error': str(e)
                    })
    
    def generate_cloud_benchmark_report(self):
        """Generate comprehensive cloud AI benchmark report"""
        df = pd.DataFrame(self.results)
        
        # Calculate statistics
        summary = df.groupby(['benchmark', 'cloud', 'method']).agg({
            'duration_seconds': ['mean', 'std', 'min', 'max'],
            'throughput_rows_per_second': ['mean', 'std'],
            'throughput_samples_per_second': ['mean', 'std']
        }).round(2)
        
        # Generate recommendations
        recommendations = []
        
        # Best data ingestion
        if 'data_ingestion' in df['benchmark'].values:
            best_ingestion = df[df['benchmark'] == 'data_ingestion'].loc[
                df[df['benchmark'] == 'data_ingestion']['duration_seconds'].idxmin()
            ]
            recommendations.append(
                f"Best data ingestion: {best_ingestion['method']} ({best_ingestion['cloud']}) "
                f"({best_ingestion['duration_seconds']:.2f}s for {best_ingestion['data_size']} rows)"
            )
        
        # Best model training
        if 'model_training' in df['benchmark'].values:
            best_training = df[df['benchmark'] == 'model_training'].loc[
                df[df['benchmark'] == 'model_training']['duration_seconds'].idxmin()
            ]
            recommendations.append(
                f"Best model training: {best_training['method']} ({best_training['cloud']}) "
                f"({best_training['duration_seconds']:.2f}s for {best_training['dataset_size']} samples)"
            )
        
        return {
            'summary': summary,
            'detailed_results': df,
            'recommendations': recommendations,
            'cloud_selection_tips': [
                "Use AWS for complex SageMaker workflows with RDS integration",
                "Use GCP for BigQuery-native ML with Vertex AI",
                "Use Azure for SQL Database integration with Azure ML",
                "Consider cost-performance tradeoffs for your specific workload",
                "Test with realistic data volumes before production deployment"
            ]
        }

# Usage example
benchmark = CloudAIBenchmark()

# Define test methods (simplified)
def test_aws_ingestion(size: int):
    """aws data ingestion"""
    time.sleep(0.1 * (size / 10000))

def test_gcp_ingestion(size: int):
    """gcp data ingestion"""
    time.sleep(0.08 * (size / 10000))

def test_azure_ingestion(size: int):
    """azure data ingestion"""
    time.sleep(0.09 * (size / 10000))

# Run benchmarks
benchmark.benchmark_data_ingestion(
    [test_aws_ingestion, test_gcp_ingestion, test_azure_ingestion],
    [1000, 10000, 100000]
)

report = benchmark.generate_cloud_benchmark_report()
print("Cloud AI Benchmark Report:")
print(report['summary'])
print("\nRecommendations:")
for rec in report['recommendations']:
    print(f"- {rec}")
```

## Hands-on Exercises

### Exercise 1: AWS SageMaker integration
1. Set up SageMaker with RDS integration
2. Implement Feature Store integration
3. Train and deploy model
4. Monitor production performance

### Exercise 2: Google Vertex AI integration
1. Set up Vertex AI with BigQuery
2. Implement Feature Store integration
3. Train AutoML model
4. Deploy endpoint and test

### Exercise 3: Azure ML integration
1. Set up Azure ML with SQL Database
2. Create Synapse pipeline for ML
3. Train and deploy model
4. Monitor with Azure Monitor

### Exercise 4: Cross-cloud comparison
1. Implement unified integration pattern
2. Benchmark performance across clouds
3. Compare costs and features
4. Choose optimal cloud for your use case

## Best Practices Summary

1. **Cloud Selection**: Choose based on existing infrastructure and specific requirements
2. **Database Integration**: Use native cloud database services for best performance
3. **Serverless Architecture**: Leverage Lambda/Cloud Functions for event-driven AI
4. **Feature Stores**: Use cloud-native feature stores for production ML
5. **Cost Optimization**: Monitor and optimize cloud AI service costs
6. **Security**: Implement proper IAM and network security
7. **Monitoring**: Use cloud-native monitoring tools
8. **Testing**: Test with realistic data volumes and query patterns

This tutorial provides practical, hands-on experience with database integration for cloud AI services. Complete all exercises to master these critical skills for building scalable AI systems in the cloud.