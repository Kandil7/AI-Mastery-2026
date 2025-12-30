"""
Deployment Module

This module implements deployment strategies for ML models in production,
including model versioning, A/B testing, canary deployments, and model serving.
"""

import os
import json
import pickle
import yaml
import hashlib
import time
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import subprocess
import docker
from docker.types import Mount
import kubernetes
from kubernetes import client, config
import requests
import threading
from pathlib import Path
import logging
from datetime import datetime


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Enumeration of model statuses."""
    PENDING = "pending"
    TRAINING = "training"
    EVALUATING = "evaluating"
    DEPLOYING = "deploying"
    SERVING = "serving"
    FAILED = "failed"
    RETIRED = "retired"


class DeploymentStrategy(Enum):
    """Enumeration of deployment strategies."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    A_B_TESTING = "a_b_testing"


@dataclass
class ModelVersion:
    """Represents a version of a model."""
    version_id: str
    model_path: str
    metadata: Dict[str, Any]
    created_at: float
    accuracy: Optional[float] = None
    performance_metrics: Optional[Dict[str, float]] = None


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    model_name: str
    model_version: str
    strategy: DeploymentStrategy
    replicas: int = 1
    resources: Dict[str, str] = None  # CPU, memory limits
    environment: Dict[str, str] = None
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"
    port: int = 8000
    
    def __post_init__(self):
        if self.resources is None:
            self.resources = {"cpu": "500m", "memory": "1Gi"}
        if self.environment is None:
            self.environment = {}


class ModelRegistry:
    """Manages model versions and metadata."""
    
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.models_file = self.registry_path / "models.json"
        
        # Load existing models
        self.models: Dict[str, List[ModelVersion]] = self._load_models()
    
    def _load_models(self) -> Dict[str, List[ModelVersion]]:
        """Load models from registry file."""
        if self.models_file.exists():
            with open(self.models_file, 'r') as f:
                data = json.load(f)
                
                # Convert dict back to ModelVersion objects
                models = {}
                for model_name, versions_data in data.items():
                    versions = []
                    for v_data in versions_data:
                        version = ModelVersion(
                            version_id=v_data['version_id'],
                            model_path=v_data['model_path'],
                            metadata=v_data['metadata'],
                            created_at=v_data['created_at'],
                            accuracy=v_data.get('accuracy'),
                            performance_metrics=v_data.get('performance_metrics')
                        )
                        versions.append(version)
                    models[model_name] = versions
                return models
        return {}
    
    def _save_models(self):
        """Save models to registry file."""
        # Convert ModelVersion objects to dict
        data = {}
        for model_name, versions in self.models.items():
            versions_data = []
            for version in versions:
                v_data = {
                    'version_id': version.version_id,
                    'model_path': version.model_path,
                    'metadata': version.metadata,
                    'created_at': version.created_at,
                    'accuracy': version.accuracy,
                    'performance_metrics': version.performance_metrics
                }
                versions_data.append(v_data)
            data[model_name] = versions_data
        
        with open(self.models_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_model(self, model_name: str, model_path: str, metadata: Dict[str, Any], 
                      accuracy: Optional[float] = None, performance_metrics: Optional[Dict[str, float]] = None) -> str:
        """Register a new model version."""
        version_id = f"{model_name}-v{len(self.models.get(model_name, [])) + 1}-{int(time.time())}"
        
        version = ModelVersion(
            version_id=version_id,
            model_path=model_path,
            metadata=metadata,
            created_at=time.time(),
            accuracy=accuracy,
            performance_metrics=performance_metrics
        )
        
        if model_name not in self.models:
            self.models[model_name] = []
        
        self.models[model_name].append(version)
        self._save_models()
        
        logger.info(f"Registered model {model_name} with version {version_id}")
        return version_id
    
    def get_model_version(self, model_name: str, version_id: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        if model_name in self.models:
            for version in self.models[model_name]:
                if version.version_id == version_id:
                    return version
        return None
    
    def get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the latest version of a model."""
        if model_name in self.models and self.models[model_name]:
            return self.models[model_name][-1]
        return None
    
    def list_versions(self, model_name: str) -> List[ModelVersion]:
        """List all versions of a model."""
        return self.models.get(model_name, [])


class ModelDeployer:
    """Handles model deployment to various platforms."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.active_deployments: Dict[str, Dict[str, Any]] = {}
        self.deployment_lock = threading.Lock()
    
    def deploy_model(self, config: DeploymentConfig) -> str:
        """Deploy a model with the specified configuration."""
        with self.deployment_lock:
            deployment_id = f"{config.model_name}-{config.model_version}-{int(time.time())}"
            
            logger.info(f"Deploying model {config.model_name} version {config.model_version}")
            
            # Get model from registry
            model_version = self.registry.get_model_version(config.model_name, config.model_version)
            if not model_version:
                raise ValueError(f"Model {config.model_name} version {config.model_version} not found in registry")
            
            # Deploy based on strategy
            if config.strategy == DeploymentStrategy.BLUE_GREEN:
                deployment_info = self._blue_green_deploy(config, model_version)
            elif config.strategy == DeploymentStrategy.CANARY:
                deployment_info = self._canary_deploy(config, model_version)
            elif config.strategy == DeploymentStrategy.ROLLING:
                deployment_info = self._rolling_deploy(config, model_version)
            elif config.strategy == DeploymentStrategy.A_B_TESTING:
                deployment_info = self._ab_testing_deploy(config, model_version)
            else:
                raise ValueError(f"Unknown deployment strategy: {config.strategy}")
            
            # Store deployment info
            self.active_deployments[deployment_id] = {
                "config": config,
                "model_version": model_version,
                "deployment_info": deployment_info,
                "status": ModelStatus.DEPLOYING,
                "deployed_at": time.time()
            }
            
            logger.info(f"Model deployed with ID: {deployment_id}")
            return deployment_id
    
    def _blue_green_deploy(self, config: DeploymentConfig, model_version: ModelVersion) -> Dict[str, Any]:
        """Deploy using blue-green strategy."""
        logger.info("Starting blue-green deployment")
        
        # In a real implementation, this would:
        # 1. Deploy the new version (green) alongside the current version (blue)
        # 2. Run health checks on the new version
        # 3. Switch traffic to the new version
        # 4. Keep the old version as backup
        
        # For this implementation, we'll simulate the deployment
        deployment_info = {
            "strategy": "blue-green",
            "blue_endpoint": f"http://blue-{config.model_name}:{config.port}",
            "green_endpoint": f"http://green-{config.model_name}:{config.port}",
            "status": "active"
        }
        
        return deployment_info
    
    def _canary_deploy(self, config: DeploymentConfig, model_version: ModelVersion) -> Dict[str, Any]:
        """Deploy using canary strategy."""
        logger.info("Starting canary deployment")
        
        # In a real implementation, this would:
        # 1. Deploy the new version to a small subset of instances
        # 2. Gradually increase traffic to the new version based on metrics
        # 3. Roll back if issues are detected
        
        deployment_info = {
            "strategy": "canary",
            "primary_endpoint": f"http://primary-{config.model_name}:{config.port}",
            "canary_endpoint": f"http://canary-{config.model_name}:{config.port}",
            "canary_traffic_percentage": 10,  # Start with 10% traffic
            "status": "active"
        }
        
        return deployment_info
    
    def _rolling_deploy(self, config: DeploymentConfig, model_version: ModelVersion) -> Dict[str, Any]:
        """Deploy using rolling strategy."""
        logger.info("Starting rolling deployment")
        
        # In a real implementation, this would:
        # 1. Gradually replace instances of the old version with the new version
        # 2. Monitor health during the process
        # 3. Pause or rollback if issues are detected
        
        deployment_info = {
            "strategy": "rolling",
            "endpoint": f"http://{config.model_name}:{config.port}",
            "replicas": config.replicas,
            "status": "active"
        }
        
        return deployment_info
    
    def _ab_testing_deploy(self, config: DeploymentConfig, model_version: ModelVersion) -> Dict[str, Any]:
        """Deploy using A/B testing strategy."""
        logger.info("Starting A/B testing deployment")
        
        # In a real implementation, this would:
        # 1. Deploy both versions simultaneously
        # 2. Split traffic between versions based on defined rules
        # 3. Collect metrics on both versions
        # 4. Determine winner based on metrics
        
        deployment_info = {
            "strategy": "a_b_testing",
            "model_a_endpoint": f"http://model-a-{config.model_name}:{config.port}",
            "model_b_endpoint": f"http://model-b-{config.model_name}:{config.port}",
            "traffic_split": {"model_a": 50, "model_b": 50},  # 50-50 split
            "status": "active"
        }
        
        return deployment_info
    
    def undeploy_model(self, deployment_id: str):
        """Undeploy a model."""
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self.active_deployments[deployment_id]
        logger.info(f"Undeploying model {deployment['config'].model_name}")
        
        # In a real implementation, this would:
        # 1. Remove the deployed resources
        # 2. Clean up any associated resources
        
        del self.active_deployments[deployment_id]
        logger.info(f"Model undeployed: {deployment_id}")
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a deployment."""
        if deployment_id not in self.active_deployments:
            return None
        
        deployment = self.active_deployments[deployment_id]
        return {
            "deployment_id": deployment_id,
            "model_name": deployment["config"].model_name,
            "model_version": deployment["config"].model_version,
            "status": deployment["status"],
            "deployed_at": deployment["deployed_at"],
            "deployment_info": deployment["deployment_info"]
        }
    
    def scale_deployment(self, deployment_id: str, replicas: int):
        """Scale a deployment to the specified number of replicas."""
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self.active_deployments[deployment_id]
        logger.info(f"Scaling deployment {deployment_id} to {replicas} replicas")
        
        # In a real implementation, this would:
        # 1. Update the deployment configuration
        # 2. Scale the underlying resources
        deployment["config"].replicas = replicas


class DockerDeployer(ModelDeployer):
    """Docker-based model deployment."""
    
    def __init__(self, registry: ModelRegistry, docker_client=None):
        super().__init__(registry)
        self.docker_client = docker_client or docker.from_env()
    
    def _create_docker_deployment(self, config: DeploymentConfig, model_version: ModelVersion) -> Dict[str, Any]:
        """Create a Docker-based deployment."""
        # Create a simple Docker image for the model
        dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE {config.port}

CMD ["python", "-m", "uvicorn", "src.production.api:app", "--host", "0.0.0.0", "--port", "{config.port}"]
"""
        
        # Write Dockerfile
        dockerfile_path = f"./temp_dockerfile_{config.model_name}"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Build image
        image_tag = f"{config.model_name}:{config.model_version}"
        logger.info(f"Building Docker image: {image_tag}")
        
        # In a real implementation, we would build the actual image
        # For this example, we'll just return a placeholder
        image_id = hashlib.md5(f"{image_tag}{time.time()}".encode()).hexdigest()
        
        # Remove temporary Dockerfile
        os.remove(dockerfile_path)
        
        # Run container
        container_name = f"{config.model_name}-{config.model_version}"
        
        try:
            # Stop and remove any existing container with the same name
            try:
                existing_container = self.docker_client.containers.get(container_name)
                existing_container.stop()
                existing_container.remove()
            except docker.errors.NotFound:
                pass  # Container doesn't exist, which is fine
            
            # Run new container
            container = self.docker_client.containers.run(
                image=image_tag,
                name=container_name,
                ports={f"{config.port}/tcp": config.port},
                environment=config.environment,
                detach=True,
                remove=False  # Don't auto-remove when stopped
            )
            
            return {
                "image_id": image_id,
                "container_id": container.id,
                "container_name": container_name,
                "endpoint": f"http://localhost:{config.port}",
                "status": "running"
            }
        except Exception as e:
            logger.error(f"Docker deployment failed: {str(e)}")
            raise


class KubernetesDeployer(ModelDeployer):
    """Kubernetes-based model deployment."""
    
    def __init__(self, registry: ModelRegistry, kube_config_path: Optional[str] = None):
        super().__init__(registry)
        
        # Load Kubernetes configuration
        if kube_config_path:
            config.load_kube_config(config_file=kube_config_path)
        else:
            try:
                config.load_incluster_config()  # For in-cluster config
            except:
                config.load_kube_config()  # For local development
        
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
    
    def _create_k8s_deployment(self, config: DeploymentConfig, model_version: ModelVersion) -> Dict[str, Any]:
        """Create a Kubernetes deployment."""
        # Define the deployment manifest
        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{config.model_name}-{config.model_version}",
                "labels": {
                    "app": config.model_name,
                    "version": config.model_version
                }
            },
            "spec": {
                "replicas": config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": config.model_name,
                        "version": config.model_version
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": config.model_name,
                            "version": config.model_version
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": config.model_name,
                                "image": f"{config.model_name}:{config.model_version}",
                                "ports": [
                                    {
                                        "containerPort": config.port
                                    }
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": config.resources["cpu"],
                                        "memory": config.resources["memory"]
                                    },
                                    "limits": {
                                        "cpu": config.resources["cpu"],
                                        "memory": config.resources["memory"]
                                    }
                                },
                                "env": [{"name": k, "value": v} for k, v in config.environment.items()]
                            }
                        ]
                    }
                }
            }
        }
        
        # Create the deployment
        try:
            deployment = self.apps_v1.create_namespaced_deployment(
                body=deployment_manifest,
                namespace="default"
            )
            
            # Create a service for the deployment
            service_manifest = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": f"{config.model_name}-service",
                    "labels": {
                        "app": config.model_name
                    }
                },
                "spec": {
                    "selector": {
                        "app": config.model_name,
                        "version": config.model_version
                    },
                    "ports": [
                        {
                            "protocol": "TCP",
                            "port": config.port,
                            "targetPort": config.port
                        }
                    ],
                    "type": "LoadBalancer"
                }
            }
            
            service = self.core_v1.create_namespaced_service(
                body=service_manifest,
                namespace="default"
            )
            
            return {
                "deployment_name": deployment.metadata.name,
                "service_name": service.metadata.name,
                "namespace": "default",
                "endpoint": f"http://{service.metadata.name}:{config.port}",
                "status": "created"
            }
        except Exception as e:
            logger.error(f"Kubernetes deployment failed: {str(e)}")
            raise


class ModelDeploymentManager:
    """High-level manager for model deployments."""
    
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry = ModelRegistry(registry_path)
        self.deployer: Optional[ModelDeployer] = None
        self.deployment_history: List[Dict[str, Any]] = []
    
    def set_deployer(self, deployer: ModelDeployer):
        """Set the deployment backend."""
        self.deployer = deployer
    
    def register_and_deploy(
        self, 
        model_name: str, 
        model_path: str, 
        config: DeploymentConfig,
        metadata: Dict[str, Any] = None,
        accuracy: Optional[float] = None,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """Register a model and deploy it in one step."""
        if metadata is None:
            metadata = {}
        
        # Register the model
        version_id = self.registry.register_model(
            model_name, 
            model_path, 
            metadata, 
            accuracy, 
            performance_metrics
        )
        
        # Update config with the new version
        config.model_version = version_id
        
        # Deploy the model
        deployment_id = self.deployer.deploy_model(config)
        
        # Record in deployment history
        self.deployment_history.append({
            "deployment_id": deployment_id,
            "model_name": model_name,
            "version_id": version_id,
            "deployed_at": time.time(),
            "config": config
        })
        
        return deployment_id
    
    def update_deployment(
        self, 
        deployment_id: str, 
        new_model_path: str, 
        strategy: DeploymentStrategy,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Update an existing deployment with a new model version."""
        if metadata is None:
            metadata = {}
        
        # Get the current deployment
        current_deployment = self.deployer.get_deployment_status(deployment_id)
        if not current_deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        model_name = current_deployment["model_name"]
        
        # Register the new model version
        version_id = self.registry.register_model(model_name, new_model_path, metadata)
        
        # Undeploy the old version
        self.deployer.undeploy_model(deployment_id)
        
        # Deploy the new version with the same configuration but updated version
        new_config = DeploymentConfig(
            model_name=model_name,
            model_version=version_id,
            strategy=strategy,
            replicas=current_deployment["deployment_info"].get("replicas", 1),
            resources=current_deployment["config"].resources,
            environment=current_deployment["config"].environment,
            health_check_path=current_deployment["config"].health_check_path,
            readiness_check_path=current_deployment["config"].readiness_check_path,
            port=current_deployment["config"].port
        )
        
        new_deployment_id = self.deployer.deploy_model(new_config)
        
        return new_deployment_id
    
    def get_model_performance(self, model_name: str, version_id: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a specific model version."""
        model_version = self.registry.get_model_version(model_name, version_id)
        if not model_version:
            return None
        
        return {
            "model_name": model_name,
            "version_id": version_id,
            "accuracy": model_version.accuracy,
            "performance_metrics": model_version.performance_metrics,
            "created_at": model_version.created_at
        }
    
    def rollback_deployment(self, deployment_id: str, to_version: str) -> str:
        """Rollback a deployment to a previous version."""
        # Get the current deployment
        current_deployment = self.deployer.get_deployment_status(deployment_id)
        if not current_deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        model_name = current_deployment["model_name"]
        
        # Check if the target version exists
        target_version = self.registry.get_model_version(model_name, to_version)
        if not target_version:
            raise ValueError(f"Model {model_name} version {to_version} not found in registry")
        
        # Undeploy the current version
        self.deployer.undeploy_model(deployment_id)
        
        # Deploy the target version
        config = current_deployment["config"]
        config.model_version = to_version
        
        new_deployment_id = self.deployer.deploy_model(config)
        
        return new_deployment_id


# Global deployment manager instance
deployment_manager = ModelDeploymentManager()


def get_deployment_manager() -> ModelDeploymentManager:
    """Get the global deployment manager instance."""
    return deployment_manager


def deploy_model_simple(model_name: str, model_path: str, port: int = 8000) -> str:
    """
    Simple function to deploy a model with default settings.
    
    Args:
        model_name: Name of the model
        model_path: Path to the model file
        port: Port to deploy on
        
    Returns:
        Deployment ID
    """
    config = DeploymentConfig(
        model_name=model_name,
        model_version="temp",  # Will be replaced by register_and_deploy
        strategy=DeploymentStrategy.ROLLING,
        port=port
    )
    
    deployment_id = deployment_manager.register_and_deploy(
        model_name=model_name,
        model_path=model_path,
        config=config
    )
    
    return deployment_id


# Initialize the deployment manager when module is loaded
logger.info("Deployment module initialized")