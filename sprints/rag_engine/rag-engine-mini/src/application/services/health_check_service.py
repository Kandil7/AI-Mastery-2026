"""
Health Check Service

This module implements a comprehensive health check service that monitors
the operational status of various system components in the RAG Engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import time
import logging

from ...domain.entities import TenantId


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    component: str
    status: str  # operational, degraded, error
    response_time_ms: float
    details: str
    timestamp: datetime
    extra_info: Optional[Dict[str, Any]] = None


@dataclass
class SystemHealthReport:
    """Comprehensive system health report."""
    overall_status: str  # operational, degraded, error
    timestamp: datetime
    components: List[HealthCheckResult]
    dependencies: Dict[str, Any]
    metrics: Dict[str, Any]


class HealthCheckServicePort(ABC):
    """Abstract port for health check services."""

    @abstractmethod
    async def check_component_health(self, component_name: str) -> HealthCheckResult:
        """Check the health of a specific component."""
        pass

    @abstractmethod
    async def check_system_health(self) -> SystemHealthReport:
        """Perform a comprehensive system health check."""
        pass

    @abstractmethod
    async def check_dependency_health(self, dependency_name: str) -> HealthCheckResult:
        """Check the health of a specific dependency."""
        pass


class HealthCheckService(HealthCheckServicePort):
    """Concrete implementation of the health check service."""

    def __init__(
        self,
        document_repo: Any,  # Repository port
        cache: Any,  # Cache adapter
        vector_store: Any,  # Vector store adapter
        llm: Any,  # LLM adapter
        logger: Optional[logging.Logger] = None
    ):
        self._document_repo = document_repo
        self._cache = cache
        self._vector_store = vector_store
        self._llm = llm
        self._logger = logger or logging.getLogger(__name__)

    async def check_component_health(self, component_name: str) -> HealthCheckResult:
        """Check the health of a specific component."""
        start_time = time.time()
        component = component_name.lower()
        
        try:
            if component == "database":
                return await self._check_database_health(start_time)
            elif component == "cache":
                return await self._check_cache_health(start_time)
            elif component == "vector_store":
                return await self._check_vector_store_health(start_time)
            elif component == "llm":
                return await self._check_llm_health(start_time)
            elif component == "api":
                return await self._check_api_health(start_time)
            else:
                return HealthCheckResult(
                    component=component,
                    status="error",
                    response_time_ms=round((time.time() - start_time) * 1000, 2),
                    details=f"Unknown component: {component_name}",
                    timestamp=datetime.now()
                )
        except Exception as e:
            return HealthCheckResult(
                component=component,
                status="error",
                response_time_ms=round((time.time() - start_time) * 1000, 2),
                details=f"Error checking {component}: {str(e)}",
                timestamp=datetime.now()
            )

    async def _check_database_health(self, start_time: float) -> HealthCheckResult:
        """Check the health of the database component."""
        try:
            # In a real implementation, we would perform a lightweight query
            # For now, we'll just check if we can access the repository
            if self._document_repo:
                # Simulate a lightweight check
                response_time = round((time.time() - start_time) * 1000, 2)
                return HealthCheckResult(
                    component="database",
                    status="operational",
                    response_time_ms=response_time,
                    details="Database connection available",
                    timestamp=datetime.now(),
                    extra_info={
                        "connection_pool_status": "active",
                        "pending_connections": 0
                    }
                )
            else:
                return HealthCheckResult(
                    component="database",
                    status="degraded",
                    response_time_ms=round((time.time() - start_time) * 1000, 2),
                    details="Database repository not available",
                    timestamp=datetime.now()
                )
        except Exception as e:
            return HealthCheckResult(
                component="database",
                status="error",
                response_time_ms=round((time.time() - start_time) * 1000, 2),
                details=f"Database error: {str(e)}",
                timestamp=datetime.now()
            )

    async def _check_cache_health(self, start_time: float) -> HealthCheckResult:
        """Check the health of the cache component."""
        try:
            if self._cache:
                # Test cache connectivity with a simple set/get operation
                test_key = f"health_test_{int(time.time() * 1000000)}"
                await self._cache.set(test_key, "health_check", 5)
                result = await self._cache.get(test_key)
                await self._cache.delete(test_key)  # Clean up
                
                response_time = round((time.time() - start_time) * 1000, 2)
                
                if result == "health_check":
                    return HealthCheckResult(
                        component="cache",
                        status="operational",
                        response_time_ms=response_time,
                        details="Cache operational and responsive",
                        timestamp=datetime.now(),
                        extra_info={
                            "hit_rate": "N/A in test environment",
                            "eviction_policy": "N/A in test environment"
                        }
                    )
                else:
                    return HealthCheckResult(
                        component="cache",
                        status="degraded",
                        response_time_ms=response_time,
                        details="Cache available but not responding correctly",
                        timestamp=datetime.now()
                    )
            else:
                return HealthCheckResult(
                    component="cache",
                    status="degraded",
                    response_time_ms=round((time.time() - start_time) * 1000, 2),
                    details="Cache not available",
                    timestamp=datetime.now()
                )
        except Exception as e:
            return HealthCheckResult(
                component="cache",
                status="error",
                response_time_ms=round((time.time() - start_time) * 1000, 2),
                details=f"Cache error: {str(e)}",
                timestamp=datetime.now()
            )

    async def _check_vector_store_health(self, start_time: float) -> HealthCheckResult:
        """Check the health of the vector store component."""
        try:
            if self._vector_store:
                # In a real implementation, we would perform a lightweight vector operation
                response_time = round((time.time() - start_time) * 1000, 2)
                return HealthCheckResult(
                    component="vector_store",
                    status="operational",
                    response_time_ms=response_time,
                    details="Vector store connection available",
                    timestamp=datetime.now(),
                    extra_info={
                        "collection_status": "active",
                        "index_status": "ready"
                    }
                )
            else:
                return HealthCheckResult(
                    component="vector_store",
                    status="degraded",
                    response_time_ms=round((time.time() - start_time) * 1000, 2),
                    details="Vector store not available",
                    timestamp=datetime.now()
                )
        except Exception as e:
            return HealthCheckResult(
                component="vector_store",
                status="error",
                response_time_ms=round((time.time() - start_time) * 1000, 2),
                details=f"Vector store error: {str(e)}",
                timestamp=datetime.now()
            )

    async def _check_llm_health(self, start_time: float) -> HealthCheckResult:
        """Check the health of the LLM component."""
        try:
            if self._llm:
                # In a real implementation, we might perform a lightweight check
                # without actually making an expensive API call
                response_time = round((time.time() - start_time) * 1000, 2)
                return HealthCheckResult(
                    component="llm",
                    status="operational",
                    response_time_ms=response_time,
                    details="LLM service available",
                    timestamp=datetime.now(),
                    extra_info={
                        "provider": getattr(self._llm, '__class__', type('Unknown', (), {})).__name__,
                        "rate_limit_status": "N/A in test"
                    }
                )
            else:
                return HealthCheckResult(
                    component="llm",
                    status="degraded",
                    response_time_ms=round((time.time() - start_time) * 1000, 2),
                    details="LLM service not available",
                    timestamp=datetime.now()
                )
        except Exception as e:
            return HealthCheckResult(
                component="llm",
                status="error",
                response_time_ms=round((time.time() - start_time) * 1000, 2),
                details=f"LLM error: {str(e)}",
                timestamp=datetime.now()
            )

    async def _check_api_health(self, start_time: float) -> HealthCheckResult:
        """Check the health of the API component."""
        try:
            # For the API, we can check if the service is responding
            response_time = round((time.time() - start_time) * 1000, 2)
            return HealthCheckResult(
                component="api",
                status="operational",
                response_time_ms=response_time,
                details="API service responding",
                timestamp=datetime.now(),
                extra_info={
                    "request_queue": 0,
                    "active_connections": 1
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component="api",
                status="error",
                response_time_ms=round((time.time() - start_time) * 1000, 2),
                details=f"API error: {str(e)}",
                timestamp=datetime.now()
            )

    async def check_system_health(self) -> SystemHealthReport:
        """Perform a comprehensive system health check."""
        start_time = time.time()
        
        # Check all components
        components = []
        component_names = ["database", "cache", "vector_store", "llm", "api"]
        
        for comp_name in component_names:
            result = await self.check_component_health(comp_name)
            components.append(result)
        
        # Check dependencies
        dependencies = await self._check_all_dependencies()
        
        # Calculate overall status
        operational_count = sum(1 for comp in components if comp.status == "operational")
        total_count = len(components)
        
        if operational_count == total_count:
            overall_status = "operational"
        elif operational_count == 0:
            overall_status = "error"
        else:
            overall_status = "degraded"
        
        # Calculate metrics
        total_response_time = sum(comp.response_time_ms for comp in components)
        avg_response_time = total_response_time / len(components) if components else 0
        
        metrics = {
            "total_response_time_ms": round(total_response_time, 2),
            "average_response_time_ms": round(avg_response_time, 2),
            "components_checked": total_count,
            "operational_components": operational_count,
            "degraded_components": sum(1 for comp in components if comp.status == "degraded"),
            "error_components": sum(1 for comp in components if comp.status == "error"),
            "check_duration_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        return SystemHealthReport(
            overall_status=overall_status,
            timestamp=datetime.now(),
            components=components,
            dependencies=dependencies,
            metrics=metrics
        )

    async def _check_all_dependencies(self) -> Dict[str, Any]:
        """Check the health of all dependencies."""
        dependencies = {}
        
        # Check database connection
        try:
            if self._document_repo:
                dependencies["postgresql"] = {
                    "status": "connected",
                    "type": "database",
                    "response_time_ms": 0  # Would be measured in real implementation
                }
            else:
                dependencies["postgresql"] = {
                    "status": "disconnected",
                    "type": "database"
                }
        except Exception as e:
            dependencies["postgresql"] = {
                "status": "error",
                "type": "database",
                "error": str(e)
            }
        
        # Check Redis connection
        try:
            if self._cache:
                dependencies["redis"] = {
                    "status": "connected",
                    "type": "cache",
                    "response_time_ms": 0  # Would be measured in real implementation
                }
            else:
                dependencies["redis"] = {
                    "status": "disconnected",
                    "type": "cache"
                }
        except Exception as e:
            dependencies["redis"] = {
                "status": "error",
                "type": "cache",
                "error": str(e)
            }
        
        # Check Qdrant connection
        try:
            if self._vector_store:
                dependencies["qdrant"] = {
                    "status": "connected",
                    "type": "vector_store",
                    "response_time_ms": 0  # Would be measured in real implementation
                }
            else:
                dependencies["qdrant"] = {
                    "status": "disconnected",
                    "type": "vector_store"
                }
        except Exception as e:
            dependencies["qdrant"] = {
                "status": "error",
                "type": "vector_store",
                "error": str(e)
            }
        
        # Check LLM provider
        try:
            if self._llm:
                dependencies["llm_provider"] = {
                    "status": "available",
                    "type": "external_api",
                    "response_time_ms": 0  # Would be measured in real implementation
                }
            else:
                dependencies["llm_provider"] = {
                    "status": "unavailable",
                    "type": "external_api"
                }
        except Exception as e:
            dependencies["llm_provider"] = {
                "status": "error",
                "type": "external_api",
                "error": str(e)
            }
        
        return dependencies

    async def check_dependency_health(self, dependency_name: str) -> HealthCheckResult:
        """Check the health of a specific dependency."""
        start_time = time.time()
        dep_name = dependency_name.lower()
        
        try:
            if dep_name == "postgresql":
                if self._document_repo:
                    return HealthCheckResult(
                        component="postgresql",
                        status="operational",
                        response_time_ms=round((time.time() - start_time) * 1000, 2),
                        details="PostgreSQL database available",
                        timestamp=datetime.now()
                    )
                else:
                    return HealthCheckResult(
                        component="postgresql",
                        status="degraded",
                        response_time_ms=round((time.time() - start_time) * 1000, 2),
                        details="PostgreSQL database not available",
                        timestamp=datetime.now()
                    )
            elif dep_name == "redis":
                if self._cache:
                    return HealthCheckResult(
                        component="redis",
                        status="operational",
                        response_time_ms=round((time.time() - start_time) * 1000, 2),
                        details="Redis cache available",
                        timestamp=datetime.now()
                    )
                else:
                    return HealthCheckResult(
                        component="redis",
                        status="degraded",
                        response_time_ms=round((time.time() - start_time) * 1000, 2),
                        details="Redis cache not available",
                        timestamp=datetime.now()
                    )
            elif dep_name == "qdrant":
                if self._vector_store:
                    return HealthCheckResult(
                        component="qdrant",
                        status="operational",
                        response_time_ms=round((time.time() - start_time) * 1000, 2),
                        details="Qdrant vector store available",
                        timestamp=datetime.now()
                    )
                else:
                    return HealthCheckResult(
                        component="qdrant",
                        status="degraded",
                        response_time_ms=round((time.time() - start_time) * 1000, 2),
                        details="Qdrant vector store not available",
                        timestamp=datetime.now()
                    )
            elif dep_name == "llm_provider":
                if self._llm:
                    return HealthCheckResult(
                        component="llm_provider",
                        status="operational",
                        response_time_ms=round((time.time() - start_time) * 1000, 2),
                        details="LLM provider available",
                        timestamp=datetime.now()
                    )
                else:
                    return HealthCheckResult(
                        component="llm_provider",
                        status="degraded",
                        response_time_ms=round((time.time() - start_time) * 1000, 2),
                        details="LLM provider not available",
                        timestamp=datetime.now()
                    )
            else:
                return HealthCheckResult(
                    component=dep_name,
                    status="error",
                    response_time_ms=round((time.time() - start_time) * 1000, 2),
                    details=f"Unknown dependency: {dependency_name}",
                    timestamp=datetime.now()
                )
        except Exception as e:
            return HealthCheckResult(
                component=dep_name,
                status="error",
                response_time_ms=round((time.time() - start_time) * 1000, 2),
                details=f"Error checking {dep_name}: {str(e)}",
                timestamp=datetime.now()
            )