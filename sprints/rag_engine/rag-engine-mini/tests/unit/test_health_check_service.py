"""
Tests for Health Check Service

This module tests the health check service functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from src.application.services.health_check_service import (
    HealthCheckService,
    HealthCheckResult,
    SystemHealthReport
)
from src.domain.entities import TenantId


@pytest.fixture
def mock_document_repo():
    """Mock document repository for testing."""
    repo = AsyncMock()
    return repo


@pytest.fixture
def mock_cache():
    """Mock cache for testing."""
    cache = AsyncMock()
    cache.set = AsyncMock(return_value=None)
    cache.get = AsyncMock(return_value="health_check")
    cache.delete = AsyncMock(return_value=None)
    return cache


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    return AsyncMock()


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    return AsyncMock()


@pytest.fixture
def health_check_service(mock_document_repo, mock_cache, mock_vector_store, mock_llm):
    """Create a health check service instance for testing."""
    return HealthCheckService(
        document_repo=mock_document_repo,
        cache=mock_cache,
        vector_store=mock_vector_store,
        llm=mock_llm
    )


@pytest.mark.asyncio
async def test_check_database_health(health_check_service):
    """Test database health check."""
    result = await health_check_service._check_database_health(start_time=0)
    
    assert isinstance(result, HealthCheckResult)
    assert result.component == "database"
    assert result.status in ["operational", "degraded", "error"]
    assert result.response_time_ms >= 0
    assert isinstance(result.timestamp, datetime)


@pytest.mark.asyncio
async def test_check_cache_health(health_check_service, mock_cache):
    """Test cache health check."""
    result = await health_check_service._check_cache_health(start_time=0)
    
    assert isinstance(result, HealthCheckResult)
    assert result.component == "cache"
    assert result.status in ["operational", "degraded", "error"]
    assert result.response_time_ms >= 0
    assert isinstance(result.timestamp, datetime)


@pytest.mark.asyncio
async def test_check_vector_store_health(health_check_service):
    """Test vector store health check."""
    result = await health_check_service._check_vector_store_health(start_time=0)
    
    assert isinstance(result, HealthCheckResult)
    assert result.component == "vector_store"
    assert result.status in ["operational", "degraded", "error"]
    assert result.response_time_ms >= 0
    assert isinstance(result.timestamp, datetime)


@pytest.mark.asyncio
async def test_check_llm_health(health_check_service):
    """Test LLM health check."""
    result = await health_check_service._check_llm_health(start_time=0)
    
    assert isinstance(result, HealthCheckResult)
    assert result.component == "llm"
    assert result.status in ["operational", "degraded", "error"]
    assert result.response_time_ms >= 0
    assert isinstance(result.timestamp, datetime)


@pytest.mark.asyncio
async def test_check_api_health(health_check_service):
    """Test API health check."""
    result = await health_check_service._check_api_health(start_time=0)
    
    assert isinstance(result, HealthCheckResult)
    assert result.component == "api"
    assert result.status in ["operational", "degraded", "error"]
    assert result.response_time_ms >= 0
    assert isinstance(result.timestamp, datetime)


@pytest.mark.asyncio
async def test_check_component_health(health_check_service):
    """Test checking health of specific components."""
    components_to_test = ["database", "cache", "vector_store", "llm", "api"]
    
    for component in components_to_test:
        result = await health_check_service.check_component_health(component)
        assert isinstance(result, HealthCheckResult)
        assert result.component == component
        assert result.status in ["operational", "degraded", "error"]
        assert result.response_time_ms >= 0


@pytest.mark.asyncio
async def test_check_system_health(health_check_service):
    """Test comprehensive system health check."""
    report = await health_check_service.check_system_health()
    
    assert isinstance(report, SystemHealthReport)
    assert report.overall_status in ["operational", "degraded", "error"]
    assert isinstance(report.timestamp, datetime)
    assert isinstance(report.components, list)
    assert isinstance(report.dependencies, dict)
    assert isinstance(report.metrics, dict)
    
    # Check that we have results for all components
    expected_components = ["database", "cache", "vector_store", "llm", "api"]
    actual_components = [comp.component for comp in report.components]
    for expected in expected_components:
        assert expected in actual_components


@pytest.mark.asyncio
async def test_check_dependency_health(health_check_service):
    """Test checking health of specific dependencies."""
    dependencies_to_test = ["postgresql", "redis", "qdrant", "llm_provider"]
    
    for dependency in dependencies_to_test:
        result = await health_check_service.check_dependency_health(dependency)
        assert isinstance(result, HealthCheckResult)
        assert result.component == dependency
        assert result.status in ["operational", "degraded", "error"]
        assert result.response_time_ms >= 0


@pytest.mark.asyncio
async def test_check_unknown_component(health_check_service):
    """Test checking health of an unknown component."""
    result = await health_check_service.check_component_health("unknown_component")
    
    assert isinstance(result, HealthCheckResult)
    assert result.component == "unknown_component"
    assert result.status == "error"
    assert "Unknown component" in result.details


@pytest.mark.asyncio
async def test_check_system_health_error_handling(health_check_service, mock_document_repo):
    """Test system health check with error conditions."""
    # Force an error in one of the components
    original_method = health_check_service._check_database_health
    async def error_method(start_time):
        raise Exception("Database unavailable")
    
    health_check_service._check_database_health = error_method
    
    try:
        report = await health_check_service.check_system_health()
        
        # The overall status should reflect the error
        assert isinstance(report, SystemHealthReport)
        # Depending on other components, it could be degraded or error
        assert report.overall_status in ["degraded", "error"]
    finally:
        # Restore original method
        health_check_service._check_database_health = original_method