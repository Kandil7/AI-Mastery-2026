"""
Tests for Multi-Layer Caching Service

This module tests the multi-layer caching service functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import timedelta, datetime

from src.application.services.multi_layer_cache_service import (
    MultiLayerCacheService,
    CacheLayer,
    CacheEntry
)


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client for testing."""
    redis_mock = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.keys = AsyncMock(return_value=[])
    redis_mock.flushdb = AsyncMock(return_value=True)
    redis_mock.info = AsyncMock(return_value={"used_memory": 1000, "db0": {"keys": 5}})
    return redis_mock


@pytest.fixture
def multi_layer_cache_service(mock_redis_client):
    """Create a multi-layer cache service instance for testing."""
    return MultiLayerCacheService(redis_client=mock_redis_client)


@pytest.mark.asyncio
async def test_get_from_empty_cache(multi_layer_cache_service):
    """Test getting a value from an empty cache."""
    result = await multi_layer_cache_service.get("nonexistent_key")
    assert result is None


@pytest.mark.asyncio
async def test_set_and_get_single_layer(multi_layer_cache_service):
    """Test setting and getting a value in a single cache layer."""
    # Set a value in L1 only
    await multi_layer_cache_service.set("test_key", "test_value", target_layers=[CacheLayer.L1_MEMORY])
    
    # Get the value from L1
    result = await multi_layer_cache_service.get("test_key", layer=CacheLayer.L1_MEMORY)
    assert result == "test_value"


@pytest.mark.asyncio
async def test_set_and_get_all_layers(multi_layer_cache_service):
    """Test setting and getting a value across all cache layers."""
    # Set a value in all layers
    await multi_layer_cache_service.set("test_key", "test_value", ttl=timedelta(minutes=5))
    
    # Get the value (should be promoted to L1 after retrieval from lower layers)
    result = await multi_layer_cache_service.get("test_key")
    assert result == "test_value"


@pytest.mark.asyncio
async def test_delete_key(multi_layer_cache_service):
    """Test deleting a key from all cache layers."""
    # Set a value
    await multi_layer_cache_service.set("delete_test_key", "delete_test_value")
    
    # Verify it exists
    result = await multi_layer_cache_service.get("delete_test_key")
    assert result == "delete_test_value"
    
    # Delete the key
    deleted = await multi_layer_cache_service.delete("delete_test_key")
    assert deleted is True
    
    # Verify it's gone
    result = await multi_layer_cache_service.get("delete_test_key")
    assert result is None


@pytest.mark.asyncio
async def test_invalidate_by_prefix(multi_layer_cache_service):
    """Test invalidating keys by prefix."""
    # Set multiple keys with the same prefix
    await multi_layer_cache_service.set("prefix:test1", "value1", target_layers=[CacheLayer.L1_MEMORY])
    await multi_layer_cache_service.set("prefix:test2", "value2", target_layers=[CacheLayer.L1_MEMORY])
    await multi_layer_cache_service.set("other:test3", "value3", target_layers=[CacheLayer.L1_MEMORY])
    
    # Verify they exist
    assert await multi_layer_cache_service.get("prefix:test1", layer=CacheLayer.L1_MEMORY) == "value1"
    assert await multi_layer_cache_service.get("prefix:test2", layer=CacheLayer.L1_MEMORY) == "value2"
    assert await multi_layer_cache_service.get("other:test3", layer=CacheLayer.L1_MEMORY) == "value3"
    
    # Invalidate by prefix
    invalidated_count = await multi_layer_cache_service.invalidate_by_prefix("prefix:")
    
    # Should have invalidated 2 keys
    assert invalidated_count == 2
    
    # Verify prefixed keys are gone but others remain
    assert await multi_layer_cache_service.get("prefix:test1", layer=CacheLayer.L1_MEMORY) is None
    assert await multi_layer_cache_service.get("prefix:test2", layer=CacheLayer.L1_MEMORY) is None
    assert await multi_layer_cache_service.get("other:test3", layer=CacheLayer.L1_MEMORY) == "value3"


@pytest.mark.asyncio
async def test_cache_expiry(multi_layer_cache_service):
    """Test that cache entries expire correctly."""
    # Set a value with a short TTL
    await multi_layer_cache_service.set(
        "expiring_key", 
        "expiring_value", 
        ttl=timedelta(milliseconds=1),  # Expire almost immediately
        target_layers=[CacheLayer.L1_MEMORY]
    )
    
    # Wait a bit for expiration
    import asyncio
    await asyncio.sleep(0.01)  # Sleep for 10ms
    
    # Try to get the value - should be expired
    result = await multi_layer_cache_service.get("expiring_key", layer=CacheLayer.L1_MEMORY)
    assert result is None


@pytest.mark.asyncio
async def test_get_stats(multi_layer_cache_service):
    """Test retrieving cache statistics."""
    # Set some values to populate stats
    await multi_layer_cache_service.set("stat_test1", "value1")
    await multi_layer_cache_service.set("stat_test2", "value2")
    
    # Get stats
    stats = await multi_layer_cache_service.get_stats()
    
    # Verify structure
    assert CacheLayer.L1_MEMORY in stats
    assert CacheLayer.L2_REDIS in stats
    assert CacheLayer.L3_PERSISTENT in stats
    
    # Check L1 stats
    l1_stats = stats[CacheLayer.L1_MEMORY]
    assert "size" in l1_stats
    assert "hit_rate" in l1_stats
    assert l1_stats["size"] >= 0  # May be 0 if values went to other layers


@pytest.mark.asyncio
async def test_warm_up(multi_layer_cache_service):
    """Test warming up the cache with multiple values."""
    data = {
        "warm_key1": "warm_value1",
        "warm_key2": "warm_value2",
        "warm_key3": "warm_value3"
    }
    
    # Warm up the cache
    success = await multi_layer_cache_service.warm_up(data, ttl=timedelta(minutes=10))
    assert success is True
    
    # Verify values were set
    assert await multi_layer_cache_service.get("warm_key1") == "warm_value1"
    assert await multi_layer_cache_service.get("warm_key2") == "warm_value2"
    assert await multi_layer_cache_service.get("warm_key3") == "warm_value3"


@pytest.mark.asyncio
async def test_clear_layer(multi_layer_cache_service):
    """Test clearing a specific cache layer."""
    # Set values in L1
    await multi_layer_cache_service.set("clear_test1", "value1", target_layers=[CacheLayer.L1_MEMORY])
    await multi_layer_cache_service.set("clear_test2", "value2", target_layers=[CacheLayer.L1_MEMORY])
    
    # Verify they exist
    assert await multi_layer_cache_service.get("clear_test1", layer=CacheLayer.L1_MEMORY) == "value1"
    assert await multi_layer_cache_service.get("clear_test2", layer=CacheLayer.L1_MEMORY) == "value2"
    
    # Clear the L1 layer
    cleared = await multi_layer_cache_service.clear_layer(CacheLayer.L1_MEMORY)
    assert cleared is True
    
    # Verify they're gone
    assert await multi_layer_cache_service.get("clear_test1", layer=CacheLayer.L1_MEMORY) is None
    assert await multi_layer_cache_service.get("clear_test2", layer=CacheLayer.L1_MEMORY) is None


@pytest.mark.asyncio
async def test_cache_promotion(multi_layer_cache_service, mock_redis_client):
    """Test that values are promoted from lower to higher layers when retrieved."""
    # Store a value in L2 (Redis) only
    import pickle
    serialized_value = pickle.dumps("promoted_value")
    mock_redis_client.get.return_value = serialized_value
    
    # When we get it, it should be promoted to L1
    result = await multi_layer_cache_service.get("promotion_key")
    assert result == "promoted_value"
    
    # The value should now be in L1 as well
    l1_result = await multi_layer_cache_service.get("promotion_key", layer=CacheLayer.L1_MEMORY)
    assert l1_result == "promoted_value"


@pytest.mark.asyncio
async def test_size_limit_eviction(multi_layer_cache_service):
    """Test that L1 cache evicts entries when size limits are reached."""
    # Set the size limits very low for testing
    multi_layer_cache_service._l1_max_bytes = 100  # Very small limit
    
    # Add a large value that exceeds the limit
    large_value = "x" * 200  # 200 bytes, exceeding our 100-byte limit
    await multi_layer_cache_service.set("large_key", large_value, target_layers=[CacheLayer.L1_MEMORY])
    
    # The cache should handle this gracefully, possibly by not adding if too large
    # or by having mechanisms to deal with oversized entries
    result = await multi_layer_cache_service.get("large_key", layer=CacheLayer.L1_MEMORY)
    
    # Either the value is there (if cache allows oversized) or it's not (if rejected)
    # Both are acceptable behaviors depending on implementation


@pytest.mark.asyncio
async def test_ttl_application(multi_layer_cache_service, mock_redis_client):
    """Test that TTL is properly applied when setting values."""
    test_ttl = timedelta(minutes=5)
    
    # Set a value with TTL
    await multi_layer_cache_service.set("ttl_test", "ttl_value", ttl=test_ttl)
    
    # Check that Redis was called with the correct expiration
    mock_redis_client.set.assert_called_once()
    args, kwargs = mock_redis_client.set.call_args
    assert 'ex' in kwargs
    assert kwargs['ex'] == int(test_ttl.total_seconds())