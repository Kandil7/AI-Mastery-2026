"""
API Dependencies
==================
Request-scoped dependencies for FastAPI routes.

تبعيات الطلب لمسارات FastAPI
"""

from fastapi import Header, HTTPException

from src.core.config import settings


async def get_tenant_id(
    api_key: str = Header(..., alias="X-API-KEY"),
) -> str:
    """
    Extract tenant_id from API key header.
    
    In production, this would:
    1. Look up API key in database
    2. Verify it's valid and active
    3. Return associated user_id as tenant_id
    
    For now, we use the API key directly as tenant_id
    (suitable for development/demo).
    
    استخراج معرف المستأجر من رأس مفتاح API
    """
    if not api_key or len(api_key) < 8:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )
    
    # In production: look up in database
    # user_lookup_repo.get_user_id_by_api_key(api_key)
    
    # For now, use API key as tenant_id directly
    # This is fine for development/demos
    return api_key


def get_api_key_header_name() -> str:
    """Get the configured API key header name."""
    return settings.api_key_header
