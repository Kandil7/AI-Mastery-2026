"""
API Authentication & Authorization Module
=========================================
JWT-based authentication with API key support and rate limiting.

Features:
- JWT token generation and validation
- API key management
- Rate limiting per user
- Role-based access control (RBAC)

Author: AI-Mastery-2026
"""

from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
from typing import Optional, Dict
import jwt
import hashlib
import time
from collections import defaultdict
import threading

# Configuration
SECRET_KEY = "your-secret-key-change-in-production"  # Use env variable in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()


# ============================================================
# JWT TOKEN MANAGEMENT
# ============================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token.
    
    Args:
        data: Payload data (user_id, roles, etc.)
        expires_delta: Token expiration time
    
    Returns:
        JWT token string
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Dict:
    """
    Verify and decode JWT token.
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded payload
    
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """
    Dependency to get current authenticated user.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user = Depends(get_current_user)):
            return {"user_id": user["user_id"]}
    """
    token = credentials.credentials
    payload = verify_token(token)
    return payload


# ============================================================
# API KEY MANAGEMENT
# ============================================================

class APIKeyManager:
    """Manage API keys for service-to-service auth."""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict] = {}
        # In production, store in database
    
    def create_api_key(self, user_id: str, description: str = "") -> str:
        """
        Generate new API key for user.
        
        Returns:
            API key string
        """
        # Generate key hash
        timestamp = str(time.time())
        key_string = f"{user_id}:{timestamp}"
        api_key = hashlib.sha256(key_string.encode()).hexdigest()
        
        self.api_keys[api_key] = {
            "user_id": user_id,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "last_used": None,
            "is_active": True
        }
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """
        Validate API key and return user info.
        
        Returns:
            User info or None if invalid
        """
        key_info = self.api_keys.get(api_key)
        
        if not key_info or not key_info["is_active"]:
            return None
        
        # Update last used
        key_info["last_used"] = datetime.utcnow().isoformat()
        
        return key_info
    
    def revoke_api_key(self, api_key: str):
        """Revoke an API key."""
        if api_key in self.api_keys:
            self.api_keys[api_key]["is_active"] = False


# Global API key manager
api_key_manager = APIKeyManager()


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> Dict:
    """
    Dependency to verify API key from header.
    
    Usage:
        @app.get("/api-protected")
        async def api_protected(user = Depends(verify_api_key)):
            return {"user_id": user["user_id"]}
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    user_info = api_key_manager.validate_api_key(x_api_key)
    
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return user_info


# ============================================================
# RATE LIMITING
# ============================================================

class RateLimiter:
    """
    Token bucket rate limiter.
    
    Limits requests per user to prevent abuse.
    """
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.buckets: Dict[str, Dict] = defaultdict(lambda: {
            "tokens": requests_per_minute,
            "last_update": time.time()
        })
        self.lock = threading.Lock()
    
    def _refill_bucket(self, user_id: str):
        """Refill tokens based on time elapsed."""
        bucket = self.buckets[user_id]
        now = time.time()
        time_elapsed = now - bucket["last_update"]
        
        # Refill rate: requests_per_minute / 60 = requests per second
        tokens_to_add = time_elapsed * (self.requests_per_minute / 60)
        bucket["tokens"] = min(
            self.requests_per_minute,
            bucket["tokens"] + tokens_to_add
        )
        bucket["last_update"] = now
    
    def allow_request(self, user_id: str) -> bool:
        """
        Check if request is allowed for user.
        
        Args:
            user_id: User identifier
        
        Returns:
            True if allowed, False if rate limited
        """
        with self.lock:
            self._refill_bucket(user_id)
            bucket = self.buckets[user_id]
            
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True
            else:
                return False
    
    def get_remaining(self, user_id: str) -> int:
        """Get remaining requests for user."""
        with self.lock:
            self._refill_bucket(user_id)
            return int(self.buckets[user_id]["tokens"])


# Global rate limiter
rate_limiter = RateLimiter(requests_per_minute=60)


async def check_rate_limit(user = Depends(get_current_user)):
    """
    Dependency to enforce rate limiting.
    
    Usage:
        @app.get("/limited")
        async def limited_route(user = Depends(check_rate_limit)):
            return {"message": "success"}
    """
    user_id = user.get("user_id", "anonymous")
    
    if not rate_limiter.allow_request(user_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later.",
            headers={"Retry-After": "60"}
        )
    
    return user


# ============================================================
# ROLE-BASED ACCESS CONTROL (RBAC)
# ============================================================

def require_role(required_role: str):
    """
    Dependency factory for role-based access.
    
    Usage:
        @app.delete("/admin/users/{user_id}")
        async def delete_user(user = Depends(require_role("admin"))):
            return {"deleted": True}
    """
    async def role_checker(user = Depends(get_current_user)):
        user_roles = user.get("roles", [])
        
        if required_role not in user_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        
        return user
    
    return role_checker


# ============================================================
# USAGE EXAMPLES
# ============================================================

if __name__ == "__main__":
    from fastapi import FastAPI
    from pydantic import BaseModel
    
    app = FastAPI()
    
    class LoginRequest(BaseModel):
        username: str
        password: str
    
    @app.post("/auth/login")
    async def login(request: LoginRequest):
        """Login endpoint - returns JWT token."""
        # In production: verify credentials against database
        if request.username == "demo" and request.password == "password":
            token = create_access_token(
                data={"user_id": "user_123", "roles": ["user", "admin"]}
            )
            return {"access_token": token, "token_type": "bearer"}
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    
    @app.post("/auth/api-key")
    async def create_key(user = Depends(get_current_user)):
        """Create API key for authenticated user."""
        api_key = api_key_manager.create_api_key(
            user["user_id"],
            description="Auto-generated key"
        )
        return {"api_key": api_key}
    
    @app.get("/protected")
    async def protected_route(user = Depends(check_rate_limit)):
        """Protected route with rate limiting."""
        return {
            "message": "Access granted",
            "user_id": user["user_id"],
            "remaining_requests": rate_limiter.get_remaining(user["user_id"])
        }
    
    @app.get("/admin/stats")
    async def admin_stats(user = Depends(require_role("admin"))):
        """Admin-only endpoint."""
        return {"total_users": 100, "active_sessions": 15}
    
    print("✅ Authentication module ready")
    print("Example endpoints:")
    print("  POST /auth/login - Get JWT token")
    print("  POST /auth/api-key - Create API key")
    print("  GET /protected - Rate-limited endpoint")
    print("  GET /admin/stats - Admin-only endpoint")
