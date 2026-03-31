"""
Auth Routes - User Registration & Login
======================================
Endpoints for authentication flows.

نقاط نهاية مصادقة المستخدم
"""

from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, Field, EmailStr

from src.application.use_cases.register_user import (
    RegisterUserUseCase,
    RegisterUserRequest,
    RegisterUserResponse,
)
from src.application.use_cases.login_user import (
    LoginUserUseCase,
    LoginUserRequest,
)
from src.core.bootstrap import get_container

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


# ============================================================================
# Request/Response Models
# ============================================================================


class RegisterRequest(BaseModel):
    """Request model for user registration."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password (8+ chars, mixed case, number, special char)",
    )


class RegisterResponse(BaseModel):
    """Response model for user registration."""

    user_id: str = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    message: str = Field(..., description="Registration status message")


class LoginRequest(BaseModel):
    """Request model for user login."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class LoginResponse(BaseModel):
    """Response model for user login."""

    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration in seconds")


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/register", response_model=RegisterResponse, status_code=201)
async def register_user(
    request: RegisterRequest,
) -> RegisterResponse:
    """
    Register a new user account.

    Flow:
    1. Validate email format and uniqueness
    2. Validate password strength
    3. Hash password with Argon2
    4. Create user record
    5. Generate JWT tokens (immediate login)

    Returns:
        User ID, email, and success message

    تسجيل مستخدم جديد
    """
    container = get_container()
    user_repo = container.get("user_repo")
    if not user_repo:
        raise HTTPException(status_code=501, detail="User repository not configured")

    use_case = RegisterUserUseCase(user_repo=user_repo)

    # Execute registration
    try:
        response = use_case.execute(
            RegisterUserRequest(
                email=request.email,
                password=request.password,
            )
        )
        return RegisterResponse(
            user_id=response.user_id,
            email=response.email,
            message=response.message,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.post("/login", response_model=LoginResponse)
async def login_user(
    request: LoginRequest,
) -> LoginResponse:
    """
    Login with email and password.

    Flow:
    1. Validate credentials
    2. Generate JWT access + refresh tokens
    3. Return tokens to client

    Returns:
        JWT access and refresh tokens

    تسجيل الدخول للمستخدم
    """
    container = get_container()
    user_repo = container.get("user_repo")
    if not user_repo:
        raise HTTPException(status_code=501, detail="User repository not configured")

    use_case = LoginUserUseCase(user_repo=user_repo)
    try:
        result = use_case.execute(
            LoginUserRequest(email=request.email, password=request.password)
        )
        return LoginResponse(
            access_token=result.access_token,
            refresh_token=result.refresh_token,
            token_type="Bearer",
            expires_in=15 * 60,
        )
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")


@router.post("/refresh")
async def refresh_tokens(
    refresh_token: str = Header(..., alias="X-Refresh-Token"),
) -> dict:
    """
    Refresh access token using refresh token.

    Flow:
    1. Verify refresh token
    2. Generate new access + refresh tokens
    3. Invalidate old refresh token

    تجديد رمز الوصول باستخدام رمز التحديث
    """
    container = get_container()
    jwt_provider = container.get("jwt_provider")

    if not jwt_provider:
        raise HTTPException(status_code=501, detail="JWT provider not configured")

    try:
        # Rotate refresh token
        new_access, new_refresh = jwt_provider.rotate_refresh_token(refresh_token)

        return {
            "access_token": new_access,
            "refresh_token": new_refresh,
            "token_type": "Bearer",
            "expires_in": 15 * 60,  # 15 minutes in seconds
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token refresh failed: {str(e)}")
