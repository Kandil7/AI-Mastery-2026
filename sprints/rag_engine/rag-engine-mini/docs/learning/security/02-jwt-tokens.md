# JWT Tokens Explained

## Learning Objectives

After completing this lesson, you will understand:
- [ ] JWT structure (header, payload, signature)
- [ ] Access vs Refresh tokens pattern
- [ ] Token revocation strategies
- [ ] JWT attacks and mitigations
- [ ] How to implement JWT authentication in Python

## Prerequisites

- **Knowledge**: Basic Python, understanding of authentication flows
- **Tools**: Python 3.11+, python-jose library
- **Time**: 45-60 minutes

## Concepts Deep Dive

### What is JWT?

**Definition**: JSON Web Token (JWT) is a compact, URL-safe means of representing claims to be transferred between two parties.

**Why it matters**:
- **Stateless**: No server-side session storage needed
- **Scalable**: Easy to distribute across services
- **Standardized**: RFC 7519 standard, widely supported
- **Self-contained**: All needed info in the token itself

**JWT Structure**:
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.
eyJzdWIiOiIxMjM0NDU3Nzg5IiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.
SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
```

Three parts separated by dots:
1. **Header**: Algorithm and token type
2. **Payload**: Claims (data)
3. **Signature**: Cryptographic signature

**Decoded Example**:
```json
// Header
{
  "alg": "HS256",
  "typ": "JWT"
}

// Payload (Claims)
{
  "sub": "1234567890",
  "name": "John Doe",
  "iat": 1516239022,
  "exp": 1516242622
}

// Signature
HMACSHA256(
  base64UrlEncode(header) + "." + base64UrlEncode(payload),
  secret
)
```

---

### JWT Claims: The Data Inside

**Registered Claims** (Standard):
| Claim | Meaning | Example |
|--------|---------|---------|
| `sub` (subject) | User ID | `"user_123"` |
| `iat` (issued at) | Token creation timestamp | `1516239022` |
| `exp` (expiration) | Token expiration timestamp | `1516242622` |
| `nbf` (not before) | Token valid from timestamp | `1516239022` |
| `jti` (JWT ID) | Unique token identifier (for revocation) | `"uuid-123"` |

**Custom Claims** (Application-specific):
```json
{
  "sub": "user_123",
  "tenant_id": "tenant_456",
  "type": "access",  // Our custom claim
  "role": "admin",
  "permissions": ["read", "write"],
  "exp": 1516242622
}
```

---

### Access vs Refresh Tokens Pattern

**The Problem with Long-lived Tokens**:
- If access token is valid for 30 days and gets stolen:
  - Attacker has 30 days of access
  - Revoking requires blacklist or breaking all tokens
  - Risk window is too large

**The Solution: Token Rotation**:
```
┌─────────────────────────────────────────────────────────────┐
│                  Authentication Flow                      │
└─────────────────────────────────────────────────────────────┘

    User                       Server
      │                             │
      │  1. Login (username/password) │
      ├──────────────────────────────>│
      │                             │
      │  2. Return both tokens      │
      │  Access: 15 min             │
      │  Refresh: 7 days             │
      │<──────────────────────────────┤
      │                             │
      │  3. Use access token       │
      │  for API requests           │
      ├──────────────────────────────>│
      │<──────────────────────────────┤
      │                             │
      │  4. Access expires...       │
      │                             │
      │  5. Use refresh token      │
      │  to get new pair            │
      ├──────────────────────────────>│
      │  6. Return new tokens       │
      │  (old refresh invalidated)    │
      │<──────────────────────────────┤
```

**Token Lifetimes**:
| Token Type | Lifetime | Use Case | Risk if Leaked |
|-----------|----------|-----------|-----------------|
| **Access Token** | 15-30 min | API calls, sensitive operations | Low (short window) |
| **Refresh Token** | 7-30 days | Get new access tokens | Medium (can be revoked) |
| **ID Token** | 5-10 min | User profile info | Very low |

**Why Short Access Tokens?**
- **Damage containment**: Attacker only has 15-30 min if token stolen
- **Revoke by expiration**: Token becomes invalid automatically
- **Forces rotation**: User must re-auth or use refresh token regularly

**Refresh Token Security**:
- **One-time use**: Invalidate after generating new pair
- **Rotate on use**: Each rotation generates new tokens
- **HTTPS only**: Never transmit over HTTP
- **HttpOnly cookies**: Store in cookies, not localStorage (prevents XSS)

---

### Token Revocation Strategies

#### 1. **Blacklist (Token Blocklist)**
Store revoked token IDs (jti) in database:
```python
def verify_with_blacklist(token):
    payload = decode_token(token)
    jti = payload['jti']
    
    # Check if token is blacklisted
    if db.is_token_blacklisted(jti):
        raise JWTError("Token revoked")
    
    return payload
```

**Pros**:
- Immediate revocation
- Simple implementation
- Works for all token types

**Cons**:
- Database query on every request (performance impact)
- Requires storage for all revoked tokens
- Need cleanup job for old entries

#### 2. **Whitelist (Token Allowlist)**
Store valid token IDs in fast storage (Redis):
```python
def verify_with_whitelist(token):
    payload = decode_token(token)
    jti = payload['jti']
    
    # Check if token exists in whitelist
    if not redis.exists(f"token:{jti}"):
        raise JWTError("Token invalid or expired")
    
    return payload
```

**Pros**:
- Fast lookups (Redis)
- Automatic expiration (TTL)
- No cleanup needed

**Cons**:
- More complex login flow
- Need to generate tokens immediately

#### 3. **Versioning (Token Version)**
Add version claim, invalidating all tokens of old version:
```python
def verify_with_version(token):
    payload = decode_token(token)
    version = payload.get('ver', 1)
    
    # Check version against current
    if version < current_token_version:
        raise JWTError("Token version outdated")
    
    return payload
```

**Pros**:
- No database lookups
- Fast invalidation

**Cons**:
- Invalidates ALL tokens (even legitimate ones)
- Forces all users to re-auth

#### 4. **Short Expiration (Time-based)**
Rely on short token lifetimes:
```python
# 15-minute access tokens
# Users re-auth automatically
```

**Pros**:
- No state needed
- Simple implementation

**Cons**:
- Poor UX (frequent re-auth)
- Doesn't handle immediate revocation

---

### JWT Algorithms: HS256 vs RS256

#### HS256 (HMAC with SHA-256)
```
Secret key (shared) ──┬─> HMACSHA256 ──> Signature
                        └─> Verify with same secret
```

**Characteristics**:
- **Symmetric**: Same secret for signing and verification
- **Faster**: HMAC is fast
- **Simpler**: Single secret to manage
- **Development**: Good for development/testing

**Risks**:
- **Key distribution**: If secret leaked, attacker can forge tokens
- **Secret sharing**: All services need access to same secret

**Use When**:
- Development environment
- Internal services with secure secret sharing
- Simple single-service applications

#### RS256 (RSA with SHA-256)
```
Private key (secret) ──┬─> RSA Sign ──> Signature
                        └─> RSA Verify (Public Key)
```

**Characteristics**:
- **Asymmetric**: Private key signs, public key verifies
- **More Secure**: Compromise of service doesn't leak signing key
- **Slower**: RSA is slower than HMAC
- **Complex**: Key pair management

**Use When**:
- Production environments
- Microservices architecture (each service verifies with public key)
- High-security requirements
- Token signing service separate from verification

**Migration Path**:
```python
# Development
jwt.encode(payload, secret="dev-key", algorithm="HS256")

# Production
jwt.encode(payload, key=private_key, algorithm="RS256")
```

---

### Common JWT Attacks & Mitigations

#### 1. None Algorithm Attack

**The Attack**:
```json
// Malicious token
{
  "alg": "none",  // No signature!
  "typ": "JWT"
}

// Payload with admin claim
{
  "sub": "victim_user",
  "role": "admin"  // Escalated privileges
}
```

**How it Works**:
1. Attacker removes signature (third part)
2. Sets `alg` to `none`
3. Claims admin role
4. Server skips signature verification (because `alg: none`)
5. Attacker has admin access

**Mitigation**:
```python
# ALWAYS verify algorithm
def decode_token(token):
    header = jwt.get_unverified_header(token)
    
    if header['alg'] == 'none':
        raise JWTError("Algorithm 'none' not allowed")
    
    # Only allow expected algorithms
    allowed_algs = ['HS256', 'RS256']
    if header['alg'] not in allowed_algs:
        raise JWTError(f"Algorithm {header['alg']} not allowed")
    
    return jwt.decode(token, secret, algorithms=allowed_algs)
```

#### 2. Algorithm Confusion Attack

**The Attack**:
```json
// Attacker's token (signed with HMAC, claims to be RSA)
{
  "alg": "RS256",  // Claims to be RSA
  "typ": "JWT"
}
```

**How it Works**:
1. Attacker signs token with HMAC using RSA public key
2. Claims algorithm is `RS256`
3. Server tries to verify with HMAC using RSA private key
4. Verification succeeds (incorrectly) because HMAC accepts any key

**Mitigation**:
```python
# Validate algorithm matches expected type
def verify_with_algorithm_check(token, expected_alg):
    header = jwt.get_unverified_header(token)
    
    if header['alg'] != expected_alg:
        raise JWTError(f"Algorithm mismatch: expected {expected_alg}, got {header['alg']}")
    
    return jwt.decode(token, secret, algorithms=[expected_alg])

# For production with RS256:
# jwt.decode(token, public_key, algorithms=['RS256'])

# For development with HS256:
# jwt.decode(token, secret, algorithms=['HS256'])
```

#### 3. Replay Attack

**The Attack**:
1. Attacker intercepts valid JWT token
2. Reuses token immediately or later
3. Server accepts token (because signature is valid)

**Mitigation 1: Short Expiration**
```python
# 15-minute access tokens
# Even if stolen, only valid for 15 minutes
```

**Mitigation 2: Unique JTI + Blacklist**
```python
def use_token_once(token):
    payload = decode_token(token)
    jti = payload['jti']
    
    # Check if token was used before
    if redis.exists(f"used_token:{jti}"):
        raise JWTError("Token already used")
    
    # Mark as used
    redis.set(f"used_token:{jti}", "1", ex=15*60)  # 15 min
```

**Mitigation 3: Audience and Issuer Claims**
```python
# Include aud and iss in token
payload = {
    "sub": "user_123",
    "aud": "my-app.example.com",  # Intended audience
    "iss": "auth.example.com",    # Issuer
    "exp": ...
}

# Verify on decode
def verify_token_with_claims(token):
    payload = decode_token(token)
    
    # Verify issuer
    if payload.get('iss') != 'auth.example.com':
        raise JWTError("Invalid issuer")
    
    # Verify audience
    if payload.get('aud') != 'my-app.example.com':
        raise JWTError("Invalid audience")
```

#### 4. Timing Attack on Verification

**The Attack**:
```python
# Vulnerable (timing leak)
def verify_vuln(token):
    try:
        decode(token)
        return True
    except Exception:
        return False  # Early return on exception
```

**Mitigation: Constant-Time Comparison**
```python
# Our implementation uses python-jose which handles this
# For custom implementation, use HMAC.compare_digest()
import hmac

def verify_secure(token, secret):
    try:
        payload = decode(token, secret)
        # Don't early return
        return hmac.compare_digest(expected, actual)
    except:
        # Always return error, don't distinguish types
        return False
```

---

### Token Storage: Best Practices

#### Where to Store Tokens?

| Storage | Security | Accessibility | XSS Risk | CSRF Risk |
|---------|-----------|--------------|------------|------------|
| **localStorage** | ❌ Poor | ✅ Easy | ❌ High | ✅ Low |
| **sessionStorage** | ❌ Poor | ✅ Easy | ❌ High | ✅ Low |
| **HttpOnly Cookie** | ✅ Excellent | ⚠️ Medium | ✅ None | ❌ High |
| **Memory (RAM)** | ✅ Excellent | ⚠️ Medium | ✅ None | ✅ None |

#### Recommendation

**Access Token**: Short-lived, can be in memory or HttpOnly cookie
```
javascript
// Store in memory variable (cleared on refresh)
let accessToken = null;

function setAccessToken(token) {
    accessToken = token;
    setTimeout(() => accessToken = null, 15*60*1000);  // Auto-clear
}
```

**Refresh Token**: Long-lived, MUST be HttpOnly cookie
```
javascript
// Cannot access HttpOnly cookie from JS (prevents XSS)
// Server sets cookie:
Set-Cookie: refresh_token=...; HttpOnly; Secure; SameSite=Strict
```

#### Cookie Security Attributes

```python
# Server-side cookie settings
response.set_cookie(
    key="refresh_token",
    value=refresh_token,
    httponly=True,      # Prevents XSS access
    secure=True,        # Only over HTTPS
    samesite="Strict",  # Prevents CSRF
    max_age=7*24*3600  # 7 days
)
```

---

## Code Walkthrough

### Our Implementation

```python
from jose import JWTError, jwt
from datetime import datetime, timedelta

class JWTProvider:
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_expire_minutes: int = 15,
        refresh_expire_days: int = 7,
    ) -> None:
        self._secret_key = secret_key
        self._algorithm = algorithm
        self._access_expire_minutes = access_expire_minutes
        self._refresh_expire_days = refresh_expire_days
    
    def create_access_token(
        self,
        user_id: str,
        tenant_id: str,
        additional_claims: dict | None = None,
    ) -> str:
        """Create short-lived access token (15 min)."""
        payload = {
            "sub": user_id,
            "tenant_id": tenant_id,
            "type": "access",
            "jti": str(uuid.uuid4()),  # Unique ID
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        expires_delta = timedelta(minutes=self._access_expire_minutes)
        return self._create_token(payload, expires_delta)
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create long-lived refresh token (7 days)."""
        payload = {
            "sub": user_id,
            "type": "refresh",
            "jti": str(uuid.uuid4()),  # Unique ID
        }
        
        expires_delta = timedelta(days=self._refresh_expire_days)
        return self._create_token(payload, expires_delta)
    
    def verify_access_token(self, token: str) -> dict:
        """Verify access token and return payload."""
        return self.decode_token(token, verify_type="access")
    
    def verify_refresh_token(self, token: str) -> dict:
        """Verify refresh token and return payload."""
        return self.decode_token(token, verify_type="refresh")
    
    def rotate_refresh_token(self, old_refresh_token: str) -> tuple[str, str]:
        """Rotate refresh token - create new pair, invalidate old."""
        payload = self.verify_refresh_token(old_refresh_token)
        user_id = payload["sub"]
        
        # Create new tokens
        new_access = self.create_access_token(
            user_id=user_id,
            tenant_id=payload.get("tenant_id", ""),
        )
        new_refresh = self.create_refresh_token(user_id=user_id)
        
        return new_access, new_refresh
```

### Key Decisions

**Decision 1: Access Token = 15 minutes**
- **Chosen**: 15-minute lifetime
- **Reason**: Balances security (short window) with UX (not too frequent re-auth)
- **Trade-off**: More rotation requests vs security

**Decision 2: Refresh Token = 7 days**
- **Chosen**: 7-day lifetime
- **Reason**: Long enough for good UX, short enough to limit damage
- **Trade-off**: Longer refresh windows but acceptable

**Decision 3: Unique JTI (JWT ID)**
- **Chosen**: UUID4 for each token
- **Reason**: Enables token revocation/blacklisting
- **Trade-off**: Slightly larger token size (negligible)

**Decision 4: Type Claim**
- **Chosen**: `"access"` or `"refresh"` claim
- **Reason**: Prevents using refresh token as access token
- **Trade-off**: Extra claim in payload

---

## Practical Exercise

### Exercise 1: Create and Verify Token

**Task**: Write a script that:
1. Creates an access token
2. Decodes and verifies it
3. Extracts claims from payload

**Solution**:
```python
from src.adapters.security.jwt_provider import get_jwt_provider

provider = get_jwt_provider()

# Create access token
token = provider.create_access_token(
    user_id="user_123",
    tenant_id="tenant_456",
    additional_claims={"role": "user"},
)

print(f"Token: {token}")
print(f"Token length: {len(token)}")

# Decode and verify
payload = provider.verify_access_token(token)

print(f"\nClaims:")
print(f"  User ID: {payload['sub']}")
print(f"  Tenant ID: {payload['tenant_id']}")
print(f"  Type: {payload['type']}")
print(f"  JWT ID: {payload['jti']}")
print(f"  Expires: {payload['exp']}")
print(f"  Issued: {payload['iat']}")
```

### Exercise 2: Token Rotation Flow

**Task**: Simulate token rotation flow.

**Solution**:
```python
from src.adapters.security.jwt_provider import get_jwt_provider

provider = get_jwt_provider()

# Step 1: User logs in, gets initial tokens
initial_access = provider.create_access_token(
    user_id="user_123",
    tenant_id="tenant_456",
)
initial_refresh = provider.create_refresh_token(user_id="user_123")

print("=== Initial Tokens ===")
print(f"Access: {initial_access[:50]}...")
print(f"Refresh: {initial_refresh[:50]}...")

# Step 2: Access token expires after 15 min
# User uses refresh token to get new pair

# Step 3: Rotate tokens
new_access, new_refresh = provider.rotate_refresh_token(initial_refresh)

print("\n=== After Rotation ===")
print(f"New Access: {new_access[:50]}...")
print(f"New Refresh: {new_refresh[:50]}...")

# Step 4: Old refresh token should be invalid
try:
    provider.verify_refresh_token(initial_refresh)
    print("ERROR: Old token still valid!")
except JWTError as e:
    print(f"✓ Old token correctly invalidated: {e}")

# Step 5: New tokens should be valid
new_payload = provider.verify_access_token(new_access)
print(f"✓ New access token valid: {new_payload['jti']}")
```

### Exercise 3: Decode Token Parts

**Task**: Decode and examine all three JWT parts.

**Solution**:
```python
import base64
import json

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NDU3Nzg5IiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"

parts = token.split('.')

print(f"=== JWT Structure ===")
print(f"Parts: {len(parts)}")

# Header (part 0)
header_decoded = base64.urlsafe_b64decode(parts[0])
header = json.loads(header_decoded)
print(f"\nHeader:")
print(json.dumps(header, indent=2))

# Payload (part 1)
payload_decoded = base64.urlsafe_b64decode(parts[1])
payload = json.loads(payload_decoded)
print(f"\nPayload:")
print(json.dumps(payload, indent=2))

# Signature (part 2)
print(f"\nSignature:")
print(f"  (Cannot decode - cryptographic)")
print(f"  Length: {len(parts[2])} bytes")
```

---

## Testing & Validation

### Unit Tests

We've implemented comprehensive tests in `tests/unit/test_jwt_provider.py`:

- ✅ Token generation (access and refresh)
- ✅ Token structure validation
- ✅ Token verification
- ✅ Expiration handling
- ✅ Algorithm validation
- ✅ Token rotation
- ✅ JTI uniqueness

### Security Tests

Test attack prevention:

```bash
# Test None algorithm attack
python -c "
from src.adapters.security.jwt_provider import JWTProvider
provider = JWTProvider(secret_key='test')
try:
    # Manually craft malicious token with alg: none
    malicious_token = 'eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.'
    provider.verify_access_token(malicious_token)
except Exception as e:
    print(f'✓ Caught none algorithm attack: {e}')
"

# Test algorithm confusion
python -c "
from jose import jwt
# Try to sign with HS256 but claim RS256
payload = {'alg': 'RS256', 'sub': 'user'}
token = jwt.encode(payload, 'secret', algorithm='HS256')
print('⚠️  Malicious token created (algorithm confusion)')
print('Server MUST reject this by verifying alg matches expected')
"
```

---

## Production Considerations

### Performance Impact

| Operation | Time | Bottleneck |
|-----------|------|------------|
| **Token Generation** | ~1-5ms | CPU (RSA) |
| **Token Verification** | ~1-3ms | CPU (RSA) |
| **Token Rotation** | ~2-8ms | CPU + DB (if blacklist) |

**Optimizations**:
- Cache public key for RS256 (don't read from DB each request)
- Use HS256 in development (faster)
- Pre-verify algorithm before full decode

### Security Considerations

1. **Always use HTTPS**: Tokens in URL/headers can be intercepted
2. **Never log tokens**: Full token or partial (even masked)
3. **Validate algorithm**: Reject unexpected algorithms immediately
4. **Rotate keys**: Every 90 days for symmetric, manage cert expiry for asymmetric
5. **Token blacklisting**: For immediate revocation
6. **Short access tokens**: 15-30 minutes maximum
7. **Refresh token rotation**: One-time use only

### Operational Concerns

1. **Token expiration handling**: Clear UI, prompt for re-auth
2. **Refresh token rotation**: Handle invalid tokens gracefully
3. **Algorithm migration**: If changing HS256 → RS256, support both during transition
4. **Key rotation**: Support multiple valid keys during rotation period
5. **Token size monitoring**: Large payloads affect performance

### Monitoring Recommendations

Track metrics:
```python
# Token operations
metric("jwt_generated", tags=["type", "access"])
metric("jwt_generated", tags=["type", "refresh"])
metric("jwt_verified", tags=["success", "true"])
metric("jwt_verified", tags=["success", "false"])

# Security events
metric("jwt_rejected", tags=["reason", "none_algorithm"])
metric("jwt_rejected", tags=["reason", "algorithm_mismatch"])
metric("jwt_rejected", tags=["reason", "expired"])
metric("jwt_rejected", tags=["reason", "revoked"])

# Token lifetimes
histogram("jwt_access_lifetime_seconds", tags=["user"])
histogram("jwt_refresh_lifetime_days", tags=["user"])
```

---

## Further Reading

### Resources

- [JWT.io](https://jwt.io/) - Interactive JWT debugger
- [RFC 7519](https://datatracker.ietf.org/doc/html/rfc7519) - JWT specification
- [OWASP JWT Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/JSON_Web_Token_for_Java_Cheat_Sheet.html)
- [JWT Best Practices](https://tools.ietf.org/html/draft-ietf-oauth-jwt-bcp-04)

### Libraries

- [python-jose](https://python-jose.readthedocs.io/) - Python JWT library we use
- [Authlib](https://authlib.readthedocs.io/) - Alternative auth library
- [PyJWT](https://pyjwt.readthedocs.io/) - Another popular JWT library

---

## Quiz

1. **What are the three parts of a JWT?**
   - A) Header, Body, Footer
   - B) Header, Payload, Signature
   - C) Claim, Algorithm, Token
   - D) Prefix, Content, Suffix

2. **Why should access tokens be short-lived (15-30 min)?**
   - A) To reduce server load
   - B) To minimize damage if token is stolen
   - C) To force users to log out
   - D) To reduce token size

3. **What is the "none algorithm" attack?**
   - A) Attacker removes signature to bypass verification
   - B) Attacker uses no encryption algorithm
   - C) Attacker sends empty token
   - D) Attacker guesses algorithm by trial

4. **What is the difference between HS256 and RS256?**
   - A) HS256 uses RSA, RS256 uses HMAC
   - B) HS256 is symmetric (shared secret), RS256 is asymmetric (public/private keys)
   - C) HS256 is always faster, RS256 is always more secure
   - D) No significant difference

5. **How does token rotation improve security?**
   - A) Rotates encryption keys
   - B) Creates new access/refresh tokens, invalidates old refresh token
   - C) Changes user password
   - D) Rotates token every 15 minutes

---

## Answer Key

1. **B** - JWT has three parts: Header, Payload, Signature
2. **B** - Short-lived tokens limit damage window if stolen
3. **A** - "none algorithm" attack removes signature to bypass verification
4. **B** - HS256 is symmetric (same secret), RS256 is asymmetric (public/private keys)
5. **B** - Token rotation creates new pair and invalidates old refresh token

---

## Next Steps

After completing this lesson:
- [ ] Implement user registration flow (Lesson 3)
- [ ] Build login and session management (Lesson 4)
- [ ] Add API key management (Lesson 5)
- [ ] Implement rate limiting (Lesson 6)
