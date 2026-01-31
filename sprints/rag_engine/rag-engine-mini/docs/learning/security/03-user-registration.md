# User Registration Best Practices

## Learning Objectives

After completing this lesson, you will understand:
- [ ] User registration best practices
- [ ] Email validation strategies
- [ ] Password strength requirements
- [ ] Initial API key generation
- [ ] Registration security considerations

## Prerequisites

- **Knowledge**: Basic Python, understanding of authentication flows
- **Tools**: Python 3.11+, regex for validation
- **Time**: 40-60 minutes

## Concepts Deep Dive

### User Registration Flow

**The Registration Journey**:
```
┌─────────────────────────────────────────────────────┐
│          User Registration Flow                 │
└─────────────────────────────────────────────────────┘

    User                        Server
      │                             │
      │  1. Submit registration    │
      │  (email, password)          │
      ├──────────────────────────────>│
      │                             │
      │  2. Validate email format  │
      │                             │
      │<──────────────────────────────┤
      │  Error if invalid          │
      │                             │
      │  3. Validate email exists? │
      │                             │
      │                             │
      │  4. Validate password     │
      │  5. Hash password           │
      │                             │
      │  6. Create user record      │
      │  7. Generate API key        │
      │  8. Generate JWT tokens      │
      │                             │
      │  9. Return success          │
      │  (user_id, tokens)          │
      │<──────────────────────────────┤
      │                             │
```

**Why Email First?**
- **User Identity**: Email is most common user identifier
- **Communication**: Enables password reset, notifications
- **Verification**: Email can be verified (SMTP check)
- **Uniqueness**: Natural unique constraint

---

### Email Validation Strategies

#### 1. Format Validation

**Regex Approach**:
```python
import re

email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
```

**RFC 5322 Compliance**:
- Local part: `username@domain.com` (username)
- Domain part: Must have at least one dot
- TLD: 2-6+ characters
- Max length: 320 characters (local part 64, domain 255)

**Edge Cases**:
| Email | Valid? | Reason |
|-------|---------|--------|
| `user@example` | ❌ | Missing TLD |
| `user@.com` | ❌ | Missing domain name |
| `user..name@example.com` | ❌ | Consecutive dots |
| `"quoted"@example.com` | ✅ | Quoted local part |
| `user+tag@example.com` | ✅ | Plus addressing |

#### 2. Uniqueness Check

**Database Query**:
```python
def email_exists(email):
    # Check if email already registered
    return db.query(User).filter_by(email=email).first() is not None
```

**Case Sensitivity**:
- **Emails are case-insensitive**: `User@Example.com` = `user@example.com`
- **Store lowercase**: Always `.lower()` before storage
- **Index properly**: Create case-insensitive index

```sql
CREATE UNIQUE INDEX ix_users_email_lower
ON users (LOWER(email));
```

#### 3. Email Verification (Optional but Recommended)

**Flow**:
```
1. Register user → Send verification email
2. User clicks link → Verify email
3. Account becomes fully active
```

**Benefits**:
- Prevents disposable email addresses
- Ensures email is deliverable
- Reduces spam accounts

**Implementation**:
```python
def register_user(email, password):
    # Create user with email_verified=False
    user_id = create_user(email, password, verified=False)
    
    # Send verification email
    send_verification_email(user_id, email)
    
    return user_id
```

---

### Password Strength Requirements

**Why Strong Passwords?**
```
Brute Force Attack Speed (by password length):

Password Length  | Possible Combinations | Time to Crack (GPU)
---------------|----------------------|--------------------
8 chars (lower)    | 26^8 = 208B           | ~0.1 seconds
12 chars (mixed)    | 94^12 = 4.7T          | ~1 million years
16 chars (mixed)    | 94^16 = 44,000T       | > universe age
```

**Industry Standards**:
| Requirement | OWASP Recommendation | Our Implementation |
|-----------|---------------------|--------------------|
| Minimum Length | 8 characters | ✅ 8 characters |
| Complexity | Mixed case + number + special | ✅ All four |
| No Common Passwords | Reject top 1000 | ✅ Basic check |
| No User Info | Don't include name/email | ✅ No enforcement (UI) |

**Our Validation Rules**:
```python
def validate_password(password):
    errors = []
    
    # Length: 8-128 characters
    if len(password) < 8:
        errors.append("At least 8 characters")
    elif len(password) > 128:
        errors.append("Less than 128 characters")
    
    # Uppercase: A-Z
    if not re.search(r'[A-Z]', password):
        errors.append("At least one uppercase letter")
    
    # Lowercase: a-z
    if not re.search(r'[a-z]', password):
        errors.append("At least one lowercase letter")
    
    # Number: 0-9
    if not re.search(r'[0-9]', password):
        errors.append("At least one number")
    
    # Special character: Not A-Za-z0-9
    if not re.search(r'[^A-Za-z0-9]', password):
        errors.append("At least one special character")
    
    return len(errors) == 0, errors
```

**Password Examples**:
| Password | Valid? | Reason |
|----------|---------|--------|
| `Password123` | ❌ | No uppercase, no special char |
| `p@ssw0rd` | ❌ | Too short (7 chars) |
| `MyS3cureP@ss!` | ✅ | Meets all requirements |
| `CorrectHorseBatteryStaple` | ❌ | Common password |
| `UserEmail123!` | ⚠️ | Valid but contains user info |

---

### Password Hashing in Registration

**Why Hash Before Storage?**
```python
❌ BAD:
user = User(email="alice@example.com", password="MyPassword123!")
db.save(user)  # Stored in plain text!

✅ GOOD:
hashed = argon2id.hash("MyPassword123!")
user = User(email="alice@example.com", hashed_password=hashed)
db.save(user)  # Secure!
```

**Benefits**:
- **Security**: Even DB breach doesn't expose passwords
- **Consistency**: Same hash algorithm for all passwords
- **Reproducibility**: Hashing is deterministic (with salt)

**Best Practices**:
1. **Use Argon2id**: Current industry standard
2. **Unique Salt Per User**: Never reuse salts
3. **Configurable Parameters**: Allow tuning as hardware improves
4. **Never Log Passwords**: Not even in debug mode

---

### API Key Generation

**Why API Keys?**
- **Service Authentication**: Different from user password
- **Rotatable**: Can be revoked without password change
- **Traceable**: Track usage per key
- **Scalable**: Multiple keys per user (different applications)

**API Key Format Options**:
| Format | Example | Security | Collision Risk |
|--------|---------|-----------|-----------------|
| **UUID4** | `550e8400-e29b-41d4-a716-446655440000` | High | Very Low |
| **Nanoid** | `V1StGXRg_ZV4GtgaN3` | High | Near Zero |
| **Base64(32)** | `xK5m2pQ7nT9jC8vL1wN3Y1kF` | Medium | Low |
| **Hash** | `8a9f5d...` | Low | Zero |

**Our Choice**: UUID4 (in User model)

**Generation**:
```python
import uuid

api_key = str(uuid.uuid4())
# Output: '550e8400-e29b-41d4-a716-446655440000'

# Or with prefix (easier to identify)
prefix = "rag_"
api_key = f"{prefix}{uuid.uuid4()}"
# Output: 'rag_550e8400-e29b-41d4-a716-446655440000'
```

**API Key Security**:
1. **Generate server-side**: Never client-generated
2. **Show once**: Only on creation (never retrievable)
3. **Hash for storage**: Store hash, not plain key
4. **Revoke capability**: Allow invalidation
5. **Track usage**: Last used, usage count

---

### JWT Tokens on Registration

**Immediate Login Pattern**:
```python
def register_user(email, password):
    # 1. Create user
    user_id = create_user(email, hash_password(password))
    
    # 2. Generate API key
    api_key = generate_api_key()
    
    # 3. Generate JWT tokens (immediate login!)
    access_token = create_access_token(user_id)
    refresh_token = create_refresh_token(user_id)
    
    # 4. Return everything
    return {
        "user_id": user_id,
        "api_key": api_key,
        "access_token": access_token,
        "refresh_token": refresh_token,
    }
```

**Benefits**:
- **Better UX**: No extra login step after registration
- **Faster Onboarding**: User can start immediately
- **Consistency**: Same token flow as login

**When to NOT Auto-Login**:
- **Email verification required**: Wait for email click
- **Admin approval needed**: Wait for manual approval
- **Risk assessment**: High-risk registrations need manual review

---

## Code Walkthrough

### Our Implementation

```python
class RegisterUserUseCase:
    def __init__(self, user_repo, email_validator=None):
        self._repo = user_repo
        self._email_validator = email_validator or SimpleEmailValidator()
    
    def execute(self, request):
        # Step 1: Validate email
        email_valid, email_error = self._email_validator.validate(request.email)
        if not email_valid:
            raise ValueError(f"Email validation failed: {email_error}")
        
        # Step 2: Validate password
        password_valid, password_error = self._validate_password(request.password)
        if not password_valid:
            raise ValueError(f"Password validation failed: {password_error}")
        
        # Step 3: Hash password
        hashed_password = hash_password(request.password)
        
        # Step 4: Create user
        user_id = self._repo.create_user(
            email=request.email,
            hashed_password=hashed_password,
        )
        
        # Step 5: Generate tokens (immediate login)
        jwt_provider = get_jwt_provider()
        access_token = jwt_provider.create_access_token(
            user_id=user_id,
            tenant_id=user_id,
            additional_claims={"email": request.email},
        )
        refresh_token = jwt_provider.create_refresh_token(user_id=user_id)
        
        return RegisterUserResponse(
            user_id=user_id,
            email=request.email,
            message="User registered successfully",
        )
```

### Key Decisions

**Decision 1: Password Requirements**
- **Chosen**: 8+ chars, mixed case, number, special char
- **Reason**: Balances security with user experience
- **Trade-off**: More requirements = more support requests

**Decision 2: Email Validation**
- **Chosen**: Regex format + uniqueness check
- **Reason**: Fast validation, prevents duplicates
- **Trade-off**: Simple regex vs complex RFC parser

**Decision 3: Auto-Login**
- **Chosen**: Generate JWT tokens on registration
- **Reason**: Better UX, faster onboarding
- **Trade-off**: Security risk if email verification required

---

## Practical Exercise

### Exercise 1: Validate Password Strength

**Task**: Implement a password strength checker.

**Solution**:
```python
import re

def check_password_strength(password):
    """Check password strength and return score 0-100."""
    score = 0
    feedback = []
    
    # Length (0-25 points)
    length = len(password)
    if length >= 8:
        score += 10
    if length >= 12:
        score += 10
    if length >= 16:
        score += 5
    
    # Character variety (0-30 points)
    has_lower = re.search(r'[a-z]', password)
    has_upper = re.search(r'[A-Z]', password)
    has_number = re.search(r'[0-9]', password)
    has_special = re.search(r'[^A-Za-z0-9]', password)
    
    variety = sum([has_lower, has_upper, has_number, has_special])
    score += variety * 7.5
    
    # Pattern complexity (0-20 points)
    # Check for sequential, repeated, common patterns
    if not re.search(r'(012|123|234|345|456|567|678|789|890)', password):
        score += 5
    if not re.search(r'(.)\1{2,}', password):  # Repeated characters
        score += 5
    if not re.search(r'(qwerty|asdf|zxcv)', password.lower()):
        score += 5
    if not re.search(r'(password|admin|welcome)', password.lower()):
        score += 5
    
    # Feedback
    if score < 30:
        feedback.append("Very weak")
    elif score < 50:
        feedback.append("Weak")
    elif score < 70:
        feedback.append("Moderate")
    elif score < 90:
        feedback.append("Strong")
    else:
        feedback.append("Very strong")
    
    return score, feedback

# Test
passwords = [
    "password",
    "Password123",
    "MyS3cureP@ss!",
    "C0rr3ctH0rs3B@tt3ryStapl3!2024"
]

for pwd in passwords:
    score, feedback = check_password_strength(pwd)
    print(f"Password: {pwd}")
    print(f"  Score: {score}/100")
    print(f"  Feedback: {', '.join(feedback)}")
    print()
```

### Exercise 2: Email Validation Edge Cases

**Task**: Test various email formats for validity.

**Solution**:
```python
import re

email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

test_emails = [
    "user@example.com",           # ✅ Valid
    "user.name@example.com",       # ✅ Valid (dot in local)
    "user+tag@example.com",       # ✅ Valid (plus)
    "user@sub.example.com",       # ✅ Valid (subdomain)
    "user@123.456.789.com",     # ✅ Valid (numeric domain)
    "user@example",              # ❌ Invalid (no TLD)
    "user@.com",               # ❌ Invalid (no domain)
    "user@com",                # ❌ Invalid (no @)
    "user@@example.com",          # ❌ Invalid (double @)
    "user@example..com",         # ❌ Invalid (double dot)
]

for email in test_emails:
    if re.match(email_regex, email):
        print(f"✓ Valid: {email}")
    else:
        print(f"✗ Invalid: {email}")
```

### Exercise 3: Complete Registration Flow

**Task**: Simulate complete registration with API.

**Solution**:
```python
from src.api.v1.routes_auth import RegisterRequest
import requests

# Step 1: Register user
registration_data = {
    "email": "newuser@example.com",
    "password": "MyS3cureP@ss!",
}

response = requests.post(
    "http://localhost:8000/api/v1/auth/register",
    json=registration_data,
)

print("=== Registration Response ===")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

# Step 2: Extract tokens
if response.status_code == 201:
    data = response.json()
    access_token = data.get("access_token")
    user_id = data.get("user_id")
    
    print(f"\nUser ID: {user_id}")
    print(f"Access Token: {access_token[:50]}...")
    
    # Step 3: Use token to authenticate
    # (In a real app, you'd store this and use it for API calls)
    print("\n✓ Registration complete!")
    print("  Next: Use access_token in Authorization header")
else:
    print(f"\n✗ Registration failed: {response.text}")
```

---

## Testing & Validation

### Unit Tests

We've implemented tests in `tests/integration/test_auth_flow.py`:

- ✅ Email format validation
- ✅ Password strength validation
- ✅ Duplicate email detection
- ✅ Password hashing verification
- ✅ API endpoint validation

### Integration Tests

Test the complete flow:

```bash
# Test registration with valid data
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "SecurePass123!"
  }'

# Test registration with short password
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "short1!"
  }'

# Test registration with invalid email
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "not-an-email",
    "password": "SecurePass123!"
  }'
```

### Security Tests

```bash
# Test SQL injection in email
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com'\'' OR '1'='1",
    "password": "SecurePass123!"
  }'

# Expected: Email validation should reject SQLi pattern

# Test XSS in email
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "<script>alert(1)</script>@example.com",
    "password": "SecurePass123!"
  }'

# Expected: Email validation should reject HTML/script patterns
```

---

## Production Considerations

### Performance Impact

| Operation | Time | Bottleneck |
|-----------|------|------------|
| Email validation | <1ms | Regex (negligible) |
| Password hashing | ~250ms | Argon2 (intentional) |
| User creation | <10ms | Database insert |
| JWT generation | <5ms | CPU (negligible) |
| **Total** | **~266ms** | Acceptable |

**Optimizations**:
- Cache email regex (compiled pattern)
- Async password hashing (if very slow)
- Batch insertions (bulk registration)

### Security Considerations

1. **Rate limiting**: Prevent registration abuse (10 per IP per minute)
2. **CAPTCHA**: For suspicious registrations
3. **Email verification**: Verify email is real and deliverable
4. **Password logging**: Never log passwords (even in debug)
5. **Error messages**: Generic to prevent enumeration
6. **API key security**: Hash for storage, show only once

### Operational Concerns

1. **Email delivery**: Ensure SMTP reliability, handle bounces
2. **Database constraints**: Unique email index, case-insensitive
3. **Password policies**: Document requirements clearly to users
4. **Support workflow**: How users reset passwords
5. **Account recovery**: Email-based password reset flow

### Monitoring Recommendations

Track metrics:
```python
# Registration metrics
metric("user_registered", tags=["source", "web"])
metric("registration_failed", tags=["reason", "email_taken"])
metric("registration_failed", tags=["reason", "weak_password"])

# Security events
metric("suspicious_registration", tags=["reason", "rate_limit"])
metric("suspicious_registration", tags=["reason", "disposable_email"])
metric("suspicious_registration", tags=["reason", "known_spammer"])

# Performance metrics
histogram("registration_latency_ms", tags=["step", "email_validation"])
histogram("registration_latency_ms", tags=["step", "password_hashing"])
histogram("registration_latency_ms", tags=["step", "user_creation"])
```

---

## Further Reading

### Resources

- [OWASP Registration Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Registration_Cheat_Sheet.html)
- [RFC 5322 Email Format](https://datatracker.ietf.org/doc/html/rfc5322)
- [Password Strength Meter](https://github.com/dropbox/zxcvbn)
- [Email Validation Best Practices](https://emailregex.com/)

### Libraries

- [email-validator](https://pypi.org/project/email-validator/) - Email validation
- [zxcvbn](https://github.com/dropbox/zxcvbn) - Password strength estimator
- [haveibeenpwned](https://haveibeenpwned.com/) - Check if password was breached

---

## Quiz

1. **Why should emails be stored in lowercase in the database?**
   - A) To save storage space
   - B) Emails are case-insensitive, so 'A@B.com' = 'a@b.com'
   - C) To make searching faster
   - D) To prevent duplicate emails with different cases

2. **What is the minimum password length requirement in our implementation?**
   - A) 6 characters
   - B) 8 characters
   - C) 10 characters
   - D) 12 characters

3. **Why generate JWT tokens immediately after registration?**
   - A) To save an API call
   - B) To improve user experience (no extra login step)
   - C) To verify the user's email
   - D) To generate an API key

4. **What information should be included in an API key?**
   - A) User's email and password
   - B) User's ID and permissions
   - C) Unique identifier (no sensitive data)
   - D) User's full name and address

5. **What is the primary security benefit of hashing passwords before storage?**
   - A) Faster database queries
   - B) Even if database is breached, attackers cannot recover original passwords
   - C) Passwords are easier to remember
   - D) Reduces database storage requirements

---

## Answer Key

1. **B** - Emails are case-insensitive ('A@B.com' = 'a@b.com')
2. **B** - Our implementation requires at least 8 characters
3. **B** - Better UX, faster onboarding (no extra login)
4. **C** - API keys should contain unique identifier only (no sensitive data)
5. **B** - Hashing is one-way; even with hash, cannot recover password

---

## Next Steps

After completing this lesson:
- [ ] User Login and Session Management (Lesson 4)
- [ ] API Key Management (Lesson 5)
- [ ] Rate Limiting (Lesson 6)
- [ ] Input Validation and Sanitization (Lesson 7)
