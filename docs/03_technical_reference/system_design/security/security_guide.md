# Security Guide

Security best practices for deploying AI-Mastery-2026 in production.

---

## 1. API Security

### Authentication

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
import jwt

# API Key Authentication
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    valid_keys = os.environ.get("API_KEYS", "").split(",")
    if api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(request: PredictionRequest):
    ...
```

### Rate Limiting

```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("100/minute")
async def predict(request: Request):
    ...
```

### Input Validation

```python
from pydantic import BaseModel, Field, validator

class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_items=1, max_items=1000)
    
    @validator('features')
    def validate_features(cls, v):
        if any(math.isnan(x) or math.isinf(x) for x in v):
            raise ValueError("Invalid feature values")
        return v
```

---

## 2. Model Security

### Adversarial Detection

```python
class AdversarialDetector:
    def is_suspicious(self, input_data, threshold=3.0):
        z_scores = np.abs((input_data - self.mean) / self.std)
        return np.any(z_scores > threshold)
    
    def sanitize(self, input_data):
        return np.clip(input_data, self.min_val, self.max_val)
```

### Model Integrity

```python
import hashlib

def verify_model_integrity(model_path, expected_hash):
    sha256 = hashlib.sha256()
    with open(model_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_hash
```

### Prompt Injection Prevention

```python
INJECTION_PATTERNS = [
    r"ignore previous",
    r"disregard prior",
    r"system prompt",
]

def is_safe_input(user_input):
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, user_input, re.IGNORECASE):
            return False
    return True
```

---

## 3. Data Privacy

### PII Detection

```python
import re

PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
}

def detect_pii(text):
    found = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            found[pii_type] = matches
    return found

def redact_pii(text):
    for pattern in PII_PATTERNS.values():
        text = re.sub(pattern, '[REDACTED]', text)
    return text
```

### Data Encryption

```python
from cryptography.fernet import Fernet

class DataEncryptor:
    def __init__(self, key=None):
        self.key = key or Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def encrypt(self, data: str) -> bytes:
        return self.cipher.encrypt(data.encode())
    
    def decrypt(self, encrypted: bytes) -> str:
        return self.cipher.decrypt(encrypted).decode()
```

---

## 4. Infrastructure Security

### Docker Hardening

```dockerfile
# Use non-root user
RUN useradd -m -s /bin/bash appuser
USER appuser

# Read-only filesystem
docker run --read-only --tmpfs /tmp app:latest
```

### Secret Management

```yaml
# docker-compose.yml with secrets
services:
  api:
    secrets:
      - db_password
      - api_key

secrets:
  db_password:
    file: ./secrets/db_password.txt
```

### Network Security

```yaml
services:
  api:
    networks:
      - frontend
  
  postgres:
    networks:
      - backend  # Not exposed externally
```

---

## 5. Compliance Checklist

| Requirement | Implementation |
|-------------|----------------|
| Authentication | API keys + JWT |
| Encryption | TLS 1.3 + AES-256 |
| Logging | Audit logs with retention |
| Access Control | RBAC |
| Data Retention | Auto-delete after 90 days |
| PII Handling | Detect and redact |

---

## Quick Security Checklist

- [ ] Enable HTTPS/TLS
- [ ] Implement API authentication
- [ ] Add rate limiting
- [ ] Validate all inputs
- [ ] Scan for PII in logs
- [ ] Use non-root containers
- [ ] Verify model integrity
- [ ] Monitor for anomalies
- [ ] Regular security audits
