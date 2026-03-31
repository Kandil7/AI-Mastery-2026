# Security Testing Guide

## Overview

Security testing ensures your RAG Engine is protected against common vulnerabilities and attacks. This guide covers penetration testing, vulnerability scanning, and security best practices.

## Table of Contents

1. [Types of Security Testing](#types-of-security-testing)
2. [Common Vulnerabilities](#common-vulnerabilities)
3. [XSS Prevention Testing](#xss-prevention-testing)
4. [SQL Injection Testing](#sql-injection-testing)
5. [Authentication Testing](#authentication-testing)
6. [Rate Limiting & DDoS Protection](#rate-limiting--ddos-protection)
7. [Security Headers](#security-headers)
8. [Penetration Testing Checklist](#penetration-testing-checklist)
9. [Automated Security Scanning](#automated-security-scanning)
10. [Security Testing in CI/CD](#security-testing-in-cicd)

## Types of Security Testing

### 1. Static Application Security Testing (SAST)
Analyzes source code for vulnerabilities without executing it.

**Tools:**
- Bandit (Python security linter)
- Semgrep
- SonarQube
- CodeQL

**What it finds:**
- Hardcoded secrets
- SQL injection patterns
- Unsafe deserialization
- Weak cryptographic algorithms

### 2. Dynamic Application Security Testing (DAST)
Tests running application from outside.

**Tools:**
- OWASP ZAP
- Burp Suite
- Nikto
- SQLMap

**What it finds:**
- XSS vulnerabilities
- CSRF issues
- Authentication bypasses
- Information disclosure

### 3. Interactive Application Security Testing (IAST)
Monitors application from inside during runtime.

**Tools:**
- Contrast Security
- Veracode
- Checkmarx

### 4. Software Composition Analysis (SCA)
Scans dependencies for known vulnerabilities.

**Tools:**
- Snyk
- OWASP Dependency-Check
- Safety (Python)
- GitHub Dependabot

### 5. Penetration Testing
Manual security testing by security experts.

## Common Vulnerabilities

### OWASP Top 10 (2021)

1. **Broken Access Control**
   - Missing access controls
   - Path traversal
   - Privilege escalation

2. **Cryptographic Failures**
   - Weak encryption
   - Hardcoded secrets
   - Insecure transmission

3. **Injection**
   - SQL injection
   - NoSQL injection
   - Command injection
   - LDAP injection

4. **Insecure Design**
   - Business logic flaws
   - Race conditions
   - Insecure workflows

5. **Security Misconfiguration**
   - Default credentials
   - Unnecessary features enabled
   - Verbose error messages

6. **Vulnerable and Outdated Components**
   - Known CVEs in dependencies
   - Unsupported software

7. **Identification and Authentication Failures**
   - Weak password policies
   - Brute force vulnerabilities
   - Session management issues

8. **Software and Data Integrity Failures**
   - Insecure deserialization
   - Unsigned updates
   - CI/CD pipeline vulnerabilities

9. **Security Logging and Monitoring Failures**
   - Insufficient logging
   - No intrusion detection
   - Missing audit trails

10. **Server-Side Request Forgery (SSRF)**
    - Unauthorized external requests
    - Internal network access

## XSS Prevention Testing

### What is XSS?

Cross-Site Scripting (XSS) allows attackers to inject malicious scripts into web pages viewed by other users.

**Types:**
- **Stored XSS**: Malicious script stored in database
- **Reflected XSS**: Script reflected in immediate response
- **DOM-based XSS**: Client-side script injection

### Testing for XSS

```python
# tests/security/test_xss_prevention.py

XSS_PAYLOADS = [
    "<script>alert('XSS')</script>",
    "<img src=x onerror=alert('XSS')>",
    "<svg onload=alert('XSS')>",
    "javascript:alert('XSS')",
]

@pytest.mark.parametrize("payload", XSS_PAYLOADS)
def test_ask_endpoint_xss_prevention(client, auth_headers, payload):
    """Test that XSS payloads are sanitized."""
    response = client.post(
        "/api/v1/ask",
        headers=auth_headers,
        json={"question": payload, "k": 5}
    )
    
    # Response should not contain raw script tags
    assert "<script>" not in response.text or "&lt;script&gt;" in response.text
```

### Prevention Techniques

1. **Input Validation**
   - Whitelist allowed characters
   - Validate data types
   - Sanitize special characters

2. **Output Encoding**
   ```python
   from html import escape
   
   safe_output = escape(user_input)
   ```

3. **Content Security Policy (CSP)**
   ```
   Content-Security-Policy: default-src 'self'; script-src 'none'
   ```

4. **HttpOnly Cookies**
   ```
   Set-Cookie: session=abc123; HttpOnly; Secure; SameSite=Strict
   ```

## SQL Injection Testing

### What is SQL Injection?

Attackers inject malicious SQL code through user input to:
- Access unauthorized data
- Modify database content
- Execute administrative operations

### Testing for SQL Injection

```python
# SQL Injection payloads
SQLI_PAYLOADS = [
    "' OR '1'='1",
    "' OR 1=1--",
    "1; DROP TABLE users--",
    "' UNION SELECT * FROM users--",
]

@pytest.mark.parametrize("payload", SQLI_PAYLOADS)
def test_sql_injection_prevention(client, payload):
    """Test SQL injection prevention."""
    response = client.get(
        "/api/v1/documents/search",
        params={"q": payload}
    )
    
    # Should not expose database errors
    assert response.status_code in [200, 400]
    
    # Response should not contain SQL keywords
    assert "sql" not in response.text.lower()
    assert "database" not in response.text.lower()
```

### Prevention Techniques

1. **Parameterized Queries (Always!)**
   ```python
   # ❌ NEVER do this
   query = f"SELECT * FROM users WHERE id = {user_id}"
   
   # ✅ ALWAYS do this
   query = "SELECT * FROM users WHERE id = :user_id"
   db.execute(query, {"user_id": user_id})
   ```

2. **ORM Usage**
   ```python
   # SQLAlchemy automatically parameterizes
   User.query.filter_by(id=user_id).first()
   ```

3. **Input Validation**
   ```python
   from pydantic import BaseModel, validator
   
   class SearchQuery(BaseModel):
       q: str
       
       @validator('q')
       def validate_query(cls, v):
           # Remove dangerous characters
           dangerous = [';', '--', '/*', '*/', 'xp_', 'sp_']
           for char in dangerous:
               if char in v:
                   raise ValueError(f"Invalid character: {char}")
           return v
   ```

4. **Least Privilege Database User**
   - Application user should not have DROP, CREATE privileges
   - Separate read-only user for reporting

## Authentication Testing

### Brute Force Protection

```python
def test_brute_force_protection(client):
    """Test that repeated failed logins trigger protection."""
    email = "test@example.com"
    
    # Attempt many failed logins
    for i in range(20):
        response = client.post("/api/v1/auth/login", json={
            "email": email,
            "password": "wrong_password"
        })
    
    # Should eventually be rate limited
    assert response.status_code == 429
```

### Implementation Strategies

1. **Rate Limiting**
   ```python
   from slowapi import Limiter
   
   limiter = Limiter(key_func=get_remote_address)
   
   @app.post("/api/v1/auth/login")
   @limiter.limit("5 per minute")
   async def login(credentials: LoginRequest):
       ...
   ```

2. **Account Lockout**
   - Lock account after N failed attempts
   - Unlock after time period or manual intervention
   - Notify user of lockout

3. **Exponential Backoff**
   - Increase delay between attempts
   - Makes brute force impractical

### Token Security

```python
def test_token_security(client):
    """Test JWT token security."""
    
    # Test 1: Tampered token should be rejected
    tampered_token = valid_token[:-5] + "XXXXX"
    response = client.get("/api/v1/documents", 
        headers={"Authorization": f"Bearer {tampered_token}"})
    assert response.status_code == 401
    
    # Test 2: Expired token should be rejected
    expired_token = create_expired_token()
    response = client.get("/api/v1/documents", 
        headers={"Authorization": f"Bearer {expired_token}"})
    assert response.status_code == 401
    
    # Test 3: 'none' algorithm should be rejected
    none_token = create_token_with_none_algorithm()
    response = client.get("/api/v1/documents", 
        headers={"Authorization": f"Bearer {none_token}"})
    assert response.status_code == 401
```

### JWT Best Practices

1. **Use Strong Algorithms**
   - ✅ RS256 (asymmetric) or HS256 (symmetric with strong secret)
   - ❌ None algorithm

2. **Short Expiration**
   - Access tokens: 15-60 minutes
   - Refresh tokens: 7-30 days

3. **Secure Storage**
   - Never store in localStorage (XSS risk)
   - Use httpOnly cookies
   - Or memory-only storage with refresh

4. **Token Claims**
   ```json
   {
     "sub": "user_id",
     "exp": 1234567890,
     "iat": 1234567800,
     "jti": "unique_token_id",  // For revocation
     "tenant": "tenant_id"      // Multi-tenant
   }
   ```

5. **Token Revocation**
   - Maintain blacklist of revoked tokens
   - Or use short expiration + refresh

## Rate Limiting & DDoS Protection

### Testing Rate Limits

```python
def test_rate_limiting(client):
    """Test that rate limits are enforced."""
    endpoint = "/api/v1/ask"
    
    # Make many rapid requests
    for i in range(150):  # Exceed typical limit
        response = client.post(endpoint, json={"question": "test"})
        
        if response.status_code == 429:
            print(f"Rate limited after {i} requests")
            break
    
    # Should have been rate limited
    assert response.status_code == 429
    
    # Should have rate limit headers
    assert "X-RateLimit-Limit" in response.headers
    assert "X-RateLimit-Remaining" in response.headers
```

### Rate Limiting Strategies

1. **Per-IP Limits**
   - Protects against anonymous attacks
   - Can block legitimate shared IPs

2. **Per-User Limits**
   - Based on authenticated user
   - Fairer distribution

3. **Per-Endpoint Limits**
   - Different limits for different endpoints
   - Heavy operations (upload) get lower limits

4. **Tiered Limits**
   - Free tier: 100 req/day
   - Pro tier: 10,000 req/day
   - Enterprise: Custom limits

### Implementation Example

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded", "retry_after": 60}
    )

@app.post("/api/v1/ask")
@limiter.limit("30 per minute")
async def ask_question(request, question: QuestionRequest):
    ...

@app.post("/api/v1/documents/upload")
@limiter.limit("10 per minute")
async def upload_document(request, file: UploadFile):
    ...
```

## Security Headers

### Essential Security Headers

```python
from fastapi import FastAPI
from fastapi.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # Prevent MIME sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'none'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:;"
        )
        
        # HSTS (HTTPS only)
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        
        return response

app = FastAPI(middleware=[Middleware(SecurityHeadersMiddleware)])
```

### Header Reference

| Header | Purpose | Recommended Value |
|--------|---------|-------------------|
| X-Frame-Options | Prevent clickjacking | DENY |
| X-Content-Type-Options | Prevent MIME sniffing | nosniff |
| X-XSS-Protection | XSS filter | 1; mode=block |
| Referrer-Policy | Control referrer info | strict-origin-when-cross-origin |
| Content-Security-Policy | Resource restrictions | default-src 'self' |
| Strict-Transport-Security | Force HTTPS | max-age=31536000 |
| Permissions-Policy | Feature restrictions | camera=(), microphone=() |

## Penetration Testing Checklist

### Pre-Engagement

- [ ] Define scope (in-scope/out-of-scope endpoints)
- [ ] Set testing windows
- [ ] Get written authorization
- [ ] Establish emergency contacts
- [ ] Define reporting format

### Reconnaissance

- [ ] Map all API endpoints
- [ ] Identify authentication mechanisms
- [ ] Document data flows
- [ ] Identify third-party integrations
- [ ] Check for exposed dev/staging environments

### Vulnerability Assessment

- [ ] **Authentication & Session Management**
  - [ ] Test password policies
  - [ ] Test account lockout
  - [ ] Test session timeout
  - [ ] Test concurrent sessions
  - [ ] Test logout functionality
  - [ ] Test token refresh mechanism

- [ ] **Authorization**
  - [ ] Test horizontal privilege escalation
  - [ ] Test vertical privilege escalation
  - [ ] Test IDOR (Insecure Direct Object Reference)
  - [ ] Test multi-tenant isolation

- [ ] **Input Validation**
  - [ ] Test XSS payloads
  - [ ] Test SQL injection
  - [ ] Test command injection
  - [ ] Test path traversal
  - [ ] Test NoSQL injection
  - [ ] Test XML/XXE injection
  - [ ] Test template injection

- [ ] **Business Logic**
  - [ ] Test workflow bypasses
  - [ ] Test race conditions
  - [ ] Test pricing manipulation
  - [ ] Test quantity manipulation

- [ ] **Cryptography**
  - [ ] Check for weak algorithms
  - [ ] Check for hardcoded secrets
  - [ ] Check certificate validity
  - [ ] Check randomness quality

- [ ] **Configuration**
  - [ ] Check for default credentials
  - [ ] Check for exposed admin interfaces
  - [ ] Check error handling (no stack traces)
  - [ ] Check security headers
  - [ ] Check CORS configuration

### Post-Testing

- [ ] Clean up test data
- [ ] Document all findings
- [ ] Provide risk ratings
- [ ] Suggest remediation steps
- [ ] Schedule retest

## Automated Security Scanning

### GitHub Actions Example

```yaml
name: Security Scan

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 3 * * 1'  # Weekly on Monday

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      # Python security scan
      - name: Bandit Security Scan
        uses: PyCQA/bandit@main
        with:
          args: "-r src/ -f json -o bandit-results.json || true"
      
      # Dependency vulnerability scan
      - name: Safety Check
        run: |
          pip install safety
          safety check --json --output safety-results.json || true
      
      # Secret scanning
      - name: GitLeaks
        uses: zricethezav/gitleaks-action@master
      
      # CodeQL Analysis
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: python
      
      - name: Autobuild
        uses: github/codeql-action/autobuild@v2
      
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2
      
      # OWASP ZAP Scan
      - name: ZAP Baseline Scan
        uses: zaproxy/action-baseline@v0.7.0
        with:
          target: 'http://localhost:8000'
          rules_file_name: '.zap/rules.tsv'
          cmd_options: '-a'
      
      # Upload results
      - name: Upload security results
        uses: actions/upload-artifact@v3
        with:
          name: security-scan-results
          path: |
            bandit-results.json
            safety-results.json
```

### Security Tools Configuration

#### Bandit Configuration (.bandit)

```yaml
exclude_dirs:
  - tests
  - venv
  - .venv

skips:
  - B101  # skip assert_used checks in test files

severity: medium
confidence: medium
```

#### Safety Ignore List (.safety-ignore)

```
# Format: CVE_ID,Reason
CVE-2023-1234,Not affected - feature not used
CVE-2023-5678,Low risk - internal only
```

## Security Testing in CI/CD

### Security Gates

```yaml
security-gates:
  - name: SAST Scan
    tool: bandit
    threshold: medium
    fail_on: high
  
  - name: Dependency Check
    tool: safety
    fail_on: critical
  
  - name: Secret Scan
    tool: gitleaks
    fail_on: any
  
  - name: DAST Scan
    tool: zap
    threshold: medium
    fail_on: high
```

### Security Test Pipeline

```yaml
stages:
  - build
  - test
  - security-scan
  - deploy

security-tests:
  stage: security-scan
  parallel:
    - sast:
        script:
          - bandit -r src/ -f json -o bandit.json
          - bandit-report-parser bandit.json --fail-on high
    
    - sca:
        script:
          - safety check --json --output safety.json
          - safety-report-parser safety.json --fail-on critical
    
    - secrets:
        script:
          - gitleaks detect --source . --verbose --redact
    
    - dast:
        script:
          - docker run -d --name app -p 8000:8000 $IMAGE
          - zap-baseline.py -t http://localhost:8000 -g gen.conf -r zap-report.html
          - docker stop app
```

## Incident Response

### Security Incident Playbook

1. **Detection**
   - Monitor security alerts
   - Review logs for anomalies
   - User reports

2. **Containment**
   - Isolate affected systems
   - Block malicious IPs
   - Revoke compromised tokens

3. **Eradication**
   - Remove malicious code
   - Patch vulnerabilities
   - Update dependencies

4. **Recovery**
   - Restore from clean backups
   - Re-enable services
   - Monitor for recurrence

5. **Lessons Learned**
   - Document incident
   - Update security measures
   - Train team on new threats

## Security Testing Best Practices

### 1. Shift Left
- Integrate security testing early in development
- Fix vulnerabilities before they reach production

### 2. Defense in Depth
- Multiple layers of security
- Don't rely on single control

### 3. Principle of Least Privilege
- Grant minimum necessary permissions
- Regular access reviews

### 4. Regular Testing
- Run security tests on every build
- Perform penetration testing quarterly
- Stay updated on new vulnerabilities

### 5. Secure by Default
- Secure configurations out of the box
- Require explicit action to reduce security

### 6. Fail Securely
- Default to deny
- Handle errors without information disclosure

## Conclusion

Security testing is not a one-time activity but an ongoing process. Regular testing, combined with secure development practices and continuous monitoring, helps maintain a strong security posture.

### Quick Start Checklist

- [ ] Enable security scanning in CI/CD
- [ ] Implement rate limiting
- [ ] Add security headers
- [ ] Test authentication mechanisms
- [ ] Validate all inputs
- [ ] Review dependencies monthly
- [ ] Conduct penetration testing
- [ ] Set up security monitoring
- [ ] Create incident response plan
- [ ] Train team on security awareness

### Additional Resources

- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
