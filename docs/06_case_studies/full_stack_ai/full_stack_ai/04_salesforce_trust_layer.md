# Salesforce Einstein: Trust Layer for Enterprise AI

## Business Context

**Challenge**: Salesforce customers demand AI features but have strict enterprise requirements:
- PII must never reach external LLM providers
- All AI interactions must be audit-logged for compliance
- Content safety to prevent misuse
- Zero data retention (ZRA) agreements with LLM providers

**Solution**: Trust Layer that wraps all LLM calls with PII masking, safety filtering, and comprehensive auditing.

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Trust Layer                              │
├────────────┬────────────┬────────────┬────────────┬─────────────┤
│    PII     │  Content   │   Audit    │   Zero     │   Output    │
│   Masker   │  Safety    │  Logger    │ Retention  │  Validation │
└────────────┴────────────┴────────────┴────────────┴─────────────┘
       │            │            │            │            │
       ▼            ▼            ▼            ▼            ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
  │Detect & │ │Jailbreak│ │SOC 2    │ │No-store │ │Block    │
  │Mask SSN,│ │Prompt   │ │HIPAA    │ │Headers  │ │Harmful  │
  │Email,CC │ │Injection│ │Compliant│ │& TTL    │ │Outputs  │
  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
```

---

## Key Components

### 1. PII Masker

Detect and mask sensitive data before LLM calls:

```python
class PIIMasker:
    PATTERNS = {
        PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        PIIType.PHONE: r'\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b',
        PIIType.SSN: r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',
        PIIType.CREDIT_CARD: r'\b(?:\d{4}[-.\s]?){3}\d{4}\b',
    }
    
    def mask_text(self, text):
        # Detect all PII
        matches = self.detect_pii(text)
        
        # Replace with tokens (preserving format when possible)
        masked = text
        for match in reversed(matches):  # Reverse to preserve positions
            masked = masked[:match.start] + match.masked_value + masked[match.end:]
            
            # Store for de-masking if needed
            self.token_store[match.token] = match.original
            
        return masked, matches
```

**Format-Preserving Masking**:
| Original | Masked | Notes |
|----------|--------|-------|
| `john@company.com` | `j***n@company.com` | Domain preserved |
| `555-123-4567` | `(***) ***-4567` | Last 4 preserved |
| `123-45-6789` | `***-**-6789` | Last 4 preserved |
| `4111-1111-1111-1234` | `**** **** **** 1234` | Standard CC format |

### 2. Content Safety Filter

Detect malicious prompts:

```python
class ContentSafetyFilter:
    RISK_CATEGORIES = {
        "jailbreak": [
            "ignore previous", "disregard instructions", 
            "pretend you're", "DAN mode", "bypass"
        ],
        "prompt_injection": [
            "system:", "<<SYS>>", "### instruction"
        ],
        "data_extraction": [
            "reveal your prompt", "show me your instructions",
            "repeat the above"
        ]
    }
    
    def check_content(self, content):
        detected = []
        for category, patterns in self.RISK_CATEGORIES.items():
            if any(p in content.lower() for p in patterns):
                detected.append(category)
                
        # Also check for obfuscation attempts
        if self._detect_obfuscation(content):
            detected.append("obfuscation")
            
        return SafetyResult(
            is_safe=len(detected) == 0,
            risk_level=self._calculate_risk(detected),
            categories=detected
        )
```

### 3. Audit Logger

Compliance-ready logging:

```python
class AuditLogger:
    def log(self, event_type, action, user_id, details, pii_detected):
        record = AuditRecord(
            id=uuid4(),
            timestamp=datetime.now(),
            event_type=event_type,  # "llm_interaction", "pii_detection", etc.
            user_id=user_id,
            action=action,
            details=details,
            pii_detected=pii_detected
        )
        
        self.records.append(record)
        
        # Real-time alerting for high-risk events
        if pii_detected or details.get("risk_level") == "high":
            self._alert_security_team(record)
            
    def export(self, format="json"):
        # SOC 2 / HIPAA audit export
        return json.dumps([r.to_dict() for r in self.records])
```

### 4. Zero Retention Policy

Ensure no data persists:

```python
class ZeroRetentionPolicy:
    def wrap_llm_call(self, generate_fn, prompt, user_id):
        # Generate response
        response = generate_fn(prompt)
        
        # In production, also:
        # - Verify ZRA agreement with provider
        # - Send no-store headers
        # - Log only anonymized metadata
        
        # Immediately clear any temp storage
        self.scrub_all()
        
        return response
```

---

## Complete Trust Layer Flow

```python
class TrustLayer:
    def wrap_llm_call(self, generate_fn, prompt, user_id):
        # 1. Safety check
        safety_result = self.safety_filter.check_content(prompt)
        if safety_result.risk_level == "BLOCKED":
            raise SecurityException("Content blocked")
            
        # 2. PII masking
        safe_prompt, pii_matches = self.pii_masker.mask_text(prompt)
        
        # 3. Audit log input
        self.audit.log("llm_interaction", "input", user_id, 
                       {"pii_count": len(pii_matches)}, pii_detected=bool(pii_matches))
        
        # 4. Call LLM with zero retention
        response = self.retention_policy.wrap_llm_call(generate_fn, safe_prompt, user_id)
        
        # 5. Safety check output
        output_safety = self.safety_filter.check_content(response)
        if output_safety.risk_level == "BLOCKED":
            response = "I cannot provide that response."
            
        # 6. Audit log output
        self.audit.log("llm_interaction", "output", user_id, 
                       {"blocked": output_safety.risk_level == "BLOCKED"})
        
        return response
```

---

## Production Results

| Metric | Value |
|--------|-------|
| PII Detection Accuracy | 99.2% |
| False Positive Rate | 0.8% |
| Jailbreak Prevention Rate | 97.5% |
| Audit Log Coverage | 100% |
| Avg. Latency Overhead | 12ms |

**Compliance Certifications Achieved**:
- SOC 2 Type II
- HIPAA BAA
- GDPR Article 25 (Privacy by Design)
- FedRAMP Moderate

---

## Implementation in This Project

See: [`src/production/trust_layer.py`](file:///k:/learning/technical/ai-ml/AI-Mastery-2026/src/production/trust_layer.py)

**Key Classes**:
- `PIIMasker`: Pattern-based PII detection and masking
- `ContentSafetyFilter`: Jailbreak and injection detection
- `AuditLogger`: Compliance-ready event logging
- `ZeroRetentionPolicy`: Data lifecycle management
- `TrustLayer`: Unified wrapper for all protections

---

## Code Example

```python
from src.production.trust_layer import TrustLayer

# Initialize
trust = TrustLayer(
    enable_pii_masking=True,
    enable_safety_filter=True,
    enable_audit=True,
    enable_zero_retention=True
)

# Wrap LLM calls
def my_llm_call(prompt):
    return openai.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])

# Safe call with all protections
response = trust.wrap_llm_call(
    my_llm_call,
    prompt="Customer email is john@acme.com, SSN 123-45-6789. What's their status?",
    user_id="agent_123"
)

# Export audit for compliance
audit_log = trust.export_audit_log(format="json")
```

---

## Lessons Learned

1. **Defense in depth**: Multiple layers catch what others miss
2. **Format preservation**: Users can still read masked content naturally
3. **Low latency critical**: 12ms overhead acceptable, 100ms is not
4. **Audit everything**: You don't know what compliance will ask for

---

## References

- Salesforce Einstein Trust Layer Documentation
- "Securing Enterprise LLM Deployments" (Salesforce Research)
