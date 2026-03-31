# Module 4: Privacy-Preserving RAG

## 📋 Module Overview

**Duration:** 2-3 weeks (15-20 hours)  
**Difficulty:** Advanced  
**Prerequisites:** Module 1-3 completion, Basic cryptography knowledge

This module teaches you to build RAG systems that protect sensitive data through encryption, access control, anonymization, and privacy-preserving retrieval techniques.

---

## 🎯 Learning Objectives

### Remember
- Define PII (Personally Identifiable Information) types
- Identify privacy regulations (GDPR, CCPA, HIPAA)
- Recall encryption methods for vector data

### Understand
- Explain privacy risks in RAG systems
- Describe differential privacy concepts
- Summarize access control models

### Apply
- Implement PII detection and redaction
- Build attribute-based access control
- Create encrypted vector indexes

### Analyze
- Compare privacy-preserving techniques
- Diagnose privacy leakage risks
- Evaluate access control policies

### Evaluate
- Assess compliance with regulations
- Critique anonymization effectiveness
- Judge privacy-utility trade-offs

### Create
- Design privacy-preserving RAG architectures
- Develop custom PII detection rules
- Build audit logging systems

---

## 🔒 Privacy Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              PRIVACY-PRESERVING RAG ARCHITECTURE            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Documents ──▶ [Privacy Preprocessing]                      │
│                  │                                          │
│                  ├──▶ PII Detection                         │
│                  ├──▶ Data Classification                   │
│                  └──▶ Anonymization/Tokenization            │
│                                                             │
│  Query ──▶ [Query Privacy]                                  │
│              │                                              │
│              ├──▶ Query sanitization                        │
│              ├──▶ Intent validation                         │
│              └──▶ Access control check                      │
│                                                             │
│              ▼                                              │
│         [Secure Retrieval]                                  │
│              │                                              │
│              ├──▶ Encrypted vector search                   │
│              ├──▶ Attribute-based filtering                 │
│              └──▶ Result redaction                          │
│                                                             │
│              ▼                                              │
│         [Privacy-Aware Generation]                          │
│              │                                              │
│              ├──▶ PII filtering in context                  │
│              ├──▶ Output validation                         │
│              └──▶ Audit logging                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Techniques

### PII Detection and Redaction

```python
import re
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class PIIEntity:
    text: str
    pii_type: str
    start: int
    end: int
    confidence: float

class PIIDetector:
    PATTERNS = {
        'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'PHONE': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
        'CREDIT_CARD': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
        'IP_ADDRESS': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    }
    
    def detect(self, text: str) -> List[PIIEntity]:
        entities = []
        for pii_type, pattern in self.PATTERNS.items():
            for match in re.finditer(pattern, text):
                entities.append(PIIEntity(
                    text=match.group(),
                    pii_type=pii_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95
                ))
        return entities
    
    def redact(self, text: str, replacement: str = '[REDACTED]') -> str:
        entities = self.detect(text)
        result = text
        for entity in sorted(entities, key=lambda e: e.start, reverse=True):
            result = result[:entity.start] + replacement + result[entity.end:]
        return result
```

### Attribute-Based Access Control

```python
from typing import Dict, List, Set

class AccessControlPolicy:
    def __init__(self):
        self.policies = []
    
    def add_policy(self, resource_tags: Set[str], 
                   required_attributes: Set[str],
                   action: str = 'read'):
        self.policies.append({
            'resource_tags': resource_tags,
            'required_attributes': required_attributes,
            'action': action
        })
    
    def check_access(self, user_attributes: Set[str],
                     resource_tags: Set[str],
                     action: str = 'read') -> bool:
        for policy in self.policies:
            if policy['action'] != action:
                continue
            if policy['resource_tags'].issubset(resource_tags):
                if policy['required_attributes'].issubset(user_attributes):
                    return True
        return False

# Example usage
policy = AccessControlPolicy()
policy.add_policy(
    resource_tags={'confidential', 'hr'},
    required_attributes={'department:hr', 'clearance:high'}
)

user_attrs = {'department:hr', 'clearance:high', 'role:manager'}
resource_tags = {'confidential', 'hr'}

has_access = policy.check_access(user_attrs, resource_tags)
```

### Encrypted Vector Search

```python
from cryptography.fernet import Fernet
import numpy as np

class EncryptedVectorIndex:
    def __init__(self, encryption_key: bytes = None):
        if encryption_key is None:
            encryption_key = Fernet.generate_key()
        self.cipher = Fernet(encryption_key)
        self.encrypted_vectors = {}
        self.metadata = {}
    
    def add(self, doc_id: str, vector: np.ndarray, 
            metadata: dict, sensitivity_level: str):
        # Encrypt vector
        vector_bytes = vector.tobytes()
        encrypted = self.cipher.encrypt(vector_bytes)
        
        self.encrypted_vectors[doc_id] = encrypted
        self.metadata[doc_id] = {
            **metadata,
            'sensitivity_level': sensitivity_level
        }
    
    def search(self, query_vector: np.ndarray, 
               user_clearance: str) -> List[dict]:
        # Filter by clearance first
        accessible_docs = [
            doc_id for doc_id, meta in self.metadata.items()
            if self._check_clearance(meta['sensitivity_level'], user_clearance)
        ]
        
        # Decrypt and search (simplified - production uses homomorphic encryption)
        results = []
        for doc_id in accessible_docs:
            encrypted = self.encrypted_vectors[doc_id]
            decrypted = self.cipher.decrypt(encrypted)
            vector = np.frombuffer(decrypted, dtype=np.float32)
            
            similarity = np.dot(query_vector, vector)
            results.append({
                'id': doc_id,
                'score': float(similarity),
                'metadata': self.metadata[doc_id]
            })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def _check_clearance(self, doc_level: str, user_clearance: str) -> bool:
        levels = {'public': 0, 'internal': 1, 'confidential': 2, 'secret': 3}
        return levels.get(user_clearance, 0) >= levels.get(doc_level, 0)
```

---

## 📚 Module Structure

```
module_4_privacy_preserving/
├── README.md
├── theory/
│   ├── 01_privacy_fundamentals.md
│   ├── 02_pii_detection.md
│   ├── 03_access_control.md
│   ├── 04_encryption_techniques.md
│   └── 05_compliance_frameworks.md
├── labs/
│   ├── lab_1_pii_redaction/
│   ├── lab_2_access_control/
│   └── lab_3_encrypted_search/
├── knowledge_checks/
├── coding_challenges/
├── solutions/
└── further_reading.md
```

---

## Compliance Considerations

| Regulation | Key Requirements | RAG Implications |
|------------|------------------|------------------|
| GDPR | Right to erasure, data minimization | Must delete user data on request |
| CCPA | Right to know, opt-out | Track data sources and usage |
| HIPAA | PHI protection, audit logs | Encrypt health data, log access |
| SOC 2 | Security controls, monitoring | Implement access controls, logging |

---

*Last Updated: March 30, 2026*
