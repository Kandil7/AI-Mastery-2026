# Database Encryption Patterns for AI/ML Systems

## Overview

In production AI/ML systems, data encryption is critical for protecting sensitive training data, model parameters, and inference results. This document covers advanced encryption patterns specifically tailored for AI/ML workloads.

## Encryption at Rest

### Traditional Approaches
- **Full Disk Encryption (FDE)**: Encrypts entire storage volumes (e.g., LUKS, BitLocker)
- **Filesystem Encryption**: Encrypts individual files or directories
- **Database-Level Encryption**: Built-in features like TDE (Transparent Data Encryption)

### AI/ML Specific Considerations
- **Model Parameter Encryption**: Encrypt model weights and biases separately from metadata
- **Training Data Encryption**: Encrypt raw datasets while maintaining ability to perform preprocessing
- **Checkpoint Encryption**: Secure model checkpoints during distributed training

### Implementation Examples
```sql
-- PostgreSQL with pgcrypto for field-level encryption
CREATE TABLE model_parameters (
    id UUID PRIMARY KEY,
    model_name TEXT,
    encrypted_weights BYTEA,
    encryption_key_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Using AES-256-GCM for symmetric encryption of sensitive fields
INSERT INTO model_parameters (model_name, encrypted_weights, encryption_key_id)
VALUES ('resnet50',
        pgp_sym_encrypt('{"weights": [0.1, 0.2, ...]}', 'encryption_key_123', 'compress-algo=1,cipher-algo=aes256'),
        'key_123');
```

### Performance Benchmarks
| Encryption Method | Throughput Impact | Latency Increase | Security Level |
|-------------------|-------------------|------------------|----------------|
| FDE (AES-256)     | <5%               | <1ms             | High           |
| Column Encryption | 15-25%            | 2-5ms            | Very High      |
| Homomorphic       | 100-500x          | 10-100ms         | Maximum        |

## Encryption in Transit

### TLS/SSL Best Practices
- Use TLS 1.3 with strong cipher suites (ECDHE-RSA-AES256-GCM-SHA384)
- Implement certificate pinning for internal services
- Rotate certificates automatically using ACME protocol

### AI/ML Specific Patterns
- **Secure Model Serving**: Encrypt model inference requests/responses
- **Distributed Training Channels**: Encrypt parameter updates between workers
- **Feature Store Access**: Secure connections between feature computation and serving layers

### Implementation Example
```python
# Secure connection for distributed training
import ssl
import torch.distributed as dist

context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
context.load_cert_chain(certfile="server.crt", keyfile="server.key")
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE

# Configure PyTorch distributed with secure channels
dist.init_process_group(
    backend='nccl',
    init_method=f'tcp://{master_addr}:{master_port}',
    world_size=world_size,
    rank=rank,
    timeout=datetime.timedelta(seconds=30),
    ssl_context=context
)
```

## Field-Level Encryption

### Advanced Techniques
- **Deterministic Encryption**: For searchable encrypted fields (e.g., tenant IDs)
- **Order-Preserving Encryption**: For range queries on encrypted numeric data
- **Homomorphic Encryption**: For computations on encrypted data (experimental for AI/ML)

### AI/ML Use Cases
- **Privacy-Preserving Federated Learning**: Encrypt local model updates
- **Secure Multi-Party Computation**: Collaborative model training without sharing raw data
- **Encrypted Feature Stores**: Serve encrypted features to models while maintaining privacy

### Implementation Pattern
```sql
-- PostgreSQL with deterministic encryption for searchable fields
CREATE OR REPLACE FUNCTION encrypt_deterministic(text, text)
RETURNS bytea AS $$
BEGIN
    RETURN pgp_sym_encrypt($1, $2, 'cipher-algo=aes256,s2k-digest-algo=sha512');
END;
$$ LANGUAGE plpgsql;

-- Index on encrypted field for efficient searching
CREATE INDEX idx_encrypted_tenant ON user_data (encrypt_deterministic(tenant_id, 'tenant_key'));
```

## Hybrid Encryption Strategies

### Layered Approach
1. **Key Management**: Use HSMs or cloud KMS for master keys
2. **Data Encryption Keys**: Generate per-dataset or per-model DEKs
3. **Field Encryption**: Apply different algorithms based on sensitivity

### AI/ML Specific Implementation
```python
class AIEncryptionManager:
    def __init__(self, kms_client):
        self.kms = kms_client
        self.cache = {}

    def encrypt_model_parameters(self, model_params, model_id):
        """Encrypt model parameters with hierarchical key management"""
        # Get or create dataset-specific key
        dataset_key = self._get_or_create_dataset_key(model_id)

        # Encrypt with dataset key
        encrypted_params = self._encrypt_with_key(model_params, dataset_key)

        # Wrap dataset key with master key
        wrapped_key = self.kms.wrap_key(dataset_key, self.master_key_id)

        return {
            'encrypted_params': encrypted_params,
            'wrapped_key': wrapped_key,
            'encryption_metadata': {
                'algorithm': 'AES-256-GCM',
                'timestamp': datetime.utcnow().isoformat(),
                'model_id': model_id
            }
        }
```

## Security Considerations for AI/ML Workloads

### Threat Modeling
- **Model Stealing**: Protect model architecture and parameters
- **Data Poisoning**: Ensure integrity of training data
- **Inference Attacks**: Prevent extraction of training data from model outputs

### Best Practices
- **Key Rotation**: Rotate encryption keys quarterly or after security incidents
- **Audit Logging**: Log all encryption/decryption operations
- **Access Control**: Combine encryption with RBAC for defense in depth
- **Hardware Acceleration**: Use GPU-accelerated encryption for performance-critical paths

## Real-World Examples

### Healthcare AI System
- Patient data encrypted at rest with FDE + column encryption for PHI
- Model parameters encrypted with HSM-managed keys
- Inference requests encrypted with mutual TLS
- Audit logs stored in separate encrypted database

### Financial Fraud Detection
- Transaction data encrypted with field-level encryption
- Model weights protected with homomorphic encryption for collaborative training
- Real-time feature computation with encrypted intermediate results
- Zero-trust architecture with continuous authentication

## Trade-offs and Recommendations

| Pattern | When to Use | Limitations | Recommendation |
|---------|-------------|-------------|----------------|
| Full Disk Encryption | General purpose, compliance requirements | No field-level protection | Always implement as baseline |
| Column Encryption | Sensitive fields, regulatory compliance | Query limitations, performance impact | Use for PII, financial data |
| Homomorphic Encryption | Privacy-preserving ML, collaborative training | High computational cost | Experimental, use for specific use cases |
| Deterministic Encryption | Searchable encrypted fields, indexing | Vulnerable to frequency analysis | Use with salted hashes for better security |

## References
- NIST SP 800-57: Recommendation for Key Management
- OWASP Top 10 for AI/ML Systems
- GDPR Article 32: Security of Processing
- HIPAA Security Rule: Technical Safeguards
- AWS KMS Best Practices for ML Workloads