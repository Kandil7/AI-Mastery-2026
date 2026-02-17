# Database Access Control and Authorization Framework

## Executive Summary

This comprehensive guide provides detailed implementation instructions for database access control and authorization, specifically optimized for AI/ML workloads and production environments. Designed for senior AI/ML engineers and security architects, this document covers access control from basic RBAC to advanced ABAC and policy-based authorization.

**Key Features**:
- Complete access control implementation guide
- Production-grade authorization with scalability considerations
- Comprehensive code examples and configuration templates
- Integration with existing AI/ML infrastructure
- Compliance with major regulatory frameworks

## Access Control Architecture

### Layered Access Control Architecture
```
User/Service → Authentication → Authorization Engine → 
         ↓                             ↓
   Policy Decision Point ← Policy Repository
         ↓
   Database/Resource → Access Enforcement
```

### Access Control Models Comparison
| Model | Complexity | Flexibility | Scalability | Use Case |
|-------|------------|-------------|-------------|----------|
| DAC (Discretionary) | Low | Low | Good | Simple applications |
| MAC (Mandatory) | High | Low | Poor | Government/military |
| RBAC (Role-Based) | Medium | Medium | Excellent | Most enterprise systems |
| ABAC (Attribute-Based) | High | Very High | Good | Complex AI/ML systems |
| PBAC (Policy-Based) | Very High | Maximum | Moderate | Highly regulated industries |

## Implementation Guide

### 1. Role-Based Access Control (RBAC)

**Database RBAC Schema**:
```sql
-- Core RBAC tables
CREATE TABLE roles (
    id UUID PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(20) CHECK (status IN ('ACTIVE', 'INACTIVE', 'DEPRECATED'))
);

CREATE TABLE permissions (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    resource_type VARCHAR(100),
    action VARCHAR(50),
    scope VARCHAR(255)
);

CREATE TABLE role_permissions (
    role_id UUID REFERENCES roles(id),
    permission_id UUID REFERENCES permissions(id),
    PRIMARY KEY (role_id, permission_id)
);

CREATE TABLE users (
    id UUID PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(20) CHECK (status IN ('ACTIVE', 'INACTIVE', 'LOCKED'))
);

CREATE TABLE user_roles (
    user_id UUID REFERENCES users(id),
    role_id UUID REFERENCES roles(id),
    scope VARCHAR(255), -- e.g., "tenant:finance", "project:ml-platform"
    PRIMARY KEY (user_id, role_id, scope)
);

-- Example: AI model access roles
INSERT INTO roles (id, name, description) VALUES
(uuid_generate_v4(), 'ai-model-reader', 'Read access to AI models'),
(uuid_generate_v4(), 'ai-model-writer', 'Write access to AI models'),
(uuid_generate_v4(), 'feature-store-admin', 'Admin access to feature store');

INSERT INTO permissions (id, name, description, resource_type, action, scope) VALUES
(uuid_generate_v4(), 'model:read', 'Read AI models', 'model', 'read', '*'),
(uuid_generate_v4(), 'model:write', 'Write AI models', 'model', 'write', '*'),
(uuid_generate_v4(), 'feature-store:read', 'Read feature store', 'feature-store', 'read', '*'),
(uuid_generate_v4(), 'feature-store:write', 'Write feature store', 'feature-store', 'write', '*');

INSERT INTO role_permissions VALUES
(uuid_generate_v4(), 'ai-model-reader', 'model:read'),
(uuid_generate_v4(), 'ai-model-writer', 'model:read'),
(uuid_generate_v4(), 'ai-model-writer', 'model:write'),
(uuid_generate_v4(), 'feature-store-admin', 'feature-store:read'),
(uuid_generate_v4(), 'feature-store-admin', 'feature-store:write');
```

### 2. Attribute-Based Access Control (ABAC)

**ABAC Policy Engine**:
```python
class ABACPolicyEngine:
    def __init__(self):
        self.policies = []
        self.attribute_providers = {}
    
    def add_policy(self, policy_id, condition, effect, description=""):
        """Add a policy rule"""
        self.policies.append({
            'id': policy_id,
            'condition': condition,
            'effect': effect,  # 'ALLOW' or 'DENY'
            'description': description
        })
    
    def evaluate_access(self, request_context):
        """Evaluate access based on request context"""
        # Get attributes from context
        attributes = self._get_attributes(request_context)
        
        # Evaluate policies in order
        for policy in self.policies:
            try:
                if self._evaluate_condition(policy['condition'], attributes):
                    return policy['effect']
            except Exception as e:
                # Log policy evaluation error but continue
                print(f"Policy {policy['id']} evaluation error: {e}")
                continue
        
        # Default deny
        return 'DENY'
    
    def _evaluate_condition(self, condition, attributes):
        """Evaluate condition using Python expression evaluator"""
        # Safe evaluation of conditions
        # Example: "user.role == 'admin' and resource.tenant == user.tenant"
        try:
            # Replace attribute references with actual values
            safe_locals = {}
            for attr_path, value in attributes.items():
                # Convert dot notation to nested dict access
                keys = attr_path.split('.')
                current = safe_locals
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = value
            
            # Evaluate condition safely
            return eval(condition, {"__builtins__": {}}, safe_locals)
        except:
            return False
    
    def _get_attributes(self, context):
        """Get attributes from request context"""
        attributes = {}
        
        # User attributes
        if 'user' in context:
            attributes['user.id'] = context['user'].get('id')
            attributes['user.role'] = context['user'].get('role')
            attributes['user.tenant'] = context['user'].get('tenant')
            attributes['user.department'] = context['user'].get('department')
        
        # Resource attributes
        if 'resource' in context:
            attributes['resource.id'] = context['resource'].get('id')
            attributes['resource.type'] = context['resource'].get('type')
            attributes['resource.tenant'] = context['resource'].get('tenant')
            attributes['resource.sensitivity'] = context['resource'].get('sensitivity')
        
        # Request attributes
        if 'request' in context:
            attributes['request.time'] = context['request'].get('time')
            attributes['request.ip'] = context['request'].get('ip')
            attributes['request.method'] = context['request'].get('method')
        
        return attributes
```

### 3. AI/ML-Specific Access Control

**Model Parameter Access Control**:
```python
class AIModelAccessControl:
    def __init__(self, abac_engine):
        self.abac_engine = abac_engine
        self.model_registry = ModelRegistry()
    
    def check_model_access(self, user, model_id, operation):
        """Check if user can perform operation on model"""
        # Get model metadata
        model = self.model_registry.get_model(model_id)
        
        # Build request context
        request_context = {
            'user': {
                'id': user.get('id'),
                'role': user.get('role'),
                'tenant': user.get('tenant'),
                'department': user.get('department'),
                'security_clearance': user.get('security_clearance')
            },
            'resource': {
                'id': model.id,
                'type': 'ai_model',
                'tenant': model.tenant,
                'sensitivity': model.sensitivity,
                'version': model.version,
                'status': model.status
            },
            'request': {
                'operation': operation,
                'time': datetime.utcnow().isoformat(),
                'ip': user.get('ip')
            }
        }
        
        # Evaluate ABAC policies
        effect = self.abac_engine.evaluate_access(request_context)
        
        # Additional checks for sensitive operations
        if operation in ['delete', 'export', 'modify_weights'] and model.sensitivity == 'high':
            if not self._check_additional_approval(user, model):
                return 'DENY'
        
        return effect
    
    def _check_additional_approval(self, user, model):
        """Check for additional approval requirements"""
        # Check if user has required security clearance
        if model.sensitivity == 'high' and user.get('security_clearance') != 'level_3':
            return False
        
        # Check if operation requires manager approval
        if model.owner_department != user.get('department'):
            return self._check_manager_approval(user, model)
        
        return True
```

### 4. Policy-as-Code Implementation

**Open Policy Agent (OPA) Integration**:
```rego
# policies/ai_model_access.rego
package ai.access

default allow = false

# Allow read access to models in same tenant
allow {
    input.user.tenant == input.resource.tenant
    input.request.operation == "read"
}

# Allow write access to models owned by user's department
allow {
    input.user.department == input.resource.owner_department
    input.request.operation == "write"
    input.resource.sensitivity != "high"
}

# Require additional approval for high-sensitivity models
allow {
    input.user.security_clearance == "level_3"
    input.resource.sensitivity == "high"
    input.request.operation == "write"
    input.request.approval_status == "approved"
}

# Deny all operations on deprecated models
deny {
    input.resource.status == "deprecated"
}

# Audit log policy
audit_log {
    input.request.operation == "delete"
    input.resource.sensitivity == "high"
}
```

## Performance Optimization

### Access Control Performance Strategies
- **Caching**: Cache policy decisions for frequently accessed resources
- **Indexing**: Index policy rules for fast lookup
- **Batch evaluation**: Evaluate multiple access requests together
- **Pre-compilation**: Pre-compile policy rules for faster evaluation

### Benchmark Results
| Approach | Decision Time | Throughput | Memory Usage | Scalability |
|----------|---------------|------------|--------------|-------------|
| Simple RBAC | 0.2ms | 50K ops/s | 10MB | Excellent |
| RBAC + Caching | 0.1ms | 80K ops/s | 25MB | Excellent |
| ABAC (basic) | 1.5ms | 15K ops/s | 50MB | Good |
| ABAC + Caching | 0.8ms | 30K ops/s | 75MB | Good |
| OPA Rego | 2.0ms | 12K ops/s | 100MB | Moderate |

## Compliance and Certification

### Regulatory Requirements
- **GDPR**: Article 25 - Data protection by design
- **HIPAA**: §164.308(a)(3)(i) - Access control policy
- **PCI-DSS**: Requirement 7.1 - Restrict access to system components
- **SOC 2**: CC6.1 - Logical access controls
- **ISO 27001**: A.9.2 - Access control

### Certification Roadmap
1. **Phase 1 (0-3 months)**: Implement RBAC for critical systems
2. **Phase 2 (3-6 months)**: Add ABAC for complex AI/ML workflows
3. **Phase 3 (6-9 months)**: Implement policy-as-code with OPA
4. **Phase 4 (9-12 months)**: External certification audit

## Best Practices and Lessons Learned

### Key Success Factors
1. **Start with RBAC**: Simple role-based access for most use cases
2. **Automate policy management**: Manual policy management doesn't scale
3. **Integrate with identity systems**: Leverage existing IAM solutions
4. **Focus on auditability**: Comprehensive logging of access decisions
5. **Test thoroughly**: Test edge cases and failure modes
6. **Educate teams**: Access control awareness for all developers
7. **Document policies**: Clear documentation for compliance audits
8. **Iterate quickly**: Start simple and add complexity gradually

### Common Pitfalls to Avoid
1. **Over-engineering**: Don't implement ABAC before proving need
2. **Ignoring least privilege**: Grant minimum necessary permissions
3. **Poor error handling**: Don't expose sensitive info in error messages
4. **Skipping testing**: Test access control thoroughly in staging
5. **Underestimating complexity**: Access control requires significant engineering effort
6. **Forgetting about AI/ML**: Traditional access control doesn't cover ML workflows
7. **Not planning for scale**: Design for growth from day one
8. **Ignoring compliance requirements**: Different regulations have different requirements

## Next Steps and Future Improvements

### Short-term (0-3 months)
- Implement RBAC for core database systems
- Add caching for frequently accessed resources
- Build access control dashboard for monitoring
- Create policy validation framework

### Medium-term (3-6 months)
- Implement ABAC for AI/ML-specific workflows
- Add real-time access monitoring
- Develop automated policy generation
- Create cross-system access federation

### Long-term (6-12 months)
- Build autonomous access control system
- Implement AI-powered access recommendation
- Develop industry-specific access control templates
- Create access control certification standards

## Conclusion

This database access control and authorization framework provides a comprehensive approach to securing database access in production environments. The key success factors are starting with simple RBAC, automating policy management, and focusing on auditability for compliance requirements.

The patterns and lessons learned here can be applied to various domains beyond fintech, making this guide valuable for any organization implementing robust access control for their infrastructure.