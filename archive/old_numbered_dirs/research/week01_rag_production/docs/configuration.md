# Configuration

This section provides comprehensive information about configuring the Production RAG System, including environment variables, configuration files, and runtime settings.

## Configuration Architecture

The Production RAG System uses a hierarchical configuration system based on Pydantic settings. The configuration follows the principle of "convention over configuration" with sensible defaults while allowing for extensive customization.

### Configuration Hierarchy

The system follows this priority order for configuration values:

1. **Environment Variables** (Highest Priority)
2. **.env File**
3. **Default Values** (Lowest Priority)

### Configuration Structure

The configuration is organized into logical sections:

```python
# Main configuration classes
RAGConfig
├── DatabaseConfig
├── ModelConfig
├── RetrievalConfig
├── APIConfig
├── LoggingConfig
└── SecurityConfig
```

## Environment Variables

### Database Configuration
- `DATABASE__URL`: Database connection URL
  - Default: `mongodb://localhost:27017`
  - Example: `mongodb://username:password@host:port/database`

- `DATABASE__NAME`: Database name
  - Default: `minirag`
  - Example: `rag_db`

- `DATABASE__USERNAME`: Database username
  - Optional
  - Example: `rag_user`

- `DATABASE__PASSWORD`: Database password
  - Optional (marked as secret)
  - Example: `secure_password`

- `DATABASE__POOL_SIZE`: Connection pool size
  - Default: `10`
  - Range: 1-100

- `DATABASE__MAX_OVERFLOW`: Maximum overflow connections
  - Default: `20`
  - Range: 0-100

- `DATABASE__ECHO`: Enable SQL query logging
  - Default: `false`
  - Values: `true`/`false`

### Model Configuration
- `MODELS__GENERATOR_MODEL`: Text generation model name
  - Default: `gpt2`
  - Example: `gpt2-medium`

- `MODELS__DENSE_MODEL`: Dense embedding model name
  - Default: `all-MiniLM-L6-v2`
  - Example: `all-mpnet-base-v2`

- `MODELS__SPARSE_MODEL`: Sparse embedding model name
  - Default: `bm25`
  - Example: `bm25`

- `MODELS__MAX_NEW_TOKENS`: Maximum tokens for generation
  - Default: `300`
  - Range: 1-2048

- `MODELS__TEMPERATURE`: Generation temperature for diversity
  - Default: `0.7`
  - Range: 0.0-2.0

- `MODELS__TOP_P`: Top-p sampling parameter
  - Default: `0.9`
  - Range: 0.0-1.0

- `MODELS__TOP_K`: Top-k sampling parameter
  - Default: `5`
  - Range: 1-20

### Retrieval Configuration
- `RETRIEVAL__ALPHA`: Weight for dense retrieval in hybrid fusion
  - Default: `0.5`
  - Range: 0.0-1.0

- `RETRIEVAL__FUSION_METHOD`: Fusion strategy
  - Default: `rrf`
  - Values: `rrf`, `weighted`, `densite`, `combsum`, `combmnz`

- `RETRIEVAL__RRF_K`: Smoothing constant for RRF calculation
  - Default: `60`

- `RETRIEVAL__SPARSE_K1`: BM25 k1 parameter
  - Default: `1.5`

- `RETRIEVAL__SPARSE_B`: BM25 b parameter
  - Default: `0.75`

- `RETRIEVAL__MAX_CANDIDATES`: Maximum candidates for reranking
  - Default: `50`

### API Configuration
- `API__HOST`: Host address for the API server
  - Default: `0.0.0.0`
  - Example: `localhost`

- `API__PORT`: Port number for the API server
  - Default: `8000`
  - Range: 1-65535

- `API__CORS_ORIGINS`: Allowed origins for CORS
  - Default: `["*"]`
  - Example: `["http://localhost:3000", "https://example.com"]`

- `API__RATE_LIMIT_REQUESTS`: Max requests per minute
  - Default: `100`

- `API__RATE_LIMIT_WINDOW`: Time window for rate limiting in seconds
  - Default: `60`

- `API__REQUEST_TIMEOUT`: Request timeout in seconds
  - Default: `30`

### Logging Configuration
- `LOGGING__LEVEL`: Logging level
  - Default: `INFO`
  - Values: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

- `LOGGING__FORMAT`: Log format string
  - Default: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

- `LOGGING__FILE_PATH`: Path to log file
  - Optional
  - Example: `/var/log/rag-system.log`

- `LOGGING__MAX_BYTES`: Maximum log file size in bytes
  - Default: `10485760` (10MB)

- `LOGGING__BACKUP_COUNT`: Number of backup log files
  - Default: `5`

### Security Configuration
- `SECURITY__SECRET_KEY`: Secret key for cryptographic operations
  - Default: `secret` (marked as secret)
  - Example: `your-very-secure-secret-key-here`

- `SECURITY__JWT_ALGORITHM`: Algorithm for JWT encoding
  - Default: `HS256`
  - Values: `HS256`, `RS256`, etc.

- `SECURITY__ACCESS_TOKEN_EXPIRE_MINUTES`: Access token expiration in minutes
  - Default: `30`

- `SECURITY__ENABLE_AUTHENTICATION`: Enable authentication middleware
  - Default: `false`
  - Values: `true`/`false`

- `SECURITY__ALLOWED_HOSTS`: Allowed hosts for security headers
  - Default: `["localhost", "127.0.0.1"]`
  - Example: `["localhost", "127.0.0.1", "example.com"]`

### Application Configuration
- `APP_NAME`: Name of the application
  - Default: `Production RAG API`
  - Example: `My RAG System`

- `APP_VERSION`: Version of the application
  - Default: `1.0.0`
  - Example: `2.1.0`

- `ENVIRONMENT`: Application environment
  - Default: `development`
  - Values: `development`, `testing`, `staging`, `production`

- `DEBUG`: Enable debug mode
  - Default: `false`
  - Values: `true`/`false`

- `OPENAI_API_KEY`: OpenAI API key
  - Optional (marked as secret)
  - Example: `sk-your-openai-api-key`

- `HUGGINGFACE_TOKEN`: Hugging Face token
  - Optional (marked as secret)
  - Example: `hf-your-huggingface-token`

## Configuration File (.env)

Create a `.env` file in the project root to define environment variables:

```env
# Application Settings
ENVIRONMENT=production
DEBUG=false
APP_NAME=My Production RAG System

# Database Settings
DATABASE__URL=mongodb://username:password@host:27017
DATABASE__NAME=my_rag_db
DATABASE__POOL_SIZE=20
DATABASE__MAX_OVERFLOW=40

# Model Settings
MODELS__GENERATOR_MODEL=gpt2-medium
MODELS__DENSE_MODEL=all-mpnet-base-v2
MODELS__MAX_NEW_TOKENS=500

# Retrieval Settings
RETRIEVAL__ALPHA=0.7
RETRIEVAL__FUSION_METHOD=rrf

# API Settings
API__HOST=0.0.0.0
API__PORT=8000
API__RATE_LIMIT_REQUESTS=1000
API__REQUEST_TIMEOUT=60

# Security Settings
SECURITY__SECRET_KEY=your-very-secure-secret-key-here
SECURITY__ENABLE_AUTHENTICATION=true

# Logging Settings
LOGGING__LEVEL=INFO
LOGGING__FILE_PATH=/var/log/rag-system.log
```

## Programmatic Configuration

### Accessing Configuration

Configuration values can be accessed programmatically through the `settings` instance:

```python
from src.config import settings

# Access database configuration
db_url = settings.database.url
db_name = settings.database.name

# Access model configuration
generator_model = settings.models.generator_model
dense_model = settings.models.dense_model

# Access retrieval configuration
alpha = settings.retrieval.alpha
fusion_method = settings.retrieval.fusion_method

# Access API configuration
api_host = settings.api.host
api_port = settings.api.port

# Check environment
if settings.is_development():
    print("Running in development mode")
elif settings.is_production():
    print("Running in production mode")
```

### Configuration Validation

The system includes comprehensive validation for all configuration values:

```python
from src.config import settings

try:
    # This will raise validation errors if values are invalid
    validated_config = settings.validate()
except ValidationError as e:
    print(f"Configuration validation error: {e}")
```

## Configuration Profiles

### Development Profile
```env
ENVIRONMENT=development
DEBUG=true
DATABASE__URL=mongodb://localhost:27017
DATABASE__NAME=rag_dev
LOGGING__LEVEL=DEBUG
API__RATE_LIMIT_REQUESTS=10000
```

### Staging Profile
```env
ENVIRONMENT=staging
DEBUG=false
DATABASE__URL=mongodb://staging-db:27017
DATABASE__NAME=rag_staging
LOGGING__LEVEL=INFO
API__RATE_LIMIT_REQUESTS=1000
SECURITY__ENABLE_AUTHENTICATION=true
```

### Production Profile
```env
ENVIRONMENT=production
DEBUG=false
DATABASE__URL=mongodb://prod-db:27017
DATABASE__NAME=rag_prod
DATABASE__POOL_SIZE=20
DATABASE__MAX_OVERFLOW=40
LOGGING__LEVEL=INFO
API__RATE_LIMIT_REQUESTS=1000
API__REQUEST_TIMEOUT=30
SECURITY__ENABLE_AUTHENTICATION=true
SECURITY__SECRET_KEY=production-secret-key
```

## Configuration Management Best Practices

### 1. Environment-Specific Configuration
Use different configuration profiles for different environments to ensure appropriate settings for each stage of the deployment pipeline.

### 2. Secure Handling of Sensitive Information
- Never commit sensitive values to version control
- Use environment variables for secrets
- Mark sensitive fields with `Field(secret=True)`
- Implement proper access controls for configuration files

### 3. Validation and Error Handling
- Implement comprehensive validation for all configuration values
- Provide clear error messages for invalid configurations
- Fail fast if critical configuration values are missing or invalid

### 4. Documentation and Comments
- Document all configuration options with examples
- Provide default values and acceptable ranges
- Include comments explaining the purpose of each setting

### 5. Testing Configuration Changes
- Test configuration changes in lower environments before production
- Validate that configuration changes don't break existing functionality
- Monitor system behavior after configuration changes

## Configuration Examples

### Minimal Configuration
```env
ENVIRONMENT=development
DATABASE__URL=mongodb://localhost:27017
```

### Full Configuration
```env
# Application
APP_NAME=Production RAG System
APP_VERSION=1.0.0
ENVIRONMENT=production
DEBUG=false

# Database
DATABASE__URL=mongodb://rag_user:password@mongo-cluster:27017
DATABASE__NAME=rag_production
DATABASE__POOL_SIZE=25
DATABASE__MAX_OVERFLOW=50
DATABASE__ECHO=false

# Models
MODELS__GENERATOR_MODEL=gpt2-medium
MODELS__DENSE_MODEL=all-mpnet-base-v2
MODELS__SPARSE_MODEL=bm25
MODELS__MAX_NEW_TOKENS=500
MODELS__TEMPERATURE=0.7
MODELS__TOP_P=0.9
MODELS__TOP_K=5

# Retrieval
RETRIEVAL__ALPHA=0.6
RETRIEVAL__FUSION_METHOD=rrf
RETRIEVAL__RRF_K=60
RETRIEVAL__SPARSE_K1=1.5
RETRIEVAL__SPARSE_B=0.75
RETRIEVAL__MAX_CANDIDATES=100

# API
API__HOST=0.0.0.0
API__PORT=8000
API__CORS_ORIGINS=["https://myapp.com"]
API__RATE_LIMIT_REQUESTS=1000
API__RATE_LIMIT_WINDOW=60
API__REQUEST_TIMEOUT=45

# Logging
LOGGING__LEVEL=INFO
LOGGING__FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
LOGGING__FILE_PATH=/var/log/rag-system.log
LOGGING__MAX_BYTES=20971520
LOGGING__BACKUP_COUNT=10

# Security
SECURITY__SECRET_KEY=super-secret-production-key
SECURITY__JWT_ALGORITHM=HS256
SECURITY__ACCESS_TOKEN_EXPIRE_MINUTES=60
SECURITY__ENABLE_AUTHENTICATION=true
SECURITY__ALLOWED_HOSTS=["myapp.com", "api.myapp.com"]
```

## Configuration Validation

The system includes built-in validation for all configuration values:

### Validation Rules
- Numeric values are validated for appropriate ranges
- String values are validated for acceptable formats
- URLs are validated for proper format
- Enum values are validated against allowed options
- Required values are checked for presence

### Custom Validators
Custom validators can be added for specific validation requirements:

```python
from pydantic import validator

class CustomConfig(BaseModel):
    custom_value: str
    
    @validator('custom_value')
    def validate_custom_value(cls, v):
        if not v.startswith('prefix_'):
            raise ValueError('Value must start with "prefix_"')
        return v
```

## Configuration Updates

### Runtime Configuration Updates
Some configuration values can be updated at runtime without restarting the application:

- Logging levels
- Rate limiting parameters
- Feature flags
- Performance tuning parameters

### Configuration Reload
The system supports configuration reloading:

```python
from src.config import settings

# Reload configuration from environment variables
settings._reload()

# Or create a new configuration instance
new_settings = RAGConfig()
```

This configuration system provides a flexible, secure, and maintainable approach to managing the Production RAG System's settings across different environments and deployment scenarios.