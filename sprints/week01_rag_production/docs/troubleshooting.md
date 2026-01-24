# Troubleshooting

This section provides solutions to common issues and troubleshooting steps for the Production RAG System.

## Common Issues and Solutions

### 1. Startup Issues

#### Issue: Application fails to start
**Symptoms**: 
- Error during application startup
- Import errors
- Missing dependencies

**Solutions**:
1. **Check dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Python version**:
   ```bash
   python --version
   # Should be Python 3.10 or higher
   ```

3. **Check for missing modules**:
   ```bash
   python -c "import src.config"
   # If this fails, check your PYTHONPATH
   ```

4. **Verify environment variables**:
   ```bash
   echo $DATABASE__URL
   # Should return your database URL
   ```

#### Issue: Database connection errors
**Symptoms**:
- "Connection refused" errors
- "Authentication failed" errors
- "Database not found" errors

**Solutions**:
1. **Check if MongoDB is running**:
   ```bash
   # Using Docker
   docker ps | grep mongo
   
   # Using system service
   sudo systemctl status mongod
   ```

2. **Verify connection parameters**:
   ```bash
   # Check environment variables
   echo $DATABASE__URL
   echo $DATABASE__NAME
   ```

3. **Test connection manually**:
   ```bash
   # Using MongoDB client
   mongosh mongodb://localhost:27017
   ```

4. **Check MongoDB logs**:
   ```bash
   # Docker logs
   docker logs rag-mongo
   
   # System logs
   sudo tail -f /var/log/mongodb/mongod.log
   ```

### 2. Performance Issues

#### Issue: Slow query responses
**Symptoms**:
- Query response times > 5 seconds
- High memory usage
- High CPU usage

**Solutions**:
1. **Check system resources**:
   ```bash
   # Monitor system resources
   htop
   # or
   top
   ```

2. **Optimize retrieval parameters**:
   ```python
   # Reduce top_k value
   result = rag_pipeline.query("query", top_k=3)  # Instead of top_k=10
   ```

3. **Check vector database performance**:
   - Ensure proper indexing
   - Consider using more efficient vector databases (FAISS vs ChromaDB)
   - Monitor vector database logs

4. **Implement caching**:
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=128)
   def cached_query(query: str):
       return rag_pipeline.query(query)
   ```

#### Issue: High memory usage
**Symptoms**:
- Memory usage > 80% of available RAM
- Out of memory errors
- Slow performance

**Solutions**:
1. **Reduce model size**:
   ```python
   # Use smaller models
   config = RAGConfig(
       dense_model="all-MiniLM-L6-v2"  # Smaller model
   )
   ```

2. **Implement batch processing**:
   ```python
   # Process documents in smaller batches
   BATCH_SIZE = 10  # Instead of processing all at once
   ```

3. **Monitor memory usage**:
   ```python
   import psutil
   
   def monitor_memory():
       process = psutil.Process()
       memory_info = process.memory_info()
       print(f"RSS: {memory_info.rss / 1024 / 1024:.2f} MB")
   ```

### 3. Configuration Issues

#### Issue: Environment variables not loaded
**Symptoms**:
- Default configuration values used instead of environment values
- Wrong database connection
- Incorrect model selection

**Solutions**:
1. **Verify .env file**:
   ```bash
   # Check if .env file exists and has correct format
   cat .env
   ```

2. **Check environment variable format**:
   ```bash
   # Should use double underscores for nested settings
   export DATABASE__URL="mongodb://localhost:27017"
   ```

3. **Restart application after changes**:
   ```bash
   # Environment variables are loaded at startup
   # Need to restart application after changes
   ```

#### Issue: Model loading failures
**Symptoms**:
- "Model not found" errors
- Long startup times
- Download failures

**Solutions**:
1. **Check model names**:
   ```python
   # Verify model names are correct
   config = RAGConfig(
       generator_model="gpt2",  # Verify this model exists
       dense_model="all-MiniLM-L6-v2"  # Verify this model exists
   )
   ```

2. **Check internet connectivity**:
   ```bash
   # Models need to be downloaded on first use
   ping google.com
   ```

3. **Clear model cache**:
   ```bash
   # Clear Hugging Face cache
   rm -rf ~/.cache/huggingface/
   ```

### 4. API Issues

#### Issue: API endpoints return 404
**Symptoms**:
- All endpoints return 404
- API documentation not accessible

**Solutions**:
1. **Check if server is running**:
   ```bash
   # Verify server is listening on correct port
   netstat -tuln | grep 8000
   ```

2. **Check FastAPI routes**:
   ```python
   # Verify routes are properly defined
   from api import app
   print(app.routes)
   ```

3. **Restart the server**:
   ```bash
   # Kill any existing processes
   pkill -f "uvicorn"
   # Restart server
   uvicorn api:app --reload --port 8000
   ```

#### Issue: Query endpoint returns errors
**Symptoms**:
- 500 errors on query endpoint
- "RAG model not initialized" errors
- Validation errors

**Solutions**:
1. **Check if RAG pipeline is initialized**:
   ```python
   # Verify in the application logs
   # Look for "RAG model initialized" message
   ```

2. **Verify request format**:
   ```bash
   # Check request body format
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "test query", "k": 3}'
   ```

3. **Check validation errors**:
   ```python
   # Look for validation error messages in logs
   # Ensure query length is between 1-500 characters
   # Ensure k is between 1-20
   ```

### 5. Docker Issues

#### Issue: Docker containers fail to start
**Symptoms**:
- Containers exit immediately
- Port binding errors
- Volume mounting errors

**Solutions**:
1. **Check Docker logs**:
   ```bash
   docker-compose logs
   docker-compose logs rag-api
   docker-compose logs mongo
   ```

2. **Check port availability**:
   ```bash
   # Check if ports are already in use
   netstat -tuln | grep 8000
   netstat -tuln | grep 27017
   ```

3. **Verify volume permissions**:
   ```bash
   # Check if volumes have correct permissions
   ls -la ./uploads
   ls -la ./data
   ```

4. **Clean Docker system**:
   ```bash
   # Remove unused containers, networks, images
   docker system prune
   ```

#### Issue: Database connection in Docker
**Symptoms**:
- Connection timeouts in Docker
- Cannot connect to MongoDB from API container

**Solutions**:
1. **Check Docker network**:
   ```bash
   # Verify containers are on the same network
   docker-compose exec rag-api ping mongo
   ```

2. **Verify database URL**:
   ```yaml
   # In docker-compose.yml, should use service name
   DATABASE__URL=mongodb://rag_user:rag_password@mongo:27017
   ```

3. **Check MongoDB initialization**:
   ```bash
   # Verify MongoDB is ready before API starts
   docker-compose logs mongo
   ```

## Diagnostic Commands

### System Diagnostics
```bash
# Check Python version
python --version

# Check installed packages
pip list

# Check environment variables
env | grep RAG

# Check disk space
df -h

# Check memory usage
free -h
```

### Application Diagnostics
```bash
# Check application logs
tail -f logs/app.log

# Monitor application performance
python -m cProfile -o profile.stats api.py

# Check for memory leaks
python -m tracemalloc api.py
```

### Network Diagnostics
```bash
# Check if ports are open
netstat -tuln | grep :8000

# Test database connectivity
telnet localhost 27017

# Check DNS resolution
nslookup localhost
```

## Debugging Tips

### 1. Enable Debug Mode
Set `DEBUG=true` in your environment to get more detailed error messages:

```bash
export DEBUG=true
uvicorn api:app --reload --port 8000
```

### 2. Add Logging
Add detailed logging to troubleshoot specific issues:

```python
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add logging to specific functions
def problematic_function():
    logger.debug("Entering function")
    # Your code here
    logger.debug("Exiting function")
```

### 3. Use Development Mode
Run in development mode with auto-reload for faster debugging:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### 4. Check Configuration
Verify your configuration is loaded correctly:

```python
from src.config import settings
print(settings.dict())  # Print all configuration values
```

## Monitoring and Health Checks

### Application Health
Regularly check the health endpoint:
```bash
curl http://localhost:8000/health
```

### Resource Monitoring
Monitor system resources:
```bash
# CPU and memory usage
htop

# Disk usage
du -sh ./data/

# Network usage
iftop
```

### Log Monitoring
Monitor application logs for errors:
```bash
# Tail application logs
tail -f logs/app.log

# Search for errors
grep -i error logs/app.log

# Monitor in real-time
tail -f logs/app.log | grep -i error
```

## When to Seek Help

Contact support or team members when experiencing:

1. **Persistent startup failures** after trying all troubleshooting steps
2. **Data corruption issues** that affect production data
3. **Security vulnerabilities** or suspected breaches
4. **Performance degradation** that impacts users significantly
5. **Configuration issues** that affect multiple environments

## Prevention Strategies

### 1. Regular Backups
- Backup database regularly
- Version control configuration files
- Backup model caches

### 2. Monitoring Setup
- Set up alerts for critical errors
- Monitor resource usage
- Track performance metrics

### 3. Testing
- Run tests before deployment
- Test configuration changes in staging
- Monitor for regressions

### 4. Documentation
- Keep configuration documentation updated
- Document troubleshooting steps
- Maintain runbooks for common issues