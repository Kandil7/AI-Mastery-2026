# Troubleshooting Guide

Common issues and solutions for AI-Mastery-2026.

---

## Installation Issues

### Issue: `pip install -e ".[dev]"` fails

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement...
```

**Solutions:**

1. **Upgrade pip:**
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Check Python version:**
   ```bash
   python --version  # Must be 3.10+
   ```

3. **Install system dependencies (Linux):**
   ```bash
   sudo apt-get install python3-dev build-essential
   ```

4. **Use conda/mamba:**
   ```bash
   conda create -n ai-mastery python=3.11
   conda activate ai-mastery
   pip install -e ".[dev]"
   ```

---

### Issue: Import errors after installation

**Symptoms:**
```
ImportError: cannot import name 'X' from 'src.module'
```

**Solutions:**

1. **Ensure you're in the project directory:**
   ```bash
   cd AI-Mastery-2026
   ```

2. **Reinstall in editable mode:**
   ```bash
   pip install -e .
   ```

3. **Clear Python cache:**
   ```bash
   find . -type d -name __pycache__ -exec rm -rf {} +
   find . -name "*.pyc" -delete
   ```

4. **Check PYTHONPATH:**
   ```bash
   # Should NOT include src/ in PYTHONPATH
   echo $PYTHONPATH
   ```

---

## Module-Specific Issues

### LLM Module

**Issue:** `from src.llm import Transformer` fails

**Solution:** Check if transformers library is installed:
```bash
pip install transformers torch
```

---

### RAG Module

**Issue:** Vector store initialization fails

**Solution:** Install required vector database:
```bash
# For FAISS
pip install faiss-cpu

# For Qdrant
pip install qdrant-client
```

---

### Production Module

**Issue:** API doesn't start

**Solution:**
1. Check if port is available:
   ```bash
   netstat -ano | findstr :8000  # Windows
   lsof -i :8000  # Linux/Mac
   ```

2. Use different port:
   ```bash
   uvicorn src.production.api:app --port 8001
   ```

---

## Performance Issues

### Issue: Slow model training

**Solutions:**

1. **Use GPU if available:**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```

2. **Reduce batch size:**
   ```python
   from src.config import TrainingConfig
   config = TrainingConfig(batch_size=8)  # Reduce from default 32
   ```

3. **Enable mixed precision:**
   ```python
   from torch.cuda.amp import autocast
   ```

---

### Issue: High memory usage

**Solutions:**

1. **Reduce model size:**
   ```python
   from src.config import TransformerConfig
   config = TransformerConfig(hidden_dim=256, num_layers=4)
   ```

2. **Use gradient accumulation:**
   ```python
   # Accumulate gradients over multiple batches
   accumulation_steps = 4
   ```

3. **Clear cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

---

## Testing Issues

### Issue: Tests fail with `ModuleNotFoundError`

**Solution:**
```bash
# Install test dependencies
pip install -e ".[dev]"

# Run from project root
pytest tests/ -v
```

---

### Issue: Tests timeout

**Solution:**
```bash
# Increase timeout
pytest tests/ --timeout=600

# Skip slow tests
pytest tests/ -m "not slow"
```

---

## Development Issues

### Issue: Pre-commit hooks fail

**Solution:**
```bash
# Update pre-commit
pre-commit autoupdate

# Run hooks manually
pre-commit run --all-files

# Skip hooks (not recommended)
git commit -m "message" --no-verify
```

---

### Issue: Black formatting conflicts

**Solution:**
```bash
# Format all code
black src/ tests/

# Check configuration
black --version
cat pyproject.toml | grep -A 10 "\[tool.black\]"
```

---

## Environment Issues

### Issue: Virtual environment problems

**Solution:**
```bash
# Recreate virtual environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -e ".[dev]"
```

---

### Issue: Docker build fails

**Solution:**
```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t ai-mastery .
```

---

## Getting Help

If you can't find a solution here:

1. **Search existing issues:** [GitHub Issues](https://github.com/Kandil7/AI-Mastery-2026/issues)
2. **Create a new issue:** Use the bug report template
3. **Check discussions:** [GitHub Discussions](https://github.com/Kandil7/AI-Mastery-2026/discussions)
4. **Contact:** medokandeal7@gmail.com

---

## Debug Mode

Enable debug mode for more detailed error messages:

```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
python your_script.py
```

Or in Python:
```python
from src.config import get_settings, Settings, Environment

settings = Settings(
    environment=Environment.DEVELOPMENT,
    debug=True,
    log_level="DEBUG"
)
```

---

**Last Updated:** March 31, 2026
