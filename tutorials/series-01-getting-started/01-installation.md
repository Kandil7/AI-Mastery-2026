# 🚀 Tutorial 1: Installation & Setup

**Get your AI development environment up and running in 30 minutes**

---

## 🎯 What You'll Accomplish

By the end of this tutorial, you will have:

- ✅ Python 3.10+ installed and verified
- ✅ VS Code configured for AI development
- ✅ Git installed and configured
- ✅ AI-Mastery-2026 repository cloned
- ✅ Dependencies installed
- ✅ First Python script running

**Time Required:** 30 minutes  
**Difficulty:** ⭐☆☆☆☆ (Beginner)

---

## 📋 Prerequisites

- Computer with 8GB+ RAM (16GB recommended)
- 10GB+ free disk space
- Internet connection
- Administrator access

---

## 🛠️ Step 1: Install Python (10 minutes)

### Windows

1. **Download Python:**
   - Visit: https://www.python.org/downloads/
   - Click "Download Python 3.12.x" (latest version)

2. **Run Installer:**
   - ✅ **IMPORTANT:** Check "Add Python to PATH"
   - Click "Install Now"
   - Wait for installation to complete

3. **Verify Installation:**
   ```cmd
   python --version
   ```
   Expected output: `Python 3.12.x`

   ```cmd
   pip --version
   ```
   Expected output: `pip 24.x.x from ...`

### macOS

1. **Using Homebrew (Recommended):**
   ```bash
   # Install Homebrew if not installed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install Python
   brew install python
   ```

2. **Verify Installation:**
   ```bash
   python3 --version
   pip3 --version
   ```

### Linux (Ubuntu/Debian)

1. **Install Python:**
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   ```

2. **Verify Installation:**
   ```bash
   python3 --version
   pip3 --version
   ```

---

## 🛠️ Step 2: Install VS Code (5 minutes)

### Download & Install

1. **Download VS Code:**
   - Visit: https://code.visualstudio.com/
   - Download for your OS
   - Run installer with default settings

2. **Install Python Extension:**
   - Open VS Code
   - Press `Ctrl+Shift+X` (or `Cmd+Shift+X` on Mac)
   - Search for "Python"
   - Click "Install" on the Microsoft Python extension

3. **Install Recommended Extensions:**
   - **Pylance** - Python language server
   - **Jupyter** - Notebook support
   - **GitLens** - Git integration
   - **Error Lens** - Inline error highlighting

### Configure VS Code

1. **Open Settings:**
   - Press `Ctrl+,` (or `Cmd+,` on Mac)

2. **Set Python Interpreter:**
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P`)
   - Type "Python: Select Interpreter"
   - Choose Python 3.10+ from the list

3. **Enable Auto-Save:**
   - File → Auto Save (check it)

---

## 🛠️ Step 3: Install Git (5 minutes)

### Windows

1. **Download Git:**
   - Visit: https://git-scm.com/download/win
   - Run installer
   - Use default settings (recommended)

2. **Verify Installation:**
   ```cmd
   git --version
   ```
   Expected: `git version 2.x.x`

### macOS

```bash
# Git usually comes pre-installed, verify:
git --version

# If not installed, use Homebrew:
brew install git
```

### Linux

```bash
sudo apt install git
git --version
```

### Configure Git

```bash
# Set your name
git config --global user.name "Your Name"

# Set your email
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

---

## 🛠️ Step 4: Clone AI-Mastery-2026 Repository (5 minutes)

### Open Terminal/Command Prompt

**Windows:**
- Press `Win+R`, type `cmd`, press Enter

**macOS/Linux:**
- Open Terminal app

### Clone Repository

```bash
# Navigate to your projects folder
cd Documents  # or wherever you keep projects

# Clone the repository
git clone https://github.com/Kandil7/AI-Mastery-2026.git

# Enter the project directory
cd AI-Mastery-2026
```

### Verify Clone

```bash
# List files
ls  # or 'dir' on Windows

# You should see:
# src/, tests/, docs/, notebooks/, pyproject.toml, etc.
```

---

## 🛠️ Step 5: Create Virtual Environment (5 minutes)

### Why Virtual Environments?

Virtual environments isolate project dependencies, preventing conflicts between projects.

### Create & Activate

**Windows (Command Prompt):**
```cmd
# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate

# You should see (venv) at the start of your prompt
```

**Windows (PowerShell):**
```powershell
# Create
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# If you get an error about execution policy:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS/Linux:**
```bash
# Create
python3 -m venv venv

# Activate
source venv/bin/activate

# You should see (venv) at the start of your prompt
```

### Verify Activation

```bash
# Check Python path (should point to venv)
which python  # or 'where python' on Windows

# Check pip path
which pip  # or 'where pip' on Windows
```

---

## 🛠️ Step 6: Install Dependencies (5 minutes)

### Install Project Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install project dependencies
pip install -e ".[dev]"

# This will install:
# - Core: numpy, pandas, scikit-learn, fastapi, pydantic
# - Dev: pytest, black, isort, mypy, etc.
# - ML: torch, transformers (optional)
```

### Verify Installation

```bash
# Test core packages
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
```

Expected output:
```
NumPy: 1.26.x
Pandas: 2.2.x
Scikit-learn: 1.4.x
FastAPI: 0.110.x
```

---

## 🛠️ Step 7: Run Your First Python Script (5 minutes)

### Create Test Script

1. **Open VS Code:**
   ```bash
   # From project directory
   code .
   ```

2. **Create New File:**
   - Press `Ctrl+N` (or `Cmd+N`)
   - Save as `test_setup.py`

3. **Add Code:**
   ```python
   """Test script to verify setup."""
   
   import sys
   import numpy as np
   import pandas as pd
   from sklearn.datasets import make_classification
   
   print("=" * 60)
   print("AI-MASTERY-2026 SETUP VERIFICATION")
   print("=" * 60)
   
   # Python version
   print(f"\n✅ Python Version: {sys.version}")
   
   # NumPy test
   arr = np.array([1, 2, 3, 4, 5])
   print(f"✅ NumPy Working: {arr}")
   print(f"   NumPy Version: {np.__version__}")
   
   # Pandas test
   df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
   print(f"\n✅ Pandas Working:")
   print(df)
   print(f"   Pandas Version: {pd.__version__}")
   
   # Scikit-learn test
   X, y = make_classification(n_samples=100, n_features=10, random_state=42)
   print(f"\n✅ Scikit-learn Working")
   print(f"   Generated dataset: {X.shape}")
   print(f"   Scikit-learn Version: {sklearn.__version__}")
   
   print("\n" + "=" * 60)
   print("🎉 ALL CHECKS PASSED! Setup is complete!")
   print("=" * 60)
   ```

4. **Run Script:**
   - Press `Ctrl+F5` (or `Cmd+F5`)
   - Or in terminal: `python test_setup.py`

### Expected Output

```
============================================================
AI-MASTERY-2026 SETUP VERIFICATION
============================================================

✅ Python Version: 3.12.x
✅ NumPy Working: [1 2 3 4 5]
   NumPy Version: 1.26.x

✅ Pandas Working:
   A  B
0  1  4
1  2  5
2  3  6
   Pandas Version: 2.2.x

✅ Scikit-learn Working
   Generated dataset: (100, 10)
   Scikit-learn Version: 1.4.x

============================================================
🎉 ALL CHECKS PASSED! Setup is complete!
============================================================
```

---

## ✅ Verification Checklist

Before proceeding, verify:

- [ ] Python 3.10+ installed and in PATH
- [ ] VS Code installed with Python extension
- [ ] Git installed and configured
- [ ] Repository cloned successfully
- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] Test script runs without errors

---

## 🐛 Troubleshooting

### Issue: "python is not recognized"

**Solution:**
- Windows: Reinstall Python, make sure to check "Add to PATH"
- Or manually add to PATH:
  1. Search "Environment Variables" in Windows
  2. Edit PATH variable
  3. Add: `C:\Users\YourName\AppData\Local\Programs\Python\Python312\`

### Issue: "Permission denied" when activating venv

**Solution (Windows PowerShell):**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
# Make sure venv is activated
# Then reinstall dependencies
pip install -e ".[dev]"
```

### Issue: Slow installation

**Solution:**
- Check internet connection
- Try using a mirror:
  ```bash
  pip install -e ".[dev]" -i https://pypi.org/simple/
  ```

---

## 🎓 Knowledge Check

**Q1.** What command creates a virtual environment?

A) `python create venv`  
B) `python -m venv venv`  
C) `pip install venv`  
D) `conda create venv`  

**Answer:** B) `python -m venv venv`

---

**Q2.** Why do we use virtual environments?

A) To make Python faster  
B) To isolate project dependencies  
C) To install Python  
D) To run Jupyter notebooks  

**Answer:** B) To isolate project dependencies

---

**Q3.** What does `pip install -e ".[dev]"` do?

A) Installs only dev dependencies  
B) Installs project in editable mode with dev dependencies  
C) Updates pip  
D) Creates a virtual environment  

**Answer:** B) Installs project in editable mode with dev dependencies

---

## 🚀 Next Steps

Now that your environment is set up:

1. **Continue to Tutorial 2:** [Your First ML Model](02-first-ml-model.md)
2. **Explore the Repository:** Browse the `src/` and `notebooks/` folders
3. **Join Discord:** Get help and connect with other learners

---

## 📞 Get Help

- **Stuck?** Join Discord `#tutorial-help`
- **Found a bug?** Open GitHub issue
- **Questions?** Email: support@ai-mastery-2026.dev

---

**Tutorial Created:** April 2, 2026  
**Last Updated:** April 2, 2026  
**Estimated Time:** 30 minutes  
**Difficulty:** Beginner

---

[← Back to Tutorials](../README.md) | [Next: Your First ML Model](02-first-ml-model.md) | [Get Help](https://discord.gg/aimastery2026)
