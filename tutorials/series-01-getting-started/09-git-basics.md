# 🔧 Tutorial 9: Git for AI Projects

**Master version control for machine learning in 45 minutes**

---

## 🎯 What You'll Build

By the end of this tutorial, you will have:

- ✅ Installed and configured Git
- ✅ Created your first repository
- ✅ Committed and tracked changes
- ✅ Created branches for experiments
- ✅ Pushed to GitHub
- ✅ Used .gitignore for ML projects
- ✅ Collaborated with others

**Time Required:** 45 minutes  
**Difficulty:** ⭐⭐☆☆☆ (Beginner)  
**Prerequisites:** Tutorial 1 (Installation), command line basics

---

## 📋 What You'll Learn

- What is Git and why it matters
- Version control fundamentals
- Core Git commands
- Branching and merging
- GitHub collaboration
- Best practices for ML projects
- Using .gitignore effectively

---

## 🧠 Step 1: Understand Git (5 minutes)

### What is Git?

**Git = Version Control System**

Git tracks changes to your code over time, allowing you to:
- **Save snapshots** of your work
- **Revert** to previous versions
- **Experiment** without breaking things
- **Collaborate** with others

### Why Git Matters for AI/ML

```
Without Git:
- "final_model_v1.py"
- "final_model_v2_REAL.py"
- "final_model_v3_ACTUAL_FINAL.py" 😱

With Git:
- Clean history of all changes
- Easy to revert bad changes
- Collaborate without conflicts
```

### Key Concepts

| Term | Definition |
|------|------------|
| **Repository** | Project folder tracked by Git |
| **Commit** | Snapshot of changes |
| **Branch** | Parallel version of code |
| **Merge** | Combine branches |
| **Push** | Send to remote (GitHub) |
| **Pull** | Get from remote |
| **Clone** | Copy repository |

---

## 🛠️ Step 2: Setup Git (5 minutes)

### Install Git

```bash
# Check if Git is installed
git --version

# If not installed:
# Windows: Download from https://git-scm.com/
# macOS: brew install git
# Linux: sudo apt install git
```

### Configure Git

```bash
# Set your identity
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Set default editor
git config --global core.editor "code --wait"  # VS Code

# Verify configuration
git config --list
```

### Create GitHub Account

1. Visit: https://github.com
2. Sign up (free)
3. Verify email
4. You're ready!

---

## 📂 Step 3: Create Your First Repository (10 minutes)

### Initialize Repository

```bash
# Create project folder
mkdir my-first-ml-project
cd my-first-ml-project

# Initialize Git
git init

# You should see:
# Initialized empty Git repository in ...
```

### Create First File

```python
# Create file: train_model.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("🤖 My First ML Project")
print("=" * 40)

# Sample data
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)

# Train model
model = LogisticRegression()
model.fit(X, y)

# Evaluate
predictions = model.predict(X)
accuracy = accuracy_score(y, predictions)

print(f"✅ Model trained!")
print(f"   Accuracy: {accuracy:.2%}")
```

### Check Status

```bash
# See what Git is tracking
git status

# You should see:
# Untracked files:
#   train_model.py
```

### Add and Commit

```bash
# Stage the file
git add train_model.py

# Commit with message
git commit -m "feat: add initial model training script"

# You should see:
# [main (root-commit) abc1234] feat: add initial model training script
#  1 file changed, 20 insertions(+)
#  create mode 100644 train_model.py
```

### View History

```bash
# See commit history
git log --oneline

# You should see:
# abc1234 (HEAD -> main) feat: add initial model training script
```

---

## 🔄 Step 4: Make Changes and Commit (5 minutes)

### Modify File

```python
# Update train_model.py to add visualization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

print("🤖 My First ML Project")
print("=" * 40)

# Generate better data
np.random.seed(42)
X = np.random.randn(200, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Train model
model = LogisticRegression()
model.fit(X, y)

# Evaluate
predictions = model.predict(X)
accuracy = accuracy_score(y, predictions)

print(f"✅ Model trained!")
print(f"   Accuracy: {accuracy:.2%}")

# Visualize confusion matrix
cm = confusion_matrix(y, predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=300)
print("📊 Confusion matrix saved!")
```

### Commit Changes

```bash
# Check what changed
git status
git diff train_model.py

# Stage changes
git add train_model.py

# Commit
git commit -m "feat: add visualization and improve data generation"

# View history
git log --oneline

# You should see:
# def5678 (HEAD -> main) feat: add visualization and improve data generation
# abc1234 feat: add initial model training script
```

---

## 🌿 Step 5: Branching and Experimenting (10 minutes)

### Why Branch?

**Branches let you experiment without breaking main code!**

```
main (stable)
  ├── experiment-random-forest
  ├── experiment-neural-network
  └── experiment-svm
```

### Create Branch

```bash
# Create and switch to new branch
git checkout -b experiment-random-forest

# You should see:
# Switched to a new branch 'experiment-random-forest'
```

### Make Changes on Branch

```python
# Update train_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

print("🌲 Random Forest Experiment")
print("=" * 40)

# Generate data
np.random.seed(42)
X = np.random.randn(500, 10)
y = (X[:, 0] * X[:, 1] > 0).astype(int)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Evaluate
accuracy = accuracy_score(y, model.predict(X))
print(f"✅ Random Forest trained!")
print(f"   Accuracy: {accuracy:.2%}")
```

### Commit on Branch

```bash
# Stage and commit
git add train_model.py
git commit -m "exp: try random forest classifier"
```

### Switch Back to Main

```bash
# Switch to main branch
git checkout main

# Your file is back to the previous version!
cat train_model.py
```

### Merge Branch

```bash
# Merge experiment into main
git merge experiment-random-forest

# If no conflicts, you'll see:
# Updating abc1234..def5678
# Fast-forward
#  train_model.py | 15 +++++++++++++++
#  1 file changed, 15 insertions(+)
```

### Delete Experiment Branch

```bash
# Delete experiment branch
git branch -d experiment-random-forest
```

---

## 🌐 Step 6: Push to GitHub (5 minutes)

### Create Remote Repository

1. Go to: https://github.com/new
2. Repository name: `my-first-ml-project`
3. Description: "My first ML project with Git"
4. Public or Private (your choice)
5. **DO NOT** initialize with README
6. Click "Create repository"

### Connect and Push

```bash
# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/my-first-ml-project.git

# Push to GitHub
git push -u origin main

# You should see:
# Enumerating objects: 6, done.
# Counting objects: 100% (6/6), done.
# Delta compression using up to 8 threads
# Compressing objects: 100% (4/4), done.
# Writing objects: 100% (6/6), 1.23 KiB | 1.23 MiB/s, done.
# Total 6 (delta 0), reused 0 (delta 0)
# To https://github.com/YOUR_USERNAME/my-first-ml-project.git
#  * [new branch]      main -> main
```

### Verify on GitHub

1. Visit: https://github.com/YOUR_USERNAME/my-first-ml-project
2. You should see your code!
3. Click "Commits" to see history

---

## 🚫 Step 7: Using .gitignore (5 minutes)

### What is .gitignore?

A file that tells Git what **NOT** to track.

### Why .gitignore for ML?

```
DON'T track:
- Large datasets (GBs of data)
- Model checkpoints (can be retrained)
- Virtual environments
- Jupyter checkpoint files
- IDE settings

DO track:
- Code (.py files)
- Notebooks (.ipynb)
- Requirements.txt
- README.md
- Config files
```

### Create .gitignore

```bash
# Create .gitignore file
touch .gitignore
```

```
# .gitignore for ML Projects

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual environments
venv/
env/
.venv/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data (too large)
data/*.csv
data/*.parquet
*.h5
*.pkl

# Model checkpoints (optional - track if small)
models/*.pth
models/*.ckpt

# OS files
.DS_Store
Thumbs.db

# Environment variables
.env
*.env
```

### Add and Commit

```bash
# Add .gitignore
git add .gitignore
git commit -m "chore: add .gitignore for ML project"

# Push to GitHub
git push origin main
```

---

## 🤝 Step 8: Collaboration Basics (5 minutes)

### Clone Someone's Repository

```bash
# Clone a repository
git clone https://github.com/username/project.git
cd project
```

### Pull Latest Changes

```bash
# Get updates from remote
git pull origin main
```

### Typical Workflow

```bash
# 1. Get latest changes
git pull origin main

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Make changes and commit
git add .
git commit -m "feat: add my feature"

# 4. Push branch
git push origin feature/my-feature

# 5. Create Pull Request on GitHub
# 6. After review, merge on GitHub
# 7. Delete local branch
git branch -d feature/my-feature
```

---

## ✅ Tutorial Checklist

- [ ] Installed and configured Git
- [ ] Created first repository
- [ ] Made commits with messages
- [ ] Viewed commit history
- [ ] Created and switched branches
- [ ] Merged branches
- [ ] Pushed to GitHub
- [ ] Created .gitignore
- [ ] Cloned a repository

---

## 🎓 Key Takeaways

1. **Git tracks changes** - Save snapshots of your work
2. **Commits need messages** - Describe what changed
3. **Branches for experiments** - Don't break main
4. **Push to GitHub** - Backup and share
5. **.gitignore** - Don't track large files
6. **Pull before work** - Stay up to date

---

## 🚀 Next Steps

1. **Continue to Tutorial 10:** [Jupyter Notebooks Mastery](10-jupyter.md)
2. **Practice:** Use Git for all your projects
3. **Learn More:** Pro Git book at git-scm.com/book

---

## 💡 Challenge (Optional)

**Create a public ML project!**

1. Create new repository on GitHub
2. Add README with project description
3. Add your best ML code
4. Add requirements.txt
5. Share link in Discord

**Best project gets featured in showcase!** 🏆

---

**Tutorial Created:** April 2, 2026  
**Last Updated:** April 2, 2026  
**Estimated Time:** 45 minutes  
**Difficulty:** Beginner

---

[← Back to Tutorials](../README.md) | [Previous: First Neural Network](08-first-neural-net.md) | [Next: Jupyter Mastery](10-jupyter.md)
