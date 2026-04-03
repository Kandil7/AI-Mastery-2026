# 📓 Tutorial 10: Jupyter Notebooks Mastery

**Become productive with Jupyter in 40 minutes**

---

## 🎯 What You'll Build

By the end of this tutorial, you will have:

- ✅ Created and navigated notebooks
- ✅ Used keyboard shortcuts efficiently
- ✅ Written markdown with formatting
- ✅ Created visualizations inline
- ✅ Debugged and inspected variables
- ✅ Exported to multiple formats
- ✅ Organized a complete analysis

**Time Required:** 40 minutes  
**Difficulty:** ⭐⭐☆☆☆ (Beginner)  
**Prerequisites:** Tutorial 1 (Installation), basic Python

---

## 📋 What You'll Learn

- What are Jupyter Notebooks
- Notebook interface basics
- Cell types (code, markdown, raw)
- Essential keyboard shortcuts
- Magic commands
- Debugging techniques
- Best practices for ML projects
- Exporting and sharing

---

## 🧠 Step 1: Understand Jupyter (5 minutes)

### What are Notebooks?

**Jupyter Notebooks = Interactive documents combining:**
- Live code
- Visualizations
- Explanations (markdown)
- Equations
- Images

### Why Notebooks for AI/ML?

✅ **Exploratory** - Try things interactively  
✅ **Visual** - See results immediately  
✅ **Documented** - Code + explanations  
✅ **Shareable** - Send .ipynb files  
✅ **Standard** - Used by all data scientists  

### Notebook Structure

```
Cell 1: [Markdown] Title and introduction
Cell 2: [Code] Import libraries
Cell 3: [Code] Load data
Cell 4: [Markdown] Data exploration
Cell 5: [Code] Visualize data
Cell 6: [Markdown] Model training
Cell 7: [Code] Train model
Cell 8: [Markdown] Results and conclusions
```

---

## 🛠️ Step 2: Setup and Launch (5 minutes)

### Install Jupyter

```bash
# Install Jupyter Lab (recommended)
pip install jupyterlab

# Verify installation
jupyter lab --version
```

### Launch Jupyter

```bash
# Navigate to your project
cd my-first-ml-project

# Launch Jupyter Lab
jupyter lab

# Opens browser at http://localhost:8888
```

### Create New Notebook

1. Click "Python 3" under Notebook
2. New notebook opens with one empty cell
3. Rename: Click "Untitled" → "tutorial_10_jupyter_mastery.ipynb"

---

## ⌨️ Step 3: Essential Keyboard Shortcuts (10 minutes)

### Command Mode (Press Esc)

| Shortcut | Action |
|----------|--------|
| `Esc` | Enter command mode |
| `A` | Insert cell above |
| `B` | Insert cell below |
| `D, D` | Delete cell (press D twice) |
| `M` | Convert to markdown |
| `Y` | Convert to code |
| `Shift+Up/Down` | Select multiple cells |
| `Shift+M` | Merge selected cells |
| `Z` | Undo cell deletion |

### Edit Mode (Press Enter)

| Shortcut | Action |
|----------|--------|
| `Enter` | Enter edit mode |
| `Shift+Enter` | Run cell, go to next |
| `Ctrl+Enter` | Run cell, stay |
| `Alt+Enter` | Run cell, insert below |
| `Tab` | Autocomplete |
| `Shift+Tab` | Show docstring |
| `Ctrl+/` | Toggle comment |
| `Ctrl+Z` | Undo |

### Practice Exercise

```python
# Cell 1: Type and run
print("Hello Jupyter!")

# Press Shift+Enter to run
```

```python
# Cell 2: Try autocomplete
import numpy as np
arr = np.random.rand(10)  # Type "np.ra" then Tab
print(arr)
```

```python
# Cell 3: Try docstring
# Type sorted() then Shift+Tab
sorted([3, 1, 4, 1, 5, 9])
```

---

## 📝 Step 4: Markdown Cells (5 minutes)

### Create Markdown Cell

1. Press `Esc` to enter command mode
2. Press `M` to convert to markdown
3. Double-click to edit

### Markdown Basics

```markdown
# Heading 1
## Heading 2
### Heading 3

**Bold text**
*Italic text*
`inline code`

- Bullet point
- Another point
  - Sub-point

1. Numbered list
2. Second item

[Link text](https://example.com)

![Image](image.png)

> Blockquote

--- (horizontal line)
```

### Equations (LaTeX)

```markdown
Inline equation: $E = mc^2$

Block equation:
$$\frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)}) x^{(i)}$$
```

### Practice

Create a markdown cell with:
- Title: "My Analysis"
- Brief description
- Bullet list of steps
- One equation

---

## 🪄 Step 5: Magic Commands (5 minutes)

### What are Magics?

Special commands that start with `%` or `%%`

### Line Magics (single %)

```python
# Time execution
%timeit sum(range(1000))

# List variables
%who

# List variables with details
%whos

# Run external Python script
%run my_script.py

# Load code from file
%load my_script.py
```

### Cell Magics (double %%)

```python
# Time entire cell
%%time
import time
time.sleep(1)
print("Done!")
```

```python
# Write cell content to file
%%writefile hello.py
print("Hello from file!")
```

```python
# Run shell commands
%%bash
ls -la
pwd
```

### Useful Magics for ML

```python
# Auto-reload modules
%load_ext autoreload
%autoreload 2

# Now imports auto-reload when you modify them!
```

---

## 🔍 Step 6: Debugging and Inspection (5 minutes)

### Inspect Variables

```python
# Create some variables
import pandas as pd
import numpy as np

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
arr = np.random.randn(100, 5)
```

```python
# List all variables
%whos
```

### View Help

```python
# Show docstring
# Type function name then Shift+Tab
help(pd.DataFrame)

# Or use ?
pd.DataFrame?
```

### Debug Errors

```python
# Intentional error
try:
    result = 1 / 0
except Exception as e:
    print(f"Error: {e}")

# Use %debug to enter debugger
# %debug  # Uncomment to try
```

### Check Memory

```python
# Check memory usage
import sys

print(f"DataFrame size: {sys.getsizeof(df)} bytes")
print(f"Array size: {sys.getsizeof(arr)} bytes")
```

---

## 📊 Step 7: Visualizations in Notebooks (5 minutes)

### Inline Plots

```python
import matplotlib.pyplot as plt
import numpy as np

# Enable inline plots (usually automatic in Jupyter)
%matplotlib inline

# Create plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True, alpha=0.3)
plt.show()
```

### Interactive Plots (Optional)

```python
# Install ipywidgets for interactivity
# pip install ipywidgets

from ipywidgets import interact

@interact
def plot_sine(frequency=(1, 10, 0.5)):
    x = np.linspace(0, 10, 100)
    y = np.sin(frequency * x)
    
    plt.figure(figsize=(10, 4))
    plt.plot(x, y)
    plt.title(f'Sine Wave (frequency={frequency})')
    plt.grid(True, alpha=0.3)
    plt.show()
```

---

## 💾 Step 8: Export and Share (5 minutes)

### Export Notebook

**From Jupyter Lab:**
1. File → Export Notebook As...
2. Choose format:
   - HTML (web page)
   - PDF (document)
   - Markdown (.md)
   - Python script (.py)
   - Reveal.js (slides)

### Command Line Export

```bash
# Export to HTML
jupyter nbconvert --to html tutorial_10_jupyter_mastery.ipynb

# Export to PDF (requires pandoc)
jupyter nbconvert --to pdf tutorial_10_jupyter_mastery.ipynb

# Export to Python script
jupyter nbconvert --to script tutorial_10_jupyter_mastery.ipynb
```

### Share on GitHub

```bash
# Add notebook to Git
git add tutorial_10_jupyter_mastery.ipynb
git commit -m "docs: add Jupyter tutorial notebook"
git push origin main
```

---

## ✅ Tutorial Checklist

- [ ] Launched Jupyter Lab
- [ ] Created new notebook
- [ ] Used keyboard shortcuts
- [ ] Created markdown cells
- [ ] Used magic commands
- [ ] Inspected variables
- [ ] Created visualizations
- [ ] Exported notebook

---

## 🎓 Key Takeaways

1. **Shortcuts save time** - Learn Esc, A, B, M, Shift+Enter
2. **Mix code and markdown** - Document your work
3. **Use magics** - %timeit, %who, %load_ext
4. **Inspect with ?** - Shift+Tab for help
5. **Visualize inline** - %matplotlib inline
6. **Export easily** - HTML, PDF, Markdown

---

## 🚀 Next Steps

1. **Practice:** Use notebooks for all your ML work
2. **Learn More:** Jupyter documentation at jupyter.org
3. **Explore Extensions:**
   - Jupyter Lab extensions
   - nbdev (notebooks as libraries)
   - Papermill (parameterized notebooks)

---

## 💡 Challenge (Optional)

**Create a complete analysis notebook!**

1. Load a dataset (e.g., from sklearn or Kaggle)
2. Explore and visualize data
3. Train a model
4. Evaluate and document results
5. Export to HTML
6. Share on GitHub

**Best notebook gets featured!** 🏆

---

**Tutorial Created:** April 2, 2026  
**Last Updated:** April 2, 2026  
**Estimated Time:** 40 minutes  
**Difficulty:** Beginner

---

[← Back to Tutorials](../README.md) | [Previous: Git Basics](09-git-basics.md) | [Series Complete! 🎉](README.md)
