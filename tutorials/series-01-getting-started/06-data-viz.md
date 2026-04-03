# 📊 Tutorial 6: Data Visualization Basics

**Create insightful data visualizations in 45 minutes**

---

## 🎯 What You'll Build

By the end of this tutorial, you will have:

- ✅ Created line plots for time series
- ✅ Built scatter plots for relationships
- ✅ Made histograms for distributions
- ✅ Designed bar charts for comparisons
- ✅ Created heatmaps for correlations
- ✅ Customized plots with labels, titles, and styles

**Time Required:** 45 minutes  
**Difficulty:** ⭐⭐☆☆☆ (Beginner)  
**Prerequisites:** Tutorial 1 (Installation), basic Python

---

## 📋 What You'll Learn

- Why visualization matters in data science
- Matplotlib fundamentals
- Seaborn for statistical plots
- Choosing the right chart type
- Customizing and saving plots
- Best practices for clear visualizations

---

## 🧠 Step 1: Understand Data Visualization (5 minutes)

### Why Visualize Data?

**A picture is worth a thousand numbers.**

Visualization helps you:
- **Discover patterns** in data
- **Communicate insights** to others
- **Detect anomalies** and outliers
- **Validate assumptions** about data

### Anscombe's Quartet

Four datasets with identical statistics but completely different patterns:

```
Dataset 1: Linear relationship
Dataset 2: Curved relationship
Dataset 3: Linear with outlier
Dataset 4: One outlier drives correlation
```

**Lesson:** Always visualize your data before analyzing!

### Chart Types Guide

| Goal | Best Chart |
|------|------------|
| Show trend over time | Line plot |
| Compare categories | Bar chart |
| Show distribution | Histogram |
| Show relationship | Scatter plot |
| Show correlation | Heatmap |
| Show proportions | Pie chart (use sparingly) |

---

## 🛠️ Step 2: Setup (5 minutes)

### Install Libraries

```bash
# Install visualization libraries
pip install matplotlib seaborn pandas numpy

# Verify installation
python -c "import matplotlib; print(f'✅ Matplotlib {matplotlib.__version__}')"
python -c "import seaborn; print(f'✅ Seaborn {seaborn.__version__}')"
python -c "import pandas; print(f'✅ Pandas {pandas.__version__}')"
```

### Create Notebook

```python
# Create file: tutorial_06_data_viz.ipynb
# Open in VS Code or Jupyter Lab
```

### Import Libraries

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configure display
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("✅ Visualization libraries loaded!")
```

---

## 📈 Step 3: Line Plots (10 minutes)

### What are Line Plots?

Line plots show **trends over time** or **continuous relationships**.

### Create Sample Data

```python
# Generate time series data
np.random.seed(42)

# Stock price simulation
days = np.arange(1, 31)
price_a = 100 + np.cumsum(np.random.randn(30) * 2)
price_b = 50 + np.cumsum(np.random.randn(30) * 1.5)

# Create DataFrame
df_stocks = pd.DataFrame({
    'Day': days,
    'Stock_A': price_a,
    'Stock_B': price_b
})

print("📊 Sample Data:")
df_stocks.head()
```

### Basic Line Plot

```python
# Create figure
plt.figure(figsize=(12, 6))

# Plot line
plt.plot(df_stocks['Day'], df_stocks['Stock_A'], 
         'b-', linewidth=2, label='Stock A')

# Add labels and title
plt.xlabel('Day', fontsize=14)
plt.ylabel('Price ($)', fontsize=14)
plt.title('Stock A Price Over Time', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Show plot
plt.tight_layout()
plt.savefig('01_line_plot_basic.png', dpi=300)
plt.show()

print("✅ Basic line plot created!")
```

### Multiple Lines

```python
# Create figure
plt.figure(figsize=(12, 6))

# Plot multiple lines
plt.plot(df_stocks['Day'], df_stocks['Stock_A'], 
         'b-', linewidth=2, label='Stock A', marker='o', markersize=4)
plt.plot(df_stocks['Day'], df_stocks['Stock_B'], 
         'r-', linewidth=2, label='Stock B', marker='s', markersize=4)

# Add labels and title
plt.xlabel('Day', fontsize=14)
plt.ylabel('Price ($)', fontsize=14)
plt.title('Stock Prices Over 30 Days', fontsize=16)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, alpha=0.3)

# Add annotations
plt.annotate('Peak', xy=(15, price_a[14]), 
             xytext=(18, price_a[14]+10),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=12)

plt.tight_layout()
plt.savefig('02_line_plot_multiple.png', dpi=300)
plt.show()

print("✅ Multiple line plot created!")
```

---

## 🔵 Step 4: Scatter Plots (10 minutes)

### What are Scatter Plots?

Scatter plots show **relationships between two variables**.

### Create Sample Data

```python
# Generate correlation data
np.random.seed(42)

n_samples = 100
x = np.random.randn(n_samples)
y = 2 * x + np.random.randn(n_samples) * 0.5  # Strong correlation

# Create DataFrame
df_scatter = pd.DataFrame({
    'Feature_X': x,
    'Feature_Y': y
})

print(f"📊 Generated {n_samples} samples")
print(f"   Correlation: {df_scatter['Feature_X'].corr(df_scatter['Feature_Y']):.3f}")
```

### Basic Scatter Plot

```python
# Create figure
plt.figure(figsize=(10, 6))

# Create scatter plot
plt.scatter(df_scatter['Feature_X'], df_scatter['Feature_Y'], 
           alpha=0.6, s=50, c='steelblue', edgecolors='black')

# Add labels and title
plt.xlabel('Feature X', fontsize=14)
plt.ylabel('Feature Y', fontsize=14)
plt.title('Relationship Between X and Y', fontsize=16)
plt.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(df_scatter['Feature_X'], df_scatter['Feature_Y'], 1)
p = np.poly1d(z)
plt.plot(df_scatter['Feature_X'], 
         p(df_scatter['Feature_X']), 
         "r--", alpha=0.8, linewidth=2, label=f'Trend (slope={z[0]:.2f})')

plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('03_scatter_plot.png', dpi=300)
plt.show()

print("✅ Scatter plot with trend line created!")
```

### Scatter with Color Mapping

```python
# Add third dimension with color
np.random.seed(42)
n = 200
x = np.random.randn(n)
y = x + np.random.randn(n) * 0.3
colors = np.random.randn(n)  # Color values

# Create figure
plt.figure(figsize=(10, 6))

# Scatter with color mapping
scatter = plt.scatter(x, y, c=colors, cmap='viridis', 
                     alpha=0.7, s=60, edgecolors='black')

# Add colorbar
plt.colorbar(scatter, label='Value')

# Add labels
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.title('Scatter Plot with Color Mapping', fontsize=16)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_scatter_colormap.png', dpi=300)
plt.show()

print("✅ Colored scatter plot created!")
```

---

## 📊 Step 5: Histograms (10 minutes)

### What are Histograms?

Histograms show **distribution of a single variable**.

### Create Sample Data

```python
# Generate different distributions
np.random.seed(42)

normal_data = np.random.randn(1000)  # Normal distribution
skewed_data = np.random.exponential(2, 1000)  # Skewed distribution
bimodal_data = np.concatenate([
    np.random.randn(500) - 2,
    np.random.randn(500) + 2
])  # Bimodal distribution

print("📊 Generated 3 distributions:")
print(f"   Normal: mean={normal_data.mean():.2f}, std={normal_data.std():.2f}")
print(f"   Skewed: mean={skewed_data.mean():.2f}, std={skewed_data.std():.2f}")
print(f"   Bimodal: mean={bimodal_data.mean():.2f}, std={bimodal_data.std():.2f}")
```

### Basic Histogram

```python
# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Normal distribution
axes[0].hist(normal_data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_title('Normal Distribution', fontsize=14)
axes[0].set_xlabel('Value', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].grid(True, alpha=0.3)

# Skewed distribution
axes[1].hist(skewed_data, bins=30, edgecolor='black', alpha=0.7, color='coral')
axes[1].set_title('Skewed Distribution', fontsize=14)
axes[1].set_xlabel('Value', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].grid(True, alpha=0.3)

# Bimodal distribution
axes[2].hist(bimodal_data, bins=30, edgecolor='black', alpha=0.7, color='seagreen')
axes[2].set_title('Bimodal Distribution', fontsize=14)
axes[2].set_xlabel('Value', fontsize=12)
axes[2].set_ylabel('Frequency', fontsize=12)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_histograms.png', dpi=300)
plt.show()

print("✅ Histograms created!")
```

### Histogram with KDE

```python
# Create figure
plt.figure(figsize=(10, 6))

# Histogram with KDE (Kernel Density Estimate)
plt.hist(normal_data, bins=30, density=True, alpha=0.6, 
         color='steelblue', edgecolor='black', label='Histogram')

# KDE line
sns.kdeplot(normal_data, color='red', linewidth=2, label='KDE')

# Add labels
plt.xlabel('Value', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title('Histogram with KDE', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('06_histogram_kde.png', dpi=300)
plt.show()

print("✅ Histogram with KDE created!")
```

---

## 📊 Step 6: Bar Charts (5 minutes)

### What are Bar Charts?

Bar charts **compare categories**.

### Create Sample Data

```python
# Sample data: AI course enrollment
courses = ['Python Basics', 'ML Fundamentals', 'Deep Learning', 
           'NLP', 'Computer Vision', 'RAG Systems']
enrollments = [1200, 950, 800, 650, 550, 1500]

# Create DataFrame
df_courses = pd.DataFrame({
    'Course': courses,
    'Enrollments': enrollments
}).sort_values('Enrollments', ascending=True)

print("📊 Course Enrollment Data:")
df_courses
```

### Horizontal Bar Chart

```python
# Create figure
plt.figure(figsize=(10, 6))

# Horizontal bar chart
bars = plt.barh(df_courses['Course'], df_courses['Enrollments'], 
               color='steelblue', edgecolor='black')

# Add value labels
for bar, val in zip(bars, df_courses['Enrollments']):
    plt.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', fontsize=11, fontweight='bold')

# Add labels
plt.xlabel('Number of Students', fontsize=14)
plt.title('AI Course Enrollments', fontsize=16)
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('07_bar_chart.png', dpi=300)
plt.show()

print("✅ Bar chart created!")
```

---

## 🔥 Step 7: Heatmaps (5 minutes)

### What are Heatmaps?

Heatmaps show **correlations** or **relationships** between variables.

### Create Correlation Data

```python
# Generate multivariate data
np.random.seed(42)
n = 200

data = {
    'Study_Hours': np.random.uniform(1, 8, n),
    'Sleep_Hours': np.random.uniform(5, 9, n),
    'Exercise_Hours': np.random.uniform(0, 3, n),
    'Stress_Level': np.random.uniform(1, 10, n),
    'GPA': np.random.uniform(2.0, 4.0, n)
}

df_students = pd.DataFrame(data)

# Add realistic correlations
df_students['GPA'] = (0.5 * df_students['Study_Hours'] + 
                     0.3 * df_students['Sleep_Hours'] - 
                     0.2 * df_students['Stress_Level'] + 
                     np.random.randn(n) * 0.3)
df_students['GPA'] = df_students['GPA'].clip(2.0, 4.0)

print("📊 Student Data:")
df_students.head()
```

### Correlation Heatmap

```python
# Calculate correlation matrix
corr_matrix = df_students.corr()

# Create figure
plt.figure(figsize=(10, 8))

# Create heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
           fmt='.2f', square=True, linewidths=1, 
           cbar_kws={'label': 'Correlation Coefficient'})

# Add title
plt.title('Student Lifestyle Correlations', fontsize=16)

plt.tight_layout()
plt.savefig('08_heatmap.png', dpi=300)
plt.show()

print("✅ Heatmap created!")
print("\n💡 Key Insights:")
print(f"   Study Hours ↔ GPA: {corr_matrix.loc['Study_Hours', 'GPA']:.2f}")
print(f"   Sleep Hours ↔ GPA: {corr_matrix.loc['Sleep_Hours', 'GPA']:.2f}")
print(f"   Stress Level ↔ GPA: {corr_matrix.loc['Stress_Level', 'GPA']:.2f}")
```

---

## 🎨 Step 8: Customization Tips (5 minutes)

### Best Practices

**Do ✅:**
- Use clear titles and labels
- Choose appropriate color schemes
- Include legends when needed
- Use grid lines for readability
- Save in high resolution (300 DPI)

**Don't ❌:**
- Use 3D charts (distorts data)
- Overload with too much information
- Use misleading scales
- Forget to label axes
- Use too many colors

### Quick Customization Examples

```python
# Change style
plt.style.use('seaborn-v0_8-darkgrid')

# Custom colors
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

# Font sizes
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16

# Figure size
plt.figure(figsize=(12, 8))

# Save with high DPI
plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')
```

---

## ✅ Tutorial Checklist

- [ ] Created line plots for trends
- [ ] Built scatter plots for relationships
- [ ] Made histograms for distributions
- [ ] Designed bar charts for comparisons
- [ ] Created heatmaps for correlations
- [ ] Customized plots with labels and styles
- [ ] Saved plots in high resolution

---

## 🎓 Key Takeaways

1. **Line plots** - Trends over time
2. **Scatter plots** - Relationships between variables
3. **Histograms** - Distribution of single variable
4. **Bar charts** - Compare categories
5. **Heatmaps** - Show correlations
6. **Always label** - Titles, axes, legends
7. **Save properly** - 300 DPI for publications

---

## 🚀 Next Steps

1. **Continue to Tutorial 7:** [Working with CSV Data](07-csv-data.md)
2. **Practice:** Visualize your own datasets
3. **Learn More:** Explore Seaborn gallery at seaborn.pydata.org

---

## 💡 Challenge (Optional)

**Create a data dashboard!**

1. Load a real dataset (e.g., from Kaggle)
2. Create 5 different visualizations
3. Add insights as text annotations
4. Save as PDF report
5. Share in Discord `#showcase`

**Best dashboard wins a shoutout!** 🏆

---

**Tutorial Created:** April 2, 2026  
**Last Updated:** April 2, 2026  
**Estimated Time:** 45 minutes  
**Difficulty:** Beginner

---

[← Back to Tutorials](../README.md) | [Previous: Deploy API](05-deploy-api.md) | [Next: CSV Data](07-csv-data.md)
