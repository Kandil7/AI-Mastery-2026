# 📁 Tutorial 7: Working with CSV Data

**Load, clean, and analyze CSV files in 40 minutes**

---

## 🎯 What You'll Build

By the end of this tutorial, you will have:

- ✅ Loaded CSV files with Pandas
- ✅ Explored and understood data structure
- ✅ Cleaned missing values and duplicates
- ✅ Transformed and filtered data
- ✅ Performed basic analysis
- ✅ Exported cleaned data

**Time Required:** 40 minutes  
**Difficulty:** ⭐⭐☆☆☆ (Beginner)  
**Prerequisites:** Tutorial 1 (Installation), basic Python

---

## 📋 What You'll Learn

- What is CSV format
- Pandas DataFrame basics
- Loading and saving CSV files
- Handling missing data
- Removing duplicates
- Filtering and sorting
- Basic data transformations
- Exporting results

---

## 🧠 Step 1: Understand CSV Files (5 minutes)

### What is CSV?

**CSV = Comma-Separated Values**

A simple text format for tabular data:

```csv
Name,Age,City,Salary
Alice,28,New York,75000
Bob,34,San Francisco,95000
Charlie,25,Chicago,65000
```

### Why CSV Matters

✅ **Universal** - Every data tool supports it  
✅ **Simple** - Human-readable text  
✅ **Lightweight** - No complex formatting  
✅ **Portable** - Works across platforms  

### Real-World CSV Sources

- Export from databases
- Downloaded from APIs
- Survey results
- Financial reports
- Sensor data
- Web scraping output

---

## 🛠️ Step 2: Setup (5 minutes)

### Install Pandas

```bash
# Install pandas (if not already installed)
pip install pandas numpy

# Verify installation
python -c "import pandas as pd; print(f'✅ Pandas {pd.__version__}')"
```

### Create Sample CSV

```python
# Create file: create_sample_data.py
import pandas as pd
import numpy as np

# Generate sample employee data
np.random.seed(42)

n_employees = 100

data = {
    'EmployeeID': range(1, n_employees + 1),
    'Name': [f'Employee_{i}' for i in range(1, n_employees + 1)],
    'Department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], n_employees),
    'Age': np.random.randint(22, 65, n_employees),
    'Salary': np.random.randint(40000, 120000, n_employees),
    'Years_Experience': np.random.randint(0, 30, n_employees),
    'Performance_Score': np.random.uniform(1.0, 5.0, n_employees).round(1),
    'Remote_Work': np.random.choice(['Yes', 'No'], n_employees),
    'Satisfaction': np.random.randint(1, 10, n_employees)
}

# Add some missing values
df = pd.DataFrame(data)
df.loc[np.random.choice(n_employees, 10), 'Salary'] = np.nan
df.loc[np.random.choice(n_employees, 5), 'Performance_Score'] = np.nan

# Save to CSV
df.to_csv('employee_data.csv', index=False)

print("✅ Sample CSV created: employee_data.csv")
print(f"   Rows: {len(df)}")
print(f"   Columns: {len(df.columns)}")
print(f"   Missing values: {df.isnull().sum().sum()}")
```

Run it:
```bash
python create_sample_data.py
```

---

## 📂 Step 3: Load CSV Data (5 minutes)

### Basic Loading

```python
import pandas as pd
import numpy as np

# Load CSV file
df = pd.read_csv('employee_data.csv')

print("✅ Data loaded successfully!")
print(f"\n📊 Dataset Info:")
print(f"   Shape: {df.shape}")
print(f"   Rows: {df.shape[0]}")
print(f"   Columns: {df.shape[1]}")
```

### First Look at Data

```python
# First 5 rows
print("\n📄 First 5 Rows:")
df.head()
```

```python
# Last 5 rows
print("\n📄 Last 5 Rows:")
df.tail()
```

```python
# Column names
print("\n📋 Columns:")
print(df.columns.tolist())
```

```python
# Data types
print("\n🔢 Data Types:")
print(df.dtypes)
```

---

## 🔍 Step 4: Explore Data (10 minutes)

### Statistical Summary

```python
# Numeric columns summary
print("📊 Statistical Summary:")
df.describe()
```

```python
# All columns info
print("\n📋 Detailed Info:")
df.info()
```

### Check Missing Values

```python
# Count missing values per column
print("❓ Missing Values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Missing_Percent': missing_pct
})
print(missing_df[missing_df['Missing_Count'] > 0])
```

### Check Duplicates

```python
# Count duplicate rows
n_duplicates = df.duplicated().sum()
print(f"\n🔄 Duplicate Rows: {n_duplicates}")

# Show duplicates if any
if n_duplicates > 0:
    print("\nDuplicate rows:")
    df[df.duplicated(keep=False)]
```

### Value Counts

```python
# Department distribution
print("\n🏢 Department Distribution:")
print(df['Department'].value_counts())
```

```python
# Remote work distribution
print("\n🏠 Remote Work Distribution:")
print(df['Remote_Work'].value_counts())
```

---

## 🧹 Step 5: Clean Data (10 minutes)

### Handle Missing Values

**Option 1: Drop rows with missing values**

```python
# Create copy to avoid modifying original
df_clean = df.copy()

# Drop rows with any missing values
df_clean_dropped = df_clean.dropna()

print(f"Before: {len(df_clean)} rows")
print(f"After dropping: {len(df_clean_dropped)} rows")
print(f"Removed: {len(df_clean) - len(df_clean_dropped)} rows")
```

**Option 2: Fill missing values**

```python
# Create copy
df_clean = df.copy()

# Fill numeric columns with median
df_clean['Salary'] = df_clean['Salary'].fillna(df_clean['Salary'].median())
df_clean['Performance_Score'] = df_clean['Performance_Score'].fillna(
    df_clean['Performance_Score'].median()
)

# Verify no missing values
print("✅ Missing values filled:")
print(f"   Remaining missing: {df_clean.isnull().sum().sum()}")
```

**Option 3: Forward/Backward fill**

```python
# For time series data
df_ffill = df.fillna(method='ffill')  # Forward fill
df_bfill = df.fillna(method='bfill')  # Backward fill
```

### Remove Duplicates

```python
# Remove duplicate rows
df_clean = df_clean.drop_duplicates()

print(f"After removing duplicates: {len(df_clean)} rows")
```

### Fix Data Types

```python
# Check if any columns need type conversion
print("Current types:")
print(df_clean.dtypes)

# Example: Convert to category
df_clean['Department'] = df_clean['Department'].astype('category')
df_clean['Remote_Work'] = df_clean['Remote_Work'].astype('category')

print("\n✅ Types updated:")
print(df_clean.dtypes)
```

---

## 🔧 Step 6: Transform Data (5 minutes)

### Add New Columns

```python
# Create copy
df_transformed = df_clean.copy()

# Add salary category
def salary_category(salary):
    if salary < 50000:
        return 'Entry'
    elif salary < 75000:
        return 'Mid'
    elif salary < 100000:
        return 'Senior'
    else:
        return 'Executive'

df_transformed['Salary_Category'] = df_transformed['Salary'].apply(salary_category)

# Add experience level
def experience_level(years):
    if years < 2:
        return 'Junior'
    elif years < 5:
        return 'Mid-Level'
    elif years < 10:
        return 'Senior'
    else:
        return 'Expert'

df_transformed['Experience_Level'] = df_transformed['Years_Experience'].apply(experience_level)

print("✅ New columns added:")
print(df_transformed[['Salary', 'Salary_Category', 'Years_Experience', 'Experience_Level']].head(10))
```

### Filter Data

```python
# Filter: High earners
high_earners = df_transformed[df_transformed['Salary'] > 80000]
print(f"\n💰 High Earners (>$80K): {len(high_earners)} employees")

# Filter: Remote workers in Engineering
remote_eng = df_transformed[
    (df_transformed['Department'] == 'Engineering') & 
    (df_transformed['Remote_Work'] == 'Yes')
]
print(f"🏠 Remote Engineers: {len(remote_eng)} employees")

# Filter: High performers
top_performers = df_transformed[df_transformed['Performance_Score'] >= 4.5]
print(f"⭐ Top Performers (≥4.5): {len(top_performers)} employees")
```

### Sort Data

```python
# Sort by salary (descending)
df_sorted = df_transformed.sort_values('Salary', ascending=False)

print("\n💰 Top 10 Salaries:")
df_sorted[['Name', 'Department', 'Salary']].head(10)
```

---

## 📊 Step 7: Analyze Data (5 minutes)

### Group By Analysis

```python
# Average salary by department
dept_salary = df_transformed.groupby('Department')['Salary'].agg(['mean', 'median', 'count'])
dept_salary.columns = ['Avg_Salary', 'Median_Salary', 'Count']
dept_salary = dept_salary.round(2)

print("📊 Department Salary Analysis:")
dept_salary
```

```python
# Performance by experience level
exp_perf = df_transformed.groupby('Experience_Level')['Performance_Score'].mean().round(2)

print("\n⭐ Performance by Experience Level:")
exp_perf
```

### Cross-Tabulation

```python
# Remote work by department
remote_dept = pd.crosstab(df_transformed['Department'], df_transformed['Remote_Work'])

print("\n🏠 Remote Work by Department:")
remote_dept
```

### Correlation Analysis

```python
# Numeric correlations
numeric_cols = ['Age', 'Salary', 'Years_Experience', 'Performance_Score', 'Satisfaction']
correlations = df_transformed[numeric_cols].corr()

print("\n🔗 Correlations:")
print(correlations['Salary'].sort_values(ascending=False))
```

---

## 💾 Step 8: Export Data (5 minutes)

### Save Cleaned Data

```python
# Save to CSV
df_transformed.to_csv('employee_data_cleaned.csv', index=False)

print("✅ Cleaned data saved to: employee_data_cleaned.csv")
```

### Save Analysis Results

```python
# Save summary statistics
summary_stats = df_transformed.describe()
summary_stats.to_csv('summary_statistics.csv')

print("✅ Summary statistics saved to: summary_statistics.csv")
```

### Export to Excel (Optional)

```python
# Install openpyxl if needed
# pip install openpyxl

# Save to Excel with multiple sheets
with pd.ExcelWriter('employee_analysis.xlsx') as writer:
    df_transformed.to_excel(writer, sheet_name='Cleaned_Data', index=False)
    dept_salary.to_excel(writer, sheet_name='Dept_Salary')
    exp_perf.to_excel(writer, sheet_name='Experience_Performance')

print("✅ Analysis saved to: employee_analysis.xlsx")
```

---

## ✅ Tutorial Checklist

- [ ] Created sample CSV data
- [ ] Loaded CSV with Pandas
- [ ] Explored data structure
- [ ] Checked missing values
- [ ] Cleaned missing data
- [ ] Removed duplicates
- [ ] Added new columns
- [ ] Filtered and sorted data
- [ ] Performed group analysis
- [ ] Exported cleaned data

---

## 🎓 Key Takeaways

1. **Load CSV:** `pd.read_csv('file.csv')`
2. **Explore:** `head()`, `info()`, `describe()`
3. **Clean:** `dropna()`, `fillna()`, `drop_duplicates()`
4. **Transform:** Add columns, filter, sort
5. **Analyze:** `groupby()`, `crosstab()`, `corr()`
6. **Export:** `to_csv()`, `to_excel()`

---

## 🚀 Next Steps

1. **Continue to Tutorial 8:** [Your First Neural Network](08-first-neural-net.md)
2. **Practice:** Load your own CSV files and explore
3. **Learn More:** Pandas documentation at pandas.pydata.org

---

## 💡 Challenge (Optional)

**Analyze a real dataset!**

1. Download dataset from Kaggle (e.g., Titanic, Iris, Housing)
2. Load and explore
3. Clean missing values
4. Create 3 insightful visualizations
5. Export your findings

**Share your analysis in Discord!** 📊

---

**Tutorial Created:** April 2, 2026  
**Last Updated:** April 2, 2026  
**Estimated Time:** 40 minutes  
**Difficulty:** Beginner

---

[← Back to Tutorials](../README.md) | [Previous: Data Visualization](06-data-viz.md) | [Next: First Neural Network](08-first-neural-net.md)
