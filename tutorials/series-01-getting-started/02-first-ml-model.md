# 🤖 Tutorial 2: Your First ML Model

**Build and train a machine learning model from scratch in 45 minutes**

---

## 🎯 What You'll Build

By the end of this tutorial, you will have:

- ✅ Loaded and explored a real dataset
- ✅ Preprocessed data for ML
- ✅ Trained your first ML model (Linear Regression)
- ✅ Evaluated model performance
- ✅ Made predictions on new data
- ✅ Visualized results

**Time Required:** 45 minutes  
**Difficulty:** ⭐⭐☆☆☆ (Beginner)  
**Prerequisites:** Tutorial 1 (Installation & Setup)

---

## 📋 What You'll Learn

- What is machine learning?
- Supervised vs unsupervised learning
- The ML workflow
- Linear regression intuition
- Model evaluation metrics
- Making predictions

---

## 🛠️ Step 1: Setup (5 minutes)

### Open Jupyter Notebook

**Option 1: VS Code (Recommended)**
```bash
# From project directory
code .

# Create new file: tutorial_02_first_ml_model.ipynb
# VS Code will open it as a Jupyter notebook
```

**Option 2: Jupyter Lab**
```bash
# From project directory
jupyter lab

# Create new notebook in browser
```

### Import Libraries

```python
# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("✅ All libraries imported successfully!")
```

---

## 📊 Step 2: Load and Explore Data (10 minutes)

### What is the California Housing Dataset?

This dataset contains information about California districts from the 1990 census. We'll use it to predict house prices based on various features.

**Features:**
- Median Income
- House Age
- Average Rooms
- Average Bedrooms
- Population
- Average Occupancy
- Latitude
- Longitude

**Target:** Median House Value (in $100,000s)

### Load Dataset

```python
# Load dataset
california = fetch_california_housing()

# Create DataFrame
df = pd.DataFrame(california.data, columns=california.feature_names)
df['Price'] = california.target

print(f"✅ Dataset loaded!")
print(f"   Samples: {df.shape[0]}")
print(f"   Features: {df.shape[1] - 1}")
print(f"   Target: Price (Median House Value)")
```

### Explore Data

```python
# First 5 rows
print("\n📄 First 5 rows:")
df.head()
```

```python
# Statistical summary
print("\n📊 Statistical Summary:")
df.describe()
```

```python
# Check for missing values
print("\n❓ Missing Values:")
print(df.isnull().sum())
```

### Visualize Data

```python
# Distribution of house prices
plt.figure(figsize=(10, 6))
plt.hist(df['Price'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Median House Value ($100,000s)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of House Prices in California', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()

print(f"\n💰 Average House Price: ${df['Price'].mean()*100000:,.0f}")
print(f"💰 Most Expensive: ${df['Price'].max()*100000:,.0f}")
print(f"💰 Cheapest: ${df['Price'].min()*100000:,.0f}")
```

```python
# Correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
           fmt='.2f', square=True, linewidths=1)
plt.title('Feature Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.show()

print("\n🔍 Key Insight: Median Income has highest correlation with Price (0.69)")
```

```python
# Scatter plot: Income vs Price
plt.figure(figsize=(10, 6))
plt.scatter(df['MedInc'], df['Price'], alpha=0.3, s=10)
plt.xlabel('Median Income', fontsize=12)
plt.ylabel('House Price ($100,000s)', fontsize=12)
plt.title('Income vs House Price', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()

print("\n💡 Observation: Higher income areas tend to have higher house prices")
```

---

## 🎓 Step 3: Understand the ML Workflow (5 minutes)

### The ML Workflow

```
1. Collect Data → 2. Explore → 3. Preprocess → 4. Split → 5. Train → 6. Evaluate → 7. Predict
```

### Why Split Data?

We split data into **training** and **testing** sets to:
- Train model on one portion (80%)
- Test on unseen data (20%)
- Ensure model generalizes well

### Train-Test Split

```python
# Separate features and target
X = df.drop('Price', axis=1)  # Features
y = df['Price']               # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,    # 20% for testing
    random_state=42   # For reproducibility
)

print(f"✅ Data Split Complete!")
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Testing samples: {X_test.shape[0]}")
print(f"   Features: {X_train.shape[1]}")
```

---

## 🤖 Step 4: Train Your First Model (10 minutes)

### What is Linear Regression?

Linear Regression finds the **best-fit line** through your data:

```
Price = w₁×Income + w₂×Age + w₃×Rooms + ... + b
```

Where:
- **w₁, w₂, ...** are weights (learned from data)
- **b** is bias (intercept)

### Intuition

Imagine plotting income vs price. Linear regression draws the line that minimizes the distance to all points.

### Train Model

```python
# Create model
model = LinearRegression()

# Train model
print("🏋️ Training model...")
model.fit(X_train, y_train)
print("✅ Model trained successfully!")
```

### View Model Parameters

```python
# View weights
print("\n⚖️ Model Weights:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"   {feature:20s}: {coef:8.4f}")

print(f"\n📍 Bias (intercept): {model.intercept_:.4f}")
```

### Interpret Weights

```python
# Create weight visualization
plt.figure(figsize=(10, 6))
features = X.columns
coefficients = model.coef_

bars = plt.barh(features, coefficients, color='steelblue')
plt.xlabel('Coefficient Value', fontsize=12)
plt.title('Feature Importance (Linear Regression Weights)', fontsize=14)
plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.show()

print("\n💡 Interpretation:")
print("   - Positive weight: Feature increases price")
print("   - Negative weight: Feature decreases price")
print("   - Larger absolute value: More important feature")
```

---

## 📈 Step 5: Evaluate Model (10 minutes)

### Make Predictions

```python
# Predict on test set
y_pred = model.predict(X_test)

print(f"✅ Predictions made!")
print(f"   Number of predictions: {len(y_pred)}")
```

### Compare Predictions vs Actual

```python
# Create comparison DataFrame
comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Difference': y_test.values - y_pred
})

print("\n📊 Sample Predictions:")
comparison.head(10)
```

### Evaluation Metrics

```python
# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("=" * 60)
print("📊 MODEL EVALUATION METRICS")
print("=" * 60)
print(f"\n1. Mean Squared Error (MSE): {mse:.4f}")
print(f"   - Average squared difference")
print(f"   - Lower is better")

print(f"\n2. Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"   - Average error in same units as target")
print(f"   - In $100,000s: ${rmse*100000:,.0f}")

print(f"\n3. R² Score: {r2:.4f}")
print(f"   - Percentage of variance explained")
print(f"   - Range: -∞ to 1.0")
print(f"   - 1.0 = Perfect prediction")
print(f"   - 0.0 = Predicts mean only")
print(f"   - Your model explains {r2*100:.1f}% of variance!")

print("\n" + "=" * 60)
```

### Visualize Results

```python
# Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3, s=10)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price ($100,000s)', fontsize=12)
plt.ylabel('Predicted Price ($100,000s)', fontsize=12)
plt.title('Actual vs Predicted House Prices', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n💡 Good predictions cluster around the red line")
```

```python
# Residual plot
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.3, s=10)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Price ($100,000s)', fontsize=12)
plt.ylabel('Residual (Error)', fontsize=12)
plt.title('Residual Plot', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()

print("\n💡 Good model: Residuals randomly scattered around 0")
print("   Pattern in residuals = Model missing something")
```

---

## 🔮 Step 6: Make Predictions (5 minutes)

### Predict on New Data

```python
# Create sample houses
new_houses = pd.DataFrame({
    'MedInc': [8.5, 3.2, 5.0],        # High, Low, Medium income
    'HouseAge': [10, 40, 25],          # New, Old, Medium age
    'AveRooms': [5.5, 4.0, 5.0],       # Rooms
    'AveBedrms': [1.0, 1.5, 1.2],      # Bedrooms
    'Population': [1200, 800, 1000],   # Population
    'AveOccup': [3.0, 4.5, 3.5],       # Occupancy
    'Latitude': [34.0, 37.5, 36.0],    # Location
    'Longitude': [-118.0, -122.0, -120.0]
})

print("🏠 Sample Houses:")
new_houses
```

```python
# Make predictions
predictions = model.predict(new_houses)

print("\n🔮 Predictions:")
for i, (house, pred) in enumerate(zip(new_houses.iterrows(), predictions), 1):
    print(f"\nHouse {i}:")
    print(f"   Income: ${house[1]['MedInc']*10000:,.0f}/year")
    print(f"   Age: {house[1]['HouseAge']} years")
    print(f"   Predicted Price: ${pred*100000:,.0f}")
```

---

## 🎯 Step 7: Understanding Check (5 minutes)

### Knowledge Check

**Q1:** What does the R² score of 0.60 mean?

A) Model is 60% accurate  
B) Model explains 60% of price variance  
C) Model is wrong 40% of time  
D) Model needs more data  

**Answer:** B) Model explains 60% of price variance

---

**Q2:** Why do we split data into train and test sets?

A) To make training faster  
B) To evaluate on unseen data  
C) To reduce memory usage  
D) It's required by sklearn  

**Answer:** B) To evaluate on unseen data

---

**Q3:** If RMSE is 0.5, what does that mean?

A) Model is 50% accurate  
B) Average error is $50,000  
C) Average error is $5,000  
D) Model needs improvement  

**Answer:** B) Average error is $50,000 (0.5 × $100,000)

---

## 🚀 Step 8: Experiment! (5 minutes)

### Try Different Models

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)

print("=" * 60)
print("🏆 MODEL COMPARISON")
print("=" * 60)
print(f"\nLinear Regression R²: {r2:.4f}")
print(f"Random Forest R²:     {rf_r2:.4f}")
print(f"\n🎉 Better model: {'Random Forest' if rf_r2 > r2 else 'Linear Regression'}")
print(f"   Improvement: {abs(rf_r2 - r2)*100:.1f}%")
```

### Feature Importance (Random Forest)

```python
# Get feature importance
importances = rf.feature_importances_

# Sort and plot
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)
plt.title('Random Forest Feature Importance', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## ✅ Tutorial Checklist

- [ ] Loaded and explored dataset
- [ ] Visualized data distributions
- [ ] Split data into train/test
- [ ] Trained Linear Regression model
- [ ] Evaluated with MSE, RMSE, R²
- [ ] Made predictions on new data
- [ ] Compared with Random Forest
- [ ] Understood feature importance

---

## 🎓 Key Takeaways

1. **ML Workflow:** Load → Explore → Split → Train → Evaluate → Predict
2. **Linear Regression:** Finds best-fit line through data
3. **R² Score:** Measures how well model explains variance
4. **Train-Test Split:** Essential for evaluating generalization
5. **Feature Importance:** Shows which features matter most

---

## 🚀 Next Steps

Now that you've built your first ML model:

1. **Continue to Tutorial 3:** [Build a Simple Chatbot](03-simple-chatbot.md)
2. **Experiment:** Try different datasets from sklearn
3. **Learn More:** Dive into Linear Regression math in Tier 1, Module 1.7

---

## 📞 Get Help

- **Stuck?** Join Discord `#tutorial-help`
- **Questions?** Check FAQ in docs
- **Share your work!** Post in `#showcase`

---

## 💡 Challenge (Optional)

**Improve the model!** Try:

1. Feature engineering (create new features)
2. Data preprocessing (scaling, normalization)
3. Different models (Gradient Boosting, SVM)
4. Hyperparameter tuning

**Best R² score wins a shoutout in Discord!** 🏆

---

**Tutorial Created:** April 2, 2026  
**Last Updated:** April 2, 2026  
**Estimated Time:** 45 minutes  
**Difficulty:** Beginner

---

[← Back to Tutorials](../README.md) | [Previous: Installation](01-installation.md) | [Next: Build a Chatbot](03-simple-chatbot.md)
