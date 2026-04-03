# 🎓 Tier 0: Absolute Beginner

**Start your AI journey here!** No programming experience required.

---

## 📋 Overview

**Duration:** 4-6 weeks  
**Commitment:** 8-10 hours/week  
**Prerequisites:** None  
**Outcome:** Python programming basics + mathematical foundations

---

## 🎯 Learning Objectives

By the end of Tier 0, you will be able to:

- ✅ Write Python programs with variables, functions, and control flow
- ✅ Solve basic algebra and calculus problems
- ✅ Understand fundamental AI/ML concepts
- ✅ Be ready for Tier 1 (Fundamentals)

---

## 📚 Module List

### Module 0.1: Python Basics - Variables & Data Types

**Duration:** 5 hours  
**Difficulty:** ⭐☆☆☆☆

#### What You'll Learn

- Installing Python and setting up your environment
- Variables and naming conventions
- Basic data types (int, float, str, bool)
- Input and output
- Simple operations

#### Topics Covered

1. **Setup & Installation** (30 min)
   - Installing Python 3.10+
   - Choosing a code editor (VS Code recommended)
   - Your first Python program

2. **Variables** (1 hour)
   - What are variables?
   - Naming rules and conventions
   - Assignment and reassignment

3. **Data Types** (2 hours)
   - Integers (whole numbers)
   - Floats (decimal numbers)
   - Strings (text)
   - Booleans (True/False)

4. **Operations** (1 hour)
   - Arithmetic operations (+, -, *, /)
   - String concatenation
   - Type conversion

5. **Input/Output** (30 min)
   - Getting user input
   - Printing results
   - Formatting output

#### Hands-On Lab

**Exercise:** Build a Simple Calculator

```python
# Your task: Create a calculator that:
# 1. Asks user for two numbers
# 2. Asks user for operation (+, -, *, /)
# 3. Prints the result

# Starter code:
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))
operation = input("Enter operation (+, -, *, /): ")

# Your code here...
```

#### Quiz

5 questions to test your understanding

#### Resources

- [Python Official Documentation](https://docs.python.org/3/)
- [Python for Beginners (Microsoft)](https://learn.microsoft.com/en-us/training/paths/introduction-to-python/)
- Video: [Python in 100 Seconds](https://www.youtube.com/watch?v=H2LbABwAWkE)

---

### Module 0.2: Control Flow - If/Else & Loops

**Duration:** 5 hours  
**Difficulty:** ⭐⭐☆☆☆

#### What You'll Learn

- Conditional statements (if, elif, else)
- Comparison operators
- Logical operators (and, or, not)
- For loops
- While loops

#### Topics Covered

1. **Conditionals** (1.5 hours)
   - if statements
   - elif and else
   - Nested conditionals

2. **Comparison Operators** (30 min)
   - Equal to (==)
   - Not equal to (!=)
   - Greater than, less than (>, <)
   - Greater/less than or equal (>=, <=)

3. **Logical Operators** (30 min)
   - AND (both conditions true)
   - OR (at least one true)
   - NOT (inverse)

4. **For Loops** (1.5 hours)
   - Iterating over ranges
   - Iterating over strings
   - Loop variables

5. **While Loops** (1 hour)
   - Condition-based looping
   - Infinite loops and how to avoid them
   - Break and continue

#### Hands-On Lab

**Exercise:** Number Guessing Game

```python
# Build a game where:
# 1. Computer picks a random number (1-100)
# 2. User guesses the number
# 3. Computer gives hints (higher/lower)
# 4. Game ends when user guesses correctly

import random

secret_number = random.randint(1, 100)
guesses = 0

# Your code here...
```

#### Quiz

8 questions including code tracing

---

### Module 0.3: Functions & Modules

**Duration:** 5 hours  
**Difficulty:** ⭐⭐☆☆☆

#### What You'll Learn

- Defining functions
- Parameters and arguments
- Return values
- Scope (local vs global)
- Importing modules

#### Topics Covered

1. **Function Basics** (1.5 hours)
   - Why use functions?
   - def keyword
   - Calling functions

2. **Parameters & Arguments** (1.5 hours)
   - Positional arguments
   - Keyword arguments
   - Default parameters

3. **Return Values** (1 hour)
   - return statement
   - Returning multiple values
   - None return

4. **Scope** (30 min)
   - Local variables
   - Global variables
   - Best practices

5. **Modules** (30 min)
   - Import statement
   - Standard library modules
   - Creating your own modules

#### Hands-On Lab

**Exercise:** Build a Unit Converter

```python
# Create functions to convert:
# - Celsius to Fahrenheit
# - Kilometers to Miles
# - Kilograms to Pounds

def celsius_to_fahrenheit(celsius):
    # Formula: F = C * 9/5 + 32
    pass  # Your code here

# Test your functions...
```

---

### Module 0.4: Math Essentials - Algebra

**Duration:** 8 hours  
**Difficulty:** ⭐⭐☆☆☆

#### What You'll Learn

- Variables and expressions
- Linear equations
- Functions and graphs
- Systems of equations
- Why algebra matters for AI

#### Topics Covered

1. **Variables & Expressions** (2 hours)
   - Algebraic notation
   - Simplifying expressions
   - Substitution

2. **Linear Equations** (2 hours)
   - Solving for x
   - Graphing lines
   - Slope and intercept

3. **Functions** (2 hours)
   - Function notation f(x)
   - Domain and range
   - Common functions

4. **Systems of Equations** (1.5 hours)
   - Two equations, two unknowns
   - Substitution method
   - Elimination method

5. **AI Connection** (30 min)
   - How algebra is used in ML
   - Linear regression preview
   - Cost functions

#### Hands-On Lab

**Exercise:** Visualize Linear Functions

```python
import matplotlib.pyplot as plt
import numpy as np

# Plot y = mx + b for different m and b
x = np.linspace(-10, 10, 100)

# Your code: Plot multiple lines
```

---

### Module 0.5: Math Essentials - Functions & Calculus Preview

**Duration:** 7 hours  
**Difficulty:** ⭐⭐⭐☆☆

#### What You'll Learn

- Function types (linear, quadratic, exponential)
- Rate of change
- Introduction to derivatives
- Gradients (preview for ML)

#### Topics Covered

1. **Function Types** (2 hours)
   - Linear functions
   - Quadratic functions
   - Exponential functions
   - Logarithmic functions

2. **Rate of Change** (2 hours)
   - Average rate of change
   - Instantaneous rate of change
   - Slope of tangent line

3. **Derivatives Introduction** (2 hours)
   - What is a derivative?
   - Basic rules
   - Power rule

4. **Gradients** (1 hour)
   - Multi-variable functions
   - Partial derivatives
   - Gradient vectors

#### Hands-On Lab

**Exercise:** Visualize Derivatives

```python
# Plot a function and its derivative
def f(x):
    return x**2

def derivative_f(x):
    return 2*x  # Derivative of x^2

# Visualize both
```

---

### Module 0.6: Introduction to AI Concepts

**Duration:** 5 hours  
**Difficulty:** ⭐☆☆☆☆

#### What You'll Learn

- What is AI?
- AI vs ML vs Deep Learning
- Real-world AI applications
- Ethics in AI
- Your AI learning journey ahead

#### Topics Covered

1. **AI Overview** (1 hour)
   - History of AI
   - Current state
   - Future trends

2. **AI, ML, DL Differences** (1 hour)
   - Artificial Intelligence (broad)
   - Machine Learning (subset)
   - Deep Learning (subset of ML)

3. **Applications** (1.5 hours)
   - Computer Vision
   - Natural Language Processing
   - Recommendation Systems
   - Autonomous Vehicles

4. **Ethics** (1 hour)
   - Bias in AI
   - Privacy concerns
   - Responsible AI development

5. **Career Paths** (30 min)
   - AI Researcher
   - ML Engineer
   - Data Scientist
   - AI Product Manager

#### Hands-On Lab

**Exercise:** Explore Pre-trained AI Models

```python
# Use a pre-trained model to classify images
# or analyze sentiment in text

from transformers import pipeline

# Try sentiment analysis
classifier = pipeline("sentiment-analysis")
result = classifier("I love learning AI!")
print(result)
```

---

## 🎓 Tier 0 Final Project

**Build: Personal Budget Tracker**

Combine everything you've learned:

### Requirements

1. **Python Basics**
   - Use variables for income/expenses
   - Store transactions in lists

2. **Control Flow**
   - Menu system (if/else)
   - Loop for continuous operation

3. **Functions**
   - Separate functions for:
     - Add transaction
     - View balance
     - Generate report

4. **Math**
   - Calculate totals
   - Calculate percentages
   - Show trends

### Example Output

```
=== Personal Budget Tracker ===
1. Add Income
2. Add Expense
3. View Balance
4. Generate Report
5. Exit

Choose option: 1
Enter amount: 5000
Enter description: Salary
Income added!

Choose option: 3
Current Balance: $4200
```

---

## ✅ Tier 0 Completion Checklist

- [ ] Complete all 6 modules
- [ ] Pass all module quizzes (80%+)
- [ ] Complete all hands-on labs
- [ ] Submit final project
- [ ] Take Tier 0 Final Exam

---

## 📊 Time Commitment

| Module | Theory | Practice | Total |
|--------|--------|----------|-------|
| 0.1 | 2 hours | 3 hours | 5 hours |
| 0.2 | 2 hours | 3 hours | 5 hours |
| 0.3 | 2 hours | 3 hours | 5 hours |
| 0.4 | 4 hours | 4 hours | 8 hours |
| 0.5 | 3 hours | 4 hours | 7 hours |
| 0.6 | 3 hours | 2 hours | 5 hours |
| **Final Project** | 1 hour | 9 hours | 10 hours |
| **TOTAL** | **17 hours** | **28 hours** | **45 hours** |

---

## 🎯 What's Next?

After completing Tier 0:

1. **Celebrate!** 🎉 You've built a strong foundation
2. **Review** any concepts you found challenging
3. **Proceed to Tier 1** - Fundamentals

### Ready for Tier 1?

You should be comfortable with:
- ✅ Python variables, functions, loops
- ✅ Basic algebra (solving equations)
- ✅ Function concepts (f(x), graphs)
- ✅ Running Python programs

If yes → [Start Tier 1: Fundamentals](../tier-1-fundamentals/README.md)

If no → Review Tier 0 modules or join study group

---

## 💬 Get Help

- **Stuck on a concept?** → Join Discord `#tier-0-help`
- **Need motivation?** → Share progress in `#progress`
- **Questions?** → Weekly office hours (Tuesdays 6 PM UTC)

---

**Last Updated:** April 2, 2026  
**Authors:** AI-Mastery-2026 Education Team  
**Version:** 1.0

---

[← Back to Course Catalog](../COURSE_CATALOG.md) | [Start Module 0.1](module-01-python-basics/README.md) | [Join Discord](https://discord.gg/aimastery2026)
