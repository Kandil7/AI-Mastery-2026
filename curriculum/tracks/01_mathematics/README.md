# Track 01: Mathematics for AI

## 🎯 Track Overview

This comprehensive track provides the mathematical foundations essential for Artificial Intelligence and Machine Learning. Built on a **"white-box" philosophy**, every concept is implemented from scratch using pure NumPy before introducing higher-level libraries.

### Track Philosophy

> **"Understand the internals before using the abstractions"**

- **From Scratch First**: Every algorithm is implemented using only NumPy
- **Mathematical Rigor**: Proofs and derivations are provided for key results
- **Visual Intuition**: Interactive visualizations complement formal mathematics
- **Production Ready**: Code follows software engineering best practices

---

## 📚 Module Structure

| Module | Topic | Duration | Difficulty | Prerequisites |
|--------|-------|----------|------------|---------------|
| 01 | Vectors and Matrices | 2 weeks | Beginner | High school algebra |
| 02 | Matrix Operations and Transformations | 2 weeks | Beginner | Module 01 |
| 03 | Eigenvalues and Eigenvectors | 2 weeks | Intermediate | Modules 01-02 |
| 04 | Matrix Decompositions | 3 weeks | Intermediate | Modules 01-03 |
| 05 | Derivatives and Gradients | 2 weeks | Intermediate | Calculus basics |
| 06 | Chain Rule and Backpropagation | 3 weeks | Advanced | Modules 01-05 |
| 07 | Probability Distributions | 2 weeks | Intermediate | Statistics basics |
| 08 | Bayes Theorem and Statistics | 3 weeks | Advanced | Modules 01-07 |

**Total Duration**: 19 weeks (~5 months)

---

## 🎓 Learning Outcomes

After completing this track, you will be able to:

### Linear Algebra
- ✅ Implement vector and matrix operations from scratch
- ✅ Perform matrix decompositions (LU, QR, SVD, Eigendecomposition)
- ✅ Understand geometric interpretations of linear transformations
- ✅ Apply eigenvalue analysis to real-world problems

### Calculus
- ✅ Compute derivatives and gradients manually and programmatically
- ✅ Implement automatic differentiation from scratch
- ✅ Understand and implement backpropagation for neural networks
- ✅ Optimize functions using gradient-based methods

### Probability & Statistics
- ✅ Work with major probability distributions
- ✅ Implement Bayesian inference from scratch
- ✅ Perform statistical hypothesis testing
- ✅ Apply probabilistic reasoning to ML problems

---

## 📁 Directory Structure

```
curriculum/tracks/01_mathematics/
├── README.md                    # This file
├── module_01_vectors_matrices/
│   ├── README.md               # Module overview
│   ├── theory/                 # Theoretical content
│   │   └── 01_vectors_matrices_theory.md
│   ├── labs/                   # Hands-on implementations
│   │   ├── lab_01_vector_operations.py
│   │   ├── lab_02_matrix_basics.py
│   │   └── lab_03_vector_spaces.py
│   ├── knowledge_checks/       # Quiz questions
│   │   └── kc_01_vectors_matrices.md
│   ├── challenges/             # Coding challenges
│   │   ├── challenge_01_easy.py
│   │   ├── challenge_02_medium.py
│   │   └── challenge_03_hard.py
│   ├── solutions/              # Complete solutions
│   │   ├── solution_lab_01.py
│   │   ├── solution_lab_02.py
│   │   ├── solution_lab_03.py
│   │   └── solution_challenges.py
│   └── further_reading.md
├── module_02_matrix_operations/
│   └── ... (same structure)
├── module_03_eigenvalues_eigenvectors/
│   └── ... (same structure)
├── module_04_matrix_decompositions/
│   └── ... (same structure)
├── module_05_derivatives_gradients/
│   └── ... (same structure)
├── module_06_chain_rule_backprop/
│   └── ... (same structure)
├── module_07_probability_distributions/
│   └── ... (same structure)
└── module_08_bayes_statistics/
    └── ... (same structure)
```

---

## 🛠️ Technical Requirements

### Software
- Python 3.10+
- NumPy 1.24+
- Matplotlib 3.7+
- Jupyter Notebook/Lab (optional)

### Installation

```bash
# Create virtual environment
python -m venv math-env
math-env\Scripts\activate  # Windows
source math-env/bin/activate  # Linux/Mac

# Install dependencies
pip install numpy matplotlib jupyterlab pytest
```

### Running Labs

```bash
# Run a specific lab
python curriculum/tracks/01_mathematics/module_01_vectors_matrices/labs/lab_01_vector_operations.py

# Run all tests for a module
pytest curriculum/tracks/01_mathematics/module_01_vectors_matrices/
```

---

## 📖 How to Use This Track

### Recommended Workflow

1. **Read Theory**: Start with the theory content in each module
2. **Study Examples**: Work through code examples line by line
3. **Complete Labs**: Implement labs from scratch (solutions available)
4. **Test Knowledge**: Complete knowledge check questions
5. **Solve Challenges**: Attempt coding challenges at your level
6. **Review Solutions**: Compare with provided solutions
7. **Explore Further**: Dive into recommended readings

### Bloom's Taxonomy Alignment

Each module targets multiple cognitive levels:

| Level | Description | Assessment |
|-------|-------------|------------|
| Remember | Recall definitions, formulas | Knowledge checks |
| Understand | Explain concepts in own words | Theory questions |
| Apply | Use concepts in new situations | Labs |
| Analyze | Break down complex problems | Medium challenges |
| Evaluate | Judge quality of solutions | Code reviews |
| Create | Build original implementations | Hard challenges |

---

## 🔗 Connections to Other Tracks

This mathematics track is foundational for:

- **Track 02**: Machine Learning Algorithms
- **Track 03**: Deep Learning & Neural Networks
- **Track 04**: Natural Language Processing
- **Track 05**: Computer Vision
- **Track 06**: Reinforcement Learning

### Prerequisite Map

```
Module 01-02 ──┬──> Module 03-04 ──> ML Algorithms (PCA, SVD)
               │
               ├──> Module 05-06 ──> Neural Networks (Backprop)
               │
               └──> Module 07-08 ──> Probabilistic ML
```

---

## 📊 Assessment Strategy

### Formative Assessment (Ongoing)
- Knowledge check questions after each section
- Lab completion with automated tests
- Code quality reviews

### Summative Assessment (End of Module)
- Coding challenges (3 difficulty levels)
- Written explanations of key concepts
- Integration projects combining multiple modules

### Grading Rubric

| Component | Weight | Criteria |
|-----------|--------|----------|
| Labs | 40% | Correctness, completeness, code quality |
| Challenges | 30% | Problem-solving, optimization, creativity |
| Knowledge Checks | 20% | Accuracy, understanding |
| Participation | 10% | Discussion, peer review |

---

## 🎯 Module Quick Links

| Module | README | Theory | Labs | Challenges |
|--------|--------|--------|------|------------|
| 01 | [Link](module_01_vectors_matrices/README.md) | [Link](module_01_vectors_matrices/theory/) | [Link](module_01_vectors_matrices/labs/) | [Link](module_01_vectors_matrices/challenges/) |
| 02 | [Link](module_02_matrix_operations/README.md) | [Link](module_02_matrix_operations/theory/) | [Link](module_02_matrix_operations/labs/) | [Link](module_02_matrix_operations/challenges/) |
| 03 | [Link](module_03_eigenvalues_eigenvectors/README.md) | [Link](module_03_eigenvalues_eigenvectors/theory/) | [Link](module_03_eigenvalues_eigenvectors/labs/) | [Link](module_03_eigenvalues_eigenvectors/challenges/) |
| 04 | [Link](module_04_matrix_decompositions/README.md) | [Link](module_04_matrix_decompositions/theory/) | [Link](module_04_matrix_decompositions/labs/) | [Link](module_04_matrix_decompositions/challenges/) |
| 05 | [Link](module_05_derivatives_gradients/README.md) | [Link](module_05_derivatives_gradients/theory/) | [Link](module_05_derivatives_gradients/labs/) | [Link](module_05_derivatives_gradients/challenges/) |
| 06 | [Link](module_06_chain_rule_backprop/README.md) | [Link](module_06_chain_rule_backprop/theory/) | [Link](module_06_chain_rule_backprop/labs/) | [Link](module_06_chain_rule_backprop/challenges/) |
| 07 | [Link](module_07_probability_distributions/README.md) | [Link](module_07_probability_distributions/theory/) | [Link](module_07_probability_distributions/labs/) | [Link](module_07_probability_distributions/challenges/) |
| 08 | [Link](module_08_bayes_statistics/README.md) | [Link](module_08_bayes_statistics/theory/) | [Link](module_08_bayes_statistics/labs/) | [Link](module_08_bayes_statistics/challenges/) |

---

## 📚 Recommended References

### Textbooks
1. **Linear Algebra**: Gilbert Strang - "Introduction to Linear Algebra"
2. **Calculus**: James Stewart - "Calculus: Early Transcendentals"
3. **Probability**: Sheldon Ross - "A First Course in Probability"
4. **ML Math**: Marc Peter Deisenroth - "Mathematics for Machine Learning"

### Online Resources
- [3Blue1Brown Linear Algebra](https://youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [Khan Academy Calculus](https://khanacademy.org/math/calculus-1)
- [StatQuest Statistics](https://youtube.com/c/statquest)

### Research Papers
- Golub & Kahan (1965) - "Calculating the Singular Values..."
- Rumelhart et al. (1986) - "Learning representations by back-propagating errors"

---

## 👥 Contributing

Contributions welcome! Please follow these guidelines:

1. All code must include tests
2. Mathematical proofs must be verified
3. Visualizations should be interactive where possible
4. Follow PEP 8 style guidelines
5. Include docstrings for all functions

---

## 📄 License

This curriculum is licensed under MIT License. See LICENSE file for details.

---

**Track Maintainer**: AI-Mastery-2026 Curriculum Team  
**Last Updated**: March 30, 2026  
**Version**: 1.0.0
