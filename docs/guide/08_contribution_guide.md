# Guide: Contribution & Development

Thank you for your interest in contributing to the `AI-Mastery-2026` project. This guide provides detailed instructions for setting up your development environment, following our coding standards, and submitting your contributions.

## 1. Core Philosophy

Contributions should align with the project's **White-Box Approach**. This means new algorithms or components should, where feasible:
1.  Be justified by their mathematical foundations.
2.  Be implemented from scratch (or with minimal abstractions) to clearly demonstrate the mechanics.
3.  Be accompanied by tests and documentation.
4.  Consider production implications (e.g., performance, monitoring).

## 2. Setting Up Your Development Environment

### Step 2.1: Fork and Clone

First, create a fork of the main repository on GitHub. Then, clone your fork to your local machine:

```bash
git clone https://github.com/YourUsername/AI-Mastery-2026.git
cd AI-Mastery-2026
```

### Step 2.2: Create and Activate Environment

We recommend using a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the environment
# On Windows: .venv\Scripts\activate
# On macOS/Linux: source .venv/bin/activate
```

### Step 2.3: Install Development Dependencies

The `Makefile` includes a target to install all necessary dependencies for both running the project and developing it (including testing and linting tools).

```bash
make install
```
*(Note: The `CONTRIBUTING.md` mentions `install-dev`, but the provided `Makefile` consolidates this into the `install` target and the dependencies are in `requirements.txt`.)*

### Step 2.4: Create a New Branch

Create a new branch for your feature or bugfix. Use a descriptive name.

```bash
# Example for a new feature
git checkout -b feature/from-scratch-svm

# Example for a bug fix
git checkout -b fix/pca-inverse-transform
```

## 3. Code Quality and Standards

We use a suite of tools to maintain code quality and a consistent style. The `Makefile` provides commands to run these tools easily.

### Formatting

The project uses `black` for code formatting and `isort` for organizing imports. Before committing your code, always run the formatter:

```bash
make format
```

To check if your code is formatted correctly without applying changes, run:
```bash
make format-check
```

### Linting and Type Checking

We use `flake8` for general linting and `mypy` for static type checking. All new code must be fully type-hinted.

To run all checks:
```bash
make lint
```
This will report any style violations or type errors in your code.

### Docstring Standards

Clear documentation is critical. All public functions and classes must have a docstring that includes:
1.  A brief, one-line summary.
2.  (If applicable) The mathematical definition or formula.
3.  An `Args` section describing the parameters.
4.  A `Returns` section describing the output.
5.  A simple `Example` of how to use the function.

**Example:**
```python
def cosine_similarity(v1: Vector, v2: Vector) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Mathematical Definition:
        cos(θ) = (a · b) / (||a|| × ||b||)
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        Cosine similarity score
    """
    # ... implementation ...
```

## 4. Testing

All new features must be accompanied by unit tests. The tests are located in the `tests/` directory.

### Running Tests

*   **Run all tests:**
    ```bash
    make test
    ```

*   **Run tests with coverage:**
    To ensure your changes are well-tested, run the coverage command. This will report the percentage of your code that is covered by tests.
    ```bash
    make test-cov
    ```
    Open the generated `htmlcov/index.html` file in a browser to see a detailed line-by-line report.

*   **Run a specific test file:**
    ```bash
    make test-file FILE=tests/test_your_new_feature.py
    ```

## 5. Submitting a Pull Request

Once your changes are complete, tested, and linted:

1.  **Commit your changes:**
    Use a clear and descriptive commit message.
    ```bash
    git commit -m "feat: Implement Support Vector Machine from scratch"
    ```

2.  **Push to your fork:**
    ```bash
    git push origin feature/from-scratch-svm
    ```

3.  **Open a Pull Request:**
    Go to the original repository on GitHub. You will see a prompt to open a Pull Request from your new branch.

4.  **Describe your PR:**
    *   Write a clear title and description of the changes you have made.
    *   Reference any related issues (e.g., "Closes #42").
    *   Ensure that all automated checks (CI) are passing.

Thank you for helping to improve the `AI-Mastery-2026` toolkit!
