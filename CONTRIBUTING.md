# Contributing to AI-Mastery-2026

Thank you for your interest in contributing to the AI-Mastery-2026 project! This toolkit is designed to help engineers understand AI from mathematical foundations to production systems using the "white-box" approach.

## Getting Started

1. Fork the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Philosophy

This project follows the "white-box" approach:
- Understand the mathematical foundations before using abstractions
- Implement algorithms from scratch with NumPy
- Use libraries like sklearn/PyTorch knowing what happens underneath
- Consider production aspects throughout the development process

## Code Style

- Python 3.10+ compatible
- Type hints for all functions
- 100 character line limit
- Black formatting with 100 character line length
- MyPy type checking

## Documentation Standards

Each function should include:
1. Brief description
2. Mathematical definition (using Unicode symbols where applicable)
3. Args and Returns sections
4. Example usage

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build
2. Update the README.md with details of changes to the interface
3. Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would represent
4. You may merge the Pull Request in once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you