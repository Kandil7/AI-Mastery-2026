# Senior Engineering Git Practices for RAG Engine Contributions

## 🎯 Overview

This guide outlines the git and commit practices expected when contributing to the RAG Engine Mini project. These practices ensure a clean, educational, and maintainable codebase that serves both production and learning purposes.

## 📚 Fundamental Principles

### 1. Every Commit Tells a Story

Each commit should represent a complete, logical change that can stand alone:

- **Clear Purpose**: Each commit addresses a single, well-defined problem
- **Complete Solution**: All necessary changes for that problem are included
- **Self-Documenting**: The commit message explains the "why" not just the "what"
- **Verifiable**: The change can be tested and validated independently

### 2. Educational Value First

Since this is an educational project, commits should be instructive:

- Include educational comments explaining complex concepts
- Ensure code is understandable to learners at different levels
- Provide context for design decisions in commit messages
- Make refactoring changes educational, not just functional

## 🧱 Atomic Commit Guidelines

### What Makes a Commit Atomic

An atomic commit should:

1. **Address One Concern**: Focus on a single issue, feature, or bug fix
2. **Be Internally Complete**: Include all related changes that make sense together
3. **Pass All Tests**: Function correctly in isolation
4. **Have Clear Documentation**: Include comments and commit messages that explain the change

### Examples of Atomic Commits

**Good:**
```
feat(embeddings): add Cohere embedding adapter with error handling

Implements a new embedding adapter for the Cohere API following
the existing adapter pattern. Includes proper error handling and
retry logic for API timeouts.

The adapter supports:
- Text embedding generation
- Batch processing for efficiency
- Retry logic with exponential backoff
- Proper error mapping to internal exceptions

Resolves #123
```

**Bad:**
```
add cohere adapter and fix some bugs and refactor things
```

## 🛠️ Practical Commit Workflow

### Step 1: Prepare Your Work Environment

```bash
# Ensure you're on the latest main branch
git checkout main
git pull origin main

# Create a feature branch with descriptive name
git checkout -b feat/query-intent-classification
# or
git checkout -b fix/embedding-cache-invalidatation
```

### Step 2: Make Focused Changes

Work on one logical change at a time:

1. **Implement the core functionality**
2. **Add tests for your changes**
3. **Update documentation as needed**
4. **Verify everything works together**

### Step 3: Stage Changes Thoughtfully

```bash
# Review your changes
git diff

# Stage related changes together
git add src/adapters/embeddings/cohere_adapter.py
git add tests/unit/test_cohere_adapter.py
git add docs/adr/006-cohere-adapter.md

# Or use patch mode to selectively stage parts of files
git add -p
```

### Step 4: Craft Your Commit Message

```bash
git commit
```

Use the template:
```
<type>(<scope>): <short description>

<thorough explanation of WHY this change is made>

<optional: additional context or migration notes>

Resolves #<issue-number>
```

## 📝 Commit Type Guidelines

### Type Definitions

- `feat`: New feature for the user (also for new educational content)
- `fix`: Bug fix for the user (also for educational corrections)
- `docs`: Documentation changes (educational content, ADRs, guides)
- `style`: Formatting changes that don't affect meaning
- `refactor`: Code changes that neither fix bugs nor add features
- `perf`: Performance improvements
- `test`: Adding or modifying tests
- `chore`: Other changes that don't modify src or test files

### Scope Examples

- `api`: Changes to API layer
- `domain`: Changes to domain layer
- `application`: Changes to application services
- `adapters`: Changes to adapters layer
- `tests`: Changes to testing infrastructure
- `docs`: Changes to documentation
- `config`: Changes to configuration
- `ci`: Changes to CI/CD configuration

## 📘 Educational Commit Examples

### Adding Educational Content

```
docs(educational): add comprehensive guide to hybrid search implementation

Creates detailed educational guide explaining how hybrid search works
in the RAG Engine Mini, with mathematical explanations and code examples.

The guide covers:
- RRF (Reciprocal Rank Fusion) algorithm
- How vector and keyword search are combined
- Performance characteristics of each approach
- When to use hybrid vs pure approaches

Part of educational enhancement initiative #456
```

### Adding a New Adapter

```
feat(adapters): implement Anthropic Claude adapter with safety features

Adds support for Anthropic's Claude model with their safety features
integrated into our LLM adapter interface.

Implementation includes:
- Async generation methods
- Safety prompt handling
- Token usage tracking
- Error mapping to internal exceptions

Educational: Demonstrates how to integrate a new LLM provider
following our adapter pattern.

Resolves #789
```

### Refactoring for Education

```
refactor(application): simplify retrieval service for educational clarity

Restructures the retrieval service to make the hybrid search logic
more transparent for learners studying the implementation.

Changes include:
- Breaking down the complex retrieve method into smaller parts
- Adding detailed comments explaining each step
- Extracting the RRF fusion logic to a dedicated helper

The functionality remains identical but the code is more educational.

Improve #654
```

## 🔧 Handling Complex Features

### Multi-Commit Features

For complex features, break them into logical commits:

1. **Interface/Contract Changes**
2. **Core Implementation**
3. **Tests**
4. **Documentation**
5. **Integration**

### Example Multi-Commit Sequence

**Commit 1:**
```
feat(domain): add QueryIntent enum and classification result

Defines the domain concepts for query intent classification
to enable intent-aware RAG processing.

This establishes the vocabulary for the feature and provides
a foundation for the implementation.

Part of #112
```

**Commit 2:**
```
feat(adapters): implement rule-based query intent classifier

Creates a rule-based classifier that identifies query intent
using keyword patterns, following the ports and adapters pattern.

This provides the core logic for the intent classification feature
while maintaining clean architecture principles.

Part of #112
```

**Commit 3:**
```
feat(application): add intent-aware RAG service

Implements an enhanced RAG service that uses intent classification
to optimize the retrieval and generation process.

The service adjusts retrieval parameters based on query intent:
- Factual queries: More focused retrieval
- Comparative queries: More diverse retrieval
- Procedural queries: Instructional content prioritization

Part of #112
```

## 🧪 Testing Your Commits

### Before Committing

1. **Run All Tests**:
   ```bash
   pytest
   ```

2. **Check Code Quality**:
   ```bash
   mypy src/
   black --check src/
   flake8 src/
   ```

3. **Verify Documentation**: Ensure your changes include necessary documentation

### Testing Individual Commits

Use `git bisect` to ensure each commit works:

```bash
# Verify a commit works in isolation
git checkout <commit-hash>
pytest  # Should pass
```

## 🔄 Handling Mistakes

### Amending Commits

```bash
# Add more changes to the last commit
git add .
git commit --amend

# Change only the commit message
git commit --amend -m "New message"
```

### Interactive Rebase

```bash
# Clean up commit history before submitting
git rebase -i HEAD~3  # Interact with last 3 commits

# Options in rebase:
# pick: keep the commit
# reword: change commit message
# edit: modify the commit
# squash: combine with previous commit
# drop: remove the commit
```

## 🚀 Pull Request Preparation

### Before Submitting

1. **Squash Minor Commits**: Combine small "fixup" commits
2. **Rebase onto Main**: Keep history linear
3. **Update Branch**: Ensure latest changes from main are included
4. **Final Test**: Verify everything works together

### PR Title and Description

Match your main commit message but add broader context:

```
feat(adapters): implement Anthropic Claude adapter with safety features

This PR adds support for Anthropic's Claude model with their safety 
features integrated into our LLM adapter interface.

Implementation includes:
- Async generation methods
- Safety prompt handling
- Token usage tracking
- Error mapping to internal exceptions

Educational: Demonstrates how to integrate a new LLM provider
following our adapter pattern.

Resolves #789

How to verify:
1. Run `pytest tests/unit/test_claude_adapter.py`
2. Test with a sample query through the API
3. Verify safety features work as expected
```

## 🏗️ Architecture-Specific Commit Patterns

### Domain Layer Changes
- Focus on business logic and entities
- Ensure immutability where appropriate
- Add comprehensive documentation for new concepts

### Application Layer Changes
- Maintain separation of concerns
- Update interfaces appropriately
- Include use case examples

### Adapter Layer Changes
- Follow the existing adapter pattern
- Include error handling and retry logic
- Add tests for external dependencies

### API Layer Changes
- Maintain backward compatibility where possible
- Update API documentation
- Include example requests/responses

## ✅ Senior Engineering Checklist

Before finalizing your commits, verify:

- [ ] Each commit addresses a single logical change
- [ ] Commit messages explain the "why" not just the "what"
- [ ] All tests pass for each commit
- [ ] Code follows existing patterns and conventions
- [ ] New code is properly documented
- [ ] Educational value is maintained or improved
- [ ] Changes are properly scoped (not too big or too small)
- [ ] Security implications are considered
- [ ] Performance impacts are addressed
- [ ] Error handling is comprehensive

## 📚 Continuous Improvement

### Learning from Examples

Study well-crafted commits in the repository:
```bash
git log --oneline -10
git show <commit-hash>
```

### Getting Feedback

- Request detailed code reviews focusing on commit quality
- Participate in commit history reviews
- Contribute to improving commit standards

By following these practices, you'll contribute to a codebase that serves both production needs and educational purposes, creating a valuable resource for current and future developers learning RAG systems.