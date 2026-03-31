# AI-Mastery-2026 Documentation

[![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg)](https://kandil7.github.io/AI-Mastery-2026/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Complete documentation for AI-Mastery-2026 - The Ultimate AI Engineering Toolkit.

---

## Quick Start

### View Documentation Online

Visit [https://kandil7.github.io/AI-Mastery-2026/](https://kandil7.github.io/AI-Mastery-2026/) for the latest documentation.

### Build Locally

```bash
# Install documentation dependencies
pip install mkdocs mkdocs-material mkdocstrings[python]

# Serve locally (http://localhost:8000)
mkdocs serve

# Build static site
mkdocs build
```

---

## Documentation Structure

### Getting Started
- [Introduction](docs/00_introduction/01_getting_started.md)
- [Installation](docs/00_introduction/01_user_guide.md)
- [Quick Start](docs/00_introduction/QUICK_START.md)

### Learning
- [Learning Roadmap](docs/01_learning_roadmap/README.md)
- [Student Guide](docs/01_student_guide/README.md)
- [Instructor Guide](docs/02_instructor_guide/README.md)

### Technical Reference
- [Architecture](docs/architecture/README.md)
- [API Reference](docs/api/README.md)
- [Technical Reference](docs/03_technical_reference/README.md)

### Contributing
- [Contributing Guide](CONTRIBUTING.md)
- [Code Style](docs/guides/code-style.md)
- [Migration Guide](MIGRATION_GUIDE.md)

---

## Navigation

```yaml
nav:
  - Home: index.md
  - Getting Started:
    - Introduction: 00_introduction/01_getting_started.md
    - Installation: 00_introduction/01_user_guide.md
    - Quick Start: 00_introduction/QUICK_START.md
  - Learning:
    - Roadmap: 01_learning_roadmap/README.md
    - Student Guide: 01_student_guide/README.md
    - Instructor Guide: 02_instructor_guide/README.md
  - Technical:
    - Architecture: architecture/README.md
    - API Reference: api/README.md
    - Technical Reference: 03_technical_reference/README.md
  - Contributing:
    - Guide: CONTRIBUTING.md
    - Code Style: guides/code-style.md
    - Migration: MIGRATION_GUIDE.md
  - Project Info:
    - Changelog: CHANGELOG.md
    - Security: SECURITY.md
    - Technical Debt: TECHNICAL_DEBT.md
```

---

## For Maintainers

### Deploy Documentation

```bash
# Deploy to GitHub Pages
mkdocs gh-deploy --force
```

### Update Navigation

Edit `mkdocs.yml` to update the navigation structure.

### Add New Pages

1. Create markdown file in appropriate `docs/` subdirectory
2. Add to navigation in `mkdocs.yml`
3. Build and verify locally
4. Deploy to GitHub Pages

---

## License

This documentation is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Last Updated:** March 31, 2026
