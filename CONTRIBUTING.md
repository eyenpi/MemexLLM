# Contributing to MemexLLM

Thank you for your interest in contributing to MemexLLM! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Contributions](#making-contributions)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please read it before contributing. We expect all contributors to:
- Be respectful and inclusive
- Use welcoming and inclusive language
- Be collaborative
- Gracefully accept constructive criticism

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/memexllm.git
   cd memexllm
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/eyenpi/memexllm.git
   ```

## Development Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Making Contributions

### Types of Contributions
We welcome various types of contributions:
- Bug fixes
- Feature implementations
- Documentation improvements
- Test coverage improvements
- Performance optimizations

### Before Contributing
1. Check the [issue tracker](https://github.com/eyenpi/memexllm/issues) for existing issues
2. For new features, create an issue to discuss the proposal first
3. For bug fixes, check if there's an existing issue and comment that you're working on it

### Development Workflow
1. Create a new branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "descriptive commit message"
   ```

3. Keep your branch updated with upstream:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

## Pull Request Process

1. Update your fork with the latest upstream changes
2. Ensure all tests pass locally
3. Update documentation if needed
4. Submit a pull request with:
   - Clear description of changes
   - Reference to related issues

### PR Review Process
1. At least one maintainer review is required
2. CI checks must pass
3. Documentation must be updated
4. All discussions must be resolved

## Coding Standards

We follow these coding standards:
- PEP 8 style guide for Python code
- Use [Black](https://github.com/psf/black) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Maximum line length of 88 characters (Black default)
- Meaningful variable and function names
- Clear and concise comments
- Type hints for function arguments and return values

### Code Style Enforcement
Run these commands before committing:
```bash
black .
isort .
mypy .
```

## Testing

### Writing Tests
- Write tests for all new features and bug fixes
- Use pytest for testing
- Maintain or improve code coverage
- Include both unit and integration tests

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=memexllm

# Run specific test file
pytest tests/test_specific.py
```

## Documentation

### Writing Documentation
- Use clear and concise language
- Include code examples
- Document all public APIs
- Update the changelog
- Add docstrings to all public functions and classes

## Questions or Need Help?

Feel free to:
- Open an issue for questions
- Join our community discussions
- Reach out to maintainers

Thank you for contributing to MemexLLM! 