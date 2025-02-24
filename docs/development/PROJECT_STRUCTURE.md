# MemexLLM Project Structure

This document provides an overview of the MemexLLM project structure and organization.

## Project Overview

MemexLLM is a Python package that provides a framework for managing and integrating LLM (Large Language Model) contexts and memory. The project follows a modular architecture with clear separation of concerns.

## Directory Structure

```
MemexLLM/
├── src/                    # Source code directory
│   └── memexllm/          # Main package
│       ├── algorithms/    # Algorithm implementations
│       ├── core/         # Core functionality
│       ├── integrations/ # External integrations (e.g., OpenAI)
│       ├── storage/      # Storage implementations
│       ├── utils/        # Utility functions
│       ├── types.py      # Type definitions
│       └── __init__.py   # Package initialization
│
├── examples/              # Example implementations and usage
│   ├── basic_usage.py
│   ├── quickstart.py
│   ├── openai_integration.py
│   ├── sqlite_storage.py
│   └── algorithm_storage_demo.py
│
├── tests/                # Test suite
├── docs/                 # Documentation
├── scripts/              # Utility scripts
└── .github/              # GitHub workflows and configurations

## Key Components

### Core Package (src/memexllm)

- **algorithms/**: Contains implementations of various algorithms for context management and memory operations
- **core/**: Core functionality and base classes
- **integrations/**: Integration modules for external services (e.g., OpenAI)
- **storage/**: Different storage backend implementations
- **utils/**: Helper functions and utilities
- **types.py**: Common type definitions

### Examples

The `examples/` directory contains various demonstration scripts showing how to use different features:
- Basic usage patterns
- Integration with OpenAI
- SQLite storage implementation
- Algorithm demonstrations

### Development Tools

- **pyproject.toml**: Project configuration and dependencies
- **poetry.lock**: Locked dependencies
- **.pre-commit-config.yaml**: Pre-commit hook configurations
- **mypy.ini**: Type checking configuration
- **.coveragerc**: Code coverage configuration

## Development Environment

The project uses:
- Poetry for dependency management
- Pre-commit hooks for code quality
- MyPy for type checking
- Pytest for testing
- GitHub Actions for CI/CD