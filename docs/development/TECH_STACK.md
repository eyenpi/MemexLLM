# MemexLLM Technology Stack

## Core Technologies

### Programming Language
- Python 3.10+ (Primary language)
- Type hints and static type checking with MyPy

### Project Management & Build Tools
- Poetry (Dependency management and packaging)
- pyproject.toml (Project configuration)
- Pre-commit hooks for code quality

## Dependencies

### Core Dependencies
- pydantic (^2.0.0) - Data validation and settings management
- typing-extensions (^4.0.0) - Additional typing features

### Storage Backends
- aiosqlite (^0.19.0) - Async SQLite support (Optional)
- Support planned for:
  - MongoDB
  - Redis
  - PostgreSQL

### LLM Integrations
- OpenAI (^1.63.0)
- Planned support for:
  - Anthropic
  - LiteLLM

## Development Tools

### Testing
- pytest (^7.0.0) - Testing framework
- pytest-cov - Code coverage reporting
- pytest-asyncio - Async test support

### Code Quality
- black (^25.1.0) - Code formatting
- isort (^6.0.0) - Import sorting
- mypy (1.15.0) - Static type checking

### Documentation
- Sphinx (^6.2.1) - Documentation generator
- sphinx-rtd-theme - Read the Docs theme
- sphinx-autodoc-typehints - Type hints in documentation
- m2r2 - Markdown to reStructuredText conversion

## Development Environment

### Code Style
- PEP 8 compliant
- 88 character line length (Black configuration)
- Strict type checking with MyPy
- Automated code formatting with Black and isort

### CI/CD
- GitHub Actions for continuous integration
- Automated testing and code coverage reporting
- Code quality checks with pre-commit hooks

## Testing Strategy
- Unit tests with pytest
- Integration tests marked with custom markers
- Coverage reporting with pytest-cov
- Automated test running in CI pipeline

### Scalability Considerations
- Async I/O support for better performance
- Modular architecture for easy extensions
- Support for distributed systems
- Enterprise-grade feature planning 