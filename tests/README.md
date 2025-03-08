# MemexLLM Test Suite

This directory contains the test suite for MemexLLM. The tests are organized into three main categories:

## Test Structure

```
tests/
├── conftest.py            # Global test fixtures and configuration
├── unit/                  # Unit tests
│   ├── conftest.py        # Unit test fixtures
│   ├── core/              # Tests for core functionality
│   ├── storage/           # Tests for storage modules
│   ├── algorithms/        # Tests for algorithms
│   ├── integrations/      # Unit tests for integrations
│   └── utils/             # Tests for utility functions
├── integration/           # Integration tests
│   ├── conftest.py        # Integration test fixtures
│   ├── openai/            # OpenAI integration tests
│   └── other/             # Other integration tests
└── performance/           # Performance tests
    ├── conftest.py        # Performance test fixtures
    ├── storage/           # Storage performance tests
    └── algorithms/        # Algorithm performance tests
```

## Running Tests

### Running All Tests

```bash
pytest
```

### Running Specific Test Categories

```bash
# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run performance tests only
pytest tests/performance/
```

### Running Tests for Specific Modules

```bash
# Run tests for core functionality
pytest tests/unit/core/

# Run tests for storage
pytest tests/unit/storage/

# Run OpenAI integration tests
pytest tests/integration/openai/
```

## Writing Tests

### Unit Tests

Unit tests should be placed in the appropriate subdirectory under `tests/unit/` based on the module they test. Each test file should be named `test_*.py`.

### Integration Tests

Integration tests should be placed in the appropriate subdirectory under `tests/integration/` based on the integration they test. Each test file should be named `test_*.py`.

### Performance Tests

Performance tests should be placed in the appropriate subdirectory under `tests/performance/` based on the module they test. Each test file should be named `test_*_performance.py`. 