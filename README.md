# MemexLLM

[![CI](https://github.com/eyenpi/memexllm/actions/workflows/ci.yml/badge.svg)](https://github.com/eyenpi/memexllm/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/eyenpi/memexllm/branch/main/graph/badge.svg?token=7C386MR8T9)](https://codecov.io/gh/eyenpi/memexllm)
[![PyPI version](https://badge.fury.io/py/memexllm.svg)](https://badge.fury.io/py/memexllm)
[![Python versions](https://img.shields.io/pypi/pyversions/memexllm.svg)](https://pypi.org/project/memexllm/)

## Overview

MemexLLM is a Python library for managing and storing LLM conversations. It provides a flexible and extensible framework for history management, storage, and retrieval of conversations.

## Installation

```bash
pip install memexllm
```

## Quick Usage

```python
from memexllm.storage import MemoryStorage
from memexllm.algorithms import FIFOAlgorithm
from memexllm.core import HistoryManager

# Initialize components
storage = MemoryStorage()
algorithm = FIFOAlgorithm(max_messages=100)
history_manager = HistoryManager(storage=storage, algorithm=algorithm)

# Create a conversation thread
thread = history_manager.create_thread(metadata={"user": "alice"})

# Add messages
history_manager.add_message(
    thread_id=thread.id,
    content="Hello, how can I help you today?",
    role="assistant"
)

# Retrieve conversation
thread = history_manager.get_thread(thread.id)
for msg in thread.messages:
    print(f"{msg.role}: {msg.content}")
```

For more examples, check out the [examples](examples/) directory.

## Development

### Setup

1. Install pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

This will automatically format your code with black and isort before each commit.

### Running Tests

```bash
pytest tests/
```
