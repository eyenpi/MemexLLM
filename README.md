# MemexLLM

[![CI](https://github.com/eyenpi/memexllm/actions/workflows/ci.yml/badge.svg)](https://github.com/eyenpi/memexllm/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/eyenpi/memexllm/branch/main/graph/badge.svg?token=7C386MR8T9)](https://codecov.io/gh/eyenpi/memexllm)
[![PyPI version](https://badge.fury.io/py/memexllm.svg)](https://badge.fury.io/py/memexllm)
[![Python versions](https://img.shields.io/pypi/pyversions/memexllm.svg)](https://pypi.org/project/memexllm/)

## Overview

MemexLLM is a Python library for managing and storing LLM conversations. It provides a flexible and extensible framework for history management, storage, and retrieval of conversations.

## Quick Start

### Installation

```bash
pip install memexllm  # Basic installation
pip install memexllm[openai]  # With OpenAI support
```

### Basic Usage

```python
from memexllm.storage import MemoryStorage
from memexllm.algorithms import FIFOAlgorithm
from memexllm.core import HistoryManager

# Initialize components
storage = MemoryStorage()
algorithm = FIFOAlgorithm(max_messages=100)
history_manager = HistoryManager(storage=storage, algorithm=algorithm)

# Create a conversation thread
thread = history_manager.create_thread()

# Add messages
history_manager.add_message(
    thread_id=thread.id,
    content="Hello, how can I help you today?",
    role="assistant"
)
```

## Documentation

For detailed documentation, including:
- Complete API reference
- Advanced usage examples
- Available storage backends
- Contributing guidelines
- Feature roadmap

Visit our documentation at: https://eyenpi.github.io/MemexLLM/

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

## License

This project is licensed under the MIT License.