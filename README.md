<h1 align="center">🐟 MemexLLM</h1>

<p align="center">
  <img src="https://raw.githubusercontent.com/eyenpi/memexllm/main/docs/website/static/img/memex_logo.svg" alt="MemexLLM Logo" width="200"/>
</p>

<p align="center">
  <a href="https://github.com/eyenpi/memexllm/actions/workflows/ci.yml"><img src="https://github.com/eyenpi/memexllm/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/eyenpi/memexllm"><img src="https://codecov.io/gh/eyenpi/memexllm/branch/main/graph/badge.svg?token=7C386MR8T9" alt="codecov"></a>
  <a href="https://badge.fury.io/py/memexllm"><img src="https://badge.fury.io/py/memexllm.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/memexllm/"><img src="https://img.shields.io/pypi/pyversions/memexllm.svg" alt="Python versions"></a>
</p>

## Overview

MemexLLM is a Python library for managing and storing LLM conversations. It provides a flexible and extensible framework for history management, storage, and retrieval of conversations.

## Features

- **Drop-in Integrations**: Add conversation management to your LLM applications with **zero code changes** using our provider integrations
- **Flexible Storage**: Choose from memory, SQLite, or bring your own storage backend
- **Conversation Management**: Organize, retrieve, and manipulate conversation threads with ease
- **Memory Management Algorithms**: Control conversation context with built-in algorithms (FIFO, summarization, etc.)
- **Provider Agnostic**: Works with OpenAI, Anthropic, and other LLM providers
- **Extensible Architecture**: Build custom storage backends and memory management algorithms

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

### Zero-Code-Change Integration

Add conversation management to your OpenAI application with no code changes:

```python
from openai import OpenAI
from memexllm.integrations.openai import with_history
from memexllm.storage import MemoryStorage
from memexllm.algorithms import FIFOAlgorithm

# Initialize your OpenAI client as usual
client = OpenAI(api_key="your-api-key")

# Add conversation memory with history management
storage = MemoryStorage()
algorithm = FIFOAlgorithm(max_messages=100)
history_manager = HistoryManager(storage=storage, algorithm=algorithm)
client = with_history(history_manager=history_manager)(client)

# Use the client as you normally would - conversations are now managed automatically
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello, who are you?"}]
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