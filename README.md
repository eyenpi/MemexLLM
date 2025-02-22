# MemexLLM

[![CI](https://github.com/eyenpi/memexllm/actions/workflows/ci.yml/badge.svg)](https://github.com/eyenpi/memexllm/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/eyenpi/memexllm/branch/main/graph/badge.svg?token=7C386MR8T9)](https://codecov.io/gh/eyenpi/memexllm)
[![PyPI version](https://badge.fury.io/py/memexllm.svg)](https://badge.fury.io/py/memexllm)
[![Python versions](https://img.shields.io/pypi/pyversions/memexllm.svg)](https://pypi.org/project/memexllm/)

## Overview

MemexLLM is a Python library for managing and storing LLM conversations. It provides a flexible and extensible framework for history management, storage, and retrieval of conversations.

## Installation

Choose the installation option that best suits your needs:

### Basic Installation
```bash
pip install memexllm
```

### OpenAI Installation
```bash
pip install memexllm[openai]
```

### Development Installation
```bash
pip install memexllm[dev]
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
thread = history_manager.create_thread()

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


For more examples check out the [examples](examples/) directory.
## Feature Roadmap

Here are the planned features and improvements for MemexLLM:

### Short-term Goals
- [x] OpenAI integration
- [x] Memory storage backend
- [x] FIFO algorithm
- [ ] Anthropic integration
- [ ] LiteLLM integration
- [ ] MongoDB storage backend support
- [ ] SQLite storage backend support
- [ ] Redis storage backend support
- [ ] PostgreSQL storage backend support
- [ ] Conversation summarization algorithm
- [ ] Other algorithms
- [ ] Export conversations to various formats (JSON, CSV, PDF)

### Medium-term Goals
- [ ] Advanced conversation analytics
- [ ] Integration with popular LLM providers
- [ ] Conversation branching and versioning

### Long-term Goals
- [ ] Distributed storage support
- [ ] Multi-modal conversation support
- [ ] Advanced privacy and security features
- [ ] API Gateway integration
- [ ] Enterprise-grade features

## Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute
1. **Code Contributions**
   - Fork the repository
   - Create a feature branch
   - Write clean, documented, and tested code
   - Submit a pull request

2. **Bug Reports**
   - Use the GitHub issue tracker
   - Include detailed steps to reproduce
   - Provide system information and context

3. **Feature Requests**
   - Open a GitHub issue with the "enhancement" label
   - Describe the feature and its use cases
   - Discuss with the community

4. **Documentation**
   - Help improve documentation
   - Write tutorials and examples
   - Fix typos and clarify explanations

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/eyenpi/memexllm.git
   cd memexllm
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

### Running Tests

```bash
pytest tests/
```

### Code Style
- We follow PEP 8 guidelines
- Use Black for code formatting
- Use isort for import sorting
- Write meaningful commit messages

### Review Process
1. All code changes require tests
2. CI must pass
3. Code review by maintainers
4. Documentation updates if needed