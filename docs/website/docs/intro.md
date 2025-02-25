# Introduction

MemexLLM is a Python library that enhances Large Language Models (LLMs) with seamless conversation management. It provides a simple way to add persistent storage and smart context management to your existing LLM applications.

## Core Components

MemexLLM is built around three main components:

### 1. Integrations
Drop-in integrations with popular LLM providers that automatically handle conversation management. Just wrap your existing client and everything works as before, but with added conversation memory.

### 2. Storage Backends
Flexible storage options for your conversations, from in-memory storage for development to SQLite for production. Choose the storage that fits your needs or create your own.

### 3. Context Algorithms
Smart algorithms that manage how conversation history is presented to the LLM, ensuring optimal context while respecting token limits.

## Quick Start

```bash
pip install memexllm # OR install with integrations
pip install memexllm[openai] # install with OpenAI integration OR
pip install memexllm[openai,sqlite] # install with OpenAI and SQLite support
```

```python
from openai import OpenAI
from memexllm.storage import SQLiteStorage
from memexllm.algorithms import FIFOAlgorithm
from memexllm.integrations.openai import with_history
from memexllm.history import HistoryManager

# Create enhanced OpenAI client
client = OpenAI()
history_manager = HistoryManager(storage=SQLiteStorage("chat.db"), algorithm=FIFOAlgorithm())
client = with_history(history_manager=history_manager)(client)

# Use as normal - conversation history is handled automatically
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}],
    thread_id="my-thread"  # Optional, will be created if not provided
)
```

## Key Features

- **Zero-Change Integration**: Add conversation management without changing your existing code
- **Persistent Memory**: Conversations survive application restarts
- **Smart Context**: Automatically manage conversation history within token limits
- **Thread Organization**: Keep conversations separate and organized
- **Async Support**: Works with both sync and async clients
- **Type Safety**: Full type hints support for better development experience

## Next Steps

- Learn about available [integrations](./integrations/overview.md)
- Explore [storage options](./storage/overview.md)
- Understand [context algorithms](./algorithms/overview.md)
- See [example applications](./examples/simple_chatbot.md)
