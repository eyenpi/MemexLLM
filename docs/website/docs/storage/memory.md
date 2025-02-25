# Memory Storage

The Memory Storage backend keeps all conversations in memory, making it perfect for development and testing.

## Features

- **In-Memory**: All data stored in memory for fast access
- **No Setup**: Just import and use
- **Thread Support**: Full thread and message management
- **Message Limiting**: Optional limit on messages per thread
- **Type Safety**: Full type hints support

## Installation

Memory Storage is included with the base package:

```bash
pip install memexllm
```

## Basic Usage

```python
from memexllm.storage import MemoryStorage
from memexllm.algorithms import FIFOAlgorithm
from memexllm.core import HistoryManager

# Create storage instance
storage = MemoryStorage()

# Keep last 100 messages in memory
storage = MemoryStorage(max_messages=100)

# Create history manager
manager = HistoryManager(
    storage=storage,
    algorithm=FIFOAlgorithm(max_messages=50)  # Use last 50 messages for context
)
```

## Use with OpenAI

```python
from openai import OpenAI
from memexllm.storage import MemoryStorage
from memexllm.algorithms import FIFOAlgorithm
from memexllm.integrations.openai import with_history
from memexllm.history import HistoryManager

# Create client with memory storage and FIFO algorithm
client = OpenAI()

history_manager = HistoryManager(
    storage=MemoryStorage(),
    algorithm=FIFOAlgorithm(max_messages=50)  # Use last 50 messages for context
)

client = with_history(history_manager=history_manager)(client)

# Use as normal
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## When to Use

✅ **Good For**:
- Development and testing
- Prototyping
- Small applications
- Learning MemexLLM
- Unit tests

❌ **Not Good For**:
- Production use
- Data persistence
- Large conversations
- Multiple processes
- Critical data

## Next Steps

- Use [SQLite Storage](./sqlite.md) for persistence
- Learn about [context algorithms](../algorithms/overview.md)
- See [example applications](../examples/simple_chatbot.md) 