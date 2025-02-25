# OpenAI Integration

MemexLLM's OpenAI integration adds conversation memory to your existing OpenAI applications with minimal code changes.

## Installation

```bash
pip install memexllm[openai]
```

## Basic Usage

Here's how to enhance your OpenAI client with conversation memory using the FIFO algorithm:

```python
from openai import OpenAI
from memexllm.integrations.openai import with_history
from memexllm.history import HistoryManager
from memexllm.storage import SQLiteStorage
from memexllm.algorithms import FIFOAlgorithm

# Your existing OpenAI code
client = OpenAI()

# Create history manager
history_manager = HistoryManager(
    storage=SQLiteStorage("chat.db"),
    algorithm=FIFOAlgorithm(max_messages=50)  # Keep last 50 messages
)

# Add conversation memory with FIFO algorithm
client = with_history(history_manager=history_manager)(client)

# Use the client as normal - conversation history is handled automatically
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}],
    thread_id="my-thread"  # Optional, will be created if not provided
)
```

That's it! The only changes needed are:
1. Import the `with_history` wrapper and `FIFOAlgorithm`
2. Choose a storage backend
3. Configure the FIFO algorithm
4. Wrap your OpenAI client

## Async Support

The integration works the same way with async clients:

```python
from openai import AsyncOpenAI
from memexllm.integrations.openai import with_history
from memexllm.storage import SQLiteStorage
from memexllm.algorithms import FIFOAlgorithm

# Your existing async OpenAI code
client = AsyncOpenAI()

# Add conversation memory with FIFO algorithm
client = with_history(
    storage=SQLiteStorage("chat.db"),
    algorithm=FIFOAlgorithm(max_messages=50)  # Keep last 50 messages
)(client)

# Use as normal
response = await client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}],
    thread_id="my-thread"
)
```

## Next Steps

- Learn about [storage options](../storage/overview.md)
- Explore [context algorithms](../algorithms/overview.md)
- See [example applications](../examples/simple_chatbot.md) 