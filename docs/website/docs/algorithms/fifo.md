# FIFO Algorithm

The FIFO (First-In-First-Out) algorithm is a simple but effective way to manage conversation context. It keeps the most recent messages while respecting token limits.

## Features

- **Simple**: First-in, first-out message selection
- **Predictable**: Always keeps most recent messages
- **Efficient**: No complex computations
- **Configurable**: Adjustable message window size
- **Type Safety**: Full type hints support

## Installation

FIFO Algorithm is included with the base package:

```bash
pip install memexllm
```

## Basic Usage

```python
from memexllm.algorithms import FIFOAlgorithm

# Create algorithm instance
algorithm = FIFOAlgorithm()

# Optional: limit context window size
algorithm = FIFOAlgorithm(max_messages=50)
```

## Use with OpenAI

```python
from openai import OpenAI
from memexllm.storage import SQLiteStorage
from memexllm.algorithms import FIFOAlgorithm
from memexllm.integrations.openai import with_history
from memexllm.history import HistoryManager
# Create client with FIFO algorithm
client = OpenAI()
history_manager = HistoryManager(
    storage=SQLiteStorage("chat.db"),
    algorithm=FIFOAlgorithm(max_messages=50)
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
- Most chat applications
- Simple chatbots
- When order matters
- General conversation
- Learning MemexLLM

❌ **Not Good For**:
- Complex context needs
- Semantic relevance
- Token optimization
- Time-based selection

## Best Practices

1. **Message Window Size**:
   - Consider model token limits
   - Account for message lengths
   - Include buffer for system messages

2. **System Messages**:
   - Always included in context
   - Count towards message limit
   - Keep them concise

3. **Performance**:
   - Efficient by default
   - No optimization needed
   - Minimal memory usage

## Next Steps

- Learn about [custom algorithms](./custom.md)
- See [example applications](../examples/simple_chatbot.md)
- Check out other [algorithms](./overview.md) 