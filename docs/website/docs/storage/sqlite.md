# SQLite Storage

The SQLite Storage backend provides persistent storage using SQLite, making it perfect for production applications that run on a single machine.

## Features

- **Persistent**: Data survives application restarts
- **ACID Compliant**: Reliable data storage
- **No Server**: Self-contained in a single file
- **Message Limiting**: Optional limit on messages per thread
- **Type Safety**: Full type hints support

## Installation

```bash
pip install memexllm[sqlite]
```

## Basic Usage

```python
from memexllm.storage import SQLiteStorage
from memexllm.algorithms import FIFOAlgorithm
from memexllm.core import HistoryManager

# Create storage instance
storage = SQLiteStorage("conversations.db")

# Keep last 1000 messages in the database
storage = SQLiteStorage(
    db_path="conversations.db",
    max_messages=1000
)

# Create history manager with FIFO algorithm
manager = HistoryManager(
    storage=storage,
    algorithm=FIFOAlgorithm(max_messages=50)  # Use last 50 messages for context
)
```

## When to Use

✅ **Good For**:
- Production applications
- Single-machine deployments
- Desktop applications
- CLI tools
- Web applications
- Reliable data storage

❌ **Not Good For**:
- Distributed applications
- High-concurrency workloads
- Cloud-native applications
- Serverless environments

## Best Practices

1. **File Location**:
   - Use absolute paths in production
   - Store in appropriate data directory
   - Set proper file permissions

2. **Message Limits**:
   - Set `max_messages` to control database size
   - Consider your application's needs
   - Monitor database growth

3. **Backup**:
   - Regularly backup the database file
   - Use SQLite's backup API for live backups
   - Test backup restoration

## Next Steps

- Learn about [context algorithms](../algorithms/overview.md)
- Create [custom storage](./custom.md)
- See [example applications](../examples/simple_chatbot.md) 