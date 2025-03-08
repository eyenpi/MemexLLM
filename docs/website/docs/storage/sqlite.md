# SQLite Storage

The SQLite Storage backend provides persistent storage using SQLite, making it perfect for production applications that run on a single machine.

## Features

- **Persistent**: Data survives application restarts
- **ACID Compliant**: Reliable data storage
- **No Server**: Self-contained in a single file
- **Message Limiting**: Optional limit on messages per thread
- **Type Safety**: Full type hints support
- **Comprehensive Error Handling**: Robust error handling for all database operations

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

## Error Handling

The SQLite storage backend provides comprehensive error handling with specific exception types:

- **SQLiteStorageError**: Base exception for all SQLite storage errors
- **DatabaseConnectionError**: Raised when database connection fails
- **DatabaseOperationError**: Raised when a database operation fails
- **DatabaseIntegrityError**: Raised when a database integrity constraint is violated

Example of handling database errors:

```python
from memexllm.storage import SQLiteStorage
from memexllm.storage.sqlite import (
    SQLiteStorageError,
    DatabaseConnectionError,
    DatabaseOperationError,
    DatabaseIntegrityError
)

try:
    storage = SQLiteStorage("conversations.db")
    # Perform operations...
except DatabaseConnectionError as e:
    print(f"Failed to connect to database: {e}")
    # Handle connection error
except DatabaseIntegrityError as e:
    print(f"Database integrity error: {e}")
    # Handle integrity error
except DatabaseOperationError as e:
    print(f"Database operation error: {e}")
    # Handle operation error
except SQLiteStorageError as e:
    print(f"General SQLite storage error: {e}")
    # Handle general storage error
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

4. **Error Handling**:
   - Catch specific exceptions for different error types
   - Implement appropriate recovery strategies
   - Log errors for debugging
   - Consider retry mechanisms for transient errors

5. **Transaction Management**:
   - Use transactions for operations that modify multiple records
   - Implement proper rollback on errors
   - Consider using SQLite's WAL mode for better concurrency

## Next Steps

- Learn about [context algorithms](../algorithms/overview.md)
- Create [custom storage](./custom.md)
- See [example applications](../examples/simple_chatbot.md) 