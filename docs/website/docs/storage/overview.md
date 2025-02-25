# Storage Overview

MemexLLM provides multiple storage backends for persisting conversation history. Choose the storage that best fits your needs or create your own custom storage.

## Available Storage Backends

### Memory Storage
- In-memory storage for development and testing
- No persistence across application restarts
- Fast and simple
- [Learn more about Memory Storage](./memory.md)

### SQLite Storage
- File-based storage using SQLite
- Persistent across application restarts
- Great for single-machine applications
- [Learn more about SQLite Storage](./sqlite.md)

## Coming Soon

We're working on additional storage backends:

### PostgreSQL Storage
- Full SQL database support
- Great for distributed applications

### Redis Storage
- Fast in-memory storage with persistence
- Great for high-performance applications

### MongoDB Storage
- Document-based storage
- Flexible schema for complex metadata

### S3 Storage
- Cloud storage support
- Great for serverless applications

## Storage Features

All storage backends in MemexLLM provide:

- **Thread Management**: Create, read, update, and delete conversation threads
- **Message Storage**: Store and retrieve messages within threads
- **Metadata Support**: Attach custom metadata to threads and messages
- **Message Limiting**: Control the number of messages stored per thread
- **Type Safety**: Full type hints support

## Custom Storage

Need a different storage backend? You can create your own:
- Implement the storage interface
- Full control over persistence and retrieval
- [Learn how to create custom storage](./custom.md)

## Next Steps

- Try [Memory Storage](./memory.md) for development
- Use [SQLite Storage](./sqlite.md) for production
- Learn to [create custom storage](./custom.md)
- See [example applications](../examples/simple_chatbot.md) 