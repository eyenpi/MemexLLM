# MemexLLM Application Flow

## 1. Overview

MemexLLM is a Python library designed to provide robust conversation management capabilities for Large Language Models (LLMs). It offers a flexible and extensible framework for storing, managing, and retrieving LLM conversations.

## 2. Core Components

### 2.1 Core Classes
1. **HistoryManager**
   - Central orchestrator for conversation management
   - Manages threads and messages
   - Coordinates between storage and algorithms
   - Provides high-level API for conversation operations

2. **Thread**
   - Represents a conversation thread
   - Contains messages and metadata
   - Has unique identifier
   - Supports thread-level operations

3. **Message**
   - Represents individual messages in conversations
   - Contains content, role (user/assistant/system)
   - Includes metadata and timestamps
   - Supports token counting

## 3. Architecture Layers

### 3.1 Storage Layer
1. **BaseStorage** (Abstract Base Class)
   - Defines interface for storage implementations
   - Core operations: save, get, list, delete, search

2. **Implementations**:
   - **MemoryStorage**: In-memory storage for development/testing
   - **SQLiteStorage**: Persistent storage using SQLite
   - Planned: MongoDB, Redis, PostgreSQL

### 3.2 Algorithm Layer
1. **BaseAlgorithm** (Abstract Base Class)
   - Defines interface for history management algorithms
   - Controls conversation context and window management

2. **Implementations**:
   - **FIFOAlgorithm**: First-In-First-Out message management
   - Planned: Sliding Window, Semantic Chunking

### 3.3 Integration Layer
1. **OpenAI Integration**
   - Seamless integration with OpenAI's chat completions
   - Automatic history tracking
   - Context management
   - Support for both sync and async clients

2. **Planned Integrations**:
   - Anthropic Claude
   - LiteLLM
   - Other LLM providers

## 4. Flow Sequences

### 4.1 Basic Conversation Flow
1. Initialize components:
   ```python
   storage = MemoryStorage()  # or other storage backend
   algorithm = FIFOAlgorithm(max_messages=100)
   history_manager = HistoryManager(storage=storage, algorithm=algorithm)
   ```

2. Create conversation thread:
   ```python
   thread = history_manager.create_thread(metadata={"user": "alice"})
   ```

3. Add messages:
   ```python
   history_manager.add_message(
       thread_id=thread.id,
       content="Hello!",
       role="user"
   )
   ```

4. Retrieve conversation:
   ```python
   thread = history_manager.get_thread(thread.id)
   for msg in thread.messages:
       print(f"{msg.role}: {msg.content}")
   ```

### 4.2 OpenAI Integration Flow
1. Setup with history management:
   ```python
   client = OpenAI()
   wrapped_client = with_history(history_manager=history_manager)(client)
   ```

2. Create conversation:
   ```python
   response = wrapped_client.chat.completions.create(
       model="gpt-3.5-turbo",
       messages=[{"role": "user", "content": "Hello!"}],
       thread_id="my-thread"  # Optional, auto-created if not provided
   )
   ```

3. Automatic history tracking:
   - Input messages stored automatically
   - Responses stored with metadata
   - Context managed by algorithm

## 5. Data Flow

### 5.1 Message Storage Flow
1. Message created with content and role
2. Timestamp and UUID added automatically
3. Optional metadata attached
4. Message added to thread
5. Thread saved to storage backend
6. Algorithm processes thread for context management

### 5.2 Message Retrieval Flow
1. Thread requested by ID
2. Storage backend retrieves thread
3. Algorithm applies context management (if configured)
4. Messages returned in chronological order
5. Optional message limit applied

## 6. Extension Points

### 6.1 Custom Storage Backends
- Implement `BaseStorage` abstract class
- Provide CRUD operations for threads
- Handle message persistence
- Support search capabilities

### 6.2 Custom Algorithms
- Implement `BaseAlgorithm` abstract class
- Define message selection/pruning logic
- Control context window management
- Support custom retention policies

### 6.3 Custom Integrations
- Use `with_history` decorator pattern
- Handle provider-specific message formats
- Manage context windows
- Track conversation state

## 7. Best Practices

### 7.1 Storage Selection
- Use `MemoryStorage` for development/testing
- Use `SQLiteStorage` for single-user applications
- Plan to use MongoDB/PostgreSQL for multi-user applications
- Consider Redis for caching layer

### 7.2 Algorithm Selection
- Use `FIFOAlgorithm` for simple use cases
- Consider token limits of LLM providers
- Balance context retention vs. memory usage
- Consider conversation semantics

### 7.3 Integration Usage
- Always handle API errors gracefully
- Monitor token usage
- Implement rate limiting
- Use appropriate context windows