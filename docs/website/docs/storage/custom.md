# Custom Storage

Need a different storage backend? MemexLLM makes it easy to create your own storage implementation.

## Overview

Custom storage backends must:
1. Inherit from `BaseStorage`
2. Implement required methods
3. Handle thread and message management
4. Support metadata

## Basic Implementation

Here's a minimal custom storage implementation:

```python
from typing import Optional, List
from memexllm.storage.base import BaseStorage
from memexllm.core.models import Thread

class CustomStorage(BaseStorage):
    def __init__(self, max_messages: Optional[int] = None):
        super().__init__(max_messages=max_messages)
        self.threads = {}  # Your storage mechanism
    
    def save_thread(self, thread: Thread) -> None:
        """Save or update a thread."""
        # Apply message limit if set
        if self.max_messages and len(thread.messages) > self.max_messages:
            thread.messages = thread.messages[-self.max_messages:]
        # Save thread
        self.threads[thread.id] = thread
    
    def get_thread(self, thread_id: str) -> Optional[Thread]:
        """Get a thread by ID."""
        return self.threads.get(thread_id)
    
    def list_threads(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Thread]:
        """List threads with pagination."""
        threads = list(self.threads.values())
        return threads[offset:offset + limit]
    
    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread."""
        if thread_id in self.threads:
            del self.threads[thread_id]
            return True
        return False
```

## Advanced Implementation

Here's a more complete example using Redis:

```python
import json
from typing import Optional, List
from redis import Redis
from memexllm.storage.base import BaseStorage
from memexllm.core.models import Thread, Message

class RedisStorage(BaseStorage):
    def __init__(
        self,
        redis_url: str,
        max_messages: Optional[int] = None
    ):
        super().__init__(max_messages=max_messages)
        self.redis = Redis.from_url(redis_url)
        
    def _thread_key(self, thread_id: str) -> str:
        return f"thread:{thread_id}"
        
    def _serialize_thread(self, thread: Thread) -> str:
        return json.dumps({
            "id": thread.id,
            "metadata": thread.metadata,
            "messages": [
                {
                    "content": msg.content,
                    "role": msg.role,
                    "metadata": msg.metadata,
                    "created_at": msg.created_at.isoformat()
                }
                for msg in thread.messages
            ]
        })
        
    def _deserialize_thread(self, data: str) -> Thread:
        thread_data = json.loads(data)
        thread = Thread(
            id=thread_data["id"],
            metadata=thread_data["metadata"]
        )
        thread.messages = [
            Message(
                content=msg["content"],
                role=msg["role"],
                metadata=msg["metadata"],
                created_at=msg["created_at"]
            )
            for msg in thread_data["messages"]
        ]
        return thread
        
    def save_thread(self, thread: Thread) -> None:
        if self.max_messages and len(thread.messages) > self.max_messages:
            thread.messages = thread.messages[-self.max_messages:]
        self.redis.set(
            self._thread_key(thread.id),
            self._serialize_thread(thread)
        )
        # Update thread index
        self.redis.sadd("threads", thread.id)
        
    def get_thread(self, thread_id: str) -> Optional[Thread]:
        data = self.redis.get(self._thread_key(thread_id))
        if data:
            return self._deserialize_thread(data)
        return None
        
    def list_threads(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Thread]:
        thread_ids = self.redis.smembers("threads")
        threads = []
        for thread_id in list(thread_ids)[offset:offset + limit]:
            thread = self.get_thread(thread_id.decode())
            if thread:
                threads.append(thread)
        return threads
        
    def delete_thread(self, thread_id: str) -> bool:
        if self.redis.exists(self._thread_key(thread_id)):
            self.redis.delete(self._thread_key(thread_id))
            self.redis.srem("threads", thread_id)
            return True
        return False
```

## Using Custom Storage

Use your custom storage like any other:

```python
from openai import OpenAI
from memexllm.integrations.openai import with_history

# Create your custom storage
storage = RedisStorage(
    redis_url="redis://localhost:6379/0",
    max_messages=1000
)

# Create history manager
manager = HistoryManager(
    storage=storage,
    algorithm=FIFOAlgorithm(max_messages=50)  # Use last 50 messages for context
)

# Use with OpenAI
client = OpenAI()
client = with_history(history_manager=manager)(client)

# Use as normal
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Best Practices

1. **Error Handling**:
   - Handle storage errors gracefully
   - Provide meaningful error messages
   - Consider retries for transient failures

2. **Performance**:
   - Implement efficient storage operations
   - Consider caching if needed
   - Monitor resource usage

3. **Thread Safety**:
   - Handle concurrent access
   - Use appropriate locking mechanisms
   - Consider transaction support

4. **Testing**:
   - Write comprehensive tests
   - Test edge cases
   - Verify data consistency

## Next Steps

- Learn about [context algorithms](../algorithms/overview.md)
- See [example applications](../examples/simple_chatbot.md)
- Check out [available storage](./overview.md) options 