# Custom Algorithms

Need a different context selection strategy? MemexLLM makes it easy to create your own algorithm implementation.

## Overview

Custom algorithms must:
1. Inherit from `BaseAlgorithm`
2. Implement required methods
3. Handle message selection
4. Support thread processing

## Basic Implementation

Here's a minimal custom algorithm implementation:

```python
from typing import List
from memexllm.algorithms.base import BaseAlgorithm
from memexllm.core.models import Message, Thread

class CustomAlgorithm(BaseAlgorithm):
    def __init__(self, max_messages: int = 100):
        super().__init__(max_messages=max_messages)
    
    def get_message_window(self, messages: List[Message]) -> List[Message]:
        """Select messages for context window."""
        # Simple example: keep last N messages
        return messages[-self.max_messages:]
    
    def process_thread(self, thread: Thread, new_message: Message) -> None:
        """Optional: Process thread when message is added."""
        pass  # No special processing needed
```

## Advanced Implementation

Here's a more complete example using metadata and time:

```python
from datetime import datetime, timedelta, timezone
from typing import List
from memexllm.algorithms.base import BaseAlgorithm
from memexllm.core.models import Message, Thread

class SmartAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        max_messages: int = 100,
        time_window_hours: int = 24,
        importance_threshold: float = 0.8
    ):
        super().__init__(max_messages=max_messages)
        self.time_window = timedelta(hours=time_window_hours)
        self.importance_threshold = importance_threshold
    
    def get_message_window(self, messages: List[Message]) -> List[Message]:
        """Select messages based on time and importance."""
        now = datetime.now(timezone.utc)
        selected = []
        
        # Always include system messages
        system_messages = [
            msg for msg in messages
            if msg.role == "system"
        ]
        selected.extend(system_messages)
        
        # Select recent important messages
        recent_messages = [
            msg for msg in messages
            if (
                msg.role != "system" and
                now - msg.created_at <= self.time_window and
                self._is_important(msg)
            )
        ]
        selected.extend(recent_messages[-self.max_messages:])
        
        return selected
    
    def process_thread(self, thread: Thread, new_message: Message) -> None:
        """Update thread metadata with message stats."""
        # Count messages by role
        role_counts = {}
        for msg in thread.messages:
            role_counts[msg.role] = role_counts.get(msg.role, 0) + 1
        
        # Update thread metadata
        thread.metadata.update({
            "message_counts": role_counts,
            "last_message": new_message.created_at.isoformat(),
            "has_important": any(
                self._is_important(msg)
                for msg in thread.messages
            )
        })
    
    def _is_important(self, message: Message) -> bool:
        """Check if message is important."""
        # Use metadata if available
        if "importance" in message.metadata:
            return message.metadata["importance"] >= self.importance_threshold
        
        # Simple heuristic: longer messages might be more important
        return len(message.content) > 100
```

## Using Custom Algorithms

Use your custom algorithm like any other:

```python
from openai import OpenAI
from memexllm.storage import SQLiteStorage
from memexllm.integrations.openai import with_history
from memexllm.history import HistoryManager

# Create your custom algorithm
algorithm = SmartAlgorithm(
    max_messages=50,
    time_window_hours=12,
    importance_threshold=0.7
)

# Use with OpenAI
client = OpenAI()
history_manager = HistoryManager(
    storage=SQLiteStorage("chat.db"),
    algorithm=algorithm
)
client = with_history(history_manager=history_manager)(client)

# Use as normal
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Best Practices

1. **Message Selection**:
   - Consider message relevance
   - Respect token limits
   - Handle system messages
   - Maintain conversation flow

2. **Performance**:
   - Keep selection logic efficient
   - Avoid expensive operations
   - Consider caching if needed
   - Profile with large threads

3. **Thread Processing**:
   - Update metadata thoughtfully
   - Handle errors gracefully
   - Consider thread lifecycle
   - Document behavior

4. **Testing**:
   - Test with various inputs
   - Verify selection logic
   - Check edge cases
   - Test performance

## Next Steps

- Try the [FIFO Algorithm](./fifo.md)
- See [example applications](../examples/simple_chatbot.md)
- Check out [available algorithms](./overview.md) 