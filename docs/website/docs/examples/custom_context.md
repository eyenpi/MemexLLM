# Custom Context Management

This example shows how to build a chat application with custom context management using MemexLLM.

## Overview

We'll create a chat application that:
- Uses custom context selection
- Tracks message importance
- Manages conversation topics
- Adapts to user preferences

## Implementation

```python
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from openai import OpenAI
from memexllm.storage import SQLiteStorage
from memexllm.algorithms import BaseAlgorithm
from memexllm.core.models import Message, Thread
from memexllm.integrations.openai import with_history
from memexllm.history import HistoryManager

class SmartContextAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        max_messages: int = 50,
        time_window: timedelta = timedelta(hours=24),
        topic_weight: float = 0.6,
        length_weight: float = 0.4
    ):
        super().__init__(max_messages=max_messages)
        self.time_window = time_window
        self.topic_weight = topic_weight
        self.length_weight = length_weight
    
    def get_message_window(self, messages: List[Message]) -> List[Message]:
        """Select messages based on relevance."""
        now = datetime.now(timezone.utc)
        selected = []
        
        # Always include system messages
        system_messages = [
            msg for msg in messages
            if msg.role == "system"
        ]
        selected.extend(system_messages)
        
        # Score and select other messages
        scored_messages = [
            (msg, self._score_message(msg, now))
            for msg in messages
            if msg.role != "system"
        ]
        
        # Sort by score and take top N
        scored_messages.sort(key=lambda x: x[1], reverse=True)
        selected.extend(
            msg for msg, _ in scored_messages[:self.max_messages]
        )
        
        return selected
    
    def process_thread(self, thread: Thread, new_message: Message) -> None:
        """Update thread metadata with topic info."""
        # Extract topics from message
        topics = self._extract_topics(new_message.content)
        
        # Update thread metadata
        thread.metadata.update({
            "topics": list(set(
                thread.metadata.get("topics", []) + topics
            )),
            "last_topic": topics[0] if topics else None,
            "message_count": len(thread.messages)
        })
    
    def _score_message(
        self,
        message: Message,
        now: datetime
    ) -> float:
        """Score message relevance."""
        # Time score (newer is better)
        age = now - message.created_at
        time_score = max(0, 1 - age / self.time_window)
        
        # Topic score
        topic_score = message.metadata.get("topic_relevance", 0.5)
        
        # Length score (longer might be more important)
        length = len(message.content)
        length_score = min(1.0, length / 500)  # Cap at 500 chars
        
        # Combine scores
        return (
            self.topic_weight * topic_score +
            self.length_weight * length_score +
            (1 - self.topic_weight - self.length_weight) * time_score
        )
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract topics from message content."""
        # Simple keyword-based topic extraction
        topics = []
        keywords = ["python", "javascript", "database", "api", "web"]
        for keyword in keywords:
            if keyword in content.lower():
                topics.append(keyword)
        return topics or ["general"]

class SmartChat:
    def __init__(self):
        # Create algorithm
        algorithm = SmartContextAlgorithm(
            max_messages=50,
            time_window=timedelta(hours=12),
            topic_weight=0.7,
            length_weight=0.3
        )
        
        # Create enhanced OpenAI client
        client = OpenAI()
        history_manager = HistoryManager(
            storage=SQLiteStorage("smart_chat.db"),
            algorithm=algorithm
        )
        self.client = with_history(history_manager=history_manager)(client)
    
    def chat(
        self,
        message: str,
        thread_id: Optional[str] = None,
        topic_relevance: float = 0.5
    ) -> str:
        """Send a message with topic relevance."""
        # Add message metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "topic_relevance": topic_relevance,
            "length": len(message)
        }
        
        # Get response
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": message,
                "metadata": metadata
            }],
            thread_id=thread_id
        )
        
        return response.choices[0].message.content

def main():
    # Create chat
    chat = SmartChat()
    
    print("Smart Chat - Type 'bye' to exit")
    print("Rate topic relevance (0-1) after each message")
    
    thread_id = None
    while True:
        # Get message
        message = input("\nYou: ").strip()
        if message.lower() == "bye":
            break
        
        # Get topic relevance
        relevance = float(input("Topic relevance (0-1): ").strip())
        
        # Get response
        response = chat.chat(
            message=message,
            thread_id=thread_id,
            topic_relevance=relevance
        )
        print("Assistant:", response)

if __name__ == "__main__":
    main()
```

## Usage

Run the smart chat:

```bash
python smart_chat.py
```

Example session:
```
Smart Chat - Type 'bye' to exit
Rate topic relevance (0-1) after each message

You: What's the best way to handle API errors in Python?
Topic relevance (0-1): 0.9
Assistant: When handling API errors in Python, it's best to...

You: How about logging those errors?
Topic relevance (0-1): 0.8
Assistant: For logging API errors, you can use Python's logging module...
```

## Key Features

1. **Smart Context Selection**:
   - Topic-based relevance
   - Time-based decay
   - Length consideration
   - System message priority

2. **Topic Management**:
   - Automatic topic extraction
   - Topic tracking
   - Relevance scoring

3. **Adaptive Context**:
   - User feedback integration
   - Dynamic scoring
   - Flexible weighting

4. **Metadata Tracking**:
   - Message timestamps
   - Topic relevance
   - Message length
   - Thread statistics

## Next Steps

- Try different [storage backends](../storage/overview.md)
- Create your own [algorithms](../algorithms/custom.md)
- Explore [OpenAI features](../integrations/openai.md) 