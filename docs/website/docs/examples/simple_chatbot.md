# Simple Chatbot

This example shows how to build a basic chatbot with conversation memory using MemexLLM and OpenAI.

## Overview

We'll create a chatbot that:
- Remembers conversation history
- Uses SQLite for storage
- Uses FIFO for context management
- Supports multiple users

## Implementation

```python
from openai import OpenAI
from memexllm.storage import SQLiteStorage
from memexllm.algorithms import FIFOAlgorithm
from memexllm.integrations.openai import with_history
from memexllm.history import HistoryManager

class Chatbot:
    def __init__(self, personality: str = "helpful assistant"):
        # Create enhanced OpenAI client
        client = OpenAI()
        history_manager = HistoryManager(
            storage=SQLiteStorage("chatbot.db"),
            algorithm=FIFOAlgorithm(max_messages=50)
        )
        self.client = with_history(history_manager=history_manager)(client)
        
        # Store personality
        self.personality = personality
    
    def chat(self, message: str, user_id: str) -> str:
        """Send a message and get response."""
        # Create thread ID from user ID
        thread_id = f"user_{user_id}"
        
        # Get response
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": message}],
            thread_id=thread_id
        )
        
        return response.choices[0].message.content

def main():
    # Create chatbot
    bot = Chatbot(personality="friendly tech support")
    
    # Chat loop
    print("Chatbot: Hi! How can I help you today?")
    while True:
        # Get user input
        message = input("You: ").strip()
        if message.lower() in ["exit", "quit", "bye"]:
            break
        
        # Get bot response
        response = bot.chat(message, user_id="demo_user")
        print("Chatbot:", response)

if __name__ == "__main__":
    main()
```

## Usage

Run the chatbot:

```bash
python chatbot.py
```

Example conversation:
```
Chatbot: Hi! How can I help you today?
You: How do I install Python?
Chatbot: I'll help you install Python. What operating system are you using?
You: I'm using macOS
Chatbot: Great! Here's how to install Python on macOS...
```

## Key Features

1. **Persistent Memory**:
   - Uses SQLite storage
   - Conversations survive restarts
   - Each user has their own thread

2. **Context Management**:
   - Uses FIFO algorithm
   - Keeps last 50 messages
   - Maintains conversation flow

3. **Simple Interface**:
   - Easy to use chat method
   - Automatic thread management
   - Clean user experience

## Next Steps

- Add [system messages](../integrations/openai.md)
- Try different [storage backends](../storage/overview.md)
- Explore other [algorithms](../algorithms/overview.md) 