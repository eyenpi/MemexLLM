"""
Quickstart example showing the basic usage of MemexLLM with clean imports.
This example demonstrates the core functionality: storage, algorithms, and history management.
"""

from typing import cast

from memexllm.algorithms import FIFOAlgorithm
from memexllm.core import HistoryManager, Message, MessageRole
from memexllm.storage import MemoryStorage


def main() -> None:
    # 1. Initialize components
    storage = MemoryStorage()  # In-memory storage
    algorithm = FIFOAlgorithm(max_messages=100)  # Keep last 100 messages
    history_manager = HistoryManager(storage=storage, algorithm=algorithm)

    # 2. Create a conversation thread
    thread = history_manager.create_thread(
        metadata={"user": "alice", "topic": "python"}
    )
    print(f"Created thread: {thread.id}")

    # 3. Add messages to the conversation
    messages = [
        ("system", "You are a helpful coding assistant."),
        ("user", "How do I read a file in Python?"),
        (
            "assistant",
            "Here's a simple way to read a file in Python:\n\n```python\nwith open('file.txt', 'r') as f:\n    content = f.read()\n```",
        ),
        ("user", "What if I want to read it line by line?"),
    ]

    for role, content in messages:
        history_manager.add_message(
            thread_id=thread.id,
            content=content,
            role=role,
        )

    # 4. Retrieve and display conversation
    thread = history_manager.get_thread(thread.id)
    print("\nConversation:")
    print("-" * 50)
    for msg in thread.messages:
        print(f"{msg.role.upper()}: {msg.content}\n")

    # 5. Demonstrate metadata and thread management
    print("Thread Information:")
    print(f"- ID: {thread.id}")
    print(f"- Metadata: {thread.metadata}")
    print(f"- Message count: {len(thread.messages)}")

    # 6. List all threads
    all_threads = history_manager.list_threads()
    print(f"\nTotal threads in storage: {len(all_threads)}")


if __name__ == "__main__":
    main()
