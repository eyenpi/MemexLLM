import pytest

from memexllm.algorithms.fifo import FIFOAlgorithm
from memexllm.core.history import HistoryManager
from memexllm.storage.memory import MemoryStorage


def test_complete_conversation_flow():
    # Initialize system
    storage = MemoryStorage()
    algorithm = FIFOAlgorithm(max_messages=50)
    manager = HistoryManager(storage=storage, algorithm=algorithm)

    # Create thread with metadata
    thread = manager.create_thread(metadata={"user_id": "user123"})

    # Simulate conversation
    manager.add_message(
        thread_id=thread.id, content="Hello, how can you help me?", role="user"
    )

    manager.add_message(
        thread_id=thread.id,
        content="I'm here to help! What's your question?",
        role="assistant",
    )

    # Verify thread state
    retrieved_thread = manager.get_thread(thread.id)
    assert len(retrieved_thread.messages) == 2
    assert retrieved_thread.messages[0].role == "user"
    assert retrieved_thread.messages[1].role == "assistant"

    # Verify metadata persistence
    assert retrieved_thread.metadata["user_id"] == "user123"


def test_message_truncation():
    manager = HistoryManager(
        storage=MemoryStorage(), algorithm=FIFOAlgorithm(max_messages=2)
    )

    thread = manager.create_thread()

    # Add more messages than the limit
    messages = [
        ("user", "Message 1"),
        ("assistant", "Response 1"),
        ("user", "Message 2"),
    ]

    for role, content in messages:
        manager.add_message(thread_id=thread.id, role=role, content=content)

    # Verify only last 2 messages are kept
    thread = manager.get_thread(thread.id)
    assert len(thread.messages) == 2
    assert thread.messages[0].content == "Response 1"
    assert thread.messages[1].content == "Message 2"
