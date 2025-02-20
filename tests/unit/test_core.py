import pytest

from memexllm.algorithms.fifo import FIFOAlgorithm
from memexllm.core.history import HistoryManager
from memexllm.storage.memory import MemoryStorage


def test_history_manager_creation() -> None:
    storage = MemoryStorage()
    algorithm = FIFOAlgorithm(max_messages=50)
    manager = HistoryManager(storage=storage, algorithm=algorithm)
    assert manager.storage == storage
    assert manager.algorithm == algorithm


def test_thread_creation() -> None:
    manager = HistoryManager(storage=MemoryStorage())
    metadata = {"user_id": "test123"}
    thread = manager.create_thread(metadata=metadata)

    assert thread.id is not None
    assert thread.metadata == metadata
    assert len(thread.messages) == 0


def test_message_management() -> None:
    manager = HistoryManager(storage=MemoryStorage())
    thread = manager.create_thread()

    # Add messages
    manager.add_message(thread_id=thread.id, content="Hello", role="user")
    manager.add_message(thread_id=thread.id, content="Hi there!", role="assistant")

    # Retrieve and verify
    updated_thread = manager.get_thread(thread.id)
    assert updated_thread is not None  # Type check
    assert len(updated_thread.messages) == 2
    assert updated_thread.messages[0].content == "Hello"
    assert updated_thread.messages[0].role == "user"
    assert updated_thread.messages[1].content == "Hi there!"
    assert updated_thread.messages[1].role == "assistant"
