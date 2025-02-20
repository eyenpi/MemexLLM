import pytest

from memexllm.core.models import Thread
from memexllm.storage.base import BaseStorage
from memexllm.storage.memory import MemoryStorage


def test_memory_storage_operations() -> None:
    storage = MemoryStorage()

    # Test thread creation and retrieval
    thread = Thread(id="test123", metadata={"user_id": "user1"})
    storage.save_thread(thread)

    retrieved_thread = storage.get_thread("test123")
    assert retrieved_thread is not None  # Type check
    assert retrieved_thread.id == "test123"
    assert retrieved_thread.metadata == {"user_id": "user1"}

    # Test thread listing
    threads = storage.list_threads()
    assert len(threads) == 1
    # Since list_threads returns Thread objects, not just IDs
    assert any(t.id == "test123" for t in threads)

    # Test thread deletion
    storage.delete_thread("test123")
    assert storage.get_thread("test123") is None


def test_base_storage_is_abstract() -> None:
    # Test that BaseStorage is an abstract class
    with pytest.raises(TypeError) as exc_info:
        BaseStorage()  # type: ignore

    error_msg = str(exc_info.value)
    # Verify that all required abstract methods are mentioned
    assert "abstract class" in error_msg
    assert "delete_thread" in error_msg
    assert "get_thread" in error_msg
    assert "list_threads" in error_msg
    assert "save_thread" in error_msg
    assert "search_threads" in error_msg


def test_save_and_get_thread() -> None:
    storage = MemoryStorage()
    thread = Thread(id="test_thread")

    storage.save_thread(thread)
    retrieved = storage.get_thread("test_thread")

    assert retrieved is not None
    assert retrieved.id == "test_thread"


def test_list_threads() -> None:
    storage = MemoryStorage()
    thread1 = Thread(id="thread1")
    thread2 = Thread(id="thread2")

    storage.save_thread(thread1)
    storage.save_thread(thread2)

    threads = storage.list_threads()
    assert len(threads) == 2
