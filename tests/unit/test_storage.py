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
        # We intentionally try to instantiate an abstract class to test that it fails
        BaseStorage()  # type: ignore[abstract]

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


def test_memory_storage_search() -> None:
    """Test MemoryStorage search functionality"""
    storage = MemoryStorage()

    # Create threads with different metadata
    thread1 = Thread(id="1", metadata={"category": "work", "priority": "high"})
    thread2 = Thread(id="2", metadata={"category": "personal", "priority": "low"})
    thread3 = Thread(id="3", metadata={"category": "work", "priority": "low"})

    storage.save_thread(thread1)
    storage.save_thread(thread2)
    storage.save_thread(thread3)

    # Test searching by metadata
    work_threads = storage.search_threads({"metadata": {"category": "work"}})
    assert len(work_threads) == 2
    assert all(t.metadata["category"] == "work" for t in work_threads)

    # Test searching with multiple criteria
    high_priority_work = storage.search_threads(
        {"metadata": {"category": "work", "priority": "high"}}
    )
    assert len(high_priority_work) == 1
    assert high_priority_work[0].id == "1"

    # Test search with non-existent criteria
    no_results = storage.search_threads({"metadata": {"category": "non-existent"}})
    assert len(no_results) == 0


def test_memory_storage_pagination() -> None:
    """Test MemoryStorage pagination functionality"""
    storage = MemoryStorage()

    # Create multiple threads
    for i in range(5):
        thread = Thread(id=str(i))
        storage.save_thread(thread)

    # Test pagination
    first_page = storage.list_threads(limit=2, offset=0)
    assert len(first_page) == 2

    second_page = storage.list_threads(limit=2, offset=2)
    assert len(second_page) == 2

    last_page = storage.list_threads(limit=2, offset=4)
    assert len(last_page) == 1
