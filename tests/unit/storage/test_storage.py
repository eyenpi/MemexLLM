import pytest

from memexllm.core import Thread
from memexllm.core.models import Message
from memexllm.storage import BaseStorage, MemoryStorage


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

    # Add some messages
    thread1.add_message(Message(content="Important work meeting", role="user"))
    thread2.add_message(Message(content="Grocery list", role="user"))
    thread3.add_message(Message(content="Work report", role="user"))

    storage.save_thread(thread1)
    storage.save_thread(thread2)
    storage.save_thread(thread3)

    # Test searching by metadata
    work_threads = storage.search_threads({"metadata": {"category": "work"}})
    assert len(work_threads) == 2
    assert all(t.metadata["category"] == "work" for t in work_threads)

    high_priority = storage.search_threads({"metadata": {"priority": "high"}})
    assert len(high_priority) == 1
    assert high_priority[0].metadata["priority"] == "high"

    # Test searching by content
    work_content = storage.search_threads({"content": "work"})
    assert len(work_content) == 2

    grocery_content = storage.search_threads({"content": "grocery"})
    assert len(grocery_content) == 1

    # Test combined search
    work_high = storage.search_threads(
        {"metadata": {"category": "work", "priority": "high"}, "content": "meeting"}
    )
    assert len(work_high) == 1
    assert work_high[0].id == "1"


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


def test_memory_storage_init() -> None:
    """Test MemoryStorage initialization"""
    storage = MemoryStorage()
    assert storage.threads == {}
    assert storage.max_messages is None

    storage = MemoryStorage(max_messages=10)
    assert storage.threads == {}
    assert storage.max_messages == 10


def test_memory_storage_save_and_get() -> None:
    """Test basic save and get operations"""
    storage = MemoryStorage()
    thread = Thread(id="test")
    storage.save_thread(thread)

    retrieved = storage.get_thread("test")
    assert retrieved is not None
    assert retrieved.id == "test"
    assert retrieved is not thread  # Should be a copy


def test_memory_storage_message_limit() -> None:
    """Test message limiting in storage"""
    storage = MemoryStorage(max_messages=2)
    thread = Thread(id="test")

    # Add 3 messages
    for i in range(3):
        thread.add_message(Message(content=f"msg{i}", role="user"))

    storage.save_thread(thread)
    retrieved = storage.get_thread("test")
    assert retrieved is not None
    assert len(retrieved.messages) == 2
    # Should keep the most recent 2 messages (msg1, msg2)
    assert [m.content for m in retrieved.messages] == ["msg1", "msg2"]

    # Add another message to verify sliding window behavior
    thread.add_message(Message(content="msg3", role="user"))
    storage.save_thread(thread)
    retrieved = storage.get_thread("test")
    assert retrieved is not None
    assert len(retrieved.messages) == 2
    # Should now have msg2 and msg3, as they are the most recent
    assert [m.content for m in retrieved.messages] == ["msg2", "msg3"]


def test_memory_storage_get_with_limit() -> None:
    """Test get_thread with message_limit parameter"""
    storage = MemoryStorage()
    thread = Thread(id="test")

    # Add 5 messages
    for i in range(5):
        thread.add_message(Message(content=f"msg{i}", role="user"))

    storage.save_thread(thread)

    # Get with limit
    retrieved = storage.get_thread("test", message_limit=3)
    assert retrieved is not None
    assert len(retrieved.messages) == 3
    assert [m.content for m in retrieved.messages] == ["msg2", "msg3", "msg4"]

    # Get without limit
    retrieved = storage.get_thread("test")
    assert retrieved is not None
    assert len(retrieved.messages) == 5


def test_memory_storage_delete() -> None:
    """Test thread deletion"""
    storage = MemoryStorage()
    thread = Thread(id="test")
    storage.save_thread(thread)

    assert storage.delete_thread("test") is True
    assert storage.get_thread("test") is None
    assert storage.delete_thread("test") is False  # Already deleted
