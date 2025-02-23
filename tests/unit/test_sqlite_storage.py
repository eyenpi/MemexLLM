"""Tests for SQLite storage implementation."""

import os
from datetime import datetime, timezone
from typing import Generator

import pytest

from memexllm.algorithms import FIFOAlgorithm
from memexllm.core import HistoryManager, Message, Thread
from memexllm.storage.sqlite import SQLiteStorage


@pytest.fixture
def db_path() -> Generator[str, None, None]:
    """Fixture to provide a test database path."""
    path = "test_memexllm.db"
    yield path
    # Cleanup after test
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def storage(db_path: str) -> SQLiteStorage:
    """Fixture to provide a SQLite storage instance."""
    return SQLiteStorage(db_path=db_path)


def test_sqlite_init(db_path: str) -> None:
    """Test SQLite storage initialization and schema creation."""
    storage = SQLiteStorage(db_path=db_path)

    # Verify database file was created
    assert os.path.exists(db_path)

    # Verify tables were created
    with storage._get_connection() as conn:
        # Check threads table
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='threads'"
        )
        assert cursor.fetchone() is not None

        # Check messages table
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
        )
        assert cursor.fetchone() is not None

        # Check message index
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_messages_thread_id'"
        )
        assert cursor.fetchone() is not None


def test_sqlite_persistence(db_path: str) -> None:
    """Test that data persists between storage instances."""
    # Create first storage instance and add data
    storage1 = SQLiteStorage(db_path=db_path)
    thread = Thread(id="test", metadata={"key": "value"})
    thread.add_message(Message(content="Hello", role="user"))
    storage1.save_thread(thread)

    # Create second storage instance and verify data
    storage2 = SQLiteStorage(db_path=db_path)
    loaded_thread = storage2.get_thread("test")
    assert loaded_thread is not None
    assert loaded_thread.id == "test"
    assert loaded_thread.metadata == {"key": "value"}
    assert len(loaded_thread.messages) == 1
    assert loaded_thread.messages[0].content == "Hello"
    assert loaded_thread.messages[0].role == "user"


def test_sqlite_message_ordering(storage: SQLiteStorage) -> None:
    """Test that message order is preserved."""
    thread = Thread(id="test")
    messages = [
        Message(
            content=f"msg{i}",
            role="user",
            created_at=datetime.fromtimestamp(i, tz=timezone.utc),
        )
        for i in range(5)
    ]
    for msg in messages:
        thread.add_message(msg)

    storage.save_thread(thread)
    loaded_thread = storage.get_thread("test")
    assert loaded_thread is not None
    assert len(loaded_thread.messages) == 5
    assert [m.content for m in loaded_thread.messages] == [
        "msg0",
        "msg1",
        "msg2",
        "msg3",
        "msg4",
    ]


def test_sqlite_metadata_serialization(storage: SQLiteStorage) -> None:
    """Test complex metadata serialization."""
    thread = Thread(
        id="test",
        metadata={
            "str": "value",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "none": None,
        },
    )
    storage.save_thread(thread)

    loaded_thread = storage.get_thread("test")
    assert loaded_thread is not None
    assert loaded_thread.metadata == thread.metadata


def test_sqlite_storage_limits() -> None:
    """Test storage message limits with SQLite."""
    storage = SQLiteStorage(db_path="test_limits.db", max_messages=3)
    try:
        thread = Thread(id="test")
        for i in range(5):
            thread.add_message(Message(content=f"msg{i}", role="user"))

        storage.save_thread(thread)
        loaded_thread = storage.get_thread("test")
        assert loaded_thread is not None
        assert len(loaded_thread.messages) == 3
        assert [m.content for m in loaded_thread.messages] == ["msg2", "msg3", "msg4"]
    finally:
        if os.path.exists("test_limits.db"):
            os.remove("test_limits.db")


def test_sqlite_message_limit_parameter(storage: SQLiteStorage) -> None:
    """Test message_limit parameter in get_thread."""
    thread = Thread(id="test")
    for i in range(5):
        thread.add_message(Message(content=f"msg{i}", role="user"))

    storage.save_thread(thread)

    # Test with different limits
    loaded_thread = storage.get_thread("test", message_limit=3)
    assert loaded_thread is not None
    assert len(loaded_thread.messages) == 3
    assert [m.content for m in loaded_thread.messages] == ["msg2", "msg3", "msg4"]

    # Test with no limit
    loaded_thread = storage.get_thread("test")
    assert loaded_thread is not None
    assert len(loaded_thread.messages) == 5


def test_sqlite_search() -> None:
    """Test SQLite search functionality."""
    storage = SQLiteStorage(db_path="test_search.db")
    try:
        # Create threads with different metadata and content
        thread1 = Thread(id="1", metadata={"category": "work", "priority": "high"})
        thread1.add_message(Message(content="Important work meeting", role="user"))

        thread2 = Thread(id="2", metadata={"category": "personal", "priority": "low"})
        thread2.add_message(Message(content="Grocery list", role="user"))

        thread3 = Thread(id="3", metadata={"category": "work", "priority": "low"})
        thread3.add_message(Message(content="Work report", role="user"))

        storage.save_thread(thread1)
        storage.save_thread(thread2)
        storage.save_thread(thread3)

        # Test metadata search
        work_threads = storage.search_threads({"metadata": {"category": "work"}})
        assert len(work_threads) == 2
        assert all(t.metadata["category"] == "work" for t in work_threads)

        # Test content search
        work_content = storage.search_threads({"content": "work"})
        assert len(work_content) == 2

        # Test combined search
        work_high = storage.search_threads(
            {"metadata": {"category": "work", "priority": "high"}, "content": "meeting"}
        )
        assert len(work_high) == 1
        assert work_high[0].id == "1"
    finally:
        if os.path.exists("test_search.db"):
            os.remove("test_search.db")


def test_sqlite_with_algorithm() -> None:
    """Test SQLite storage with algorithm."""
    storage = SQLiteStorage(db_path="test_algo.db", max_messages=5)
    algorithm = FIFOAlgorithm(max_messages=3)
    manager = HistoryManager(storage=storage, algorithm=algorithm)

    try:
        # Create thread and add messages
        thread = manager.create_thread()
        for i in range(7):
            manager.add_message(thread.id, f"msg{i}", "user")

        # Verify storage limit
        stored_thread = storage.get_thread(thread.id)
        assert stored_thread is not None
        assert len(stored_thread.messages) == 5
        assert [m.content for m in stored_thread.messages] == [
            "msg2",
            "msg3",
            "msg4",
            "msg5",
            "msg6",
        ]

        # Verify algorithm window
        context_thread = manager.get_thread(thread.id)
        assert context_thread is not None
        assert len(context_thread.messages) == 3
        assert [m.content for m in context_thread.messages] == ["msg4", "msg5", "msg6"]
    finally:
        if os.path.exists("test_algo.db"):
            os.remove("test_algo.db")


def test_sqlite_concurrent_access(db_path: str) -> None:
    """Test concurrent access to SQLite storage."""
    storage1 = SQLiteStorage(db_path=db_path)
    storage2 = SQLiteStorage(db_path=db_path)

    # Create thread in first storage
    thread = Thread(id="test")
    thread.add_message(Message(content="msg1", role="user"))
    storage1.save_thread(thread)

    # Read and modify in second storage
    loaded_thread = storage2.get_thread("test")
    assert loaded_thread is not None
    loaded_thread.add_message(Message(content="msg2", role="user"))
    storage2.save_thread(loaded_thread)

    # Verify in first storage
    final_thread = storage1.get_thread("test")
    assert final_thread is not None
    assert len(final_thread.messages) == 2
    assert [m.content for m in final_thread.messages] == ["msg1", "msg2"]
