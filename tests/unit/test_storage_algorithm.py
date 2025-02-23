"""Tests for storage and algorithm capacity independence."""

from memexllm.algorithms import FIFOAlgorithm
from memexllm.core import HistoryManager, Message, Thread
from memexllm.storage import MemoryStorage


def create_thread_with_messages(num_messages: int) -> Thread:
    """Helper to create a thread with specified number of messages."""
    thread = Thread(id="test")
    for i in range(num_messages):
        thread.add_message(Message(content=f"msg{i}", role="user"))
    return thread


def test_storage_larger_than_algorithm() -> None:
    """Test when storage capacity is larger than algorithm window."""
    storage = MemoryStorage(max_messages=10)  # Store up to 10 messages
    algorithm = FIFOAlgorithm(max_messages=5)  # But only use last 5 for context
    manager = HistoryManager(storage=storage, algorithm=algorithm)

    # Create thread and add messages
    thread = manager.create_thread()
    for i in range(8):  # Add 8 messages
        manager.add_message(thread.id, f"msg{i}", "user")

    # Verify storage has more messages than algorithm window
    stored_thread = storage.get_thread(thread.id)
    assert stored_thread is not None
    assert len(stored_thread.messages) == 8
    assert [m.content for m in stored_thread.messages] == [
        "msg0",
        "msg1",
        "msg2",
        "msg3",
        "msg4",
        "msg5",
        "msg6",
        "msg7",
    ]

    # Verify algorithm window only shows last 5
    context_thread = manager.get_thread(thread.id)
    assert context_thread is not None
    assert len(context_thread.messages) == 5
    assert [m.content for m in context_thread.messages] == [
        "msg3",
        "msg4",
        "msg5",
        "msg6",
        "msg7",
    ]


def test_storage_smaller_than_algorithm() -> None:
    """Test when storage capacity is smaller than algorithm window."""
    storage = MemoryStorage(max_messages=3)  # Only store last 3 messages
    algorithm = FIFOAlgorithm(max_messages=5)  # Try to use last 5 for context
    manager = HistoryManager(storage=storage, algorithm=algorithm)

    # Create thread and add messages
    thread = manager.create_thread()
    for i in range(5):  # Add 5 messages
        manager.add_message(thread.id, f"msg{i}", "user")

    # Verify storage only keeps last 3
    stored_thread = storage.get_thread(thread.id)
    assert stored_thread is not None
    assert len(stored_thread.messages) == 3
    assert [m.content for m in stored_thread.messages] == ["msg2", "msg3", "msg4"]

    # Verify algorithm can't exceed what's in storage
    context_thread = manager.get_thread(thread.id)
    assert context_thread is not None
    assert len(context_thread.messages) == 3
    assert [m.content for m in context_thread.messages] == ["msg2", "msg3", "msg4"]


def test_unlimited_storage_with_algorithm() -> None:
    """Test when storage is unlimited but algorithm has a window."""
    storage = MemoryStorage()  # No message limit
    algorithm = FIFOAlgorithm(max_messages=3)  # Only use last 3 for context
    manager = HistoryManager(storage=storage, algorithm=algorithm)

    # Create thread and add messages
    thread = manager.create_thread()
    for i in range(5):  # Add 5 messages
        manager.add_message(thread.id, f"msg{i}", "user")

    # Verify storage keeps all messages
    stored_thread = storage.get_thread(thread.id)
    assert stored_thread is not None
    assert len(stored_thread.messages) == 5
    assert [m.content for m in stored_thread.messages] == [
        "msg0",
        "msg1",
        "msg2",
        "msg3",
        "msg4",
    ]

    # Verify algorithm only shows last 3
    context_thread = manager.get_thread(thread.id)
    assert context_thread is not None
    assert len(context_thread.messages) == 3
    assert [m.content for m in context_thread.messages] == ["msg2", "msg3", "msg4"]


def test_storage_limit_with_no_algorithm() -> None:
    """Test when storage has a limit but no algorithm is used."""
    storage = MemoryStorage(max_messages=3)  # Only store last 3 messages
    manager = HistoryManager(storage=storage)  # No algorithm

    # Create thread and add messages
    thread = manager.create_thread()
    for i in range(5):  # Add 5 messages
        manager.add_message(thread.id, f"msg{i}", "user")

    # Verify both direct storage and manager return same limited messages
    stored_thread = storage.get_thread(thread.id)
    assert stored_thread is not None
    assert len(stored_thread.messages) == 3
    assert [m.content for m in stored_thread.messages] == ["msg2", "msg3", "msg4"]

    managed_thread = manager.get_thread(thread.id)
    assert managed_thread is not None
    assert len(managed_thread.messages) == 3
    assert [m.content for m in managed_thread.messages] == ["msg2", "msg3", "msg4"]


def test_incremental_message_addition() -> None:
    """Test how limits work as messages are added incrementally."""
    storage = MemoryStorage(max_messages=4)  # Store last 4 messages
    algorithm = FIFOAlgorithm(max_messages=2)  # Use last 2 for context
    manager = HistoryManager(storage=storage, algorithm=algorithm)

    # Create empty thread
    thread = manager.create_thread()

    # Add messages one by one and verify limits
    for i in range(6):  # Add 6 messages total
        manager.add_message(thread.id, f"msg{i}", "user")

        # Check storage
        stored_thread = storage.get_thread(thread.id)
        assert stored_thread is not None
        expected_storage = [f"msg{j}" for j in range(max(0, i - 3), i + 1)]
        assert [m.content for m in stored_thread.messages] == expected_storage

        # Check algorithm window
        context_thread = manager.get_thread(thread.id)
        assert context_thread is not None
        expected_context = [f"msg{j}" for j in range(max(0, i - 1), i + 1)]
        assert [m.content for m in context_thread.messages] == expected_context


def test_message_limit_parameter() -> None:
    """Test interaction between storage limit, algorithm window, and message_limit parameter."""
    storage = MemoryStorage(max_messages=5)  # Store last 5 messages
    algorithm = FIFOAlgorithm(max_messages=3)  # Use last 3 for context
    manager = HistoryManager(storage=storage, algorithm=algorithm)

    # Create thread and add messages
    thread = manager.create_thread()
    for i in range(7):  # Add 7 messages
        manager.add_message(thread.id, f"msg{i}", "user")

    # Test different message_limit values
    # 1. No limit (should use algorithm's limit)
    context_thread = manager.get_thread(thread.id)
    assert context_thread is not None
    assert len(context_thread.messages) == 3  # Algorithm's limit
    assert [m.content for m in context_thread.messages] == ["msg4", "msg5", "msg6"]

    # 2. Limit smaller than both storage and algorithm
    stored_thread = storage.get_thread(thread.id, message_limit=2)
    assert stored_thread is not None
    assert len(stored_thread.messages) == 2
    assert [m.content for m in stored_thread.messages] == ["msg5", "msg6"]

    # 3. Limit larger than storage capacity
    stored_thread = storage.get_thread(thread.id, message_limit=10)
    assert stored_thread is not None
    assert len(stored_thread.messages) == 5  # Storage's limit
    assert [m.content for m in stored_thread.messages] == [
        "msg2",
        "msg3",
        "msg4",
        "msg5",
        "msg6",
    ]
