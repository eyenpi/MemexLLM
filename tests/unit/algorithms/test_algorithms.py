from typing import cast

import pytest

from memexllm.algorithms import BaseAlgorithm, FIFOAlgorithm
from memexllm.core import Message, MessageRole, Thread
from memexllm.types import MessageType, ThreadType


def test_base_algorithm() -> None:
    # Create a concrete class that inherits from BaseAlgorithm but doesn't implement process_thread
    class TestAlgorithm(BaseAlgorithm):
        pass

    with pytest.raises(TypeError) as exc_info:
        TestAlgorithm()  # type: ignore[abstract]

    error_msg = str(exc_info.value)
    assert "abstract class" in error_msg
    assert "process_thread" in error_msg


def test_fifo_algorithm_initialization() -> None:
    algorithm = FIFOAlgorithm(max_messages=5)
    assert algorithm.max_messages == 5


def test_fifo_algorithm_processing() -> None:
    algorithm = FIFOAlgorithm(max_messages=2)
    thread = Thread()

    # Add first message
    msg1 = Message(role=cast(MessageRole, "user"), content="First")
    algorithm.process_thread(thread, msg1)
    assert len(thread.messages) == 1

    # Add second message
    msg2 = Message(role=cast(MessageRole, "assistant"), content="Second")
    algorithm.process_thread(thread, msg2)
    assert len(thread.messages) == 2

    # Add third message (should remove first message)
    msg3 = Message(role=cast(MessageRole, "user"), content="Third")
    algorithm.process_thread(thread, msg3)
    assert len(thread.messages) == 2
    assert thread.messages[0].content == "Second"
    assert thread.messages[1].content == "Third"


def test_fifo_with_max_messages() -> None:
    algo = FIFOAlgorithm(max_messages=3)
    messages = [
        Message(role=cast(MessageRole, "user"), content=str(i)) for i in range(4)
    ]
    thread = Thread(id="test", messages=messages)

    new_message = Message(role=cast(MessageRole, "user"), content="4")
    algo.process_thread(thread, new_message)  # Modifies thread in-place

    assert len(thread.messages) == 3
    assert thread.messages[0].content == "2"
    assert thread.messages[1].content == "3"
    assert thread.messages[2].content == "4"


def test_fifo_empty_thread() -> None:
    algo = FIFOAlgorithm(max_messages=2)
    thread = Thread(id="test")

    new_message = Message(role=cast(MessageRole, "user"), content="First")
    algo.process_thread(thread, new_message)

    assert len(thread.messages) == 1
    assert thread.messages[0].content == "First"


def test_fifo_exact_max_messages() -> None:
    algo = FIFOAlgorithm(max_messages=3)
    thread = Thread(
        id="test",
        messages=[
            Message(role=cast(MessageRole, "user"), content="First"),
            Message(role=cast(MessageRole, "assistant"), content="Second"),
        ],
    )

    new_message = Message(role=cast(MessageRole, "user"), content="Third")
    algo.process_thread(thread, new_message)

    assert len(thread.messages) == 3
    assert [msg.content for msg in thread.messages] == ["First", "Second", "Third"]


def test_fifo_large_message_count() -> None:
    algo = FIFOAlgorithm(max_messages=5)
    messages = [
        Message(role=cast(MessageRole, "user"), content=str(i)) for i in range(10)
    ]
    thread = Thread(id="test", messages=messages)

    new_message = Message(role=cast(MessageRole, "user"), content="10")
    algo.process_thread(thread, new_message)

    assert len(thread.messages) == 5
    assert [msg.content for msg in thread.messages] == ["6", "7", "8", "9", "10"]


def test_fifo_metadata_preservation() -> None:
    algo = FIFOAlgorithm(max_messages=2)
    messages = [
        Message(
            role=cast(MessageRole, "user"),
            content="First",
            metadata={"important": True},
        ),
        Message(
            role=cast(MessageRole, "assistant"),
            content="Second",
            metadata={"processed": True},
        ),
    ]
    thread = Thread(id="test", messages=messages)

    new_message = Message(
        role=cast(MessageRole, "user"), content="Third", metadata={"final": True}
    )
    algo.process_thread(thread, new_message)

    assert len(thread.messages) == 2
    assert thread.messages[0].metadata == {"processed": True}
    assert thread.messages[1].metadata == {"final": True}


def test_type_hints_usage() -> None:
    """Test that type hints from types.py are used correctly"""
    # This test ensures types.py is imported and used
    msg: MessageType = Message(content="Test", role="user")
    thread: ThreadType = Thread(messages=[msg])

    algo = FIFOAlgorithm(max_messages=1)
    algo.process_thread(thread, msg)

    assert isinstance(thread, Thread)
    assert isinstance(msg, Message)
    assert len(thread.messages) == 1
