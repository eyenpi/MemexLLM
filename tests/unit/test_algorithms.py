import pytest
from memexllm.algorithms.fifo import FIFOAlgorithm
from memexllm.core.models import Message, Thread
from memexllm.algorithms.base import BaseAlgorithm

def test_base_algorithm():
    # Create a concrete class that inherits from BaseAlgorithm but doesn't implement process_thread
    class TestAlgorithm(BaseAlgorithm):
        pass
        
    with pytest.raises(TypeError) as exc_info:
        TestAlgorithm()
    
    error_msg = str(exc_info.value)
    assert "abstract class" in error_msg
    assert "process_thread" in error_msg

def test_fifo_algorithm():
    algo = FIFOAlgorithm(max_messages=2)
    
    thread = Thread(
        id="test",
        messages=[
            Message(role="user", content="First"),
            Message(role="assistant", content="Second"),
        ]
    )
    
    new_message = Message(role="user", content="Third")
    algo.process_thread(thread, new_message)  # Modifies thread in-place
    
    assert len(thread.messages) == 2
    assert thread.messages[0].content == "Second"
    assert thread.messages[1].content == "Third"

def test_fifo_with_max_messages():
    algo = FIFOAlgorithm(max_messages=3)
    messages = [Message(role="user", content=str(i)) for i in range(4)]
    thread = Thread(id="test", messages=messages)
    
    new_message = Message(role="user", content="4")
    algo.process_thread(thread, new_message)  # Modifies thread in-place
    
    assert len(thread.messages) == 3
    assert thread.messages[0].content == "2"
    assert thread.messages[1].content == "3"
    assert thread.messages[2].content == "4"

def test_fifo_empty_thread():
    algo = FIFOAlgorithm(max_messages=2)
    thread = Thread(id="test")
    
    new_message = Message(role="user", content="First")
    algo.process_thread(thread, new_message)
    
    assert len(thread.messages) == 1
    assert thread.messages[0].content == "First"

def test_fifo_exact_max_messages():
    algo = FIFOAlgorithm(max_messages=3)
    thread = Thread(
        id="test",
        messages=[
            Message(role="user", content="First"),
            Message(role="assistant", content="Second"),
        ]
    )
    
    new_message = Message(role="user", content="Third")
    algo.process_thread(thread, new_message)
    
    assert len(thread.messages) == 3
    assert [msg.content for msg in thread.messages] == ["First", "Second", "Third"]

def test_fifo_large_message_count():
    algo = FIFOAlgorithm(max_messages=5)
    messages = [Message(role="user", content=str(i)) for i in range(10)]
    thread = Thread(id="test", messages=messages)
    
    new_message = Message(role="user", content="10")
    algo.process_thread(thread, new_message)
    
    assert len(thread.messages) == 5
    assert [msg.content for msg in thread.messages] == ["6", "7", "8", "9", "10"]

def test_fifo_metadata_preservation():
    algo = FIFOAlgorithm(max_messages=2)
    messages = [
        Message(role="user", content="First", metadata={"important": True}),
        Message(role="assistant", content="Second", metadata={"processed": True})
    ]
    thread = Thread(id="test", messages=messages)
    
    new_message = Message(role="user", content="Third", metadata={"final": True})
    algo.process_thread(thread, new_message)
    
    assert len(thread.messages) == 2
    assert thread.messages[0].metadata == {"processed": True}
    assert thread.messages[1].metadata == {"final": True} 