"""
Example demonstrating the difference between storage capacity and algorithm context window.

This example shows how:
1. Storage controls how many messages are actually saved
2. Algorithms control how many messages are used for context
3. These two concerns are independent of each other
"""

from memexllm.algorithms import FIFOAlgorithm
from memexllm.core import HistoryManager
from memexllm.core.models import Thread
from memexllm.storage.memory import MemoryStorage


def print_thread_info(thread: Thread, source: str) -> None:
    """Helper to print thread information"""
    print(f"\n{source}:")
    print(f"Number of messages: {len(thread.messages)}")
    if thread.messages:
        print("Message contents:")
        for msg in thread.messages:
            print(f"- {msg.content}")


def main() -> None:
    # Example 1: Storage with more capacity than algorithm window
    print("\n=== Example 1: Storage(1000) > Algorithm(5) ===")
    storage = MemoryStorage(max_messages=1000)  # Store up to 1000 messages
    algorithm = FIFOAlgorithm(max_messages=5)  # But only use last 5 for context
    manager = HistoryManager(storage=storage, algorithm=algorithm)

    # Create a thread and add 10 messages
    thread = manager.create_thread()
    for i in range(10):
        manager.add_message(thread_id=thread.id, content=f"Message {i}", role="user")

    # Show what's in storage vs what algorithm returns
    raw_thread = storage.get_thread(thread.id)  # Get directly from storage
    algo_thread = manager.get_thread(thread.id)  # Get through manager (uses algorithm)

    print_thread_info(raw_thread, "Storage contents (up to 1000 messages)")
    print_thread_info(algo_thread, "Algorithm window (last 5 messages)")

    # Example 2: Storage with less capacity than algorithm window
    print("\n=== Example 2: Storage(3) < Algorithm(5) ===")
    storage = MemoryStorage(max_messages=3)  # Only store last 3 messages
    algorithm = FIFOAlgorithm(max_messages=5)  # Try to use last 5 for context
    manager = HistoryManager(storage=storage, algorithm=algorithm)

    # Create a thread and add 10 messages
    thread = manager.create_thread()
    for i in range(10):
        manager.add_message(thread_id=thread.id, content=f"Message {i}", role="user")

    # Show what's in storage vs what algorithm returns
    raw_thread = storage.get_thread(thread.id)  # Get directly from storage
    algo_thread = manager.get_thread(thread.id)  # Get through manager (uses algorithm)

    print_thread_info(raw_thread, "Storage contents (up to 3 messages)")
    print_thread_info(
        algo_thread, "Algorithm window (tries for 5, but only 3 available)"
    )

    # Example 3: Storage without algorithm
    print("\n=== Example 3: Only Storage(3), No Algorithm ===")
    storage = MemoryStorage(max_messages=3)  # Only store last 3 messages
    manager = HistoryManager(storage=storage)  # No algorithm

    # Create a thread and add 10 messages
    thread = manager.create_thread()
    for i in range(10):
        manager.add_message(thread_id=thread.id, content=f"Message {i}", role="user")

    # Show what's in storage
    thread = manager.get_thread(thread.id)
    print_thread_info(thread, "Storage contents (up to 3 messages)")

    # Example 4: Algorithm without storage limits
    print("\n=== Example 4: Unlimited Storage, Algorithm(3) ===")
    storage = MemoryStorage()  # No storage limit
    algorithm = FIFOAlgorithm(max_messages=3)  # But only use last 3 for context
    manager = HistoryManager(storage=storage, algorithm=algorithm)

    # Create a thread and add 10 messages
    thread = manager.create_thread()
    for i in range(10):
        manager.add_message(thread_id=thread.id, content=f"Message {i}", role="user")

    # Show what's in storage vs what algorithm returns
    raw_thread = storage.get_thread(thread.id)  # Get directly from storage
    algo_thread = manager.get_thread(thread.id)  # Get through manager (uses algorithm)

    print_thread_info(raw_thread, "Storage contents (unlimited)")
    print_thread_info(algo_thread, "Algorithm window (last 3 messages)")


if __name__ == "__main__":
    main()
