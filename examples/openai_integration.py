import asyncio

from openai import AsyncOpenAI, OpenAI

from memexllm.algorithms.fifo import FIFOAlgorithm
from memexllm.core.history import HistoryManager
from memexllm.integrations.openai import with_history
from memexllm.storage.memory import MemoryStorage


def sync_example() -> None:
    """Example showing how the history-enabled client maintains conversation context."""
    # Create storage, algorithm, and history manager instances
    storage = MemoryStorage()
    algorithm = FIFOAlgorithm(max_messages=10)  # Keep last 10 messages
    history_manager = HistoryManager(storage=storage, algorithm=algorithm)

    # Create and wrap OpenAI client
    client = OpenAI()  # Make sure OPENAI_API_KEY is set in environment
    wrapped_client = with_history(history_manager=history_manager)(client)

    # Create a new thread using history manager
    thread = history_manager.create_thread()

    # Start a conversation with a system message
    messages = [
        {
            "role": "system",
            "content": "You are a helpful math tutor. Keep answers brief.",
        },
        {"role": "user", "content": "What is 2+2?"},
    ]

    # First request - use the created thread
    response1 = wrapped_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, thread_id=thread.id
    )
    print("Q: What is 2+2?")
    print("A:", response1.choices[0].message.content)

    # Second request - reuse the same thread
    messages2 = [{"role": "user", "content": "Now multiply that by 2"}]
    response2 = wrapped_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages2, thread_id=thread.id
    )
    print("\nQ: Now multiply that by 2")
    print("A:", response2.choices[0].message.content)

    # Third request - the model remembers the entire conversation
    messages3 = [
        {"role": "user", "content": "What calculations did we do to get here?"}
    ]
    response3 = wrapped_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages3, thread_id=thread.id
    )
    print("\nQ: What calculations did we do to get here?")
    print("A:", response3.choices[0].message.content)

    # Print conversation history
    print("\nFull Conversation History:")
    thread = history_manager.get_thread(thread.id)
    for msg in thread.messages:
        print(f"{msg.role.upper()}: {msg.content}")


async def async_example() -> None:
    """Async version of the history-enabled client example."""
    # Create storage, algorithm, and history manager instances
    storage = MemoryStorage()
    algorithm = FIFOAlgorithm(max_messages=10)
    history_manager = HistoryManager(storage=storage, algorithm=algorithm)

    # Create and wrap AsyncOpenAI client
    client = AsyncOpenAI()  # Make sure OPENAI_API_KEY is set in environment
    wrapped_client = with_history(history_manager=history_manager)(client)

    # Create a new thread using history manager
    thread = history_manager.create_thread()

    # Start conversation
    messages = [
        {
            "role": "system",
            "content": "You are a helpful math tutor. Keep answers brief.",
        },
        {"role": "user", "content": "What is 5+7?"},
    ]

    # First request
    response1 = await wrapped_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, thread_id=thread.id
    )
    print("Q: What is 5+7?")
    print("A:", response1.choices[0].message.content)

    # Second request - context is maintained
    messages2 = [{"role": "user", "content": "Divide that by 3"}]
    response2 = await wrapped_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages2, thread_id=thread.id
    )
    print("\nQ: Divide that by 3")
    print("A:", response2.choices[0].message.content)

    # Print conversation history
    print("\nFull Conversation History:")
    thread = history_manager.get_thread(thread.id)
    for msg in thread.messages:
        print(f"{msg.role.upper()}: {msg.content}")


if __name__ == "__main__":
    print("Running synchronous example...")
    sync_example()

    print("\nRunning asynchronous example...")
    asyncio.run(async_example())
