"""
Example demonstrating how to use SQLite storage with OpenAI integration in MemexLLM.

This example shows:
1. Using SQLite for persistent conversation storage
2. Integrating with OpenAI's chat completions
3. Maintaining conversation history across sessions
4. Managing multiple conversation threads
5. Using metadata for conversation tracking

Installation:
    # Install the package with SQLite and OpenAI support
    poetry install --with sqlite,openai

    # Or if using pip
    pip install "memexllm[sqlite,openai]"

Running:
    # Set your OpenAI API key
    export OPENAI_API_KEY=your_api_key_here

    # Run the example
    poetry run python examples/sqlite_openai_integration.py
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI, OpenAI

from memexllm.algorithms import FIFOAlgorithm
from memexllm.core import HistoryManager
from memexllm.integrations.openai import with_history
from memexllm.storage.sqlite import SQLiteStorage


def create_history_manager(db_path: str) -> HistoryManager:
    """Create a HistoryManager with SQLite storage."""
    storage = SQLiteStorage(db_path=db_path)
    algorithm = FIFOAlgorithm(max_messages=10)  # Keep last 10 messages
    return HistoryManager(storage=storage, algorithm=algorithm)


def sync_example(history_manager: HistoryManager) -> None:
    """Example showing persistent conversation history with SQLite storage."""
    # Create and wrap OpenAI client
    client = OpenAI()  # Make sure OPENAI_API_KEY is set in environment
    wrapped_client = with_history(history_manager=history_manager)(client)

    # Create a new math tutoring thread
    math_thread = history_manager.create_thread(
        metadata={
            "topic": "math",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "subject": "basic arithmetic",
        }
    )
    print(f"Created math thread with ID: {math_thread.id}")

    # Start a conversation about math
    messages = [
        {
            "role": "system",
            "content": "You are a helpful math tutor. Keep answers brief.",
        },
        {"role": "user", "content": "What is 15 × 7?"},
    ]

    # First math question
    response1 = wrapped_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, thread_id=math_thread.id
    )
    print("\nMath Thread:")
    print("Q: What is 15 × 7?")
    print("A:", response1.choices[0].message.content)

    # Create another thread for physics
    physics_thread = history_manager.create_thread(
        metadata={
            "topic": "physics",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "subject": "basic mechanics",
        }
    )
    print(f"\nCreated physics thread with ID: {physics_thread.id}")

    # Start a conversation about physics
    physics_messages = [
        {
            "role": "system",
            "content": "You are a helpful physics tutor. Keep answers brief.",
        },
        {"role": "user", "content": "What is Newton's first law?"},
    ]

    # First physics question
    physics_response = wrapped_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=physics_messages, thread_id=physics_thread.id
    )
    print("\nPhysics Thread:")
    print("Q: What is Newton's first law?")
    print("A:", physics_response.choices[0].message.content)

    # Continue math conversation
    math_followup = [{"role": "user", "content": "Now divide that by 3"}]
    response2 = wrapped_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=math_followup, thread_id=math_thread.id
    )
    print("\nBack to Math Thread:")
    print("Q: Now divide that by 3")
    print("A:", response2.choices[0].message.content)

    # List all threads
    print("\nAll conversation threads:")
    threads = history_manager.list_threads()
    for thread in threads:
        print(f"\nThread {thread.id}:")
        print(f"Topic: {thread.metadata.get('topic')}")
        print(f"Subject: {thread.metadata.get('subject')}")
        print("Messages:")
        for msg in thread.messages:
            print(f"{msg.role.upper()}: {msg.content}")


async def async_example(history_manager: HistoryManager) -> None:
    """Async version of the SQLite-backed conversation example."""
    # Create and wrap AsyncOpenAI client
    client = AsyncOpenAI()
    wrapped_client = with_history(history_manager=history_manager)(client)

    # Create a new thread for chemistry
    chem_thread = history_manager.create_thread(
        metadata={
            "topic": "chemistry",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "subject": "basic reactions",
        }
    )
    print(f"\nCreated chemistry thread with ID: {chem_thread.id}")

    # Start conversation about chemistry
    messages = [
        {
            "role": "system",
            "content": "You are a helpful chemistry tutor. Keep answers brief.",
        },
        {"role": "user", "content": "What happens when you mix an acid and a base?"},
    ]

    # First chemistry question
    response1 = await wrapped_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, thread_id=chem_thread.id
    )
    print("\nChemistry Thread:")
    print("Q: What happens when you mix an acid and a base?")
    print("A:", response1.choices[0].message.content)

    # Follow-up question
    messages2 = [{"role": "user", "content": "Give an example of this reaction"}]
    response2 = await wrapped_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages2, thread_id=chem_thread.id
    )
    print("\nQ: Give an example of this reaction")
    print("A:", response2.choices[0].message.content)

    # Print final conversation state
    print("\nFinal Chemistry Thread:")
    thread = history_manager.get_thread(chem_thread.id)
    for msg in thread.messages:
        print(f"{msg.role.upper()}: {msg.content}")


def main():
    """Run both sync and async examples with persistent storage."""
    # Use a persistent database file
    db_path = "conversations.db"

    # Create history manager with SQLite storage
    history_manager = create_history_manager(db_path)

    print("Running synchronous example...")
    sync_example(history_manager)

    print("\nRunning asynchronous example...")
    asyncio.run(async_example(history_manager))

    print("\nAll conversations have been saved to:", db_path)
    print("You can run this example multiple times to see how conversations persist!")


if __name__ == "__main__":
    main()
