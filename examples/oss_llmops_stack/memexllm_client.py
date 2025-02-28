#!/usr/bin/env python3
"""
Example client for LiteLLM proxy with Langfuse integration and MemexLLM history management.

This script demonstrates how to combine:
1. LiteLLM proxy for model routing
2. Langfuse for observability
3. MemexLLM for conversation history management

The integration provides:
- A unified API for multiple LLM providers through LiteLLM
- Automatic logging of all requests to Langfuse
- Persistent conversation history with MemexLLM

Requirements:
- Running LiteLLM proxy (start with docker-compose up -d)
- Install required packages: `pip install "memexllm[sqlite,openai]" python-dotenv`
"""

import os
import uuid

from dotenv import load_dotenv
from openai import OpenAI

from memexllm.algorithms import FIFOAlgorithm
from memexllm.core import HistoryManager
from memexllm.integrations.openai import with_history
from memexllm.storage import SQLiteStorage

# Load environment variables from .env file
load_dotenv()

# LiteLLM Proxy configuration
LITELLM_PROXY_BASE_URL = "http://localhost:4000"
LITELLM_VIRTUAL_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-master-1234567890")

# MemexLLM configuration
DB_PATH = "conversations.db"
MODEL = "gpt-4o"  # This will be routed through LiteLLM


def main():
    """Run the example demonstrating memory capabilities."""
    # Initialize OpenAI client with LiteLLM proxy
    client = OpenAI(api_key=LITELLM_VIRTUAL_KEY, base_url=LITELLM_PROXY_BASE_URL)

    # Set up SQLite storage for conversation history
    storage = SQLiteStorage(DB_PATH)

    # Use FIFO algorithm to limit history to last 20 messages
    algorithm = FIFOAlgorithm(max_messages=20)

    # Create HistoryManager
    history_manager = HistoryManager(storage=storage, algorithm=algorithm)

    # Wrap the OpenAI client with history management
    client = with_history(history_manager=history_manager)(client)

    # Generate a unique ID to test memory
    unique_id = str(uuid.uuid4())[:8]

    print("\n=== Memory Capability Demonstration ===")

    # First message - introduce a unique fact
    print(f"\n1. Sending message with unique ID: {unique_id}")

    first_message = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant with excellent memory. Keep your answers concise.",
        },
        {
            "role": "user",
            "content": f"My favorite number is {unique_id}. What can you tell me about the OSS LLMOps stack?",
        },
    ]

    # Create a new thread for this conversation
    thread = history_manager.create_thread()
    thread_id = thread.id

    # Prepare Langfuse metadata with thread_id
    langfuse_metadata = {
        "metadata": {
            "thread_id": thread_id,
            "conversation_name": f"MemexLLM Demo {unique_id}",
        }
    }

    response = client.chat.completions.create(
        model=MODEL,
        messages=first_message,
        thread_id=thread_id,
        extra_body=langfuse_metadata,
    )

    print("\nResponse:")
    print(f"Model: {response.model}")
    print(f"Total tokens: {response.usage.total_tokens}")
    print("\nAssistant's response:")
    print(response.choices[0].message.content)

    # Second message - ask about a different topic
    print("\n2. Sending message about a different topic...")

    second_message = [
        {
            "role": "user",
            "content": "How does MemexLLM help with conversation history management?",
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=second_message,
        thread_id=thread_id,
        extra_body=langfuse_metadata,
    )

    print("\nResponse:")
    print(f"Model: {response.model}")
    print(f"Total tokens: {response.usage.total_tokens}")
    print("\nAssistant's response:")
    print(response.choices[0].message.content)

    # Third message - test if the model remembers the unique ID
    print("\n3. Testing if the model remembers the unique ID...")

    third_message = [
        {
            "role": "user",
            "content": "What was my favorite number that I mentioned earlier?",
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=third_message,
        thread_id=thread_id,
        extra_body=langfuse_metadata,
    )

    print("\nResponse:")
    print(f"Model: {response.model}")
    print(f"Total tokens: {response.usage.total_tokens}")
    print("\nAssistant's response:")
    print(response.choices[0].message.content)

    # Check if the unique ID is in the response
    if unique_id in response.choices[0].message.content:
        print(
            f"\n✅ Memory Test PASSED: The model correctly remembered the unique ID: {unique_id}"
        )
    else:
        print(
            f"\n❌ Memory Test FAILED: The model did not remember the unique ID: {unique_id}"
        )

    print("\n=== End of Demonstration ===")
    print(f"\nThis entire conversation has been:")
    print(f"1. Routed through the LiteLLM proxy")
    print(f"2. Logged to Langfuse for observability")
    print(f"3. Saved to {DB_PATH} by MemexLLM for history management")
    print("\nYou can run this script again to continue the conversation.")


if __name__ == "__main__":
    main()
