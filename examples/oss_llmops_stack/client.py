#!/usr/bin/env python3
"""
Example client for LiteLLM proxy with Langfuse integration.
This script demonstrates how to make requests to the LiteLLM proxy
which automatically logs traces to Langfuse.
"""

import os

import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LiteLLM Proxy configuration
LITELLM_PROXY_BASE_URL = "http://localhost:4000"
LITELLM_VIRTUAL_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-master-1234567890")


def main():
    # Initialize OpenAI client with LiteLLM proxy
    client = openai.OpenAI(api_key=LITELLM_VIRTUAL_KEY, base_url=LITELLM_PROXY_BASE_URL)

    # Create a conversation with multiple messages to demonstrate tracing
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant that provides concise answers.",
        },
        {
            "role": "user",
            "content": "What is Langfuse and how does it integrate with LiteLLM?",
        },
    ]

    # Make the request to the LiteLLM proxy
    print("Sending request to LiteLLM proxy...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

    # Print the response
    print("\nResponse from LiteLLM proxy:")
    print(f"Model: {response.model}")
    print(f"Completion tokens: {response.usage.completion_tokens}")
    print(f"Prompt tokens: {response.usage.prompt_tokens}")
    print(f"Total tokens: {response.usage.total_tokens}")
    print("\nAssistant's response:")
    print(response.choices[0].message.content)

    # The request has been automatically logged to Langfuse
    print("\nThis request has been automatically logged to Langfuse.")
    print("Check your Langfuse dashboard to see the trace.")


if __name__ == "__main__":
    main()
