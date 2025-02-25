# Integrations Overview

MemexLLM provides seamless integrations with popular LLM providers. These integrations allow you to add conversation management to your existing LLM applications with minimal code changes.

## Available Integrations

### OpenAI
- Full support for OpenAI's chat completions API
- Works with both sync and async clients
- Supports all OpenAI chat models
- [Learn more about OpenAI integration](./openai.md)

## Coming Soon

We're working on integrations with other popular LLM providers:

### Anthropic Claude
- Support for Claude and Claude 2 models
- Integration with official Anthropic Python client

### Azure OpenAI
- Support for Azure-hosted OpenAI models
- Compatible with Azure OpenAI client

### LiteLLM
- Universal support for multiple LLM providers
- Compatible with LiteLLM's unified interface

### Llama.cpp
- Support for local LLM models
- Integration with llama.cpp Python bindings

## Integration Features

All MemexLLM integrations provide:

- **Automatic History**: Conversations are automatically stored and managed
- **Thread Organization**: Keep different conversations separate
- **Persistent Storage**: Conversation history survives application restarts
- **Context Management**: Smart selection of relevant conversation history
- **Metadata Support**: Track additional information with conversations

## Next Steps

- Get started with [OpenAI integration](./openai.md)
- Learn about [storage options](../storage/overview.md)
- Explore [context algorithms](../algorithms/overview.md) 