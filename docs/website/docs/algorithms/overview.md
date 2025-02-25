# Context Algorithms

MemexLLM uses context algorithms to manage how conversation history is presented to LLMs. These algorithms ensure optimal context while respecting token limits.

## Available Algorithms

### FIFO Algorithm
- First-In-First-Out message selection
- Simple and predictable behavior
- Great for most use cases
- [Learn more about FIFO Algorithm](./fifo.md)

## Coming Soon

We're working on additional context algorithms:

### Semantic Search
- Select messages based on relevance
- Use embeddings for similarity

### Token-Aware
- Smart token counting
- Model-specific optimizations

### Time-Based
- Select messages by recency
- Configurable time windows

### Hybrid
- Combine multiple algorithms
- Custom selection strategies

## Algorithm Features

All context algorithms in MemexLLM provide:

- **Message Selection**: Choose which messages to include in context
- **Token Management**: Respect LLM token limits
- **Thread Support**: Work with conversation threads
- **Metadata Support**: Use metadata for selection
- **Type Safety**: Full type hints support

## Custom Algorithms

Need a different selection strategy? You can create your own:
- Implement the algorithm interface
- Full control over message selection
- [Learn how to create custom algorithms](./custom.md)

## Next Steps

- Try [FIFO Algorithm](./fifo.md) for simple cases
- Learn to [create custom algorithms](./custom.md)
- See [example applications](../examples/simple_chatbot.md) 