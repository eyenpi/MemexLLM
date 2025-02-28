# LiteLLM Proxy with Langfuse and MemexLLM Integration Example

This example demonstrates how to set up and use a complete OSS LLMOps stack with:

- **LiteLLM Proxy**: A proxy server that provides a unified interface to multiple LLM providers
- **Langfuse**: An open-source LLM observability and evaluation platform
- **MemexLLM**: A conversation history management library for LLMs

## Overview

This integration allows you to:
- Make requests to various LLM providers through a single OpenAI-compatible API
- Automatically log all requests and responses to Langfuse for monitoring and evaluation
- Maintain persistent conversation history across multiple sessions with MemexLLM

## Prerequisites

- Docker and Docker Compose
- OpenAI API key
- Langfuse account (you can sign up for free at [cloud.langfuse.com](https://cloud.langfuse.com))
- Python 3.8+

## Setup

1. **Configure environment variables**

   Edit the `.env` file and add your API keys:

   ```
   # OpenAI API Key
   OPENAI_API_KEY=your_openai_api_key_here

   # LiteLLM Master Key (can be any string you choose)
   LITELLM_MASTER_KEY=sk-master-1234567890

   # Langfuse Credentials (from your Langfuse project)
   LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
   LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

2. **Start the LiteLLM proxy**

   ```bash
   docker-compose up -d
   ```

3. **Install Python dependencies**

   ```bash
   pip install "memexllm[sqlite,openai]" python-dotenv
   ```

## Usage

### Basic Client

Run the basic example client script:

```bash
python client.py
```

This will:
1. Connect to the LiteLLM proxy
2. Send a request to the OpenAI GPT-4o model
3. Print the response
4. Automatically log the request and response to Langfuse

### MemexLLM Integration

Run the MemexLLM integration client:

```bash
python memexllm_client.py
```

This enhanced client:
1. Connects to the LiteLLM proxy
2. Wraps the OpenAI client with MemexLLM's history management
3. Sends multiple requests while maintaining conversation context
4. Logs all interactions to Langfuse
5. Stores conversation history in a SQLite database

The conversation history is persisted between runs, so you can run the script multiple times and continue the same conversation.

## Viewing Results

### Langfuse Traces

After running the examples, you can view the traces in your Langfuse dashboard:

1. Go to [cloud.langfuse.com](https://cloud.langfuse.com)
2. Navigate to the "Traces" section
3. You should see your requests logged with details including:
   - Model used
   - Input and output
   - Token usage
   - Latency
   - Cost (if configured)

### MemexLLM Conversation History

The conversation history is stored in a SQLite database (`conversations.db`) in the current directory. You can view it using any SQLite browser or the SQLite command line tool.

## Customizing the Example

- **Add more models**: Edit `litellm_config.yaml` to add more LLM providers
- **Modify client parameters**: Edit the client scripts to change request parameters
- **Adjust history management**: Modify the FIFO algorithm parameters or use a different algorithm
- **Add custom metadata**: Use the `extra_headers` parameter to add custom metadata to your Langfuse traces

## Learn More

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Langfuse Documentation](https://langfuse.com/docs)
- [MemexLLM Github](https://github.com/memexllm)
- [MemexLLM Documentation](https://eyenpi.github.io/MemexLLM/)
- [OSS LLMOps Stack](https://oss-llmops-stack.com/docs) 