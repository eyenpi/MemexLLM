from memexllm.core.history import HistoryManager
from memexllm.storage.memory import MemoryStorage
from memexllm.algorithms.fifo import FIFOAlgorithm

def main():
    # Initialize storage backend
    storage = MemoryStorage()
    
    # Initialize history algorithm (optional)
    algorithm = FIFOAlgorithm(max_messages=50)
    
    # Create history manager
    history_manager = HistoryManager(storage=storage, algorithm=algorithm)
    
    # Create a new conversation thread
    thread = history_manager.create_thread(metadata={"user_id": "user123"})
    print(f"Created thread with ID: {thread.id}")
    
    # Add messages to the thread
    history_manager.add_message(
        thread_id=thread.id,
        content="Hello, how can you help me with my project?",
        role="user"
    )
    
    history_manager.add_message(
        thread_id=thread.id,
        content="I'm an AI assistant. I'd be happy to help with your project. What are you working on?",
        role="assistant"
    )
    
    # Retrieve the thread
    retrieved_thread = history_manager.get_thread(thread.id)
    print(f"Thread has {len(retrieved_thread.messages)} messages")
    
    # Print all messages
    for msg in retrieved_thread.messages:
        print(f"[{msg.role}]: {msg.content}")
    
    # List all threads
    all_threads = history_manager.list_threads()
    print(f"Total threads: {len(all_threads)}")
    
if __name__ == "__main__":
    main()