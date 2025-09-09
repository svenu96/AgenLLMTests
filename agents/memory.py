"""
Memory Systems for Agents

Different types of memory systems for managing conversation history
and knowledge retention.
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import List, Dict, Any, Optional
import json
import uuid
from datetime import datetime

from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
import numpy as np


class BaseMemory(ABC):
    """Base class for all memory systems."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.created_at = datetime.now()
    
    @abstractmethod
    def add_message(self, message: BaseMessage):
        """Add a message to memory."""
        pass
    
    @abstractmethod
    def get_relevant_messages(self, query: Optional[str] = None, limit: int = 10) -> List[BaseMessage]:
        """Get relevant messages from memory."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all memory."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        pass


class ConversationMemory(BaseMemory):
    """
    Simple conversation memory that maintains recent messages.
    
    Good for short-term context and basic conversation flow.
    """
    
    def __init__(self, max_size: int = 20):
        super().__init__(max_size)
        self.messages = deque(maxlen=max_size)
        self.message_count = 0
    
    def add_message(self, message: BaseMessage):
        """Add a message to the conversation memory."""
        self.messages.append({
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(),
            "message": message,
            "content": message.content,
            "type": type(message).__name__
        })
        self.message_count += 1
    
    def get_relevant_messages(self, query: Optional[str] = None, limit: int = 10) -> List[BaseMessage]:
        """
        Get recent messages from conversation memory.
        
        Args:
            query: Not used in conversation memory
            limit: Maximum number of messages to return
            
        Returns:
            List of recent messages
        """
        # Return the most recent messages
        recent_messages = list(self.messages)[-limit:]
        return [msg["message"] for msg in recent_messages]
    
    def get_messages_by_type(self, message_type: str) -> List[BaseMessage]:
        """Get messages of a specific type."""
        return [
            msg["message"] for msg in self.messages
            if msg["type"] == message_type
        ]
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation."""
        human_count = len(self.get_messages_by_type("HumanMessage"))
        ai_count = len(self.get_messages_by_type("AIMessage"))
        system_count = len(self.get_messages_by_type("SystemMessage"))
        
        return {
            "total_messages": len(self.messages),
            "human_messages": human_count,
            "ai_messages": ai_count,
            "system_messages": system_count,
            "memory_utilization": len(self.messages) / self.max_size
        }
    
    def clear(self):
        """Clear all conversation memory."""
        self.messages.clear()
        self.message_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "type": "ConversationMemory",
            "max_size": self.max_size,
            "current_size": len(self.messages),
            "total_processed": self.message_count,
            "created_at": self.created_at.isoformat(),
            **self.get_conversation_summary()
        }


class VectorMemory(BaseMemory):
    """
    Vector-based memory using embeddings for semantic search.
    
    Good for long-term memory and finding relevant context based on similarity.
    """
    
    def __init__(self, max_size: int = 1000, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__(max_size)
        self.embedding_model = embedding_model
        self.messages_store = []
        self.message_count = 0
        
        # Initialize embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            self.vector_store = None
            self._initialize_vector_store()
        except Exception as e:
            print(f"Warning: Could not initialize embeddings: {e}")
            self.embeddings = None
            self.vector_store = None
    
    def _initialize_vector_store(self):
        """Initialize the vector store."""
        if self.embeddings:
            try:
                # Create a temporary vector store (in practice, you'd want persistent storage)
                self.vector_store = Chroma(embedding_function=self.embeddings)
            except Exception as e:
                print(f"Warning: Could not initialize vector store: {e}")
                self.vector_store = None
    
    def add_message(self, message: BaseMessage):
        """Add a message to vector memory."""
        message_data = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(),
            "message": message,
            "content": message.content,
            "type": type(message).__name__
        }
        
        self.messages_store.append(message_data)
        self.message_count += 1
        
        # Add to vector store if available
        if self.vector_store and message.content.strip():
            try:
                self.vector_store.add_texts(
                    texts=[message.content],
                    metadatas=[{
                        "id": message_data["id"],
                        "timestamp": message_data["timestamp"].isoformat(),
                        "type": message_data["type"]
                    }]
                )
            except Exception as e:
                print(f"Warning: Could not add to vector store: {e}")
        
        # Maintain max size
        if len(self.messages_store) > self.max_size:
            self.messages_store.pop(0)
    
    def get_relevant_messages(self, query: Optional[str] = None, limit: int = 10) -> List[BaseMessage]:
        """
        Get relevant messages using vector similarity search.
        
        Args:
            query: Query to search for similar messages
            limit: Maximum number of messages to return
            
        Returns:
            List of relevant messages
        """
        if not query or not self.vector_store:
            # Fallback to recent messages
            recent_messages = self.messages_store[-limit:]
            return [msg["message"] for msg in recent_messages]
        
        try:
            # Perform similarity search
            docs = self.vector_store.similarity_search(query, k=limit)
            
            # Get corresponding messages
            relevant_messages = []
            for doc in docs:
                # Find the message by content (in practice, you'd use IDs)
                for msg_data in self.messages_store:
                    if msg_data["content"] == doc.page_content:
                        relevant_messages.append(msg_data["message"])
                        break
            
            return relevant_messages
            
        except Exception as e:
            print(f"Warning: Vector search failed: {e}")
            # Fallback to recent messages
            recent_messages = self.messages_store[-limit:]
            return [msg["message"] for msg in recent_messages]
    
    def search_by_content(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for messages by content similarity.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of message data with similarity scores
        """
        if not self.vector_store:
            return []
        
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=limit)
            
            results = []
            for doc, score in docs_with_scores:
                # Find corresponding message data
                for msg_data in self.messages_store:
                    if msg_data["content"] == doc.page_content:
                        results.append({
                            "message_data": msg_data,
                            "similarity_score": score,
                            "content": doc.page_content
                        })
                        break
            
            return results
            
        except Exception as e:
            print(f"Warning: Content search failed: {e}")
            return []
    
    def clear(self):
        """Clear all vector memory."""
        self.messages_store.clear()
        self.message_count = 0
        
        # Reinitialize vector store
        if self.embeddings:
            self._initialize_vector_store()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "type": "VectorMemory",
            "max_size": self.max_size,
            "current_size": len(self.messages_store),
            "total_processed": self.message_count,
            "embedding_model": self.embedding_model,
            "vector_store_available": self.vector_store is not None,
            "created_at": self.created_at.isoformat()
        }


class HybridMemory(BaseMemory):
    """
    Hybrid memory combining conversation and vector memory.
    
    Maintains both recent conversation context and long-term semantic memory.
    """
    
    def __init__(self, max_size: int = 1000, recent_limit: int = 20):
        super().__init__(max_size)
        self.conversation_memory = ConversationMemory(max_size=recent_limit)
        self.vector_memory = VectorMemory(max_size=max_size)
        self.recent_limit = recent_limit
    
    def add_message(self, message: BaseMessage):
        """Add message to both memory systems."""
        self.conversation_memory.add_message(message)
        self.vector_memory.add_message(message)
    
    def get_relevant_messages(self, query: Optional[str] = None, limit: int = 10) -> List[BaseMessage]:
        """
        Get relevant messages from both memory systems.
        
        Combines recent conversation context with semantically relevant historical messages.
        """
        # Get recent messages from conversation memory
        recent_messages = self.conversation_memory.get_relevant_messages(limit=self.recent_limit // 2)
        
        # Get relevant historical messages from vector memory
        historical_limit = limit - len(recent_messages)
        if historical_limit > 0 and query:
            historical_messages = self.vector_memory.get_relevant_messages(query, historical_limit)
            
            # Remove duplicates (messages that are both recent and relevant)
            recent_content = {msg.content for msg in recent_messages}
            historical_messages = [
                msg for msg in historical_messages 
                if msg.content not in recent_content
            ]
            
            # Combine recent and historical
            return recent_messages + historical_messages[:historical_limit]
        
        return recent_messages
    
    def clear(self):
        """Clear both memory systems."""
        self.conversation_memory.clear()
        self.vector_memory.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined memory statistics."""
        conv_stats = self.conversation_memory.get_stats()
        vector_stats = self.vector_memory.get_stats()
        
        return {
            "type": "HybridMemory",
            "conversation_memory": conv_stats,
            "vector_memory": vector_stats,
            "total_unique_messages": vector_stats["current_size"],
            "recent_messages": conv_stats["current_size"]
        }


def demo_memory_systems():
    """Demonstrate different memory systems."""
    print("🧠 Memory Systems Demo")
    print("=" * 50)
    
    # Test messages
    test_messages = [
        HumanMessage(content="Hello, I'm learning about AI"),
        AIMessage(content="Great! AI is a fascinating field. What specifically interests you?"),
        HumanMessage(content="I want to understand neural networks"),
        AIMessage(content="Neural networks are the foundation of deep learning. They consist of layers of interconnected nodes."),
        HumanMessage(content="Can you explain transformers?"),
        AIMessage(content="Transformers are a type of neural network architecture that uses attention mechanisms for processing sequences."),
        HumanMessage(content="What about machine learning?"),
        AIMessage(content="Machine learning is a subset of AI that enables systems to learn from data without explicit programming.")
    ]
    
    # Test Conversation Memory
    print("\n💬 Conversation Memory Test:")
    conv_memory = ConversationMemory(max_size=10)
    
    for msg in test_messages:
        conv_memory.add_message(msg)
    
    print(f"Stats: {conv_memory.get_stats()}")
    
    recent = conv_memory.get_relevant_messages(limit=3)
    print(f"Recent messages: {len(recent)}")
    
    # Test Vector Memory
    print("\n🔍 Vector Memory Test:")
    vector_memory = VectorMemory(max_size=100)
    
    for msg in test_messages:
        vector_memory.add_message(msg)
    
    print(f"Stats: {vector_memory.get_stats()}")
    
    relevant = vector_memory.get_relevant_messages("neural networks", limit=2)
    print(f"Relevant to 'neural networks': {len(relevant)} messages")
    
    # Test Hybrid Memory
    print("\n🔀 Hybrid Memory Test:")
    hybrid_memory = HybridMemory(max_size=100, recent_limit=5)
    
    for msg in test_messages:
        hybrid_memory.add_message(msg)
    
    print(f"Stats: {hybrid_memory.get_stats()}")
    
    combined = hybrid_memory.get_relevant_messages("transformers", limit=4)
    print(f"Combined relevant messages: {len(combined)}")


if __name__ == "__main__":
    demo_memory_systems()