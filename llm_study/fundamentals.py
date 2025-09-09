"""
LLM Fundamentals

Core concepts and basic implementations for understanding Large Language Models.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np


class LLMBasics:
    """
    Educational class for understanding LLM fundamentals.
    Provides interactive examples and visualizations.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """Initialize with a small model for learning purposes."""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the tokenizer and model."""
        try:
            print(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Try using 'distilbert-base-uncased' for a smaller model")
    
    def tokenization_demo(self, text: str) -> Dict[str, Any]:
        """
        Demonstrate tokenization process with visualizations.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Dictionary with tokenization results
        """
        if not self.tokenizer:
            return {"error": "Tokenizer not loaded"}
        
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.encode(text)
        
        # Create visualization
        print(f"Original text: '{text}'")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        
        # Token-to-ID mapping
        token_mapping = {}
        for i, token in enumerate(tokens):
            token_id = self.tokenizer.convert_tokens_to_ids([token])[0]
            token_mapping[token] = token_id
        
        return {
            "original_text": text,
            "tokens": tokens,
            "token_ids": token_ids,
            "token_mapping": token_mapping,
            "vocab_size": self.tokenizer.vocab_size
        }
    
    def attention_visualization(self, text: str, layer: int = 0):
        """
        Visualize attention patterns for input text.
        
        Args:
            text: Input text
            layer: Which attention layer to visualize
        """
        if not self.model or not self.tokenizer:
            print("❌ Model or tokenizer not loaded")
            return
        
        # Tokenize and encode
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Extract attention weights
        attention = outputs.attentions[layer][0]  # First layer, first batch
        
        # Average across attention heads
        attention_avg = attention.mean(dim=0)
        
        # Get tokens for visualization
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(attention_avg.numpy(), cmap='Blues')
        plt.colorbar(label='Attention Weight')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.title(f'Attention Patterns - Layer {layer}')
        
        # Set tick labels
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        plt.yticks(range(len(tokens)), tokens)
        plt.tight_layout()
        plt.show()
        
        return {
            "tokens": tokens,
            "attention_weights": attention_avg.numpy(),
            "layer": layer
        }
    
    def embedding_exploration(self, words: List[str]):
        """
        Explore word embeddings and their relationships.
        
        Args:
            words: List of words to analyze
        """
        if not self.model or not self.tokenizer:
            print("❌ Model or tokenizer not loaded")
            return
        
        embeddings = []
        valid_words = []
        
        for word in words:
            # Tokenize word
            inputs = self.tokenizer(word, return_tensors="pt")
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling for word representation
                embedding = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(embedding.squeeze().numpy())
                valid_words.append(word)
        
        if len(embeddings) < 2:
            print("Need at least 2 valid words for comparison")
            return
        
        # Calculate similarities
        embeddings = np.array(embeddings)
        similarities = np.dot(embeddings, embeddings.T)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1)
        similarities = similarities / np.outer(norms, norms)
        
        # Visualize similarity matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(similarities, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Cosine Similarity')
        plt.xlabel('Words')
        plt.ylabel('Words')
        plt.title('Word Embedding Similarities')
        plt.xticks(range(len(valid_words)), valid_words, rotation=45)
        plt.yticks(range(len(valid_words)), valid_words)
        plt.tight_layout()
        plt.show()
        
        return {
            "words": valid_words,
            "embeddings": embeddings,
            "similarities": similarities
        }


class TransformerConcepts:
    """
    Detailed explanation and implementation of Transformer concepts.
    """
    
    @staticmethod
    def attention_mechanism_demo():
        """
        Demonstrate the attention mechanism with a simple example.
        """
        print("🎯 Attention Mechanism Demo")
        print("=" * 50)
        
        # Simple example with 3 words
        words = ["The", "cat", "sat"]
        
        # Simulate query, key, value vectors (simplified)
        d_model = 4  # embedding dimension
        
        # Random vectors for demonstration
        torch.manual_seed(42)
        Q = torch.randn(3, d_model)  # Queries
        K = torch.randn(3, d_model)  # Keys  
        V = torch.randn(3, d_model)  # Values
        
        print(f"Words: {words}")
        print(f"Query shape: {Q.shape}")
        print(f"Key shape: {K.shape}")
        print(f"Value shape: {V.shape}")
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))
        print(f"\nAttention Scores:\n{scores}")
        
        # Apply softmax
        attention_weights = torch.softmax(scores / np.sqrt(d_model), dim=-1)
        print(f"\nAttention Weights (after softmax):\n{attention_weights}")
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        print(f"\nOutput (attended values):\n{output}")
        
        return {
            "words": words,
            "attention_weights": attention_weights.numpy(),
            "output": output.numpy()
        }
    
    @staticmethod
    def positional_encoding_demo(seq_len: int = 10, d_model: int = 8):
        """
        Demonstrate positional encoding.
        
        Args:
            seq_len: Sequence length
            d_model: Model dimension
        """
        print("📍 Positional Encoding Demo")
        print("=" * 50)
        
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pos_encoding = torch.zeros(seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        # Visualize
        plt.figure(figsize=(12, 6))
        plt.imshow(pos_encoding.T, cmap='RdYlBu', aspect='auto')
        plt.colorbar(label='Value')
        plt.xlabel('Position')
        plt.ylabel('Dimension')
        plt.title('Positional Encoding Pattern')
        plt.tight_layout()
        plt.show()
        
        print(f"Positional encoding shape: {pos_encoding.shape}")
        print(f"First position encoding: {pos_encoding[0]}")
        
        return pos_encoding.numpy()


# Example usage and demonstrations
def run_fundamentals_demo():
    """Run a comprehensive demo of LLM fundamentals."""
    print("🚀 LLM Fundamentals Demo")
    print("=" * 50)
    
    # Initialize LLM Basics
    llm = LLMBasics()
    
    # 1. Tokenization Demo
    print("\n1. 📝 Tokenization Demo")
    text = "Hello, how are you today?"
    result = llm.tokenization_demo(text)
    print(f"Vocab size: {result.get('vocab_size', 'N/A')}")
    
    # 2. Transformer Concepts
    print("\n2. 🔄 Transformer Concepts")
    concepts = TransformerConcepts()
    concepts.attention_mechanism_demo()
    concepts.positional_encoding_demo()
    
    # 3. Embedding Exploration
    print("\n3. 🎨 Embedding Exploration")
    words = ["happy", "joy", "sad", "angry", "excited"]
    llm.embedding_exploration(words)
    
    print("\n✅ Demo completed! Check the visualizations above.")


if __name__ == "__main__":
    run_fundamentals_demo()