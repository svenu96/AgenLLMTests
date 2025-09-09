"""
Embedding Management Module

Tools for creating and managing document embeddings for semantic search.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pickle
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch


class EmbeddingManager:
    """
    Manages text embeddings for semantic search and similarity matching.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "auto"):
        """
        Initialize embedding manager.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run the model on (auto, cpu, cuda)
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.embedding_dim = None
        
        self._load_model()
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Get embedding dimension
            test_embedding = self.model.encode("test")
            self.embedding_dim = len(test_embedding)
            
            print(f"✅ Model loaded successfully!")
            print(f"   - Device: {self.device}")
            print(f"   - Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            self.model = None
    
    def create_embeddings(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            return embeddings
            
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return np.array([])
    
    def create_single_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of the embedding
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
            
        except Exception as e:
            print(f"Error creating single embedding: {e}")
            return np.array([])
    
    def similarity_search(self, query: str, embeddings: np.ndarray, texts: List[str], 
                         top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform similarity search using cosine similarity.
        
        Args:
            query: Query text
            embeddings: Document embeddings
            texts: Original texts corresponding to embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (text, similarity_score) tuples
        """
        if len(embeddings) == 0 or len(texts) != len(embeddings):
            return []
        
        # Create query embedding
        query_embedding = self.create_single_embedding(query)
        if len(query_embedding) == 0:
            return []
        
        # Calculate cosine similarities
        similarities = self._cosine_similarity(query_embedding, embeddings)
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(texts):
                results.append((texts[idx], float(similarities[idx])))
        
        return results
    
    def _cosine_similarity(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query and document embeddings.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: Matrix of document embeddings
            
        Returns:
            Array of similarity scores
        """
        # Normalize vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        
        # Calculate dot product (cosine similarity for normalized vectors)
        similarities = np.dot(doc_norms, query_norm)
        
        return similarities
    
    def save_embeddings(self, embeddings: np.ndarray, texts: List[str], metadata: Dict[str, Any], 
                       file_path: str):
        """
        Save embeddings and associated data to file.
        
        Args:
            embeddings: Embedding matrix
            texts: Associated texts
            metadata: Additional metadata
            file_path: Path to save the data
        """
        data = {
            "embeddings": embeddings,
            "texts": texts,
            "metadata": metadata,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim
        }
        
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"✅ Embeddings saved to: {file_path}")
            
        except Exception as e:
            print(f"❌ Error saving embeddings: {e}")
    
    def load_embeddings(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load embeddings and associated data from file.
        
        Args:
            file_path: Path to the saved data
            
        Returns:
            Dictionary containing embeddings, texts, and metadata
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            print(f"✅ Embeddings loaded from: {file_path}")
            print(f"   - Model: {data.get('model_name', 'Unknown')}")
            print(f"   - Texts: {len(data.get('texts', []))}")
            print(f"   - Embedding dim: {data.get('embedding_dim', 'Unknown')}")
            
            return data
            
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            return None
    
    def cluster_embeddings(self, embeddings: np.ndarray, n_clusters: int = 5, method: str = "kmeans") -> np.ndarray:
        """
        Cluster embeddings using specified method.
        
        Args:
            embeddings: Embedding matrix
            n_clusters: Number of clusters
            method: Clustering method (kmeans, hierarchical)
            
        Returns:
            Cluster labels
        """
        try:
            if method == "kmeans":
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(embeddings)
                
            elif method == "hierarchical":
                from sklearn.cluster import AgglomerativeClustering
                clustering = AgglomerativeClustering(n_clusters=n_clusters)
                labels = clustering.fit_predict(embeddings)
                
            else:
                raise ValueError(f"Unknown clustering method: {method}")
            
            return labels
            
        except ImportError:
            print("❌ Scikit-learn not available for clustering")
            return np.array([])
        except Exception as e:
            print(f"❌ Error clustering embeddings: {e}")
            return np.array([])
    
    def visualize_embeddings(self, embeddings: np.ndarray, texts: List[str], 
                           labels: Optional[np.ndarray] = None, method: str = "tsne"):
        """
        Visualize embeddings in 2D space.
        
        Args:
            embeddings: Embedding matrix
            texts: Associated texts for labeling
            labels: Optional cluster labels for coloring
            method: Dimensionality reduction method (tsne, pca, umap)
        """
        try:
            import matplotlib.pyplot as plt
            
            # Reduce dimensionality
            if method == "tsne":
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            elif method == "pca":
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2, random_state=42)
            elif method == "umap":
                try:
                    import umap
                    reducer = umap.UMAP(n_components=2, random_state=42)
                except ImportError:
                    print("UMAP not available, falling back to t-SNE")
                    from sklearn.manifold import TSNE
                    reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            else:
                raise ValueError(f"Unknown dimensionality reduction method: {method}")
            
            embeddings_2d = reducer.fit_transform(embeddings)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            if labels is not None:
                scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                    c=labels, cmap='tab10', alpha=0.7)
                plt.colorbar(scatter, label='Cluster')
            else:
                plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
            
            # Add text labels for a subset of points
            n_labels = min(20, len(texts))
            indices = np.linspace(0, len(texts)-1, n_labels).astype(int)
            
            for idx in indices:
                plt.annotate(texts[idx][:30] + "..." if len(texts[idx]) > 30 else texts[idx],
                           (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
            
            plt.title(f'Embedding Visualization ({method.upper()})')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.tight_layout()
            plt.show()
            
        except ImportError as e:
            print(f"❌ Visualization libraries not available: {e}")
        except Exception as e:
            print(f"❌ Error visualizing embeddings: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self.embedding_dim,
            "model_loaded": self.model is not None
        }


class EmbeddingCache:
    """
    Cache system for storing and retrieving embeddings to avoid recomputation.
    """
    
    def __init__(self, cache_dir: str = "embedding_cache"):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load cache index
        self.index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """Load cache index from file."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {}
    
    def _save_index(self):
        """Save cache index to file."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache index: {e}")
    
    def _get_cache_key(self, texts: List[str], model_name: str) -> str:
        """Generate cache key for texts and model."""
        import hashlib
        
        # Create hash of texts and model
        content = f"{model_name}:{':'.join(texts)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_embeddings(self, texts: List[str], model_name: str) -> Optional[np.ndarray]:
        """
        Get embeddings from cache if available.
        
        Args:
            texts: List of texts
            model_name: Name of the embedding model
            
        Returns:
            Cached embeddings or None if not found
        """
        cache_key = self._get_cache_key(texts, model_name)
        
        if cache_key in self.cache_index:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    return data["embeddings"]
                    
                except Exception as e:
                    print(f"Warning: Could not load cached embeddings: {e}")
                    # Remove invalid cache entry
                    del self.cache_index[cache_key]
                    self._save_index()
        
        return None
    
    def save_embeddings(self, texts: List[str], model_name: str, embeddings: np.ndarray):
        """
        Save embeddings to cache.
        
        Args:
            texts: List of texts
            model_name: Name of the embedding model
            embeddings: Computed embeddings
        """
        cache_key = self._get_cache_key(texts, model_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            data = {
                "texts": texts,
                "model_name": model_name,
                "embeddings": embeddings,
                "timestamp": str(np.datetime64('now'))
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Update index
            self.cache_index[cache_key] = {
                "file": cache_file.name,
                "model_name": model_name,
                "num_texts": len(texts),
                "timestamp": data["timestamp"]
            }
            
            self._save_index()
            
        except Exception as e:
            print(f"Warning: Could not cache embeddings: {e}")
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            
            self.cache_index.clear()
            self._save_index()
            
            print("✅ Cache cleared successfully")
            
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")


def demo_embeddings():
    """Demonstrate embedding functionality."""
    print("🎯 Embeddings Demo")
    print("=" * 50)
    
    # Initialize embedding manager
    manager = EmbeddingManager()
    
    if not manager.model:
        print("❌ Could not initialize embedding model")
        return
    
    # Demo texts
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information.",
        "Reinforcement learning learns through trial and error.",
        "The weather is sunny today.",
        "I love eating pizza for dinner.",
        "Cats are wonderful pets to have."
    ]
    
    print(f"\n📝 Creating embeddings for {len(texts)} texts...")
    embeddings = manager.create_embeddings(texts)
    
    print(f"✅ Embeddings created: {embeddings.shape}")
    
    # Similarity search demo
    print(f"\n🔍 Similarity Search Demo:")
    query = "artificial intelligence and machine learning"
    results = manager.similarity_search(query, embeddings, texts, top_k=3)
    
    print(f"Query: '{query}'")
    print("Top results:")
    for i, (text, score) in enumerate(results, 1):
        print(f"{i}. ({score:.3f}) {text}")
    
    # Clustering demo
    print(f"\n🎯 Clustering Demo:")
    labels = manager.cluster_embeddings(embeddings, n_clusters=3)
    
    if len(labels) > 0:
        for i, (text, label) in enumerate(zip(texts, labels)):
            print(f"Cluster {label}: {text[:50]}...")
    
    print(f"\n📊 Model Info:")
    info = manager.get_model_info()
    for key, value in info.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    demo_embeddings()