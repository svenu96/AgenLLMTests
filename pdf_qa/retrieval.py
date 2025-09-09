"""
Document Retrieval Module

Advanced retrieval systems for finding relevant document chunks based on queries.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import heapq
from collections import defaultdict

from .embeddings import EmbeddingManager


@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    text: str
    score: float
    document_id: str
    chunk_index: int
    metadata: Dict[str, Any] = None


class DocumentRetriever:
    """
    Advanced document retrieval system with multiple search strategies.
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """
        Initialize document retriever.
        
        Args:
            embedding_manager: Embedding manager for semantic search
        """
        self.embedding_manager = embedding_manager
        self.documents = {}  # Store document chunks and embeddings
        self.document_metadata = {}  # Store document metadata
        
    def add_document(self, document_id: str, chunks: List[str], 
                    embeddings: np.ndarray, metadata: Dict[str, Any] = None):
        """
        Add a document to the retrieval system.
        
        Args:
            document_id: Unique identifier for the document
            chunks: List of text chunks
            embeddings: Embeddings for the chunks
            metadata: Optional metadata for the document
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        self.documents[document_id] = {
            "chunks": chunks,
            "embeddings": embeddings,
            "chunk_count": len(chunks)
        }
        
        self.document_metadata[document_id] = metadata or {}
        
        print(f"✅ Added document '{document_id}' with {len(chunks)} chunks")
    
    def retrieve(self, query: str, k: int = 5, method: str = "semantic", 
                filters: Dict[str, Any] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            k: Number of results to return
            method: Retrieval method (semantic, keyword, hybrid)
            filters: Optional filters for documents
            
        Returns:
            List of retrieval results
        """
        if method == "semantic":
            return self._semantic_search(query, k, filters)
        elif method == "keyword":
            return self._keyword_search(query, k, filters)
        elif method == "hybrid":
            return self._hybrid_search(query, k, filters)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
    
    def _semantic_search(self, query: str, k: int, filters: Dict[str, Any] = None) -> List[RetrievalResult]:
        """Perform semantic search using embeddings."""
        results = []
        
        # Get filtered documents
        filtered_docs = self._apply_filters(filters) if filters else self.documents.keys()
        
        for doc_id in filtered_docs:
            doc_data = self.documents[doc_id]
            chunks = doc_data["chunks"]
            embeddings = doc_data["embeddings"]
            
            # Perform similarity search for this document
            similarity_results = self.embedding_manager.similarity_search(
                query, embeddings, chunks, top_k=len(chunks)
            )
            
            # Convert to RetrievalResult objects
            for chunk_text, score in similarity_results:
                chunk_index = chunks.index(chunk_text)
                
                result = RetrievalResult(
                    text=chunk_text,
                    score=score,
                    document_id=doc_id,
                    chunk_index=chunk_index,
                    metadata=self.document_metadata.get(doc_id, {})
                )
                
                results.append(result)
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]
    
    def _keyword_search(self, query: str, k: int, filters: Dict[str, Any] = None) -> List[RetrievalResult]:
        """Perform keyword-based search."""
        import re
        from collections import Counter
        
        # Tokenize query
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        query_counter = Counter(query_tokens)
        
        results = []
        filtered_docs = self._apply_filters(filters) if filters else self.documents.keys()
        
        for doc_id in filtered_docs:
            doc_data = self.documents[doc_id]
            chunks = doc_data["chunks"]
            
            for chunk_index, chunk in enumerate(chunks):
                # Tokenize chunk
                chunk_tokens = re.findall(r'\b\w+\b', chunk.lower())
                chunk_counter = Counter(chunk_tokens)
                
                # Calculate TF-IDF-like score
                score = self._calculate_keyword_score(query_counter, chunk_counter, chunk_tokens)
                
                if score > 0:  # Only include chunks with some relevance
                    result = RetrievalResult(
                        text=chunk,
                        score=score,
                        document_id=doc_id,
                        chunk_index=chunk_index,
                        metadata=self.document_metadata.get(doc_id, {})
                    )
                    
                    results.append(result)
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]
    
    def _hybrid_search(self, query: str, k: int, filters: Dict[str, Any] = None, 
                      semantic_weight: float = 0.7) -> List[RetrievalResult]:
        """Combine semantic and keyword search with weighted scores."""
        # Get results from both methods
        semantic_results = self._semantic_search(query, k * 2, filters)
        keyword_results = self._keyword_search(query, k * 2, filters)
        
        # Normalize scores
        if semantic_results:
            max_semantic_score = max(r.score for r in semantic_results)
            for result in semantic_results:
                result.score = result.score / max_semantic_score
        
        if keyword_results:
            max_keyword_score = max(r.score for r in keyword_results)
            for result in keyword_results:
                result.score = result.score / max_keyword_score
        
        # Combine results
        combined_scores = defaultdict(list)
        
        # Add semantic scores
        for result in semantic_results:
            key = (result.document_id, result.chunk_index)
            combined_scores[key].append(('semantic', result.score, result))
        
        # Add keyword scores
        for result in keyword_results:
            key = (result.document_id, result.chunk_index)
            combined_scores[key].append(('keyword', result.score, result))
        
        # Calculate combined scores
        final_results = []
        for key, scores in combined_scores.items():
            semantic_score = 0
            keyword_score = 0
            result_obj = None
            
            for method, score, result in scores:
                if method == 'semantic':
                    semantic_score = score
                    result_obj = result
                elif method == 'keyword':
                    keyword_score = score
                    if result_obj is None:
                        result_obj = result
            
            # Calculate weighted combined score
            combined_score = (semantic_weight * semantic_score + 
                            (1 - semantic_weight) * keyword_score)
            
            if result_obj:
                result_obj.score = combined_score
                final_results.append(result_obj)
        
        # Sort and return top-k
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:k]
    
    def _calculate_keyword_score(self, query_counter: Counter, chunk_counter: Counter, 
                               chunk_tokens: List[str]) -> float:
        """Calculate keyword-based relevance score."""
        score = 0
        
        for token, query_freq in query_counter.items():
            if token in chunk_counter:
                # Simple TF score
                tf = chunk_counter[token] / len(chunk_tokens)
                
                # Boost score based on query term frequency
                score += tf * query_freq
        
        return score
    
    def _apply_filters(self, filters: Dict[str, Any]) -> List[str]:
        """Apply filters to documents and return matching document IDs."""
        matching_docs = []
        
        for doc_id, metadata in self.document_metadata.items():
            matches = True
            
            for filter_key, filter_value in filters.items():
                if filter_key not in metadata:
                    matches = False
                    break
                
                if isinstance(filter_value, str):
                    if filter_value.lower() not in str(metadata[filter_key]).lower():
                        matches = False
                        break
                elif metadata[filter_key] != filter_value:
                    matches = False
                    break
            
            if matches:
                matching_docs.append(doc_id)
        
        return matching_docs
    
    def get_document_chunks(self, document_id: str) -> List[str]:
        """Get all chunks for a specific document."""
        if document_id not in self.documents:
            return []
        
        return self.documents[document_id]["chunks"]
    
    def get_chunk_context(self, document_id: str, chunk_index: int, 
                         context_size: int = 1) -> str:
        """
        Get a chunk with surrounding context.
        
        Args:
            document_id: Document identifier
            chunk_index: Index of the target chunk
            context_size: Number of chunks before and after to include
            
        Returns:
            Combined text with context
        """
        if document_id not in self.documents:
            return ""
        
        chunks = self.documents[document_id]["chunks"]
        
        if chunk_index < 0 or chunk_index >= len(chunks):
            return ""
        
        # Calculate context range
        start_idx = max(0, chunk_index - context_size)
        end_idx = min(len(chunks), chunk_index + context_size + 1)
        
        # Combine chunks with context
        context_chunks = chunks[start_idx:end_idx]
        return "\n\n".join(context_chunks)
    
    def search_within_document(self, document_id: str, query: str, 
                             k: int = 3, method: str = "semantic") -> List[RetrievalResult]:
        """
        Search for relevant chunks within a specific document.
        
        Args:
            document_id: Target document ID
            query: Search query
            k: Number of results to return
            method: Search method
            
        Returns:
            List of retrieval results from the specified document
        """
        if document_id not in self.documents:
            return []
        
        # Create temporary filter for single document
        filters = {"document_id": document_id}
        
        # Temporarily modify documents to only include target document
        original_docs = self.documents.copy()
        self.documents = {document_id: self.documents[document_id]}
        
        try:
            results = self.retrieve(query, k, method)
            return results
        finally:
            # Restore original documents
            self.documents = original_docs
    
    def get_similar_chunks(self, reference_chunk: str, k: int = 5, 
                          exclude_document: str = None) -> List[RetrievalResult]:
        """
        Find chunks similar to a reference chunk.
        
        Args:
            reference_chunk: Text to find similar chunks for
            k: Number of similar chunks to return
            exclude_document: Optional document ID to exclude from search
            
        Returns:
            List of similar chunks
        """
        # Create query embedding
        query_embedding = self.embedding_manager.create_single_embedding(reference_chunk)
        
        if len(query_embedding) == 0:
            return []
        
        all_results = []
        
        for doc_id, doc_data in self.documents.items():
            if exclude_document and doc_id == exclude_document:
                continue
            
            chunks = doc_data["chunks"]
            embeddings = doc_data["embeddings"]
            
            # Calculate similarities
            similarities = self.embedding_manager._cosine_similarity(query_embedding, embeddings)
            
            # Create results
            for i, (chunk, score) in enumerate(zip(chunks, similarities)):
                result = RetrievalResult(
                    text=chunk,
                    score=float(score),
                    document_id=doc_id,
                    chunk_index=i,
                    metadata=self.document_metadata.get(doc_id, {})
                )
                all_results.append(result)
        
        # Sort and return top-k
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:k]
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system."""
        total_chunks = sum(doc["chunk_count"] for doc in self.documents.values())
        
        chunk_sizes = []
        for doc_data in self.documents.values():
            chunk_sizes.extend([len(chunk) for chunk in doc_data["chunks"]])
        
        return {
            "num_documents": len(self.documents),
            "total_chunks": total_chunks,
            "avg_chunks_per_doc": total_chunks / len(self.documents) if self.documents else 0,
            "avg_chunk_length": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            "min_chunk_length": min(chunk_sizes) if chunk_sizes else 0,
            "max_chunk_length": max(chunk_sizes) if chunk_sizes else 0,
            "embedding_dimension": self.embedding_manager.embedding_dim
        }


def demo_retrieval():
    """Demonstrate retrieval functionality."""
    print("🔍 Document Retrieval Demo")
    print("=" * 50)
    
    # Initialize embedding manager and retriever
    from .embeddings import EmbeddingManager
    
    embedding_manager = EmbeddingManager()
    if not embedding_manager.model:
        print("❌ Could not initialize embedding model")
        return
    
    retriever = DocumentRetriever(embedding_manager)
    
    # Demo documents
    doc1_chunks = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming.",
        "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
        "Supervised learning uses labeled data to train models that can make predictions on new data.",
        "Unsupervised learning finds hidden patterns in data without using labeled examples."
    ]
    
    doc2_chunks = [
        "Natural language processing helps computers understand and generate human language.",
        "Sentiment analysis determines the emotional tone of text using NLP techniques.",
        "Machine translation automatically converts text from one language to another.",
        "Chatbots use NLP to understand user queries and provide relevant responses."
    ]
    
    # Create embeddings
    doc1_embeddings = embedding_manager.create_embeddings(doc1_chunks, show_progress=False)
    doc2_embeddings = embedding_manager.create_embeddings(doc2_chunks, show_progress=False)
    
    # Add documents to retriever
    retriever.add_document("ml_basics", doc1_chunks, doc1_embeddings, 
                          {"topic": "machine_learning", "difficulty": "beginner"})
    retriever.add_document("nlp_guide", doc2_chunks, doc2_embeddings,
                          {"topic": "nlp", "difficulty": "intermediate"})
    
    # Demo semantic search
    print("\n🎯 Semantic Search Demo:")
    query = "neural networks and deep learning"
    results = retriever.retrieve(query, k=3, method="semantic")
    
    print(f"Query: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"{i}. ({result.score:.3f}) [{result.document_id}] {result.text[:80]}...")
    
    # Demo keyword search
    print(f"\n🔤 Keyword Search Demo:")
    results = retriever.retrieve(query, k=3, method="keyword")
    
    print(f"Query: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"{i}. ({result.score:.3f}) [{result.document_id}] {result.text[:80]}...")
    
    # Demo hybrid search
    print(f"\n🔀 Hybrid Search Demo:")
    results = retriever.retrieve(query, k=3, method="hybrid")
    
    print(f"Query: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"{i}. ({result.score:.3f}) [{result.document_id}] {result.text[:80]}...")
    
    # Demo filtered search
    print(f"\n🔍 Filtered Search Demo:")
    results = retriever.retrieve("learning", k=2, method="semantic", 
                               filters={"topic": "machine_learning"})
    
    print("Query: 'learning' (filtered by topic: machine_learning)")
    for i, result in enumerate(results, 1):
        print(f"{i}. ({result.score:.3f}) [{result.document_id}] {result.text[:80]}...")
    
    # Demo stats
    print(f"\n📊 Retrieval Stats:")
    stats = retriever.get_retrieval_stats()
    for key, value in stats.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    demo_retrieval()