"""
PDF Question-Answering System

A comprehensive system for processing PDFs and answering questions about their content.
"""

from .processing import PDFProcessor, DocumentChunker
from .embeddings import EmbeddingManager
from .retrieval import DocumentRetriever
from .generation import AnswerGenerator

__all__ = [
    "PDFProcessor",
    "DocumentChunker",
    "EmbeddingManager", 
    "DocumentRetriever",
    "AnswerGenerator",
    "PDFQASystem",
]


class PDFQASystem:
    """
    Complete PDF Question-Answering System.
    
    Combines document processing, embedding generation, retrieval, and answer generation
    into a unified system for PDF-based Q&A.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the PDF Q&A system."""
        self.processor = PDFProcessor()
        self.chunker = DocumentChunker()
        self.embedding_manager = EmbeddingManager(model_name=embedding_model)
        self.retriever = DocumentRetriever(self.embedding_manager)
        self.generator = AnswerGenerator()
        
        self.documents = {}  # Store processed documents
        self.is_ready = False
    
    def add_pdf(self, pdf_path: str, document_id: str = None) -> str:
        """
        Add a PDF document to the system.
        
        Args:
            pdf_path: Path to the PDF file
            document_id: Optional custom ID for the document
            
        Returns:
            Document ID
        """
        if document_id is None:
            import os
            document_id = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Process PDF
        text = self.processor.extract_text(pdf_path)
        
        # Chunk the text
        chunks = self.chunker.chunk_text(text)
        
        # Create embeddings
        embeddings = self.embedding_manager.create_embeddings(chunks)
        
        # Store document
        self.documents[document_id] = {
            "path": pdf_path,
            "text": text,
            "chunks": chunks,
            "embeddings": embeddings
        }
        
        # Update retriever
        self.retriever.add_document(document_id, chunks, embeddings)
        
        self.is_ready = True
        return document_id
    
    def ask_question(self, question: str, num_chunks: int = 3) -> dict:
        """
        Ask a question about the loaded documents.
        
        Args:
            question: Question to ask
            num_chunks: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        if not self.is_ready:
            return {
                "answer": "No documents loaded. Please add a PDF first.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Retrieve relevant chunks
        relevant_chunks = self.retriever.retrieve(question, k=num_chunks)
        
        # Generate answer
        answer_result = self.generator.generate_answer(question, relevant_chunks)
        
        return answer_result
    
    def get_document_summary(self, document_id: str) -> dict:
        """Get summary information about a document."""
        if document_id not in self.documents:
            return {"error": "Document not found"}
        
        doc = self.documents[document_id]
        return {
            "document_id": document_id,
            "path": doc["path"],
            "text_length": len(doc["text"]),
            "num_chunks": len(doc["chunks"]),
            "avg_chunk_length": sum(len(chunk) for chunk in doc["chunks"]) / len(doc["chunks"])
        }
    
    def list_documents(self) -> list:
        """List all loaded documents."""
        return [
            self.get_document_summary(doc_id) 
            for doc_id in self.documents.keys()
        ]