"""
PDF Processing Module

Tools for extracting and processing text from PDF documents.
"""

import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
import re
from pathlib import Path


class PDFProcessor:
    """
    PDF text extraction with multiple backends for reliability.
    """
    
    def __init__(self, preferred_method: str = "pdfplumber"):
        """
        Initialize PDF processor.
        
        Args:
            preferred_method: Preferred extraction method (pypdf2, pdfplumber, pymupdf)
        """
        self.preferred_method = preferred_method
        self.available_methods = self._check_available_methods()
    
    def _check_available_methods(self) -> List[str]:
        """Check which PDF processing libraries are available."""
        methods = []
        
        try:
            import PyPDF2
            methods.append("pypdf2")
        except ImportError:
            pass
        
        try:
            import pdfplumber
            methods.append("pdfplumber")
        except ImportError:
            pass
        
        try:
            import fitz
            methods.append("pymupdf")
        except ImportError:
            pass
        
        return methods
    
    def extract_text(self, pdf_path: str, method: Optional[str] = None) -> str:
        """
        Extract text from PDF using specified or preferred method.
        
        Args:
            pdf_path: Path to PDF file
            method: Extraction method to use
            
        Returns:
            Extracted text
        """
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        extraction_method = method or self.preferred_method
        
        if extraction_method not in self.available_methods:
            # Fallback to first available method
            if self.available_methods:
                extraction_method = self.available_methods[0]
            else:
                raise RuntimeError("No PDF processing libraries available")
        
        try:
            if extraction_method == "pdfplumber":
                return self._extract_with_pdfplumber(pdf_path)
            elif extraction_method == "pymupdf":
                return self._extract_with_pymupdf(pdf_path)
            elif extraction_method == "pypdf2":
                return self._extract_with_pypdf2(pdf_path)
            else:
                raise ValueError(f"Unknown extraction method: {extraction_method}")
        
        except Exception as e:
            # Try fallback methods
            for fallback_method in self.available_methods:
                if fallback_method != extraction_method:
                    try:
                        return self.extract_text(pdf_path, fallback_method)
                    except:
                        continue
            
            raise RuntimeError(f"All PDF extraction methods failed: {str(e)}")
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber."""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF."""
        text = ""
        doc = fitz.open(pdf_path)
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text() + "\n"
        
        doc.close()
        return text
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2."""
        text = ""
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        return text
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract metadata from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing metadata
        """
        metadata = {}
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if pdf_reader.metadata:
                    metadata.update({
                        'title': pdf_reader.metadata.get('/Title', ''),
                        'author': pdf_reader.metadata.get('/Author', ''),
                        'subject': pdf_reader.metadata.get('/Subject', ''),
                        'creator': pdf_reader.metadata.get('/Creator', ''),
                        'producer': pdf_reader.metadata.get('/Producer', ''),
                        'creation_date': pdf_reader.metadata.get('/CreationDate', ''),
                        'modification_date': pdf_reader.metadata.get('/ModDate', '')
                    })
                
                metadata['num_pages'] = len(pdf_reader.pages)
        
        except Exception as e:
            metadata['error'] = str(e)
        
        # Add file metadata
        file_path = Path(pdf_path)
        metadata.update({
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'file_path': str(file_path.absolute())
        })
        
        return metadata


class DocumentChunker:
    """
    Text chunking for efficient processing and retrieval.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document chunker.
        
        Args:
            chunk_size: Target size for each chunk (in characters)
            chunk_overlap: Overlap between chunks (in characters)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, method: str = "recursive") -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            method: Chunking method (simple, sentence, recursive)
            
        Returns:
            List of text chunks
        """
        if method == "simple":
            return self._simple_chunk(text)
        elif method == "sentence":
            return self._sentence_chunk(text)
        elif method == "recursive":
            return self._recursive_chunk(text)
        else:
            raise ValueError(f"Unknown chunking method: {method}")
    
    def _simple_chunk(self, text: str) -> List[str]:
        """Simple character-based chunking."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def _sentence_chunk(self, text: str) -> List[str]:
        """Sentence-aware chunking."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _recursive_chunk(self, text: str) -> List[str]:
        """Recursive chunking that tries to split at natural boundaries."""
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        
        # Try to split at paragraph boundaries first
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            return self._chunk_by_separator(text, '\n\n')
        
        # Try to split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 1:
            return self._chunk_by_separator(text, r'(?<=[.!?])\s+')
        
        # Fall back to simple chunking
        return self._simple_chunk(text)
    
    def _chunk_by_separator(self, text: str, separator: str) -> List[str]:
        """Chunk text by a specific separator."""
        if separator == '\n\n':
            parts = text.split(separator)
        else:
            parts = re.split(separator, text)
        
        chunks = []
        current_chunk = ""
        
        for part in parts:
            if len(current_chunk) + len(part) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Handle overlap
                if self.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + " " + part
                else:
                    current_chunk = part
            else:
                if current_chunk:
                    current_chunk += (" " if separator != '\n\n' else '\n\n') + part
                else:
                    current_chunk = part
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def get_chunk_info(self, chunks: List[str]) -> Dict[str, Any]:
        """Get information about the chunks."""
        if not chunks:
            return {"error": "No chunks provided"}
        
        chunk_lengths = [len(chunk) for chunk in chunks]
        
        return {
            "num_chunks": len(chunks),
            "total_characters": sum(chunk_lengths),
            "avg_chunk_length": sum(chunk_lengths) / len(chunks),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "target_chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }


def demo_pdf_processing():
    """Demonstrate PDF processing capabilities."""
    print("📄 PDF Processing Demo")
    print("=" * 50)
    
    # Initialize processor
    processor = PDFProcessor()
    print(f"Available methods: {processor.available_methods}")
    
    # Initialize chunker
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
    
    # Demo text for chunking (since we might not have a PDF file)
    demo_text = """
    Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.
    
    Machine Learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.
    
    Deep Learning is a subset of machine learning that uses neural networks with multiple layers (hence "deep") to model and understand complex patterns in data. It's particularly effective for tasks like image recognition, natural language processing, and speech recognition.
    
    Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding.
    """
    
    print("\n📝 Text Chunking Demo:")
    chunks = chunker.chunk_text(demo_text, method="sentence")
    
    print(f"Original text length: {len(demo_text)} characters")
    print(f"Number of chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
    
    # Chunk information
    info = chunker.get_chunk_info(chunks)
    print(f"\n📊 Chunk Information:")
    for key, value in info.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    demo_pdf_processing()