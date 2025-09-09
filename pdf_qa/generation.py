"""
Answer Generation Module

Advanced answer generation for PDF Q&A using retrieval-augmented generation (RAG).
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
from langchain.llms.base import LLM
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from .retrieval import RetrievalResult


@dataclass
class AnswerResult:
    """Container for generated answers with metadata."""
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    method: str
    tokens_used: int = 0
    processing_time: float = 0.0


class AnswerGenerator:
    """
    Advanced answer generation system using retrieval-augmented generation.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.1):
        """
        Initialize answer generator.
        
        Args:
            model_name: Language model to use for generation
            temperature: Temperature for generation (0.0 = deterministic, 1.0 = creative)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = None
        
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the language model."""
        try:
            if "gpt" in self.model_name.lower():
                self.llm = ChatOpenAI(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=1000
                )
                print(f"✅ Initialized {self.model_name} for answer generation")
            else:
                print(f"⚠️ Model {self.model_name} not directly supported, using GPT-3.5 as fallback")
                self.llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=self.temperature,
                    max_tokens=1000
                )
        except Exception as e:
            print(f"❌ Failed to initialize language model: {e}")
            print("💡 Make sure you have set your OpenAI API key")
            self.llm = None
    
    def generate_answer(self, question: str, retrieved_chunks: List[RetrievalResult], 
                       method: str = "comprehensive") -> AnswerResult:
        """
        Generate an answer based on the question and retrieved chunks.
        
        Args:
            question: User's question
            retrieved_chunks: Relevant chunks from retrieval
            method: Generation method (comprehensive, concise, analytical)
            
        Returns:
            AnswerResult with generated answer and metadata
        """
        if not self.llm:
            return AnswerResult(
                answer="❌ Language model not available. Please configure your OpenAI API key.",
                confidence=0.0,
                sources=[],
                method=method
            )
        
        if not retrieved_chunks:
            return AnswerResult(
                answer="❌ No relevant information found to answer your question.",
                confidence=0.0,
                sources=[],
                method=method
            )
        
        # Select generation strategy
        if method == "comprehensive":
            return self._generate_comprehensive_answer(question, retrieved_chunks)
        elif method == "concise":
            return self._generate_concise_answer(question, retrieved_chunks)
        elif method == "analytical":
            return self._generate_analytical_answer(question, retrieved_chunks)
        else:
            raise ValueError(f"Unknown generation method: {method}")
    
    def _generate_comprehensive_answer(self, question: str, chunks: List[RetrievalResult]) -> AnswerResult:
        """Generate a comprehensive, detailed answer."""
        import time
        start_time = time.time()
        
        # Prepare context from chunks
        context = self._prepare_context(chunks, max_chunks=5)
        
        # Create prompt
        system_prompt = """You are an expert assistant that provides comprehensive and accurate answers based on the provided context. 

Instructions:
1. Answer the question thoroughly using the provided context
2. Include specific details and examples when available
3. If the context doesn't fully answer the question, acknowledge what's missing
4. Structure your answer clearly with appropriate formatting
5. Be accurate and don't make up information not in the context
6. Cite relevant sources when possible"""
        
        user_prompt = f"""Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm(messages)
            answer = response.content
            
            # Calculate confidence based on relevance scores
            confidence = self._calculate_confidence(chunks)
            
            # Prepare sources
            sources = self._prepare_sources(chunks)
            
            processing_time = time.time() - start_time
            
            return AnswerResult(
                answer=answer,
                confidence=confidence,
                sources=sources,
                method="comprehensive",
                processing_time=processing_time
            )
            
        except Exception as e:
            return AnswerResult(
                answer=f"❌ Error generating answer: {str(e)}",
                confidence=0.0,
                sources=[],
                method="comprehensive"
            )
    
    def _generate_concise_answer(self, question: str, chunks: List[RetrievalResult]) -> AnswerResult:
        """Generate a concise, to-the-point answer."""
        import time
        start_time = time.time()
        
        # Use fewer chunks for concise answers
        context = self._prepare_context(chunks, max_chunks=3)
        
        system_prompt = """You are an expert assistant that provides concise and direct answers based on the provided context.

Instructions:
1. Answer the question directly and concisely
2. Focus on the most important information
3. Keep your response brief but accurate
4. Don't include unnecessary details
5. If the context doesn't answer the question, say so briefly"""
        
        user_prompt = f"""Context:
{context}

Question: {question}

Please provide a concise, direct answer."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm(messages)
            answer = response.content
            
            confidence = self._calculate_confidence(chunks)
            sources = self._prepare_sources(chunks[:3])  # Fewer sources for concise
            processing_time = time.time() - start_time
            
            return AnswerResult(
                answer=answer,
                confidence=confidence,
                sources=sources,
                method="concise",
                processing_time=processing_time
            )
            
        except Exception as e:
            return AnswerResult(
                answer=f"❌ Error generating answer: {str(e)}",
                confidence=0.0,
                sources=[],
                method="concise"
            )
    
    def _generate_analytical_answer(self, question: str, chunks: List[RetrievalResult]) -> AnswerResult:
        """Generate an analytical answer with reasoning and evidence."""
        import time
        start_time = time.time()
        
        context = self._prepare_context(chunks, max_chunks=6)
        
        system_prompt = """You are an expert analyst that provides detailed, analytical answers based on the provided context.

Instructions:
1. Analyze the question and break it down into components
2. Examine the evidence from the context systematically
3. Present your reasoning clearly with supporting evidence
4. Discuss any limitations or gaps in the available information
5. Structure your analysis logically with clear sections
6. Draw well-supported conclusions"""
        
        user_prompt = f"""Context:
{context}

Question: {question}

Please provide an analytical answer that includes:
1. Analysis of the question
2. Examination of relevant evidence
3. Reasoning process
4. Conclusions with supporting evidence"""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm(messages)
            answer = response.content
            
            confidence = self._calculate_confidence(chunks)
            sources = self._prepare_sources(chunks)
            processing_time = time.time() - start_time
            
            return AnswerResult(
                answer=answer,
                confidence=confidence,
                sources=sources,
                method="analytical",
                processing_time=processing_time
            )
            
        except Exception as e:
            return AnswerResult(
                answer=f"❌ Error generating answer: {str(e)}",
                confidence=0.0,
                sources=[],
                method="analytical"
            )
    
    def _prepare_context(self, chunks: List[RetrievalResult], max_chunks: int = 5) -> str:
        """Prepare context string from retrieved chunks."""
        context_parts = []
        
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            # Add source information
            source_info = f"[Source {i}: {chunk.document_id}]"
            context_parts.append(f"{source_info}\n{chunk.text}")
        
        return "\n\n".join(context_parts)
    
    def _prepare_sources(self, chunks: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Prepare source information for the answer."""
        sources = []
        
        for i, chunk in enumerate(chunks, 1):
            source = {
                "id": i,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "relevance_score": round(chunk.score, 3),
                "text_preview": chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text
            }
            
            # Add metadata if available
            if chunk.metadata:
                source["metadata"] = chunk.metadata
            
            sources.append(source)
        
        return sources
    
    def _calculate_confidence(self, chunks: List[RetrievalResult]) -> float:
        """Calculate confidence score based on retrieval results."""
        if not chunks:
            return 0.0
        
        # Base confidence on average relevance score
        avg_score = sum(chunk.score for chunk in chunks) / len(chunks)
        
        # Boost confidence if we have multiple relevant chunks
        chunk_bonus = min(0.2, len(chunks) * 0.05)
        
        # Penalty if top score is low
        top_score = chunks[0].score if chunks else 0
        score_penalty = 0.0 if top_score > 0.7 else (0.7 - top_score) * 0.3
        
        confidence = min(1.0, avg_score + chunk_bonus - score_penalty)
        return round(confidence, 3)
    
    def generate_followup_questions(self, question: str, answer: str, 
                                  chunks: List[RetrievalResult]) -> List[str]:
        """Generate relevant follow-up questions based on the Q&A context."""
        if not self.llm:
            return []
        
        system_prompt = """You are an expert at generating relevant follow-up questions based on a Q&A conversation.

Instructions:
1. Generate 3-5 relevant follow-up questions
2. Questions should be related to the original question and answer
3. Questions should help the user explore the topic further
4. Make questions specific and actionable
5. Ensure questions can potentially be answered using similar context"""
        
        user_prompt = f"""Original Question: {question}

Answer Provided: {answer}

Based on this Q&A, generate 3-5 relevant follow-up questions that would help the user explore this topic further."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm(messages)
            
            # Extract questions from response
            questions = []
            lines = response.content.split('\n')
            
            for line in lines:
                line = line.strip()
                # Look for numbered or bulleted questions
                if re.match(r'^\d+\.|\-|\*', line):
                    question = re.sub(r'^\d+\.|\-|\*\s*', '', line).strip()
                    if question.endswith('?'):
                        questions.append(question)
            
            return questions[:5]  # Return max 5 questions
            
        except Exception as e:
            print(f"Error generating follow-up questions: {e}")
            return []
    
    def explain_answer(self, question: str, answer: str, chunks: List[RetrievalResult]) -> str:
        """Generate an explanation of how the answer was derived."""
        if not self.llm:
            return "Explanation not available - language model not initialized."
        
        system_prompt = """You are an expert at explaining how answers are derived from source material.

Instructions:
1. Explain how the answer relates to the original question
2. Identify which parts of the source material were most relevant
3. Explain the reasoning process used
4. Highlight any assumptions or limitations
5. Keep the explanation clear and educational"""
        
        context = self._prepare_context(chunks, max_chunks=3)
        
        user_prompt = f"""Question: {question}

Answer: {answer}

Source Material:
{context}

Please explain how this answer was derived from the source material and what reasoning was used."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm(messages)
            return response.content
            
        except Exception as e:
            return f"Error generating explanation: {str(e)}"


class MultiStepGenerator(AnswerGenerator):
    """
    Generator that breaks down complex questions into steps.
    """
    
    def generate_multistep_answer(self, question: str, chunks: List[RetrievalResult]) -> AnswerResult:
        """Generate answer by breaking down complex questions into steps."""
        if not self.llm:
            return AnswerResult(
                answer="❌ Language model not available.",
                confidence=0.0,
                sources=[],
                method="multistep"
            )
        
        import time
        start_time = time.time()
        
        # Step 1: Break down the question
        sub_questions = self._decompose_question(question)
        
        # Step 2: Answer each sub-question
        sub_answers = []
        for sub_q in sub_questions:
            sub_answer = self._answer_subquestion(sub_q, chunks)
            sub_answers.append((sub_q, sub_answer))
        
        # Step 3: Synthesize final answer
        final_answer = self._synthesize_answers(question, sub_answers, chunks)
        
        confidence = self._calculate_confidence(chunks)
        sources = self._prepare_sources(chunks)
        processing_time = time.time() - start_time
        
        return AnswerResult(
            answer=final_answer,
            confidence=confidence,
            sources=sources,
            method="multistep",
            processing_time=processing_time
        )
    
    def _decompose_question(self, question: str) -> List[str]:
        """Break down a complex question into simpler sub-questions."""
        system_prompt = """Break down complex questions into 2-4 simpler sub-questions that together would answer the original question.

Instructions:
1. Identify the key components of the question
2. Create specific, focused sub-questions
3. Ensure sub-questions are answerable independently
4. List each sub-question on a separate line starting with a number"""
        
        user_prompt = f"Break down this question into simpler sub-questions: {question}"
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm(messages)
            
            # Extract sub-questions
            sub_questions = []
            lines = response.content.split('\n')
            
            for line in lines:
                line = line.strip()
                if re.match(r'^\d+\.', line):
                    sub_q = re.sub(r'^\d+\.\s*', '', line).strip()
                    if sub_q:
                        sub_questions.append(sub_q)
            
            return sub_questions if sub_questions else [question]
            
        except Exception:
            return [question]  # Fallback to original question
    
    def _answer_subquestion(self, sub_question: str, chunks: List[RetrievalResult]) -> str:
        """Answer a single sub-question."""
        context = self._prepare_context(chunks, max_chunks=3)
        
        system_prompt = "Answer the specific question concisely using the provided context."
        user_prompt = f"Context:\n{context}\n\nQuestion: {sub_question}"
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm(messages)
            return response.content
            
        except Exception as e:
            return f"Could not answer: {str(e)}"
    
    def _synthesize_answers(self, original_question: str, sub_answers: List[tuple], 
                          chunks: List[RetrievalResult]) -> str:
        """Synthesize sub-answers into a comprehensive final answer."""
        sub_answers_text = "\n\n".join([
            f"Q: {sub_q}\nA: {sub_a}" 
            for sub_q, sub_a in sub_answers
        ])
        
        system_prompt = """Synthesize the answers to sub-questions into a comprehensive response to the original question.

Instructions:
1. Integrate information from all sub-answers
2. Provide a coherent, complete answer
3. Eliminate redundancy
4. Ensure the answer directly addresses the original question"""
        
        user_prompt = f"""Original Question: {original_question}

Sub-Questions and Answers:
{sub_answers_text}

Please synthesize these into a comprehensive answer to the original question."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm(messages)
            return response.content
            
        except Exception as e:
            # Fallback: concatenate sub-answers
            return f"Based on the analysis:\n\n{sub_answers_text}"


def demo_answer_generation():
    """Demonstrate answer generation capabilities."""
    print("💬 Answer Generation Demo")
    print("=" * 50)
    
    # Mock retrieval results for demo
    mock_chunks = [
        RetrievalResult(
            text="Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming. It uses algorithms to find patterns in data.",
            score=0.92,
            document_id="ml_guide",
            chunk_index=0,
            metadata={"topic": "machine_learning"}
        ),
        RetrievalResult(
            text="Deep learning is a subset of machine learning that uses neural networks with multiple layers. It's particularly effective for image recognition and natural language processing.",
            score=0.87,
            document_id="ml_guide", 
            chunk_index=1,
            metadata={"topic": "deep_learning"}
        ),
        RetrievalResult(
            text="Supervised learning uses labeled training data to learn a mapping from inputs to outputs. Common supervised learning tasks include classification and regression.",
            score=0.81,
            document_id="ml_guide",
            chunk_index=2, 
            metadata={"topic": "supervised_learning"}
        )
    ]
    
    # Initialize generator (will use mock mode if OpenAI not available)
    generator = AnswerGenerator()
    
    question = "What is machine learning and how does it work?"
    
    if generator.llm:
        print(f"Question: {question}\n")
        
        # Test different generation methods
        methods = ["comprehensive", "concise", "analytical"]
        
        for method in methods:
            print(f"🎯 {method.title()} Answer:")
            result = generator.generate_answer(question, mock_chunks, method=method)
            
            print(f"Answer: {result.answer}")
            print(f"Confidence: {result.confidence}")
            print(f"Sources: {len(result.sources)}")
            print(f"Processing time: {result.processing_time:.2f}s")
            print("-" * 30)
        
        # Test follow-up questions
        print("🔄 Follow-up Questions:")
        comprehensive_result = generator.generate_answer(question, mock_chunks, "comprehensive")
        followups = generator.generate_followup_questions(question, comprehensive_result.answer, mock_chunks)
        
        for i, followup in enumerate(followups, 1):
            print(f"{i}. {followup}")
        
    else:
        print("❌ OpenAI API not configured - showing mock response structure")
        
        # Show structure without actual generation
        mock_result = AnswerResult(
            answer="This would be the generated answer based on the retrieved context.",
            confidence=0.85,
            sources=[
                {
                    "id": 1,
                    "document_id": "ml_guide",
                    "chunk_index": 0,
                    "relevance_score": 0.92,
                    "text_preview": "Machine learning is a subset of artificial intelligence..."
                }
            ],
            method="comprehensive",
            processing_time=1.23
        )
        
        print(f"Mock Answer Result:")
        print(f"- Answer: {mock_result.answer}")
        print(f"- Confidence: {mock_result.confidence}")
        print(f"- Sources: {len(mock_result.sources)}")
        print(f"- Method: {mock_result.method}")


if __name__ == "__main__":
    demo_answer_generation()