"""
RAG Retriever - Integrates search, prompts, and LLM
Complete Q&A pipeline
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
import os

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.vector_store import VectorStore
from src.prompts.templates import PromptBuilder, PromptTemplates
from src.prompts.query_reformulator import QueryReformulator
from src.utils.config import config

class RAGRetriever:
    """
    Complete RAG system with query reformulation and LLM integration
    """
    
    def __init__(
        self,
        vector_store: VectorStore = None,
        use_fine_tuned: bool = True
    ):
        """
        Initialize RAG retriever
        
        Args:
            vector_store: Vector store instance
            use_fine_tuned: Whether to use fine-tuned model
        """
        # Load vector store
        self.vector_store = vector_store
        if not self.vector_store:
            self.vector_store = VectorStore()
            try:
                self.vector_store.load(config.paths.vector_db)
                print("✅ Vector store loaded")
            except FileNotFoundError:
                print("⚠️  No vector store found. Add papers first.")
        
        # Initialize components
        self.client = OpenAI(api_key=config.models.openai_api_key)
        self.query_reformulator = QueryReformulator()
        self.prompt_builder = PromptBuilder()
        self.templates = PromptTemplates()
        
        # Model selection
        self.model = config.models.llm_model if use_fine_tuned else "gpt-3.5-turbo"
        print(f"🤖 Using model: {self.model}")
        
        # Conversation history for multi-turn
        self.conversation_history = []
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        use_reformulation: bool = True
    ) -> List[Dict]:
        """
        Retrieve relevant context from papers
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            use_reformulation: Whether to reformulate query
            
        Returns:
            List of relevant chunks
        """
        if use_reformulation:
            # Reformulate query for better search
            reformulated = self.query_reformulator.smart_reformulate(query)
            search_query = reformulated['optimized']
            
            # Try multiple search variants if needed
            all_results = []
            for variant in reformulated['search_variants'][:2]:
                results = self.vector_store.search(variant, top_k=top_k)
                all_results.extend(results)
            
            # Deduplicate and sort by score
            seen_ids = set()
            unique_results = []
            for chunk, score in all_results:
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    unique_results.append((chunk, score))
            
            # Sort by score and take top_k
            unique_results.sort(key=lambda x: x[1], reverse=True)
            results = unique_results[:top_k]
        else:
            results = self.vector_store.search(query, top_k=top_k)
        
        # Format results
        context_chunks = []
        for chunk, score in results:
            context_chunks.append({
                'content': chunk.content,
                'metadata': chunk.metadata,
                'page_number': chunk.page_number,
                'chunk_id': chunk.chunk_id,
                'similarity_score': score
            })
        
        return context_chunks
    
    def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict],
        use_chain_of_thought: bool = False,
        temperature: float = 0.7
    ) -> Dict[str, any]:
        """
        Generate answer using LLM
        
        Args:
            question: User question
            context_chunks: Retrieved context
            use_chain_of_thought: Whether to use CoT reasoning
            temperature: Sampling temperature
            
        Returns:
            Dictionary with answer and metadata
        """
        # Build prompt
        prompt = self.prompt_builder.build_qa_prompt(
            question=question,
            context_chunks=context_chunks,
            use_chain_of_thought=use_chain_of_thought,
            include_examples=True
        )
        
        # Generate response
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.templates.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=temperature
            )
            
            answer = response.choices[0].message.content
            
            # Extract citations
            citations = self._extract_citations(answer, context_chunks)
            
            return {
                'answer': answer,
                'citations': citations,
                'sources': context_chunks,
                'model': self.model,
                'question': question
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'citations': [],
                'sources': context_chunks,
                'error': str(e)
            }
    
    def _extract_citations(self, answer: str, context_chunks: List[Dict]) -> List[Dict]:
        """
        Extract citations from answer
        
        Args:
            answer: Generated answer
            context_chunks: Source chunks
            
        Returns:
            List of citation dictionaries
        """
        import re
        
        citations = []
        
        # Find citation patterns like [Author, Year] or [Source 1]
        citation_pattern = r'\[([^\]]+)\]'
        matches = re.findall(citation_pattern, answer)
        
        for match in matches:
            # Try to find corresponding source
            for i, chunk in enumerate(context_chunks):
                metadata = chunk['metadata']
                title = metadata.get('title', '')
                author = metadata.get('author', '')
                
                if author.lower() in match.lower() or f"source {i+1}" in match.lower():
                    citations.append({
                        'citation_text': match,
                        'title': title,
                        'author': author,
                        'page': chunk['page_number'],
                        'chunk_id': chunk['chunk_id']
                    })
                    break
        
        return citations
    
    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        use_reformulation: bool = True,
        use_chain_of_thought: bool = False
    ) -> Dict[str, any]:
        """
        Complete Q&A pipeline
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            use_reformulation: Whether to reformulate query
            use_chain_of_thought: Whether to use CoT
            
        Returns:
            Complete answer with sources and citations
        """
        print(f"\n{'='*70}")
        print(f"❓ Question: {question}")
        print(f"{'='*70}")
        
        # Step 1: Retrieve context
        print(f"\n🔍 Retrieving relevant context...")
        context_chunks = self.retrieve_context(
            query=question,
            top_k=top_k,
            use_reformulation=use_reformulation
        )
        
        if not context_chunks:
            return {
                'answer': "I couldn't find relevant information in the papers to answer this question.",
                'citations': [],
                'sources': [],
                'question': question
            }
        
        print(f"  ✅ Found {len(context_chunks)} relevant sources")
        
        # Step 2: Generate answer
        print(f"\n🤖 Generating answer...")
        result = self.generate_answer(
            question=question,
            context_chunks=context_chunks,
            use_chain_of_thought=use_chain_of_thought
        )
        
        print(f"  ✅ Answer generated")
        
        # Add to conversation history
        self.conversation_history.append({
            'question': question,
            'answer': result['answer'],
            'sources': context_chunks
        })
        
        return result
    
    def multi_turn_conversation(
        self,
        question: str,
        use_history: bool = True
    ) -> Dict[str, any]:
        """
        Handle multi-turn conversations with context
        
        Args:
            question: Current question
            use_history: Whether to use conversation history
            
        Returns:
            Answer with conversation context
        """
        # Build conversation context
        if use_history and self.conversation_history:
            history_context = "\n\nPrevious conversation:\n"
            for turn in self.conversation_history[-3:]:  # Last 3 turns
                history_context += f"Q: {turn['question']}\nA: {turn['answer'][:200]}...\n\n"
            
            # Augment current question with history
            augmented_question = f"{history_context}Current question: {question}"
        else:
            augmented_question = question
        
        # Answer with context
        return self.answer_question(augmented_question)
    
    def summarize_paper(self, paper_title: str, num_sentences: int = 5) -> str:
        """
        Summarize a specific paper
        
        Args:
            paper_title: Title of paper to summarize
            num_sentences: Number of sentences in summary
            
        Returns:
            Summary string
        """
        # Search for paper
        results = self.vector_store.search(paper_title, top_k=10)
        
        # Filter for the specific paper
        paper_chunks = []
        for chunk, score in results:
            if paper_title.lower() in chunk.metadata.get('title', '').lower():
                paper_chunks.append(chunk.content)
        
        if not paper_chunks:
            return f"Paper '{paper_title}' not found in the database."
        
        # Combine chunks
        paper_content = "\n\n".join(paper_chunks[:5])  # First 5 chunks
        
        # Generate summary
        prompt = self.prompt_builder.build_summarization_prompt(
            content=paper_content,
            num_sentences=num_sentences
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert research paper summarizer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✅ Conversation history cleared")

# Usage example
if __name__ == "__main__":
    # Initialize retriever
    retriever = RAGRetriever(use_fine_tuned=True)
    
    # Test question
    result = retriever.answer_question(
        question="What are transformers and how do they work?",
        use_reformulation=True,
        use_chain_of_thought=False
    )
    
    print(f"\n{'='*70}")
    print(f"📝 Answer:")
    print(f"{'='*70}")
    print(result['answer'])
    
    print(f"\n{'='*70}")
    print(f"📚 Sources Used:")
    print(f"{'='*70}")
    for i, source in enumerate(result['sources'], 1):
        print(f"{i}. {source['metadata'].get('title', 'Unknown')} (Page {source['page_number']})")