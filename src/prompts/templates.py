"""
Prompt templates for research assistant
Includes templates for Q&A, summarization, and citation
"""

from typing import Dict, List

class PromptTemplates:
    """Collection of prompt templates for different tasks"""
    
    # System prompt for research assistant
    SYSTEM_PROMPT = """You are an expert research assistant specializing in academic paper analysis. Your role is to:

1. Provide accurate, well-researched answers based on the provided context
2. Always cite your sources using [Author, Year] format
3. Be concise but comprehensive
4. Acknowledge when information is uncertain or not in the provided context
5. Use clear, academic language appropriate for researchers

Guidelines:
- Base your answers on the provided research paper excerpts
- Always include citations for specific claims
- If the context doesn't contain enough information, say so
- Maintain objectivity and academic rigor"""

    # Q&A with context
    QA_WITH_CONTEXT = """Based on the following research paper excerpts, answer the question comprehensively.

Research Context:
{context}

Question: {question}

Instructions:
- Provide a clear, well-structured answer
- Cite specific papers using [Author, Year] format
- If multiple papers discuss the topic, synthesize the information
- Note any conflicting viewpoints if present
- If the context doesn't fully answer the question, acknowledge the limitations

Answer:"""

    # Summarization prompt
    SUMMARIZATION = """Summarize the following research content in {num_sentences} clear, concise sentences.

Content:
{content}

Focus on:
- Main contributions
- Key findings
- Methodology (if relevant)
- Implications

Summary:"""

    # Chain-of-thought reasoning
    CHAIN_OF_THOUGHT = """Let's approach this research question step by step.

Question: {question}

Context:
{context}

Think through this systematically:

Step 1 - Understanding the question:
[What is being asked? What key concepts are involved?]

Step 2 - Analyzing the context:
[What relevant information do the papers provide?]

Step 3 - Synthesizing the answer:
[How does the evidence come together?]

Step 4 - Final answer:
[Comprehensive answer with citations]

Analysis:"""

    @staticmethod
    def format_context(chunks: List[Dict]) -> str:
        """Format document chunks into context string"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get('metadata', {}).get('title', 'Unknown')
            author = chunk.get('metadata', {}).get('author', 'Unknown')
            page = chunk.get('page_number', '?')
            content = chunk.get('content', '')
            
            context_part = f"""[Source {i}] {title}
Author: {author} | Page: {page}
{content}
"""
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)
    
    @staticmethod
    def create_few_shot_examples() -> str:
        """Create few-shot examples for in-context learning"""
        examples = """Here are examples of how to answer research questions:

Example 1:
Q: What is the main contribution of transformers?
A: Transformers introduced a novel attention mechanism that processes sequences in parallel rather than sequentially. The key innovation is the self-attention mechanism, which allows the model to weigh the importance of different parts of the input when producing each output.

Example 2:
Q: How does transfer learning work?
A: Transfer learning involves pre-training a model on a large dataset and then fine-tuning it on a smaller, task-specific dataset. The pre-trained model learns general features that can be adapted to new tasks with minimal additional training.

Now answer the user's question following this style:
"""
        return examples

class PromptBuilder:
    """Helper class to build prompts dynamically"""
    
    def __init__(self):
        self.templates = PromptTemplates()
    
    def build_qa_prompt(
        self, 
        question: str, 
        context_chunks: List[Dict],
        use_chain_of_thought: bool = False,
        include_examples: bool = False
    ) -> str:
        """Build a Q&A prompt with context"""
        # Format context
        context = self.templates.format_context(context_chunks)
        
        # Choose template
        if use_chain_of_thought:
            template = self.templates.CHAIN_OF_THOUGHT
        else:
            template = self.templates.QA_WITH_CONTEXT
        
        # Add few-shot examples if requested
        prompt_parts = []
        if include_examples:
            prompt_parts.append(self.templates.create_few_shot_examples())
        
        # Fill template
        prompt = template.format(question=question, context=context)
        prompt_parts.append(prompt)
        
        return "\n\n".join(prompt_parts)
    
    def build_summarization_prompt(self, content: str, num_sentences: int = 3) -> str:
        """Build a summarization prompt"""
        return self.templates.SUMMARIZATION.format(
            content=content,
            num_sentences=num_sentences
        )