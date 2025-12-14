"""
Query reformulation for better search results
Expands and optimizes user queries
"""

import re
from typing import List, Dict, Optional
from openai import OpenAI
import os

class QueryReformulator:
    """Reformulate user queries for better retrieval"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize reformulator
        
        Args:
            api_key: OpenAI API key (optional, will use env var if not provided)
        """
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        
        # Keywords that indicate different query types
        self.question_types = {
            'definition': ['what is', 'define', 'meaning of', 'explain'],
            'comparison': ['compare', 'difference between', 'versus', 'vs'],
            'how_to': ['how to', 'how can', 'how do', 'steps to'],
            'why': ['why', 'reason for', 'causes of'],
            'list': ['list', 'examples of', 'types of', 'categories'],
            'evaluation': ['advantages', 'disadvantages', 'pros and cons', 'benefits']
        }
    
    def detect_query_type(self, query: str) -> str:
        """
        Detect the type of query
        
        Args:
            query: User query
            
        Returns:
            Query type string
        """
        query_lower = query.lower()
        
        for qtype, keywords in self.question_types.items():
            if any(kw in query_lower for kw in keywords):
                return qtype
        
        return 'general'
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and related terms
        
        Args:
            query: Original query
            
        Returns:
            List of expanded query variations
        """
        # Use LLM to generate query variations
        prompt = f"""Generate 3 alternative search queries that would help find information about:
"{query}"

Requirements:
- Keep the core meaning
- Use synonyms and related terms
- Make queries more specific where possible
- Each query should be on a new line
- No numbering or bullets

Alternative queries:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a search query optimization expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            variations = response.choices[0].message.content.strip().split('\n')
            variations = [v.strip() for v in variations if v.strip()]
            
            # Add original query
            return [query] + variations[:3]
            
        except Exception as e:
            print(f"⚠️  Query expansion failed: {str(e)}")
            return [query]
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from query
        
        Args:
            query: User query
            
        Returns:
            List of keywords
        """
        # Remove common words
        stop_words = {
            'what', 'is', 'are', 'how', 'why', 'when', 'where', 'who',
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'can', 'could', 'should', 'would', 'do', 'does', 'did',
            'tell', 'me', 'about', 'explain', 'describe'
        }
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def reformulate_with_llm(self, query: str) -> Dict[str, any]:
        """
        Use LLM to reformulate query comprehensively
        
        Args:
            query: Original query
            
        Returns:
            Dictionary with reformulated queries and metadata
        """
        prompt = f"""Analyze and reformulate this research question for better paper search:

Original Query: "{query}"

Provide:
1. Query Type: (e.g., definition, comparison, methodology, etc.)
2. Main Concepts: Key concepts to search for (comma-separated)
3. Search Query: Optimized search query (keywords only, no question words)
4. Alternative Query: Alternative phrasing
5. Broader Query: A broader query if the specific one fails
6. Narrower Query: A more specific query

Format your response as:
Type: [type]
Concepts: [concepts]
Search: [search query]
Alternative: [alternative]
Broader: [broader query]
Narrower: [narrower query]"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a research query optimization expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.5
            )
            
            content = response.choices[0].message.content
            
            # Parse response
            result = {
                'original': query,
                'type': self._extract_field(content, 'Type'),
                'concepts': self._extract_field(content, 'Concepts'),
                'optimized': self._extract_field(content, 'Search'),
                'alternative': self._extract_field(content, 'Alternative'),
                'broader': self._extract_field(content, 'Broader'),
                'narrower': self._extract_field(content, 'Narrower')
            }
            
            return result
            
        except Exception as e:
            print(f"⚠️  LLM reformulation failed: {str(e)}")
            return {
                'original': query,
                'optimized': query,
                'keywords': self.extract_keywords(query)
            }
    
    def _extract_field(self, text: str, field: str) -> str:
        """Extract a field from formatted text"""
        pattern = f"{field}:\\s*(.+?)(?=\\n[A-Z]|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def get_search_queries(self, query: str, max_variants: int = 3) -> List[str]:
        """
        Get multiple search query variants
        
        Args:
            query: Original query
            max_variants: Maximum number of variants to return
            
        Returns:
            List of search queries
        """
        reformulated = self.reformulate_with_llm(query)
        
        queries = [
            reformulated.get('optimized', query),
            reformulated.get('alternative', ''),
            reformulated.get('broader', ''),
            reformulated.get('narrower', '')
        ]
        
        # Filter empty and duplicates
        queries = list(dict.fromkeys([q for q in queries if q]))
        
        return queries[:max_variants]
    
    def smart_reformulate(self, query: str) -> Dict[str, any]:
        """
        Comprehensive query reformulation with all strategies
        
        Args:
            query: User query
            
        Returns:
            Dictionary with all reformulation results
        """
        print(f"\n🔄 Reformulating query: '{query}'")
        
        # Detect type
        qtype = self.detect_query_type(query)
        print(f"  📋 Query type: {qtype}")
        
        # Get LLM reformulation
        llm_result = self.reformulate_with_llm(query)
        
        # Extract keywords
        keywords = self.extract_keywords(query)
        
        # Get search variants
        search_queries = self.get_search_queries(query)
        
        result = {
            'original': query,
            'type': qtype,
            'keywords': keywords,
            'optimized': llm_result.get('optimized', query),
            'search_variants': search_queries,
            'concepts': llm_result.get('concepts', ''),
            'llm_analysis': llm_result
        }
        
        print(f"  ✅ Optimized: '{result['optimized']}'")
        print(f"  🔑 Keywords: {', '.join(keywords)}")
        
        return result

# Usage example
if __name__ == "__main__":
    reformulator = QueryReformulator()
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "How do transformers work?",
        "Compare BERT and GPT",
        "Why is attention mechanism important?"
    ]
    
    for query in test_queries:
        result = reformulator.smart_reformulate(query)
        print(f"\nOriginal: {result['original']}")
        print(f"Optimized: {result['optimized']}")
        print(f"Variants: {result['search_variants']}")
        print("-" * 70)