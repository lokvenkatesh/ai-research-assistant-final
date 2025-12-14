"""
Prepare training data for OpenAI fine-tuning
Creates properly formatted JSONL files
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import requests
import time
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.document_processor import DocumentProcessor
from src.rag.vector_store import VectorStore
from src.utils.config import config

class OpenAIDataPreparator:
    """Prepare data for OpenAI fine-tuning"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("data/training/fine_tuning")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def format_for_openai(self, examples: List[Dict]) -> List[Dict]:
        """
        Format data for OpenAI fine-tuning
        OpenAI format: {"messages": [{"role": "system"/"user"/"assistant", "content": "..."}]}
        """
        formatted_data = []
        
        for example in examples:
            formatted_example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert research assistant specializing in analyzing academic papers. Provide clear, accurate, and well-cited responses."
                    },
                    {
                        "role": "user",
                        "content": f"{example['instruction']}\n\n{example['input']}"
                    },
                    {
                        "role": "assistant",
                        "content": example['output']
                    }
                ]
            }
            formatted_data.append(formatted_example)
        
        return formatted_data
    
    def create_from_existing_papers(self, use_vector_store: bool = True) -> List[Dict]:
        """
        Create training examples from papers already in your RAG system
        """
        print("\n📚 Creating training data from your papers...")
        
        examples = []
        
        if use_vector_store:
            # Load existing vector store
            try:
                vector_store = VectorStore()
                vector_store.load(config.paths.vector_db)
                
                print(f"✅ Loaded {vector_store.index.ntotal} chunks from vector store")
                
                # Create examples from chunks
                for i, chunk in enumerate(vector_store.chunk_metadata[:100]):  # Use first 100 chunks
                    # Summarization task
                    if len(chunk.content) > 200:
                        examples.append({
                            "instruction": "Summarize the following research content concisely.",
                            "input": chunk.content,
                            "output": self._create_summary(chunk.content)
                        })
                    
                    # Q&A task
                    if chunk.metadata.get('title'):
                        examples.append({
                            "instruction": "Answer this question about the research paper.",
                            "input": f"What does this paper discuss?\n\nContent: {chunk.content[:300]}",
                            "output": f"This paper, titled '{chunk.metadata['title']}', discusses {chunk.content[:200]}..."
                        })
                
                print(f"✅ Created {len(examples)} examples from your papers")
                
            except FileNotFoundError:
                print("⚠️  No vector store found. Add papers first or use arXiv collection.")
        
        return examples
    
    def collect_from_arxiv(self, query: str = "cat:cs.AI", max_results: int = 50) -> List[Dict]:
        """
        Collect papers from arXiv and create training examples
        """
        print(f"\n📥 Collecting papers from arXiv: {query}")
        
        arxiv_api = "http://export.arxiv.org/api/query"
        examples = []
        
        try:
            params = {
                'search_query': query,
                'start': 0,
                'max_results': max_results,
                'sortBy': 'lastUpdatedDate',
                'sortOrder': 'descending'
            }
            
            response = requests.get(arxiv_api, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            entries = root.findall('{http://www.w3.org/2005/Atom}entry')
            
            print(f"✅ Found {len(entries)} papers")
            
            for entry in tqdm(entries, desc="Processing papers"):
                paper = self._parse_arxiv_entry(entry)
                
                # Create multiple training examples per paper
                paper_examples = self._create_paper_examples(paper)
                examples.extend(paper_examples)
            
            print(f"✅ Created {len(examples)} training examples from arXiv")
            
        except Exception as e:
            print(f"❌ Error collecting from arXiv: {str(e)}")
        
        return examples
    
    def _parse_arxiv_entry(self, entry) -> Dict:
        """Parse arXiv XML entry"""
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        title = entry.find('atom:title', ns).text.strip()
        summary = entry.find('atom:summary', ns).text.strip()
        
        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns).text
            authors.append(name)
        
        return {
            'title': title,
            'summary': summary,
            'authors': ', '.join(authors)
        }
    
    def _create_paper_examples(self, paper: Dict) -> List[Dict]:
        """Create multiple training examples from a paper"""
        examples = []
        
        # Example 1: Summarization
        examples.append({
            "instruction": "Summarize this research paper abstract in 2-3 clear sentences.",
            "input": paper['summary'],
            "output": self._create_summary(paper['summary'])
        })
        
        # Example 2: Main topic
        examples.append({
            "instruction": "What is the main topic of this research paper?",
            "input": f"Title: {paper['title']}\n\nAbstract: {paper['summary'][:300]}...",
            "output": f"The main topic is {paper['title'].lower()}. {paper['summary'][:150]}..."
        })
        
        # Example 3: Key findings
        if len(paper['summary']) > 400:
            examples.append({
                "instruction": "What are the key findings of this research?",
                "input": paper['summary'],
                "output": self._extract_findings(paper['summary'])
            })
        
        return examples
    
    def _create_summary(self, text: str) -> str:
        """Create a concise summary"""
        sentences = text.split('. ')
        if len(sentences) >= 3:
            return '. '.join(sentences[:3]) + '.'
        return text
    
    def _extract_findings(self, text: str) -> str:
        """Extract key findings from abstract"""
        sentences = text.split('. ')
        if len(sentences) >= 4:
            # Usually findings are in the middle/end
            return '. '.join(sentences[-3:]) + '.'
        return text
    
    def save_for_openai(self, examples: List[Dict], filename: str = "training_data"):
        """
        Save in OpenAI fine-tuning format (JSONL)
        """
        # Format for OpenAI
        formatted_examples = self.format_for_openai(examples)
        
        # Save as JSONL
        output_file = self.output_dir / f"{filename}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in formatted_examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"\n💾 Saved {len(formatted_examples)} examples to: {output_file}")
        print(f"📊 File size: {output_file.stat().st_size / 1024:.2f} KB")
        
        # Validate format
        self._validate_format(output_file)
        
        return output_file
    
    def _validate_format(self, filepath: Path):
        """Validate the JSONL format"""
        print("\n✅ Validating format...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines[:5]):  # Check first 5
            try:
                data = json.loads(line)
                assert "messages" in data
                assert len(data["messages"]) >= 2
                assert all("role" in msg and "content" in msg for msg in data["messages"])
            except Exception as e:
                print(f"❌ Line {i+1} format error: {str(e)}")
                return False
        
        print("✅ Format validation passed!")
        return True
    
    def create_train_val_split(self, examples: List[Dict], train_ratio: float = 0.8):
        """Split data into training and validation sets"""
        import random
        random.shuffle(examples)
        
        split_idx = int(len(examples) * train_ratio)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        print(f"\n📊 Dataset split:")
        print(f"  Training: {len(train_examples)} examples")
        print(f"  Validation: {len(val_examples)} examples")
        
        return train_examples, val_examples

def main():
    """Main data preparation pipeline"""
    
    print("=" * 70)
    print("🔧 OpenAI Fine-Tuning - Data Preparation")
    print("=" * 70)
    
    preparator = OpenAIDataPreparator()
    
    all_examples = []
    
    # Option 1: Use your existing papers
    print("\n📌 Option 1: Creating from your existing papers...")
    existing_examples = preparator.create_from_existing_papers()
    all_examples.extend(existing_examples)
    
    # Option 2: Collect from arXiv (multiple categories)
    print("\n📌 Option 2: Collecting from arXiv...")
    categories = ["cat:cs.AI", "cat:cs.LG", "cat:cs.CL"]
    
    for category in categories:
        print(f"\n  Fetching {category}...")
        arxiv_examples = preparator.collect_from_arxiv(category, max_results=30)
        all_examples.extend(arxiv_examples)
        time.sleep(3)  # Rate limiting
    
    print(f"\n✅ Total examples collected: {len(all_examples)}")
    
    # Split into train and validation
    train_examples, val_examples = preparator.create_train_val_split(all_examples)
    
    # Save both sets
    train_file = preparator.save_for_openai(train_examples, "train")
    val_file = preparator.save_for_openai(val_examples, "validation")
    
    print("\n" + "=" * 70)
    print("✅ Data preparation complete!")
    print("=" * 70)
    print(f"\n📁 Files created:")
    print(f"  Training: {train_file}")
    print(f"  Validation: {val_file}")
    print(f"\n🚀 Next step: Run fine-tuning with these files!")
    print(f"   python src/fine_tuning/openai_finetune.py")

if __name__ == "__main__":
    main()