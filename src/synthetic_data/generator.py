"""
Synthetic Data Generation - Q&A Pair Generation
Uses LLM to generate training data from research papers
"""

import sys
from pathlib import Path
from typing import List, Dict
import json
from openai import OpenAI
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import config

class SyntheticDataGenerator:
    """Generate synthetic Q&A pairs from research papers"""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.models.openai_api_key)
        self.output_dir = Path("data/training/synthetic")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_qa_from_text(self, text: str, num_pairs: int = 5) -> List[Dict]:
        """
        Generate Q&A pairs from a text chunk
        
        Args:
            text: Text from research paper
            num_pairs: Number of Q&A pairs to generate
            
        Returns:
            List of Q&A dictionaries
        """
        prompt = f"""Generate {num_pairs} question-answer pairs from this research text. 
Each pair should be suitable for training a Q&A model.

Text:
{text[:2000]}

Generate questions that:
1. Ask about key concepts
2. Request explanations of methods
3. Inquire about findings or results
4. Compare different aspects

Format each pair as:
Q: [question]
A: [answer]

Generate {num_pairs} pairs:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at creating educational Q&A pairs from research papers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            
            # Parse Q&A pairs
            pairs = self._parse_qa_pairs(content)
            return pairs
            
        except Exception as e:
            print(f"❌ Error generating Q&A: {str(e)}")
            return []
    
    def _parse_qa_pairs(self, content: str) -> List[Dict]:
        """Parse Q&A pairs from LLM response"""
        pairs = []
        lines = content.split('\n')
        
        current_q = None
        current_a = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q:'):
                if current_q and current_a:
                    pairs.append({'question': current_q, 'answer': current_a})
                current_q = line[2:].strip()
                current_a = None
            elif line.startswith('A:'):
                current_a = line[2:].strip()
        
        # Add last pair
        if current_q and current_a:
            pairs.append({'question': current_q, 'answer': current_a})
        
        return pairs
    
    def generate_from_paper(self, paper_content: str, paper_title: str) -> List[Dict]:
        """
        Generate multiple Q&A pairs from a complete paper
        
        Args:
            paper_content: Full paper text
            paper_title: Paper title
            
        Returns:
            List of Q&A pairs with metadata
        """
        print(f"📝 Generating Q&A pairs from: {paper_title}")
        
        # Split into chunks
        chunk_size = 2000
        chunks = [paper_content[i:i+chunk_size] for i in range(0, len(paper_content), chunk_size)]
        
        all_pairs = []
        
        for i, chunk in enumerate(chunks[:3], 1):  # First 3 chunks only for demo
            print(f"  Processing chunk {i}/3...")
            pairs = self.generate_qa_from_text(chunk, num_pairs=3)
            
            for pair in pairs:
                pair['source_paper'] = paper_title
                pair['chunk_index'] = i
                all_pairs.append(pair)
        
        print(f"  ✅ Generated {len(all_pairs)} Q&A pairs")
        return all_pairs
    
    def augment_existing_data(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        Augment existing Q&A pairs by rephrasing
        
        Args:
            qa_pairs: Original Q&A pairs
            
        Returns:
            Augmented pairs
        """
        print(f"🔄 Augmenting {len(qa_pairs)} Q&A pairs...")
        
        augmented = []
        
        for pair in qa_pairs[:5]:  # Demo with first 5
            prompt = f"""Rephrase this question in 2 different ways while keeping the same meaning:

Original: {pair['question']}

Provide 2 variations:"""
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.8
                )
                
                variations = response.choices[0].message.content.strip().split('\n')
                
                for variation in variations[:2]:
                    if variation.strip():
                        augmented.append({
                            'question': variation.strip(),
                            'answer': pair['answer'],
                            'source_paper': pair.get('source_paper', ''),
                            'original_question': pair['question'],
                            'augmented': True
                        })
                
            except Exception as e:
                print(f"  ⚠️ Error augmenting: {str(e)}")
        
        print(f"  ✅ Created {len(augmented)} augmented pairs")
        return augmented
    
    def create_evaluation_set(self, qa_pairs: List[Dict], ratio: float = 0.2) -> tuple:
        """
        Split data into training and evaluation sets
        
        Args:
            qa_pairs: All Q&A pairs
            ratio: Ratio for evaluation set
            
        Returns:
            (training_set, evaluation_set)
        """
        import random
        random.shuffle(qa_pairs)
        
        split_idx = int(len(qa_pairs) * (1 - ratio))
        training = qa_pairs[:split_idx]
        evaluation = qa_pairs[split_idx:]
        
        print(f"📊 Data split:")
        print(f"  Training: {len(training)} pairs")
        print(f"  Evaluation: {len(evaluation)} pairs")
        
        return training, evaluation
    
    def save_dataset(self, qa_pairs: List[Dict], filename: str):
        """Save Q&A pairs to file"""
        # Save as JSON
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        
        # Save as JSONL (for training)
        jsonl_path = self.output_dir / f"{filename}.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for pair in qa_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        print(f"💾 Saved {len(qa_pairs)} pairs to:")
        print(f"  {json_path}")
        print(f"  {jsonl_path}")
        
        return json_path
    
    def generate_statistics(self, qa_pairs: List[Dict]) -> Dict:
        """Generate statistics about the dataset"""
        stats = {
            'total_pairs': len(qa_pairs),
            'avg_question_length': sum(len(p['question']) for p in qa_pairs) / len(qa_pairs),
            'avg_answer_length': sum(len(p['answer']) for p in qa_pairs) / len(qa_pairs),
            'unique_papers': len(set(p.get('source_paper', '') for p in qa_pairs)),
            'augmented_count': sum(1 for p in qa_pairs if p.get('augmented', False))
        }
        
        return stats

def demo_synthetic_generation():
    """Demo synthetic data generation"""
    print("=" * 70)
    print("🔄 Synthetic Data Generation Demo")
    print("=" * 70)
    
    generator = SyntheticDataGenerator()
    
    # Load existing papers
    from src.rag.document_processor import DocumentProcessor
    processor = DocumentProcessor()
    
    papers_dir = Path("data/papers/raw")
    
    if not papers_dir.exists():
        print("❌ Papers directory not found")
        return
    
    # Process first paper
    pdf_files = list(papers_dir.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found")
        return
    
    print(f"\n📄 Processing: {pdf_files[0].name}")
    doc = processor.process_document(pdf_files[0])
    
    # Generate Q&A pairs
    qa_pairs = generator.generate_from_paper(
        doc.content,
        doc.metadata['title']
    )
    
    # Augment data
    augmented = generator.augment_existing_data(qa_pairs)
    
    # Combine
    all_pairs = qa_pairs + augmented
    
    # Create train/eval split
    train_set, eval_set = generator.create_evaluation_set(all_pairs)
    
    # Save datasets
    generator.save_dataset(train_set, "synthetic_train")
    generator.save_dataset(eval_set, "synthetic_eval")
    
    # Statistics
    stats = generator.generate_statistics(all_pairs)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total pairs: {stats['total_pairs']}")
    print(f"  Avg question length: {stats['avg_question_length']:.0f} chars")
    print(f"  Avg answer length: {stats['avg_answer_length']:.0f} chars")
    print(f"  Augmented pairs: {stats['augmented_count']}")
    
    # Show examples
    print(f"\n📝 Example Q&A Pairs:")
    for i, pair in enumerate(qa_pairs[:2], 1):
        print(f"\n{i}. Q: {pair['question']}")
        print(f"   A: {pair['answer'][:150]}...")

if __name__ == "__main__":
    demo_synthetic_generation()