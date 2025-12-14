"""
Evaluate fine-tuned model performance
Compare against base model
"""

import sys
from pathlib import Path
import json
from openai import OpenAI
import os
from typing import List, Dict
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import config

class ModelEvaluator:
    """Evaluate and compare models"""
    
    def __init__(self):
        api_key = config.models.openai_api_key or os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)
        self.results_dir = Path("data/training/fine_tuning/eval_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_test_set(self, test_file: Path = None) -> List[Dict]:
        """Load test examples"""
        if not test_file:
            test_file = Path("data/training/fine_tuning/validation.jsonl")
        
        test_examples = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                test_examples.append(json.loads(line))
        
        return test_examples[:20]  # Use first 20 for evaluation
    
    def evaluate_model(self, model_id: str, test_examples: List[Dict]) -> List[Dict]:
        """
        Evaluate model on test set
        
        Args:
            model_id: Model to evaluate
            test_examples: Test examples
            
        Returns:
            List of results with predictions
        """
        print(f"\n🧪 Evaluating model: {model_id}")
        print(f"   Test examples: {len(test_examples)}")
        
        results = []
        
        for i, example in enumerate(test_examples, 1):
            print(f"   Progress: {i}/{len(test_examples)}", end='\r')
            
            # Get user message
            messages = example['messages']
            test_messages = [msg for msg in messages if msg['role'] != 'assistant']
            expected_output = next(msg['content'] for msg in messages if msg['role'] == 'assistant')
            
            # Get model prediction
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=test_messages,
                    max_tokens=300,
                    temperature=0.7
                )
                
                prediction = response.choices[0].message.content
                
                results.append({
                    'input': test_messages[-1]['content'],
                    'expected': expected_output,
                    'predicted': prediction,
                    'model': model_id
                })
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"\n❌ Error on example {i}: {str(e)}")
                continue
        
        print(f"\n✅ Evaluation complete: {len(results)} examples")
        return results
    
    def compare_models(
        self,
        base_model: str = "gpt-3.5-turbo",
        fine_tuned_model: str = None,
        num_examples: int = 10
    ):
        """
        Compare base model vs fine-tuned model
        
        Args:
            base_model: Base model ID
            fine_tuned_model: Fine-tuned model ID
            num_examples: Number of test examples
        """
        print("=" * 70)
        print("📊 Model Comparison")
        print("=" * 70)
        
        # Load test examples
        test_examples = self.load_test_set()[:num_examples]
        
        # Evaluate base model
        print(f"\n1️⃣  Evaluating base model: {base_model}")
        base_results = self.evaluate_model(base_model, test_examples)
        
        # Evaluate fine-tuned model
        if fine_tuned_model:
            print(f"\n2️⃣  Evaluating fine-tuned model: {fine_tuned_model}")
            ft_results = self.evaluate_model(fine_tuned_model, test_examples)
        else:
            print("\n⚠️  No fine-tuned model provided. Skipping comparison.")
            ft_results = []
        
        # Save results
        self._save_comparison(base_results, ft_results)
        
        # Display comparison
        if ft_results:
            self._display_comparison(base_results, ft_results)
        
        return base_results, ft_results
    
    def _save_comparison(self, base_results: List[Dict], ft_results: List[Dict]):
        """Save comparison results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"comparison_{timestamp}.json"
        
        comparison = {
            'base_model_results': base_results,
            'fine_tuned_results': ft_results,
            'timestamp': timestamp
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def _display_comparison(self, base_results: List[Dict], ft_results: List[Dict]):
        """Display side-by-side comparison"""
        print("\n" + "=" * 70)
        print("📋 Sample Comparisons")
        print("=" * 70)
        
        for i in range(min(3, len(base_results))):
            print(f"\n{'─' * 70}")
            print(f"Example {i+1}")
            print(f"{'─' * 70}")
            
            print(f"\n📝 Input:")
            print(base_results[i]['input'][:200] + "...")
            
            print(f"\n🎯 Expected:")
            print(base_results[i]['expected'][:200] + "...")
            
            print(f"\n🤖 Base Model:")
            print(base_results[i]['predicted'][:200] + "...")
            
            if i < len(ft_results):
                print(f"\n✨ Fine-Tuned Model:")
                print(ft_results[i]['predicted'][:200] + "...")
        
        print(f"\n{'─' * 70}")
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate evaluation metrics"""
        # Simple metrics
        metrics = {
            'num_examples': len(results),
            'avg_response_length': sum(len(r['predicted']) for r in results) / len(results)
        }
        
        return metrics
    
    def human_evaluation(self, base_results: List[Dict], ft_results: List[Dict]):
        """Interactive human evaluation"""
        print("\n" + "=" * 70)
        print("👤 Human Evaluation")
        print("=" * 70)
        print("Compare responses and rate which is better (1=base, 2=fine-tuned, 3=tie)")
        
        scores = {'base': 0, 'fine_tuned': 0, 'tie': 0}
        
        for i in range(min(5, len(base_results))):
            print(f"\n{'─' * 70}")
            print(f"Example {i+1}/{min(5, len(base_results))}")
            print(f"{'─' * 70}")
            
            print(f"\nInput: {base_results[i]['input'][:200]}...")
            
            print(f"\n[1] Base Model:\n{base_results[i]['predicted'][:300]}...")
            
            if i < len(ft_results):
                print(f"\n[2] Fine-Tuned Model:\n{ft_results[i]['predicted'][:300]}...")
            
            print(f"\n[3] Tie / Both good")
            
            choice = input("\nWhich is better? (1/2/3 or skip): ").strip()
            
            if choice == '1':
                scores['base'] += 1
            elif choice == '2':
                scores['fine_tuned'] += 1
            elif choice == '3':
                scores['tie'] += 1
        
        print(f"\n📊 Human Evaluation Results:")
        print(f"   Base Model: {scores['base']}")
        print(f"   Fine-Tuned: {scores['fine_tuned']}")
        print(f"   Tie: {scores['tie']}")
        
        return scores

def main():
    """Run evaluation"""
    
    print("=" * 70)
    print("📊 Model Evaluation")
    print("=" * 70)
    
    evaluator = ModelEvaluator()
    
    # Check for fine-tuned models
    models_file = Path("data/training/fine_tuning/fine_tuned_models.json")
    
    if not models_file.exists():
        print("\n❌ No fine-tuned models found!")
        print("   Complete fine-tuning first")
        return
    
    # Load fine-tuned models
    with open(models_file, 'r') as f:
        models = json.load(f)
    
    if not models:
        print("\n❌ No fine-tuned models available")
        return
    
    # Get latest model
    latest_model = models[-1]['model_id']
    print(f"\n✅ Found fine-tuned model: {latest_model}")
    
    # Run comparison
    evaluator.compare_models(
        base_model="gpt-3.5-turbo",
        fine_tuned_model=latest_model,
        num_examples=10
    )

if __name__ == "__main__":
    main()