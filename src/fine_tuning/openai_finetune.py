"""
OpenAI Fine-Tuning Manager
Upload data, start fine-tuning, and manage jobs
"""

import os
import sys
from pathlib import Path
import time
from openai import OpenAI
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import config

class OpenAIFineTuner:
    """Manage OpenAI fine-tuning jobs"""
    
    def __init__(self):
        # Initialize OpenAI client
        api_key = config.models.openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError(
                "OpenAI API key not found! "
                "Add OPENAI_API_KEY to your .env file"
            )
        
        self.client = OpenAI(api_key=api_key)
        self.data_dir = Path("data/training/fine_tuning")
        
        print("✅ OpenAI client initialized")
    
    def upload_training_file(self, filepath: Path):
        """Upload training data to OpenAI"""
        print(f"\n📤 Uploading training file: {filepath.name}")
        
        with open(filepath, 'rb') as f:
            response = self.client.files.create(
                file=f,
                purpose='fine-tune'
            )
        
        file_id = response.id
        print(f"✅ File uploaded successfully!")
        print(f"   File ID: {file_id}")
        
        # Wait for processing
        print("⏳ Waiting for file to be processed...")
        while True:
            file_info = self.client.files.retrieve(file_id)
            if file_info.status == 'processed':
                print("✅ File processed successfully!")
                break
            elif file_info.status == 'error':
                print(f"❌ File processing failed: {file_info}")
                return None
            time.sleep(2)
        
        return file_id
    
    def start_fine_tuning(
        self,
        training_file_id: str,
        validation_file_id: str = None,
        model: str = "gpt-3.5-turbo",
        suffix: str = "research-assistant",
        n_epochs: int = 3
    ):
        """
        Start a fine-tuning job
        
        Args:
            training_file_id: ID of uploaded training file
            validation_file_id: ID of uploaded validation file (optional)
            model: Base model to fine-tune (gpt-3.5-turbo or gpt-4o-mini)
            suffix: Name suffix for the fine-tuned model
            n_epochs: Number of training epochs
        """
        print(f"\n🚀 Starting fine-tuning job...")
        print(f"   Base model: {model}")
        print(f"   Epochs: {n_epochs}")
        
        # Create fine-tuning job
        job_params = {
            "training_file": training_file_id,
            "model": model,
            "suffix": suffix,
            "hyperparameters": {
                "n_epochs": n_epochs
            }
        }
        
        if validation_file_id:
            job_params["validation_file"] = validation_file_id
        
        job = self.client.fine_tuning.jobs.create(**job_params)
        
        job_id = job.id
        print(f"✅ Fine-tuning job created!")
        print(f"   Job ID: {job_id}")
        
        # Save job info
        self._save_job_info(job_id, model, suffix)
        
        return job_id
    
    def monitor_job(self, job_id: str, check_interval: int = 60):
        """
        Monitor fine-tuning job progress
        
        Args:
            job_id: Job ID to monitor
            check_interval: Seconds between status checks
        """
        print(f"\n👀 Monitoring job: {job_id}")
        print("   This typically takes 10-30 minutes...")
        print("   You can close this and check later with: list_jobs()")
        
        while True:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            
            print(f"\n⏱️  Status: {status}")
            
            if status == 'succeeded':
                print(f"\n🎉 Fine-tuning completed successfully!")
                print(f"   Fine-tuned model: {job.fine_tuned_model}")
                
                # Save model info
                self._save_model_info(job.fine_tuned_model, job_id)
                
                return job.fine_tuned_model
            
            elif status == 'failed':
                print(f"\n❌ Fine-tuning failed!")
                print(f"   Error: {job.error}")
                return None
            
            elif status == 'cancelled':
                print(f"\n⚠️  Fine-tuning was cancelled")
                return None
            
            # Show progress if available
            if hasattr(job, 'trained_tokens') and job.trained_tokens:
                print(f"   Trained tokens: {job.trained_tokens}")
            
            time.sleep(check_interval)
    
    def list_jobs(self, limit: int = 10):
        """List recent fine-tuning jobs"""
        print(f"\n📋 Recent fine-tuning jobs:")
        
        jobs = self.client.fine_tuning.jobs.list(limit=limit)
        
        for i, job in enumerate(jobs.data, 1):
            print(f"\n{i}. Job ID: {job.id}")
            print(f"   Status: {job.status}")
            print(f"   Model: {job.model}")
            if job.fine_tuned_model:
                print(f"   Fine-tuned: {job.fine_tuned_model}")
            print(f"   Created: {job.created_at}")
    
    def cancel_job(self, job_id: str):
        """Cancel a fine-tuning job"""
        print(f"\n🛑 Cancelling job: {job_id}")
        
        self.client.fine_tuning.jobs.cancel(job_id)
        print("✅ Job cancelled")
    
    def test_fine_tuned_model(self, model_id: str, test_prompt: str):
        """
        Test a fine-tuned model
        
        Args:
            model_id: Fine-tuned model ID (e.g., ft:gpt-3.5-turbo:org:suffix:id)
            test_prompt: Test prompt
        """
        print(f"\n🧪 Testing model: {model_id}")
        print(f"Prompt: {test_prompt}\n")
        
        response = self.client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are an expert research assistant."},
                {"role": "user", "content": test_prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        result = response.choices[0].message.content
        print("Response:")
        print("=" * 70)
        print(result)
        print("=" * 70)
        
        return result
    
    def get_cost_estimate(self, training_file: Path):
        """Estimate fine-tuning cost"""
        # Count tokens (rough estimate: 1 token ≈ 4 characters)
        with open(training_file, 'r', encoding='utf-8') as f:
            content = f.read()
            approx_tokens = len(content) / 4
        
        # GPT-3.5-turbo fine-tuning costs (as of 2024)
        # Training: $0.008 / 1K tokens
        # Usage: $0.003 input / $0.006 output per 1K tokens
        
        epochs = 3
        training_cost = (approx_tokens * epochs / 1000) * 0.008
        
        print(f"\n💰 Cost Estimate (GPT-3.5-turbo):")
        print(f"   Approximate tokens: {approx_tokens:,.0f}")
        print(f"   Training cost: ${training_cost:.2f}")
        print(f"   Usage cost: ~$0.003-0.006 per 1K tokens")
        print(f"\n💡 Actual cost may vary based on token count")
    
    def _save_job_info(self, job_id: str, model: str, suffix: str):
        """Save job information"""
        jobs_file = self.data_dir / "fine_tuning_jobs.json"
        
        job_info = {
            "job_id": job_id,
            "model": model,
            "suffix": suffix,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Load existing jobs
        jobs = []
        if jobs_file.exists():
            with open(jobs_file, 'r') as f:
                jobs = json.load(f)
        
        jobs.append(job_info)
        
        # Save
        with open(jobs_file, 'w') as f:
            json.dump(jobs, f, indent=2)
    
    def _save_model_info(self, model_id: str, job_id: str):
        """Save fine-tuned model information"""
        models_file = self.data_dir / "fine_tuned_models.json"
        
        model_info = {
            "model_id": model_id,
            "job_id": job_id,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Load existing models
        models = []
        if models_file.exists():
            with open(models_file, 'r') as f:
                models = json.load(f)
        
        models.append(model_info)
        
        # Save
        with open(models_file, 'w') as f:
            json.dump(models, f, indent=2)
        
        print(f"\n💾 Model info saved to: {models_file}")

def main():
    """Main fine-tuning workflow"""
    
    print("=" * 70)
    print("🔧 OpenAI Fine-Tuning Manager")
    print("=" * 70)
    
    # Initialize
    tuner = OpenAIFineTuner()
    
    # Check for training files
    data_dir = Path("data/training/fine_tuning")
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "validation.jsonl"
    
    if not train_file.exists():
        print("\n❌ Training file not found!")
        print("   Run data preparation first:")
        print("   python src/fine_tuning/openai_data_prep.py")
        return
    
    # Show cost estimate
    tuner.get_cost_estimate(train_file)
    
    # Ask for confirmation
    print("\n" + "=" * 70)
    response = input("Start fine-tuning? This will cost money. (yes/no): ")
    
    if response.lower() != 'yes':
        print("❌ Fine-tuning cancelled")
        return
    
    # Upload files
    train_file_id = tuner.upload_training_file(train_file)
    
    val_file_id = None
    if val_file.exists():
        val_file_id = tuner.upload_training_file(val_file)
    
    # Start fine-tuning
    job_id = tuner.start_fine_tuning(
        training_file_id=train_file_id,
        validation_file_id=val_file_id,
        model="gpt-3.5-turbo",  # or "gpt-4o-mini"
        suffix="research-assistant",
        n_epochs=3
    )
    
    # Monitor job
    print("\n" + "=" * 70)
    monitor = input("Monitor job progress? (yes/no): ")
    
    if monitor.lower() == 'yes':
        model_id = tuner.monitor_job(job_id)
        
        if model_id:
            # Test the model
            test_prompt = "Summarize what machine learning is in 2-3 sentences."
            tuner.test_fine_tuned_model(model_id, test_prompt)
    else:
        print(f"\n💡 Check job status later with:")
        print(f"   python -c \"from src.fine_tuning.openai_finetune import OpenAIFineTuner; t=OpenAIFineTuner(); t.list_jobs()\"")

if __name__ == "__main__":
    main()