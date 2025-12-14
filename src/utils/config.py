"""
Configuration management for the Research Assistant
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass

# Load environment variables
load_dotenv()

@dataclass
class ModelConfig:
    """LLM and embedding model configurations"""
    openai_api_key: str = os.getenv('OPENAI_API_KEY', '')
    anthropic_api_key: str = os.getenv('ANTHROPIC_API_KEY', '')
    embedding_model: str = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    llm_model: str = os.getenv('LLM_MODEL', 'ft:gpt-3.5-turbo-0125:personal:research-assistant:Cm91GJSD')
    fine_tuned_model_path: str = os.getenv('FINE_TUNED_MODEL_PATH', './models/fine_tuned/')
    
@dataclass
class RAGConfig:
    """RAG system configurations"""
    chunk_size: int = int(os.getenv('CHUNK_SIZE', 500))
    chunk_overlap: int = int(os.getenv('CHUNK_OVERLAP', 50))
    top_k_results: int = int(os.getenv('TOP_K_RESULTS', 5))
    vector_db_path: str = os.getenv('VECTOR_DB_PATH', './data/vector_db/')
    
@dataclass
class PathConfig:
    """Path configurations"""
    root_dir: Path = Path(__file__).parent.parent.parent
    data_dir: Path = root_dir / 'data'
    papers_raw: Path = data_dir / 'papers' / 'raw'
    papers_processed: Path = data_dir / 'papers' / 'processed'
    vector_db: Path = data_dir / 'vector_db'
    models_dir: Path = root_dir / 'models'
    outputs_dir: Path = root_dir / 'outputs'
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for path in [self.papers_raw, self.papers_processed, 
                     self.vector_db, self.models_dir, self.outputs_dir]:
            path.mkdir(parents=True, exist_ok=True)

class Config:
    """Main configuration class"""
    def __init__(self):
        self.models = ModelConfig()
        self.rag = RAGConfig()
        self.paths = PathConfig()
        
    def validate(self):
        """Validate that required API keys are set"""
        errors = []
        
        if not self.models.openai_api_key:
            errors.append("OPENAI_API_KEY not set in .env file")
            
        if len(errors) > 0:
            print("⚠️  Configuration warnings:")
            for error in errors:
                print(f"  - {error}")
            print("\nSome features may not work without proper API keys.")
            
        return len(errors) == 0

# Global config instance
config = Config()