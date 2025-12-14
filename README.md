# AI-Powered Research Assistant

An advanced research assistant that combines RAG, Fine-tuning, Prompt Engineering, Multimodal Processing, and Synthetic Data Generation.

## 🎯 Features

- 📚 **RAG System**: Semantic search across research papers
- 🎯 **Prompt Engineering**: Advanced query reformulation
- 🔧 **Fine-tuned Models**: Domain-specific expertise
- 🖼️ **Multimodal Processing**: Images, tables, charts
- 🔄 **Synthetic Data Generation**: Training data creation

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install packages
pip install -r requirements.txt
```

### 2. Add Research Papers

Place PDF files in: `data/papers/raw/`

### 3. Run the Application
```bash
# Option 1: Command line demo
python demo_rag.py

# Option 2: Web interface
cd web
streamlit run app.py
```

## 📁 Project Structure
```
research_assistant/
├── src/
│   ├── rag/              # RAG system
│   ├── prompts/          # Prompt engineering
│   ├── fine_tuning/      # Model fine-tuning
│   ├── multimodal/       # Image/table processing
│   └── synthetic_data/   # Data generation
├── data/
│   ├── papers/raw/       # PDF files
│   └── vector_db/        # Vector store
├── web/                  # Web interface
└── demo_rag.py          # Demo script
```

## 🔧 Configuration

Edit `.env` file:
```bash
OPENAI_API_KEY=your_key_here
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=500
TOP_K_RESULTS=5
```

## 📖 Usage

### Search Papers
```python
from src.rag.vector_store import VectorStore

vector_store = VectorStore()
vector_store.load("./data/vector_db")

results = vector_store.search("What is deep learning?", top_k=5)
```

### Process New Papers
```python
from src.rag.document_processor import DocumentProcessor

processor = DocumentProcessor()
docs = processor.process_directory("./data/papers/raw")
```

## 🎓 Project for Academic Assignment

This project fulfills requirements for:
- ✅ RAG implementation
- ✅ Prompt engineering
- ✅ Fine-tuning (coming in Part 3)
- ✅ Multimodal processing (coming in Part 4)
- ✅ Synthetic data generation (coming in Part 5)

## 📝 License

MIT License - Created for educational purposes
```

---

## 📄 File 9: `.gitignore`
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Project specific
.env
data/papers/raw/*.pdf
data/vector_db/*
models/fine_tuned/*
*.log

# Jupyter
.ipynb_checkpoints/
*.ipynb

# OS
.DS_Store
Thumbs.db