# 🤖 AI-Powered Research Assistant

**Search research papers, get cited answers, and analyze figures with RAG, a fine-tuned LLM, and multimodal processing.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5%20fine--tuned-412991) ![LangChain](https://img.shields.io/badge/LangChain-RAG-green) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Overview

Researchers lose hours skimming papers to find one answer. This assistant lets you upload PDFs, ask questions in plain language, and get answers grounded in the papers, with citations and page numbers you can verify.

It combines five generative AI techniques in one working application:

| Technique | What it does here |
|---|---|
| **RAG** | Retrieves the most relevant passages before the model answers |
| **Fine-tuning** | A GPT-3.5 model specialized for academic paper analysis |
| **Prompt engineering** | Query reformulation, context-aware templates, citation extraction |
| **Multimodal processing** | Extracts and indexes figures, charts, and diagrams from PDFs |
| **Synthetic data** | Generates Q&A pairs to build training and evaluation sets |

### Key results

| Metric | Result |
|---|---|
| Average response time | **~1.8 s** |
| Search latency | **< 1 s** |
| Fine-tuned model preferred over base | **72%** of comparisons |
| Retrieval accuracy gain from prompt optimization | **15–20%** |
| Semantic chunks indexed | **98** (from 5 papers) |
| Figures and charts extracted | **35** |
| Synthetic Q&A pairs generated | **29** (19 original + 10 augmented) |

---

## ✨ Features

### 1. Dual RAG system
- **Custom RAG** built from scratch on FAISS, with full control over chunking and retrieval
- **LangChain RAG** for a standard framework implementation
- Switch between the two in the web interface to compare them
- 384-dimensional Sentence-BERT embeddings (`all-MiniLM-L6-v2`)

### 2. Fine-tuned GPT-3.5
- Trained on 220 research examples (56 held out for validation) over 3 epochs via the OpenAI API
- Produces more specific, research-focused answers with citations
- Side-by-side comparison with the base model in the UI

### 3. Prompt engineering
- Automatic LLM-based query reformulation
- Context-aware prompt template selection
- Chain-of-thought reasoning and automatic citation extraction

### 4. Multimodal processing
- Automated extraction of figures, charts, and diagrams from PDFs
- Metadata tracking (page, dimensions, format) and a searchable image index
- Table detection

### 5. Synthetic data generation
- LLM-generated Q&A pairs from source papers
- Augmentation through question rephrasing
- Automatic train/eval split, exported as JSON and JSONL

### Also included
- **Conversation history** stored in SQLite, searchable by keyword
- **Analytics dashboard** with real-time Plotly charts
- **User feedback** with thumbs up/down on answers
- **Export** conversations as JSON

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│           Web Interface (Streamlit)                 │
│   Q&A | Search | History | Upload | Analytics       │
└────────────────┬────────────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼─────────┐         ┌────▼──────────┐
│  RAG System │◄───────►│  Fine-Tuned   │
│  Custom +   │         │   GPT-3.5     │
│  LangChain  │         │               │
└───┬─────────┘         └────┬──────────┘
    │                        │
┌───▼──────────┐      ┌─────▼───────────┐
│ Vector Store │      │ SQLite Database │
│ FAISS        │      │ Conversations   │
│ 98 Chunks    │      │ Analytics       │
└──────────────┘      └─────────────────┘
```

### Request flow
1. A PDF is uploaded and its text and images are extracted
2. Text is split into overlapping chunks and embedded
3. Vectors are stored and indexed in FAISS
4. A user question is reformulated for better retrieval
5. The most relevant chunks are retrieved and assembled into a prompt
6. The fine-tuned model generates an answer, and citations are extracted
7. The exchange is saved and analytics are updated
8. The answer is shown with its sources and similarity scores

---

## 🚀 Quick start

### Prerequisites
- Python 3.10+
- An OpenAI API key
- 4 GB+ RAM and 2 GB+ free disk space

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/lokvenkatesh/ai-research-assistant-final.git
cd ai-research-assistant-final

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
```

### Configuration

Add your key to `.env`:

```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# Model settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gpt-3.5-turbo        # or your own fine-tuned model ID

# RAG settings
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=5
```

> **Note:** Fine-tuned OpenAI models are private to the account that trained them. To use a fine-tuned model, run the fine-tuning scripts in `src/fine_tuning/` with your own key and set `LLM_MODEL` to the resulting model ID.

---

## 💻 Usage

### Web interface (recommended)

```bash
streamlit run web/app.py
```

Then open http://localhost:8501.

| Tab | Purpose |
|---|---|
| 💬 Ask Questions | Q&A with citations |
| 🔍 Search Papers | Semantic search across the knowledge base |
| 📜 History | Past conversations and usage tracking |
| 📤 Upload Papers | Add new PDFs |
| 📊 Analytics | System metrics |
| 🖼️ Multimodal | Extracted images and synthetic data |

### Command line

```bash
python demo_qa.py --mode interactive              # Interactive Q&A
python demo_qa.py --mode demo                     # Preset demo questions
python demo_qa.py --question "What is transfer learning?"
python demo_rag.py                                # RAG search only
```

---

## 📊 Example outputs

### Semantic search

**Query:** `transfer learning`

```
Result #1 - Similarity: 0.480
Document: Guided Transfer Learning for Discrete Diffusion Models
Page: 3
Content: Transfer learning involves pre-training a model on a large
dataset and then fine-tuning it on a smaller, task-specific dataset...
```

### Q&A with citations

**Question:** "What is transfer learning?"

**Answer (fine-tuned model):**
> Transfer learning is a machine learning technique that leverages knowledge from pre-trained models and adapts it to new, related tasks with limited data [Kleutgens et al., 2024]. The approach is particularly effective in discrete diffusion models, where guided transfer learning enables efficient domain adaptation [Kleutgens et al., 2024].

**Sources:** Guided Transfer Learning for Discrete Diffusion Models (pages 3, 4, 15) · Similarity: 0.480, 0.466, 0.456

### Fine-tuned vs. base model

| Base GPT-3.5 | Fine-tuned |
|---|---|
| "Transfer learning is a technique in machine learning where a model developed for one task is reused as the starting point for a model on a second task..." | "Transfer learning in the context of discrete diffusion models involves adapting pre-trained models through guided learning approaches that enable efficient knowledge transfer with minimal additional training [Kleutgens et al., 2024]..." |
| Generic, uncited | Specific, research-focused, cited |

### Synthetic Q&A pair

```json
{
  "question": "What are the main contributions of guided transfer learning?",
  "answer": "A unified framework for transfer learning in discrete diffusion models, a guided formulation that improves adaptation efficiency, and demonstrated improvements across multiple domains.",
  "source_paper": "Guided Transfer Learning for Discrete Diffusion Models",
  "augmented": false
}
```

---

## 🗂️ Data and models

### Knowledge base
- **Papers:** 5 AI/ML research papers from arXiv, including *Guided Transfer Learning for Discrete Diffusion Models*
- **Chunks:** 98, split on sentence boundaries (500 words, 50-word overlap)
- **Metadata:** title, author, page number, source file
- **Index:** FAISS FlatL2, 384 dimensions

### Fine-tuning dataset
| Split | Examples |
|---|---|
| Training | 220 |
| Validation | 56 |

About 300,000 tokens processed across 3 epochs. Sample format:

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert research assistant specializing in academic paper analysis."},
    {"role": "user", "content": "Summarize this research paper abstract in 2-3 clear sentences.\n\nAbstract: [abstract text]"},
    {"role": "assistant", "content": "This paper presents a novel approach to transfer learning in discrete diffusion models..."}
  ]
}
```

---

## 📁 Project structure

```
ai-research-assistant-final/
├── src/
│   ├── rag/
│   │   ├── document_processor.py    # PDF processing and chunking
│   │   ├── vector_store.py          # FAISS vector database
│   │   ├── retriever.py             # Custom RAG pipeline
│   │   └── langchain_retriever.py   # LangChain RAG
│   ├── prompts/
│   │   ├── templates.py             # Prompt templates
│   │   └── query_reformulator.py    # Query optimization
│   ├── fine_tuning/
│   │   ├── openai_data_prep.py      # Data preparation
│   │   ├── openai_finetune.py       # Fine-tuning job
│   │   └── evaluate.py              # Model evaluation
│   ├── multimodal/
│   │   └── image_extractor.py       # Image extraction from PDFs
│   ├── synthetic_data/
│   │   └── generator.py             # Q&A pair generation
│   └── utils/
│       ├── config.py                # Configuration
│       └── database.py              # SQLite storage
├── web/
│   └── app.py                       # Streamlit app
├── data/
│   ├── papers/raw/                  # Source PDFs
│   ├── vector_db/                   # FAISS index and metadata
│   ├── images/                      # Extracted figures
│   └── training/                    # Fine-tuning datasets
├── docs/
│   └── project_report.pdf           # Full technical report (25+ pages)
├── screenshots/
├── demo_rag.py
├── demo_qa.py
├── test_finetuned.py
├── requirements.txt
└── .env.example
```

---

## 🧪 Testing

```bash
python demo_rag.py --test            # Document processing, search, retrieval
python demo_qa.py --mode test        # Full Q&A pipeline
python test_finetuned.py             # Fine-tuned vs. base model
pytest tests/                        # Unit tests
```

Individual components:

```bash
python src/rag/langchain_retriever.py
python src/multimodal/image_extractor.py
python src/synthetic_data/generator.py
```

---

## 🛠️ Tech stack

| Layer | Technologies |
|---|---|
| AI / ML | OpenAI GPT-3.5 (fine-tuned), Sentence-BERT, FAISS, LangChain, Hugging Face Transformers |
| Backend | Python, SQLite, PyMuPDF, Pillow |
| Frontend | Streamlit, Plotly, custom CSS |

---

## 🔧 Troubleshooting

| Error | Fix |
|---|---|
| `No module named 'fitz'` | `pip install PyMuPDF` |
| `OpenAI API key not found` | Check that `.env` exists and contains `OPENAI_API_KEY` |
| `No papers found` | Add PDFs to `data/papers/raw/`, then run `python demo_rag.py` |
| FAISS import error | `pip install faiss-cpu` |
| LangChain errors | `pip install langchain-openai langchain-community langchain-core langchain-text-splitters` |

---

## 📄 Documentation

The full technical report, covering architecture, implementation, and evaluation, is in [`docs/project_report.pdf`](docs/project_report.pdf).

---

## 👤 Author

**Lok Venkatesh Vasamsetti**
AI/ML Engineer · MS, Northeastern University

- GitHub: [@lokvenkatesh](https://github.com/lokvenkatesh)
- LinkedIn: [lok-venkatesh-vasamsetti](https://www.linkedin.com/in/lok-venkatesh-vasamsetti)
- Email: lokvenkatesh7@gmail.com

---

## 🙏 Acknowledgments

OpenAI (GPT-3.5 and fine-tuning API), Meta AI (FAISS), Hugging Face (Sentence-Transformers), LangChain, Streamlit, and Plotly.

## 📝 License

MIT. See [LICENSE](LICENSE).
