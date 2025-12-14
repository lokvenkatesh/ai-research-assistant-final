"""
LangChain-based RAG Implementation
Alternative implementation using LangChain framework
"""

import sys
from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import config

class LangChainRAG:
    """
    RAG implementation using LangChain framework
    Alternative to custom implementation for comparison
    """
    
    def __init__(self, openai_api_key: str = None):
        """Initialize LangChain RAG system"""
        self.api_key = openai_api_key or config.models.openai_api_key
        
        # Initialize embeddings
        print("🔧 Initializing LangChain components...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.models.embedding_model
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.models.llm_model,
            api_key=self.api_key,
            temperature=0.7
        )
        
        # Vector store will be initialized after loading documents
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        
        print("✅ LangChain components initialized")
    
    def load_documents_from_directory(self, directory: Path) -> List[Document]:
        """
        Load PDFs using LangChain's PyPDFLoader
        
        Args:
            directory: Path to directory containing PDFs
            
        Returns:
            List of LangChain Document objects
        """
        print(f"📚 Loading documents from {directory}")
        
        pdf_files = list(directory.glob("*.pdf"))
        all_documents = []
        
        for pdf_path in pdf_files:
            print(f"  Loading: {pdf_path.name}")
            try:
                loader = PyPDFLoader(str(pdf_path))
                documents = loader.load()
                all_documents.extend(documents)
            except Exception as e:
                print(f"  ⚠️ Error loading {pdf_path.name}: {str(e)}")
        
        print(f"✅ Loaded {len(all_documents)} pages from {len(pdf_files)} papers")
        return all_documents
    
    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks using LangChain's text splitter
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        print("🔪 Splitting documents into chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        print(f"✅ Created {len(chunks)} chunks")
        return chunks
    
    def create_vectorstore(self, documents: List[Document]):
        """
        Create FAISS vector store from documents
        
        Args:
            documents: List of chunked Document objects
        """
        print("🔄 Creating vector store...")
        
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.rag.top_k_results}
        )
        
        print("✅ Vector store created")
    
    def setup_rag_chain(self):
        """Setup the RAG chain using LCEL"""
        print("⚙️ Setting up RAG chain...")
        
        # Custom prompt template
        template = """You are an expert research assistant. Use the following context from research papers to answer the question.

Context:
{context}

Question: {question}

Provide a comprehensive answer with citations in [Author, Year] format. If the context doesn't contain enough information, acknowledge the limitation.

Answer:"""

        prompt = PromptTemplate.from_template(template)
        
        def format_docs(docs):
            """Format documents for context"""
            return "\n\n".join([
                f"[Source {i+1}] {doc.page_content}" 
                for i, doc in enumerate(docs)
            ])
        
        # Create RAG chain using LCEL
        self.rag_chain = (
            {
                "context": self.retriever | format_docs, 
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("✅ RAG chain ready")
    
    def process_papers(self, papers_dir: Path):
        """
        Complete pipeline: load, chunk, embed, index
        
        Args:
            papers_dir: Directory containing PDF papers
        """
        # Load documents
        documents = self.load_documents_from_directory(papers_dir)
        
        if not documents:
            print("❌ No documents loaded!")
            return
        
        # Create chunks
        chunks = self.create_chunks(documents)
        
        # Create vector store
        self.create_vectorstore(chunks)
        
        # Setup RAG chain
        self.setup_rag_chain()
        
        print("🎉 LangChain RAG pipeline ready!")
    
    def ask(self, question: str) -> Dict:
        """
        Ask a question using LangChain
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and sources
        """
        if not self.rag_chain:
            raise ValueError("RAG chain not initialized. Run process_papers() first.")
        
        print(f"\n❓ Question: {question}")
        
        # Get relevant documents using invoke (updated method)
        source_docs = self.retriever.invoke(question)
        
        # Generate answer using the chain
        answer = self.rag_chain.invoke(question)
        
        print(f"✅ Answer generated")
        
        return {
            'answer': answer,
            'sources': source_docs,
            'num_sources': len(source_docs)
        }
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Direct similarity search
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of relevant documents
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized.")
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def save_vectorstore(self, save_path: Path):
        """Save vector store to disk"""
        if self.vectorstore:
            save_path.mkdir(parents=True, exist_ok=True)
            self.vectorstore.save_local(str(save_path))
            print(f"💾 Vector store saved to {save_path}")
    
    def load_vectorstore(self, load_path: Path):
        """Load vector store from disk"""
        self.vectorstore = FAISS.load_local(
            str(load_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.rag.top_k_results}
        )
        
        self.setup_rag_chain()
        
        print(f"📂 LangChain vector store loaded from {load_path}")

def demo_langchain():
    """Demo LangChain RAG implementation"""
    print("=" * 70)
    print("🦜 LangChain RAG Implementation Demo")
    print("=" * 70)
    
    # Initialize
    langchain_rag = LangChainRAG()
    
    # Process papers
    papers_dir = config.paths.papers_raw
    
    if not papers_dir.exists() or not list(papers_dir.glob("*.pdf")):
        print(f"❌ No PDF files found in {papers_dir}")
        print("   Please add papers to data/papers/raw/")
        return
    
    langchain_rag.process_papers(papers_dir)
    
    # Ask questions
    test_questions = [
        "What is transfer learning?",
        "How do diffusion models work?"
    ]
    
    for question in test_questions:
        try:
            result = langchain_rag.ask(question)
            
            print(f"\n{'='*70}")
            print(f"Question: {question}")
            print(f"{'='*70}")
            print(f"Answer: {result['answer'][:400]}...")
            print(f"\nSources: {result['num_sources']} documents")
            for i, doc in enumerate(result['sources'][:2], 1):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', '?')
                print(f"  {i}. {Path(source).name} (Page {page})")
            print("=" * 70)
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    demo_langchain()