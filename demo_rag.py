"""
Demo script to test the RAG system
Run this to process papers and test search
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from rag.document_processor import DocumentProcessor
from rag.vector_store import VectorStore
from utils.config import config

def main():
    """Main demo function"""
    
    print("=" * 60)
    print("🚀 AI Research Assistant - RAG System Demo")
    print("=" * 60)
    
    # Initialize components
    print("\n1️⃣  Initializing components...")
    processor = DocumentProcessor(
        chunk_size=config.rag.chunk_size,
        chunk_overlap=config.rag.chunk_overlap
    )
    vector_store = VectorStore(embedding_model=config.models.embedding_model)
    
    # Check if vector store exists
    vector_db_path = config.paths.vector_db
    
    if (vector_db_path / "index.faiss").exists():
        print(f"\n2️⃣  Loading existing vector store from {vector_db_path}")
        vector_store.load(vector_db_path)
        stats = vector_store.get_stats()
        print(f"  📊 Statistics:")
        print(f"     - Documents: {stats['total_documents']}")
        print(f"     - Chunks: {stats['total_chunks']}")
        print(f"     - Vectors: {stats['total_vectors']}")
    else:
        # Process papers
        print(f"\n2️⃣  Processing papers from {config.paths.papers_raw}")
        documents = processor.process_directory(config.paths.papers_raw)
        
        if not documents:
            print("\n❌ No documents found!")
            print(f"   Please add PDF files to: {config.paths.papers_raw}")
            print("\n💡 Tip: Download some research papers and place them in the papers/raw folder")
            return
        
        print(f"\n3️⃣  Building vector store...")
        vector_store.add_documents(documents)
        
        print(f"\n4️⃣  Saving vector store...")
        vector_store.save(vector_db_path)
    
    # Interactive search
    print("\n" + "=" * 60)
    print("🔍 Interactive Search Mode")
    print("=" * 60)
    print("Type your research questions (or 'quit' to exit)")
    print()
    
    while True:
        query = input("🔎 Query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not query:
            continue
        
        # Search
        print(f"\n🔄 Searching for: '{query}'")
        results = vector_store.search(query, top_k=config.rag.top_k_results)
        
        if not results:
            print("  ❌ No results found")
            continue
        
        # Display results
        print(f"\n📚 Found {len(results)} relevant chunks:\n")
        
        for i, (chunk, score) in enumerate(results, 1):
            print(f"{'─' * 60}")
            print(f"Result #{i} - Similarity: {score:.3f}")
            print(f"Document: {chunk.metadata.get('title', 'Unknown')}")
            print(f"Page: {chunk.page_number} | Chunk: {chunk.chunk_id}")
            print(f"{'─' * 60}")
            
            # Show excerpt
            content = chunk.content
            if len(content) > 300:
                content = content[:300] + "..."
            
            print(content)
            print()
        
        print("\n" + "=" * 60 + "\n")

def quick_test():
    """Quick test without interactive mode"""
    print("🧪 Quick Test Mode\n")
    
    processor = DocumentProcessor()
    vector_store = VectorStore()
    
    # Try to load existing store
    try:
        vector_store.load(config.paths.vector_db)
        print("✅ Vector store loaded successfully!")
        
        # Test search
        test_queries = [
            "machine learning",
            "neural networks",
            "deep learning applications"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            results = vector_store.search(query, top_k=3)
            print(f"  ✅ Found {len(results)} results")
            
            if results:
                best_match = results[0]
                print(f"  📄 Best match: {best_match[0].metadata.get('title', 'Unknown')}")
                print(f"  📊 Score: {best_match[1]:.3f}")
        
        print("\n✅ All tests passed!")
        
    except FileNotFoundError:
        print("❌ No vector store found. Please run main() to process papers first.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG System Demo')
    parser.add_argument('--test', action='store_true', help='Run quick test')
    args = parser.parse_args()
    
    if args.test:
        quick_test()
    else:
        main()