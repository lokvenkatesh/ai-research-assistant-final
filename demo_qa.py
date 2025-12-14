"""
Interactive Q&A demo for research assistant
Test the complete RAG + LLM pipeline
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.rag.retriever import RAGRetriever
from src.utils.config import config

def print_header():
    """Print demo header"""
    print("=" * 70)
    print("🎓 AI Research Assistant - Interactive Q&A")
    print("=" * 70)
    print(f"Model: {config.models.llm_model}")
    print("=" * 70)
    print()

def print_result(result: dict):
    """Print formatted result"""
    print(f"\n{'─'*70}")
    print("📝 ANSWER:")
    print(f"{'─'*70}")
    print(result['answer'])
    
    if result.get('citations'):
        print(f"\n{'─'*70}")
        print("📚 CITATIONS:")
        print(f"{'─'*70}")
        for i, citation in enumerate(result['citations'], 1):
            print(f"{i}. [{citation['citation_text']}]")
            print(f"   Title: {citation['title']}")
            print(f"   Page: {citation['page']}")
    
    print(f"\n{'─'*70}")
    print("📖 SOURCES:")
    print(f"{'─'*70}")
    for i, source in enumerate(result['sources'], 1):
        title = source['metadata'].get('title', 'Unknown')
        author = source['metadata'].get('author', 'Unknown')
        page = source['page_number']
        score = source['similarity_score']
        print(f"{i}. {title}")
        print(f"   Author: {author} | Page: {page} | Score: {score:.3f}")
    
    print(f"\n{'='*70}\n")

def interactive_mode(retriever: RAGRetriever):
    """Run interactive Q&A session"""
    print("\n💬 Interactive Mode")
    print("Commands:")
    print("  - Type your question")
    print("  - 'history' - Show conversation history")
    print("  - 'clear' - Clear conversation history")
    print("  - 'summarize [paper_title]' - Summarize a paper")
    print("  - 'quit' or 'q' - Exit")
    print()
    
    while True:
        try:
            user_input = input("❓ You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'q', 'exit']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                retriever.clear_history()
                continue
            
            if user_input.lower() == 'history':
                if not retriever.conversation_history:
                    print("  No conversation history yet")
                else:
                    print(f"\n📜 Conversation History ({len(retriever.conversation_history)} turns):")
                    for i, turn in enumerate(retriever.conversation_history, 1):
                        print(f"\n{i}. Q: {turn['question']}")
                        print(f"   A: {turn['answer'][:150]}...")
                continue
            
            if user_input.lower().startswith('summarize'):
                paper_title = user_input[9:].strip()
                if not paper_title:
                    print("  ❌ Please provide a paper title")
                    continue
                
                print(f"\n📄 Summarizing: {paper_title}")
                summary = retriever.summarize_paper(paper_title)
                print(f"\n{summary}\n")
                continue
            
            # Regular question
            result = retriever.answer_question(
                question=user_input,
                use_reformulation=True,
                use_chain_of_thought=False
            )
            
            print_result(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")

def demo_mode(retriever: RAGRetriever):
    """Run demo with preset questions"""
    demo_questions = [
        "What is machine learning?",
        "How do neural networks work?",
        "What are the applications of AI?",
        "Explain transformers in simple terms"
    ]
    
    print("\n🎬 Demo Mode - Running preset questions...\n")
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n{'='*70}")
        print(f"Demo Question {i}/{len(demo_questions)}")
        print(f"{'='*70}")
        
        result = retriever.answer_question(
            question=question,
            use_reformulation=True
        )
        
        print_result(result)
        
        if i < len(demo_questions):
            input("Press Enter for next question...")

def quick_test(retriever: RAGRetriever):
    """Quick single question test"""
    test_question = "What are the main benefits of transfer learning?"
    
    print(f"\n🧪 Quick Test")
    print(f"Question: {test_question}\n")
    
    result = retriever.answer_question(
        question=test_question,
        use_reformulation=True
    )
    
    print_result(result)

def main():
    """Main demo function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Research Assistant Q&A Demo')
    parser.add_argument('--mode', choices=['interactive', 'demo', 'test'], 
                       default='interactive',
                       help='Demo mode')
    parser.add_argument('--no-finetune', action='store_true',
                       help='Use base model instead of fine-tuned')
    parser.add_argument('--question', type=str,
                       help='Single question to ask')
    
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # Initialize retriever
    print("🔧 Initializing system...")
    retriever = RAGRetriever(use_fine_tuned=not args.no_finetune)
    
    # Check if vector store has data
    if retriever.vector_store.index.ntotal == 0:
        print("\n❌ No papers found in vector store!")
        print("   Please add papers first using: python demo_rag.py")
        return
    
    print(f"✅ Loaded {retriever.vector_store.index.ntotal} document chunks")
    print()
    
    # Single question mode
    if args.question:
        result = retriever.answer_question(args.question)
        print_result(result)
        return
    
    # Run selected mode
    if args.mode == 'interactive':
        interactive_mode(retriever)
    elif args.mode == 'demo':
        demo_mode(retriever)
    elif args.mode == 'test':
        quick_test(retriever)

if __name__ == "__main__":
    main()