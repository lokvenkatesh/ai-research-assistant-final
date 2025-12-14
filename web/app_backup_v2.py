# """
# Streamlit Web Interface for AI Research Assistant
# Run with: streamlit run web/app.py
# """

# import streamlit as st
# import sys
# from pathlib import Path

# # Add src to path
# # Add src to path
# import os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# from rag.document_processor import DocumentProcessor
# from rag.vector_store import VectorStore
# from utils.config import config

# # Page config
# st.set_page_config(
#     page_title="AI Research Assistant",
#     page_icon="📚",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         font-weight: bold;
#         text-align: center;
#         margin-bottom: 1rem;
#     }
#     .result-box {
#         background-color: #f0f2f6;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin-bottom: 1rem;
#     }
#     .metric-box {
#         background-color: #e6f3ff;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         text-align: center;
#     }
# </style>
# """, unsafe_allow_html=True)

# @st.cache_resource
# def load_components():
#     """Load RAG components (cached)"""
#     processor = DocumentProcessor(
#         chunk_size=config.rag.chunk_size,
#         chunk_overlap=config.rag.chunk_overlap
#     )
#     vector_store = VectorStore(embedding_model=config.models.embedding_model)
    
#     # Try to load existing vector store
#     try:
#         vector_store.load(config.paths.vector_db)
#         status = "loaded"
#     except FileNotFoundError:
#         status = "empty"
    
#     return processor, vector_store, status

# def main():
#     """Main app function"""
    
#     # Header
#     st.markdown('<div class="main-header">📚 AI Research Assistant</div>', unsafe_allow_html=True)
#     st.markdown("### Powered by RAG, Fine-tuning, Prompt Engineering & More")
    
#     # Load components
#     processor, vector_store, status = load_components()
    
#     # Sidebar
#     with st.sidebar:
#         st.header("⚙️ Settings")
        
#         # Vector store status
#         if status == "loaded":
#             stats = vector_store.get_stats()
#             st.success("✅ Vector Store Loaded")
            
#             st.markdown("### 📊 Statistics")
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.metric("Documents", stats['total_documents'])
#             with col2:
#                 st.metric("Chunks", stats['total_chunks'])
#         else:
#             st.warning("⚠️ Vector Store Empty")
#             st.info("Upload papers to get started!")
        
#         st.divider()
        
#         # Search settings
#         st.markdown("### 🔍 Search Settings")
#         top_k = st.slider("Results to show", 1, 10, config.rag.top_k_results)
        
#         st.divider()
        
#         # About
#         st.markdown("### ℹ️ About")
#         st.markdown("""
#         This is an advanced research assistant featuring:
#         - 📚 RAG (Retrieval-Augmented Generation)
#         - 🎯 Prompt Engineering
#         - 🔧 Fine-tuned Models
#         - 🖼️ Multimodal Processing
#         - 🔄 Synthetic Data Generation
#         """)
    
#     # Main tabs
#     tab1, tab2, tab3 = st.tabs(["🔍 Search", "📤 Upload Papers", "📊 Analytics"])
    
#     with tab1:
#         st.header("Search Research Papers")
        
#         # Search interface
#         query = st.text_input(
#             "Enter your research question:",
#             placeholder="E.g., What are the latest developments in transformer models?"
#         )
        
#         col1, col2, col3 = st.columns([1, 1, 3])
#         with col1:
#             search_button = st.button("🔎 Search", type="primary", use_container_width=True)
#         with col2:
#             if st.button("🔄 Clear", use_container_width=True):
#                 st.rerun()
        
#         if search_button and query:
#             if status == "empty":
#                 st.error("❌ Please upload papers first!")
#             else:
#                 with st.spinner("🔄 Searching..."):
#                     results = vector_store.search(query, top_k=top_k)
                
#                 if not results:
#                     st.warning("No results found. Try a different query.")
#                 else:
#                     st.success(f"✅ Found {len(results)} relevant results")
                    
#                     # Display results
#                     for i, (chunk, score) in enumerate(results, 1):
#                         with st.expander(
#                             f"📄 Result #{i} - {chunk.metadata.get('title', 'Unknown')} (Score: {score:.3f})",
#                             expanded=(i == 1)
#                         ):
#                             # Metadata
#                             col1, col2, col3 = st.columns(3)
#                             with col1:
#                                 st.markdown(f"**Page:** {chunk.page_number}")
#                             with col2:
#                                 st.markdown(f"**Chunk:** {chunk.metadata.get('chunk_number', 0)}")
#                             with col3:
#                                 st.markdown(f"**Similarity:** {score:.3f}")
                            
#                             st.divider()
                            
#                             # Content
#                             st.markdown("**Content:**")
#                             st.markdown(f'<div class="result-box">{chunk.content}</div>', 
#                                       unsafe_allow_html=True)
                            
#                             # Additional metadata
#                             if chunk.metadata.get('author'):
#                                 st.caption(f"Author: {chunk.metadata['author']}")
    
#     with tab2:
#         st.header("Upload Research Papers")
        
#         st.info("📝 Upload PDF research papers to build your knowledge base")
        
#         uploaded_files = st.file_uploader(
#             "Choose PDF files",
#             type=['pdf'],
#             accept_multiple_files=True
#         )
        
#         if uploaded_files:
#             st.write(f"📁 {len(uploaded_files)} file(s) selected")
            
#             if st.button("🚀 Process Papers", type="primary"):
#                 progress_bar = st.progress(0)
#                 status_text = st.empty()
                
#                 documents = []
                
#                 for i, uploaded_file in enumerate(uploaded_files):
#                     # Save temporarily
#                     temp_path = config.paths.papers_raw / uploaded_file.name
#                     with open(temp_path, 'wb') as f:
#                         f.write(uploaded_file.getbuffer())
                    
#                     # Process
#                     status_text.text(f"Processing {uploaded_file.name}...")
#                     try:
#                         doc = processor.process_document(temp_path)
#                         documents.append(doc)
#                         st.success(f"✅ {uploaded_file.name}")
#                     except Exception as e:
#                         st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    
#                     progress_bar.progress((i + 1) / len(uploaded_files))
                
#                 if documents:
#                     # Add to vector store
#                     status_text.text("Building vector store...")
#                     vector_store.add_documents(documents)
#                     vector_store.save(config.paths.vector_db)
                    
#                     st.success(f"🎉 Successfully processed {len(documents)} papers!")
#                     st.balloons()
                    
#                     # Clear cache to reload
#                     st.cache_resource.clear()
#                     st.rerun()
    
#     with tab3:
#         st.header("Knowledge Base Analytics")
        
#         if status == "empty":
#             st.info("📊 Upload papers to see analytics")
#         else:
#             stats = vector_store.get_stats()
            
#             # Metrics
#             col1, col2, col3, col4 = st.columns(4)
            
#             with col1:
#                 st.markdown('<div class="metric-box">', unsafe_allow_html=True)
#                 st.metric("Total Documents", stats['total_documents'])
#                 st.markdown('</div>', unsafe_allow_html=True)
            
#             with col2:
#                 st.markdown('<div class="metric-box">', unsafe_allow_html=True)
#                 st.metric("Total Chunks", stats['total_chunks'])
#                 st.markdown('</div>', unsafe_allow_html=True)
            
#             with col3:
#                 st.markdown('<div class="metric-box">', unsafe_allow_html=True)
#                 st.metric("Vector Dimension", stats['embedding_dimension'])
#                 st.markdown('</div>', unsafe_allow_html=True)
            
#             with col4:
#                 st.markdown('<div class="metric-box">', unsafe_allow_html=True)
#                 avg_chunks = stats['total_chunks'] / max(stats['total_documents'], 1)
#                 st.metric("Avg Chunks/Doc", f"{avg_chunks:.1f}")
#                 st.markdown('</div>', unsafe_allow_html=True)
            
#             st.divider()
            
#             # Document list
#             st.subheader("📚 Indexed Documents")
            
#             doc_data = []
#             for doc_id, metadata in vector_store.document_map.items():
#                 doc_data.append({
#                     'Title': metadata.get('title', 'Unknown'),
#                     'Filename': metadata.get('filename', 'Unknown'),
#                     'Pages': metadata.get('num_pages', 0),
#                     'Author': metadata.get('author', 'Unknown')
#                 })
            
#             if doc_data:
#                 st.dataframe(doc_data, use_container_width=True)

# if __name__ == "__main__":
#     main()

"""
Complete Streamlit Web Interface for AI Research Assistant
Includes RAG Search + Q&A System
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rag.document_processor import DocumentProcessor
from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever
from utils.config import config

# Page config
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
        color: #000000;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #2ecc71;
        color: #000000;
    }
    .source-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #ffc107;
        color: #000000;
    }
    .metric-box {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .citation-badge {
        background-color: #6c757d;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 0.3rem;
        font-size: 0.9rem;
        margin-right: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_components():
    """Load RAG components (cached)"""
    processor = DocumentProcessor(
        chunk_size=config.rag.chunk_size,
        chunk_overlap=config.rag.chunk_overlap
    )
    vector_store = VectorStore(embedding_model=config.models.embedding_model)
    
    # Try to load existing vector store
    try:
        vector_store.load(config.paths.vector_db)
        status = "loaded"
    except FileNotFoundError:
        status = "empty"
    
    return processor, vector_store, status

def get_retriever(vector_store, use_fine_tuned=True):
    """Get retriever with selected model"""
    return RAGRetriever(vector_store=vector_store, use_fine_tuned=use_fine_tuned)

def main():
    """Main app function"""
    
    # Header
    st.markdown('<div class="main-header">📚 AI Research Assistant</div>', unsafe_allow_html=True)
    st.markdown("### Complete RAG + Fine-Tuned Q&A System")
    
    # Load components
    processor, vector_store, status = load_components()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # Vector store status
        if status == "loaded":
            stats = vector_store.get_stats()
            st.success("✅ System Ready")
            
            st.markdown("### 📊 Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats['total_documents'])
                st.metric("Chunks", stats['total_chunks'])
            with col2:
                st.metric("Vectors", stats['total_vectors'])
                st.metric("Dimension", stats['embedding_dimension'])
        else:
            st.warning("⚠️ No Data")
            st.info("Upload papers to get started!")
        
        st.divider()
        
        # Model Selection
        st.markdown("### 🤖 Model Selection")
        model_choice = st.radio(
            "Choose Model:",
            ["Fine-Tuned Model ✨", "Base GPT-3.5"],
            index=0,
            help="Fine-tuned model is trained on your research papers"
        )
        use_fine_tuned = (model_choice == "Fine-Tuned Model ✨")
        
        if use_fine_tuned:
            st.success("Using custom fine-tuned model")
            st.caption("Specialized for research papers")
        else:
            st.info("Using base GPT-3.5-turbo")
            st.caption("General purpose model")
        
        st.divider()
        
        # Settings
        st.markdown("### 🔧 Settings")
        
        if status == "loaded":
            top_k = st.slider("Results to retrieve", 1, 10, 5)
            use_reformulation = st.checkbox("Smart query reformulation", value=True)
            use_cot = st.checkbox("Chain-of-thought reasoning", value=False)
        else:
            top_k = 5
            use_reformulation = True
            use_cot = False
        
        st.divider()
        
        # Model info
        st.markdown("### 📋 Current Model")
        if use_fine_tuned:
            st.code(config.models.llm_model, language="text")
        else:
            st.code("gpt-3.5-turbo", language="text")
        
        st.divider()
        
        # About
        st.markdown("### ℹ️ Features")
        st.markdown("""
        ✅ RAG Search  
        ✅ Fine-Tuned Model  
        ✅ Query Reformulation  
        ✅ Citation Tracking  
        ✅ Multi-turn Q&A  
        ✅ Smart Prompting
        """)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Ask Questions", "🔍 Search Papers", "📤 Upload Papers", "📊 Analytics"])
    
    # ============================================================
    # TAB 1: Q&A SYSTEM (NEW!)
    # ============================================================
    with tab1:
        st.header("Ask Research Questions")
        
        if status == "empty":
            st.warning("⚠️ Please upload papers first!")
        else:
            # Initialize retriever with selected model
            retriever = get_retriever(vector_store, use_fine_tuned=use_fine_tuned)
            
            st.markdown(f"""
            Ask questions about your research papers. The system will:
            - 🔍 Search for relevant context
            - 🤖 Generate comprehensive answers using **{model_choice}**
            - 📚 Provide citations and sources
            """)
            
            # Initialize session state for conversation
            if 'conversation_history' not in st.session_state:
                st.session_state.conversation_history = []
            
            # Question input
            question = st.text_input(
                "Your Question:",
                placeholder="e.g., What are the main benefits of transfer learning?",
                key="question_input"
            )
            
            col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
            with col1:
                ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)
            with col2:
                clear_button = st.button("🔄 Clear", use_container_width=True)
            with col3:
                show_history = st.checkbox("Show History", value=False)
            
            if clear_button:
                st.session_state.conversation_history = []
                retriever.clear_history()
                st.success("✅ History cleared!")
                st.rerun()
            
            if ask_button and question:
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Get answer
                        result = retriever.answer_question(
                            question=question,
                            top_k=top_k,
                            use_reformulation=use_reformulation,
                            use_chain_of_thought=use_cot
                        )
                        
                        # Add to session history
                        st.session_state.conversation_history.append({
                            'question': question,
                            'answer': result['answer'],
                            'sources': result['sources'],
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        })
                        
                        # Display answer
                        st.markdown("### 💡 Answer")
                        
                        # Show which model was used
                        model_badge = "Fine-Tuned ✨" if use_fine_tuned else "Base GPT-3.5"
                        st.caption(f"Generated by: **{model_badge}**")
                        
                        st.markdown(f'<div class="answer-box">{result["answer"]}</div>', 
                                  unsafe_allow_html=True)
                        
                        # Display sources
                        st.markdown("### 📚 Sources")
                        for i, source in enumerate(result['sources'], 1):
                            with st.expander(
                                f"📄 Source {i}: {source['metadata'].get('title', 'Unknown')} (Similarity: {source['similarity_score']:.3f})",
                                expanded=(i == 1)
                            ):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.markdown(f"**Author:** {source['metadata'].get('author', 'Unknown')}")
                                with col2:
                                    st.markdown(f"**Page:** {source['page_number']}")
                                with col3:
                                    st.markdown(f"**Score:** {source['similarity_score']:.3f}")
                                
                                st.markdown("**Content:**")
                                st.markdown(f'<div class="source-box">{source["content"][:500]}...</div>', 
                                          unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            # Show conversation history
            if show_history and st.session_state.conversation_history:
                st.divider()
                st.markdown("### 📜 Conversation History")
                
                for i, turn in enumerate(reversed(st.session_state.conversation_history), 1):
                    with st.expander(f"Q{len(st.session_state.conversation_history) - i + 1}: {turn['question'][:60]}... ({turn['timestamp']})"):
                        st.markdown(f"**Q:** {turn['question']}")
                        st.markdown(f"**A:** {turn['answer'][:300]}...")
                        st.markdown(f"**Sources:** {len(turn['sources'])} papers")
    
    # ============================================================
    # TAB 2: SEARCH (EXISTING - Enhanced)
    # ============================================================
    with tab2:
        st.header("Search Research Papers")
        st.markdown("Find specific passages in your papers using semantic search.")
        
        # Search interface
        query = st.text_input(
            "Search query:",
            placeholder="E.g., transformer architecture, attention mechanism"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            search_button = st.button("🔎 Search", type="primary", use_container_width=True)
        
        if search_button and query:
            if status == "empty":
                st.error("❌ Please upload papers first!")
            else:
                with st.spinner("🔄 Searching..."):
                    results = vector_store.search(query, top_k=top_k)
                
                if not results:
                    st.warning("No results found. Try a different query.")
                else:
                    st.success(f"✅ Found {len(results)} relevant results")
                    
                    # Display results
                    for i, (chunk, score) in enumerate(results, 1):
                        with st.expander(
                            f"📄 Result #{i} - {chunk.metadata.get('title', 'Unknown')} (Score: {score:.3f})",
                            expanded=(i == 1)
                        ):
                            # Metadata
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"**Page:** {chunk.page_number}")
                            with col2:
                                st.markdown(f"**Chunk:** {chunk.metadata.get('chunk_number', 0)}")
                            with col3:
                                st.markdown(f"**Similarity:** {score:.3f}")
                            
                            st.divider()
                            
                            # Content
                            st.markdown("**Content:**")
                            st.markdown(f'<div class="result-box">{chunk.content}</div>', 
                                      unsafe_allow_html=True)
                            
                            # Additional metadata
                            if chunk.metadata.get('author'):
                                st.caption(f"Author: {chunk.metadata['author']}")
    
    # ============================================================
    # TAB 3: UPLOAD (EXISTING)
    # ============================================================
    with tab3:
        st.header("Upload Research Papers")
        
        st.info("📝 Upload PDF research papers to build your knowledge base")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"📁 {len(uploaded_files)} file(s) selected")
            
            if st.button("🚀 Process Papers", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                documents = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    # Save temporarily
                    temp_path = config.paths.papers_raw / uploaded_file.name
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process
                    status_text.text(f"Processing {uploaded_file.name}...")
                    try:
                        doc = processor.process_document(temp_path)
                        documents.append(doc)
                        st.success(f"✅ {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                if documents:
                    # Add to vector store
                    status_text.text("Building vector store...")
                    vector_store.add_documents(documents)
                    vector_store.save(config.paths.vector_db)
                    
                    st.success(f"🎉 Successfully processed {len(documents)} papers!")
                    st.balloons()
                    
                    # Clear cache to reload
                    st.cache_resource.clear()
                    st.rerun()
    
    # ============================================================
    # TAB 4: ANALYTICS (EXISTING - Enhanced)
    # ============================================================
    with tab4:
        st.header("Knowledge Base Analytics")
        
        if status == "empty":
            st.info("📊 Upload papers to see analytics")
        else:
            stats = vector_store.get_stats()
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Total Documents", stats['total_documents'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Total Chunks", stats['total_chunks'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Vector Dimension", stats['embedding_dimension'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                avg_chunks = stats['total_chunks'] / max(stats['total_documents'], 1)
                st.metric("Avg Chunks/Doc", f"{avg_chunks:.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.divider()
            
            # System capabilities
            st.subheader("🚀 System Capabilities")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Implemented Features:**")
                st.markdown("""
                - ✅ Semantic search across papers
                - ✅ Fine-tuned model (GPT-3.5)
                - ✅ Query reformulation
                - ✅ Citation tracking
                - ✅ Multi-turn conversations
                - ✅ Chain-of-thought reasoning
                - ✅ Source attribution
                """)
            
            with col2:
                st.markdown("**Technologies Used:**")
                st.markdown(f"""
                - 🔧 RAG (Retrieval-Augmented Generation)
                - 🎯 Prompt Engineering
                - 🔬 Fine-Tuning (OpenAI)
                - 📊 FAISS Vector Database
                - 🤖 Sentence Transformers
                - 🌐 Streamlit Web Interface
                
                **Active Model:**
                - {model_choice}
                """)
            
            st.divider()
            
            # Document list
            st.subheader("📚 Indexed Documents")
            
            doc_data = []
            for doc_id, metadata in vector_store.document_map.items():
                doc_data.append({
                    'Title': metadata.get('title', 'Unknown'),
                    'Filename': metadata.get('filename', 'Unknown'),
                    'Pages': metadata.get('num_pages', 0),
                    'Author': metadata.get('author', 'Unknown')
                })
            
            if doc_data:
                st.dataframe(doc_data, use_container_width=True)

if __name__ == "__main__":
    main()