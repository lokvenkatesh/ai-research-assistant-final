"""
Complete Streamlit Web Interface for AI Research Assistant
Includes RAG Search + Q&A System + LangChain Integration
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rag.document_processor import DocumentProcessor
from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever
from utils.config import config
from utils.database import ConversationDB
import uuid

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
        padding: 1.5rem;
        border-radius: 0.8rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #d1e7f5;
    }
    .insight-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #6c757d;
    }
    .langchain-badge {
        background-color: #10b981;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 0.4rem;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .custom-badge {
        background-color: #3b82f6;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 0.4rem;
        font-size: 0.85rem;
        font-weight: bold;
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

@st.cache_resource
def init_database():
    """Initialize database connection (cached)"""
    return ConversationDB()

def get_retriever(vector_store, use_fine_tuned=True):
    """Get retriever with selected model"""
    return RAGRetriever(vector_store=vector_store, use_fine_tuned=use_fine_tuned)

def get_session_id():
    """Get or create session ID"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def main():
    """Main app function"""
    
    # Header
    st.markdown('<div class="main-header">📚 AI Research Assistant</div>', unsafe_allow_html=True)
    st.markdown("### Complete RAG + Fine-Tuned Q&A System with LangChain")
    
    # Load components
    processor, vector_store, status = load_components()
    db = init_database()
    session_id = get_session_id()
    
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
        
        # RAG Implementation Selection
        st.markdown("### 🔧 RAG Backend")
        rag_choice = st.radio(
            "Choose Implementation:",
            ["Custom RAG 🏗️", "LangChain RAG 🦜"],
            index=0,
            help="Compare custom implementation vs LangChain framework"
        )
        use_langchain = (rag_choice == "LangChain RAG 🦜")
        
        if use_langchain:
            st.success("Using LangChain framework")
            st.caption("High-level abstractions, faster development")
        else:
            st.info("Using custom implementation")
            st.caption("Full control, deeper understanding")
        
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
        st.markdown("### 📋 Current Configuration")
        config_display = f"""
**RAG:** {rag_choice}
**Model:** {model_choice}
**Top-K:** {top_k}
**Reformulation:** {"On" if use_reformulation else "Off"}
"""
        st.markdown(config_display)
        
        st.divider()
        
        # About
        st.markdown("### ℹ️ Features")
        st.markdown("""
        ✅ Custom RAG Implementation
        ✅ LangChain RAG Framework
        ✅ Fine-Tuned Model  
        ✅ Query Reformulation  
        ✅ Citation Tracking  
        ✅ Multi-turn Q&A  
        ✅ Smart Prompting
        """)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💬 Ask Questions", 
        "🔍 Search Papers", 
        "📜 History", 
        "📤 Upload Papers", 
        "📊 Analytics",
        "🖼️ Multimodal"
    ])
    
    # ============================================================
    # TAB 1: Q&A SYSTEM with LangChain Option
    # ============================================================
    with tab1:
        st.header("Ask Research Questions")
        
        if status == "empty":
            st.warning("⚠️ Please upload papers first!")
        else:
            # Display current configuration
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"""
                Ask questions about your research papers. The system will:
                - 🔍 Search for relevant context using **{rag_choice}**
                - 🤖 Generate comprehensive answers using **{model_choice}**
                - 📚 Provide citations and sources
                """)
            with col2:
                if use_langchain:
                    st.markdown('<span class="langchain-badge">🦜 LangChain</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="custom-badge">🏗️ Custom</span>', unsafe_allow_html=True)
            
            # Initialize retriever based on RAG choice
            if use_langchain:
                # Try to import and initialize LangChain
                try:
                    from src.rag.langchain_retriever import LangChainRAG
                    
                    # Initialize or load LangChain RAG
                    if 'langchain_rag' not in st.session_state:
                        with st.spinner("🦜 Initializing LangChain RAG..."):
                            st.session_state.langchain_rag = LangChainRAG()
                            # Check if vectorstore exists
                            lc_vectorstore_path = config.paths.vector_db / "langchain"
                            if lc_vectorstore_path.exists():
                                st.session_state.langchain_rag.load_vectorstore(lc_vectorstore_path)
                                st.success("✅ Loaded LangChain vector store")
                            else:
                                # Create from papers
                                st.session_state.langchain_rag.process_papers(config.paths.papers_raw)
                                st.session_state.langchain_rag.save_vectorstore(lc_vectorstore_path)
                                st.success("✅ Created LangChain vector store")
                    
                    langchain_rag = st.session_state.langchain_rag
                    
                except Exception as e:
                    st.error(f"⚠️ LangChain error: {str(e)}")
                    st.info("Falling back to custom RAG...")
                    use_langchain = False
                    retriever = get_retriever(vector_store, use_fine_tuned=use_fine_tuned)
            
            if not use_langchain:
                # Use custom retriever
                retriever = get_retriever(vector_store, use_fine_tuned=use_fine_tuned)
            
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
                if not use_langchain:
                    retriever.clear_history()
                st.success("✅ History cleared!")
                st.rerun()
            
            if ask_button and question:
                with st.spinner("🤔 Thinking..."):
                    try:
                        import time
                        start_time = time.time()
                        
                        # Get answer based on RAG choice
                        if use_langchain:
                            # Use LangChain implementation
                            lc_result = langchain_rag.ask(question)
                            
                            # Convert to standard format
                            result = {
                                'answer': lc_result['answer'],
                                'sources': [
                                    {
                                        'content': doc.page_content[:500],
                                        'metadata': doc.metadata,
                                        'page_number': doc.metadata.get('page', 1),
                                        'similarity_score': 0.85,
                                        'chunk_id': f"lc_chunk_{i}"
                                    }
                                    for i, doc in enumerate(lc_result['sources'])
                                ]
                            }
                        else:
                            # Use custom implementation
                            result = retriever.answer_question(
                                question=question,
                                top_k=top_k,
                                use_reformulation=use_reformulation,
                                use_chain_of_thought=use_cot
                            )
                        
                        response_time = time.time() - start_time
                        
                        # Save to database
                        model_name = f"{model_choice} ({rag_choice})"
                        sources_for_db = [
                            {
                                'title': s['metadata'].get('title', s['metadata'].get('source', 'Unknown')),
                                'author': s['metadata'].get('author', 'Unknown'),
                                'page': s['page_number']
                            }
                            for s in result['sources']
                        ]
                        
                        conv_id = db.save_conversation(
                            session_id=session_id,
                            question=question,
                            answer=result['answer'],
                            model_used=model_name,
                            sources=sources_for_db,
                            response_time=response_time
                        )
                        
                        # Add to session history
                        st.session_state.conversation_history.append({
                            'id': conv_id,
                            'question': question,
                            'answer': result['answer'],
                            'sources': result['sources'],
                            'model': model_name,
                            'rag_type': rag_choice,
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        })
                        
                        # Display answer
                        st.markdown("### 💡 Answer")
                        
                        # Show which model and RAG was used
                        model_badge = "Fine-Tuned ✨" if use_fine_tuned else "Base GPT-3.5"
                        rag_badge = "🦜 LangChain" if use_langchain else "🏗️ Custom"
                        
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.caption(f"**Model:** {model_badge} | **RAG:** {rag_badge} | **Time:** {response_time:.2f}s")
                        with col2:
                            # Feedback buttons
                            fcol1, fcol2 = st.columns(2)
                            with fcol1:
                                if st.button("👍", key=f"thumbs_up_{conv_id}"):
                                    db.add_feedback(conv_id, 1)
                                    st.success("Thanks!")
                            with fcol2:
                                if st.button("👎", key=f"thumbs_down_{conv_id}"):
                                    db.add_feedback(conv_id, -1)
                                    st.info("Feedback saved")
                        
                        st.markdown(f'<div class="answer-box">{result["answer"]}</div>', 
                                  unsafe_allow_html=True)
                        
                        # Display sources
                        st.markdown("### 📚 Sources")
                        for i, source in enumerate(result['sources'], 1):
                            with st.expander(
                                f"📄 Source {i}: {source['metadata'].get('title', source['metadata'].get('source', 'Unknown'))} (Score: {source['similarity_score']:.3f})",
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
                        st.info("Try switching RAG implementation or check your API key.")
            
            # Show conversation history
            if show_history and st.session_state.conversation_history:
                st.divider()
                st.markdown("### 📜 Conversation History")
                
                for i, turn in enumerate(reversed(st.session_state.conversation_history), 1):
                    with st.expander(f"Q{len(st.session_state.conversation_history) - i + 1}: {turn['question'][:60]}... ({turn['timestamp']})"):
                        st.markdown(f"**Q:** {turn['question']}")
                        st.markdown(f"**A:** {turn['answer'][:300]}...")
                        st.caption(f"RAG: {turn.get('rag_type', 'Unknown')} | Model: {turn['model']}")
    
    # ============================================================
    # TAB 2: SEARCH
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
    # TAB 3: HISTORY
    # ============================================================
    with tab3:
        st.header("📜 Conversation History")
        
        subtab1, subtab2, subtab3 = st.tabs(["Current Session", "Search History", "Analytics"])
        
        with subtab1:
            st.markdown("### Current Session History")
            
            # Get current session history
            history = db.get_session_history(session_id)
            
            if not history:
                st.info("No conversations yet in this session. Ask a question to get started!")
            else:
                st.success(f"📊 {len(history)} conversations in this session")
                
                # Display history
                for i, conv in enumerate(history, 1):
                    with st.expander(
                        f"Q{i}: {conv['question'][:60]}... ({conv['timestamp']})",
                        expanded=(i == 1)
                    ):
                        st.markdown(f"**Question:** {conv['question']}")
                        st.markdown(f"**Configuration:** {conv['model_used']}")
                        st.markdown(f"**Answer:**")
                        st.markdown(f'<div class="answer-box">{conv["answer"]}</div>', 
                                  unsafe_allow_html=True)
                        
                        if conv['sources']:
                            st.markdown(f"**Sources:** {len(conv['sources'])} papers")
                        
                        # Feedback display
                        if conv['feedback'] == 1:
                            st.success("👍 Positive feedback")
                        elif conv['feedback'] == -1:
                            st.warning("👎 Negative feedback")
                
                # Export button
                st.divider()
                if st.button("📥 Export Current Session", type="primary"):
                    export_path = db.export_history(session_id=session_id)
                    st.success(f"✅ Exported to: {export_path}")
                    with open(export_path, 'r') as f:
                        st.download_button(
                            label="⬇️ Download JSON",
                            data=f.read(),
                            file_name=f"session_{session_id[:8]}.json",
                            mime="application/json"
                        )
        
        with subtab2:
            st.markdown("### Search Conversations")
            
            search_term = st.text_input("Search for:", placeholder="e.g., machine learning")
            
            if search_term:
                results = db.search_conversations(search_term, limit=20)
                
                if not results:
                    st.warning("No matching conversations found.")
                else:
                    st.success(f"Found {len(results)} matching conversations")
                    
                    for i, conv in enumerate(results, 1):
                        with st.expander(f"{i}. {conv['question'][:70]}...", expanded=(i == 1)):
                            st.markdown(f"**Question:** {conv['question']}")
                            st.markdown(f"**Answer:** {conv['answer'][:300]}...")
                            st.caption(f"Model: {conv['model_used']} | {conv['timestamp']}")
            
            st.divider()
            st.markdown("### 🔥 Popular Queries")
            
            popular = db.get_popular_queries(limit=10)
            if popular:
                for i, item in enumerate(popular, 1):
                    st.markdown(f"{i}. **{item['query']}** (asked {item['count']} times)")
        
        with subtab3:
            st.markdown("### 📊 Advanced Analytics Dashboard")
            
            analytics = db.get_analytics()
            
            # Top metrics row with styling
            st.markdown("#### 🎯 Key Performance Indicators")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric(
                    "Total Questions", 
                    analytics['total_conversations'],
                    delta="+New" if analytics['total_conversations'] > 0 else None
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Active Sessions", analytics['total_sessions'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                avg_time = analytics['avg_response_time']
                st.metric(
                    "Avg Response Time", 
                    f"{avg_time:.2f}s" if avg_time else "N/A",
                    delta="-Fast" if avg_time and avg_time < 2 else None,
                    delta_color="inverse"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                current_session = len(history) if history else 0
                st.metric(
                    "This Session", 
                    current_session,
                    delta=f"{current_session} Q&A"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.divider()
            
            # Model comparison section
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 🤖 Model Performance Comparison")
                model_usage = analytics['model_usage']
                
                if model_usage:
                    import pandas as pd
                    import plotly.graph_objects as go
                    
                    # Prepare data
                    models = []
                    counts = []
                    for model, count in model_usage.items():
                        # Shorten model names for display
                        if "LangChain" in model:
                            model_name = "LangChain " + ("Fine-Tuned" if "Fine" in model else "Base")
                        elif "Custom" in model:
                            model_name = "Custom " + ("Fine-Tuned" if "Fine" in model else "Base")
                        else:
                            model_name = "Fine-Tuned ✨" if "ft:" in model else "Base GPT-3.5"
                        models.append(model_name)
                        counts.append(count)
                    
                    # Create pie chart
                    fig = go.Figure(data=[go.Pie(
                        labels=models,
                        values=counts,
                        hole=0.4,
                        marker=dict(colors=['#2ecc71', '#3498db', '#10b981', '#6366f1']),
                        textinfo='label+percent',
                        textfont=dict(size=12)
                    )])
                    
                    fig.update_layout(
                        showlegend=True,
                        height=300,
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No model usage data yet. Ask some questions!")
            
            with col2:
                st.markdown("#### 📈 Usage Stats")
                
                if model_usage:
                    for model, count in model_usage.items():
                        # Simplify model name
                        if "LangChain" in model:
                            display_name = "🦜 LangChain"
                        elif "Custom" in model:
                            display_name = "🏗️ Custom"
                        elif "Fine" in model or "ft:" in model:
                            display_name = "✨ Fine-Tuned"
                        else:
                            display_name = "Base Model"
                        
                        percentage = (count / analytics['total_conversations']) * 100
                        
                        st.markdown(f"**{display_name}**")
                        st.progress(percentage / 100)
                        st.caption(f"{count} queries ({percentage:.1f}%)")
                        st.markdown("")
                
                # Performance indicator
                if avg_time:
                    st.markdown("**⚡ Performance**")
                    if avg_time < 2:
                        st.success("Excellent (< 2s)")
                    elif avg_time < 3:
                        st.info("Good (< 3s)")
                    else:
                        st.warning("Acceptable (> 3s)")
            
            st.divider()
            
            # Activity trends
            st.markdown("#### 📈 Activity Trends (Last 7 Days)")
            daily = analytics['daily_activity']
            
            if daily:
                import pandas as pd
                import plotly.express as px
                
                df = pd.DataFrame(daily)
                df['date'] = pd.to_datetime(df['date'])
                
                # Create bar chart
                fig = px.bar(
                    df, 
                    x='date', 
                    y='count',
                    title="Questions Asked per Day",
                    labels={'count': 'Number of Questions', 'date': 'Date'},
                    color='count',
                    color_continuous_scale='Blues'
                )
                
                fig.update_layout(
                    showlegend=False,
                    height=300,
                    xaxis_title="",
                    yaxis_title="Questions"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total This Week", df['count'].sum())
                with col2:
                    st.metric("Daily Average", f"{df['count'].mean():.1f}")
                with col3:
                    st.metric("Peak Day", df['count'].max())
            else:
                st.info("📊 No activity data yet. Start asking questions to see trends!")
            
            st.divider()
            
            # Top queries section
            st.markdown("#### 🔥 Trending Questions")
            popular = db.get_popular_queries(limit=5)
            
            if popular:
                for i, item in enumerate(popular, 1):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{i}.** {item['query']}")
                        st.caption(f"Last asked: {item['last_asked']}")
                    with col2:
                        st.markdown(f"**{item['count']}x**")
                    
                    if i < len(popular):
                        st.markdown("---")
            else:
                st.info("No trending queries yet")
            
            st.divider()
            
            # Insights section
            st.markdown("#### 💡 Quick Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**System Health** ✅")
                health_items = [
                    f"✅ Vector Store: {vector_store.index.ntotal} vectors",
                    f"✅ Documents: {stats['total_documents']} papers",
                    f"✅ Database: {analytics['total_conversations']} conversations",
                    f"✅ Sessions: {analytics['total_sessions']} active"
                ]
                for item in health_items:
                    st.markdown(item)
            
            with col2:
                st.markdown("**RAG Implementations** 🔧")
                st.markdown("✅ Custom RAG (from scratch)")
                st.markdown("✅ LangChain RAG (framework)")
                st.markdown("💡 Try both and compare!")
    
    # ============================================================
    # TAB 4: UPLOAD
    # ============================================================
    with tab4:
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
    # TAB 5: ANALYTICS
    # ============================================================
    with tab5:
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
                - ✅ Custom RAG implementation
                - ✅ LangChain RAG framework
                - ✅ Fine-tuned model (GPT-3.5)
                - ✅ Query reformulation
                - ✅ Citation tracking
                - ✅ Multi-turn conversations
                - ✅ Conversation database
                - ✅ Analytics dashboard
                - ✅ Model comparison
                """)
            
            with col2:
                st.markdown("**Technologies Used:**")
                st.markdown("""
                - 🔧 RAG (Custom + LangChain)
                - 🎯 Prompt Engineering
                - 🔬 Fine-Tuning (OpenAI)
                - 📊 FAISS Vector Database
                - 🤖 Sentence Transformers (Hugging Face)
                - 🦜 LangChain Framework
                - 🌐 Streamlit Web Interface
                - 💾 SQLite Database
                - 📊 Plotly Visualizations
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
    
    # ============================================================
    # TAB 6: MULTIMODAL
    # ============================================================
    with tab6:
        st.header("🖼️ Multimodal Features")
        
        subtab1, subtab2 = st.tabs(["Extracted Images", "Synthetic Data"])
        
        with subtab1:
            st.markdown("### 📸 Extracted Images from Papers")
            
            image_dir = Path("data/images")
            if image_dir.exists():
                images = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
                
                if images:
                    st.success(f"✅ Extracted {len(images)} images from papers")
                    
                    # Filter options
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown("**Image Gallery**")
                    with col2:
                        num_cols = st.selectbox("Columns", [2, 3, 4], index=1)
                    
                    # Display in grid
                    cols = st.columns(num_cols)
                    for i, img_path in enumerate(images[:15]):
                        with cols[i % num_cols]:
                            try:
                                st.image(str(img_path), caption=img_path.stem, use_column_width=True)
                            except Exception as e:
                                st.error(f"Error loading {img_path.name}")
                    
                    if len(images) > 15:
                        st.info(f"Showing 15 of {len(images)} images")
                    
                    st.divider()
                    with st.expander("📋 View Complete Image Index"):
                        index_file = image_dir / "image_index.md"
                        if index_file.exists():
                            st.markdown(index_file.read_text(encoding='utf-8'))
                else:
                    st.info("No images found.")
                    if st.button("🚀 Extract Images Now", type="primary"):
                        with st.spinner("Extracting images..."):
                            try:
                                from src.multimodal.image_extractor import ImageExtractor
                                extractor = ImageExtractor()
                                all_images = extractor.extract_from_directory(config.paths.papers_raw)
                                total = sum(len(imgs) for imgs in all_images.values())
                                st.success(f"✅ Extracted {total} images!")
                                st.balloons()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
            else:
                if st.button("🚀 Extract Images Now", type="primary"):
                    with st.spinner("Extracting images..."):
                        try:
                            from src.multimodal.image_extractor import ImageExtractor
                            extractor = ImageExtractor()
                            all_images = extractor.extract_from_directory(config.paths.papers_raw)
                            total = sum(len(imgs) for imgs in all_images.values())
                            st.success(f"✅ Extracted {total} images!")
                            st.balloons()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        with subtab2:
            st.markdown("### 🔄 Synthetic Training Data")
            
            synthetic_dir = Path("data/training/synthetic")
            
            if synthetic_dir.exists():
                json_files = list(synthetic_dir.glob("*.json"))
                
                if json_files:
                    st.success(f"✅ Found {len(json_files)} synthetic datasets")
                    
                    for json_file in json_files:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        with st.expander(f"📊 {json_file.stem.replace('_', ' ').title()}", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Pairs", len(data))
                            with col2:
                                avg_q = sum(len(p['question']) for p in data) / len(data) if data else 0
                                st.metric("Avg Q Length", f"{avg_q:.0f} chars")
                            with col3:
                                avg_a = sum(len(p['answer']) for p in data) / len(data) if data else 0
                                st.metric("Avg A Length", f"{avg_a:.0f} chars")
                            
                            st.divider()
                            
                            st.markdown("**Sample Q&A Pairs:**")
                            for i, pair in enumerate(data[:3], 1):
                                st.markdown(f"**{i}. Question:**")
                                st.info(pair['question'])
                                st.markdown(f"**Answer:**")
                                st.markdown(f'<div class="answer-box">{pair["answer"][:250]}...</div>', unsafe_allow_html=True)
                                
                                if i < 3:
                                    st.markdown("---")
                            
                            st.divider()
                            st.download_button(
                                label=f"📥 Download {json_file.stem}.json",
                                data=json.dumps(data, indent=2),
                                file_name=json_file.name,
                                mime="application/json"
                            )
                else:
                    st.info("No synthetic data found.")
                    if st.button("🚀 Generate Synthetic Q&A Pairs Now", type="primary"):
                        with st.spinner("Generating Q&A pairs... This may take 2-3 minutes."):
                            try:
                                from src.synthetic_data.generator import demo_synthetic_generation
                                demo_synthetic_generation()
                                st.success("✅ Generated synthetic Q&A pairs!")
                                st.balloons()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
            else:
                if st.button("🚀 Generate Synthetic Q&A Pairs Now", type="primary"):
                    with st.spinner("Generating Q&A pairs..."):
                        try:
                            from src.synthetic_data.generator import demo_synthetic_generation
                            demo_synthetic_generation()
                            st.success("✅ Generated synthetic Q&A pairs!")
                            st.balloons()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()