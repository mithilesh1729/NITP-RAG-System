"""
NITP RAG System - Web Interface
Streamlit-based chat UI for the NIT Patna Institutional Knowledge Assistant.
Reuses the exact same RAG pipeline as chat.py (no code duplication).

Run with: streamlit run app.py
"""
import streamlit as st
import time

# ============================================================
# Page Configuration (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="NIT Patna RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Theme-aware CSS (works in BOTH light and dark mode)
# ============================================================
st.markdown("""
<style>
    /* Main container width */
    .main .block-container {
        max-width: 900px;
        padding-top: 1.5rem;
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 0.8rem 0 1rem 0;
        border-bottom: 2px solid rgba(128,128,128,0.3);
        margin-bottom: 1.2rem;
    }
    .main-header h1 {
        font-size: 1.8rem;
        margin-bottom: 0.1rem;
    }
    .main-header .subtitle {
        opacity: 0.7;
        font-size: 0.9rem;
    }
    
    /* Source badges — theme-aware */
    .source-badge {
        display: inline-block;
        background: rgba(26, 115, 232, 0.15);
        color: #58a6ff;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        margin: 2px 4px;
        border: 1px solid rgba(88, 166, 255, 0.3);
    }
    
    /* Score breakdown — theme-aware */
    .score-line {
        font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
        font-size: 0.76rem;
        padding: 3px 0;
        border-bottom: 1px solid rgba(128,128,128,0.15);
        line-height: 1.5;
    }
    .score-line:last-child {
        border-bottom: none;
    }
    
    /* Sidebar config cards — theme-aware */
    .config-card {
        padding: 0.55rem 0.75rem;
        margin-bottom: 0.5rem;
        border-radius: 6px;
        border-left: 3px solid #58a6ff;
        background: rgba(88, 166, 255, 0.08);
        font-size: 0.82rem;
    }
    .config-card .card-label {
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.7;
        margin-bottom: 2px;
    }
    .config-card .card-value {
        font-family: 'Cascadia Code', 'Consolas', monospace;
        font-size: 0.82rem;
    }
    
    /* Sample query buttons */
    .stButton > button {
        width: 100%;
        text-align: left;
        font-size: 0.8rem;
        padding: 0.4rem 0.6rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Initialize RAG Backend (cached — loads only once)
# ============================================================
@st.cache_resource(show_spinner="🔄 Loading RAG Pipeline... (first run takes ~10s)")
def load_rag_pipeline():
    """Load all RAG components once and cache them across sessions."""
    from src.embeddings.generator import GeminiEmbedder
    from src.storage.vector_store import ChromaVectorStore
    from src.generation.llm import GeminiLLM
    
    embedder = GeminiEmbedder()
    db = ChromaVectorStore()
    llm = GeminiLLM()
    
    # Get stats for sidebar
    doc_count = db.collection.count()
    
    return embedder, db, llm, doc_count


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("## 🎓 NIT Patna RAG")
    st.caption("Institutional Knowledge Assistant")
    
    st.divider()
    
    st.markdown("### ⚙️ System Configuration")
    
    st.markdown("""
    <div class="config-card">
        <div class="card-label">Embedding Model</div>
        <div class="card-value">gemini-embedding-2</div>
        <div style="font-size:0.72rem; opacity:0.6;">3072-dim · 8K token capacity</div>
    </div>
    <div class="config-card">
        <div class="card-label">Generation Model</div>
        <div class="card-value">Gemini Flash</div>
        <div style="font-size:0.72rem; opacity:0.6;">1M token context window</div>
    </div>
    <div class="config-card">
        <div class="card-label">Search Strategy</div>
        <div class="card-value">Hybrid: Vector + BM25</div>
        <div style="font-size:0.72rem; opacity:0.6;">Reciprocal Rank Fusion (k=60)</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Top-K slider
    top_k = st.slider(
        "📊 Chunks to retrieve (Top-K)", 
        min_value=5, max_value=35, value=25,
        help="Number of chunks retrieved by hybrid search. With Gemini Flash's 1M context window, 25 chunks gives high recall without latency penalty."
    )
    
    show_scores = st.toggle("🔍 Show retrieval scores", value=True, 
                            help="Display Hybrid Search scoring (Vector Rank, BM25 Rank, RRF Score) for each chunk.")
    
    st.divider()
    
    # Load pipeline and show stats
    try:
        embedder, db, llm, doc_count = load_rag_pipeline()
        st.success(f"✅ Pipeline Active — {doc_count} chunks indexed")
    except Exception as e:
        st.error(f"❌ Pipeline Error: {e}")
        st.stop()
    
    st.divider()
    st.caption("Built by **Mithilesh Kumar** (2246022)")
    st.caption("Dual Degree CSE · NIT Patna")
    st.caption("Guide: **Prof. M.P. Singh**")


# ============================================================
# Main Chat Interface
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🎓 NIT Patna RAG Assistant</h1>
    <p class="subtitle">Ask questions about fees, exams, regulations, faculty, and more — powered by Hybrid RAG</p>
</div>
""", unsafe_allow_html=True)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sample queries (shown only when chat is empty)
if not st.session_state.messages:
    st.markdown("##### 💡 Try a sample query:")
    sample_cols = st.columns(3)
    sample_queries = [
        "What is the tuition fee for SC/ST students?",
        "When is the first MSE for 3rd sem CSE-AI-DS?",
        "What are the eligibility criteria for JRF?"
    ]
    for i, query in enumerate(sample_queries):
        with sample_cols[i]:
            if st.button(f"📝 {query}", key=f"sample_{i}", use_container_width=True):
                st.session_state.sample_query = query
                st.rerun()


# Display chat history
for message in st.session_state.messages:
    avatar = "🧑‍🎓" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        
        # Re-render sources and scores for assistant messages
        if message["role"] == "assistant":
            if message.get("sources"):
                source_html = " ".join(
                    f'<span class="source-badge">📄 {s}</span>' for s in message["sources"]
                )
                st.markdown(source_html, unsafe_allow_html=True)
            
            if message.get("scores") and show_scores:
                with st.expander("🔍 Hybrid Search Scoring Breakdown", expanded=False):
                    for score in message["scores"]:
                        st.markdown(f'<div class="score-line">{score}</div>', unsafe_allow_html=True)
            
            if message.get("elapsed"):
                st.caption(f"⏱️ {message['elapsed']:.1f}s")


# ============================================================
# Handle User Input
# ============================================================
# Check for sample query button click
user_query = st.chat_input("Ask about NIT Patna...")

# If sample query was clicked, use that
if not user_query and "sample_query" in st.session_state:
    user_query = st.session_state.pop("sample_query")

if user_query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_query)
    
    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        start_time = time.time()
        
        # Status updates during processing
        status = st.status("🔍 Processing your query...", expanded=True)
        
        try:
            # Step 1: Embed the query
            status.write("⚡ Embedding query with Gemini...")
            query_vector = embedder.embed_text(user_query)
            
            # Step 2: Hybrid Search
            status.write("🔎 Running Hybrid Search (Vector + BM25 → RRF)...")
            results = db.hybrid_search(
                query_text=user_query, 
                query_embedding=query_vector, 
                n_results=top_k
            )
            
            if not results['documents'] or len(results['documents'][0]) == 0:
                answer = "I couldn't find any relevant information in the database for your question."
                sources = []
                scores = []
            else:
                # Step 3: Prepare context
                retrieved_text = "\n\n".join(results['documents'][0])
                
                # Step 4: Extract deterministic sources (NOT LLM-generated)
                sources = list(set(
                    meta.get('source', 'Unknown') for meta in results['metadatas'][0]
                ))
                
                # Step 5: Get scoring info
                scores = results.get('hybrid_scores', [[]])[0]
                
                # Step 6: Generate answer
                status.write(f"🧠 Generating answer from {len(results['documents'][0])} chunks...")
                answer = llm.generate_answer(query=user_query, context=retrieved_text)
            
            elapsed = time.time() - start_time
            status.update(label=f"✅ Done in {elapsed:.1f}s", state="complete", expanded=False)
            
            # Display the answer
            st.markdown(answer)
            
            # Display sources as badges (only if answer is not "I don't know")
            if sources and "I don't know" not in answer:
                source_html = " ".join(
                    f'<span class="source-badge">📄 {s}</span>' for s in sources
                )
                st.markdown(source_html, unsafe_allow_html=True)
            
            # Display hybrid search scores
            if scores and show_scores:
                with st.expander("🔍 Hybrid Search Scoring Breakdown", expanded=False):
                    for score in scores:
                        st.markdown(f'<div class="score-line">{score}</div>', unsafe_allow_html=True)
            
            # Show latency
            st.caption(f"⏱️ {elapsed:.1f}s")
            
            # Save to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources if "I don't know" not in answer else [],
                "scores": scores,
                "elapsed": elapsed
            })
            
        except Exception as e:
            status.update(label="❌ Error", state="error", expanded=False)
            error_msg = f"⚠️ Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "sources": [],
                "scores": [],
                "elapsed": 0
            })
