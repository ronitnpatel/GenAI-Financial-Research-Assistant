"""
Streamlit web UI for the JPMorgan 10-K RAG system
"""

import streamlit as st
from rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="JPMorgan 10-K RAG",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 JPMorgan Chase 10-K RAG System")
st.caption("Ask questions about JPMorgan Chase annual filings (last 5 years)")

# Load pipeline and cache

@st.cache_resource(show_spinner="Loading RAG pipeline...")
def load_rag():
    return RAGPipeline()

rag = load_rag()

# Sidebar 
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=5)
    show_context = st.checkbox("Show retrieved context", value=True)

    st.markdown("---")
    st.markdown("**💡 Sample Questions**")
    samples = [
        "How did JPMorgan characterise its exposure to interest rate risk?",
        "What major acquisition did JPMorgan complete in 2023?",
        "How does JPMorgan describe its liquidity risk management approach?",
        "What are the main business segments JPMorgan operates in?",
        "How does JPMorgan describe its approach to credit risk management?",
        "How does JPMorgan describe the competitive environment it operates in?",
        "What is JPMorgan's dividend policy?",
    ]
    for s in samples:
        if st.button(s, use_container_width=True, key=s):
            st.session_state["q_input"] = s

# Main area
question = st.text_input(
    "Ask a question about JPMorgan's 10-K filings:",
    key="q_input",
    placeholder="e.g. What major acquisition did JPMorgan complete in 2023?",
)

if st.button("🔍 Ask", type="primary", use_container_width=False):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            result = rag.query(question, top_k=top_k)

        st.subheader("📝 Answer")
        st.success(result["answer"])

        if show_context:
            with st.expander(f"📚 Retrieved Context ({top_k} chunks)", expanded=False):
                for i, chunk in enumerate(result["source_chunks"], 1):
                    meta = chunk["metadata"]
                    st.markdown(
                        f"**[{i}]  Year: `{meta.get('year','?')}`  |  "
                        f"File: `{meta.get('filename','?')}`  |  "
                        f"Score: `{chunk['score']}`**"
                    )
                    st.text(chunk["text"][:600] + ("..." if len(chunk["text"]) > 600 else ""))
                    st.divider()
