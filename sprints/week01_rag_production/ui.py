
import streamlit as st
import requests
import pandas as pd
import json

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Week 1: RAG Master", layout="wide")

st.title("🤖 Week 1: Production RAG Demo")
st.markdown("""
This dashboard demonstrates a **Hybrid Retrieval** system (Dense + Sparse) integrated with an LLM.
""")

# --- Sidebar: System Status & Indexing ---
with st.sidebar:
    st.header("⚙️ Knowledge Base")
    
    # Check API Status
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            st.success("Backend Online")
        else:
            st.error("Backend Error")
    except requests.exceptions.ConnectionError:
        st.error("Backend Offline (Run `python api.py`)")
        st.stop()
        
    st.subheader("Add Document")
    with st.form("add_doc_form"):
        doc_id = st.text_input("Doc ID", value=f"doc_{pd.Timestamp.now().strftime('%f')}")
        doc_content = st.text_area("Content", "Enter text to index...")
        submitted = st.form_submit_button("Index Document")
        
        if submitted:
            payload = [{"id": doc_id, "content": doc_content, "metadata": {"source": "ui"}}]
            res = requests.post(f"{API_URL}/index", json=payload)
            if res.status_code == 200:
                st.toast(f"Indexed! Total docs: {res.json()['total_docs']}")
            else:
                st.error("Failed to index")

# --- Main Area: Chat/Query ---
st.header("🔍 Semantic Search")

query = st.text_input("Ask a question:", "What is Hybrid Retrieval?")
top_k = st.slider("Retrieval Count (k)", 1, 5, 3)

if st.button("Search & Generate"):
    with st.spinner("Retrieving & Thinking..."):
        try:
            payload = {"query": query, "k": top_k}
            response = requests.post(f"{API_URL}/query", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # Layout: Answer on top, sources below
                st.subheader("💡 Answer")
                st.info(data["response"])
                
                st.subheader("📚 Retrieved Context (Evidence)")
                
                source_data = []
                for s in data["sources"]:
                   source_data.append({
                       "Rank": s["rank"],
                       "Score": f"{s['score']:.4f}",
                       "Content": s["content"],
                       "ID": s["id"]
                   })
                
                st.dataframe(pd.DataFrame(source_data).set_index("Rank"), use_container_width=True)
                
            else:
                st.error(f"Error: {response.text}")
                
        except Exception as e:
            st.error(f"Request failed: {e}")

st.markdown("---")
st.caption("AI-Mastery-2026 | Week 1 Sprint | Hybrid RAG Implementation")
