
"""
Production-Ready RAG Dashboard with Enhanced Visualization and Debugging

This Streamlit dashboard provides a comprehensive interface for interacting with
the production RAG system. It includes advanced visualization capabilities,
debugging tools, and performance monitoring features to help users understand
and optimize the RAG system's behavior.

Key Features:
- Interactive query interface with adjustable parameters
- Detailed source document visualization
- Performance metrics and timing information
- Document management and indexing tools
- System health and status monitoring
- Debugging and inspection tools
- Export capabilities for analysis

Sections:
- System Status: Real-time health and performance metrics
- Document Management: Tools for adding and managing documents
- Query Interface: Advanced query controls with parameter tuning
- Results Visualization: Detailed answer and source display
- Performance Metrics: Timing and quality metrics
- Debugging Tools: Internal state inspection and analysis
"""

import streamlit as st
import requests
import pandas as pd
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Production RAG Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .source-highlight {
        background-color: #fff3cd;
        padding: 5px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Production RAG Dashboard")
st.markdown("""
**Enterprise-grade Retrieval Augmented Generation System**
Monitor, query, and optimize your RAG pipeline with advanced visualization and debugging tools.
""")

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'document_stats' not in st.session_state:
    st.session_state.document_stats = {}

# --- Sidebar: System Status & Controls ---
with st.sidebar:
    st.header("⚙️ System Control Panel")

    # API Connection Status
    st.subheader("📡 Connection Status")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            status = health_data.get("status", "unknown")
            if status == "healthy":
                st.success(f"✅ API Healthy | Docs: {health_data['details'].get('document_count', 0)}")
            else:
                st.warning(f"⚠️ API Degraded: {status}")
        else:
            st.error("❌ API Error")
    except requests.exceptions.ConnectionError:
        st.error("🔴 API Offline (Run `python api.py`)")

    # Query Parameters
    st.subheader("🎯 Query Parameters")
    top_k = st.slider("Retrieved Documents (k)", 1, 20, 3)
    include_sources = st.checkbox("Include Sources", value=True)
    timeout_seconds = st.slider("Timeout (seconds)", 1.0, 60.0, 30.0, step=1.0)

    # Debug Mode
    debug_mode = st.checkbox("Enable Debug Mode", value=False)

    # Document Management
    st.subheader("📚 Document Management")
    with st.expander("Add New Document"):
        with st.form("add_doc_form"):
            col1, col2 = st.columns(2)
            with col1:
                doc_id = st.text_input("Document ID", value=f"doc_{int(time.time())}")
            with col2:
                doc_type = st.selectbox("Document Type", ["policy", "manual", "faq", "article", "report"])

            doc_content = st.text_area("Content", height=150)
            doc_metadata = st.text_area("Metadata (JSON)", value='{"source": "dashboard", "category": "general"}', height=100)

            submitted = st.form_submit_button("📥 Index Document")

            if submitted:
                try:
                    metadata_dict = json.loads(doc_metadata)
                except json.JSONDecodeError:
                    st.error("Invalid JSON in metadata")
                    st.stop()

                payload = [{
                    "id": doc_id,
                    "content": doc_content,
                    "metadata": metadata_dict,
                    "doc_type": doc_type
                }]

                try:
                    res = requests.post(f"{API_URL}/index", json=payload, timeout=30)
                    if res.status_code == 200:
                        st.success(f"✅ Indexed document: {doc_id}")
                        st.rerun()
                    else:
                        st.error(f"❌ Indexing failed: {res.text}")
                except Exception as e:
                    st.error(f"❌ Error indexing: {str(e)}")

# --- Main Dashboard Area ---
tab_overview, tab_query, tab_insights, tab_debug = st.tabs([
    "📊 Overview",
    "💬 Query Interface",
    "📈 Insights",
    "🛠️ Debug Tools"
])

with tab_overview:
    col1, col2, col3, col4 = st.columns(4)

    # Fetch system stats
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=5)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            doc_count = health_data['details'].get('document_count', 0)
        else:
            doc_count = 0
    except:
        doc_count = 0

    with col1:
        st.metric(label="Documents Indexed", value=doc_count)
    with col2:
        st.metric(label="Active Queries", value=len(st.session_state.query_history))
    with col3:
        avg_latency = np.mean([q.get('query_time_ms', 0) for q in st.session_state.query_history]) if st.session_state.query_history else 0
        st.metric(label="Avg. Latency", value=f"{avg_latency:.2f}ms")
    with col4:
        success_rate = np.mean([q.get('success', True) for q in st.session_state.query_history]) if st.session_state.query_history else 1.0
        st.metric(label="Success Rate", value=f"{success_rate*100:.1f}%")

    # Recent queries chart
    if st.session_state.query_history:
        df_history = pd.DataFrame(st.session_state.query_history)
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])

        st.subheader("Recent Query Performance")
        fig = px.line(
            df_history,
            x='timestamp',
            y='query_time_ms',
            title="Query Latency Over Time",
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab_query:
    st.header("💬 Advanced Query Interface")

    # Query input
    query = st.text_area(
        "Enter your query:",
        value="What is Hybrid Retrieval?",
        height=100
    )

    # Query execution
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚀 Execute Query", type="primary", use_container_width=True):
            with st.spinner("Processing query..."):
                try:
                    start_time = time.time()

                    payload = {
                        "query": query,
                        "k": top_k,
                        "include_sources": include_sources,
                        "timeout_seconds": timeout_seconds
                    }

                    response = requests.post(f"{API_URL}/query", json=payload, timeout=timeout_seconds)

                    if response.status_code == 200:
                        data = response.json()

                        # Store query in history
                        query_record = {
                            "query": query,
                            "response": data["response"],
                            "timestamp": datetime.now().isoformat(),
                            "query_time_ms": data["query_time_ms"],
                            "sources_count": len(data["sources"]),
                            "total_docs": data["total_documents_indexed"],
                            "success": True
                        }
                        st.session_state.query_history.append(query_record)

                        # Display results
                        st.success("✅ Query executed successfully!")

                        # Answer section
                        with st.container(border=True):
                            st.subheader("💡 Generated Response")
                            st.write(data["response"])

                        # Sources section
                        if data["sources"]:
                            with st.expander(f"📚 Retrieved Sources ({len(data['sources'])})", expanded=True):
                                for source in data["sources"]:
                                    with st.container(border=True):
                                        st.markdown(f"**Rank {source['rank']}** | Score: `{source['score']:.4f}` | ID: `{source['id']}`")

                                        # Show content with highlighting
                                        content_preview = source['content'][:500] + "..." if len(source['content']) > 500 else source['content']
                                        st.text_area("Content Preview", value=content_preview, height=100, key=f"content_{source['id']}")

                                        if 'metadata' in source and source['metadata']:
                                            with st.expander("Metadata"):
                                                st.json(source['metadata'])

                        # Performance metrics
                        with st.expander("⏱️ Performance Metrics"):
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Query Time", f"{data['query_time_ms']:.2f}ms")
                            with col_b:
                                st.metric("Retrieved Docs", len(data["sources"]))
                            with col_c:
                                st.metric("Total Indexed", data["total_documents_indexed"])

                    else:
                        st.error(f"❌ Query failed: {response.text}")
                        st.session_state.query_history.append({
                            "query": query,
                            "timestamp": datetime.now().isoformat(),
                            "success": False
                        })

                except requests.exceptions.Timeout:
                    st.error(f"⏰ Query timed out after {timeout_seconds} seconds")
                    st.session_state.query_history.append({
                        "query": query,
                        "timestamp": datetime.now().isoformat(),
                        "success": False
                    })
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.query_history.append({
                        "query": query,
                        "timestamp": datetime.now().isoformat(),
                        "success": False
                    })

with tab_insights:
    st.header("📈 System Insights")

    if st.session_state.query_history:
        df_insights = pd.DataFrame(st.session_state.query_history)
        df_insights['timestamp'] = pd.to_datetime(df_insights['timestamp'])

        # Performance trends
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Query Latency Distribution")
            fig_hist = px.histogram(df_insights, x='query_time_ms', nbins=20, title="Latency Distribution")
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            st.subheader("Sources vs Latency")
            if 'sources_count' in df_insights.columns:
                fig_scatter = px.scatter(df_insights, x='sources_count', y='query_time_ms',
                                       title="Sources Count vs Query Time", trendline="ols")
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)

        # Query patterns
        st.subheader("Query Patterns")
        query_lengths = df_insights['query'].apply(len)
        fig_box = px.box(y=query_lengths, title="Distribution of Query Lengths")
        fig_box.update_layout(height=300)
        st.plotly_chart(fig_box, use_container_width=True)

    else:
        st.info("Execute some queries to see performance insights here.")

with tab_debug:
    st.header("🛠️ Debug Tools")

    if debug_mode:
        st.warning("⚠️ Debug mode enabled. Additional internal information will be displayed.")

        # System information
        with st.expander("🔧 System Information"):
            try:
                health_resp = requests.get(f"{API_URL}/health", timeout=5)
                if health_resp.status_code == 200:
                    st.json(health_resp.json())
                else:
                    st.error("Could not fetch health info")
            except Exception as e:
                st.error(f"Error fetching health info: {str(e)}")

        # Query history details
        with st.expander("📋 Query History Details"):
            if st.session_state.query_history:
                for i, query in enumerate(st.session_state.query_history[-10:], 1):  # Last 10 queries
                    with st.container(border=True):
                        st.markdown(f"**Query {i}** - {query['timestamp']}")
                        st.text(f"Query: {query.get('query', 'N/A')[:100]}...")
                        st.text(f"Time: {query.get('query_time_ms', 0):.2f}ms")
                        st.text(f"Success: {query.get('success', False)}")
            else:
                st.info("No query history available.")

    else:
        st.info("Enable debug mode in the sidebar to access advanced debugging tools.")

# Footer
st.markdown("---")
col_footer1, col_footer2 = st.columns([3, 1])
with col_footer1:
    st.caption("AI-Mastery-2026 | Production RAG System | Enterprise-grade Retrieval-Augmented Generation")
with col_footer2:
    st.caption(f"Queries processed: {len(st.session_state.query_history)}")
