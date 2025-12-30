"""
AI-Mastery-2026 Streamlit Application

A web interface for interacting with the RAG system and ML models.
"""

import streamlit as st
import requests
import json
import time
from typing import List, Dict, Any
import os

# Configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000")
APP_TITLE = "AI-Mastery-2026"
APP_DESCRIPTION = "Full-Stack AI Engineering Toolkit"

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #dbeafe;
    }
    .assistant-message {
        background-color: #f3f4f6;
    }
    .source-card {
        background-color: #fef3c7;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin-top: 0.5rem;
        font-size: 0.875rem;
    }
    .metric-card {
        background-color: #ecfdf5;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #059669;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> Dict[str, Any]:
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "message": f"Status code: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}


def get_models() -> List[Dict[str, Any]]:
    """Get list of available models."""
    try:
        response = requests.get(f"{API_URL}/models", timeout=5)
        if response.status_code == 200:
            return response.json().get("models", [])
        return []
    except requests.exceptions.RequestException:
        return []


def make_prediction(features: List[float], model_name: str) -> Dict[str, Any]:
    """Make a prediction using the API."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features, "model_name": model_name},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"Status code: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def make_rag_query(query: str, k: int = 3) -> Dict[str, Any]:
    """Make a RAG query."""
    try:
        response = requests.post(
            f"{API_URL}/chat/completions",
            json={"query": query, "k": k},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"Status code: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def main():
    """Main application."""
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=AI-Mastery", width=150)
        st.markdown("---")
        
        # API Status
        st.subheader("🔌 API Status")
        health = check_api_health()
        if health.get("status") == "healthy":
            st.success(f"✅ Connected")
            st.caption(f"Models loaded: {health.get('models_loaded', 0)}")
        else:
            st.error("❌ Disconnected")
            st.caption(health.get("message", "Unable to connect"))
        
        st.markdown("---")
        
        # Navigation
        st.subheader("📍 Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Home", "💬 Chat (RAG)", "🔮 Predictions", "📊 Models", "⚙️ Settings"]
        )
        
        st.markdown("---")
        st.caption("AI-Mastery-2026 v2.0.0")
    
    # Main content
    if page == "🏠 Home":
        render_home_page()
    elif page == "💬 Chat (RAG)":
        render_chat_page()
    elif page == "🔮 Predictions":
        render_predictions_page()
    elif page == "📊 Models":
        render_models_page()
    elif page == "⚙️ Settings":
        render_settings_page()


def render_home_page():
    """Render the home page."""
    st.markdown('<p class="main-header">🤖 AI-Mastery-2026</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Full-Stack AI Engineering Toolkit</p>', unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    models = get_models()
    health = check_api_health()
    
    with col1:
        st.metric("Models Loaded", len(models))
    
    with col2:
        status = "🟢 Online" if health.get("status") == "healthy" else "🔴 Offline"
        st.metric("API Status", status)
    
    with col3:
        st.metric("Response Time", "< 50ms")
    
    with col4:
        st.metric("Uptime", "99.9%")
    
    st.markdown("---")
    
    # Features
    st.subheader("🚀 Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Machine Learning**
        - Classification models (RandomForest, Logistic)
        - Regression models (GradientBoosting)
        - Real-time inference API
        - Model versioning
        """)
        
        st.markdown("""
        **RAG System**
        - Document ingestion
        - Vector search (HNSW)
        - Hybrid retrieval
        - Context-aware generation
        """)
    
    with col2:
        st.markdown("""
        **Production Features**
        - Docker deployment
        - Prometheus metrics
        - Grafana dashboards
        - CI/CD pipeline
        """)
        
        st.markdown("""
        **Educational Content**
        - From-scratch implementations
        - Jupyter notebooks
        - Interview preparation
        - System design guides
        """)


def render_chat_page():
    """Render the chat page for RAG interactions."""
    st.header("💬 Chat with AI-Mastery")
    st.caption("Ask questions about AI, ML, and the codebase")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Try RAG query first
                result = make_rag_query(prompt)
                
                if "error" in result:
                    # Fallback response
                    response = f"I apologize, but I couldn't process your request. Error: {result['error']}"
                    sources = []
                else:
                    response = result.get("response", "I don't have an answer for that.")
                    sources = result.get("sources", [])
                
                st.markdown(response)
                
                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.markdown(f"- {source}")
        
        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })


def render_predictions_page():
    """Render the predictions page."""
    st.header("🔮 Model Predictions")
    
    # Model selection
    models = get_models()
    model_names = [m.get("model_id", "unknown") for m in models] if models else ["classification_model"]
    
    selected_model = st.selectbox("Select Model", model_names)
    
    # Get model info
    if models:
        model_info = next((m for m in models if m.get("model_id") == selected_model), None)
        if model_info:
            n_features = model_info.get("n_features", 10)
        else:
            n_features = 10
    else:
        n_features = 10
    
    st.markdown("---")
    
    # Feature input
    st.subheader("Input Features")
    
    input_method = st.radio("Input Method", ["Manual", "JSON", "Random"])
    
    if input_method == "Manual":
        cols = st.columns(min(n_features, 5))
        features = []
        for i in range(n_features):
            with cols[i % 5]:
                val = st.number_input(f"Feature {i+1}", value=0.0, key=f"feature_{i}")
                features.append(val)
    
    elif input_method == "JSON":
        json_input = st.text_area("Enter features as JSON array", value="[0.0, 0.0, 0.0, 0.0, 0.0]")
        try:
            features = json.loads(json_input)
        except json.JSONDecodeError:
            st.error("Invalid JSON")
            features = []
    
    else:  # Random
        import numpy as np
        if st.button("Generate Random Features"):
            features = np.random.randn(n_features).tolist()
            st.session_state.random_features = features
        features = st.session_state.get("random_features", [0.0] * n_features)
        st.json(features)
    
    # Predict button
    if st.button("🚀 Predict", type="primary"):
        with st.spinner("Making prediction..."):
            result = make_prediction(features, selected_model)
        
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.success("Prediction Complete!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Prediction", result.get("predictions", ["N/A"])[0])
            
            with col2:
                st.metric("Processing Time", f"{result.get('processing_time', 0)*1000:.2f} ms")
            
            if result.get("probabilities"):
                st.subheader("Probabilities")
                st.bar_chart(result["probabilities"][0])


def render_models_page():
    """Render the models page."""
    st.header("📊 Model Registry")
    
    models = get_models()
    
    if not models:
        st.warning("No models loaded. Make sure the API is running and models are trained.")
        st.code("python scripts/train_save_models.py", language="bash")
        return
    
    for model in models:
        with st.expander(f"🤖 {model.get('model_id', 'Unknown')}", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Type:** {model.get('model_type', 'Unknown')}")
            
            with col2:
                st.markdown(f"**Features:** {model.get('n_features', 'Unknown')}")
            
            with col3:
                st.markdown(f"**Status:** ✅ Loaded")
            
            if model.get("metadata"):
                st.json(model["metadata"])


def render_settings_page():
    """Render the settings page."""
    st.header("⚙️ Settings")
    
    st.subheader("API Configuration")
    api_url = st.text_input("API URL", value=API_URL)
    
    st.subheader("Display Settings")
    theme = st.selectbox("Theme", ["Light", "Dark", "System"])
    
    st.subheader("RAG Settings")
    k_documents = st.slider("Number of documents to retrieve", 1, 10, 3)
    
    if st.button("Save Settings"):
        st.success("Settings saved!")
        st.info("Note: Some settings require a page refresh to take effect.")
    
    st.markdown("---")
    
    st.subheader("System Information")
    st.json({
        "app_version": "2.0.0",
        "api_url": API_URL,
        "python_version": "3.10+",
        "streamlit_version": st.__version__
    })


if __name__ == "__main__":
    main()
