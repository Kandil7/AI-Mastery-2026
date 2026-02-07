"""
RAG Engine Mini - Demo UI
==========================
Simple Gradio interface for interacting with the RAG API.

واجهة Gradio تجريبية للتفاعل مع محرك RAG
"""

import os
import time
import requests
import gradio as gr

# Configuration from environment
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "demo_api_key_12345678")

def upload_file(file):
    """Upload file to the API."""
    if file is None:
        return "No file selected."
    
    url = f"{API_BASE_URL}/api/v1/documents/upload"
    headers = {"X-API-KEY": API_KEY}
    
    try:
        with open(file.name, "rb") as f:
            files = {"file": (os.path.basename(file.name), f, "application/octet-stream")}
            response = requests.post(url, headers=headers, files=files)
        
        if response.status_code == 200:
            data = response.json()
            return f"✅ Upload Successful!\nDocument ID: {data['document_id']}\nStatus: {data['status']}"
        else:
            return f"❌ Upload Failed: {response.status_code}\n{response.text}"
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

def ask_question_stream(question, k_vec=30, k_kw=30, rerank_n=8, expand=False):
    """Ask a question via hybrid search with streaming."""
    if not question.strip():
        yield "Please enter a question.", ""
        return
    
    url = f"{API_BASE_URL}/api/v1/queries/ask-hybrid-stream"
    headers = {
        "X-API-KEY": API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "question": question,
        "k_vec": k_vec,
        "k_kw": k_kw,
        "rerank_top_n": rerank_n,
        "expand_query": expand
    }
    
    try:
        full_response = ""
        with requests.post(url, headers=headers, json=payload, stream=True) as response:
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_response += chunk
                        yield full_response, "Streaming in progress..."
                yield full_response, "✅ Done"
            else:
                yield f"❌ Error: {response.status_code}\n{response.text}", ""
            
    except Exception as e:
        yield f"❌ Error: {str(e)}", ""

# -----------------------------------------------------------------------------
# Build Gradio UI
# -----------------------------------------------------------------------------

with gr.Blocks(title="RAG Engine Mini - Demo") as demo:
    gr.Markdown("# 🤖 RAG Engine Mini - Stage 2 Demo")
    gr.Markdown("Advanced RAG with **Streaming**, Hybrid Search, and Query Expansion.")
    
    with gr.Tab("💬 Ask Questions"):
        with gr.Row():
            with gr.Column(scale=3):
                q_input = gr.Textbox(label="Question", placeholder="What do you want to know?", lines=3)
                expand_chk = gr.Checkbox(label="Expand Query (LLM-based)", value=False)
                ask_btn = gr.Button("Submit Query (Streaming)", variant="primary")
            with gr.Column(scale=1):
                k_vec_sld = gr.Slider(minimum=1, maximum=100, value=30, label="Vector K")
                k_kw_sld = gr.Slider(minimum=1, maximum=100, value=30, label="Keyword K")
                rerank_sld = gr.Slider(minimum=1, maximum=20, value=8, label="Rerank Top-N")
        
        a_output = gr.Markdown(label="Response")
        s_output = gr.Textbox(label="Status/Sources", lines=2)
        
        ask_btn.click(
            ask_question_stream,
            inputs=[q_input, k_vec_sld, k_kw_sld, rerank_sld, expand_chk],
            outputs=[a_output, s_output]
        )
        
    with gr.Tab("📁 Upload Documents"):
        file_input = gr.File(label="Upload PDF, DOCX, or TXT")
        upload_btn = gr.Button("Upload and Index", variant="primary")
        status_output = gr.Textbox(label="Upload Status")
        
        upload_btn.click(
            upload_file,
            inputs=[file_input],
            outputs=[status_output]
        )
    
    gr.Markdown("---")
    gr.Markdown("Developed for [AI-Mastery-2026](file:///k:/learning/technical/ai-ml/AI-Mastery-2026/README.md)")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860,share=True)
