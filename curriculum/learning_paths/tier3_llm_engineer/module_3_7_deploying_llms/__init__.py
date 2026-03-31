"""
Module 3.7: Deploying LLMs

Production-ready deployment implementations:
- Local: Ollama, LM Studio, llama.cpp deployment
- Demo: Gradio, Streamlit, Hugging Face Spaces
- Server: vLLM, TGI, FastAPI servers
- Edge: MLC LLM, mobile, WebLLM deployment
"""

from .local import (
    LocalDeployment,
    OllamaDeployment,
    LMStudioDeployment,
    LlamaCppDeployment,
    LocalServerConfig,
)
from .demo import (
    DemoApp,
    GradioApp,
    StreamlitApp,
    HuggingFaceSpaces,
    DemoConfig,
)
from .server import (
    LLMServer,
    VLLMServer,
    TGIServer,
    FastAPIServer,
    ServerConfig,
    LoadBalancer,
)
from .edge import (
    EdgeDeployment,
    MLCDeployment,
    MobileDeployment,
    WebLLMDeployment,
    EdgeConfig,
)

__all__ = [
    # Local
    "LocalDeployment",
    "OllamaDeployment",
    "LMStudioDeployment",
    "LlamaCppDeployment",
    "LocalServerConfig",
    # Demo
    "DemoApp",
    "GradioApp",
    "StreamlitApp",
    "HuggingFaceSpaces",
    "DemoConfig",
    # Server
    "LLMServer",
    "VLLMServer",
    "TGIServer",
    "FastAPIServer",
    "ServerConfig",
    "LoadBalancer",
    # Edge
    "EdgeDeployment",
    "MLCDeployment",
    "MobileDeployment",
    "WebLLMDeployment",
    "EdgeConfig",
]

__version__ = "1.0.0"
