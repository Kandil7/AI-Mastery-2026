"""
Demo App Module

Production-ready demo applications:
- Gradio apps
- Streamlit apps
- Hugging Face Spaces

Features:
- Quick deployment
- Interactive UIs
- Sharing capabilities
"""

from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class DemoConfig:
    """Configuration for demo applications."""

    # App settings
    title: str = "LLM Demo"
    description: str = "Interactive LLM demonstration"
    theme: str = "default"

    # Model settings
    model_name: str = ""
    model_endpoint: str = ""
    api_key: Optional[str] = None

    # UI settings
    max_tokens: int = 2048
    temperature: float = 0.7
    show_advanced: bool = True

    # Sharing settings
    share: bool = False
    enable_queue: bool = True

    # Server settings
    host: str = "0.0.0.0"
    port: int = 7860


class DemoApp(ABC):
    """Abstract base class for demo applications."""

    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self._running = False

    @abstractmethod
    def create_ui(self) -> Any:
        """Create the UI."""
        pass

    @abstractmethod
    async def launch(self) -> str:
        """Launch the application."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the application."""
        pass

    def get_url(self) -> Optional[str]:
        """Get application URL."""
        return None


class GradioApp(DemoApp):
    """
    Gradio-based demo application.

    Provides interactive UI with minimal code.
    """

    def __init__(
        self,
        config: DemoConfig,
        generate_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(config)
        self.generate_fn = generate_fn
        self._app = None
        self._gradio = None

        self._try_import_gradio()

    def _try_import_gradio(self) -> None:
        """Try to import Gradio."""
        try:
            import gradio as gr
            self._gradio = gr
            logger.info("Gradio imported successfully")
        except ImportError:
            logger.warning("Gradio not installed. Run: pip install gradio")

    def create_ui(self) -> Any:
        """Create Gradio UI."""
        if not self._gradio:
            raise RuntimeError("Gradio not available")

        gr = self._gradio

        with gr.Blocks(
            title=self.config.title,
            theme=self.config.theme,
        ) as demo:
            gr.Markdown(f"# {self.config.title}")
            gr.Markdown(self.config.description)

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="Conversation", height=400)
                    msg = gr.Textbox(
                        label="Message",
                        placeholder="Type your message...",
                        lines=2,
                    )
                    clear = gr.Button("Clear")

                with gr.Column(scale=1):
                    with gr.Accordion("Settings", open=not self.config.show_advanced):
                        temperature = gr.Slider(
                            minimum=0,
                            maximum=2,
                            value=self.config.temperature,
                            label="Temperature",
                        )
                        max_tokens = gr.Slider(
                            minimum=64,
                            maximum=8192,
                            value=self.config.max_tokens,
                            label="Max Tokens",
                        )
                        stream = gr.Checkbox(
                            value=True,
                            label="Stream Response",
                        )

            # Event handlers
            msg.submit(
                self._create_respond_fn(),
                inputs=[msg, chatbot, temperature, max_tokens, stream],
                outputs=[msg, chatbot],
            )

            clear.click(
                lambda: None,
                None,
                chatbot,
                queue=False,
            )

        self._app = demo
        return demo

    def _create_respond_fn(self) -> Callable:
        """Create response function for Gradio."""
        async def respond(
            message: str,
            history: List[tuple],
            temperature: float,
            max_tokens: int,
            stream: bool,
        ) -> tuple:
            if not message:
                return "", history

            # Add user message
            history = history + [(message, None)]

            # Generate response
            if self.generate_fn:
                if asyncio.iscoroutinefunction(self.generate_fn):
                    response = await self.generate_fn(
                        message,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                else:
                    response = self.generate_fn(
                        message,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                if stream and hasattr(response, '__aiter__'):
                    partial_response = ""
                    async for chunk in response:
                        partial_response += chunk
                        history[-1] = (message, partial_response)
                        yield "", history
                else:
                    history[-1] = (message, response)
            else:
                # Fallback response
                history[-1] = (message, f"Response to: {message}")

            return "", history

        return respond

    async def launch(self) -> str:
        """Launch Gradio app."""
        if not self._gradio:
            raise RuntimeError("Gradio not available")

        if not self._app:
            self.create_ui()

        # Launch in thread
        loop = asyncio.get_event_loop()

        def run_server():
            self._app.launch(
                server_name=self.config.host,
                server_port=self.config.port,
                share=self.config.share,
                show_error=True,
            )

        await loop.run_in_executor(None, run_server)

        self._running = True
        return self.get_url() or f"http://{self.config.host}:{self.config.port}"

    async def stop(self) -> None:
        """Stop Gradio app."""
        if self._app:
            self._app.close()
        self._running = False

    def get_url(self) -> Optional[str]:
        """Get Gradio URL."""
        if self._app and hasattr(self._app, 'local_url'):
            return self._app.local_url
        return f"http://{self.config.host}:{self.config.port}"


class StreamlitApp(DemoApp):
    """
    Streamlit-based demo application.

    Provides clean, modern UI for LLM demos.
    """

    def __init__(
        self,
        config: DemoConfig,
        generate_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(config)
        self.generate_fn = generate_fn
        self._streamlit = None

        self._try_import_streamlit()

    def _try_import_streamlit(self) -> None:
        """Try to import Streamlit."""
        try:
            import streamlit as st
            self._streamlit = st
            logger.info("Streamlit imported successfully")
        except ImportError:
            logger.warning("Streamlit not installed. Run: pip install streamlit")

    def create_ui(self) -> Any:
        """Create Streamlit UI code."""
        if not self._streamlit:
            raise RuntimeError("Streamlit not available")

        st = self._streamlit

        st.set_page_config(
            page_title=self.config.title,
            page_icon="🤖",
            layout="wide",
        )

        st.title(self.config.title)
        st.markdown(self.config.description)

        # Sidebar settings
        with st.sidebar:
            st.header("Settings")
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=self.config.temperature,
            )
            max_tokens = st.slider(
                "Max Tokens",
                min_value=64,
                max_value=8192,
                value=self.config.max_tokens,
            )

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Type your message..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                if self.generate_fn:
                    response = self.generate_fn(
                        prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    st.markdown(response)
                else:
                    response = f"Response to: {prompt}"
                    st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

        return st

    async def launch(self) -> str:
        """Launch Streamlit app."""
        # Streamlit needs to be run via CLI
        # Create app file
        app_code = self._generate_app_code()

        app_path = Path("streamlit_app.py")
        app_path.write_text(app_code)

        logger.info(f"Streamlit app created at {app_path}")
        logger.info(f"Run with: streamlit run {app_path}")

        self._running = True
        return f"http://{self.config.host}:{self.config.port}"

    def _generate_app_code(self) -> str:
        """Generate Streamlit app code."""
        return f'''
import streamlit as st

st.set_page_config(
    page_title="{self.config.title}",
    page_icon="🤖",
    layout="wide",
)

st.title("{self.config.title}")
st.markdown("{self.config.description}")

with st.sidebar:
    temperature = st.slider("Temperature", 0.0, 2.0, {self.config.temperature})
    max_tokens = st.slider("Max Tokens", 64, 8192, {self.config.max_tokens})

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({{"role": "user", "content": prompt}})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = f"Response to: {{prompt}}"
        st.markdown(response)
    st.session_state.messages.append({{"role": "assistant", "content": response}})
'''

    async def stop(self) -> None:
        """Stop Streamlit app."""
        self._running = False


class HuggingFaceSpaces(DemoApp):
    """
    Hugging Face Spaces deployment.

    Deploys demo to Hugging Face Spaces platform.
    """

    def __init__(
        self,
        config: DemoConfig,
        generate_fn: Optional[Callable] = None,
        space_id: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        super().__init__(config)
        self.generate_fn = generate_fn
        self.space_id = space_id
        self.token = token or os.getenv("HF_TOKEN")

        self._gradio_app: Optional[GradioApp] = None

    def create_ui(self) -> Any:
        """Create UI for Spaces."""
        self._gradio_app = GradioApp(self.config, self.generate_fn)
        return self._gradio_app.create_ui()

    async def launch(self) -> str:
        """Deploy to Hugging Face Spaces."""
        if not self.space_id:
            raise ValueError("space_id required for Hugging Face Spaces")

        if not self.token:
            raise ValueError("HF token required for deployment")

        # Create app files
        await self._create_space_files()

        # Deploy using huggingface_hub
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self.token)

            # Upload files
            api.upload_folder(
                folder_path="./space",
                repo_id=self.space_id,
                repo_type="space",
            )

            logger.info(f"Deployed to https://huggingface.co/spaces/{self.space_id}")
            self._running = True

            return f"https://huggingface.co/spaces/{self.space_id}"

        except ImportError:
            logger.warning("huggingface_hub not installed. Run: pip install huggingface_hub")
            return ""
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return ""

    async def _create_space_files(self) -> None:
        """Create files for Spaces deployment."""
        space_dir = Path("./space")
        space_dir.mkdir(exist_ok=True)

        # Create requirements.txt
        requirements = """
gradio>=4.0.0
huggingface_hub
"""
        (space_dir / "requirements.txt").write_text(requirements)

        # Create app.py
        app_code = self._generate_space_app_code()
        (space_dir / "app.py").write_text(app_code)

        # Create README.md
        readme = f"""---
title: {self.config.title}
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# {self.config.title}

{self.config.description}
"""
        (space_dir / "README.md").write_text(readme)

        logger.info(f"Created Space files in {space_dir}")

    def _generate_space_app_code(self) -> str:
        """Generate app code for Spaces."""
        return f'''
import gradio as gr

TITLE = "{self.config.title}"
DESCRIPTION = "{self.config.description}"

def respond(message, history, temperature, max_tokens):
    if not message:
        return "", history
    history = history + [(message, None)]
    response = f"Response to: {{message}}"
    history[-1] = (message, response)
    return "", history

with gr.Blocks() as demo:
    gr.Markdown(f"# {{TITLE}}")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=400)
            msg = gr.Textbox(label="Message", placeholder="Type your message...", lines=2)
            clear = gr.Button("Clear")

        with gr.Column(scale=1):
            temperature = gr.Slider(minimum=0, maximum=2, value={self.config.temperature}, label="Temperature")
            max_tokens = gr.Slider(minimum=64, maximum=8192, value={self.config.max_tokens}, label="Max Tokens")

    msg.submit(respond, [msg, chatbot, temperature, max_tokens], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
'''

    async def stop(self) -> None:
        """Stop Spaces deployment."""
        # Can't really stop - would need to delete the Space
        self._running = False

    def get_url(self) -> Optional[str]:
        """Get Spaces URL."""
        if self.space_id:
            return f"https://huggingface.co/spaces/{self.space_id}"
        return None


class DemoAppFactory:
    """Factory for creating demo applications."""

    @staticmethod
    def create(
        app_type: str,
        config: DemoConfig,
        generate_fn: Optional[Callable] = None,
        **kwargs: Any,
    ) -> DemoApp:
        """
        Create demo application.

        Args:
            app_type: Type of app (gradio, streamlit, spaces)
            config: App configuration
            generate_fn: Generation function
            **kwargs: Additional arguments

        Returns:
            Demo application
        """
        apps = {
            "gradio": GradioApp,
            "streamlit": StreamlitApp,
            "spaces": HuggingFaceSpaces,
        }

        app_class = apps.get(app_type.lower())
        if not app_class:
            raise ValueError(f"Unknown app type: {app_type}")

        return app_class(config, generate_fn, **kwargs)


def create_quick_demo(
    generate_fn: Callable,
    title: str = "Quick LLM Demo",
    description: str = "A quick demo of the LLM",
    app_type: str = "gradio",
) -> DemoApp:
    """
    Create a quick demo application.

    Args:
        generate_fn: Function that generates responses
        title: App title
        description: App description
        app_type: Type of app

    Returns:
        Demo application
    """
    config = DemoConfig(
        title=title,
        description=description,
    )

    return DemoAppFactory.create(app_type, config, generate_fn)
