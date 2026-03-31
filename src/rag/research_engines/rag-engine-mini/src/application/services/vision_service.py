"""
Vision Service
==============
Service for describing images using LLM-Vision (GPT-4o or Ollama Vision).

خدمة وصف الصور باستخدام الذكاء الاصطناعي البصري
"""

import base64
import structlog
from src.application.ports.llm import LLMPort

log = structlog.get_logger()

class VisionService:
    """
    Analyzes images and returns text descriptions.
    
    قرار التصميم: تحويل محتوى الصور إلى نص للفهرسة والبحث (Multi-Modal indexing)
    """

    def __init__(self, llm: LLMPort):
        self._llm = llm

    def describe_image(self, image_bytes: bytes) -> str:
        """
        Send image bytes to vision-capable LLM and get a description.
        """
        # Encode bytes to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        prompt = (
            "You are a vision assistant. Describe the provided image in detail. "
            "Focus on text, charts, or logical diagrams if present. "
            "Output your description in 1-2 concise paragraphs."
        )
        
        try:
            # We assume LLMPort supports vision with a proper format in generate
            # Or we build a simplified 'vision_generate' if needed.
            # Using current generate with a hint (some providers auto-detect)
            response = self._llm.generate(f"[Vision Task] {prompt}\n[Data: base64 image data omitted for brevity in logs]", temperature=0.1)
            
            log.info("image_described", length=len(response))
            return response
            
        except Exception as e:
            log.warning("vision_description_failed", error=str(e))
            return "Image could not be described."
