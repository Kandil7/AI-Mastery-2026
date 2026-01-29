"""
Graph Extractor Service
========================
Extracts entities and relationships from text for Knowledge Graph RAG.

خدمة استخراج الكيانات والعلاقات لبناء رسم بياني للمعرفة
"""

import json
import structlog
from typing import List, TypedDict
from src.application.ports.llm import LLMPort

log = structlog.get_logger()

class GraphTriplet(TypedDict):
    subject: str
    relation: str
    obj: str

class GraphExtractorService:
    """
    Extracts knowledge triplets (S, R, O) from text.
    
    قرار التصميم: بناء رسم بياني مبسط يسمح بربط المعلومات عبر المستندات
    """

    def __init__(self, llm: LLMPort):
        self._llm = llm

    def extract_triplets(self, text: str) -> List[GraphTriplet]:
        """
        Extract knowledge triplets from the provided text.
        """
        prompt = (
            "Extract key entities and their relationships from the text below.\n"
            "Format the output as a JSON list of objects with keys: 'subject', 'relation', 'obj'.\n"
            "Keep entities concise (1-3 words).\n\n"
            f"Text: {text[:4000]}\n\n"
            "JSON Output:"
        )
        
        try:
            response = self._llm.generate(prompt, temperature=0.1)
            
            # Clean possible markdown formatting
            clean_json = response.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:-3].strip()
            elif clean_json.startswith("```"):
                clean_json = clean_json[3:-3].strip()
                
            triplets = json.loads(clean_json)
            
            # Basic validation
            valid_triplets = []
            for t in triplets:
                if all(k in t for k in ("subject", "relation", "obj")):
                    valid_triplets.append(t)
            
            log.info("graph_triplets_extracted", count=len(valid_triplets))
            return valid_triplets
            
        except Exception as e:
            log.warning("graph_extraction_failed", error=str(e))
            return []
