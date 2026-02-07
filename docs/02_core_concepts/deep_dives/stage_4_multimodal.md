# 🖼️ Stage 4: Multi-Modal & Structural RAG

> Solving the "Hard Parts" of RAG: Tables and Images.

---

## 1. Structural Table Parsing

**The Problem**: Tables are designed for human eyes, not LLMs. In a raw PDF extract, a table row might look like: `Product A 100 Delivered 2024`. The LLM might not know if `100` is the price, quantity, or id.

**Our Solution**:
1.  **Detection**: We use `PyMuPDF`'s structural detection to find table boundaries.
2.  **Conversion**: We convert the table data into **Markdown**.
    *   *Result*: `| Product | Price | Status | Date |`
3.  **Context**: The LLM receives the table in a format it was trained on (Markdown / CSV), doubling the accuracy on financial/technical data.

---

## 2. Vision-Enhanced Indexing (Multi-Modal)

**The Problem**: "A picture is worth a thousand words," but standard RAG is blind. If a document has a diagram of a software architecture, standard RAG skips it.

**Our Solution**:
1.  **Extraction**: During indexing, we scan every page for images.
2.  **Description**: We send the image bytes to a **Vision LLM** (e.g., GPT-4o-mini or Llama-3-Vision).
3.  **Indexing**: The vision model returns a text description (e.g., *"A diagram showing a microservices architecture with a Gateway and 3 backend nodes"*).
4.  **Search**: We index this text description as a "Virtual Chunk". 

**Benefit**: You can now search for "architecture diagram" and find the exact page, even if the word "architecture" wasn't in the page text!

---

## 🛠️ How it works in RAG Engine Mini

These features are powered by:
-   `src/application/services/vision_service.py`
-   The updated `DefaultTextExtractor` with `pandas` integration.

---

## 📚 Further Learning
- [Unstructured.io: Handling Complex PDF Structures](https://unstructured.io/)
- [OpenAI: Vision API Documentation](https://platform.openai.com/docs/guides/vision)
