# 🛡️ Privacy & Compliance in AI Systems

> How to build RAG systems that don't leak sensitive data.

---

## 🔒 1. The PII Risk

**PII (Personally Identifiable Information)** includes emails, phone numbers, addresses, and ID numbers. 
When a user asks a question like *"Can my refund be sent to john.doe@email.com?"*, sending this raw text to an external LLM (OpenAI, Anthropic) means your data is now processed by a third party. For many enterprises, this is a **Blocker**.

---

## 🛡️ 2. Redaction Strategy (The Shield)

In RAG Engine Mini, we implement a **Redaction-Restoration** loop in `PrivacyGuardService`.

### Step A: Redact (Before external call)
We use highly optimized Regular Expressions to find sensitive patterns.
*   **Original**: *"Contact me at +1-555-0199"*
*   **Redacted**: *"Contact me at <PHONE_0>"*

### Step B: The LLM Pass
The LLM receives the redacted text. Since it doesn't know the actual phone number, it cannot leak it or store it in its logs.
*   **Assistant**: *"I have noted that I should contact you at <PHONE_0>."*

### Step C: Restore (Before showing to user)
The system retrieves the original value from a local, secure dictionary and replaces the placeholder.
*   **Final Result**: *"I have noted that I should contact you at +1-555-0199."*

---

## 🏛️ 3. Multi-Tenancy Isolation

Privacy isn't just about LLMs. It's about ensuring User A never sees User B's data. 
We enforce this at every layer:
1.  **Postgres**: Every row has a `user_id` column.
2.  **Qdrant**: Every point has a `user_id` payload filter.
3.  **File Store**: Files are stored in directories named after the `tenant_id`.

---

## ⚖️ 4. Senior Tip: Compliance by Design

When designing AI products, follow the **Gold Standard**:
*   **Local-first**: Use local models (Ollama) if the data is extremely sensitive.
*   **Logs**: Scrub logs of any user inputs.
*   **Encryption**: Ensure all IDs are UUIDs (non-predictable).

---

> [!IMPORTANT]
> The redaction service in this project is a **Starting Point**. For production-grade compliance (GDPR/HIPAA), we recommend specialized libraries like **Microsoft Presidio** or **Amazon Macie**.
