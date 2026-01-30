# RAG Engine JavaScript SDK
# ==========================

/**
 * RAG Engine JavaScript/TypeScript SDK
 * =====================================
 * Official SDK for RAG Engine API (Node.js & Browser)
 *
 * RAG Engine JavaScript/TypeScript SDK الرسمي
 */

// Types
export enum DocumentStatus {
  CREATED = "created",
  INDEXED = "indexed",
  FAILED = "failed",
}

export enum QuerySortBy {
  CREATED = "created",
  UPDATED = "updated",
  FILENAME = "filename",
  SIZE = "size",
}

export interface Document {
  id: string;
  filename: string;
  contentType: string;
  sizeBytes: number;
  status: DocumentStatus;
  createdAt: string;
  updatedAt?: string;
}

export interface Answer {
  text: string;
  sources: string[];
  retrievalK?: number;
  embedMs?: number;
  searchMs?: number;
  llmMs?: number;
}

export interface QueryHistoryItem {
  question: string;
  answer: string;
  sources: string[];
  timestamp: string;
  success: boolean;
}

export interface Config {
  apiKey: string;
  baseUrl?: string;
  timeout?: number;
}

// Client Class
export class RAGClient {
  private apiKey: string;
  private baseUrl: string;
  private timeout: number;
  private headers: { [key: string]: string };

  constructor(config: Config) {
    this.apiKey = config.apiKey;
    this.baseUrl = (config.baseUrl || "https://api.rag-engine.com").replace(/\/$/, "");
    this.timeout = config.timeout || 30000;
    this.headers = {
      "Authorization": `Bearer ${this.apiKey}`,
      "Content-Type": "application/json",
    };
  }

  /**
   * Ask a question to the RAG engine.
   * @param question - Question to ask
   * @param k - Number of chunks to retrieve (default: 5)
   * @param documentId - Optional document ID for chat mode
   * @param expandQuery - Use query expansion (default: false)
   * @returns Answer with text and sources
   * 
   * طرح سؤال على محرك RAG
   */
  async ask(
    question: string,
    { k = 5, documentId, expandQuery = false }:
      {
        question: string;
        k?: number;
        documentId?: string;
        expandQuery?: boolean;
      }
  ): Promise<Answer> {
    const response = await this.fetch("/api/v1/ask", {
      method: "POST",
      body: JSON.stringify({
        question,
        k,
        document_id: documentId,
        expand_query: expandQuery,
      }),
    });

    const data = await response.json();

    return {
      text: data.answer,
      sources: data.sources || [],
      retrievalK: data.retrieval_k,
      embedMs: data.embed_ms,
      searchMs: data.search_ms,
      llmMs: data.llm_ms,
    };
  }

  /**
   * Upload a document to the RAG engine.
   * @param filePath - Path or File object to upload
   * @param filename - Optional custom filename
   * @returns Document metadata
   * 
   * رفع مستند إلى محرك RAG
   */
  async uploadDocument(
    filePath: string | File,
    { filename }:
      {
        filePath: string | File;
        filename?: string;
      }
  ): Promise<Document> {
    let file: File;

    if (typeof filePath === "string") {
      // Node.js: Read file from path
      const fs = await import("fs/promises");
      const buffer = await fs.readFile(filePath as string);
      file = new File([buffer], filename || (filePath as string).split("/").pop()!);
    } else {
      // Browser: File object provided directly
      file = filePath as File;
    }

    const formData = new FormData();
    formData.append("file", file);

    const response = await this.fetch("/api/v1/documents", {
      method: "POST",
      body: formData,
      headers: {}, // Let browser set Content-Type with boundary
    });

    const data = await response.json();

    return {
      id: data.id,
      filename: data.filename,
      contentType: data.content_type,
      sizeBytes: data.size_bytes,
      status: data.status as DocumentStatus,
      createdAt: data.created_at,
      updatedAt: data.updated_at,
    };
  }

  /**
   * Search for documents.
   * @param query - Search query
   * @param k - Number of results (default: 10)
   * @param filters - Optional filters (status, type, date, size)
   * @param sortBy - Sort order (default: created)
   * @param limit - Max results (default: 20)
   * @param offset - Pagination offset (default: 0)
   * @returns List of matching documents
   * 
   * البحث عن المستندات
   */
  async searchDocuments(
    query: string,
    {
      k = 10,
      filters,
      sortBy = QuerySortBy.CREATED,
      limit = 20,
      offset = 0,
    }:
      {
        query: string;
        k?: number;
        filters?: { [key: string]: any };
        sortBy?: QuerySortBy;
        limit?: number;
        offset?: number;
      }
  ): Promise<Document[]> {
    const params = new URLSearchParams({
      query,
      k: String(k),
      sort_by: sortBy,
      limit: String(limit),
      offset: String(offset),
    });

    if (filters) {
      Object.entries(filters).forEach(([key, value]) => {
        params.append(key, String(value));
      });
    }

    const response = await this.fetch(`/api/v1/documents/search?${params.toString()}`, {
      method: "GET",
    });

    const data = await response.json();

    return data.results.map((d: any) => ({
      id: d.id,
      filename: d.filename,
      contentType: d.content_type,
      sizeBytes: d.size_bytes,
      status: d.status as DocumentStatus,
      createdAt: d.created_at,
      updatedAt: d.updated_at,
    }));
  }

  /**
   * Delete a document.
   * @param documentId - ID of document to delete
   * @returns True if successful
   * 
   * حذف مستند
   */
  async deleteDocument(documentId: string): Promise<boolean> {
    const response = await this.fetch(`/api/v1/documents/${documentId}`, {
      method: "DELETE",
    });

    return response.ok;
  }

  /**
   * Get query history.
   * @param limit - Max history items (default: 50)
   * @param offset - Pagination offset (default: 0)
   * @returns List of query history items
   * 
   * الحصول على سجل الاستعلامات
   */
  async getQueryHistory(
    { limit = 50, offset = 0 }:
      {
        limit?: number;
        offset?: number;
      }
  ): Promise<QueryHistoryItem[]> {
    const params = new URLSearchParams({
      limit: String(limit),
      offset: String(offset),
    });

    const response = await this.fetch(`/api/v1/queries/history?${params.toString()}`, {
      method: "GET",
    });

    const data = await response.json();

    return data.questions.map((q: any) => ({
      question: q.question,
      answer: q.answer || "",
      sources: q.sources || [],
      timestamp: q.timestamp,
      success: q.success ?? true,
    }));
  }

  /**
   * Helper method to make HTTP requests.
   * @param endpoint - API endpoint
   * @param options - Request options
   * @returns Response object
   */
  private async fetch(
    endpoint: string,
    options: {
      method?: string;
      body?: BodyInit | null;
      headers?: { [key: string]: string };
    }
  ): Promise<Response> {
    const url = `${this.baseUrl}${endpoint}`;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        ...options,
        headers: { ...this.headers, ...options.headers },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  }
}

// React Hook
export function useRAGClient(config: Config) {
  const client = React.useMemo(() => new RAGClient(config), [config.apiKey]);

  return client;
}

// Example Usage
// Node.js
async function exampleNode() {
  const client = new RAGClient({
    apiKey: "sk_your_api_key_here",
    baseUrl: "http://localhost:8000",
  });

  // Ask a question
  const answer = await client.ask("What is RAG?", { k: 5 });
  console.log("Answer:", answer.text);
  console.log("Sources:", answer.sources);

  // Upload a document
  const doc = await client.uploadDocument("./test.pdf");
  console.log("Document ID:", doc.id);

  // Search documents
  const results = await client.searchDocuments("vector search");
  console.log("Results:", results.length);

  // Get query history
  const history = await client.getQueryHistory();
  console.log("History:", history.length);
}

// React Example
function RAGComponent() {
  const client = useRAGClient({
    apiKey: "sk_your_api_key_here",
  });

  const [question, setQuestion] = React.useState("");
  const [answer, setAnswer] = React.useState<Answer | null>(null);
  const [loading, setLoading] = React.useState(false);

  const handleAsk = async () => {
    setLoading(true);
    try {
      const result = await client.ask(question, { k: 5 });
      setAnswer(result);
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>RAG Chat</h1>
      <input
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask a question..."
      />
      <button onClick={handleAsk} disabled={loading}>
        {loading ? "Loading..." : "Ask"}
      </button>
      {answer && (
        <div>
          <h2>Answer</h2>
          <p>{answer.text}</p>
          <h3>Sources</h3>
          <ul>
            {answer.sources.map((source) => (
              <li key={source}>{source}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default RAGClient;
