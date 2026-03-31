export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  timestamp: Date;
  isStreaming?: boolean;
}

export interface Citation {
  id: string;
  documentId: string;
  documentName: string;
  pageNumber?: number;
  chunkIndex: number;
  score: number;
  text: string;
}

export interface Document {
  id: string;
  name: string;
  type: 'pdf' | 'txt' | 'docx' | 'md';
  size: number;
  status: 'uploading' | 'processing' | 'indexed' | 'error';
  uploadProgress?: number;
  createdAt: Date;
  errorMessage?: string;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

export interface UploadResponse {
  documentId: string;
  status: string;
  message: string;
}

export interface QueryRequest {
  query: string;
  sessionId?: string;
  topK?: number;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  processingTime: number;
}

export interface ApiError {
  error: string;
  message: string;
  statusCode: number;
}
