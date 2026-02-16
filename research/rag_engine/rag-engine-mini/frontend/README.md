# RAG Engine Mini - Frontend

A modern React/Next.js frontend for the RAG Engine Mini - chat with your documents using AI-powered retrieval.

## Features

- **Real-time Chat**: Streamed responses with source citations
- **Document Management**: Upload, view, and delete documents
- **File Upload**: Drag-and-drop support for PDF, TXT, DOCX, and MD files
- **Source Citations**: Expandable citations showing document sources with relevance scores
- **Responsive Design**: Works on desktop and mobile
- **TypeScript**: Full type safety throughout

## Project Structure

```
frontend/
├── src/
│   ├── app/
│   │   ├── globals.css      # Tailwind + custom styles
│   │   ├── layout.tsx       # Root layout
│   │   └── page.tsx         # Main page
│   ├── components/
│   │   ├── ChatWindow.tsx   # Main chat interface
│   │   ├── DocumentList.tsx # Document list with status
│   │   ├── UploadModal.tsx  # File upload modal
│   │   ├── MessageBubble.tsx # Chat message component
│   │   └── Citation.tsx     # Source citation component
│   ├── lib/
│   │   └── api.ts           # API client
│   └── types/
│       └── index.ts         # TypeScript types
├── package.json
├── tailwind.config.js
├── next.config.js
└── tsconfig.json
```

## Components

### ChatWindow
Main chat interface with:
- Message history display
- Streaming responses (real-time)
- Source citations
- Input area with send button
- Loading indicators
- Suggested prompts for empty state

### DocumentList
Document management with:
- List of uploaded documents
- Status indicators (uploading, processing, indexed, error)
- File type icons
- File size formatting
- Delete functionality
- Progress bars for uploads

### UploadModal
File upload modal with:
- Drag-and-drop support
- Multiple file upload
- Progress tracking
- Error handling
- Supported formats: PDF, TXT, DOCX, MD

### MessageBubble
Individual chat message:
- User/assistant styling
- Timestamps
- Streaming indicators
- Error states
- Citation attachments

### Citation
Source citation component:
- Document name and page number
- Relevance score (color-coded)
- Expandable text preview
- Match percentage display

## Getting Started

### Prerequisites
- Node.js 18+ 
- Backend API running (default: http://localhost:8000)

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### Environment Variables

Create `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## API Integration

The frontend communicates with the backend via:
- `GET /api/documents` - List all documents
- `POST /api/documents/upload` - Upload a file
- `DELETE /api/documents/:id` - Delete a document
- `POST /api/query` - Query with streaming response
- `POST /api/query/stream` - Streaming query endpoint

## Technologies

- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- Lucide React (icons)
- React Dropzone (file upload)

## Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript checks
