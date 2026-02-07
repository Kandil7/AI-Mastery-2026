# JavaScript SDK Guide
# ===================
# JavaScript SDK for RAG Engine
# دليل JavaScript SDK لـ RAG Engine

## Overview / نظرة عامة

The JavaScript SDK provides a convenient interface for interacting with the RAG Engine REST API from Node.js and browser applications. It includes TypeScript types for full type safety.

توفر حزمة JavaScript SDK واجهة مريحة للتفاعل مع API RAG Engine من تطبيقات Node.js والمتصفح.

## Installation / التثبيت

### npm (Node.js)

```bash
npm install @ragengine/client
```

### yarn (Node.js)

```bash
yarn add @ragengine/client
```

### CDN (Browser)

```html
<script src="https://cdn.ragengine.com/sdk/v1/rag-engine.js"></script>
```

## Quick Start / البدء السريع

```javascript
import { RAGEngineClient } from '@ragengine/client';

// Initialize client
const client = new RAGEngineClient({
    apiUrl: 'https://api.ragengine.com/v1',
    apiKey: 'your-api-key',
    tenantId: 'your-tenant-id',
});

// Upload a document
const fileContent = await fs.promises.readFile('document.pdf');
const result = await client.uploadDocument({
    filename: 'document.pdf',
    content: fileContent,
    contentType: 'application/pdf',
});
console.log('Document uploaded:', result.documentId);

// Ask a question
const answer = await client.askQuestion({
    question: 'What is RAG?',
    k: 5,
});
console.log('Answer:', answer.text);
console.log('Sources:', answer.sources);
```

## Authentication / المصادقة

### API Key Authentication / مصادقة مفتاح API

```javascript
import { RAGEngineClient } from '@ragengine/client';

const client = new RAGEngineClient({
    apiUrl: 'https://api.ragengine.com/v1',
    apiKey: 'sk-xxxxxxxxxxxxx',
});
```

### OAuth Authentication / مصادقة OAuth

```javascript
import { RAGEngineClient } from '@ragengine/client';

const client = new RAGEngineClient({
    apiUrl: 'https://api.ragengine.com/v1',
    oauthToken: 'ya29.a0AfH6xxxxx',
});
```

## Documents / المستندات

### Upload Document / رفع مستند

```javascript
// Upload from file
import fs from 'fs';

const fileContent = await fs.promises.readFile('document.pdf');
const result = await client.uploadDocument({
    filename: 'document.pdf',
    content: fileContent,
    contentType: 'application/pdf',
});
console.log('Document ID:', result.documentId);
console.log('Status:', result.status); // 'queued', 'processing', 'indexed', 'failed'

// Upload from Buffer
const buffer = Buffer.from('Document content...');
const result = await client.uploadDocument({
    filename: 'document.txt',
    content: buffer,
    contentType: 'text/plain',
});
```

### List Documents / قائمة المستندات

```javascript
// List all documents
const documents = await client.listDocuments({
    limit: 20,
    offset: 0,
});

documents.forEach(doc => {
    console.log('ID:', doc.id);
    console.log('Filename:', doc.filename);
    console.log('Status:', doc.status);
    console.log('Created:', doc.createdAt);
    console.log('-'.repeat(40));
});

// Filter by status
const indexedDocs = await client.listDocuments({
    status: 'indexed',
    limit: 50,
});
```

### Get Document / الحصول على مستند

```javascript
// Get document by ID
const document = await client.getDocument('doc-123');

console.log('Filename:', document.filename);
console.log('Size:', document.sizeBytes);
console.log('Status:', document.status);
```

### Delete Document / حذف مستند

```javascript
const success = await client.deleteDocument('doc-123');
if (success) {
    console.log('Document deleted successfully');
}
```

### Bulk Operations / العمليات بالجملة

```javascript
// Bulk upload
const files = [
    { filename: 'doc1.pdf', content: Buffer.from('...'), contentType: 'application/pdf' },
    { filename: 'doc2.pdf', content: Buffer.from('...'), contentType: 'application/pdf' },
];
const results = await client.bulkUploadDocuments(files);

console.log('Success:', results.successCount);
console.log('Failures:', results.failureCount);

// Bulk delete
const success = await client.bulkDeleteDocuments([
    'doc-123',
    'doc-456',
    'doc-789',
]);
```

## Search / البحث

### Search Documents / البحث عن المستندات

```javascript
// Full-text search
const results = await client.searchDocuments({
    query: 'machine learning',
    k: 10,
    sortBy: 'created', // 'created', 'updated', 'filename', 'size'
});

console.log('Total results:', results.total);

results.results.forEach(doc => {
    console.log('Document:', doc.filename);
    console.log('Score:', doc.score);
});

// With faceted search
const results = await client.searchDocuments({
    query: 'PDF',
    k: 10,
    filters: {
        status: 'indexed',
        contentType: 'application/pdf',
    },
});
```

### Auto-Suggest / الاقتراح التلقائي

```javascript
// Get search suggestions
const suggestions = await client.getSearchSuggestions({
    query: 'vector',
    limit: 5,
    types: ['document', 'query'], // 'document', 'query', 'topic'
});

suggestions.forEach(suggestion => {
    console.log('Suggestion:', suggestion.text);
    console.log('Type:', suggestion.type);
    console.log('Relevance:', suggestion.relevanceScore);
});
```

## Chat / المحادثة

### Ask Question / طرح سؤال

```javascript
// Ask a question
const answer = await client.askQuestion({
    question: 'What is retrieval-augmented generation?',
    k: 5,
});
console.log('Answer:', answer.text);
console.log('Sources:', answer.sources);
console.log('Retrieval K:', answer.retrievalK);

// With performance metrics
const answer = await client.askQuestion({
    question: 'How does RAG work?',
    k: 10,
});

if (answer.embedMs) {
    console.log('Embedding time:', answer.embedMs, 'ms');
}
if (answer.searchMs) {
    console.log('Search time:', answer.searchMs, 'ms');
}
if (answer.llmMs) {
    console.log('LLM time:', answer.llmMs, 'ms');
}
```

### Create Chat Session / إنشاء جلسة محادثة

```javascript
// Create new session
const session = await client.createChatSession({
    title: 'RAG Architecture Discussion',
});
console.log('Session ID:', session.id);
console.log('Title:', session.title);
```

### List Chat Sessions / قائمة جلسات المحادثة

```javascript
// List sessions
const sessions = await client.listChatSessions({
    limit: 20,
    offset: 0,
});

sessions.forEach(session => {
    console.log('Session:', session.title);
    console.log('Created:', session.createdAt);
    console.log('ID:', session.id);
});
```

### Get Chat Session / الحصول على جلسة محادثة

```javascript
// Get session details
const session = await client.getChatSession('session-123');

console.log('Title:', session.title);
console.log('Turns:', session.turns ? session.turns.length : 0);
```

### Stream Answer / تدفق الإجابة

```javascript
// Stream answers in real-time
const stream = await client.askQuestionStream({
    question: 'Explain vector databases',
    k: 5,
});

for await (const chunk of stream) {
    process.stdout.write(chunk);
}
console.log();
```

## Exports / التصدير

### Export Documents / تصدير المستندات

```javascript
// Export to PDF
const pdfContent = await client.exportDocuments({
    format: 'pdf',
    documentIds: ['doc-123', 'doc-456'],
    title: 'My Documents',
});
fs.writeFileSync('documents.pdf', pdfContent);

// Export to CSV
const csvContent = await client.exportDocuments({
    format: 'csv',
    documentIds: ['doc-123', 'doc-456'],
});
fs.writeFileSync('documents.csv', csvContent);

// Export to JSON
const jsonContent = await client.exportDocuments({
    format: 'json',
    documentIds: ['doc-123', 'doc-456'],
});
fs.writeFileSync('documents.json', jsonContent);
```

### Export Chat Sessions / تصدير جلسات المحادثة

```javascript
// Export chat to Markdown
const mdContent = await client.exportChatSessions({
    format: 'markdown',
    sessionIds: ['session-123'],
});
fs.writeFileSync('chat.md', mdContent);
```

## A/B Testing / اختبار A/B

### Get Experiments / الحصول على التجارب

```javascript
// List experiments
const experiments = await client.listExperiments({
    status: 'active', // 'active', 'paused', 'completed'
    limit: 50,
});

experiments.forEach(exp => {
    console.log('Experiment:', exp.name);
    console.log('Status:', exp.status);
    console.log('Variants:', exp.variants.length);
});
```

### Assign Variant / تعيين نسخة

```javascript
// Assign user to variant
const assignment = await client.assignVariant({
    experimentId: 'exp-123',
    userId: 'user-456',
});
console.log('Assigned variant:', assignment.variantName);
console.log('Config:', assignment.variantConfig);
```

### Record Conversion / تسجيل تحويل

```javascript
// Record conversion event
await client.recordConversion({
    experimentId: 'exp-123',
    variantId: 'variant-A',
    success: true,
    value: 10.0, // Optional conversion value
});
```

## GraphQL Client / عميل GraphQL

```javascript
import { GraphQLClient } from '@ragengine/client';

// Initialize GraphQL client
const client = new GraphQLClient({
    apiUrl: 'https://api.ragengine.com/graphql',
    apiKey: 'your-api-key',
});

// Execute GraphQL query
const query = `
    query {
        documents(limit: 20) {
            id
            filename
            status
            sizeBytes
        }
    }
`;

const result = await client.execute(query);
const documents = result.data.documents;

// Execute mutation
const mutation = `
    mutation($question: String!) {
        askQuestion(question: $question, k: 5) {
            text
            sources
        }
    }
`;

const variables = { question: 'What is RAG?' };
const result = await client.execute(mutation, variables);

const answer = result.data.askQuestion;
console.log('Answer:', answer.text);
```

## React Integration / التكامل مع React

```javascript
import { RAGEngineProvider, useAskQuestion } from '@ragengine/react';

function App() {
    const askQuestion = useAskQuestion();

    const handleSubmit = async (e) => {
        e.preventDefault();
        const question = e.target.question.value;

        const result = await askQuestion(question);
        console.log('Answer:', result.text);
    };

    return (
        <RAGEngineClient
            apiKey="your-api-key"
            tenantId="your-tenant-id"
        >
            <form onSubmit={handleSubmit}>
                <input name="question" placeholder="Ask a question..." />
                <button type="submit">Ask</button>
            </form>
        </RAGEngineClient>
    );
}
```

## Error Handling / معالجة الأخطاء

```javascript
import { RAGEngineClient, RAGEngineError } from '@ragengine/client';

const client = new RAGEngineClient({
    apiUrl: 'https://api.ragengine.com/v1',
    apiKey: 'your-api-key',
});

try {
    const document = await client.getDocument('doc-123');
} catch (error) {
    if (error instanceof RAGEngineError) {
        console.error('Error:', error.message);
        console.error('Status Code:', error.statusCode);
        console.error('Request ID:', error.requestId);
    }
}

// Retry with exponential backoff
import { retry } from '@ragengine/client';

const getDocumentWithRetry = retry(
    async () => client.getDocument('doc-123'),
    { maxAttempts: 3, backoffFactor: 2 }
);

const document = await getDocumentWithRetry();
```

## Advanced Usage / الاستخدام المتقدم

### Custom Headers / رؤوس مخصصة

```javascript
const client = new RAGEngineClient({
    apiUrl: 'https://api.ragengine.com/v1',
    apiKey: 'your-api-key',
    headers: {
        'X-Custom-Header': 'value',
        'X-Request-ID': 'custom-id',
    },
});
```

### Timeout Configuration / تكوين المهلة الزمنية

```javascript
const client = new RAGEngineClient({
    apiUrl: 'https://api.ragengine.com/v1',
    apiKey: 'your-api-key',
    timeout: 30000, // milliseconds
    connectTimeout: 10000, // milliseconds
});
```

### Webhooks / خطافات الويب

```javascript
import { WebhookClient } from '@ragengine/client';

const webhookClient = new WebhookClient({
    webhookUrl: 'https://your-site.com/webhooks',
    secret: 'your-webhook-secret',
});

// Verify webhook signature
function handleWebhook(requestData, signature) {
    if (webhookClient.verifySignature(requestData, signature)) {
        console.log('Webhook verified');
        // Process webhook
    } else {
        console.log('Invalid webhook signature');
    }
}

// Register webhook
await client.registerWebhook({
    events: ['document.indexed', 'chat.updated'],
    url: 'https://your-site.com/webhooks',
});
```

## Browser Usage / الاستخدام في المتصفح

```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.ragengine.com/sdk/v1/rag-engine.js"></script>
</head>
<body>
    <input id="question" type="text" placeholder="Ask a question..." />
    <button onclick="askQuestion()">Ask</button>
    <div id="answer"></div>

    <script>
        const client = new RAGEngineClient({
            apiKey: 'your-api-key',
            tenantId: 'your-tenant-id',
        });

        async function askQuestion() {
            const question = document.getElementById('question').value;
            const result = await client.askQuestion(question);
            document.getElementById('answer').textContent = result.text;
        }
    </script>
</body>
</html>
```

## Best Practices / أفضل الممارسات

1. **Use Async/Await** - All SDK methods return Promises
2. **Implement Error Handling** - Always wrap API calls in try/catch
3. **Use Retry Logic** - Network failures are common; use exponential backoff
4. **Cache Results** - Reduce API calls by caching frequently accessed data
5. **Use Streaming** - For long responses, use streaming endpoints
6. **TypeScript Support** - Use TypeScript for full type safety
7. **Tenant Isolation** - Always specify tenantId for multi-tenant security

## Examples / أمثلة

### Complete RAG Pipeline / أنبوب RAG كامل

```javascript
import { RAGEngineClient } from '@ragengine/client';
import fs from 'fs';

const client = new RAGEngineClient({
    apiUrl: 'https://api.ragengine.com/v1',
    apiKey: 'your-api-key',
});

// Step 1: Upload document
const fileContent = await fs.promises.readFile('manual.pdf');
const doc = await client.uploadDocument({
    filename: 'manual.pdf',
    content: fileContent,
    contentType: 'application/pdf',
});

// Step 2: Wait for indexing
await new Promise(resolve => setTimeout(resolve, 5000));

// Step 3: Ask questions
const questions = [
    'What is the main topic?',
    'How do I use feature X?',
    'What are the limitations?',
];

for (const question of questions) {
    const answer = await client.askQuestion(question);
    console.log(`Q: ${question}`);
    console.log(`A: ${answer.text}\n`);
}
```

### Batch Processing / المعالجة بالجملة

```javascript
import fs from 'fs';
import path from 'path';

// Upload all PDFs in directory
const pdfFiles = fs.readdirSync('./documents')
    .filter(f => f.endsWith('.pdf'));

for (const filename of pdfFiles) {
    const fileContent = await fs.promises.readFile(path.join('./documents', filename));
    const result = await client.uploadDocument({
        filename,
        content: fileContent,
        contentType: 'application/pdf',
    });
    console.log(`Uploaded: ${filename} -> ${result.documentId}`);
}
```

## Support / الدعم

- **Documentation:** https://docs.ragengine.com
- **GitHub Issues:** https://github.com/your-org/rag-engine-mini/issues
- **Email:** support@ragengine.com

---

**JavaScript SDK Version:** 1.0.0
**Last Updated:** 2026-01-31
