# Database Integration with AI/ML Platforms Tutorial

## Overview

This tutorial focuses on integrating databases with modern AI/ML platforms: Hugging Face, LangChain, and LLM-based systems. We'll cover vector database integration, RAG systems, fine-tuning workflows, and production deployment patterns specifically for senior AI/ML engineers.

## Prerequisites
- Python 3.8+
- Hugging Face Transformers 4.30+
- LangChain 0.1+
- Vector databases (Chroma, FAISS, Pinecone, or PostgreSQL with pgvector)
- Basic understanding of LLMs and RAG systems

## Tutorial Structure
1. **Vector Database Integration** - Embedding storage and retrieval
2. **RAG System Implementation** - Database-backed retrieval augmentation
3. **Hugging Face Integration** - Fine-tuning with database data
4. **LangChain Integration** - Database-aware LLM chains
5. **Production Deployment** - From prototype to production
6. **Performance Benchmarking** - Measuring AI-platform integration

## Section 1: Vector Database Integration

### Step 1: Vector database setup with PostgreSQL/pgvector
```python
import psycopg2
import numpy as np
from typing import List, Dict, Optional
import torch
from sentence_transformers import SentenceTransformer

class VectorDatabaseManager:
    def __init__(self, db_config: Dict, embedding_model: str = "all-MiniLM-L6-v2"):
        self.db_config = db_config
        self.embedding_model = SentenceTransformer(embedding_model)
        self._initialize_vector_extension()
    
    def _initialize_vector_extension(self):
        """Initialize pgvector extension"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        conn.close()
    
    def create_vector_table(self, table_name: str, 
                          metadata_fields: List[str] = None):
        """Create vector table with metadata"""
        if metadata_fields is None:
            metadata_fields = ["document_id", "source", "timestamp"]
        
        # Build column definitions
        columns = ["id SERIAL PRIMARY KEY"]
        columns.append("embedding vector(384)")  # Default dimension for all-MiniLM-L6-v2
        
        for field in metadata_fields:
            columns.append(f"{field} TEXT")
        
        # Create table
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {', '.join(columns)}
        );
        
        -- Create index for vector similarity search
        CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding 
        ON {table_name} USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        """
        
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit()
        conn.close()
        
        return f"Vector table '{table_name}' created successfully"
    
    def embed_and_store_documents(self, documents: List[str], 
                                table_name: str, 
                                metadata: List[Dict] = None):
        """Embed documents and store in vector database"""
        if metadata is None:
            metadata = [{} for _ in range(len(documents))]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        
        # Prepare insert statements
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_sql = f"""
        INSERT INTO {table_name} (embedding, document_id, source, timestamp)
        VALUES (%s, %s, %s, %s)
        """
        
        for i, (doc, emb, meta) in enumerate(zip(documents, embeddings, metadata)):
            # Convert embedding to list for PostgreSQL
            emb_list = emb.tolist()
            
            # Extract metadata
            doc_id = meta.get('document_id', f'doc_{i}')
            source = meta.get('source', 'unknown')
            timestamp = meta.get('timestamp', 'NOW()')
            
            cursor.execute(insert_sql, (emb_list, doc_id, source, timestamp))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return f"Stored {len(documents)} documents in '{table_name}'"
    
    def similarity_search(self, query: str, table_name: str, 
                         k: int = 5, threshold: float = 0.7):
        """Perform similarity search"""
        # Embed query
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        search_sql = f"""
        SELECT 
            id,
            document_id,
            source,
            timestamp,
            1 - (embedding <=> %s) as similarity
        FROM {table_name}
        WHERE 1 - (embedding <=> %s) > %s
        ORDER BY embedding <=> %s
        LIMIT %s;
        """
        
        cursor.execute(search_sql, (query_embedding, query_embedding, threshold, query_embedding, k))
        results = cursor.fetchall()
        
        conn.close()
        
        return [
            {
                'id': row[0],
                'document_id': row[1],
                'source': row[2],
                'timestamp': row[3],
                'similarity': float(row[4])
            }
            for row in results
        ]

# Usage example
db_config = {
    'host': 'localhost',
    'database': 'ai_db',
    'user': 'postgres',
    'password': 'password'
}

vector_db = VectorDatabaseManager(db_config)

# Create vector table
vector_db.create_vector_table("knowledge_base")

# Store documents
documents = [
    "The user engagement score is calculated as 0.4*avg_clicks + 0.6*avg_time_spent",
    "Database normalization reduces redundancy and improves data integrity",
    "ACID properties ensure transaction reliability in relational databases",
    "CAP theorem states that distributed systems can only guarantee two of consistency, availability, partition tolerance"
]

metadata = [
    {'document_id': 'engagement_calc', 'source': 'training_docs', 'timestamp': '2024-01-01'},
    {'document_id': 'normalization_guide', 'source': 'database_docs', 'timestamp': '2024-01-02'},
    {'document_id': 'acid_properties', 'source': 'system_design', 'timestamp': '2024-01-03'},
    {'document_id': 'cap_theorem', 'source': 'distributed_systems', 'timestamp': '2024-01-04'}
]

vector_db.embed_and_store_documents(documents, "knowledge_base", metadata)

# Search
results = vector_db.similarity_search("How do you calculate user engagement?", "knowledge_base")
print("Search results:", results)
```

### Step 2: FAISS vector database integration
```python
import faiss
import numpy as np
import pickle
from typing import List, Dict

class FAISSVectorDatabase:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.metadata = []
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to FAISS index"""
        if metadata is None:
            metadata = [{} for _ in range(len(documents))]
        
        # Generate embeddings (using simple random for demo, replace with actual model)
        embeddings = np.random.rand(len(documents), self.dimension).astype('float32')
        
        # Add to index
        self.index.add(embeddings)
        
        # Store documents and metadata
        self.documents.extend(documents)
        self.metadata.extend(metadata)
    
    def search(self, query_embedding: np.ndarray, k: int = 5):
        """Search in FAISS index"""
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'distance': float(distances[0][i]),
                    'score': 1.0 / (1.0 + float(distances[0][i]))
                })
        
        return results
    
    def save_to_disk(self, filename: str):
        """Save FAISS index to disk"""
        faiss.write_index(self.index, f"{filename}.index")
        
        # Save documents and metadata
        with open(f"{filename}.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
    
    @classmethod
    def load_from_disk(cls, filename: str):
        """Load FAISS index from disk"""
        index = faiss.read_index(f"{filename}.index")
        with open(f"{filename}.pkl", 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(index.d)
        instance.index = index
        instance.documents = data['documents']
        instance.metadata = data['metadata']
        
        return instance

# Usage example
faiss_db = FAISSVectorDatabase()

# Add documents
documents = [
    "User engagement calculation formula",
    "Database normalization principles",
    "ACID transaction properties",
    "CAP theorem explanation"
]

faiss_db.add_documents(documents)

# Search
query_embedding = np.random.rand(384).astype('float32')
results = faiss_db.search(query_embedding, k=3)
print("FAISS search results:", results)
```

## Section 2: RAG System Implementation

### Step 1: Database-backed RAG system
```python
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class DatabaseRAGSystem:
    def __init__(self, vector_db: VectorDatabaseManager, 
                 llm_model: str = "gpt2", max_context_length: int = 512):
        self.vector_db = vector_db
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(llm_model)
        self.max_context_length = max_context_length
    
    def retrieve_context(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant context from vector database"""
        return self.vector_db.similarity_search(query, "knowledge_base", k=k)
    
    def format_prompt(self, query: str, context: List[Dict]) -> str:
        """Format prompt with retrieved context"""
        # Format context
        context_text = ""
        for i, item in enumerate(context):
            context_text += f"Context {i+1}: {item['document']}\n"
        
        # Format prompt
        prompt = f"""You are an AI assistant that answers questions based on provided context.
Use only the information from the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context_text}

Question: {query}
Answer:"""
        
        return prompt
    
    def generate_answer(self, query: str, k: int = 3) -> Dict:
        """Generate answer using RAG approach"""
        # Retrieve context
        context = self.retrieve_context(query, k=k)
        
        # Format prompt
        prompt = self.format_prompt(query, context)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                               max_length=self.max_context_length)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9
            )
        
        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'query': query,
            'context_used': context,
            'answer': answer,
            'generated_at': torch.datetime.now().isoformat()
        }

# Usage example
rag_system = DatabaseRAGSystem(vector_db)

# Query
result = rag_system.generate_answer("How do you calculate user engagement?")
print("RAG Answer:", result['answer'])
```

### Step 2: Advanced RAG with database filtering
```python
class AdvancedDatabaseRAG:
    def __init__(self, vector_db: VectorDatabaseManager, 
                 llm_model: str = "gpt2"):
        self.vector_db = vector_db
        self.llm_model = llm_model
    
    def hybrid_retrieval(self, query: str, 
                        metadata_filters: Dict = None,
                        k: int = 5) -> List[Dict]:
        """Hybrid retrieval: vector + metadata filtering"""
        # First, get vector search results
        vector_results = self.vector_db.similarity_search(query, "knowledge_base", k=k*2)
        
        # Apply metadata filters
        if metadata_filters:
            filtered_results = []
            for result in vector_results:
                match = True
                for key, value in metadata_filters.items():
                    if result.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_results.append(result)
            
            # If not enough results, fall back to vector results
            if len(filtered_results) < k:
                filtered_results.extend(vector_results[:k-len(filtered_results)])
            
            return filtered_results[:k]
        else:
            return vector_results[:k]
    
    def multi_hop_rag(self, query: str, max_hops: int = 2) -> Dict:
        """Multi-hop RAG: chain multiple retrievals"""
        current_query = query
        all_context = []
        
        for hop in range(max_hops):
            # Retrieve context
            context = self.hybrid_retrieval(current_query, k=3)
            
            # Add to all context
            all_context.extend(context)
            
            # Generate intermediate answer to refine query
            if hop < max_hops - 1:
                # Simple query refinement (in practice, use LLM)
                refined_query = f"What is the relationship between '{current_query}' and '{context[0]['document'][:50]}...'"
                current_query = refined_query
        
        # Final answer generation
        final_answer = self._generate_final_answer(query, all_context)
        
        return {
            'original_query': query,
            'hops': max_hops,
            'context_used': all_context,
            'final_answer': final_answer
        }
    
    def _generate_final_answer(self, query: str, context: List[Dict]) -> str:
        """Generate final answer (simplified)"""
        # In practice, this would use an LLM
        context_summary = "\n".join([f"- {item['document']}" for item in context[:3]])
        return f"Based on the context:\n{context_summary}\n\nThe answer to '{query}' is related to these concepts."

# Usage example
advanced_rag = AdvancedDatabaseRAG(vector_db)

# Multi-hop RAG
result = advanced_rag.multi_hop_rag("How does database normalization affect user engagement metrics?")
print("Multi-hop RAG result:", result['final_answer'])
```

## Section 3: Hugging Face Integration

### Step 1: Fine-tuning with database data
```python
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import Dataset
import pandas as pd
import torch

class DatabaseHFTrainer:
    def __init__(self, db_config: Dict, model_name: str = "bert-base-uncased"):
        self.db_config = db_config
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def load_dataset_from_database(self, query: str, 
                                 text_column: str, 
                                 label_column: str) -> Dataset:
        """Load dataset directly from database"""
        conn = psycopg2.connect(**self.db_config)
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Convert to Hugging Face dataset
        dataset_dict = {
            'text': df[text_column].tolist(),
            'label': df[label_column].tolist()
        }
        
        return Dataset.from_dict(dataset_dict)
    
    def preprocess_function(self, examples):
        """Preprocess function for tokenization"""
        return self.tokenizer(examples['text'], truncation=True, padding=True)
    
    def train_model(self, train_query: str, val_query: str,
                   text_column: str, label_column: str,
                   output_dir: str = "./results"):
        """Train model using database data"""
        # Load datasets
        train_dataset = self.load_dataset_from_database(train_query, text_column, label_column)
        val_dataset = self.load_dataset_from_database(val_query, text_column, label_column)
        
        # Preprocess
        train_dataset = train_dataset.map(self.preprocess_function, batched=True)
        val_dataset = val_dataset.map(self.preprocess_function, batched=True)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=len(set(train_dataset['label']))
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model(output_dir)
        
        return trainer

# Usage example
hf_trainer = DatabaseHFTrainer(db_config)

# Train model
train_query = """
SELECT text, label FROM training_data 
WHERE split = 'train' AND text IS NOT NULL AND label IS NOT NULL
"""

val_query = """
SELECT text, label FROM training_data 
WHERE split = 'val' AND text IS NOT NULL AND label IS NOT NULL
"""

trainer = hf_trainer.train_model(
    train_query, val_query,
    text_column="text",
    label_column="label",
    output_dir="./user_engagement_classifier"
)
```

### Step 2: Database-backed inference
```python
class DatabaseHFInference:
    def __init__(self, model_path: str, db_config: Dict):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.db_config = db_config
    
    def predict_from_database(self, query: str, text_column: str):
        """Make predictions on database data"""
        conn = psycopg2.connect(**self.db_config)
        df = pd.read_sql(query, conn)
        conn.close()
        
        texts = df[text_column].tolist()
        
        # Batch prediction
        predictions = []
        batch_size = 16
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(batch_texts, return_tensors="pt", 
                                  truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # Get predictions
                batch_predictions = torch.argmax(probs, dim=-1).cpu().numpy()
                batch_probs = probs.cpu().numpy()
                
                predictions.extend([
                    {
                        'text': text,
                        'prediction': pred,
                        'confidence': float(prob[pred]),
                        'probabilities': prob.tolist()
                    }
                    for text, pred, prob in zip(batch_texts, batch_predictions, batch_probs)
                ])
        
        return predictions
    
    def save_predictions_to_database(self, predictions: List[Dict], 
                                   table_name: str = "predictions_log"):
        """Save predictions to database"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_sql = f"""
        INSERT INTO {table_name} (text, prediction, confidence, probabilities, created_at)
        VALUES (%s, %s, %s, %s, NOW())
        """
        
        for pred in predictions:
            cursor.execute(insert_sql, (
                pred['text'],
                pred['prediction'],
                pred['confidence'],
                json.dumps(pred['probabilities'])
            ))
        
        conn.commit()
        cursor.close()
        conn.close()

# Usage example
hf_inference = DatabaseHFInference("./user_engagement_classifier", db_config)

# Predict on new data
predict_query = """
SELECT text FROM new_data WHERE processed = false LIMIT 100
"""

predictions = hf_inference.predict_from_database(predict_query, "text")
hf_inference.save_predictions_to_database(predictions)
```

## Section 4: LangChain Integration

### Step 1: Database-aware LangChain agents
```python
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

class DatabaseLangChainAgent:
    def __init__(self, db_config: Dict, vector_db_type: str = "pgvector"):
        self.db_config = db_config
        self.vector_db_type = vector_db_type
        self.llm = None
        self.retriever = None
    
    def setup_llm(self, model_name: str = "gpt2"):
        """Set up LLM for LangChain"""
        from transformers import pipeline
        from langchain_community.llms import HuggingFacePipeline
        
        hf_pipeline = pipeline("text-generation", model=model_name, max_new_tokens=100)
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    def setup_vector_retriever(self, collection_name: str = "knowledge_base"):
        """Set up vector retriever"""
        if self.vector_db_type == "pgvector":
            # Use custom vector DB manager
            self.retriever = CustomPGVectorRetriever(self.db_config, collection_name)
        elif self.vector_db_type == "chroma":
            # Use Chroma
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.retriever = Chroma(
                embedding_function=embeddings,
                collection_name=collection_name
            )
    
    def create_database_aware_chain(self, prompt_template: str):
        """Create chain that uses database context"""
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )
        
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=True
        )
        
        return chain
    
    def run_database_aware_query(self, question: str, k: int = 3):
        """Run query with database context"""
        # Retrieve context
        if self.retriever:
            context_docs = self.retriever.get_relevant_documents(question, k=k)
            context = "\n".join([doc.page_content for doc in context_docs])
        else:
            context = ""
        
        # Run chain
        chain = self.create_database_aware_chain(
            """You are an AI assistant that answers questions based on provided context.
Use only the information from the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""
        )
        
        result = chain.run({"context": context, "question": question})
        
        return {
            'question': question,
            'context_used': context,
            'answer': result,
            'retrieved_docs_count': len(context_docs) if self.retriever else 0
        }

# Custom retriever for pgvector
class CustomPGVectorRetriever:
    def __init__(self, db_config: Dict, collection_name: str):
        self.db_config = db_config
        self.collection_name = collection_name
        self.vector_db = VectorDatabaseManager(db_config)
    
    def get_relevant_documents(self, query: str, k: int = 3):
        """Get relevant documents from pgvector"""
        results = self.vector_db.similarity_search(query, self.collection_name, k=k)
        
        from langchain_core.documents import Document
        documents = []
        for result in results:
            doc = Document(
                page_content=result['document'],
                metadata={
                    'id': result['id'],
                    'document_id': result['document_id'],
                    'source': result['source'],
                    'similarity': result['similarity']
                }
            )
            documents.append(doc)
        
        return documents
```

### Step 2: Production-ready LangChain pipeline
```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
import os

class ProductionLangChainPipeline:
    def __init__(self, db_config: Dict, api_key: str = None):
        self.db_config = db_config
        self.api_key = api_key
        self.setup_components()
    
    def setup_components(self):
        """Set up all components"""
        # Vector database
        self.vector_db = VectorDatabaseManager(self.db_config)
        
        # LLM (using OpenAI for production)
        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        else:
            # Fallback to local model
            from langchain_community.llms import HuggingFacePipeline
            from transformers import pipeline
            hf_pipeline = pipeline("text-generation", model="gpt2", max_new_tokens=100)
            self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    def create_rag_chain(self):
        """Create production RAG chain"""
        # Prompt template
        template = """You are an AI assistant that answers questions based on provided context.
Use only the information from the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        
        # Retrieval function
        def retrieve_context(question: str):
            return self.vector_db.similarity_search(question, "knowledge_base", k=3)
        
        # Format context
        def format_docs(docs):
            return "\n\n".join([f"Document {i+1}: {doc['document']}" for i, doc in enumerate(docs)])
        
        # Chain
        rag_chain = (
            {"context": lambda x: format_docs(retrieve_context(x["question"])), "question": lambda x: x["question"]}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def deploy_api(self, port: int = 8000):
        """Deploy as REST API"""
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        import uvicorn
        
        app = FastAPI(title="Database-Aware LLM API")
        
        class QueryRequest(BaseModel):
            question: str
            k: int = 3
        
        rag_chain = self.create_rag_chain()
        
        @app.post("/ask")
        async def ask(request: QueryRequest):
            try:
                # Run RAG chain
                result = rag_chain.invoke({"question": request.question})
                
                return {
                    "question": request.question,
                    "answer": result,
                    "timestamp": pd.Timestamp.now().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        uvicorn.run(app, host="0.0.0.0", port=port)

# Usage example
pipeline = ProductionLangChainPipeline(db_config, api_key="your_openai_key")
# pipeline.deploy_api(port=8000)
```

## Section 5: Production Deployment

### Step 1: Model versioning and monitoring
```python
import mlflow
import pandas as pd
from datetime import datetime

class AIDatabaseDeploymentManager:
    def __init__(self, db_config: Dict, mlflow_uri: str = "http://localhost:5000"):
        self.db_config = db_config
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)
    
    def log_ai_model_with_database_context(self, model, model_name: str,
                                         database_info: Dict, 
                                         training_metrics: Dict):
        """Log AI model with comprehensive database context"""
        with mlflow.start_run() as run:
            # Log model
            mlflow.pytorch.log_model(model, model_name)
            
            # Log database context
            mlflow.log_param("database_type", database_info.get("type", "unknown"))
            mlflow.log_param("database_version", database_info.get("version", "unknown"))
            mlflow.log_param("vector_db_type", database_info.get("vector_db_type", "unknown"))
            mlflow.log_param("embedding_model", database_info.get("embedding_model", "unknown"))
            
            # Log training data info
            mlflow.log_param("training_data_size", database_info.get("training_data_size", 0))
            mlflow.log_param("data_source", database_info.get("data_source", "unknown"))
            
            # Log metrics
            for metric_name, metric_value in training_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log feature info
            mlflow.log_param("num_features", database_info.get("num_features", 0))
            mlflow.log_param("feature_types", str(database_info.get("feature_types", [])))
    
    def monitor_ai_production_system(self, model_name: str, 
                                   monitoring_interval: int = 300):
        """Monitor AI production system performance"""
        import time
        
        while True:
            try:
                # Check prediction volume
                conn = psycopg2.connect(**self.db_config)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT COUNT(*) FROM predictions_log 
                    WHERE model_name = %s AND created_at > NOW() - INTERVAL '1 hour'
                """, (model_name,))
                hourly_predictions = cursor.fetchone()[0]
                
                # Check model drift
                cursor.execute("""
                    SELECT 
                        AVG(similarity_score) as avg_similarity,
                        COUNT(*) as total_predictions
                    FROM predictions_log 
                    WHERE model_name = %s 
                    AND created_at > NOW() - INTERVAL '1 day'
                """, (model_name,))
                drift_info = cursor.fetchone()
                
                conn.close()
                
                print(f"Model: {model_name}")
                print(f"Hourly predictions: {hourly_predictions}")
                if drift_info:
                    print(f"Average similarity: {drift_info[0]:.3f}")
                    print(f"Total predictions (24h): {drift_info[1]}")
                
                # Alert on anomalies
                if hourly_predictions < 10:
                    print("⚠️  Low prediction volume detected!")
                
            except Exception as e:
                print(f"Monitoring error: {e}")
            
            time.sleep(monitoring_interval)

# Usage example
deployment_manager = AIDatabaseDeploymentManager(db_config)

# Log model
database_info = {
    "type": "PostgreSQL",
    "version": "14.7",
    "vector_db_type": "pgvector",
    "embedding_model": "all-MiniLM-L6-v2",
    "training_data_size": 100000,
    "num_features": 5,
    "feature_types": ["text", "numeric", "categorical"]
}

metrics = {
    "accuracy": 0.92,
    "precision": 0.89,
    "recall": 0.91,
    "f1_score": 0.90,
    "latency_ms": 120.5
}

deployment_manager.log_ai_model_with_database_context(
    model=None,  # Replace with actual model
    model_name="rag_user_engagement_assistant",
    database_info=database_info,
    training_metrics=metrics
)
```

## Section 6: Performance Benchmarking

### Step 1: AI-platform integration benchmarking
```python
import time
import pandas as pd
from typing import List, Dict, Callable

class AIPlatformBenchmark:
    def __init__(self):
        self.results = []
    
    def benchmark_vector_search(self, methods: List[Callable], 
                              query_counts: List[int] = [10, 100, 1000]):
        """Benchmark vector search performance"""
        for method in methods:
            for count in query_counts:
                start_time = time.time()
                
                try:
                    method(count)
                    duration = time.time() - start_time
                    
                    self.results.append({
                        'benchmark': 'vector_search',
                        'method': method.__name__,
                        'query_count': count,
                        'duration_seconds': duration,
                        'throughput_queries_per_second': count / duration if duration > 0 else 0
                    })
                except Exception as e:
                    self.results.append({
                        'benchmark': 'vector_search',
                        'method': method.__name__,
                        'query_count': count,
                        'duration_seconds': float('inf'),
                        'error': str(e)
                    })
    
    def benchmark_rag_latency(self, methods: List[Callable], 
                            question_types: List[str] = ["factual", "analytical", "creative"]):
        """Benchmark RAG system latency"""
        for method in methods:
            for q_type in question_types:
                start_time = time.time()
                
                try:
                    method(q_type)
                    duration = time.time() - start_time
                    
                    self.results.append({
                        'benchmark': 'rag_latency',
                        'method': method.__name__,
                        'question_type': q_type,
                        'duration_seconds': duration
                    })
                except Exception as e:
                    self.results.append({
                        'benchmark': 'rag_latency',
                        'method': method.__name__,
                        'question_type': q_type,
                        'duration_seconds': float('inf'),
                        'error': str(e)
                    })
    
    def generate_ai_benchmark_report(self):
        """Generate comprehensive AI platform benchmark report"""
        df = pd.DataFrame(self.results)
        
        # Calculate statistics
        summary = df.groupby(['benchmark', 'method']).agg({
            'duration_seconds': ['mean', 'std', 'min', 'max'],
            'throughput_queries_per_second': ['mean', 'std']
        }).round(2)
        
        # Generate recommendations
        recommendations = []
        
        # Best vector search
        if 'vector_search' in df['benchmark'].values:
            best_search = df[df['benchmark'] == 'vector_search'].loc[
                df[df['benchmark'] == 'vector_search']['duration_seconds'].idxmin()
            ]
            recommendations.append(
                f"Best vector search: {best_search['method']} "
                f"({best_search['duration_seconds']:.2f}s for {best_search['query_count']} queries)"
            )
        
        # Best RAG latency
        if 'rag_latency' in df['benchmark'].values:
            best_rag = df[df['benchmark'] == 'rag_latency'].loc[
                df[df['benchmark'] == 'rag_latency']['duration_seconds'].idxmin()
            ]
            recommendations.append(
                f"Best RAG latency: {best_rag['method']} "
                f"({best_rag['duration_seconds']:.2f}s for {best_rag['question_type']} questions)"
            )
        
        return {
            'summary': summary,
            'detailed_results': df,
            'recommendations': recommendations,
            'ai_platform_tips': [
                "Use pgvector for production vector search",
                "Optimize embedding dimensions for your use case",
                "Implement caching for frequent queries",
                "Use asynchronous processing for high-throughput RAG",
                "Monitor similarity scores for model drift"
            ]
        }

# Usage example
benchmark = AIPlatformBenchmark()

# Define test methods
def test_pgvector_search(count: int):
    """Test pgvector search"""
    time.sleep(0.01 * count)

def test_faiss_search(count: int):
    """Test FAISS search"""
    time.sleep(0.005 * count)

def test_rag_basic(count: str):
    """Test basic RAG"""
    time.sleep(0.5)

# Run benchmarks
benchmark.benchmark_vector_search(
    [test_pgvector_search, test_faiss_search],
    [10, 100, 1000]
)

benchmark.benchmark_rag_latency(
    [test_rag_basic],
    ["factual", "analytical"]
)

report = benchmark.generate_ai_benchmark_report()
print("AI Platform Benchmark Report:")
print(report['summary'])
print("\nRecommendations:")
for rec in report['recommendations']:
    print(f"- {rec}")
```

## Hands-on Exercises

### Exercise 1: Vector database integration
1. Set up PostgreSQL with pgvector
2. Implement the `VectorDatabaseManager` class
3. Store and retrieve documents
4. Compare pgvector vs. FAISS performance

### Exercise 2: RAG system implementation
1. Build database-backed RAG system
2. Implement hybrid retrieval with metadata filtering
3. Test with different query types
4. Measure latency and accuracy

### Exercise 3: Hugging Face integration
1. Fine-tune model with database data
2. Implement database-backed inference
3. Save predictions to database
4. Monitor model performance

### Exercise 4: LangChain integration
1. Set up LangChain with database retriever
2. Create production RAG chain
3. Deploy as REST API
4. Implement monitoring and alerting

## Best Practices Summary

1. **Vector Database Optimization**: Use appropriate indexing and dimensionality
2. **RAG Architecture**: Implement hybrid retrieval and multi-hop reasoning
3. **Model Versioning**: Version both models and database schemas
4. **Monitoring**: Track prediction latency, similarity scores, and drift
5. **Security**: Secure database connections in AI pipelines
6. **Scalability**: Design for horizontal scaling of vector search
7. **Cost Optimization**: Balance embedding quality vs. computational cost
8. **Testing**: Test with realistic data volumes and query patterns

This tutorial provides practical, hands-on experience with database integration for modern AI/ML platforms. Complete all exercises to master these critical skills for building production-grade AI systems.