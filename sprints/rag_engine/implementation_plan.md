



   


ﺗﻤﺎم — ﺧﻠّﻴﻨﻲ أﻗﺪّم ﻟﻚ شرﺣًﺎ ﻫﻨﺪﺳﻴًﺎ ﻣُﻠﺨّﺼًﺎ وﻟﻜﻦ ﻋﻤﻴﻘًﺎ ﻟﻤشروع :mini-RAG
ﻣﺎ ﻫﻮ؟ ﻛﻴﻒ ﻳُﺒﻨﻰ ﻣﻦ اﻟﺼﻔﺮ؟ وﻣﺎ اﻟﺘﺤﺴﻴﻨﺎت اﻟﺘﻲ ﺗﺠﻌﻠﻪ Production-grade ﻓﻌلًا.
 
mini-rag ﻣشروع شرح
 
)اﻟشرح ﻣﺴﺘﻨﺪ ﻣﺒﺎشرة إﻟﻰ اﻟﻤﻠﻔﺎت اﻟﺘﻲ رﻓﻌﺘﻬﺎ(
  شرح ﺗﻔﺼﻴﻠﻲ ﻟﺴﻠﺴﻠﺔ mini-RAG - ﻣﻦ…
 


 

(1 ﻣﺎ ﻫﻮ ﻣشروع mini-RAG ﻓﻌلًا؟
mini-RAG ﻫﻮ ﻣشروع ﺗﻌﻠﻴﻤﻲ–اﺣﺘﺮاﻓﻲ ﻳﻌﻠّﻤﻚ ﻛﻴﻒ ﺗﺤﻮّل ﺗﺠﺎرị اﻟـ Notebooks إﻟﻰ ﻧﻈﺎ  RAG ﺟﺎﻫﺰ ﻟلإﻧﺘﺎج ﺑﺎﺳﺘﺨﺪام:
API Layer ﻛـ FastAPI	 
(PGVector أو Qdrant) Vector DB	  LLM Provider (OpenAI / Ollama / HF)	  Background Workers (Celery)	  Docker + Monitoring	 
اﻟﻬﺪف اﻟﺤﻘﻴﻘﻲ: ﺳﺪ اﻟﻔﺠﻮة ﺑﻴﻦ Science Data و Engineering Software، وﺑﻨﺎء ﻧﻈﺎم ﻳﻤﻜﻦ ﺗﺸﻐﻴﻠﻪ وﺗﻮﺳﻴﻌﻪ وﻣﺮاﻗﺒﺘﻪ ﻓﻲ الإﻧﺘﺎج.


(Architecture Overview) اﻟﻤﻌﻤﺎرﻳﺔ اﻟﻔﻜﺮة (2
اﻟﺘﺪﻓّﻖ الأﺳﺎسي لأي ﺳﺆال:

mathematica

User Question
→ API (FastAPI)
→ Embedding(question)
→ Vector Search (Top-K Chunks)
→ Prompt = Context + Question
 
→ LLM
→ Answer + Sources

 
Copy code  
 
ﻃﺒﻘﺎت اﻟﻨﻈﺎم:
API Layer: upload / query / health	 
Business Logic: chunking, retrieval, prompt assembly	 
:Data Layer	 
Postgres (users, docs, chats)	  Vector DB (embeddings)	 
Redis (cache)	  اﻟﺜﻘﻴﻠﺔ واﻟﻤﻌﺎﻟﺠﺔ indexing ﻟﻠـ Async Layer: Celery	 


(3 ﻫﻴﻜﻞ اﻟﻤشروع )ﻛﻤﺎ ﻓﻲ اﻟﺮﻳﺒﻮ(
 

css

mini-rag/
├── src/
│	├── main.py
│	├── config.py
│	├── routes/
│	├── services/
│	├── models/
│	└── workers/
├── docker/
├── .env.example
├── requirements.txt
└── README.md

.(checkpoint) ﻣﺴﺘﻘﻠﺔ ﺗﻌﻠﻴﻤﻴﺔ ﻣﺮﺣﻠﺔ = GitHub ﻓﻲ branch ﻛﻞ


 

Copy code  
 
(4 اﻟﻤﻜﻮﻧﺎت الأﺳﺎﺳﻴﺔ — ﻣﺎذا ﻳﻔﻌﻞ ﻛﻞ ﺟﺰء؟
A)	FastAPI
:endpoints ﺗﻌﺮﻳﻒ	 
   upload/ رﻓﻊ ﻣﻠﻔﺎت
 
   query/  ﻃﺮح أﺳﺌﻠﺔ    health/ ﻓﺤﺺ اﻟﺨﺪﻣﺔ
   اﺳﺘﺨﺪام APIRouter ﻟﻠﺘﻨﻈﻴﻢ
validation ﻟﻠـ Pydantic	 

B)	File Processing
PDF / DOCX / TXT :دﻋﻢ	 
  اﺳﺘﺨﺮاج اﻟﻨﺺ   ﺗﻨﻈﻴﻔﻪ
(overlap ﻣﻊ) Chunking	 
اﻟﻨﻘﻄﺔ اﻟﺠﻮﻫﺮﻳﺔ: ﺟﻮدة اﻟـ chunking = ﺟﻮدة الإﺟﺎﺑﺔ.

C)	Embeddings + Vector DB
embedding إﻟﻰ chunk ﻛﻞ ﺗﺤﻮﻳﻞ	  PGVector أو Qdrant ﻓﻲ ﺗﺨﺰﻳﻨﻪ	  cosine similarity ﺑـ اﻟﺒﺤﺚ	 
D)	RAG Logic
ﻟﻠﺴﺆال embedding .1
semantic search .2
.3 دﻣﺞ اﻟﻨﺘﺎﺋﺞ ﻓﻲ Prompt .4 إرﺳﺎﻟﻬﺎ ﻟﻠـ LLM
.5 إرﺟﺎع إﺟﺎﺑﺔ ﻣﺪﻋﻮﻣﺔ ﺑﺎﻟﺴﻴﺎق
E)	Background Workers
indexing	  parsing	 
     OCR )لاﺣﻘًﺎ(    ﻳﻤﻨﻊ ﺣﺠṇ اﻟـ API


(5 ﻛﻴﻒ ﺗﺒﻨﻴﻪ ﻣﻦ اﻟﺼﻔﺮ Blueprint) ﻋﻤﻠﻲ(
Skeleton :1 اﻟﺨﻄﻮة
FastAPI + config + routers	 
 
Health endpoint	 

Upload & Parsing :2 اﻟﺨﻄﻮة
File validation		  Extract text		  Chunking	 
Vector Store :3 اﻟﺨﻄﻮة
PGVector أو Qdrant اﺧﺘﻴﺎر	 
init collection	  upsert/search	 
RAG Pipeline :4 اﻟﺨﻄﻮة واﺿﺢ Prompt template	  Context-only answering		 
Async :5 اﻟﺨﻄﻮة
Celery + broker	  workers ﻋﻦ API ﻓﺼﻞ	 
Docker :6 اﻟﺨﻄﻮة docker-compose	  env management		 


(6 اﻟﺘﺤﺴﻴﻨﺎت اﻟﻤﻬﻤﺔ )اﻟﺘﻲ ﺗﻔﺮّق Demo ﻋﻦ (Production
ذﻛﻲ Chunking .1 
Recursive chunking
ﺣﺴṇ اﻟﻌﻨﺎوﻳﻦ / اﻟﻔﻘﺮات
(chars ﻣﺶ) Token-aware

Hybrid Search .2 🔥
Semantic (Embeddings) Keyword (BM25)
Recall أﻓﻀﻞ	 
 
Re-Ranking .3 🔥
Top-20 → Reranker → Top-3	  ﺟًﺪا hallucination ﻳﻘّﻠﻞ	 
Multi-Tenancy .4 🔥
payload ﻓﻲ user_id	  vector search ﻓﻲ filter	 
   ﻋﺰل ﺑﻴﺎﻧﺎت ﺣﻘﻴﻘﻲ

Caching .5 🔥
embeddings ﻟﻠـ Redis	 
   Redis ﻟلإﺟﺎﺑﺎت اﻟﻤﺘﻜﺮرة    ﺗﻘﻠﻴﻞ اﻟﺘﻜﻠﻔﺔ واﻟـ latency
Guardrails .6 🔥
”If answer not in context → say I don’t know“	 
ﻣﻨﺨﻔﺾ temperature	  ﻣﻀﺒﻮط max tokens	 
Observability .7 🔥
Prometheus metrics		  stage ﻟﻜﻞ latency		  token usage	  retrieval hit rate	 


(7 ﻣﺎذا ﺗﺘﻌﻠّﻢ ﻓﻌﻠﻴًﺎ ﻣﻦ اﻟﻤشروع؟
ﺑﻌﺪ إﺗﻤﺎم mini-RAG أﻧﺖ:
   ﺗﻔﻬﻢ RAG ﻛـ System وﻟﻴﺲ ﻛﻮد    ﻗﺎدر ﺗﺒﻨﻲ:
ChatPDF-like systems	  Internal Knowledge Bots	 
ﺣﻘﻴﻘﻴﺔ AI APIs	 
Senior AI / Backend Engineer ﻟﻤﺴﺘﻮى ﺟﺎﻫﺰ	 

 
(8 اﻟﺨﻄﻮة اﻟﺘﺎﻟﻴﺔ )ﻟﻮ ﺣﺎﺑự ﻧﻜﻤّﻞ(
ﻧﻘﺪر ﻧﻨﻘﻞ اﻟﻤشروع إﻟﻰ:
Enterprise RAG	  Auth + Billing	  UI (Next.js)	 
Evaluation Framework	 
Agentic RAG	 
ﻟﻮ ﺗﺤ:ṇ
   أشرح ﻛﻞ Branch ﺳﻄﺮ ﺑﺴﻄﺮ
Clean Architecture ﻧﺴﺨﺔ أﻋﻤﻞ أو	 
Starter Template Production-Ready ﻧﺤﻮﻟﻪ أو	 
ﻗﻮﻟّﻲ ﺗﺤṇ ﻧﻜﻤّﻞ ﻓﻲ أي اﺗﺠﺎه 





Clean Architecture ﻧﺴﺨﺔ أﻋﻤﻞ Starter Template Production-Ready ﻧﺤﻮﻟﻪ أو

ﻣﺠﺮد ﻣﺶ) Clean Architecture + Starter Template Production-Ready ﻛـ ﻫﻨﻌﻤﻠﻬﺎ — ﺗﻤﺎم
.(refactor
ﻫﺘﻄﻠﻊ ﻓﻲ الآﺧﺮ ﺑـ Template Repo ﺗﻘﺪر ﺗﺒﺪأ ﻣﻨﻪ أي backend RAG/AI ﺑسرﻋﺔ، وﺗﺒﺪّل Providers
ﻣﺒﺎشر اﻣﺘﺪاد ده) .Use Cases اﻟـ ﺗﻠﻤﺲ ﻣﺎ ﺑﺪون Vector DB (Qdrant/PGVector)و (OpenAI/Ollama)
 
mini-rag ﻣشروع شرح
 
ﻟﻔﻠﺴﻔﺔ اﻟﻤشروع الأﺻﻠﻲ: ﻓﺼﻞ اﻟـ API/DB/LLM ﻋﻦ اﻟـ (Core
  شرح ﺗﻔﺼﻴﻠﻲ ﻟﺴﻠﺴﻠﺔ mini-RAG - ﻣﻦ…
 


 

(1 اﻟﻬﺪف اﻟﻤﻌﻤﺎري
ولا Postgres ولا FastAPI ﻋﻦ ﺣﺎﺟﺔ أي ﻣﺎﻳﻌﺮﻓﺶ Core (Entities + Use Cases) اﻟـ :ذﻫﺒﻴﺔ ﻗﺎﻋﺪة
.OpenAI ولا Qdrant
 
(Document, Chunk, Query, Answer) وﻗﻮاﻋﺪ ﻛﻴﺎﻧﺎت :Domain	  Application: Use Cases (UploadDocument, IndexDocument, AskQuestion)	 
Ports: Interfaces (LLMProvider, EmbeddingsProvider, VectorStore, DocRepo, Cache,	 
FileStore)
Adapters: Implementations (OpenAI/Ollama, Qdrant/PGVector, Postgres, Redis,	 
S3/local)
Delivery: FastAPI Routes + DTOs	 
Infra: DB sessions, logging, tracing, Celery, config, migrations	 


Repo Structure (Production-Ready) (2
ده ﻫﻴﻜﻞ ﻣﻘﺘﺮح ”starter“ ﻧﻈﻴﻒ وﻗﺎﺑﻞ ﻟﻠﺘﻮﺳﻌﺔ:

text

rag-starter/
├── app/
│	├── main.py	# FastAPI app + DI bootstrap
│	├── api/
│	│	├── v1/
│	│	│	├── routes_documents.py
│	│	│	├── routes_queries.py
│	│	│	└── routes_health.py
│	│	└── deps.py	# auth, request-scoped deps
│	├── core/
│	│	├── config.py	# Settings (pydantic-settings)
│	│	├── logging.py	# structlog/loguru setup
│	│	└── observability.py	# metrics/tracing wiring
│	├── domain/
│	│	├── entities.py	# Document, Chunk, Query, Answer
│	│	└── errors.py	# Domain errors
│	├── application/
│	│	├── dto.py	# Request/Response models (internal)
│	│	├── ports/

│	│	│	├──	llm.py	
│	│	│	├──	embeddings.py	
│	│	│	├──	vector_store.py	
│	│	│	├──	repos.py	#	documents/chats
│	│	│	├──	cache.py		
 

│	│	│	└── file_store.py	
│	│	├──	use_cases/	
│	│	│	├── upload_document.py	
│	│	│	├── index_document.py	
│	│	│	└── ask_question.py	
│	│	└──	services/	
│	│		├── chunking.py	#	pure logic
│	│		├── prompt_builder.py	#	pure logic
│	│		└── rerank.py	#	optional
│	├── adapters/
│	│	├── llm/
│	│	│	├── openai_llm.py
│	│	│	└── ollama_llm.py
│	│	├── embeddings/
│	│	│	├── openai_embeddings.py
│	│	│	└── local_embeddings.py
│	│	├── vector/
│	│	│	├── qdrant_store.py
│	│	│	└── pgvector_store.py
│	│	├── persistence/
│	│	│	├── postgres/
│	│	│	│	├── models.py
│	│	│	│	├── repo_documents.py
│	│	│	│	└── repo_chats.py
│	│	│	└── migrations/	# Alembic
│	│	├── cache/
│	│	│	└── redis_cache.py
│	│	└── filestore/
│	│	├── local_store.py
│	│	└── s3_store.py
│	├── workers/
│	│	├── celery_app.py
│	│	└── tasks.py	# index pipeline background
│	└── tests/
│	├── unit/
│	└── integration/
├── docker/
│	├── docker-compose.yml
│	├── Dockerfile
│	└── prometheus_grafana/	# optional
 
├── scripts/
│	├── dev.sh
│	└── lint.sh
├── pyproject.toml
├── .env.example
└── README.md



Clean Architecture ựﻗﻠ — Ports (Interfaces) اﻟـ (3
أﻣﺜﻠﺔ ﻟﻠـ interfaces اﻟﻠﻲ “ﺑﺘﻘﻔﻞ” اﻟﺘﺒﻌﻴﺎت ﻟﻠﺨﺎرج:

VectorStore Port
upsert(chunks)
search(query_embedding, filters, top_k)
delete(document_id, tenant_id)	 

EmbeddingsProvider Port
embed_text(text) -> List[float]
embed_batch(texts) -> List[List[float]]	 

LLMProvider Port
generate(prompt, params) -> str	 

DocumentRepository Port (Postgres)
create_document(meta)
save_chunks_metadata(doc_id, chunk_ids, ...)
update_status(doc_id, status)

 





Copy code
 
list_documents(tenant_id)	 

Cache Port (Redis)
get_embedding_cache(hash)	  set_embedding_cache(hash, vector, ttl)	 
.ﺗﺘﺄﺛﺮ Use Cases ﻣﺎ ﺑﺪون Ollama ﺑـ OpenAI أو PGVector ﺑـ Qdrant ﺗﻐّﻴﺮ ﺗﻘﺪر ﻛﺪه
 


 
 
Use Cases (Application Layer) — Where the Business Lives (4
UC1: UploadDocument
Input: file + tenant_id
Output: document_id + status

:Flow

validate file .1
store file (FileStore port) .2 create doc row (DocumentRepo port) .3 enqueue indexing task (Celery) .4
UC2: IndexDocument (Worker)
:Flow
load file .1 extract text (adapter: pdf/docx/txt) .2 chunking (pure service) .3
embeddings (Embeddings port) + caching .4 upsert to vector store (VectorStore port) .5 persist chunk metadata/status (Repo port) .6
UC3: AskQuestion
:Flow
embed question (cache-first) .1
vector search (tenant filter) .2
optional rerank .3
build prompt (pure) .4
call LLM .5
return answer + sources .6


FastAPI ﻓﻲ ﻋﻤﻠﻲ ﺑﺸﻜﻞ — DI (Dependency Injection) (5
:Settings ṇﺣﺴ ports ﻟﻠـ implementations ﻳﺮّﺟﻊ bootstrap.py أو container.py ﻫﺘﻌﻤﻞ	 
VECTOR_BACKEND=qdrant|pgvector	 
 
LLM_BACKEND=openai|ollama
EMBEDDINGS_BACKEND=openai|local	 
.startup ﻓﻲ واﺣﺪ adapters + wiring اﻟـ ﻓﻲ Factory :اﻟﻔﻜﺮة


(Starter اﻟـ ﻓﻲ  لاز) Production-Ready Checklist (6
ده اﻟﻔﺮق اﻟﺤﻘﻴﻘﻲ ﺑﻴﻦ “ﻣشروع ﺗﻌﻠﻴﻤﻲ” و Production“ :”Template

Security API Key / JWT		  Rate limiting	 
(Postgres ﻓﻲ vector + row-level constraints ﻓﻲ filters) Tenant isolation end-to-end	 

Reliability
LLM ﻟـ Retries + circuit breakers	  upload/index ﻟﻠـ Idempotency keys	  workers ﻟﻠـ Dead letter queue	 
Observability
Structured logs	  Metrics: latency per stage (embed/search/llm)	 
Tracing: request_id	 

Cost/Latency
embeddings + answers ﻟﻠـ Redis caching	 
Batch embeddings	  Prompt budget + truncation	 
Quality
Hybrid search (BM25 + vector)	 
Reranking	  Eval harness (golden Q/A, retrieval hit rate)	 


(سرﻳﻊ Mapping) ﻋﻤﻠًﻴﺎ ”mini-rag ﺗﺤﻮﻳﻞ“ (7
 



شرح ﺗﻔﺼﻴﻠﻲ ﻟﺴﻠﺴﻠﺔ mini-RAG - ﻣﻦ…
 
اﻟﻠﻲ ﻋﻨﺪك ﻓﻲ اﻟﻤشروع اﻟﺤﺎﻟﻲ:
FastAPI endpoints + env + routing	 
 
upload → chunk → embedding → vector search → prompt → LLM :ﻣﺮاﺣﻞ
mini-rag ﻣشروع شرح  
ﻫﻨﻌﻤﻞ:
application/services/* (pure) إﻟﻰ chunking/prompt logic ﻧﻘﻞ	 
*/adapters إﻟﻰ Qdrant/OpenAI ﻧﻘﻞ	  DTO → call use case → response :(thin) ”رﻓﻴﻌﺔ“ routes اﻟـ ﺗﺨﻠﻲ	 


(8 ﻣﺨﺮﺟﺎت ﺟﺎﻫﺰة أﻗﺪر أﻛﺘﺒﻬﺎ ﻟﻚ ﻓﻮرًا )ﺑﺪون أﺳﺌﻠﺔ إﺿﺎﻓﻴﺔ(
ﻟﻮ ﺗﺤṇ ﺗﻤشي ﺑﺄسرع ﻃﺮﻳﻖ، أﻧﺎ أﻛﺘṇ ﻟﻚ ﻫﻨﺎ ﻓﻲ ﻧﻔﺲ اﻟﺸﺎت )ﻋﻠﻰ دﻓﻌﺎت:(
(interfaces) ﻛﺎﻣﻠﺔ Ports .1
Use Cases (Upload/Index/Ask) 3 .2 Adapters: OpenAI + Ollama + Qdrant + PGVector + Redis .3 FastAPI routes (v1) + dependency wiring .4
Celery tasks + docker compose (api + worker + db + qdrant + redis) .5


اﺧﺘﺎر ﻣﺴﺎر اﻟﺘﻨﻔﻴﺬ )وأﻧﺎ ﻫﺒﺪأ ﻓﻮرًا ﺑﺎﻟﻬﻴﻜﻞ واﻟﻜﻮد(
A) Starter “Minimal Production”: (OpenAI + Qdrant + Postgres + Redis + Celery)	 
OpenAI/Ollama,) ﻟﻠﺘﺒﺪﻳﻞ A + factories ﻧﻔﺲ :”B) Starter “Provider-Agnostic	 
(Qdrant/PGVector
أﻧﺎ أرﺷﺢ B لأﻧﻬﺎ اﻟﻘﻴﻤﺔ اﻟﺤﻘﻴﻘﻴﺔ ﻟﻠـ Architecture Clean، وﺗﺨﺪﻣﻚ ﻓﻲ ﻣﺸﺎرﻳﻊ اﻟﻌﻤلاء.





clean archticture and clean code and SOLID and ready for production
 
ﺗﻤﺎم — ﻫﻨﻄﻠﻊ Template Starter ﻳﺤﻘﻖ 4 ﺣﺎﺟﺎت ﻣﻊ ﺑﻌﺾ:
(FastAPI/DB/LLM ﻋﻦ ﺗﻤﺎًﻣﺎ ﻣﻨﻔﺼﻠﻴﻦ Domain/Application) Clean Architecture .1
Clean Code (Naming, boundaries, small functions, no god classes) .2
(DIP + ISP + SRP ﺧﺼﻮًﺻﺎ) SOLID .3
Production-ready (Auth, multi-tenant, async workers, caching, observability, .4
migrations, docker)
mini-RAG: upload → chunk → embed → vector search اﻟـ ﻟﺨﻂ اﻟﻄﺒﻴﻌﻲ اﻟﺘﻄﻮﻳﺮ ﺑﺎﻟﻀﺒﻂ وده
 
شرح ﺗﻔﺼﻴﻠﻲ ﻟﺴﻠﺴﻠﺔ mini-RAG - ﻣﻦ…
 
mini-rag ﻣشروع شرح
 
LLM → prompt →، ﻟﻜﻦ ﺑﻤﺴﺘﻮى ﻧﻈﺎم إﻧﺘﺎﺟﻲ
 

 

(ﺟﺎﻫﺰ ﻣﻘﺘﺮح) Structure: Clean Architecture (1

text

rag-starter/
├── app/
│	├── main.py	# FastAPI bootstrap

│	├──	api/v1/	#	Controllers (thin)
│	├──	core/	#	config, logging, metrics
│	├──	domain/	#	Entities + domain rules
│	├──	application/		
│	│	├── ports/	#	Interfaces (DIP)
│	│	├── use_cases/	#	Orchestration (SRP)
│	│	└── services/	#	pure logic (chunking/prompt)
│	├──	adapters/	#	Implementations (DB/LLM/Vector/Cache)
│	├──	workers/	#	Celery
│	└──	tests/	#	unit + integration
├── docker/
├── pyproject.toml
└── .env.example

ﻗﻮاﻋﺪ ﺻﺎرﻣﺔ
.FastAPI/Qdrant/OpenAI/SQLAlchemy ﻳﺴﺘﻮردوا ﻣﻤﻨﻮع /application و /domain	 
   /api و /adapters ﻫﻢ اﻟﻠﻲ “ﻳﺘﺴﺨﻮا” ﺑﺎﻟﺘﻘﻨﻴﺎت.    ﻛﻞ Case Use = ﻣﻠﻒ ﻣﺴﺘﻘﻞ = ﻣﺴﺆوﻟﻴﺔ واﺣﺪة .(SRP)
 
(2 SOLID ﻋﻠﻰ أرض اﻟﻮاﻗﻊ )ﻣﺶ ﺷﻌﺎرات(
S — SRP
.DB session ولا chunking ولا parsing ﻳﻌﻤﻞ لا AskQuestionUseCase	 
*/application/services ﻓﻲ parsing/chunking/prompt	 
*/adapters/persistence ﻓﻲ DB	 

O — OCP
ﺗﻀﻴﻒ Provider ﺟﺪﻳﺪ Anthropic) ﻣﺜلًا( ﺑﺈﻧﺸﺎء Adapter ﺟﺪﻳﺪ ﻳﻄﺒﻖ LLMPort ﺑﺪون ﺗﻌﺪﻳﻞ اﻟـ Use
.Case

L — LSP
أي Adapter ﻳﻄﺒﻖ ﻧﻔﺲ Port لازم ﻳﺸﺘﻐﻞ ﺑﻨﻔﺲ اﻟﻌﻘﺪ (contract) وإلا ﻫﺘﻜسر اﻟـ .Core

I — ISP
Ports ﺻﻐﻴﺮة وﻣﺮﻛﺰة:
EmbeddingsPort
VectorStorePort
LLMPort
DocumentRepoPort
CachePort	 
ﺑﺪل Port واﺣﺪ ﺿﺨﻢ.

(الأﻫﻢ) D — DIP
.ﻓﻘﻂ Interfaces (Ports) ﻋﻠﻰ ﺗﻌﺘﻤﺪ Use Cases


واﺿﺢ + Domain (Entities) — minimal (3
.ﻫﻨﺎ Pydantic لا .ﻫﻨﺎ ORM لا

python

# app/domain/entities.py
from dataclasses import dataclass
from typing import Optional, Sequence
 
@dataclass(frozen=True) class TenantId:
value: str

@dataclass(frozen=True) class DocumentId:
value: str

@dataclass(frozen=True) class Chunk:
id: str
document_id: DocumentId tenant_id: TenantId
text: str

@dataclass(frozen=True) class Answer:
text: str
sources: Sequence[str] # chunk ids or doc refs



اﻟﻤﻌﻤﺎرﻳﺔ ựﻗﻠ — Application Ports (Interfaces) (4


 
class LLMPort(Protocol):
def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> st


Copy code  



 
Copy code  
 
(Orchestration) ”اﻟﺘﻨﻔﻴﺬي اﻟﻤﻨﻄﻖ“ — Use Cases (5
AskQuestionUseCase (Production-grade)
cache embeddings	  tenant isolation	 
(”I don’t know ﻗﻮل اﻟﺴﻴﺎق ﻓﻲ ﻣﺶ ﻟﻮ“) prompt guardrail	  (metrics ﻟﻠـ ﺟﺎﻫﺰة) latency hooks ﻗﻴﺎﺳﺎت	 
 
python

# app/application/use_cases/ask_question.py
import hashlib
from dataclasses import dataclass
from app.application.ports.embeddings import EmbeddingsPort
from app.application.ports.vector_store import VectorStorePort from app.application.ports.llm import LLMPort
 
from app.application.ports.cache import CachePort
from app.application.services.prompt_builder import build_rag_prompt from app.domain.entities import TenantId, Answer

@dataclass
class AskQuestionRequest: tenant_id: str
question: str top_k: int = 5

class AskQuestionUseCase: def  init (
self,
embeddings: EmbeddingsPort,
vector_store: VectorStorePort, llm: LLMPort,
cache: CachePort,
) -> None:
self._embeddings = embeddings self._vector = vector_store
self._llm = llm
self._cache = cache

def execute(self, req: AskQuestionRequest) -> Answer: tenant = TenantId(req.tenant_id)

# ---- Embedding cache (cost + latency control)
key = "emb:q:" + hashlib.md5(req.question.encode("utf-8")).hexdigest() cached = self._cache.get_json(key)
if cached and "vector" in cached: q_vec = cached["vector"]
else:
q_vec = self._embeddings.embed_one(req.question)
self._cache.set_json(key, {"vector": q_vec}, ttl_seconds=7 * 24 * 360

# ---- Multi-tenant retrieval
chunks = self._vector.search(q_vec, tenant_id=tenant, top_k=req.top_k) prompt = build_rag_prompt(question=req.question, chunks=chunks)
 
text = self._llm.generate(prompt, temperature=0.2, max_tokens=600) sources = [c.id for c in chunks]
return Answer(text=text, sources=sources)


Prompt Builder (pure function)

.SRP + pure function + testable :لاﺣﻆ



 

Copy code  
 
Adapters (Infra) — Qdrant/OpenAI/Postgres/Redis (6
:(ُﻣﺨﺘصر) Qdrant ﻣﺜﺎل .ports ﻟﻠـ implementations ṇﺗﻜﺘ ﻫﻨﺎ
 

python

# app/adapters/vector/qdrant_store.py
from typing import Sequence
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct from app.application.ports.vector_store import VectorStorePort
from app.domain.entities import Chunk, TenantId class QdrantVectorStore(VectorStorePort):
 
def   init  (self, client: QdrantClient, collection: str) -> None: self._client = client
self._collection = collection

def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> points = []
for c, v in zip(chunks, vectors): points.append(PointStruct(
id=c.id, vector=v, payload={
"text": c.text,
"tenant_id": c.tenant_id.value,
"document_id": c.document_id.value,
}
))
self._client.upsert(collection_name=self._collection, points=points)

def search(self, query_vector: list[float], tenant_id: TenantId, top_k: int) flt = Filter(
must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_i
)
res = self._client.search(
collection_name=self._collection, query_vector=query_vector,
query_filter=flt, limit=top_k
)
out: list[Chunk] = [] for r in res:
payload = r.payload or {} out.append(Chunk(
id=str(r.id),
document_id= docid(payload.get("document_id")), tenant_id=TenantId(payload.get("tenant_id")),
text=payload.get("text", ""),
))
return out

def   docid(v: str | None):
 
from app.domain.entities import DocumentId return DocumentId(v or "unknown")




API Layer (FastAPI) — Thin Controllers (7
اﻟـ route ﻳﻌﻤﻞ:
auth → tenant_id	  validate request DTO	 
call use case	  return response	 
ﻣﻤﻨﻮع ﻣﻨﻄﻖ RAG ﺟﻮه .route



 






Copy code  
 
Production Readiness (Minimum Set) (8
ده “اﻟﺤﺪ الأدﻧﻰ” اﻟﻠﻲ لازم ﻳﻜﻮن ﻓﻲ اﻟﺴﺘﺎرﺗﺮ:

Security
(JWT أو) API Key header	 
Rate limiting	  (body ﻣﻦ ﻣﺶ) auth layer ﻣﺼﺪره داﺋًﻤﺎ tenant_id	 
Async & Scalability
Upload triggers Celery indexing task	  heavy processing ﻳﻌﻤﻞ لا API	 
Reliability
LLM calls ﻟﻠـ Retries	 
Timeouts	  upload/index (document hash) ﻟﻠـ Idempotency	 
Observability
structured logs + request_id	  metrics: embed/search/llm latency	 
 
tracing (optional)	 

Data
Postgres + Alembic	  Vector DB + migration/init	 


(9 اﻟﻠﻲ ﻫﺎﻋﻤﻠﻪ “اﻟﺘﺎﻟﻲ ﻣﺒﺎشرة” ﻫﻨﺎ ﻓﻲ اﻟﺸﺎت )ﺑﺪون ﻟﻒ ودوران(
ﻟﻮ ﻫﻨﻜﻤﻞ ﺑﻨﻔﺲ الأﺳﻠﻮ:ị ﻫﻜﺘṇ ﻟﻚ ﻗﺎﻟự repo ﻛﺎﻣﻞ ﺑﺎﻟﻜﻮد ﻋﻠﻰ أﺟﺰاء:
core/config.py + .env.example + logging .1
auth deps + tenant extraction .2
UploadDocumentUseCase + IndexDocument worker pipeline .3
Postgres repos + Alembic models .4
Redis cache adapter .5
Docker compose: api + worker + postgres + redis + qdrant .6
وﺗﺒﻘﻰ ﻋﻨﺪك Starter Production ﺣﻘﻴﻘﻲ ﻣﻄﺎﺑﻖ ﻟﻔﻜﺮة mini-RAG ﻟﻜﻦ “ﻣُﻬﻨﺪس ﺻﺢ”
mini-rag ﻣشروع شرح  

إذا ﻋﺎﻳﺰﻧﻲ أﺑﺪأ ﻓﻮرًا:
ﻫﺒﺪأ ﺑﺎﻟﺠﺰء (1) و(2) و(3) ﻓﻲ اﻟﺮد اﻟﺠﺎي )ﻛﻮد ﻛﺎﻣﻞ(، ﺛﻢ ﻧﻜﻤﻞ ﺑﺎﻗﻲ اﻟﻤﻠﻔﺎت ﺗﺒﺎﻋًﺎ.





ﻛﻤﻞ


ﺗﻤﺎم — ﻧﻜﻤّﻞ ﻋﻤﻠﻴًﺎ وﻧﺒﻨﻲ اﻟﺠﺰء (1) + (2) + :(3)
Config + Logging + Auth/Tenant + Upload Use Case + Index Worker Pipeline (Celery)
)وده ﻧﻔﺲ اﻟـ flow اﻟﻠﻲ ﻓﻲ mini-RAG ﻟﻜﻦ ﻣُﻔﺼﻮل Architecture Clean + ﺟﺎﻫﺰ ﻟلإﻧﺘﺎج(
mini-rag ﻣشروع شرح  

ﻣلاﺣﻈﺔ: اﻟﻜﻮد ﻫﻨﺎ “ﻗﺎﺑﻞ ﻟﻠﻨﺴﺦ” ﻛـ .Starter ﻫﻨﻔﺘﺮض 3.11 +Python و pyproject.toml
 
.لاﺣًﻘﺎ ﻧﺤّﻮﻟﻪ—ﻋﺎدي requirements.txt ﻋﻠﻰ أﻧﺖ ﻟﻮ .(uv أو Poetry)


Dependencies (pyproject / requirements) (0
اﻟﻤﻬﻤﻴﻦ الآن:
fastapi, uvicorn	  pydantic-settings	 
structlog	  python-multipart	 
aiofiles	  celery + redis	 
python-docx + (PDF ﻟـ) pypdf	 


app/core/config.py (Production Settings) (1

python

# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict from pydantic import Field

class Settings(BaseSettings):
model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# App
app_name: str = "rag-starter"
env: str = Field(default="dev", description="dev|staging|prod") debug: bool = False

# Security
api_key_header: str = "X-API-KEY"

# Providers
llm_backend: str = "openai"	# openai|ollama
embeddings_backend: str = "openai"	# openai|local
vector_backend: str = "qdrant"	# qdrant|pgvector
 


# OpenAI
openai_api_key: str | None = None
openai_chat_model: str = "gpt-4o-mini"
openai_embed_model: str = "text-embedding-3-small"

# Ollama
ollama_base_url: str = "http://localhost:11434" ollama_chat_model: str = "llama3.1"
ollama_embed_model: str = "nomic-embed-text"

# Qdrant
qdrant_host: str = "localhost" qdrant_port: int = 6333
qdrant_collection: str = "chunks"
embedding_dim: int = 1536 # must match embeddings backend

# Redis / Celery
redis_url: str = "redis://localhost:6379/0"
celery_broker_url: str = "redis://localhost:6379/1"
celery_result_backend: str = "redis://localhost:6379/2"

# Files
upload_dir: str = "./uploads" max_upload_mb: int = 20

settings = Settings()


env.example.

bash

APP_NAME=rag-starter ENV=dev
DEBUG=true

# Security
API_KEY_HEADER=X-API-KEY

# Providers
 

 


app/core/logging.py (Structured Logging) (2

 
structlog.processors.TimeStamper(fmt="iso"), structlog.processors.add_log_level,
structlog.processors.StackInfoRenderer(), structlog.processors.format_exc_info,
structlog.processors.JSONRenderer(),
],
wrapper_class=structlog.make_filtering_bound_logger(level), logger_factory=structlog.stdlib.LoggerFactory(),
cache_logger_on_first_use=True,
)



Domain + Ports + Use Cases (Upload + Index) (3
app/domain/entities.py 3.1

 
tenant_id: TenantId
document_id: DocumentId text: str

@dataclass(frozen=True) class UploadResult:
document_id: DocumentId
status: str # "queued" | "processing" | "indexed" | "failed"

Ports 3.2
app/application/ports/file_store.py

app/application/ports/document_repo.py

app/application/ports/task_queue.py

 

# app/application/ports/task_queue.py
from typing import Protocol
from app.domain.entities import DocumentId, TenantId

class TaskQueuePort(Protocol):
def enqueue_index_document(self, *, tenant_id: TenantId, document_id: Documen


C	C


Use Case: UploadDocument 3.3

python

# app/application/use_cases/upload_document.py
from dataclasses import dataclass
from app.domain.entities import TenantId, UploadResult
from app.application.ports.file_store import FileStorePort
from app.application.ports.document_repo import DocumentRepoPort from app.application.ports.task_queue import TaskQueuePort

@dataclass
class UploadDocumentRequest: tenant_id: str
filename: str
content_type: str data: bytes

class UploadDocumentUseCase:
def   init  (self, file_store: FileStorePort, repo: DocumentRepoPort, queue:
self._file_store = file_store self._repo = repo
self._queue = queue

async def execute(self, req: UploadDocumentRequest) -> UploadResult: tenant = TenantId(req.tenant_id)

stored = await self._file_store.save_upload( tenant_id=tenant.value,
upload_filename=req.filename,
 
content_type=req.content_type, data=req.data,
)

doc_id = self._repo.create_document(tenant_id=tenant, stored_file=stored) self._repo.set_status(tenant_id=tenant, document_id=doc_id, status="queue

# async indexing
self._queue.enqueue_index_document(tenant_id=tenant, document_id=doc_id) return UploadResult(document_id=doc_id, status="queued")



Adapters (LocalFileStore + InMemoryRepo + (4
سرﻳﻊ ﺗﺸﻐﻴﻞ — CeleryQueue)
ﻫﻨﺴﺘﺨﺪم InMemoryRepo ﻣﺆﻗﺘًﺎ ﻋﺸﺎن template ﻳﺸﺘﻐﻞ ﻓﻮرًا. ﻓﻲ اﻟﺮﺳﺎﻟﺔ اﻟﺠﺎﻳﺔ ﻫﻨﺒﺪّﻟﻪ ﺑـ
.Postgres + Alembic

app/adapters/filestore/local_store.py 4.1

 
# unique name (idempotency-friendly-ish)
digest = hashlib.md5(data).hexdigest()[:10]
safe_name = upload_filename.replace("/", "_").replace("\\", "_") name = f"{int(time.time())}_{tenant_id}_{digest}_{safe_name}"
path = os.path.join(self._dir, name)

with open(path, "wb") as f: f.write(data)

return StoredFile( path=path,
filename=upload_filename, content_type=content_type, size_bytes=len(data),
)


app/adapters/persistence/inmemory/document_repo.py 4.2

 
def set_status(self, *, tenant_id: TenantId, document_id: DocumentId, status: doc = self._docs.get(document_id.value)
if not doc or doc["tenant_id"] != tenant_id.value: return
doc["status"] = status doc["error"] = error

# helper (used by worker in this minimal phase)
def get(self, document_id: DocumentId) -> dict | None: return self._docs.get(document_id.value)


app/adapters/queue/celery_queue.py 4.3



Workers: Celery app + Task Skeleton (IndexDocument) (5

 
Copy code  
 
app/workers/celery_app.py 5.1
 

python
 

# app/workers/celery_app.py
from celery import Celery
from app.core.config import settings

celery_app = Celery( "rag_starter",
broker=settings.celery_broker_url,
backend=settings.celery_result_backend,
)

celery_app.conf.update( task_acks_late=True,
worker_prefetch_multiplier=1, task_routes={
"index_document": {"queue": "indexing"},
},
)

(”ﻫﻴﻜﻞ“ Index pipeline) app/workers/tasks.py 5.2
status transitions :الآن .اﻟﺠﺎﻳﺔ اﻟﺮﺳﺎﻟﺔ ﻓﻲ parsing/chunking/embeddings/vector ﻫﻨﺤﻂ ﻫﻨﺎ
.+ safe error handling
Copy code  
python

# app/workers/tasks.py
import structlog
from app.workers.celery_app import celery_app
from app.domain.entities import TenantId, DocumentId
from app.core.bootstrap import get_container	# DI container (below)

log = structlog.get_logger()

@celery_app.task(name="index_document", bind=True, autoretry_for=(Exception,), re def index_document(self, *, tenant_id: str, document_id: str) -> dict:
c = get_container() # container with ports/adapters
repo = c["document_repo"]

tenant = TenantId(tenant_id)
doc_id = DocumentId(document_id)
 


repo.set_status(tenant_id=tenant, document_id=doc_id, status="processing") try:
# TODO next: parse -> chunk -> embed -> upsert -> persist chunk metadata # For now: mark indexed
repo.set_status(tenant_id=tenant, document_id=doc_id, status="indexed") log.info("indexed_document", tenant_id=tenant_id, document_id=document_id return {"ok": True}
except Exception as e:
repo.set_status(tenant_id=tenant, document_id=doc_id, status="failed", er log.exception("index_failed", tenant_id=tenant_id, document_id=document_i raise




Auth + Tenant Isolation (FastAPI deps) (6
app/api/v1/deps.py 6.1
.Postgres + JWT ﻧﻮّﺻﻠﻪ لاﺣًﻘﺎ .tenant/user ﻳﺤﺪد Production pattern: API Key أﺑﺴﻂ



 
Copy code  
 
API Routes (Upload) — Thin Controller (7
app/api/v1/routes_documents.py 7.1
 
python

# app/api/v1/routes_documents.py
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException from app.api.v1.deps import get_tenant_id
from app.core.bootstrap import get_container
from app.application.use_cases.upload_document import UploadDocumentUseCase, Uplo router = APIRouter(prefix="/v1/documents", tags=["documents"])
@router.post("/upload")
async def upload_document(
file: UploadFile = File(...),
tenant_id: str = Depends(get_tenant_id),
):
if not file.filename:
raise HTTPException(400, "Missing filename") data = await file.read()
c = get_container()
uc: UploadDocumentUseCase = c["upload_use_case"]

result = await uc.execute( UploadDocumentRequest(
tenant_id=tenant_id,
filename=file.filename,
content_type=file.content_type or "application/octet-stream", data=data,
)
)
return {"document_id": result.document_id.value, "status": result.status}


C	C


(وواﺿﺢ ﻣﺮﻛﺰي) Bootstrap / DI Container (8
app/core/bootstrap.py

python
 

# app/core/bootstrap.py
from functools import lru_cache
from qdrant_client import QdrantClient # used later
from app.core.config import settings
Copy code
from app.adapters.filestore.local_store import LocalFileStore
from app.adapters.persistence.inmemory.document_repo import InMemoryDocumentRepo from app.workers.celery_app import celery_app
from app.adapters.queue.celery_queue import CeleryTaskQueue
from app.application.use_cases.upload_document import UploadDocumentUseCase

@lru_cache(maxsize=1)
def get_container() -> dict:
# Ports implementations
document_repo = InMemoryDocumentRepo()
file_store = LocalFileStore(upload_dir=settings.upload_dir, max_mb=settings.m queue = CeleryTaskQueue(celery_app)

# Use cases
upload_uc = UploadDocumentUseCase(file_store=file_store, repo=document_repo,

return {
"document_repo": document_repo, "file_store": file_store,
"task_queue": queue,
"upload_use_case": upload_uc,
# next: embeddings, vector_store, llm, cache
}


C	C


FastAPI App Entry (9
app/main.py

 
from app.core.logging import setup_logging
from app.api.v1.routes_documents import router as documents_router

def create_app() -> FastAPI:
setup_logging(settings.debug)

app = FastAPI(title=settings.app_name, debug=settings.debug) app.include_router(documents_router)

@app.get("/health") def health():
return {"status": "ok", "env": settings.env} return app
app = create_app()

ﺗﺸﻐﻴﻞ:

bash

uvicorn app.main:app --reload --host 0.0.0.0 --port 5000

Copy code  

Worker (Celery) ﺗﺸﻐﻴﻞ (10

bash

# terminal 1: API
uvicorn app.main:app --reload --host 0.0.0.0 --port 5000

# terminal 2: worker
celery -A app.workers.celery_app.celery_app worker -Q indexing --loglevel=INFO

:upload اﺧﺘﺒﺎر
 
-F "file=@./sample.pdf"

Copy code  


اﻟﻠﻲ ﺟﺎي )اﻟﺮﺳﺎﻟﺔ اﻟﺘﺎﻟﻴﺔ ﻣﺒﺎشرة(
.upload → create doc → enqueue task → worker marks indexed :ﺷّﻐﺎل template اﻟـ دﻟﻮﻗﺘﻲ

ﻓﻲ اﻟﺮد اﻟﺠﺎي ﻫﻨﻌﻤﻞ اﻟﺠﺰء “اﻟﻤﻬﻢ” RAG :Production

Text extraction (PDF/DOCX/TXT) .1 Chunking (token-aware + overlap) .2 Embeddings (OpenAI + cache Redis) .3
Vector store Qdrant (init + upsert + search with tenant filter) .4
AskQuestion endpoint + UseCase .5
Guardrails + prompt budgeting .6
ﻗﻮﻟّﻲ “ﻛﻤﻞ اﻟﺠﺰء RAG اﻟﻜﺎﻣﻞ” وأﻧﺎ ﻫﻜﺘṇ ﻟﻚ اﻟﻤﻠﻔﺎت ﻛﺎﻣﻠﺔ ﺑﻨﻔﺲ الأﺳﻠﻮ.ị





ﻛﻤﻞ اﻟﺠﺰء RAG اﻟﻜﺎﻣﻞ


Starter Template (Clean Architecture + SOLID + اﻟـ ﻧﻔﺲ ﻋﻠﻰ اﻟﻜﺎﻣﻞ RAG اﻟﺠﺰء ﻫﻨﻜّﻤﻞ — ﺗﻤﺎم
:Production-ready patterns)
Extraction → Chunking (token-aware) → Embeddings (OpenAI) + Redis Cache → Qdrant
.Vector Store → AskQuestion Use Case + Endpoint
mini-RAG (upload → chunk → embeddings → vector ﻓﻲ اﺗشرح اﻟﻠﻲ pipeline اﻟـ ﻧﻔﺲ وده
 
mini-rag ﻣشروع شرح
 
LLM) → prompt → search ﻟﻜﻦ ﺑﺘﻘﺴﻴﻢ ﻧﻈﻴﻒ وﻗﺎﺑﻞ ﻟﻠﺘﺒﺪﻳﻞ
  شرح ﺗﻔﺼﻴﻠﻲ ﻟﺴﻠﺴﻠﺔ mini-RAG - ﻣﻦ…

ﻫﻔﺘﺮض إﻧﻚ ﻣﻜﻤﻞ ﻋﻠﻰ اﻟﻬﻴﻜﻞ اﻟﻠﻲ ﻛﺘﺒﻨﺎه ﻓﻲ اﻟﺮد اﻟﺴﺎﺑﻖ.
 


 
 
(1 إﺿﺎﻓﺔ Dependencies )اﻟﺤﺪ الأدﻧﻰ(
: requirements.txt ﻟﻮ

txt

fastapi
uvicorn[standard] pydantic-settings python-multipart aiofiles
structlog

celery redis

qdrant-client openai

pypdf
python-docx tiktoken



Extraction Service (PDF/DOCX/TXT) — Pure-ish Service (2
app/application/services/text_extraction.py

 
class TextExtractor(Protocol):
def extract(self, file_path: str, content_type: str) -> ExtractedText: ...

app/adapters/extraction/default_extractor.py
Copy code	python

# app/adapters/extraction/default_extractor.py
from pypdf import PdfReader import docx
from app.application.services.text_extraction import TextExtractor, ExtractedText

class DefaultTextExtractor(TextExtractor):
def extract(self, file_path: str, content_type: str) -> ExtractedText:
if content_type == "application/pdf" or file_path.lower().endswith(".pdf" return self._extract_pdf(file_path)
if file_path.lower().endswith(".docx") or content_type in ("application/v return self._extract_docx(file_path)
# fallback txt
return self._extract_txt(file_path)

def _extract_pdf(self, file_path: str) -> ExtractedText: reader = PdfReader(file_path)
pages = []
for i, page in enumerate(reader.pages, start=1): t = page.extract_text() or ""
if t.strip():
pages.append(f"\n--- Page {i} ---\n{t}") text = "\n".join(pages).strip()
return ExtractedText(text=text, metadata={"pages": len(reader.pages), "ty

def _extract_docx(self, file_path: str) -> ExtractedText: d = docx.Document(file_path)
paras = [p.text for p in d.paragraphs if p.text.strip()] text = "\n".join(paras).strip()
return ExtractedText(text=text, metadata={"paragraphs": len(paras), "type

def _extract_txt(self, file_path: str) -> ExtractedText: for enc in ("utf-8", "utf-8-sig", "latin-1"):
try:
 
with open(file_path, "r", encoding=enc) as f: text = f.read().strip()
return ExtractedText(text=text, metadata={"encoding": enc, "type" except UnicodeDecodeError:
continue
raise ValueError("Unable to decode text file")





Chunking (Token-aware + overlap + safe) — Pure (3
Service
app/application/services/chunking.py

 

enc = tiktoken.get_encoding(encoding_name) tokens = enc.encode(text)

chunks: List[str] = [] start = 0
max_t = max(50, spec.max_tokens)
overlap = min(spec.overlap_tokens, max_t - 1)

while start < len(tokens):
end = min(start + max_t, len(tokens)) chunk_tokens = tokens[start:end]
chunk = enc.decode(chunk_tokens).strip() if chunk:
chunks.append(chunk) if end == len(tokens):
break
start = end - overlap return chunks



Embeddings Port + OpenAI Adapter + Redis Cache (4
app/application/ports/embeddings.py

python

# app/application/ports/embeddings.py
from typing import Protocol, Sequence

class EmbeddingsPort(Protocol):
def embed_one(self, text: str) -> list[float]: ...
def embed_many(self, texts: Sequence[str]) -> list[list[float]]: ...


 

# app/application/ports/cache.py
from typing import Protocol, Optional

class CachePort(Protocol):
def get_json(self, key: str) -> Optional[dict]: ...
def set_json(self, key: str, value: dict, ttl_seconds: int) -> None: ...

 
Copy code  
 
app/adapters/cache/redis_cache.py
 

python

# app/adapters/cache/redis_cache.py
import json import redis
from app.application.ports.cache import CachePort

class RedisCache(CachePort):
def   init  (self, redis_url: str) -> None:
self._r = redis.Redis.from_url(redis_url, decode_responses=True)

def get_json(self, key: str) -> dict | None: v = self._r.get(key)
return json.loads(v) if v else None

def set_json(self, key: str, value: dict, ttl_seconds: int) -> None: self._r.setex(key, ttl_seconds, json.dumps(value))

app/adapters/embeddings/openai_embeddings.py

 

def embed_one(self, text: str) -> list[float]:
resp = self._client.embeddings.create(model=self._model, input=text) return resp.data[0].embedding

def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
resp = self._client.embeddings.create(model=self._model, input=list(texts return [d.embedding for d in resp.data]


app/application/services/embedding_cache.py (pure-ish utility)



VectorStore Port + Qdrant Adapter (init + upsert + (5
search)
 

Copy code  
 
app/application/ports/vector_store.py
python
 

# app/application/ports/vector_store.py
from typing import Protocol, Sequence
from app.domain.entities import Chunk, TenantId

class VectorStorePort(Protocol):
def ensure_collection(self) -> None: ...
def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> def search(self, query_vector: list[float], tenant_id: TenantId, top_k: int)

C	C

app/adapters/vector/qdrant_store.py
Copy code	python

# app/adapters/vector/qdrant_store.py
from typing import Sequence, List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, Fie from app.application.ports.vector_store import VectorStorePort
from app.domain.entities import Chunk, TenantId, DocumentId

 
class QdrantVectorStore(VectorStorePort):
def   init  (self, client: QdrantClient, collection: str, vector_size: self._client = client
self._collection = collection self._size = vector_size
 

int) -
 

def ensure_collection(self) -> None:
# create if not exists
exists = self._client.collection_exists(self._collection) if not exists:
self._client.create_collection(
collection_name=self._collection,
vectors_config=VectorParams(size=self._size, distance=Distance.CO
)

def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) ->
 
points: List[PointStruct] = []
for c, v in zip(chunks, vectors): points.append(
PointStruct(
id=c.id, vector=v, payload={
"text": c.text,
"tenant_id": c.tenant_id.value,
"document_id": c.document_id.value,
},
)
)
self._client.upsert(collection_name=self._collection, points=points)

def search(self, query_vector: list[float], tenant_id: TenantId, top_k: int) flt = Filter(
must=[
FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id.
]
)
res = self._client.search(
collection_name=self._collection, query_vector=query_vector,
query_filter=flt, limit=top_k,
)

out: list[Chunk] = [] for r in res:
payload = r.payload or {} out.append(
Chunk(
id=str(r.id),
tenant_id=TenantId(payload.get("tenant_id", "")),
document_id=DocumentId(payload.get("document_id", "")), text=payload.get("text", ""),
)
 
)
return out





LLM Port + OpenAI Adapter (6
app/application/ports/llm.py

python

# app/application/ports/llm.py
from typing import Protocol

class LLMPort(Protocol):
def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> st


C	C

 
Copy code  
 
app/adapters/llm/openai_llm.py
 

 
 
)
return resp.choices[0].message.content or ""





 
Copy code  
 
Use Case: AskQuestion (RAG) (7
 
app/application/services/prompt_builder.py (guardrails + citations)

app/application/use_cases/ask_question.py

 
tenant_id: str question: str top_k: int = 5

 
class AskQuestionUseCase:
def  init (self, cached_embeddings: self._emb = cached_embeddings
self._vector = vector_store self._llm = llm
 

CachedEmbeddings, vector_store:
 

VectorS
 

def execute(self, req: AskQuestionRequest) -> Answer: tenant = TenantId(req.tenant_id)
q_vec = self._emb.embed_one(req.question)

chunks = self._vector.search(q_vec, tenant_id=tenant, top_k=req.top_k) prompt = build_rag_prompt(question=req.question, chunks=chunks)

text = self._llm.generate(prompt, temperature=0.2, max_tokens=700) sources = [c.id for c in chunks]
return Answer(text=text, sources=sources)





Parse → Chunk →) اﻟﺤﻘﻴﻘﻲ Worker: IndexDocument (8
(Embed → Upsert
app/workers/tasks.py ﺗﺤﺪﻳﺚ

 

@celery_app.task(
name="index_document", bind=True,
autoretry_for=(Exception,), retry_backoff=True,
retry_kwargs={"max_retries": 5},
)
def index_document(self, *, tenant_id: str, document_id: str) -> dict: c = get_container()
repo = c["document_repo"]
extractor = c["text_extractor"]
cached_emb = c["cached_embeddings"] vector_store = c["vector_store"]

tenant = TenantId(tenant_id)
doc_id = DocumentId(document_id)

repo.set_status(tenant_id=tenant, document_id=doc_id, status="processing")

try:
# For this minimal phase, repo is in-memory, so get stored_file from it
doc = repo.get(doc_id) if not doc:
raise ValueError("Document not found") stored = doc["stored_file"]
extracted = extractor.extract(stored.path, stored.content_type) if not extracted.text.strip():
raise ValueError("No text extracted from file")

# chunking
chunks_text = chunk_text_token_aware(extracted.text, spec=ChunkSpec(max_t

# embed + upsert
vector_store.ensure_collection()

chunks: list[Chunk] = []
vectors: list[list[float]] = []
 
for t in chunks_text:
cid = str(uuid.uuid4())
chunks.append(Chunk(id=cid, tenant_id=tenant, document_id=doc_id, tex vectors.append(cached_emb.embed_one(t))

vector_store.upsert(chunks, vectors)

repo.set_status(tenant_id=tenant, document_id=doc_id, status="indexed") log.info("indexed_document", tenant_id=tenant_id, document_id=document_id return {"ok": True, "chunks": len(chunks)}

except Exception as e:
repo.set_status(tenant_id=tenant, document_id=doc_id, status="failed", er log.exception("index_failed", tenant_id=tenant_id, document_id=document_i raise



API: Ask Endpoint (Thin) + DTO (9
app/api/v1/routes_queries.py

 

 
router لإﺿﺎﻓﺔ app/main.py ﺗﺤﺪﻳﺚ



Adapters/UseCases ﻛﻞ ﻟﺮﺑﻂ DI Container ﺗﺤﺪﻳﺚ (10
app/core/bootstrap.py )اﺳﺘﺒﺪال اﻟﻨﺴﺨﺔ اﻟﺴﺎﺑﻘﺔ(

 

from app.core.config import settings

from app.adapters.filestore.local_store import LocalFileStore
from app.adapters.persistence.inmemory.document_repo import InMemoryDocumentRepo from app.adapters.queue.celery_queue import CeleryTaskQueue

from app.adapters.extraction.default_extractor import DefaultTextExtractor from app.adapters.cache.redis_cache import RedisCache

from app.adapters.embeddings.openai_embeddings import OpenAIEmbeddings from app.application.services.embedding_cache import CachedEmbeddings

from app.adapters.vector.qdrant_store import QdrantVectorStore from app.adapters.llm.openai_llm import OpenAILLM

from app.workers.celery_app import celery_app

from app.application.use_cases.upload_document import UploadDocumentUseCase from app.application.use_cases.ask_question import AskQuestionUseCase

@lru_cache(maxsize=1)
def get_container() -> dict:
# --- repositories / stores
document_repo = InMemoryDocumentRepo()
file_store = LocalFileStore(upload_dir=settings.upload_dir, max_mb=settings.m queue = CeleryTaskQueue(celery_app)

# --- extraction
text_extractor = DefaultTextExtractor()

# --- cache
cache = RedisCache(settings.redis_url)

# --- embeddings (OpenAI for now)
if not settings.openai_api_key:
# In production: raise at startup. Here: let it fail on first call if mis
pass
embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key or "", model=se cached_embeddings = CachedEmbeddings(embeddings=embeddings, cache=cache, ttl_
 


# --- vector store
qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port) vector_store = QdrantVectorStore(
client=qdrant,
collection=settings.qdrant_collection, vector_size=settings.embedding_dim,
)

# --- llm
llm = OpenAILLM(api_key=settings.openai_api_key or "", model=settings.openai_

# --- use cases
upload_uc = UploadDocumentUseCase(file_store=file_store, repo=document_repo, ask_uc = AskQuestionUseCase(cached_embeddings=cached_embeddings, vector_store

return {
"document_repo": document_repo, "file_store": file_store,
"task_queue": queue,
"text_extractor": text_extractor, "cache": cache,
"embeddings": embeddings,
"cached_embeddings": cached_embeddings, "vector_store": vector_store,
"llm": llm,
"upload_use_case": upload_uc, "ask_use_case": ask_uc,
}





(API + Redis + Qdrant + Worker) ﻛﺎﻣﻞ ﺗﺸﻐﻴﻞ (11
(ﻟﻠﺘﺸﻐﻴﻞ ﺟًﺪا ﻣﻔﻴﺪ) docker-compose
: docker/docker-compose.yml

yaml
 

version: "3.9" services:
redis:
image: redis:7-alpine ports: ["6379:6379"]

qdrant:
image: qdrant/qdrant:latest ports: ["6333:6333"]
volumes:
- qdrant_data:/qdrant/storage

volumes:
qdrant_data:

ﺗﺸﻐﻴﻞ:


 
Copy code  
 
ﺛﻢ:
 

bash

# API
uvicorn app.main:app --reload --host 0.0.0.0 --port 5000

# Worker
celery -A app.workers.celery_app.celery_app worker -Q indexing --loglevel=INFO



 
Copy code  
 
End-to-End اﺧﺘﺒﺎر (12
Upload
 

bash
 

curl -X POST "http://localhost:5000/v1/documents/upload" \
-H "X-API-KEY=tenant_demo_12345678" \
-F "file=@./sample.pdf"
Copy code

Ask

bash

curl -X POST "http://localhost:5000/v1/queries/ask" \
-H "Content-Type: application/json" \
-H "X-API-KEY=tenant_demo_12345678" \
","top_k":5}'ﻣﺎ اﻟﮭﺪف ﻣﻦ اﻟﻤﺸﺮوع؟'{"question":" -d


(13 ﻣلاﺣﻈﺎت Production )ﻣﻬﻢ ﺟﺪًا(
اﻟﻨﺴﺨﺔ دي skeleton Production-ready، ﻟﻜﻦ ﻟﺴﻪ ﻧﺎﻗﺺ 3 ﺗﺮﻗﻴﺎت “لازم” ﻗﺒﻞ أي ﻧشر ﺣﻘﻴﻘﻲ:
Users/Docs/Chunks/Chats + multi-) Postgres + Alembic ﺑـ InMemoryRepo اﺳﺘﺒﺪال .1
(ﺣﻘﻴﻘﻲ tenant (Rate limiting/دور/ﻣﺨﺰﻧﺔ API keys) ﺣﻘﻴﻘﻲ Auth .2
Observability (metrics/tracing + structured logs request_id) .3


اﻟﺨﻄﻮة اﻟﺠﺎﻳﺔ
:ﻟﻚ ṇﻫﻜﺘ ”Postgres + Alembic ﻛﻤﻞ“ ﺗﻘﻮل ﻟﻮ SQLAlchemy models (users, documents, chunks, chats)	 
DocumentRepoPort ﺗﻄﺒﻖ repos	 
ﺟﺎﻫﺰة migrations	  DB ﻣﻦ API key lookup ﺑـ auth رﺑﻂ	 
ﻣﻀﺒﻮط chunk metadata + status transitions ﺣﻔﻆ	 
أوًلا؟ Hybrid search + rerank ولا Postgres + Alembic ﺑـ ﻣﺒﺎشرة أﺑﺪأ ṇﺗﺤ


 


Postgres + Alembic ﻛﻤﻞ


Clean Architecture / ﺑـ وﻣﻠﺘﺰم Production-ready ﺑﺸﻜﻞ Postgres + Alembic ﻧﻜّﻤﻞ — ﺗﻤﺎم
:SOLID
Auth اﻟـ ﻧﺮﺑﻂ وﻛﻤﺎن ،SQLAlchemy Repo + Migrations ﺑـ InMemoryDocumentRepo ﻫﻨﺴﺘﺒﺪل
.(ﺣﻘﻴﻘﻲ multi-tenant) Users ﺟﺪول ﺑـ (API Key)
اﻟﻨﺘﻴﺠﺔ:
ﻳﺤّﺪث → DB ﻣﻦ stored_file ﻳﻘﺮأ Document record → Celery worker ﻳﺤﻔﻆ → Upload
(ṇﺗﺤ ﻟﻮ chunks metadata ﻧﻀﻴﻒ لاﺣًﻘﺎ) → status
.(body ﻣﻦ ﻣﺶ) User.api_key ﻣﻦ ﻳﺄﺗﻲ TenantId واﻟـ


Dependencies (1
أﺿﻒ:

txt

sqlalchemy>=2.0
psycopg[binary]>=3.1 alembic>=1.13

Settings: DATABASE_URL (2
: env.example. ﻓﻲ

bash

DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag

 
Copy code  
 

:أﺿﻒ app/core/config.py وﻓﻲ
 

python
 
Copy code
database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/rag"


C	C


SQLAlchemy Base + Engine + Session (3
app/adapters/persistence/postgres/db.py

python

# app/adapters/persistence/postgres/db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase from app.core.config import settings

class Base(DeclarativeBase): pass

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)



 


Copy code  
 
ORM Models (Users + Documents) (4
app/adapters/persistence/postgres/models.py
 

python

# app/adapters/persistence/postgres/models.py
from sqlalchemy import String, Text, Integer, DateTime, ForeignKey, func, Index from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.adapters.persistence.postgres.db import Base

class User(Base):
  tablename	 = "users"

id: Mapped[str] = mapped_column(String(36), primary_key=True)
email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
 
api_key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False created_at: Mapped["DateTime"] = mapped_column(DateTime(timezone=True), serve documents = relationship("Document", back_populates="user", cascade="all, del
Index("ix_users_api_key", User.api_key)



class Document(Base):
  tablename	 = "documents"

id: Mapped[str] = mapped_column(String(36), primary_key=True)
user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondel

filename: Mapped[str] = mapped_column(String(512), nullable=False)
content_type: Mapped[str] = mapped_column(String(128), nullable=False) file_path: Mapped[str] = mapped_column(Text, nullable=False)
size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

status: Mapped[str] = mapped_column(String(32), nullable=False, default="crea error: Mapped[str | None] = mapped_column(Text, nullable=True)

created_at: Mapped["DateTime"] = mapped_column(DateTime(timezone=True), serve updated_at: Mapped["DateTime"] = mapped_column(DateTime(timezone=True), serve

user = relationship("User", back_populates="documents")

Index("ix_documents_user_id", Document.user_id) Index("ix_documents_status", Document.status)


.user_id ﺑـ ﻧﻔﻠﺘﺮ لازم query ﻛﻞ ﻓﻲ . user_id = tenant boundary :لاﺣﻆ


Alembic Setup (5
5.1 إﻧﺸﺎء Alembic )ﻣﺮة واﺣﺪة(
:root ﻣﻦ
 
bash

alembic init app/adapters/persistence/postgres/migrations

 
Copy code   Copy code  
 
alembic.ini 5.2
.(practice أﻓﻀﻞ) env ﻣﻦ ﻟﻴﻘﺮأ sqlalchemy.url ﻋّﺪل
.env.py ﻓﻲ env ﻣﻦ ﻫﻨﺤﻘﻨﻪ وإﺣﻨﺎ ،placeholder ﺧﻠﻴﻪ alembic.ini ﻓﻲ
migrations/env.py 5.3
اﺳﺘﺒﺪل ﻣﺤﺘﻮى env.py ﺑـ:
 

python

# app/adapters/persistence/postgres/migrations/env.py
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool from alembic import context
import os

from app.adapters.persistence.postgres.db import Base
from app.adapters.persistence.postgres import models # noqa: F401	(ensure model

config = context.config

if config.config_file_name is not None: fileConfig(config.config_file_name)

target_metadata = Base.metadata

def get_url() -> str:
# Prefer DATABASE_URL from env url = os.getenv("DATABASE_URL") if not url:
raise RuntimeError("DATABASE_URL is not set") return url

def run_migrations_offline() -> None: context.configure(
url=get_url(),
 
target_metadata=target_metadata, literal_binds=True,
dialect_opts={"paramstyle": "named"}, compare_type=True,
)
with context.begin_transaction(): context.run_migrations()

def run_migrations_online() -> None:
configuration = config.get_section(config.config_ini_section) or {} configuration["sqlalchemy.url"] = get_url()

connectable = engine_from_config( configuration,
prefix="sqlalchemy.",
poolclass=pool.NullPool,
)

with connectable.connect() as connection: context.configure(
connection=connection,
target_metadata=target_metadata, compare_type=True,
)

with context.begin_transaction(): context.run_migrations()

if context.is_offline_mode(): run_migrations_offline()
else:
run_migrations_online()


Migration أول 5.4

bash

alembic revision --autogenerate -m "create users and documents" alembic upgrade head
 

 
Postgres Repository Adapter (implements (6
DocumentRepoPort)
.worker ﻟﻠـ document + stored_file ﻧﻘﺮأ ﻛﻤﺎن ﻧﺤﺘﺎج
app/adapters/persistence/postgres/repo_documents.py

python

# app/adapters/persistence/postgres/repo_documents.py
import uuid
from sqlalchemy import select, update
from app.adapters.persistence.postgres.db import SessionLocal from app.adapters.persistence.postgres.models import Document
Copy code
from app.application.ports.document_repo import DocumentRepoPort
from app.domain.entities import DocumentId, TenantId, StoredFile

class PostgresDocumentRepo(DocumentRepoPort):
def create_document(self, *, tenant_id: TenantId, stored_file: StoredFile) -> doc_id = DocumentId(str(uuid.uuid4()))
with SessionLocal() as db: db.add(
Document(
id=doc_id.value,
user_id=tenant_id.value,	# tenant_id == user_id in o
filename=stored_file.filename,
content_type=stored_file.content_type, file_path=stored_file.path,
size_bytes=stored_file.size_bytes, status="created",
)
)
db.commit() return doc_id

def set_status(self, *, tenant_id: TenantId, document_id: DocumentId, status: with SessionLocal() as db:
stmt = (
update(Document)
.where(Document.id == document_id.value, Document.user_id == tena
 
.values(status=status, error=error)
)
db.execute(stmt) db.commit()

def get_stored_file(self, *, tenant_id: TenantId, document_id: DocumentId) -> with SessionLocal() as db:
stmt = select(Document).where(Document.id == document_id.value, Docum doc = db.execute(stmt).scalar_one_or_none()
if not doc:
return None
return StoredFile(
path=doc.file_path,
filename=doc.filename,
content_type=doc.content_type, size_bytes=doc.size_bytes,
)


Clean ﻓﻲ . get_stored_file ﻓﻴﻪ ﻛﺎﻧﺶ ﻣﺎ اﻟﺴﺎﺑﻖ ﻓﻲ DocumentRepoPort :ﻣلاﺣﻈﺔ
:الأﻓﻀﻞ Architecture

DocumentReadPort ﻣﻨﻔﺼﻞ Port ﻧﻀﻴﻒ إﻣﺎ	 
  أو ﻧﻮﺳّﻊ اﻟـ DocumentRepoPort ﺑﺸﻜﻞ ﻣﺤﺪود. أﻧﺎ ﻫﻌﻤﻠﻪ ﺑﺸﻜﻞ SOLID أﻓﻀﻞ: Port ﺟﺪﻳﺪ.


(7 إﺿﺎﻓﺔ Port ﻗﺮاءة (ISP)
app/application/ports/document_reader.py

 
class DocumentReaderPort(Protocol):
def get_stored_file(self, *, tenant_id: TenantId, document_id: DocumentId) ->
Copy code  

.(الاﺗﻨﻴﻦ implements واﺣﺪة class ﺑﺒﺴﺎﻃﺔ) الاﺛﻨﻴﻦ ﻳﻄﺒﻖ PostgresDocumentRepo واﺟﻌﻞ

Auth: API Key → user_id (tenant_id) (8
ﺑﺪل ﻣﺎ ﻧﻌﺎﻣﻞ اﻟـ api_key ﻫﻮ tenant ﻣﺒﺎشرة، ﻧﻌﻤﻞ lookup ﻓﻲ .Postgres
C	app/adapters/persistence/postgres/repo_users.pCy

python

# app/adapters/persistence/postgres/repo_users.py
from sqlalchemy import select
from app.adapters.persistence.postgres.db import SessionLocal from app.adapters.persistence.postgres.models import User

class UserLookupRepo:
def get_user_id_by_api_key(self, api_key: str) -> str | None: with SessionLocal() as db:
stmt = select(User.id).where(User.api_key == api_key) return db.execute(stmt).scalar_one_or_none()

(ﺗﺤﺪﻳﺚ) app/api/v1/deps.py

 
if not user_id:
raise HTTPException(status_code=401, detail="Invalid API key")

# tenant_id == user_id
return user_id





InMemory ﺑﺪل DB ﻣﻦ StoredFile ﻟﻘﺮاءة Worker ﺗﺤﺪﻳﺚ (9
:doc ṇﺟﻠ ﺟﺰء ﻋّﺪل app/workers/tasks.py ﻓﻲ












Postgres Repos + ﺑـ InMemory اﺳﺘﺒﺪال :Bootstrap (10
UserLookup
(الأﺳﺎﺳﻴﺔ اﻟﺘﻐﻴﻴﺮات) app/core/bootstrap.py

python

# app/core/bootstrap.py (only show changed parts)
from app.adapters.persistence.postgres.repo_documents import PostgresDocumentRepo from app.adapters.persistence.postgres.repo_users import UserLookupRepo

@lru_cache(maxsize=1)
def get_container() -> dict:
# --- postgres repos
document_repo = PostgresDocumentRepo()
document_reader = document_repo # implements DocumentReaderPort
user_lookup_repo = UserLookupRepo()

# ... اﻟـ ﺑﺎﻗﻲ adapters (file_store, queue, extractor, cache, embeddings, vecto
 


upload_uc = UploadDocumentUseCase(file_store=file_store, repo=document_repo, ask_uc = AskQuestionUseCase(cached_embeddings=cached_embeddings, vector_store

return {
"document_repo": document_repo,
"document_reader": document_reader,
"user_lookup_repo": user_lookup_repo,
# ... rest
"upload_use_case": upload_uc, "ask_use_case": ask_uc,
}





Docker Compose: Postgres + Redis + Qdrant (11
docker/docker-compose.yml

 
volumes:
- qdrant_data:/qdrant/storage

volumes:
pg_data:
qdrant_data:

ﺗﺸﻐﻴﻞ:

bash

cd docker
docker compose up -d

 
Copy code  
 
:migrations ﺛﻢ









ﻟلاﺧﺘﺒﺎر سرﻳﻊ Seed User (API Key) (12
اﻛﺘṇ ﺳﻜﺮﻳﺒﺖ ﺑﺴﻴﻂ:
scripts/seed_user.py
 

python

# scripts/seed_user.py
import uuid
from app.adapters.persistence.postgres.db import SessionLocal from app.adapters.persistence.postgres.models import User

def main():
user_id = str(uuid.uuid4())
api_key = "tenant_demo_12345678" # test key
email = "demo@example.com"
 
with SessionLocal() as db:
db.add(User(id=user_id, email=email, api_key=api_key)) db.commit()

print("Seeded:", {"user_id": user_id, "api_key": api_key, "email": email})

if  name	 == " main ": main()

ﺗﺸﻐﻴﻞ:

bash

python scripts/seed_user.py

Postgres ﺑﻌﺪ Test Flow (13
Seed user .1 X-API-KEY=tenant_demo_12345678 ﺑــ Upload .2 DB ﻣﻦ indexing ﻳﺸﺘﻐﻞ Worker .3
results ṇﻳﺠﻴ Ask endpoint  .4


اﻟﻠﻲ ﻧﺎﻗﺺ ﻟﻴﺒﻘﻰ Production-grade“ ﺗﻤﺎﻣًﺎ”
.ﻧﻈﻴﻒ ﺑﺸﻜﻞ ”Postgres + Alembic” + “Auth lookup“ ﻧﻔﺬت ﻛﺪه أﻧﺎ
اﻟﺨﻄﻮة اﻟﺘﺎﻟﻴﺔ )أﻗﻮى ﺗﺮﻗﻴﺔ ﻟﻠﺠﻮدة/اﻟﺘﺘﺒﻊ:(
Chunks table (metadata + chunk_id + offsets + hash) .1
Chat history table (question/answer/sources/tokens/latency) .2
.3 hash document Idempotency: ﻳﻤﻨﻊ إﻋﺎدة اﻟﻔﻬﺮﺳﺔ ﻟﻨﻔﺲ اﻟﻤﻠﻒ
(أﻛﺜﺮ multi-tenant ﺗﺸﺪد ﻋﺎﻳﺰ ﻟﻮ) Row-level policies .4
و worker اﻟـ ﻓﻲ واﺳﺘﺨﺪاﻣﻬﻢ reposو migrations ﻫﻨﻀﻴﻒ ”chunks + chat tables ﻛﻤﻞ“ ﺗﻘﻮل ﻟﻮ
.ask

 


Hybrid search + rerank


وﺑـ ،Clean Architecture اﻟـ ﻧﻔﺲ داﺧﻞ ﻧﻈﻴﻒ ﺑﺸﻜﻞ Hybrid Search + Rerank ﻫﻨﻀﻴﻒ — ﺗﻤﺎم
.Production-grade knobs (latency/cost/quality)

ﻫﻨﺒﻨﻴﻬﺎ ﻋﻠﻰ 3 ﻃﺒﻘﺎت:

Candidate Generation (Recall) .1
Vector Search (Qdrant / PGVector)	  (وﻋﻤﻠﻲ سرﻳﻊ) tsvector ﺑﺎﺳﺘﺨﺪام Postgres داﺧﻞ Keyword Search (BM25-ish)	 
(weighted أو RRF :اﻟﻨﺘﺎﺋﺞ دﻣﺞ) Fusion .2
Rerank (Precision) .3
(أﻏﻠﻰ ﻟﻜﻦ أﺳﻬﻞ) LLM-as-reranker أو (أﻓﻀﻞ) Cross-Encoder	 

(1 اﻟﻤﻌﻤﺎرﻳﺔ اﻟﻤﺴﺘﻬﺪﻓﺔ

pgsql

question
→ embed(question)
→ vector_candidates = VectorStore.search(top_k=K_vec)
→ keyword_candidates = KeywordStore.search(top_k=K_kw)
→ fused = Fusion(vector_candidates, keyword_candidates)
→ reranked = Reranker.rank(question, fused)	# top_n
→ build_prompt(question, reranked)
→ LLM answer

إﻋﺪادات واﻗﻌﻴﺔ:

 

Copy code  
 
K_vec = 30
K_kw = 30
 
fused_limit = 40
(ﺣﺴﺎس latency ﻟﻮ 5 أو) rerank_top_n = 8


 
(SOLID/ISP) ﺟﺪﻳﺪة Ports (2
Keyword Search Port 2.1

python

# app/application/ports/keyword_store.py
from typing import Protocol, Sequence
from app.domain.entities import Chunk, TenantId

class KeywordStorePort(Protocol):
def search(self, *, query: str, tenant_id: TenantId, top_k: int) -> Sequence[


C	C

 

Copy code  
 
Fusion Service (pure) 2.2
 

python

# app/application/services/fusion.py
from dataclasses import dataclass from typing import Sequence
from app.domain.entities import Chunk

@dataclass(frozen=True) class ScoredChunk:
chunk: Chunk score: float

def rrf_fusion(
*,
vector_hits: Sequence[ScoredChunk], keyword_hits: Sequence[ScoredChunk], k: int = 60,
out_limit: int = 40,
) -> list[ScoredChunk]: """
Reciprocal Rank Fusion (RRF): robust, no score calibration needed. score = Σ 1 / (k + rank)
"""
acc: dict[str, float] = {}
 
def add(hits: Sequence[ScoredChunk]):
for rank, h in enumerate(hits, start=1):
acc[h.chunk.id] = acc.get(h.chunk.id, 0.0) + 1.0 / (k + rank)

add(vector_hits) add(keyword_hits)

# keep best
scored = sorted(acc.items(), key=lambda x: x[1], reverse=True)[:out_limit]
# reconstruct chunks by id
by_id = {h.chunk.id: h.chunk for h in list(vector_hits) + list(keyword_hits)} return [ScoredChunk(chunk=by_id[cid], score=s) for cid, s in scored if cid in


Reranker Port 2.3

python

# app/application/ports/reranker.py from typing import Protocol, Sequence from app.domain.entities import Chunk

class RerankerPort(Protocol):
def rerank(self, *, query: str, chunks: Sequence[Chunk], top_n: int) -> Seque


C	C


 
Copy code  
 
Postgres Keyword Search Adapter (tsvector) (3
 
.ﺟًﺪا ﻋﻤﻠﻲ BM25-ish ﻛـ Postgres Full-Text Search (FTS) ﻧﺴﺘﺨﺪم ،(ﻛﺒﻴﺮ) Elasticsearch ﺑﺪل
( لاز) chunks ﺟﺪول :DB 3.1
.tenant isolation وﺑـ ﺑسرﻋﺔ ﻳﺸﺘﻐﻞ keyword search ﻋﺸﺎن Postgres ﻓﻲ chunks ﻧﺨﺰن لازم

:(ﻣﺒﺪﺋًﻴﺎ) Migration

C	chunks(id, document_id, user_id, text, tsv, created_at)	C
tsv ﻋﻠﻰ GIN index	 

ORM model
 
python

# app/adapters/persistence/postgres/models_chunks.py
from sqlalchemy import String, Text, DateTime, ForeignKey, func, Index from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import TSVECTOR
from app.adapters.persistence.postgres.db import Base

class ChunkRow(Base):
  tablename	 = "chunks"

id: Mapped[str] = mapped_column(String(36), primary_key=True)
document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondel

text: Mapped[str] = mapped_column(Text, nullable=False)
tsv: Mapped[object] = mapped_column(TSVECTOR, nullable=False)

created_at: Mapped["DateTime"] = mapped_column(DateTime(timezone=True), serve

Index("ix_chunks_user_id", ChunkRow.user_id)
Index("ix_chunks_tsv", ChunkRow.tsv, postgresql_using="gin")


C	C

ﺗﻔﻌﻴﻞ tsv ﺗﻠﻘﺎﺋﻴًﺎ )أﻓﻀﻞ (practice
:migration SQL ﻓﻲ

sql

 
-- tsv = to_tsvector('simple', text)

 


.insert/update code ﻓﻲ أو
Keyword repo 3.2
 

python

# app/adapters/persistence/postgres/keyword_store.py
from sqlalchemy import select, text as sql_text
from app.adapters.persistence.postgres.db import SessionLocal
 
from app.adapters.persistence.postgres.models_chunks import ChunkRow from app.application.ports.keyword_store import KeywordStorePort
from app.domain.entities import Chunk, TenantId, DocumentId

class PostgresKeywordStore(KeywordStorePort):
def search(self, *, query: str, tenant_id: TenantId, top_k: int):
# websearch_to_tsquery اﻟﻄﺒﯿﻌﯿﺔ ﻟﻠﻤﺪﺧﻼت أﻓﻀﻞ
tsq = sql_text("websearch_to_tsquery('simple', :q)")
rank = sql_text("ts_rank_cd(chunks.tsv, websearch_to_tsquery('simple', :q

with SessionLocal() as db: stmt = (
select(ChunkRow.id, ChunkRow.document_id, ChunkRow.user_id, Chunk
.where(ChunkRow.user_id == tenant_id.value)
.where(ChunkRow.tsv.op("@@")(tsq))
.order_by(rank.desc())
.limit(top_k)
)
rows = db.execute(stmt, {"q": query}).all()

out = []
for cid, doc_id, uid, txt in rows: out.append(
Chunk(id=cid, tenant_id=TenantId(uid), document_id=DocumentId(doc
)
return out





 



C












Copy code
 
ﻓﻲ chunks ﺧّﺰن Qdrant ﻟـ upsert ﺑﻌﺪ :Worker ﺗﻌﺪﻳﻞ (4
PostgreCs
:worker indexing ﻓﻲ Postgres (id + text + tsv + user_id + ﻓﻲ chunks ﻟﻠـ insert اﻋﻤﻞ :chunking ﺑﻌﺪ	 
document_id)
.tracing و keyword search ﻳﺨﺪم ده	 
Port ﻟﻠﻜﺘﺎﺑﺔ:
 

python
 

# app/application/ports/chunk_repo.py from typing import Protocol, Sequence from app.domain.entities import Chunk

class ChunkRepoPort(Protocol):
def upsert_chunks(self, *, chunks: Sequence[Chunk]) -> None: ...

:Adapter Postgres

 

Copy code
 
python
 
# app/adapters/persistence/postgres/repo_chunks.py
from sqlalchemy import insert, text as sql_text
from app.adapters.persistence.postgres.db import SessionLocal
from app.adapters.persistence.postgres.models_chunks import ChunkRow from app.application.ports.chunk_repo import ChunkRepoPort
from app.domain.entities import Chunk

class PostgresChunkRepo(ChunkRepoPort): def upsert_chunks(self, *, chunks):
values = []
for c in chunks:
values.append({ "id": c.id,
"document_id": c.document_id.value, "user_id": c.tenant_id.value,
"text": c.text,
# tsv computed in SQL for consistency
})

with SessionLocal() as db:
# Insert rows
db.execute(insert(ChunkRow), values)
# Update tsv in one shot (or define generated column via migration) db.execute(sql_text("UPDATE chunks SET tsv = to_tsvector('simple', te db.commit()

C	C

.update ﻋﻦ ﻓﺘﺴﺘﻐﻨﻰ ،migration ﻓﻲ tsv Generated Column ﺗﺨﻠﻲ :ﻛﺪه ﻣﻦ أﻓﻀﻞ
 

 
(fusion ﻋﺸﺎن) scores ﻧﺤﺘﺎج :Vector Search (5
.scored hits ﻳﺮّﺟﻊ VectorStore adapter ﻧﺤّﺪث .score (similarity) ﺑﻴﺮﺟﻊ Qdrant

ﺑﺪل ﻣﺎ ﻧﻜسر Port اﻟﻘﺪﻳﻢ، ﻧﻀﻴﻒ Port ﺟﺪﻳﺪ أو :DTO


 



Copy code  
 
. (...)VectorStore adapter method search_scored ﺛﻢ

:Port
 

python

# app/application/ports/vector_store.py
from typing import Protocol, Sequence
from app.domain.entities import TenantId
from app.application.services.scoring import ScoredChunk

class VectorStorePort(Protocol):
def ensure_collection(self) -> None: ...
def search_scored(self, query_vector: list[float], tenant_id: TenantId, top_k


C	C

:(ﻣﺨﺘصر) Qdrant implementation

 
def search_scored(...):
res = self._client.search(..., limit=top_k) out=[]
for r in res:
p = r.payload or {}
c = Chunk(... text=p.get("text","") ...)
out.append(ScoredChunk(chunk=c, score=float(r.score))) return out



 

Copy code  
 
(ﺧﻴﺎرات 3) Rerank Adapter (6
(ﺟﻮدة أﻓﻀﻞ) A: Cross-Encoder ﺧﻴﺎر
(bge-reranker ﻣﺜﻞ) reranker ﻧﻤﻮذج + sentence-transformers	 
   سرﻳﻊ ﻧﺴﺒﻴًﺎ ﻋﻠﻰ GPU، وﻋﻠﻰ CPU ﻣﻤﻜﻦ ﻳﺒﻘﻰ ﺑﻄﺊ.
.Port stays same
(أﺳﻬﻞ) B: LLM Rerank ﺧﻴﺎر
ﻧﺨﻠﻲ اﻟـ LLM ﻳﺮﺟّﻊ ﺗﺮﺗﻴṇ IDs ﻓﻘﻂ. )ﻟﻜﻦ cost أﻋﻠﻰ(

:Adapter ﻣﺜﺎل
 

python

# app/adapters/rerank/llm_reranker.py
from app.application.ports.reranker import RerankerPort from app.application.ports.llm import LLMPort
from app.domain.entities import Chunk

class LLMReranker(RerankerPort):
def   init  (self, llm: LLMPort) -> None: self._llm = llm

def rerank(self, *, query: str, chunks: list[Chunk], top_n: int) -> list[Chun items = "\n".join([f"{i}. [{c.id}] {c.text[:400]}" for i,c in enumerate(c prompt = (
"Rank the passages by relevance to the query. Return ONLY the top ids f"Query: {query}\n\nPassages:\n{items}\n"
 
)
raw = self._llm.generate(prompt, temperature=0.0, max_tokens=200)
# parsing safely omitted ﻓﻲ—ھﻨﺎ prod ﻻزم json parse + fallback # fallback: return first top_n
return chunks[:top_n]


C: Lightweight heuristic rerank (fallback) ﺧﻴﺎر
overlap keywords + cosine score + length penalty	 
.fallback ﻛـ ﺟًﺪا سرﻳﻊ	 


 
Copy code  
 
Use Case: AskQuestion — Hybrid + Rerank (7
ﺑﺪل ask اﻟﺤﺎﻟﻲ، ﻫﻨﻌﻤﻞ:
 

 
 
cached_embeddings: CachedEmbeddings, vector_store: VectorStorePort,
keyword_store: KeywordStorePort, reranker: RerankerPort,
llm,
) -> None:
self._emb = cached_embeddings self._vec = vector_store
self._kw = keyword_store self._rerank = reranker self._llm = llm

def execute(self, req: AskHybridRequest) -> Answer: tenant = TenantId(req.tenant_id)
q_vec = self._emb.embed_one(req.question)

vec_hits = self._vec.search_scored(q_vec, tenant_id=tenant, top_k=req.k_v kw_chunks = self._kw.search(query=req.question, tenant_id=tenant, top_k=r

# Convert kw to scored by rank (RRF doesn't need calibrated scores)
from app.application.services.scoring import ScoredChunk
kw_hits = [ScoredChunk(chunk=c, score=1.0) for c in kw_chunks]

fused = rrf_fusion(vector_hits=vec_hits, keyword_hits=kw_hits, out_limit= fused_chunks = [s.chunk for s in fused]

reranked = self._rerank.rerank(query=req.question, chunks=fused_chunks, t

prompt = build_rag_prompt(question=req.question, chunks=reranked) text = self._llm.generate(prompt, temperature=0.2, max_tokens=700)

return Answer(text=text, sources=[c.id for c in reranked])





ﺟﺪﻳﺪ API Endpoint (8
v1/queries/ask-hybrid/

python
 

 

Bootstrap wiring (9
:أﺿﻒ ()get_container ﻓﻲ

 




Copy code  
 
()keyword_store = PostgresKeywordStore	 
(worker ﻳﺴﺘﺨﺪﻣﻪ) ()chunk_repo = PostgresChunkRepo	  (CrossEncoderReranker أو LLMReranker(llm)) = reranker	 
(...)ask_hybrid_use_case = AskQuestionHybridUseCase	 



(اﻟﻨﻘﺎط أﻫﻢ) Production tuning (10
:Latency	 
   رﻳﺮﻧﻚ top-8 → top-40 ﻣﻤﺘﺎز.
 
أو أﻗﻞ rerank_top_n ﺧﻠﻲ → bottleneck ﻳﺒﻘﻰ ﻣﻤﻜﻦ CPU ﻋﻠﻰ Cross-encoder	 
.low-confidence ﻋﻨﺪ ﻓﻘﻂ LLM rerank اﺳﺘﺨﺪم
:Quality	 
   Hybrid ﻳﺮﻓﻊ Recall ﺟﺪًا ﺧﺼﻮﺻًﺎ ﻟلأﺳﻤﺎء/الأرﻗﺎم/اﻟﺘﻮارﻳﺦ.
.hallucinationو ”context drift“ ﻳﻘﻠﻞ Rerank	 
:Multi-tenant	  Keyword: WHERE user_id = tenant_id	 
Vector: filter payload tenant_id	 
:Cost control	 
(chunks + ﺳﺆال) Cache embeddings	  (ﻛﺒﻴﺮ ﺗﺤﺴﻴﻦ) embed_one per chunk ﺑﺪل indexing ﻓﻲ Batch embeddings	 


”chunks + generated tsv ﺑـ اﻟﺨﺎﺻﺔ migrations ﻛﻤﻞ“ ﺗﻘﻮل ﻟﻮ
أﻧﺎ أﻛﺘṇ ﻟﻚ:
(chunks table + generated tsv + gin index) ﻛﺎﻣﻠﺔ Alembic migration	  indexing أﺛﻨﺎء Postgres ﻓﻲ chunks ṇﻟﻴﻜﺘ worker ﺗﻌﺪﻳﻞ	 
الاﺗﺴﺎق ﺑﻨﻔﺲ chunk_id/doc_id/user_id ﻳﺤﺘﻮي ﺑﺤﻴﺚ Qdrant payload ﺗﻌﺪﻳﻞ	 
سرﻳﻊ؟ ﻛﺤﻞ LLM Rerank ولا ﻣﺤﻠﻲ Cross-Encoder ﻳﻜﻮن reranker ṇﺗﺤ




chunks ﺑـ اﻟﺨﺎﺻﺔ migrations ﻛﻤﻞ chunks + chat tables ﻛﻤﻞ
+ generated tsv


Thought for 5s  
ﺑﺪون) generated tsvector ﻣﻊ Chunks + Chat Tables + Alembic migrations ﻫﻨﻜّﻤﻞ — ﺗﻤﺎم
.hybrid keyword search واﻟـ worker ﻟﻠـ Wiring + (ﻳﺪوي update

ﻫﻨﺎ ﻫﺪﻳﻚ:

ORM Models (ChunkRow + ChatSession + ChatTurn) .1
(generated tsv + indexes) ﺟﺎﻫﺰة Alembic Migration .2
 
Repos/Ports (ChunkRepo + ChatRepo) .3 (indexing أﺛﻨﺎء Postgres ﻓﻲ chunks ﻳﺤﻔﻆ) Worker modifications .4 tsv generated ﻋﻠﻰ ﻳﻌﺘﻤﺪ KeywordStore (Postgres FTS) .5


ORM Models (1
app/adapters/persistence/postgres/models_chunks.py 1.1
python

from sqlalchemy import String, Text, DateTime, ForeignKey, func, Index from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import TSVECTOR
from app.adapters.persistence.postgres.db import Base

class ChunkRow(Base):
  tablename	 = "chunks"

id: Mapped[str] = mapped_column(String(36), primary_key=True)

document_id: Mapped[str] = mapped_column( String(36),
ForeignKey("documents.id", ondelete="CASCADE"), nullable=False,
)
user_id: Mapped[str] = mapped_column( String(36),
ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
)

text: Mapped[str] = mapped_column(Text, nullable=False)

# NOTE: this will be GENERATED ALWAYS AS ... STORED via migration. tsv: Mapped[object] = mapped_column(TSVECTOR, nullable=False)

created_at: Mapped["DateTime"] = mapped_column(DateTime(timezone=True), serve

Index("ix_chunks_user_id", ChunkRow.user_id)
 
Index("ix_chunks_document_id", ChunkRow.document_id)
Index("ix_chunks_tsv", ChunkRow.tsv, postgresql_using="gin")





Chat Tables (Session + Turn) 1.2

app/adapters/persistence/postgres/models_chat.py

python

from sqlalchemy import String, Text, Integer, DateTime, ForeignKey, func, Index from sqlalchemy.orm import Mapped, mapped_column
from app.adapters.persistence.postgres.db import Base

CcolpayscsodCehatSession(Base):
  tablename	 = "chat_sessions"

id: Mapped[str] = mapped_column(String(36), primary_key=True)
user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondel title: Mapped[str | None] = mapped_column(String(200), nullable=True)
created_at: Mapped["DateTime"] = mapped_column(DateTime(timezone=True), serve Index("ix_chat_sessions_user_id", ChatSession.user_id)

class ChatTurn(Base):
  tablename	 = "chat_turns"

id: Mapped[str] = mapped_column(String(36), primary_key=True)
session_id: Mapped[str] = mapped_column(String(36), ForeignKey("chat_sessions
C	C
user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondel

question: Mapped[str] = mapped_column(Text, nullable=False) answer: Mapped[str] = mapped_column(Text, nullable=False)

# store sources as a compact JSON string (we can move to JSONB later)
sources_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
 


# observability/cost fields (optional but useful)
retrieval_k: Mapped[int] = mapped_column(Integer, nullable=False, default=5) embed_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
search_ms: Mapped[int | None] = mapped_column(Integer, nullable=True) llm_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True) created_at: Mapped["DateTime"] = mapped_column(DateTime(timezone=True), serve
Index("ix_chat_turns_user_id", ChatTurn.user_id)
Index("ix_chat_turns_session_id", ChatTurn.session_id) Index("ix_chat_turns_created_at", ChatTurn.created_at)




Alembic Migration (generated tsv + chat tables) (2
2.1 ﺗﺄﻛﺪ إن env.py ﺑﻴﺴﺘﻮرد ﻛﻞ models
:imports أﺿﻒ app/adapters/persistence/postgres/migrations/env.py ﻓﻲ

python

from app.adapters.persistence.postgres import models # users/documents
from app.adapters.persistence.postgres import models_chunks	# chunks
from app.adapters.persistence.postgres import models_chat # chat

 
Copy code  
 
Create revision 2.2





2.3 ﻣﺤﺘﻮى migration )اﻧﺴﺨﻪ ﻛﻤﺎ ﻫﻮ(
: migrations/versions/*.py داﺧﻞ اﻟﺠﺪﻳﺪ revision ﻣﻠﻒ ﻓﻲ ﺿﻌﻪ
 

python
 

from alembic import op import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "xxxx_add_chunks_chat"
down_revision = "<<<PUT_PREVIOUS_REVISION_ID_HERE>>>" branch_labels = None
depends_on = None



def upgrade() -> None: # --- chunks table op.create_table(
"chunks",
sa.Column("id", sa.String(length=36), primary_key=True),
sa.Column("document_id", sa.String(length=36), sa.ForeignKey("documents.i sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id", onde sa.Column("text", sa.Text(), nullable=False),

# Create column first; we'll convert to GENERATED using raw SQL
sa.Column("tsv", postgresql.TSVECTOR(), nullable=False),

sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.tex
)

op.create_index("ix_chunks_user_id", "chunks", ["user_id"])
op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")

# Make tsv a generated stored column
# Postgres syntax: GENERATED ALWAYS AS (...) STORED
# Need to drop & recreate if Alembic can't alter generated; simplest: ALTER C
op.execute("""
ALTER TABLE chunks ALTER COLUMN tsv
SET DATA TYPE tsvector
USING to_tsvector('simple', coalesce(text, '')); """)
 
# convert to generated (requires dropping default; we use raw SQL) # Approach: drop column and re-add as generated (reliable).
op.execute("""
ALTER TABLE chunks DROP COLUMN tsv; """)
op.execute("""
ALTER TABLE chunks
ADD COLUMN tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', coales """)
op.execute("""
CREATE INDEX ix_chunks_tsv ON chunks USING gin (tsv); """)

# --- chat_sessions
op.create_table(
"chat_sessions",
sa.Column("id", sa.String(length=36), primary_key=True),
sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id", onde sa.Column("title", sa.String(length=200), nullable=True),
sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.tex
)
op.create_index("ix_chat_sessions_user_id", "chat_sessions", ["user_id"])

# --- chat_turns
op.create_table( "chat_turns",
sa.Column("id", sa.String(length=36), primary_key=True),
sa.Column("session_id", sa.String(length=36), sa.ForeignKey("chat_session sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id", onde

sa.Column("question", sa.Text(), nullable=False), sa.Column("answer", sa.Text(), nullable=False),
sa.Column("sources_json", sa.Text(), nullable=False, server_default=sa.te

sa.Column("retrieval_k", sa.Integer(), nullable=False, server_default="5" sa.Column("embed_ms", sa.Integer(), nullable=True),
sa.Column("search_ms", sa.Integer(), nullable=True), sa.Column("llm_ms", sa.Integer(), nullable=True),
sa.Column("prompt_tokens", sa.Integer(), nullable=True),
sa.Column("completion_tokens", sa.Integer(), nullable=True),
 


sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.tex
)
op.create_index("ix_chat_turns_user_id", "chat_turns", ["user_id"])
op.create_index("ix_chat_turns_session_id", "chat_turns", ["session_id"]) op.create_index("ix_chat_turns_created_at", "chat_turns", ["created_at"])


def downgrade() -> None:
op.drop_index("ix_chat_turns_created_at", table_name="chat_turns") op.drop_index("ix_chat_turns_session_id", table_name="chat_turns") op.drop_index("ix_chat_turns_user_id", table_name="chat_turns")
op.drop_table("chat_turns")

op.drop_index("ix_chat_sessions_user_id", table_name="chat_sessions") op.drop_table("chat_sessions")

op.drop_index("ix_chunks_tsv", table_name="chunks")
op.drop_index("ix_chunks_document_id", table_name="chunks") op.drop_index("ix_chunks_user_id", table_name="chunks")
op.drop_table("chunks")



.(users/documents ﻋﻤﻠﺖ اﻟﻠﻲ اﻟﺴﺎﺑﻘﺔ migration ﺑﺘﺎع ID) اﻟﺼﺤﻴﺢ down_revision ﺣﻂ :ﻣﻬﻢ

ﺛﻢ:

bash

alembic upgrade head

Copy code  


Ports + Repos (3
ChunkRepo Port 3.1
app/application/ports/chunk_repo.py

python
 

from typing import Protocol, Sequence from app.domain.entities import Chunk

class ChunkRepoPort(Protocol):
def insert_many(self, *, chunks: Sequence[Chunk]) -> None: ...
Copy code  
PostgresChunkRepo 3.2
app/adapters/persistence/postgres/repo_chunks.py



ChatRepo Port 3.3
 
app/application/ports/chat_repo.py


PostgresChatRepo 3.4
app/adapters/persistence/postgres/repo_chat.py

 
sid = str(uuid.uuid4())
with SessionLocal() as db:
db.execute(insert(ChatSession).values(id=sid, user_id=tenant_id.value db.commit()
return sid

def add_turn( self,
*,
tenant_id: TenantId, session_id: str,
question: str, answer: str,
sources,
retrieval_k: int,
embed_ms: int | None = None, search_ms: int | None = None, llm_ms: int | None = None,
prompt_tokens: int | None = None,
completion_tokens: int | None = None,
) -> str:
tid = str(uuid.uuid4())
with SessionLocal() as db: db.execute(
insert(ChatTurn).values( id=tid,
session_id=session_id,
user_id=tenant_id.value, question=question,
answer=answer,
sources_json=json.dumps(list(sources)), retrieval_k=retrieval_k,
embed_ms=embed_ms,
search_ms=search_ms, llm_ms=llm_ms,
prompt_tokens=prompt_tokens,
completion_tokens=completion_tokens,
)
)
 
db.commit() return tid




generated tsv ﻋﻠﻰ ﻳﻌﺘﻤﺪ KeywordStore (4
app/adapters/persistence/postgres/keyword_store.py

python

from sqlalchemy import select, text as sql_text
from app.adapters.persistence.postgres.db import SessionLocal
from app.adapters.persistence.postgres.models_chunks import ChunkRow from app.application.ports.keyword_store import KeywordStorePort
from app.domain.entities import Chunk, TenantId, DocumentId

class PostgresKeywordStore(KeywordStorePort):
def search(self, *, query: str, tenant_id: TenantId, top_k: int): tsq = sql_text("websearch_to_tsquery('simple', :q)")
rank = sql_text("ts_rank_cd(chunks.tsv, websearch_to_tsquery('simple', :q
Copy code
with SessionLocal() as db: stmt = (
select(ChunkRow.id, ChunkRow.document_id, ChunkRow.user_id, Chunk
.where(ChunkRow.user_id == tenant_id.value)
.where(ChunkRow.tsv.op("@@")(tsq))
.order_by(rank.desc())
.limit(top_k)
)
rows = db.execute(stmt, {"q": query}).all()

return [
Chunk(
id=cid,
tenant_id=TenantId(uid),
document_id=DocumentId(doc_id), text=txt,
)
 
for (cid, doc_id, uid, txt) in rows
]





Postgres + upsert Qdrant ﻓﻲ chunks ﺧّﺰن indexing أﺛﻨﺎء :Worker (5
: chunks list ﺗﺒﻨﻲ ﻣﺎ ﺑﻌﺪ app/workers/tasks.py ﻓﻲ

وﺗﻘﺪر status failed ṇﺗﺴﻴ ﻓﺸﻞ Qdrant وﻟﻮ ،Qdrant ﺛﻢ أوًلا Postgres ựاﻛﺘ :ﻋﺎدة أﻓﻀﻞ ṇﺗﺮﺗﻴ
 
Copy code  
 
.retry ﺗﻌﻴﺪ
 


 

(repos أﺿﻒ) Bootstrap wiring (6
: app/core/bootstrap.py ﻓﻲ

 
"keyword_store": keyword_store,
}





 

Copy code  





























Copy code  
 
(7 )اﺧﺘﻴﺎري ﻟﻜﻦ ﻗﻮي( ﺗﺴﺠﻴﻞ Turn Chat ﺑﻌﺪ الإﺟﺎﺑﺔ
:answer ﺗﻌﻤﻞ ﻣﺎ ﺑﻌﺪ AskQuestionHybridUseCase ﻓﻲ
.turn ﺧّﺰن :request ﻓﻲ session_id ﻋﻨﺪك ﻟﻮ	 
:ﺛﻢ ،DTO ﻟﻠـ session_id: str | None أﺿﻒ



















Quick sanity checks (8
: alembic upgrade head ﺑﻌﺪ	 
tsv GENERATED STORED ﻳﻈﻬﺮ لازم psql ﻓﻲ d chunks\	 
:keyword اﺧﺘﺒﺮ	 
   ارﻓﻊ ﻣﻠﻒ
chunks ṇﻛﺘ worker ﺗﺄﻛﺪ	 
 
   ﺟﺮّị ask-hybrid/ ﺑﺴﺆال ﻳﺤﺘﻮي ﻛﻠﻤﺎت ﺣﺮﻓﻴًﺎ ﻣﻦ اﻟﻨﺺ → recall keyword ﻳﺘﺤﺴﻦ ﺟﺪًا


ﻟﻮ ﺗﺤṇ، اﻟﺨﻄﻮة اﻟﺘﺎﻟﻴﺔ اﻟﻤﻨﻄﻘﻴﺔ:
(indexing ﺗﻜﺮار ﻣﻨﻊ) ﻟﻠﻤﻠﻒ Idempotency hashing .1
(chunk ﻟﻜﻞ hash) Chunk de-dup .2
 
(اﻟﻄﻮﻳﻞ اﻟﻤﺪى ﻋﻠﻰ وأرﺧﺺ LLM rerank ﻣﻦ أﻓﻀﻞ) Reranker Cross-Encoder local .3
ﻗﻮﻟّﻲ أي اﺗﺠﺎه ﻧﻜﻤﻞ.




Idempotency hashing ﻟﻠﻤﻠﻒ (ﺗﻜﺮار ﻣﻨﻊ indexing) Chunk de-dup (hash ﻟﻜﻞ chunk)
Reranker Cross-Encoder local (ﻣﻦ أﻓﻀﻞ LLM rerank وأرﺧﺺ
)ﻋﻠﻰ اﻟﻤﺪى اﻟﻄﻮﻳﻞ


Thought for 21s  
ﺗﻤﺎم — ﻫﻨﻀﻴﻒ اﻟﺜلاث ﺗﺮﻗﻴﺎت دي “ﺻﺢ” وﺑﺸﻜﻞ Production-grade، ﻣﻦ ﻏﻴﺮ ﻣﺎ ﻧﻜسر Clean
:Architecture
.1 hashing Idempotency ﻟﻠﻤﻠﻒ )ﻣﻨﻊ ﺗﻜﺮار indexing ﻟﻨﻔﺲ اﻟﻤﺤﺘﻮى داﺧﻞ ﻧﻔﺲ اﻟـ (tenant
(mapping doc→chunks + ﻣﻮّﺣﺪ ﺗﺨﺰﻳﻦ + chunk ﻟﻜﻞ hash) Chunk de-dup .2
(LLM rerank ﻣﻦ وأﺛﺒﺖ أرﺧﺺ) ﻣﺤﻠﻲ Reranker Cross-Encoder .3
.DB schema + migrations + adapters + worker changes + wiring + knobs :ﻫﺪﻳﻚ


ﻟﻠﻤﻠﻒ Idempotency hashing (1
اﻟﻔﻜﺮة
.upload اﻟـ ﻋﻨﺪ اﻟﻤﻠﻒ bytes ﻋﻠﻰ file_sha256 ṇﻧﺤﺴ	 
. documents.file_sha256 ﻓﻲ ﻧﺨّﺰﻧﻪ	 
: (user_id, file_sha256) ﻋﻠﻰ unique index ﻧﻌﻤﻞ	 
   ﻟﻮ ﻧﻔﺲ اﻟﻤﻠﻒ اﺗﺮﻓﻊ ﺗﺎﻧﻲ ﻟﻨﻔﺲ اﻟﻤﺴﺘﺨﺪم → ﻧﺮﺟّﻊ ﻧﻔﺲ document_id ﺑﺪل إﻧﺸﺎء
.re-index ﻣﻨﻊ + ﺟﺪﻳﺪ Document

Migration: add file_sha256 + unique index 1.1
:revision اﻋﻤﻞ

bash
 

alembic revision -m "add document file hash idempotency"

 
Copy code  
 
:migration ﻓﻲ ﺿﻊ
 

python

from alembic import op import sqlalchemy as sa

revision = "xxxx_doc_hash" down_revision = "<<<PREV>>>" branch_labels = None
depends_on = None

def upgrade():
op.add_column("documents", sa.Column("file_sha256", sa.String(length=64), nul op.create_index(
"uq_documents_user_file_sha256", "documents",
["user_id", "file_sha256"], unique=True,
)

def downgrade():
op.drop_index("uq_documents_user_file_sha256", table_name="documents") op.drop_column("documents", "file_sha256")

C	C

ﺛﻢ:

bash

alembic upgrade head

 
Copy code   Copy code  
 

(ﻳﺘﻐﻴﺮ ﻣﺤﺘﺎج ﻣﺶ StoredFile + UploadResult) Update Domain 1.2
.”create or get existing“ ṇوﻧﻄﻠ hash ṇﻧﺤﺴ لازم UseCase upload ﻓﻲ ﻟﻜﻦ

idempotency ﻳﺪﻋﻢ Port: DocumentRepo 1.3
ﻋﺪّل/أﺿﻒ port ﺻﻐﻴﺮ :(ISP)
 
python

# app/application/ports/document_idempotency.py
from typing import Protocol
from app.domain.entities import TenantId, DocumentId

class DocumentIdempotencyPort(Protocol):
def get_by_file_hash(self, *, tenant_id: TenantId, file_sha256: str) -> Docum def create_document_with_hash(self, *, tenant_id: TenantId, stored_file, file

C	C

Postgres repo implementation 1.4
Copy code	python

# app/adapters/persistence/postgres/repo_documents_idempotency.py
import uuid
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from app.adapters.persistence.postgres.db import SessionLocal from app.adapters.persistence.postgres.models import Document
from app.domain.entities import TenantId, DocumentId, StoredFile
from app.application.ports.document_idempotency import DocumentIdempotencyPort

class PostgresDocumentIdempotencyRepo(DocumentIdempotencyPort):
def get_by_file_hash(self, *, tenant_id: TenantId, file_sha256: str) -> Docum with SessionLocal() as db:
stmt = select(Document.id).where(Document.user_id == tenant_id.value, v = db.execute(stmt).scalar_one_or_none()
return DocumentId(v) if v else None

def create_document_with_hash(self, *, tenant_id: TenantId, stored_file: Stor doc_id = DocumentId(str(uuid.uuid4()))
with SessionLocal() as db: db.add(
Document(
id=doc_id.value,
user_id=tenant_id.value,
filename=stored_file.filename,
content_type=stored_file.content_type,
 
file_path=stored_file.path,
size_bytes=stored_file.size_bytes, status="created",
file_sha256=file_sha256,
)
)
try:
db.commit()
return doc_id
except IntegrityError: db.rollback()
# someone inserted same hash concurrently → fetch existing
stmt = select(Document.id).where(Document.user_id == tenant_id.va existing = db.execute(stmt).scalar_one()
return DocumentId(existing)


Update UploadDocumentUseCase (idempotent) 1.5

python

# app/application/use_cases/upload_document.py (replace create logic)
import hashlib
from app.application.ports.document_idempotency import DocumentIdempotencyPort

class UploadDocumentUseCase:
def   init  (self, file_store, repo: DocumentRepoPort, queue, idem_repo: Docu
self._file_store = file_store self._repo = repo
self._queue = queue
self._idem = idem_repo

Copy code
async def execute(self, req: UploadDocumentRequest) -> UploadResult:
tenant = TenantId(req.tenant_id)

file_sha256 = hashlib.sha256(req.data).hexdigest()

existing = self._idem.get_by_file_hash(tenant_id=tenant, file_sha256=file if existing:
# already known → do NOT enqueue again
return UploadResult(document_id=existing, status="already_exists")
 


stored = await self._file_store.save_upload( tenant_id=tenant.value,
upload_filename=req.filename, content_type=req.content_type, data=req.data,
)

doc_id = self._idem.create_document_with_hash(tenant_id=tenant, stored_fi self._repo.set_status(tenant_id=tenant, document_id=doc_id, status="queue self._queue.enqueue_index_document(tenant_id=tenant, document_id=doc_id)

return UploadResult(document_id=doc_id, status="queued")





اﻟﺼﺤﻴﺢ اﻟﺘﺼﻤﻴﻢ — (chunk ﻟﻜﻞ hash) Chunk de-dup (2
ﻟﻴﻪ اﻟﺘﺼﻤﻴﻢ اﻟﺼﺢ لاز  ﻳﻜﻮن ﺟﺪوﻟﻴﻦ؟
ﻟﻮ ﺧﺰﻧّﺎ chunks داﺧﻞ documents ﻣﺒﺎشرة، اﻟﺪي-دị ﻫﻴﺒﻘﻰ “داﺧﻞ doc ﻓﻘﻂ.”
أﻧﺖ ﻃﻠﺒﺖ de-dup ﺣﻘﻴﻘﻲ )ﻳﻮﻓﺮ ﻣﺴﺎﺣﺔ + ﻳﺜﺒﺖ search Keyword + ﻳﺤﺴﻦ (re-use، ﻓﺎلأﻓﻀﻞ:
chunk_store : chunk unique per tenant via (user_id, chunk_hash)	 
position/ṇﺗﺮﺗﻴ + document_chunks : mapping doc → chunk	 

Migration: chunk_store + document_chunks + generated tsv 2.1
)ﺑﺪل ﺟﺪول chunks اﻟﻘﺪﻳﻢ ﻟﻮ ﻛﻨﺖ ﻋﺎﻣﻠّﻪ(                            code Copy ﻟﻮ ﻋﻨﺪك ﺟﺪول chunks ﺑﺎﻟﻔﻌﻞ: إﻣّﺎ ﻧﻌﻤﻞ migration اﻧﺘﻘﺎﻟﻴﺔ. ﻟﻮ ﻟﺴﻪ ﻓﻲ ﻣﺮﺣﻠﺔ template، الأﺳﻬﻞ: ﻧﻌﻤﻞ tables اﻟﺠﺪﻳﺪة وﻧﺴﻴṇ اﻟﻘﺪﻳﻤﺔ أو ﻧﺰﻳﻠﻬﺎ ﻓﻲ .downgrade/cleanup
:Revision


python
 

from alembic import op import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "xxxx_chunk_dedup" down_revision = "<<<PREV>>>" branch_labels = None
depends_on = None

def upgrade():
# chunk_store
op.create_table(
"chunk_store",
sa.Column("id", sa.String(36), primary_key=True),
sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="C sa.Column("chunk_hash", sa.String(64), nullable=False),
sa.Column("text", sa.Text(), nullable=False),
sa.Column("tsv", postgresql.TSVECTOR(), nullable=False),
sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.tex
)
op.create_index("uq_chunk_store_user_hash", "chunk_store", ["user_id", "chunk op.create_index("ix_chunk_store_user_id", "chunk_store", ["user_id"])

# generated tsv
op.execute("ALTER TABLE chunk_store DROP COLUMN tsv;") op.execute("""
ALTER TABLE chunk_store
ADD COLUMN tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', coales """)
op.execute("CREATE INDEX ix_chunk_store_tsv ON chunk_store USING gin (tsv);")

# document_chunks mapping
op.create_table(
"document_chunks",
sa.Column("document_id", sa.String(36), sa.ForeignKey("documents.id", ond sa.Column("chunk_id", sa.String(36), sa.ForeignKey("chunk_store.id", onde sa.Column("ord", sa.Integer(), nullable=False),
sa.PrimaryKeyConstraint("document_id", "ord"),
)
op.create_index("ix_document_chunks_document_id", "document_chunks", ["docume
 
op.create_index("ix_document_chunks_chunk_id", "document_chunks", ["chunk_id"

def downgrade():
op.drop_index("ix_document_chunks_chunk_id", table_name="document_chunks")
op.drop_index("ix_document_chunks_document_id", table_name="document_chunks") op.drop_table("document_chunks")

op.execute("DROP INDEX IF EXISTS ix_chunk_store_tsv;")
op.drop_index("ix_chunk_store_user_id", table_name="chunk_store")
op.drop_index("uq_chunk_store_user_hash", table_name="chunk_store") op.drop_table("chunk_store")


ORM Models 2.2
app/adapters/persistence/postgres/models_chunk_store.py

python

from sqlalchemy import String, Text, DateTime, ForeignKey, Integer, func, Index, from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import TSVECTOR
from app.adapters.persistence.postgres.db import Base

class ChunkStoreRow(Base):
  tablename	 = "chunk_store"
  table_args	 = (
UniqueConstraint("user_id", "chunk_hash", name="uq_chunk_store_user_hash"
)

id: Mapped[str] = mapped_column(String(36), primary_key=True)
user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondel
Copy ccohduenk_hash: Mapped[str] = mapped_column(String(64), nullable=False) text: Mapped[str] = mapped_column(Text, nullable=False)
tsv: Mapped[object] = mapped_column(TSVECTOR, nullable=False) # generated in
created_at: Mapped["DateTime"] = mapped_column(DateTime(timezone=True), serve

Index("ix_chunk_store_user_id", ChunkStoreRow.user_id)
Index("ix_chunk_store_tsv", ChunkStoreRow.tsv, postgresql_using="gin")
 
class DocumentChunkRow(Base):
  tablename	 = "document_chunks"
document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id ord: Mapped[int] = mapped_column(Integer, primary_key=True)
chunk_id: Mapped[str] = mapped_column(String(36), ForeignKey("chunk_store.id"

Index("ix_document_chunks_document_id", DocumentChunkRow.document_id) Index("ix_document_chunks_chunk_id", DocumentChunkRow.chunk_id)


Chunk hashing + de-dup repo 2.3
:Port

:Adapter

 
stmt = select(ChunkStoreRow.id).where(ChunkStoreRow.user_id == tenant existing = db.execute(stmt).scalar_one_or_none()
if existing:
return existing

cid = str(uuid.uuid4())
db.add(ChunkStoreRow(id=cid, user_id=tenant_id.value, chunk_hash=chun try:
db.commit() return cid
except IntegrityError: db.rollback()
# concurrent insert
existing = db.execute(stmt).scalar_one() return existing

def replace_document_chunks(self, *, tenant_id: TenantId, document_id: str, c # tenant_id not stored in mapping table; tenant isolation enforced via do with SessionLocal() as db:
db.execute(delete(DocumentChunkRow).where(DocumentChunkRow.document_i rows = [{"document_id": document_id, "ord": i, "chunk_id": cid} for i if rows:
db.execute(insert(DocumentChunkRow), rows) db.commit()


Worker changes: hashing + de-dup + mapping + Qdrant 2.4
:chunking ﺑﻌﺪ index_document داﺧﻞ

 
chunk_ids_in_order = [] chunks_for_qdrant = [] vectors = []

for t in chunks_text: h = chunk_sha256(t)
chunk_id = dedup_repo.upsert_chunk_store(tenant_id=tenant, chunk_hash=h, text chunk_ids_in_order.append(chunk_id)

# Qdrant uses chunk_id as point id (stable + dedup)
chunks_for_qdrant.append(Chunk(id=chunk_id, tenant_id=tenant, document_id=doc vectors.append(cached_emb.embed_one(t))

dedup_repo.replace_document_chunks( tenant_id=tenant,
document_id=doc_id.value,
chunk_ids_in_order=chunk_ids_in_order,
)

vector_store.ensure_collection()
vector_store.upsert(chunks_for_qdrant, vectors)


ﻣﻴﺰة: ﻟﻮ ﻧﻔﺲ chunk اﺗﻜﺮر ﻓﻲ ﻣﻠﻒ أو ﻣﻠﻔﺎت أﺧﺮى ﻟﻨﻔﺲ اﻟﻤﺴﺘﺨﺪم، ﻧﻔﺲ chunk_id ﻫﻴﺘﻌﺎد اﺳﺘﺨﺪاﻣﻪ.


chunk_store (generated tsv) ﻋﻠﻰ Hybrid Keyword Search (3
:(اﻟﻘﺪﻳﻤﺔ chunks ﻣﺶ) chunk_store ﻣﻦ ﻟﻴﻜﻮن KeywordStore query ﺑّﺪل

 
class PostgresKeywordStore(KeywordStorePort):
def search(self, *, query: str, tenant_id: TenantId, top_k: int): tsq = sql_text("websearch_to_tsquery('simple', :q)")
rank = sql_text("ts_rank_cd(chunk_store.tsv, websearch_to_tsquery('simple

with SessionLocal() as db: stmt = (
select(ChunkStoreRow.id, ChunkStoreRow.user_id, ChunkStoreRow.tex
.where(ChunkStoreRow.user_id == tenant_id.value)
.where(ChunkStoreRow.tsv.op("@@")(tsq))
.order_by(rank.desc())
.limit(top_k)
)
rows = db.execute(stmt, {"q": query}).all()

ﻣﺶ ﻣﺮﺑﻮط ﺑﺪوك واﺣﺪ chunk_store ھﻨﺎ اﺧﺘﯿﺎري؛ document_id #
# ﻣﺤﺘﺎج ﻟﻮ doc filter: join document_chunks (ﻻﺣًﻘﺎ ﺑﻨﻀﯿﻔﻪ).
return [Chunk(id=cid, tenant_id=TenantId(uid), document_id=DocumentId("")





Reranker Cross-Encoder Local (Production-grade) (4
Dependencies 4.1


 
Copy code  
 
Settings knobs 4.2
: app/core/config.py ﻓﻲ
 

python

rerank_backend: str = "cross_encoder" # cross_encoder|none
cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2" cross_encoder_device: str = "cpu" # cpu|cuda


Copy code  
 

 
ﻣﻤﻜﻦ ﺗﺴﺘﺨﺪم models أﻗﻮى لاﺣﻘًﺎ )زي (bge-reranker ﻟﻮ ﻋﻨﺪك .GPU

Adapter implements RerankerPort 4.3

python

# app/adapters/rerank/cross_encoder_reranker.py
from typing import Sequence
from sentence_transformers import CrossEncoder
from app.application.ports.reranker import RerankerPort from app.domain.entities import Chunk

class CrossEncoderReranker(RerankerPort):
def   init  (self, model_name: str, device: str = "cpu") -> None:
self._model = CrossEncoder(model_name, device=device)

def rerank(self, *, query: str, chunks: Sequence[Chunk], top_n: int) -> Seque if not chunks:
return []
pairs = [(query, c.text) for c in chunks]
scores = self._model.predict(pairs)	# numpy array
ranked = sorted(zip(chunks, scores), key=lambda x: float(x[1]), reverse=T return [c for c, _ in ranked[:top_n]]

C	C

bootstrap ﻓﻲ Wiring 4.4



 
Copy code  
 
(5 logging Chat + ﺟﻮدة )اﺧﺘﻴﺎري ﻟﻜﻦ ﻗﻮي(
ﺑﻌﺪ ﻣﺎ ﺗﻨﺘﺞ الإﺟﺎﺑﺔ ﻓﻲ :AskHybridUseCase
 
chat_turns (question/answer/sources + timings) ﻓﻲ turn ﺧّﺰن	 
.Eval-ready history و Observability ﺑﻴﺪي ده	 
)أﻧﺖ ﻋﻨﺪك ChatRepo ﺳﺎﺑﻘًﺎ—ﻣﻤﻜﻦ ﻧﻮﺻﻠﻪ ﻣﺒﺎشرة(.


(container ﻟﻠـ ﺗﻀﻴﻔﻪ اﻟﻠﻲ إﻳﻪ) Bootstrapping summary (6
:أﺿﻒ ()get_container ﻓﻲ
()document_idem_repo = PostgresDocumentIdempotencyRepo
()chunk_dedup_repo = PostgresChunkDedupRepo
(...)reranker = CrossEncoderReranker	 
(chunk_store ﻋﻠﻰ) ()keyword_store = PostgresKeywordStore	 
Update UploadDocumentUseCase(..., idem_repo=document_idem_repo)	 



(7 ﻣلاﺣﻈﺎت Production ﻣﻬﻤﺔ ﺟﺪًا
   إﻋﺎدة رﻓﻊ ﻧﻔﺲ اﻟﻤﻠﻒ: ”already_exists“ ﻳﺮﺟﻊ document_id اﻟﺤﺎﻟﻲ → ﻣﻔﻴﺶ queue ولا
.indexing race ﺑﺘﻘﻔﻞ اﻟﻠﻲ ﻫﻲ unique indexes + IntegrityError handling اﻟـ :Concurrency	 
.conditions
:Chunk de-dup	 
   ﻳﻘﻠﻞ DB اﻟﺤﺠﻢ
(chunk_id point id اﺳﺘﺨﺪﻣﺖ ﻟﻮ) Qdrant points ﻳﻘﻠﻞ	 
ﺛﺎﺑﺖ ﺑﺸﻜﻞ keyword recall وﻳﺤﺴﻦ	 
:Cross-Encoder	 
.latency ṇراﻗ ﻟﻜﻦ ،rerank 40 passages → top 8 ﻟـ ﻣﻤﺘﺎز :CPU ﻋﻠﻰ	 
   ﻋﻠﻰ :GPU أسرع ﺑﻜﺜﻴﺮ وﻳﻔﺘﺢ ﻟﻚ top_k أﻛﺒﺮ.


إذا ﺗﺤṇ، اﻟﺨﻄﻮة اﻟﻠﻲ ﺑﻌﺪﻫﺎ ﻣﺒﺎشرة ﻋﺸﺎن ﻳﺒﻘﻰ :”Enterprise-ready“
Doc-filtered keyword search (join document_chunks + chunk_store ) .1 (Qdrant storage ﺗﻮﻓﻴﺮ) Vector payload minimal + fetch text from DB .2 وﻛﻠﻔﺔ وﻗﺖ ﻟﺘﻘﻠﻴﻞ (embed_one ﺑﺪل) indexing ﻓﻲ Batch embeddings .3
ﻗﻮﻟّﻲ ﺗﺤṇ ﻧﻜﻤﻞ ﺑﺄي ﺗﺮﺗﻴṇ؟
 

 	 	 	 	 	 



Doc-filtered keyword search (join document_chunks + chunk_store)

Vector payload minimal + fetch text from DB (ﺗﻮﻓﻴﺮ Qdrant storage)

Batch embeddings ﻓﻲ indexing (ﺑﺪل embed_one) ﻟﺘﻘﻠﻴﻞ
وﻗﺖ وﻛﻠﻔﺔ


:Clean Architecture روح ﺑﻨﻔﺲ ”Enterprise-grade“ ﺗﺮﻗﻴﺎت 3 ﻫﻨﻀﻴﻒ — ﺗﻤﺎم
(ﻣﺤﺪد document ﻋﻠﻰ ﻣﻘﻴﺪ ﻟﻜﻦ FTS) Doc-filtered keyword search .1
(Postgres ﻣﻦ ṇُﻳﺠﻠ اﻟﻨﺺ + payload أﻗﻞ ﻳﺨﺰن Qdrant) Vector payload minimal .2
(أﻗﻞ ﺗﻜﻠﻔﺔ + سرﻋﺔ) indexing ﻓﻲ Batch embeddings .3
.Retrieval/Ask ﺗﻌﺪﻳلات + Worker ﺗﻌﺪﻳلات + Ports + Adapters + SQL :ﻫﺪﻳﻚ


Doc-filtered keyword search ( join document_chunks + (1
chunk_store )
1.1 Port )إﺿﺎﻓﺔ ﻓﻠﺘﺮ document_id اﺧﺘﻴﺎري(
ﺑﺪل ﻣﺎ ﻧﻌﻤﻞ Port ﺟﺪﻳﺪ، ﻧﺨﻠﻲ method ﺗﻘﺒﻞ None | str document_id: ISP) ﻣﻘﺒﻮل لأﻧﻪ ﻧﻔﺲ اﻟﻤﺴﺆوﻟﻴﺔ.(
python

# app/application/ports/keyword_store.py
from typing import Protocol, Sequence
from app.domain.entities import Chunk, TenantId

class KeywordStorePort(Protocol): def search(
self,
*,
 
query: str,
tenant_id: TenantId, top_k: int,
document_id: str | None = None,
) -> Sequence[Chunk]: ...

 







Copy code  
 
PostgresKeywordStore (doc-filtered) 1.2
اﻟﻤﺒﺪأ:
:ﻣﻮﺟﻮد document_id ﻟﻮ	  join: document_chunks dc → chunk_store cs ﻧﻌﻤﻞ	 
dc.document_id = :doc_id + cs.user_id = :user_id ﻧﻔﻠﺘﺮ	 
ts_rank_cd(cs.tsv, tsquery) ﺑـ ṇﻧﺮﺗ	  (ﻣﻌﺮوف document_id) Chunk id/text وﻧﺮّﺟﻊ	 
 
python

# app/adapters/persistence/postgres/keyword_store.py
from sqlalchemy import text as sql
from app.adapters.persistence.postgres.db import SessionLocal
from app.application.ports.keyword_store import KeywordStorePort from app.domain.entities import Chunk, TenantId, DocumentId

class PostgresKeywordStore(KeywordStorePort):
def search(self, *, query: str, tenant_id: TenantId, top_k: int, document_id: with SessionLocal() as db:
if document_id:
rows = db.execute(sql(""" SELECT cs.id, cs.text
FROM document_chunks dc
JOIN chunk_store cs ON cs.id = dc.chunk_id WHERE dc.document_id = :doc_id
AND cs.user_id = :user_id
AND cs.tsv @@ websearch_to_tsquery('simple', :q)
ORDER BY ts_rank_cd(cs.tsv, websearch_to_tsquery('simple', :q dc.ord ASC
LIMIT :k
"""), {"doc_id": document_id, "user_id": tenant_id.value, "q": qu

return [
 
Chunk(
id=cid,
tenant_id=tenant_id,
document_id=DocumentId(document_id), text=txt,
)
for (cid, txt) in rows
]

# tenant-wide keyword search
rows = db.execute(sql(""" SELECT cs.id, cs.text FROM chunk_store cs
WHERE cs.user_id = :user_id
AND cs.tsv @@ websearch_to_tsquery('simple', :q)
ORDER BY ts_rank_cd(cs.tsv, websearch_to_tsquery('simple', :q)) D LIMIT :k
"""), {"user_id": tenant_id.value, "q": query, "k": top_k}).all()

return [
Chunk(
id=cid,
tenant_id=tenant_id,
document_id=DocumentId(""), text=txt,
)
for (cid, txt) in rows
]


✅ ﻛﺪه ﺗﻘﺪر ﺗﻌﻤﻞ “ﺑﺤﺚ داﺧﻞ document واﺣﺪ” useful) ﺟﺪًا ﻟﻮاﺟﻬﺔ .(ChatPDF


Vector payload minimal + fetch text from DB (2
اﻟﻤﺸﻜﻠﺔ
:وده ، text ﻓﻴﻪ Qdrant payload ﺣﺎﻟًﻴﺎ
storage ﻳﺰّود	 
   ﻳﻜﺮر اﻟﺒﻴﺎﻧﺎت ﻣﻊ Postgres
 













Copy code  
 
  ﻳﺼﻌّṇ ﺗﺤﺪﻳﺚ اﻟﻨﺺ/اﻟﺘﻨﻈﻴﻒ
اﻟﺤﻞ اﻟﺼﺤﻴﺢ
:ﻳﺨﺰن Qdrant	 
tenant_id	 
(أﺻًلا id) chunk_id	 
(doc-filter vector-side ﻣﺤﺘﺎج ﻟﻮ اﺧﺘﻴﺎري) document_id ﻣﻤﻜﻦ	 
. ChunkTextReaderPort ﻋﺒﺮ Postgres ﻣﻦ ựﻳﺠﻠ اﻟﻨﺺ	 
Port: ChunkTextReader 2.1
 

python

# app/application/ports/chunk_text_reader.py
from typing import Protocol, Sequence
from app.domain.entities import TenantId

class ChunkTextReaderPort(Protocol):
def get_texts_by_ids(self, *, tenant_id: TenantId, chunk_ids: Sequence[str])


C	C

Adapter: PostgresChunkTextReader 2.2
 
AND id = ANY(:ids)
"""), {"user_id": tenant_id.value, "ids": list(chunk_ids)}).all() return {cid: txt for cid, txt in rows}

ﻣلاﺣﻈﺔ: ANY(:ids) ﻳﻌﻤﻞ ﻓﻲ psycopg ﻛﻮﻳﺲ، ﻟﻮ واﺟﻬﺘﻚ ﻣﺸﻜﻠﺔ ﻧﺒﺪّﻟﻬﺎ إﻟﻰ = id WHERE
. ANY(CAST(:ids AS text[]))

Qdrant upsert: payload minimal ﺗﻌﺪﻳﻞ 2.3


 


Copy code  
 
ﻓﻘﻂ IDs ﻳﺮّﺟﻊ Qdrant search ﺗﻌﺪﻳﻞ 2.4
ﻫﻨﺮﺟﻊ objects Chunk ﺑﺎﻟﻨﺺ ﻓﺎرغ، وﺑﻌﺪﻳﻦ ﻧﻤلأه ﻣﻦ .DB
 

python

# app/adapters/vector/qdrant_store.py (search_scored)
text="" # placeholder

 
Copy code  
 
Service: hydrate chunks texts (pure-ish) 2.5
 

python

# app/application/services/hydrate.py
from typing import Sequence
from app.domain.entities import Chunk, TenantId, DocumentId
from app.application.ports.chunk_text_reader import ChunkTextReaderPort

def hydrate_chunk_texts(
*,
tenant_id: TenantId,
chunks: Sequence[Chunk],
reader: ChunkTextReaderPort,
 
) -> list[Chunk]:
ids = [c.id for c in chunks]
texts = reader.get_texts_by_ids(tenant_id=tenant_id, chunk_ids=ids)

out = []
for c in chunks: out.append(
Chunk(
id=c.id,
tenant_id=c.tenant_id,
document_id=c.document_id, text=texts.get(c.id, ""),
)
)
return out

Update AskHybridUseCase: hydrate after retrieval + before 2.6
rerank/prompt vector hits -> get ids	  hydrate texts from DB		 
then fusion + rerank + prompt	 
ﻣﻬﻢ: reranker ﻳﺤﺘﺎج ﻧﺺ، ﻓﺎﻟﻬﻴﺪراﺷﻦ لازم ﻗﺒﻞ .rerank

(embed_one ﺑﺪل) indexing ﻓﻲ Batch embeddings (3
ﻟﻤﺎذا؟
.batch input ﻳﺪﻋﻢ OpenAI embeddings endpoint	 
overhead/network calls ﺗﻘﻠﻴﻞ	 
   أسرع وأرﺧﺺ ﻋﻤﻠﻴًﺎ.
embed_many_cached أﺿﻒ :CachedEmbeddings 3.1

python

# app/application/services/embedding_cache.py
import hashlib
from app.application.ports.cache import CachePort
 
from app.application.ports.embeddings import EmbeddingsPort

class CachedEmbeddings:
def   init  (self, embeddings: EmbeddingsPort, cache: CachePort, ttl_seconds: self._emb = embeddings
self._cache = cache
self._ttl = ttl_seconds

def _key(self, text: str) -> str:
return "emb:" + hashlib.md5(text.encode("utf-8")).hexdigest()

def embed_one(self, text: str) -> list[float]: key = self._key(text)
cached = self._cache.get_json(key) if cached and "v" in cached:
return cached["v"]
v = self._emb.embed_one(text)
self._cache.set_json(key, {"v": v}, ttl_seconds=self._ttl) return v

def embed_many(self, texts: list[str]) -> list[list[float]]:
# 1) check cache
keys = [self._key(t) for t in texts]
cached = [self._cache.get_json(k) for k in keys]

missing_idx = [i for i, c in enumerate(cached) if not (c and "v" in c)] if not missing_idx:
return [c["v"] for c in cached] # type: ignore

# 2) batch embed missing
missing_texts = [texts[i] for i in missing_idx]
missing_vecs = self._emb.embed_many(missing_texts)

# 3) store back
for i, v in zip(missing_idx, missing_vecs):
self._cache.set_json(keys[i], {"v": v}, ttl_seconds=self._ttl)

# 4) assemble output
out = []
miss_map = {idx: vec for idx, vec in zip(missing_idx, missing_vecs)}
 
for i, c in enumerate(cached): if c and "v" in c:
out.append(c["v"]) else:
out.append(miss_map[i]) return out


Worker indexing: batch embed once 3.2
:loop embed_one ﺑﺪل

python

# worker indexing snippet
texts_for_embedding = chunks_text # list[str]
vectors = cached_emb.embed_many(texts_for_embedding)
# now zip chunk_ids / vectors

 

Copy code  









Copy code  
 
de-dup ﻣﻊ
ﻟﻮ ﺑﺘﻌﻤﻞ de-dup :chunk ﻫﺘﺤﺘﺎج embed ﻟﻠﻨﺼﻮص ﺑﺎﻟﺘﺮﺗﻴṇ ﺣﺘﻰ ﻟﻮ chunk_id ﻣﻜﺮر.
:embedding call ﻣﺴﺘﻮى ﻋﻠﻰ dedup :الأﻓﻀﻞ	  mapping hash -> text unique اﺑﻨﻲ	 
ﻓﻘﻂ unique ﻟﻠـ embed_many اﻋﻤﻞ	  order ṇﺣﺴ expand vectors ﺛﻢ	 
 
:Snippet

python

import hashlib

def chunk_hash(text: str) -> str: norm = " ".join(text.split())
return hashlib.sha256(norm.encode("utf-8")).hexdigest()

hashes = [chunk_hash(t) for t in chunks_text] unique = {}
for h, t in zip(hashes, chunks_text): unique.setdefault(h, t)
 


unique_hashes = list(unique.keys())
unique_texts = [unique[h] for h in unique_hashes]
unique_vectors = cached_emb.embed_many(unique_texts)
vec_by_hash = {h: v for h, v in zip(unique_hashes, unique_vectors)} vectors_in_order = [vec_by_hash[h] for h in hashes]


payload + doc-filter vector ﻟﺘﻘﻠﻴﻞ VectorStore ﺗﻌﺪﻳﻞ (4
(اﺧﺘﻴﺎري) search
 

Copy code  
 
:ﻛﻤﺎن vector ﻟﻠـ doc-filter ﺗﻌﻤﻞ ṇﺣﺎﺑ ﻟﻮ payload ﻓﻲ document_id ﺧﺰن	 
.ﻣﻮﺟﻮد ﻟﻮ document_id == req.document_id ﻳﻀﻴﻒ search filter	 
.ChatPDF mode ﻓﻲ ﻣﻔﻴﺪ ده


(5 ﺗﺤﺪﻳﺚ AskHybridUseCase )اﻟﺼﻮرة اﻟﻨﻬﺎﺋﻴﺔ(
Strategy
vector_hits_scored (IDs + scores)	 
DB ﻣﻦ hydrate texts	  keyword_hits (already has text)	 
fusion (RRF)	  (ﻓﻘﻂ IDs رﺟﻊ keyword ﻟﻮ) hydrate fused if needed	 
rerank CrossEncoder		  prompt + LLM		  log chat turn	 


(ﺗﺘﻮه لا ﺣﺘﻰ) Minimal changes list (6
أﺿﻒ:
ChunkTextReaderPort + PostgresChunkTextReader	 
 
hydrate_chunk_texts service	  Qdrant payload (remove text) ﺗﻌﺪﻳﻞ	 
worker: batch embeddings + de-dup hash mapping ﺗﻌﺪﻳﻞ	 
KeywordStore.search(document_id=...) ﺗﺤﺪﻳﺚ	  rerank ﻗﺒﻞ hydration ﻟﻌﻤﻞ AskHybridUseCase ﺗﺤﺪﻳﺚ	 


ﻟﻮ ﺗﺤự أﻛﻤﻞ “ﻛﻮد ﻛﺎﻣﻞ ﺟﺎﻫﺰ” ﻛـ patch
ﻗﻠّﻲ ﺑﺲ:
؟document-filtered ﻛﻤﺎن ولا ﻓﻘﻂ، tenant-wide ﻳﻜﻮن Vector search ﺗﺮﻳﺪ ﻫﻞ	 
أﻧﺎ أﻣﻴﻞ: اﺗﻨﻴﻦ document_id) اﺧﺘﻴﺎري ﻓﻲ (request لأﻧﻪ ﻳﺨﺪم ﻛﻞ اﻟﺴﻴﻨﺎرﻳﻮﻫﺎت.




ﻛﻤﻞ


ﺗﻤﺎم — دي “اﻟﻨﺴﺨﺔ اﻟﻤﻜﺘﻤﻠﺔ” ﻟﻠـ 3 ﺗﺮﻗﻴﺎت ﻣﻊ ﻛﻮد ﺟﺎﻫﺰ (Patch-style) ﻋﻠﻰ اﻟـ Starter اﻟﻠﻲ ﺑﻨﻴﻨﺎه:
Doc-filtered keyword search (join document_chunks + chunk_store ) ✅	  (Postgres ﻣﻦ text + hydrate text ﺑﺪون Qdrant) Vector payload minimal ✅	  hash اﻟـ ﻣﺴﺘﻮى ﻋﻠﻰ ذﻛﻲ indexing + dedup ﻓﻲ Batch embeddings ✅	 
chunk_store + document_chunks + documents + users + :اﻟﺠﺪاول ﻋﻨﺪك إن ﻫﻨﻔﺘﺮض
*_chat ﻣﻦ اﻟﺨﻄﻮات اﻟﺴﺎﺑﻘﺔ.



Update Ports (Vector + Keyword + Text Reader) (1
app/application/services/scoring.py 1.1

python

from dataclasses import dataclass
from app.domain.entities import Chunk

@dataclass(frozen=True)
 
class ScoredChunk: chunk: Chunk
score: float

app/application/ports/vector_store.py 1.2

app/application/ports/keyword_store.py 1.3

python

from typing import Protocol, Sequence
from app.domain.entities import Chunk, TenantId
 

 
app/application/ports/chunk_text_reader.py 1.4


Copy code  
Vector payload minimal (Qdrant) + hydrate from (2
Postgres
app/adapters/vector/qdrant_store.py (REPLACE with minimal 2.1
payload)

python

from typing import Sequence
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, Fie from app.application.ports.vector_store import VectorStorePort
from app.application.services.scoring import ScoredChunk
from app.domain.entities import Chunk, TenantId, DocumentId

 
class QdrantVectorStore(VectorStorePort):
def   init  (self, client: QdrantClient, collection: str, vector_size:
 

int) -
 
self._client = client
self._collection = collection self._size = vector_size

def ensure_collection(self) -> None:
if not self._client.collection_exists(self._collection): self._client.create_collection(
collection_name=self._collection,
vectors_config=VectorParams(size=self._size, distance=Distance.CO
)

def upsert_points( self,
*,
ids: Sequence[str],
vectors: Sequence[list[float]], tenant_id: str,
document_id: str,
) -> None:
points = []
for pid, vec in zip(ids, vectors): points.append(
PointStruct(
id=pid,
vector=vec, payload={
"tenant_id": tenant_id,
"document_id": document_id,	# optional but enables doc-f
},
)
)
self._client.upsert(collection_name=self._collection, points=points)

def search_scored( self,
*,
query_vector: list[float], tenant_id: TenantId,
top_k: int,
document_id: str | None = None,
 
) -> Sequence[ScoredChunk]:
must = [FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id. if document_id:
must.append(FieldCondition(key="document_id", match=MatchValue(value= flt = Filter(must=must)
res = self._client.search(
collection_name=self._collection, query_vector=query_vector,
query_filter=flt, limit=top_k,
)

out: list[ScoredChunk] = [] for r in res:
payload = r.payload or {}
# NOTE: text is NOT stored in Qdrant anymore c = Chunk(
id=str(r.id),
tenant_id=TenantId(payload.get("tenant_id", tenant_id.value)),
document_id=DocumentId(payload.get("document_id", document_id or text="", # hydrate later
)
out.append(ScoredChunk(chunk=c, score=float(r.score))) return out


app/adapters/persistence/postgres/chunk_text_reader.py 2.2

python

from sqlalchemy import text as sql
from app.adapters.persistence.postgres.db import SessionLocal
from app.application.ports.chunk_text_reader import ChunkTextReaderPort from app.domain.entities import TenantId

class PostgresChunkTextReader(ChunkTextReaderPort):
def get_texts_by_ids(self, *, tenant_id: TenantId, chunk_ids): if not chunk_ids:
 
return {}
with SessionLocal() as db:
rows = db.execute(sql(""" SELECT id, text
FROM chunk_store
WHERE user_id = :user_id AND id = ANY(:ids)
"""), {"user_id": tenant_id.value, "ids": list(chunk_ids)}).all() return {cid: txt for cid, txt in rows}

app/application/services/hydrate.py 2.3


 
Doc-filtered keyword search ( join doc_chunks + (3
chunk_store)
app/adapters/persistence/postgres/keyword_store.py (REPLACE)
Copy code
python

from sqlalchemy import text as sql
from app.adapters.persistence.postgres.db import SessionLocal
from app.application.ports.keyword_store import KeywordStorePort from app.domain.entities import Chunk, TenantId, DocumentId

class PostgresKeywordStore(KeywordStorePort):
def search(self, *, query: str, tenant_id: TenantId, top_k: int, document_id: with SessionLocal() as db:
if document_id:
rows = db.execute(sql(""" SELECT cs.id, cs.text
FROM document_chunks dc
JOIN chunk_store cs ON cs.id = dc.chunk_id WHERE dc.document_id = :doc_id
AND cs.user_id = :user_id
AND cs.tsv @@ websearch_to_tsquery('simple', :q)
ORDER BY ts_rank_cd(cs.tsv, websearch_to_tsquery('simple', :q dc.ord ASC
LIMIT :k
"""), {"doc_id": document_id, "user_id": tenant_id.value, "q": qu

return [
Chunk(id=cid, tenant_id=tenant_id, document_id=DocumentId(doc for (cid, txt) in rows
]

rows = db.execute(sql(""" SELECT cs.id, cs.text FROM chunk_store cs
WHERE cs.user_id = :user_id
AND cs.tsv @@ websearch_to_tsquery('simple', :q)
ORDER BY ts_rank_cd(cs.tsv, websearch_to_tsquery('simple', :q)) D LIMIT :k
 
"""), {"user_id": tenant_id.value, "q": query, "k": top_k}).all()

return [
Chunk(id=cid, tenant_id=tenant_id, document_id=DocumentId(""), te for (cid, txt) in rows
]





Batch embeddings + dedup mapping in indexing (4
worker
(ده اﻟﺠﺰء replace indexing core) app/workers/tasks.py 4.1
ﻳﻔﺘﺮض وﺟﻮد:

 








Copy code  
 
document_reader (get_stored_file)
text_extractor
(embed_many ﻳﺪﻋﻢ) cached_embeddings chunk_dedup_repo (upsert chunk_store + replace_document_chunks)
vector_store (upsert_points)
 

python

import hashlib import structlog
from app.workers.celery_app import celery_app
from app.domain.entities import TenantId, DocumentId from app.core.bootstrap import get_container
from app.application.services.chunking import chunk_text_token_aware, ChunkSpec log = structlog.get_logger()
def _chunk_hash(text: str) -> str: norm = " ".join(text.split())
return hashlib.sha256(norm.encode("utf-8")).hexdigest()

@celery_app.task(
name="index_document",
 
bind=True,
autoretry_for=(Exception,), retry_backoff=True,
retry_kwargs={"max_retries": 5},
)
def index_document(self, *, tenant_id: str, document_id: str) -> dict: c = get_container()

repo = c["document_repo"]
reader = c["document_reader"] extractor = c["text_extractor"]
cached_emb = c["cached_embeddings"] dedup_repo = c["chunk_dedup_repo"] vector_store = c["vector_store"]

tenant = TenantId(tenant_id)
doc_id = DocumentId(document_id)

repo.set_status(tenant_id=tenant, document_id=doc_id, status="processing") try:
stored = reader.get_stored_file(tenant_id=tenant, document_id=doc_id) if not stored:
raise ValueError("Document not found")

extracted = extractor.extract(stored.path, stored.content_type) if not extracted.text.strip():
raise ValueError("No text extracted")

chunks_text = chunk_text_token_aware(extracted.text, spec=ChunkSpec(max_t if not chunks_text:
raise ValueError("No chunks produced")

# --- dedup hashing (stable)
hashes = [_chunk_hash(t) for t in chunks_text]

# --- unique-by-hash for batch embeddings (big speedup)
unique_text_by_hash: dict[str, str] = {} for h, t in zip(hashes, chunks_text):
unique_text_by_hash.setdefault(h, t)
 
unique_hashes = list(unique_text_by_hash.keys())
unique_texts = [unique_text_by_hash[h] for h in unique_hashes]

# Batch embeddings (with cache)
unique_vectors = cached_emb.embed_many(unique_texts)
vec_by_hash = {h: v for h, v in zip(unique_hashes, unique_vectors)}

# --- upsert into chunk_store (dedup) + build mapping order
chunk_ids_in_order: list[str] = []
for h, t in zip(hashes, chunks_text):
chunk_id = dedup_repo.upsert_chunk_store(tenant_id=tenant, chunk_hash chunk_ids_in_order.append(chunk_id)

# Map document → chunks order
dedup_repo.replace_document_chunks( tenant_id=tenant,
document_id=doc_id.value,
chunk_ids_in_order=chunk_ids_in_order,
)

# --- upsert vectors to Qdrant with minimal payload
vector_store.ensure_collection()

# For Qdrant: we must embed in same order as ids (even if ids repeat, vec
vectors_in_order = [vec_by_hash[h] for h in hashes]

vector_store.upsert_points( ids=chunk_ids_in_order,
vectors=vectors_in_order, tenant_id=tenant.value,
document_id=doc_id.value,
)

repo.set_status(tenant_id=tenant, document_id=doc_id, status="indexed") log.info("indexed_document", tenant_id=tenant_id, document_id=document_id return {"ok": True, "chunks": len(chunk_ids_in_order)}

except Exception as e:
repo.set_status(tenant_id=tenant, document_id=doc_id, status="failed", er
 
log.exception("index_failed", tenant_id=tenant_id, document_id=document_i raise




AskHybridUseCase updated: doc-filter + hydrate + (5
fusion + rerank
Fusion (RRF) + conversions 5.1
. ScoredChunk و rrf_fusion وﺟﻮد اﻓﺘﺮض

app/application/use_cases/ask_question_hybrid.py (REPLACE) 5.2

python

from dataclasses import dataclass
from app.domain.entities import TenantId, Answer, Chunk
from app.application.ports.vector_store import VectorStorePort
from app.application.ports.keyword_store import KeywordStorePort from app.application.ports.reranker import RerankerPort
from app.application.ports.llm import LLMPort
from app.application.ports.chunk_text_reader import ChunkTextReaderPort from app.application.services.embedding_cache import CachedEmbeddings from app.application.services.fusion import rrf_fusion
from app.application.services.scoring import ScoredChunk
from app.application.services.hydrate import hydrate_chunk_texts
from app.application.services.prompt_builder import build_rag_prompt

@dataclass
class AskHybridRequest: tenant_id: str
question: str

# optional: restrict search within single document (ChatPDF mode)
document_id: str | None = None

k_vec: int = 30
k_kw: int = 30
Copy code
fused_limit: int = 40
 
rerank_top_n: int = 8

class AskQuestionHybridUseCase: def  init (
self,
*,
cached_embeddings: CachedEmbeddings, vector_store: VectorStorePort,
keyword_store: KeywordStorePort,
chunk_text_reader: ChunkTextReaderPort, reranker: RerankerPort,
llm: LLMPort,
) -> None:
self._emb = cached_embeddings self._vec = vector_store
self._kw = keyword_store
self._text = chunk_text_reader self._rerank = reranker
self._llm = llm

def execute(self, req: AskHybridRequest) -> Answer: tenant = TenantId(req.tenant_id)
q_vec = self._emb.embed_one(req.question)

# 1) Vector candidates (IDs + scores, no text)
vec_hits = self._vec.search_scored( query_vector=q_vec,
tenant_id=tenant, top_k=req.k_vec,
document_id=req.document_id,
)

# hydrate vector text from DB (needed for rerank/prompt)
vec_chunks_hydrated = hydrate_chunk_texts( tenant_id=tenant,
chunks=[h.chunk for h in vec_hits], reader=self._text,
)
vec_hits_h = [ScoredChunk(chunk=c, score=h.score) for c, h in zip(vec_chu
 
# 2) Keyword candidates (already includes text)
kw_chunks = self._kw.search( query=req.question,
tenant_id=tenant, top_k=req.k_kw,
document_id=req.document_id,
)
kw_hits = [ScoredChunk(chunk=c, score=1.0) for c in kw_chunks] # rank-on

# 3) Fusion (RRF)
fused = rrf_fusion(vector_hits=vec_hits_h, keyword_hits=kw_hits, out_limi fused_chunks = [s.chunk for s in fused]

# 4) Rerank (CrossEncoder local)
reranked = self._rerank.rerank(query=req.question, chunks=fused_chunks, t

# 5) Generate answer
prompt = build_rag_prompt(question=req.question, chunks=reranked) text = self._llm.generate(prompt, temperature=0.2, max_tokens=700)

return Answer(text=text, sources=[c.id for c in reranked])





API endpoint supports document_id (Optional) (6
app/api/v1/routes_queries.py (update ask-hybrid body)

python

from pydantic import BaseModel, Field
from app.application.use_cases.ask_question_hybrid import AskHybridRequest

class AskHybridBody(BaseModel):
question: str = Field(min_length=2) document_id: str | None = None
k_vec: int = Field(default=30, ge=1, le=200) k_kw: int = Field(default=30, ge=1, le=200)
 

 


Bootstrap wiring (complete) (7
(ﺣّﺪث/أﺿﻒ) app/core/bootstrap.py

 
device=settings.cross_encoder_device,
)

ask_hybrid_uc = AskQuestionHybridUseCase( cached_embeddings=cached_embeddings, vector_store=vector_store,
keyword_store=keyword_store,
chunk_text_reader=chunk_text_reader, reranker=reranker,
llm=llm,
)

return {
# ...
"chunk_text_reader": chunk_text_reader, "keyword_store": keyword_store,
"chunk_dedup_repo": chunk_dedup_repo, "reranker": reranker,
"ask_hybrid_use_case": ask_hybrid_uc,
}





(8 ﻣلاﺣﻈﺎت ﺗﺸﻐﻴﻞ واﺧﺘﺒﺎر سرﻳﻊ
Index document 8.1
chunking + dedup + batch embeddings + Qdrant minimal ﻳﻌﻤﻞ PDF → worker ارﻓﻊ	 
.payload

Ask within doc (doc-filtered) 8.2


 
Copy code  
 
Ask tenant-wide 8.3
 
. document_id ﺑﺪون


(9 ﺗﺤﺴﻴﻦ ﺻﻐﻴﺮ “لاز ” ﻟﻮ ﻫﺘﻮﺳﻊ
ﻣﺴﺘﻨﺪات ﻋﺒﺮ chunk ﻟﻨﻔﺲ id = chunk_id (dedup)، upsert إن ﺑﻤﺎ :Qdrant points dedup	 
.payload document_id ھﯿﺴﺘﺒﺪل ﻣﺨﺘﻠﻔﺔ
:اﻟﺼﺤﻴﺢ production ﺣﻞ
Postgres ﻋﺒﺮ doc-filter وﺗﻌﻤﻞ payload (tenant-only) ﻣﻦ document_id ﺗﺸﻴﻞ إﻣﺎ	 
ﻓﻘﻂ
ﺑﻤﺮﺟﻊ وﺗﺤﺘﻔﻆ Qdrant point id = f"{doc_id}:{ord}" (unique per doc) ﺗﺨﻠﻲ أو	  (واﺣﺪ ﺣﻘﻞ ﻳﺰﻳﺪ payload minimal ﻟﻜﻦ) payload ﻓﻲ chunk_id
ﻟﻮ ﻋﺎﻳﺰ “الأدق” ﻟﻠﻤﺴﺘﻨﺪات:
، chunk_id + tenant_id + document_id ﻳﺤﺘﻮي payload و Qdrant point id = doc_id:ord اﺟﻌﻞ
.chunk_id ﻋﺒﺮ اﻟﻨﺺ hydrate ﺛﻢ

أﻗﺪر أﻛﺘṇ ﻟﻚ patch ده ﻓﻮرًا ﻟﻮ ﺗﺤ.ṇ


إذا ﻋﺎﻳﺰ ﻧﻜﻤّﻞ ﻣﺒﺎشرة:
(doc-filter vector أﻓﻀﻞ) per-doc ﻟﻴﻜﻮن Qdrant IDs ﺗﻌﺪﻳﻞ (A)
(اﻟﻨﺺ ﺑﺪون) chunk_id reference ﻣﻊ ”doc-filter vector search via payload“ إﺿﺎﻓﺔ (B)
Eval harness (retrieval hit rate + rerank gain) إﺿﺎﻓﺔ (C)




ﻣﻬﻤﺘﻚ Senior Full-Stack + Technical Writer ﻛﻤﻄﻮر ﺗﻌﻤﻞ أﻧﺖ
إﻧﺸﺎء ﻣشروع ﺑﺮﻣﺠﻲ ﻛﺎﻣﻞ ﻓﻲ ﺷﻜﻞ Repository GitHub اﺣﺘﺮاﻓﻲ.
أرﻳﺪك أن ﺗﻨﺘﺞ ﻟﻲ ﻫﻴﻜﻞ رﻳﺒﻮ ﻛﺎﻣﻞ ﻟﻤشروع ﺑﻌﻨﻮان: ]اﻛﺘṇ اﺳﻢ اﻟﻤشروع ﻫﻨﺎ[ ﺑﺎﺳﺘﺨﺪام:


[... / Python / JS / TS] :اﻟﺒﺮﻣﺠﺔ ﻟﻐﺔ
 
[FastAPI, React, Node, etc] :اﻟﻔﺮﻳﻤﻮرﻛﺎت


mini-RAG system, REST API, ML project, :ﻣﺜلًا] :اﻟﻤشروع ﻧﻮع
[etc


اﻟﻤﺘﻄﻠﺒﺎت:


ﻫﻴﻜﻞ اﻟﺮﻳﺒﻮ


أﻧشئ ﻫﻴﻜﻞ ﻣﺠﻠﺪات اﺣﺘﺮاﻓﻲ، ﻣﺜﺎل:


/src ﻟﻠﻜﻮد الأﺳﺎسي


/notebooks ﻟﻠﺘﺠﺎرị واﻟشرح اﻟﺘﻔﺎﻋﻠﻲ


/docs ﻟﻤﻠﻔﺎت اﻟﺘﻮﺛﻴﻖ ﺑﺼﻴﻐﺔ md.


/tests لاﺧﺘﺒﺎرات اﻟﻮﺣﺪة


أي ﻣﺠﻠﺪات إﺿﺎﻓﻴﺔ ضرورﻳﺔ ﻟﻠﺒﺮوﺟﻴﻜﺖ )ﻣﺜﻞ data/, configs/,
.(/scripts




اﻋﺮض اﻟﻬﻴﻜﻞ ﻓﻲ ﺷﻜﻞ ﺷﺠﺮة ﻣﻠﻔﺎت ﻣﻊ وﺻﻒ ﺳﻄﺮ واﺣﺪ ﻟﻜﻞ ﻣﻠﻒ/ﻣﺠﻠﺪ.
 
ﻣﻠﻔﺎت اﻟﺘﻮﺛﻴﻖ ﻓﻲ /docs
أﻧشئ ﻣﻠﻔﺎت Markdown اﻟﺘﺎﻟﻴﺔ ﻣﻊ ﻣﺤﺘﻮى اﺣﺘﺮاﻓﻲ، ﺑﺎﻟﻠﻐﺔ ]اﻟﻌﺮﺑﻴﺔ + الإﻧﺠﻠﻴﺰﻳﺔ[ ﻟﻮ أﻣﻜﻦ:


README.md ﻓﻲ ﺟﺬر اﻟﻤشروع:


وﺻﻒ ﻣﺨﺘصر ﻟﻠﻤشروع، اﻟﻬﺪف، ال_features_ اﻟﺮﺋﻴﺴﻴﺔ.


.(…Python version, dependencies) اﻟﺘﺸﻐﻴﻞ ﻣﺘﻄﻠﺒﺎت


ﺧﻄﻮات اﻟﺘﻨﺼﻴṇ واﻟﺘﺸﻐﻴﻞ ﺧﻄﻮة ﺑﺨﻄﻮة.


ﻣﺜﺎل ﻋﻤﻠﻲ end-to-end ﻟﺘﺸﻐﻴﻞ اﻟﻤشروع أو اﺳﺘﺪﻋﺎء .API




docs/architecture.md


شرح ﻣﻌﻤﺎري ﺗﻔﺼﻴﻠﻲ: اﻟﻤﻜﻮﻧﺎت، اﻟـmodules، اﻟـservices، وﻛﻴﻔﻴﺔ ﺗﻔﺎﻋﻠﻬﺎ.


sequence / component :ﻣﺜﺎل) ﻧصي ﻣﻌﻤﺎري رﺳﻢ
.(description




docs/modules.md


classesال أﻫﻢ ﻣﺴﺌﻮﻟﻴﺘﻪ، :Module / Package ﻟﻜﻞ شرح
.functionsوال
 




docs/workflows.md


ﺳﻴﻨﺎرﻳﻮﻫﺎت اﺳﺘﺨﺪام رﺋﻴﺴﻴﺔ، ﻣﺜﻞ: “إﺿﺎﻓﺔ ﻣﺼﺪر ﻟﻠـ”RAG،
.”inference”، “Training pipeline ﺗﺸﻐﻴﻞ“




docs/contributing.md


،naming conventions ﻟﻠﻜﻮد، style guide اﻟﻤﺴﺎﻫﻤﺔ، ﻗﻮاﻋﺪ
وإرﺷﺎدات ﻟﻜﺘﺎﺑﺔ اﺧﺘﺒﺎرات.






ﻛﻮد ﻣﻨﻈﻢ ﻣﻊ ﺗﻌﻠﻴﻘﺎت Comments ﺗﻌﻠﻴﻤﻴﺔ


اﻛﺘṇ اﻟﻜﻮد ﻓﻲ ﻣﻠﻔﺎت ﺣﻘﻴﻘﻴﺔ ﺗﺤﺖ /src وﻟﻴﺲ ﻓﻲ .Notebook


:ﺗﻮﺿﺢ Function و Class ﻟﻜﻞ ﻣﻔﺼﻠﺔ docstrings اﺳﺘﺨﺪم


اﻟﻬﺪف


parametersاﻟـ


returnاﻟـ
 




أﺿﻒ ﺗﻌﻠﻴﻘﺎت ﺳﻄﺮﻳﺔ # ﻟشرح اﻟﺨﻄﻮات اﻟﻤﻬﻤﺔ، وﺧﺎﺻﺔ الأﺟﺰاء اﻟﻤﻌﻤﺎرﻳﺔ أو اﻟﻤﻨﻄﻘﻴﺔ اﻟﻤﻌﻘﺪة.


ﺣﺎﻓﻆ ﻋﻠﻰ SOLID code, clean ﻗﺪر الإﻣﻜﺎن، وأﺳﻤﺎء واﺿﺤﺔ ﻟﻠﻤﺘﻐﻴﺮات واﻟـ.functions




/notebooks ﻓﻲ واﻟﺘﺠﺮﺑﺔ ﻟﻠشرح Notebooks


أﻧشئ واﺣﺪ أو أﻛﺜﺮ ﻣﻦ Notebooks، ﻣﺜﺎل:


notebooks/01_intro_and_setup.ipynb


notebooks/02_end_to_end_example.ipynb


notebooks/03_experiments.ipynb




ﻛﻞ Notebook ﻳﺤﺘﻮي ﻋﻠﻰ:


ﺧلاﻳﺎ Markdown ﺗشرح اﻟﻔﻜﺮة ﻧﻈﺮﻳﺎً ﺧﻄﻮة ﺑﺨﻄﻮة وﺑﺎلأﺳﻠﻮị اﻟﺘﻌﻠﻴﻤﻲ.


ﺧلاﻳﺎ ﻛﻮد ﺗﺴﺘﻮرد ﻣﻦ /src وﺗﻄﺒﻖ أﻣﺜﻠﺔ ﻋﻤﻠﻴﺔ، ﻣﻊ شرح ﻣﺎ ﻳﻔﻌﻠﻪ ﻛﻞ ﺟﺰء.
 


أﻣﺜﻠﺔ ﺣﻘﻴﻘﻴﺔ ﻟﺘﺸﻐﻴﻞ اﻟﻮﻇﺎﺋﻒ الأﺳﺎﺳﻴﺔ )ﻣﺜﺎل: رﻓﻊ docs ﻟﻠـRAG، ﺗﻨﻔﻴﺬ query، ﺗﻘﻴﻴﻢ اﻟﻨﺘﺎﺋﺞ.(…






شرح ﺗﻔﺼﻴﻠﻲ ﻟﻜﻞ ﻧﻘﻄﺔ


src/core/retriever.py, ﻣﺜلًا) ﻣﻬﻢ ﻛﻮد ﻣﻠﻒ ﻟﻜﻞ
:(…src/api/routes.py


اشرح ﻓﻲ docs أو ﻓﻲ Notebook ﻟﻤﺎذا ﺗﻢ ﺗﺼﻤﻴﻤﻪ ﺑﻬﺬا اﻟﺸﻜﻞ، ﻣﺎ اﻟﺒﺪاﺋﻞ، وﻣﺎ اﻟﻤﺰاﻳﺎ.




أرﻳﺪ ﺗﻮﺿﻴﺢ rationale ﻟﻠﺘﺼﻤﻴﻢ )ﻟﻴﻪ ﻋﻤﻠﻨﺎ ﻛﺪه؟( وﻟﻴﺲ ﻣﺠﺮد “إﻳﻪ اﻟﻠﻲ ﺑﻴﺤﺼﻞ؟.”




ﻣﻠﻔﺎت ﻣﺴﺎﻋﺪة أﺳﺎﺳﻴﺔ


dependencies ﻣﻊ requirements.txt أو pyproject.toml
ﻣﻨﻈﻤﺔ.


env.example. ﻳﻮﺿﺢ اﻟﻤﺘﻐﻴﺮات اﻟﺒﻴﺌﻴﺔ اﻟﻤﻄﻠﻮﺑﺔ.


make run, ﻣﺜﻞ) اﻟﺘﺸﻐﻴﻞ لاﺧﺘﺼﺎرات tasks.py أو Makefile
 
.(make test, make format




ﺧﺮوج ﻣﻨﻈﻢ


أﻋﻄﻨﻲ اﻟﻨﺎﺗﺞ ﻋﻠﻰ ﻣﺮاﺣﻞ:


أولاً: ﻓﻘﻂ ﺷﺠﺮة اﻟﻤﻠﻔﺎت + وﺻﻒ ﻛﻞ ﻣﻠﻒ.


ﺑﻌﺪ ﻣﺎ أواﻓﻖ، اﺑﺪأ ﺑﺈرﺳﺎل ﻣﺤﺘﻮى ﻛﻞ ﻣﻠﻒ ﻋﻠﻰ ﺣﺪة )اﺑﺪأ
.(اﻟﺘﻔﺼﻴﻠﻴﺔ docs ﺛﻢ notebooks ﺛﻢ src ﺛﻢ READMEﺑﺎﻟـ




ﻓﻲ ﻛﻞ ﻣﻠﻒ، اﺳﺘﺨﺪم blocks code ﻣﻨﺎﺳﺒﺔ ﻟﻠﻐﺔ.




أﺳﻠﻮị اﻟشرح


اﻛﺘṇ اﻟشرح داﺧﻞ اﻟـcomments وﻣﻠﻔﺎت اﻟـdocs وNotebooks ﺑﺄﺳﻠﻮị ﺗﻌﻠﻴﻤﻲ ﻣﻨﺎﺳṇ ﻟﻤﻄﻮر ﻋﻨﺪه ﺧﺒﺮة ﻣﺘﻮﺳﻄﺔ – ﻣﺘﻘﺪﻣﺔ، ﻟﻜﻦ ﻋﺎوز ﻳﻔﻬﻢ اﻟﺘﺼﻤﻴﻢ ﻣﻦ اﻟﺼﻔﺮ ﺣﺘﻰ الإﻧﺘﺎج.


رﻛﺰ ﻋﻠﻰ: أﻓﻀﻞ اﻟﻤﻤﺎرﺳﺎت، أﺧﻄﺎء ﺷﺎﺋﻌﺔ ﻳﺠṇ ﺗﺠﻨﺒﻬﺎ، وﻧﺼﺎﺋﺢ ﻟﺘﺤﻮﻳﻞ اﻟﻤشروع لإﻧﺘﺎج ﺣﻘﻴﻘﻲ. ﻋﺪل و ﺣﺴﻦ و ﺧﺼﺺ اﻟﺒﺮﻣﺒﺖ ده ﻟﻠﻤشروع ده


Production-Ready RAG Starter: FastAPI +) اﻟﺤﺎﻟﻲ ﻟﻤشروﻋﻨﺎ ﱠﺼﺺ	وُﻣﺨ ﱠﺴﻦ ُﻣﺤ Prompt ده .أﻛﻴﺪ Postgres + Qdrant + Redis + Celery + Hybrid Search + Cross-Encoder Rerank +
 
.”Senior Full-Stack + Technical Writer“ ịوﺑﺄﺳﻠﻮ (Idempotency + Chunk Dedup
ﻣﻬﻢ: ﻓﻲ اﻟﻨﺴﺨﺔ دي ﺧﻠﻴﺖ اﻟﺤﻘﻮل ﻗﺎﺑﻠﺔ ﻟﻠﺘﻌﺒﺌﺔ )اﺳﻢ اﻟﻤشروع(…، وﻗﻴّﺪت اﻟﻨﺘﻴﺠﺔ ﻋﻠﻰ ﻣﺮاﺣﻞ زي ﻣﺎ ﻃﻠﺒﺖ:
اﻟﻤﺮﺣﻠﺔ 1 = ﺷﺠﺮة اﻟﻤﻠﻔﺎت + وﺻﻒ ﺳﻄﺮ ﻟﻜﻞ ﻣﻠﻒ ﻓﻘﻂ ﺛﻢ ﺑﻌﺪ “ﻣﻮاﻓﻘﺘﻚ” ﺗﺒﺪأ اﻟﻤﺤﺘﻮﻳﺎت ﻣﻠﻔًﺎ ﻣﻠﻔًﺎ.


✅ Pack Prompt )ﻣُﺤﺴّﻦ وﻣُﺨﺼّﺺ ﻟﻠﻤشروع(
اﻧﺴﺦ اﻟﻨﺺ اﻟﺘﺎﻟﻲ ﻛﻤﺎ ﻫﻮ واﺳﺘﺨﺪﻣﻪ:

text

ﻛﻤﻄﻮر ﺗﻌﻤﻞ أﻧﺖ Senior Full-Stack + AI Systems Architect + Technical Writer.
ﺮﺑﻲ + إﻧﺠﻠﯿﺰي ،Production-Ready ،اﺣﺘﺮاﻓﻲ Repository GitHub ﻣﮭﻤﺘﻚ إﻧﺸﺎء ﻣﺸﺮوع ﺑﺮﻣﺠﻲ ﻛﺎﻣﻞ ﻓﻲ ﺷﻜﻞ

# Project Info (fill these)
]اﻛﺘﺐ اﺳﻢ اﻟﻤﺸﺮوع ھﻨﺎ[ PROJECT_NAME:
LANGUAGE: Python 3.11+
FRAMEWORKS: FastAPI + Celery + Redis + Postgres (SQLAlchemy) + Alembic + Qdrant + PROJECT_TYPE: Production-ready RAG Starter Template (Hybrid Search + Rerank)

# Core Requirements (must implement)
1)	Clean Architecture + SOLID + Clean Code
-	ﻓﺼﻞ layers: domain / application / adapters / api / workers
-	Ports (interfaces) + Adapters (implementations)
-	Use Cases واﺿﺤﺔ
-	Dependency Injection (bootstrap/container)

2)	RAG Pipeline (End-to-End)
-	Upload document (PDF/DOCX/TXT) → extract → chunk → embed → index
-	Chunking token-aware + overlap
-	Vector store: Qdrant
-	Keyword search: Postgres FTS (generated tsvector + GIN index)
-	Hybrid retrieval: vector + keyword + RRF fusion
-	Rerank: Cross-Encoder local (SentenceTransformers)
-	Answer generation: LLM adapter (OpenAI) + prompt builder
-	Multi-tenant: tenant isolation via user_id everywhere

3)	Production Features
 
-	Idempotency hashing ﻟﻠﻤﻠﻒ (sha256) + unique(user_id, file_sha256) ﺗﻜﺮار ﻟﻤﻨﻊ inde
-	Chunk de-dup per tenant: chunk_store (user_id, chunk_hash unique) + document
-	Vector payload minimal (no text in Qdrant) + hydrate text from Postgres at r
-	Batch embeddings in indexing (embed_many) + cache embeddings in Redis
-	Document-filtered search support:
-	keyword search join document_chunks + chunk_store
-	optional vector search filter by document_id
-	Observability-friendly:
-	structured logging
-	store chat sessions + turns (sources, timings fields)

4)	Developer Experience
-	pyproject.toml (preferred) أو requirements.txt ﻣﻨﻈﻢ
-	.env.example ﺷﺎﻣﻞ
-	Makefile أو tasks.py:
-	run, worker, test, format, lint, typecheck, migrate, seed
-	docker-compose: postgres + redis + qdrant
-	tests/ unit tests skeleton + minimal meaningful tests

5)	Documentation (Arabic + English in same docs when possible)
-	README.md (root): overview, features, quickstart, E2E example (upload + ask-
-	docs/architecture.md: detailed architecture + text diagrams (components + se
-	docs/modules.md: ﻛﻞ ﺷﺮح package/module أھﻢ + واﻟﻤﺴﺆوﻟﯿﺎت classes/functions
-	docs/workflows.md: workflows (upload/index, ask-hybrid, doc-filtered chat, r
-	docs/contributing.md: coding standards, naming, style, tests, git workflow

6)	Notebooks (educational, import from src)
-	notebooks/01_intro_and_setup.ipynb
-	notebooks/02_end_to_end_rag.ipynb
-	notebooks/03_hybrid_search_and_rerank.ipynb Requirements:
-	Markdown cells: theory + rationale + pitfalls
-	Code cells: import from src/ ﻓﻘﻂ (ﺗﻜﺘﺐ ﻻ core logic داﺧﻞ notebook)

# Output Rules (VERY IMPORTANT)
:أﺧﺮج اﻟﻨﺘﺎﺋﺞ ﻋﻠﻰ ﻣﺮاﺣﻞ -
.ﻓﻘﻂ ﺷﺠﺮة اﻟﻤﻠﻔﺎت + وﺻﻒ ﺳﻄﺮ واﺣﺪ ﻟﻜﻞ ﻣﻠﻒ/ﻣﺠﻠﺪ )ﺑﺪون ﻣﺤﺘﻮى اﻟﻤﻠﻔﺎت( 1: Stage
.ﻓﻘﻂ README.md ﯾﺒﺪأ ﺑﺈرﺳﺎل ﻣﺤﺘﻮى 2 Stage :ﺑﻌﺪ ﻣﺎ أواﻓﻖ
ﺛﻢ Stage 3: src/ (ﺑﻤﻠﻒ ﻣﻠﻒ) ﺛﻢ workers/ ﺛﻢ tests/ ﺛﻢ docs/ ﺛﻢ notebooks/.
- داﺧﻞ ﺗﻘﺪﯾﻤﮫ ﯾﺘﻢ ﻣﻠﻒ ﻛﻞ code block ﻣﻨﺎﺳﺐ (```python / ```md / ```toml / ```yaml ...).
 
src/. إﻻ اﻟﺘﻲ ﺗﺴﺘﻮرد ﻣﻦ notebook ﻣﻤﻨﻮع ﺗﻮﻟﯿﺪ أﻛﻮاد داﺧﻞ - ؟trade-offs ﻟﻤﺎذا ﺻﻤﻤﺖ ھﺬا؟ ﻣﺎ اﻟﺒﺪاﺋﻞ؟ rationale: رﻛّﺰ ﻋﻠﻰ -
-	اﻟﻤﻤﺎرﺳﺎت ﺑﺄﻓﻀﻞ اﻟﺘﺰم: typing, docstrings, errors, defensive coding, config-driven.
.اﻓﺘﺮض أن اﻟﻘﺎرئ ﻣﻄﻮر ﻣﺘﻮﺳﻂ-ﻣﺘﻘﺪم ﯾﺮﯾﺪ ﻓﮭﻢ إﻧﺘﺎﺟﻲ ﻣﻦ اﻟﺼﻔﺮ -

# Additional Constraints
-	ﺗﺴﺘﺨﺪم ﻻ Elasticsearchاﺳﺘﺨﺪم ؛ Postgres FTS + Qdrant ﻟﻠـ hybrid.
-	ﻓﻲ اﻟﻨﺺ ﺗﺨﺰن ﻻ Qdrant payload.
-	اﺳﺘﺨﺪم Alembic migrations (including generated tsvector).
- اﺳﺘﺨﺪم Cross-Encoder rerank ﻛـ ﻣﺤﻠﻲ default، ﻣﻦ ﺗﻌﻄﯿﻠﮫ إﻣﻜﺎﻧﯿﺔ ﻣﻊ config.
-	ﺷﻲء ﻛﻞ اﺟﻌﻞ multi-tenant-safe by design.





ﺗﺨﺼﻴﺺ سرﻳﻊ )اﺧﺘﻴﺎري ﻟﻜﻦ ﻣﻔﻴﺪ(
ﻟﻮ ﺗﺤṇ ﺗﺨﻠﻴﻪ “ﺟﺎﻫﺰ ﻟلاﺳﺘﺨﺪام” ﺑﺪون ﺗﻌﺒﺌﺔ ﻛﻞ ﻣﺮة، ﻏﻴّﺮ اﻟﺤﻘﻮل ﺑﺎلأﻋﻠﻰ ﻣﺜلًا:
PROJECT_NAME: ragforge-starter	  PROJECT_NAME: bank-rag-platform أو	 


ﻣلاﺣﻈﺔ ﻣﻬﻤﺔ ﻋﻦ شرط “ﺑﻌﺪ ﻣﺎ أواﻓﻖ”
أﻧﺖ ﻃﻠﺒﺖ اﻟﺘﺪرّج ﺑﺎﻟﻤﻮاﻓﻘﺔ. أﻧﺎ ﻫﺎﻟﺘﺰم ﺑﻪ.
اﻟﺨﻄﻮة اﻟﺘﺎﻟﻴﺔ )ﻟﻮ أﻧﺖ ﻣﺴﺘﻌﺪ( ﻫﻲ إﻧﻚ ﺗﻜﺘṇ ﺑﺲ:    اﺳﻢ اﻟﻤشروع
(اﺧﺘﻴﺎري) OpenAI + Ollama ولا ﻓﻘﻂ LLM = OpenAI ﺗﺮﻳﺪ ﻫﻞ	 
ﻟﻜﻦ ﻟﻮ ﻣﺶ ﻋﺎﻳﺰ ﺗﻜﺘṇ ﺣﺎﺟﺔ: أﻧﺎ ﻣﻤﻜﻦ أﻓﺘﺮض اﺳﻢ اﻓﺘﺮاضي ﻣﺜﻞ rag-starter-pro وأﻣشي.

ﻗﻮﻟّﻲ اﺳﻢ اﻟﻤشروع وﻫﻨﺒﺪأ ﻣﺒﺎشرة ﺑـ 1 :Stage ﺷﺠﺮة اﻟﻤﻠﻔﺎت + وﺻﻒ ﻛﻞ ﻣﻠﻒ.




full implmentaition plan in full details
 



Copy code
 
ﻟﻤشروع ”Production → ﺻﻔﺮ ﻣﻦ“ (Implementation Plan) ﻛﺎﻣﻠﺔ ﺗﻨﻔﻴﺬ ﺧﻄﺔ دي .ﺗﻤﺎم
:ﺑﻨﻴﻨﺎه اﻟﻠﻲ ﺑﻨﻔﺲ Production-Ready RAG Starter
 
FastAPI + Postgres(Alembic) + Qdrant + Redis + Celery + Hybrid Search (FTS+Vector) + RRF + Cross-Encoder Rerank + Idempotency + Chunk De-dup + Minimal payload +
.Batch embeddings + Chat logs
اﻟﺨﻄﺔ ﻣﻜﺘﻮﺑﺔ ﺑﺤﻴﺚ ﺗﻘﺪر ﺗﺤﻮﻟﻬﺎ ﻣﺒﺎشرة ﻟـ tickets + milestones + repo .GitHub



(0 اﻟﻬﺪف اﻟﻨﻬﺎﺋﻲ
ﻣﻨﺼﺔ RAG ﻣﺘﻌﺪدة اﻟﻤﺴﺘﺄﺟﺮﻳﻦ :(Multi-tenant)
Upload docs (PDF/DOCX/TXT) → Extract → Chunk → Dedup → Embed (batch) → Index	 
(Qdrant minimal) + Store text + FTS in Postgres Query: Hybrid retrieval (doc-filter optional) → Fusion (RRF) → Rerank (Cross-Encoder)	 
→ Generate answer (LLM) → Save chat turn + metrics


Milestone Map (High level) (1
Repo bootstrap + DX + CI basics .1
Clean Architecture skeleton + config/logging/DI .2
Storage layer: Postgres + Alembic schema (Users/Documents/Chunks/Chat) .3
Upload + Idempotency file hashing .4 Indexing worker: extraction/chunking/dedup/batch embed/qdrant upsert .5 Retrieval: vector minimal + hydrate + keyword FTS (tenant + doc-filter) .6 Hybrid fusion + rerank + prompt + ask endpoint .7
Observability + testing + hardening .8
Production packaging + deployment guidance .9


Repo Bootstrap & Developer Experience (DX) (2
(ﻣﺒﺪﺋًﻴﺎ) Structure 2.1
src/app/... (domain/application/adapters/api/workers/core)	 
   /docs ﺗﻮﺛﻴﻖ ﻣﻌﻤﺎري
 
(ﻓﻘﻂ import from src) ịوﺗﺠﺎر ﺗﻌﻠﻴﻢ /notebooks	 
tests/ unit/integration skeleton	 
docker/ compose for postgres/redis/qdrant	 
scripts/ seed, maintenance	 
env.example, Makefile , pyproject.toml .	 

Tooling standards 2.2
+Python 3.11	 
black أو Formatting: ruff format	 
Lint: ruff	 
(ُﻳﻀﺎف ﺛﻢ sprint أول اﺧﺘﻴﺎري) Typing: mypy	 
Testing: pytest	 
Pre-commit (optional but recommended)	 
Tickets
pyproject.toml + ruff/pytest config Makefile: run , worker , test , lint , format , migrate , seed
docker-compose: postgres/redis/qdrant
CI workflow: run lint + tests


Clean Architecture Skeleton (Core) (3
Layers 3.1
domain/: Entities + Value Objects (TenantId, DocumentId, Chunk, Answer)   application/: Ports (Protocols) + Use Cases + Services (pure)   adapters/: DB repos + Qdrant + OpenAI + Redis + extraction   
api/: FastAPI routes (thin)	 
workers/: Celery tasks (thin, orchestrate use cases/services)	 
core/: config/logging/bootstrap/DI	 

Baseline components 3.2 Config via pydantic-settings	   Structured logs via structlog	 
Container bootstrap with singletons (cached)	 
Health endpoint	 
 
config + logging setup base entities + ports skeleton DI container baseline
FastAPI app factory


”Database (Postgres + Alembic) — schema “production-first (4
Tables 4.1
users .1
id (uuid str) , email , api_key , timestamps	 
Index: users.api_key	 
documents .2
id , user_id , filename , content_type , file_path , size_bytes	 
status , error , timestamps	 
idempotency: file_sha256 + unique( user_id , file_sha256 )	 
:Chunk de-dup model .3
chunk_store	  id , user_id , chunk_hash(sha256) , text	 
tsv GENERATED ALWAYS AS to_tsvector(...) STORED	 
Unique: ( user_id , chunk_hash )	 
Index: GIN(tsv)	 
document_chunks	  document_id , ord , chunk_id	 
PK( document_id , ord )	 
Index: chunk_id , document_id	 
:Chat .4
chat_sessions : id , user_id , title , timestamps	 
chat_turns : id , session_id , user_id , question , answer , sources_json	 
metrics fields: embed_ms, search_ms, llm_ms, tokens, retrieval_k	 
Migration strategy 4.2
One migration per feature set	  Use raw SQL for generated tsvector + GIN	 
 
Alembic init + env.py imports Migration: users + documents
Migration: documents.file_sha256 + unique index Migration: chunk_store + document_chunks + generated tsv + indexes
Migration: chat tables + indexes Seed script: create demo user api_key


Auth & Tenant Isolation (5
API key header 5.1
Header: X-API-KEY		  Lookup users by api_key	  tenant_id = user_id (internal)	 
Guard rails 5.2 Always filter DB reads/writes by user_id		  Qdrant filter always includes tenant_id	  Never accept tenant_id from request body			 
Tickets
deps.py: get_tenant_id via DB lookup add tests: invalid key → 401


Upload Workflow + Idempotency hashing (6
Upload endpoint /v1/documents/upload 6.1
:Process
read bytes .1
compute sha256 .2
check existing document by (tenant, hash) .3
if exists: return {status:"already_exists", document_id}	 
save file to filesystem (LocalFileStore) OR S3 adapter later .4
 
create document with hash, set status queued .5
enqueue Celery indexing .6
Tickets
FileStore adapter + size limits  	  DocumentIdempotency repo (get/create with IntegrityError race safe)  	 
UploadDocumentUseCase update  	  API route + curl example  	  integration test (mock queue)  	 


Indexing Worker Pipeline (Celery) (7
Responsibilities (Worker thin) 7.1 Update status processing	  Extract text (pdf/docx/txt)		 
Chunk token-aware overlap	  Chunk hashing (normalize + sha256)	 
De-dup insert into chunk_store + build mapping document_chunks (ord)	 
Batch embeddings (unique hashes only) + cache	 
:Qdrant upsert minimal payload	 
:(ﻣﻬﻢ اﺧﺘﻴﺎر) point_id strategy	 
ﻓﻲ document_id ﻟﻜﻦ Option A (tenant-wide dedup): point_id = chunk_id	 
(doc-filter vector ﻟﻮ ṇﻣﻨﺎﺳ ﻏﻴﺮ) ﻳﺘﺒﺪل payload
Option B (recommended): point_id = f"{doc_id}:{ord}" payload	 
{tenant_id, document_id, chunk_id} ﺑﺪﻗﺔ doc-filter vector ﻳﺴﻤﺢ ✅	  (Postgres ﻣﻦ hydrate) اﻟﻨﺺ ﻳﻜﺮر لا ✅	 
docs ﺑﻴﻦ overwrite ṇﻳﺘﺠﻨ ✅	 
   ﻓﻲ plan ﻫﻨﺴﺘﺨﺪم B Option لأﻧﻬﺎ الأﻛﺜﺮ اﺳﺘﻘﺮارًا ﻟلإﻧﺘﺎج

Batch embeddings 7.2
build unique texts map by chunk_hash	 
embed_many(unique_texts) once	 
expand vectors by ord	 
 
Tickets
TextExtractor adapter  	  Chunking service (token-aware) + tests  	 
ChunkDedupRepo: upsert chunk_store + replace document_chunks  	 
CachedEmbeddings.embed_many  	  QdrantVectorStore.upsert_points (point_id doc:ord, payload chunk_id)  	 
worker task end-to-end + logging + retries  	 


Retrieval Layer (8
Vector search (Qdrant minimal) 8.1
search returns point ids + payload (chunk_id, doc_id)	  hydrate chunk text from Postgres by chunk_ids	 
Keyword search (Postgres FTS) 8.2
:Two modes tenant-wide: chunk_store only	 
doc-filtered: join document_chunks dc + chunk_store cs filter dc.document_id	 

Fusion (RRF) 8.3
input: scored vector hits + ranked keyword hits	 
output: fused list top N	 
Tickets
ChunkTextReaderPort + Postgres adapter KeywordStore doc-filter join + tests VectorStore search_scored (doc filter optional)
hydrate service fusion service (RRF) + tests


Rerank (Cross-Encoder Local) (9
Default 9.1
sentence-transformers CrossEncoder	 
 
Model default: cross-encoder/ms-marco-MiniLM-L-6-v2	 
Device configurable: cpu/cuda	  rerank top_n small (8) for latency	 
Fallback 9.2
if model load fails or disabled: return fused order	 
Tickets
RerankerPort + CrossEncoder adapter
config knobs tests: deterministic behavior with stub


Ask Endpoint (Hybrid) (10
v1/queries/ask-hybrid/ 10.1
:Request
question	 
optional document_id	 
knobs: k_vec, k_kw, fused_limit, rerank_top_n	 
:Flow
embed question .1 vector search (doc filter optional) .2 hydrate vector chunk texts .3 keyword search (doc filter optional) .4
fusion RRF .5
rerank cross-encoder .6
build prompt with citations [chunk_id] .7
LLM generate .8
save chat turn (optional session_id) .9
Tickets
AskHybridUseCase full  	  prompt builder (strict grounding + citations)  	 
API route + examples  	 
 
ChatRepo integration (turn logging)


Chat Sessions & Observability (11
Add endpoints (optional but recommended) 11.1
POST /v1/chat/sessions create session	 
GET /v1/chat/sessions list	 
GET /v1/chat/sessions/{id}/turns	 

Metrics fields 11.2
:measure durations in use case	  embed_ms, search_ms, llm_ms	 
tokens if available (OpenAI response usage)	 
Tickets
ChatRepo + endpoints timing instrumentation structured logs with request_id


Testing Strategy (12
Unit tests (fast) 12.1
chunking token aware	  hash normalization stable	 
RRF fusion	  prompt builder	 
keyword SQL builder (smoke) with sqlite? (FTS needs Postgres; keep as integration)	 

Integration tests (docker-compose) 12.2
run postgres+redis+qdrant	  test upload → worker index → ask-hybrid returns answer (with stub LLM if needed)	 
Tickets
pytest harness testcontainers optional
 
CI job running unit tests (integration optional)


Production Hardening Checklist (13
Config validation at startup (missing keys)	 
Rate limiting (later)	  File storage backend (S3) abstraction	 
Secrets management	  DB connection pool settings	  Celery retry policy + dead-letter	 
Qdrant collection migration strategy	  Data retention / GDPR (delete docs cascades to mappings)	 


(ﻟﻠﺘﻨﻔﻴﺬ ﻗﺎﺑﻞ Day-by-day) Delivery Plan (14
( أﻳﺎ 1–2) Phase A
Repo bootstrap + config/logging + FastAPI skeleton + docker-compose	  Alembic setup + users/documents migrations + seed user	 
Upload endpoint wired	 

( أﻳﺎ 2–4) Phase B
Worker indexing pipeline: extraction/chunking/dedup/batch embed/qdrant minimal	 
payload Document status transitions	 
Basic ask endpoint vector-only	 

( أﻳﺎ 2–3) Phase C
Keyword store (tenant + doc-filter join)	  Hybrid fusion + CrossEncoder rerank	  Ask-hybrid endpoint complete	 
( أﻳﺎ 1–2) Phase D
Chat sessions/turns + timings	  Tests + docs + notebooks	 
 

 

(ﻣﺨﺘصر Rationale) Design ﻗﺮارات (15
.use cases ﻟﻤﻌﻈﻢ ﻛﻔﺎﻳﺔ + أﺳﻬﻞ deployو ﺗﻌﻘﻴﺪ أﻗﻞ :Elasticsearch ﺑﺪل Postgres FTS	 
.أﻗﻞ cost + ﻗﻮي doc-filter + ﺣﻘﻴﻘﻲ chunk_store + document_chunks: dedup	 
.ﻟﻠﻨﺺ واﺣﺪ storage + source of truth ﺗﻮﻓﻴﺮ :Qdrant minimal payload	 
.ﻣﺮﺟﻌﻲ chunk_id ﺑـ الاﺣﺘﻔﺎظ ﻣﻊ docs ﻋﺒﺮ overwrite ﻳﻤﻨﻊ :point_id = doc:ord	 
.scores ﺑﻴﻦ calibration ﺑﺪون RRF: robust	 
.LLM rerank ﺑـ ﻣﻘﺎرﻧﺔ ﺗﻜﻠﻔﺔ ﺑﺄﻗﻞ Cross-Encoder rerank: boost precision	 


اﻟﺨﻄﻮة اﻟﺘﺎﻟﻴﺔ
ﻟﻮ ﻋﺎﻳﺰ أﺣﻮّل اﻟﺨﻄﺔ دي إﻟﻰ Milestones + Issues GitHub )ﺟﺎﻫﺰة ﻟﻠﻨﺴﺦ(، ﻗﻮﻟّﻲ اﺳﻢ اﻟﻤشروع وﻫﻨﻈﻤﻬﺎ ﻛﺎﻟﺘﺎﻟﻲ:
Milestones	 
Epics	  effort + acceptance criteria ﺑﺘﻘﺪﻳﺮ Issues	 
1/2	 
