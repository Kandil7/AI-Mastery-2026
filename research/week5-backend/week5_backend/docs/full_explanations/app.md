# App package (app/*.py)

## app/main.py

### create_app() -> FastAPI
- Purpose: constructs and configures the FastAPI application.
- Steps:
  - Loads settings to read the app version.
  - Creates FastAPI instance with title/version.
  - Registers routers: health, ingest, query, eval, feedback.
  - Returns the configured app.

### app = create_app()
- Creates a module-level FastAPI instance for ASGI servers.

## app/schemas.py

### IngestRequest (pydantic BaseModel)
- Fields:
  - `tenant_id: str` -> tenant boundary for indexing.
  - `source_type: str` -> `web`, `file`, `pdf`.
  - `uri: str` -> source location.
  - `metadata: Dict[str, Any]` -> optional metadata.

### IngestResponse
- Fields:
  - `ingestion_id: str` -> document id assigned at ingest.
  - `status: str` -> ingestion status string.

### QueryRequest
- Fields:
  - `tenant_id: str` -> tenant boundary for retrieval.
  - `question: str` -> user question.
  - `filters: Dict[str, Any]` -> metadata filters.
  - `top_k: int` -> top K requested.
  - `mode: str` -> `vector`, `hybrid`, or `agentic`.

### Citation
- Fields:
  - `chunk_id: str`
  - `doc_id: str`
  - `score: float`
  - `snippet: str`

### QueryResponse
- Fields:
  - `answer: str`
  - `citations: List[Citation]`
  - `trace_id: str`
  - `model: str`

### EvalRequest
- Fields:
  - `dataset_id: str`
  - `mode: str`

### EvalResponse
- Fields:
  - `run_id: str`
  - `status: str`

### FeedbackRequest
- Fields:
  - `trace_id: str`
  - `rating: int`
  - `notes: Optional[str]`

### FeedbackResponse
- Fields:
  - `status: str`

## app/routers/query.py

### router (APIRouter)
- Prefix: `/query`.

### query(request: QueryRequest) -> QueryResponse
- Purpose: API endpoint for online queries.
- Steps:
  - Calls `run_query_pipeline(...)` with request payload.
  - Returns a `QueryResponse` populated with result fields.

## app/routers/ingest.py

### router (APIRouter)
- Prefix: `/ingest`.

### ingest(request: IngestRequest) -> IngestResponse
- Purpose: API endpoint for ingestion.
- Steps:
  - Calls `ingest_document(...)`.
  - Returns ingestion id with status `accepted`.

## app/routers/health.py

### router (APIRouter)
- No prefix.

### healthz() -> dict
- Purpose: health check endpoint.
- Returns:
  - `status: ok`
  - `version: <app_version>`

## app/routers/feedback.py

### router (APIRouter)
- Prefix: `/feedback`.

### feedback(request: FeedbackRequest) -> FeedbackResponse
- Purpose: stub endpoint for feedback collection.
- Current behavior: accepts request and returns `status: ok`.

## app/routers/eval.py

### router (APIRouter)
- Prefix: `/eval`.

### eval_run(request: EvalRequest) -> EvalResponse
- Purpose: start an evaluation run.
- Steps:
  - Calls `run_evaluation(dataset_id, mode)`.
  - Returns run id and `status: running`.
