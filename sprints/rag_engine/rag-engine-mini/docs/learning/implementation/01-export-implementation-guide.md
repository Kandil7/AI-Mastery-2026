# Export Implementation Guide

## Overview

This document provides a comprehensive guide to the export functionality implementation in the RAG Engine Mini. The export feature allows users to download their data in various formats (PDF, Markdown, CSV, JSON), completing a critical feature that was marked as pending in the project completion checklist.

## Architecture

### Component Structure

The export functionality follows the same architectural patterns as the rest of the RAG Engine:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Layer     │────│  Application     │────│   Domain/       │
│   (routes)      │    │  Services/       │    │   Ports/        │
│                 │    │  Use Cases       │    │   Adapters      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         │ HTTP Requests          │ Business Logic        │ Interfaces &
         │                        │                       │ Implementations
         ▼                        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │  Export Service  │    │ ExportService   │
│   Endpoints     │    │  & Use Cases     │    │ Port Interface  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Components

1. **API Routes** (`src/api/v1/routes_exports.py`): FastAPI endpoints for export functionality
2. **Export Service** (`src/application/services/export_service.py`): Core export logic
3. **Export Use Case** (`src/application/use_cases/export_use_case.py`): Business logic orchestrator
4. **Dependency Injection** (`src/core/bootstrap.py`): Service registration and wiring

## Implementation Details

### 1. Export Service

The `ExportService` implements the `ExportServicePort` interface and provides:

- **Multiple Format Support**: PDF, Markdown, CSV, and JSON
- **Background Processing**: Asynchronous export jobs
- **Preview Functionality**: Sample export before full processing
- **Status Tracking**: Job status monitoring

Key methods:
```python
async def create_export_job(...) -> ExportJob
async def get_export_status(...) -> Optional[ExportJob]
async def get_export_result(...) -> Optional[ExportResult]
async def preview_export(...) -> Optional[str]
async def export_to_pdf(...) -> bytes
async def export_to_markdown(...) -> str
async def export_to_csv(...) -> str
async def export_to_json(...) -> str
```

### 2. Export Use Case

The `ExportUseCase` orchestrates the export process:

```python
class ExportUseCase:
    def __init__(self, export_service: ExportServicePort):
        self._export_service = export_service

    async def create_export_job(...)
    async def get_export_status(...)
    async def get_export_result(...)
    async def preview_export(...)
```

### 3. API Endpoints

The API provides endpoints for:
- Creating export jobs (`POST /exports`)
- Checking job status (`GET /exports/{job_id}`)
- Downloading results (`GET /exports/{job_id}/download`)
- Format listing (`GET /exports/formats`)
- Preview functionality (`POST /exports/preview`)

## API Usage

### Creating an Export Job

```bash
POST /exports
Content-Type: application/json

{
  "format": "json",
  "content_type": "documents",
  "filters": {
    "status": "indexed"
  }
}
```

### Checking Export Status

```bash
GET /exports/{job_id}
```

### Downloading Export

```bash
GET /exports/{job_id}/download
```

### Previewing Export

```bash
POST /exports/preview
Content-Type: application/json

{
  "format": "markdown",
  "content_type": "documents",
  "filters": {},
  "limit": 5
}
```

## Format-Specific Implementation

### PDF Export

Uses ReportLab to generate PDF documents:
- Converts content to styled paragraphs
- Handles multi-page documents
- Preserves formatting

### Markdown Export

Generates properly formatted Markdown:
- Headers, lists, and emphasis
- Proper document structure
- Readable formatting

### CSV Export

Creates properly formatted CSV files:
- Header row with column names
- Proper escaping of special characters
- Compliant with RFC 4180

### JSON Export

Generates structured JSON:
- Proper escaping and encoding
- Pretty-printed for readability
- Valid JSON format

## Integration Points

### Dependency Injection

The export service is registered in `src/core/bootstrap.py`:

```python
export_service = ExportService(
    document_repo=document_repo
)
export_use_case = ExportUseCase(export_service=export_service)

return {
    # ... other services
    "export_service": export_service,
    "export_use_case": export_use_case,
}
```

### API Integration

The routes are included in the main application:

```python
app.include_router(exports_router)
```

## Error Handling

The export functionality includes comprehensive error handling:

- Invalid format requests
- Unauthorized access attempts
- File system errors
- Database connection issues
- Resource limits exceeded

## Security Considerations

1. **Tenant Isolation**: Export data is limited to the requesting tenant
2. **Access Control**: Only authorized users can create/export data
3. **Rate Limiting**: Prevents abuse of export functionality
4. **File Validation**: Ensures safe file types and sizes

## Testing

The export functionality includes comprehensive tests in `tests/unit/test_export_service.py`:

- Format conversion tests
- Error condition tests
- Tenant isolation tests
- Integration tests

## Performance Considerations

1. **Background Processing**: Large exports run asynchronously
2. **Memory Management**: Efficient processing of large datasets
3. **Caching**: Frequently accessed data is cached
4. **Resource Limits**: Prevents excessive resource consumption

## Educational Value

This implementation demonstrates:

1. **Clean Architecture**: Clear separation of concerns
2. **Port/Adapter Pattern**: Interface-based design
3. **Dependency Injection**: Proper service wiring
4. **API Design**: RESTful endpoint design
5. **Async Programming**: Non-blocking operations
6. **Error Handling**: Comprehensive error management
7. **Testing**: Thorough test coverage

## Conclusion

The export functionality completes a critical feature that was marked as pending in the project completion checklist. It follows the same architectural patterns as the rest of the RAG Engine Mini, ensuring consistency and maintainability. The implementation provides a solid foundation for users to export their data in multiple formats while maintaining security and performance standards.

This addition brings the RAG Engine Mini closer to full completion, providing users with a comprehensive solution for their RAG needs including data export capabilities.