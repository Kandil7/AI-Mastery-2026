# Archive - Legacy and Deprecated Code

This directory contains legacy, deprecated, and historical code that is no longer part of the active codebase.

## ⚠️ Warning

**Do not use this code in production.** This code is preserved for:
- Historical reference
- Learning purposes
- Migration reference
- Understanding design decisions

## Directory Structure

```
archive/
├── README.md                  # This file
├── legacy/                    # Recently archived code (Phase 1 cleanup)
│   ├── agents/
│   │   └── legacy_agents.py   # Old agent implementations
│   └── rag/
│       ├── legacy_rag.py      # Original RAG pipeline
│       ├── legacy_advanced_rag.py  # Advanced RAG (deprecated)
│       ├── retrieval/
│       │   └── legacy_retrieval.py
│       └── reranking/
│           └── legacy_reranking.py
├── duplicate_root_modules/    # Duplicate modules (to be reviewed)
│   ├── arabic-llm/
│   └── benchmarks/
├── legacy_documentation/      # Old documentation
└── old_numbered_dirs/         # Historical numbered research dirs
    └── research/
```

## Recently Archived (Phase 1 - March 2026)

The following files were moved to archive during Phase 1 restructuring:

### RAG Legacy Code
- `legacy_rag.py` - Original RAG pipeline implementation
- `legacy_advanced_rag.py` - Advanced RAG with enterprise features
- `retrieval/legacy_retrieval.py` - Legacy retrieval strategies
- `reranking/legacy_reranking.py` - Legacy reranking methods

### Agents Legacy Code
- `legacy_agents.py` - Original agent implementations

**Reason for archival:** These files were replaced by newer implementations in the specialized RAG architectures and modern agent orchestration systems.

## Cleanup History

### Phase 1 (March 31, 2026)
- Created `archive/legacy/` structure
- Moved 5 legacy files from active `src/` directory
- Removed temporary markdown files

### Planned Future Cleanup
- Review and consolidate `duplicate_root_modules/`
- Archive or remove `old_numbered_dirs/`
- Clean up `legacy_documentation/`

## Migration Notes

If you're migrating code that depends on archived modules:

1. **RAG Pipeline**: Use `src.rag.RAGPipeline` instead of `legacy_rag.py`
2. **Advanced RAG**: Use specialized RAG architectures in `src.rag.specialized/`
3. **Agents**: Use `src.agents.orchestration` and `src.agents.multi_agent_systems`

## Contributing

When deprecating code:
1. Move to appropriate archive subdirectory
2. Update this README with archival date and reason
3. Update import statements in active code
4. Add deprecation notices before next release

---

**Last Updated:** March 31, 2026
**Archive Status:** Active (Phase 1 Complete)
