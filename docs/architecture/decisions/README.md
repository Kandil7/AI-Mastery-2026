# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the AI-Mastery-2026 project.

## What are ADRs?

ADRs are documents that capture significant architectural decisions made during the project's development. Each ADR describes:

- The context and problem being addressed
- The decision made and its rationale
- The consequences (both positive and negative)
- Alternatives that were considered

## ADR Index

| Number | Title | Status | Date |
|--------|-------|--------|------|
| [ADR-001](adr-001-project-structure.md) | Project Structure and Module Organization | Accepted | 2026-03-31 |
| [ADR-002](adr-002-configuration-management.md) | Configuration Management Strategy | Accepted | 2026-03-31 |
| [ADR-003](adr-003-type-definitions.md) | Type Definitions and Shared Types | Accepted | 2026-03-31 |

## ADR Template

Use the [template](template.md) when creating new ADRs.

## Creating a New ADR

1. Copy `template.md` to `adr-NNN-short-title.md`
2. Fill in all sections
3. Update this index with the new ADR
4. Reference related ADRs
5. Submit for review

## ADR Lifecycle

- **Proposed** - Under discussion
- **Accepted** - Approved and being implemented
- **Deprecated** - No longer recommended, but still valid
- **Superseded** - Replaced by a newer ADR

## Related Documentation

- [Contributing Guide](../../CONTRIBUTING.md)
- [Architecture Overview](../README.md)
- [Project Structure](../../README.md#-project-structure)

---

**Last Updated:** March 31, 2026
