# Content Management and Standards for LMS

## Table of Contents

1. [Content Standards Overview](#1-content-standards-overview)
2. [xAPI/Tin Can Deep Dive](#2-xapitin-can-deep-dive)
3. [cmi5 Implementation](#3-cmi5-implementation)
4. [Open Badges and Credentials](#4-open-badges-and-credentials)
5. [H5P Interactive Content](#5-h5p-interactive-content)
6. [Video Streaming Standards](#6-video-streaming-standards)

---

## 1. Content Standards Overview

### Standards Comparison

| Standard | Version | Primary Use | Key Feature |
|----------|---------|-------------|-------------|
| SCORM | 1.2/2004 | Legacy content | Simple tracking |
| xAPI | 1.0 | Modern learning | Flexible tracking |
| cmi5 | 1.0 | LMS-controlled | Enhanced tracking |
| LTI | 1.3 | Tool integration | SSO and grade passback |
| Open Badges | 3.0 | Credentials | Verifiable badges |

---

## 2. xAPI/Tin Can Deep Dive

### xAPI Statement Structure

```json
{
  "actor": {
    "mbox": "mailto:learner@example.com",
    "name": "John Doe"
  },
  "verb": {
    "id": "http://adlnet.gov/expapi/verbs/completed",
    "display": {"en-US": "completed"}
  },
  "object": {
    "id": "https://lms.example.com/course/intro-ml",
    "definition": {
      "type": "http://adlnet.gov/expapi/activities/course",
      "name": {"en-US": "Introduction to Machine Learning"}
    }
  },
  "result": {
    "score": {"scaled": 0.85, "raw": 85, "min": 0, "max": 100},
    "success": true,
    "completion": true,
    "duration": "PT1H30M"
  },
  "context": {
    "platform": "LMS Platform",
    "language": "en-US"
  },
  "timestamp": "2026-02-10T14:30:00Z"
}
```

---

## 3. cmi5 Implementation

### cmi5 vs SCORM

| Feature | SCORM | cmi5 |
|---------|-------|------|
| Launch Control | Package-based | LMS-controlled |
| Sequencing | Package-based | LMS-defined |
| Completion | Completion ribbon | Objective-based |
| Bookmarking | Resume capability | LMS state management |

---

## 4. Open Badges and Credentials

### Badge Assertion

```json
{
  "@context": "https://w3id.org/openbadges/v2",
  "type": "Assertion",
  "id": "https://lms.example.com/badges/12345",
  "recipient": {
    "type": "email",
    "identity": "learner@example.com",
    "hashed": false
  },
  "badge": {
    "id": "https://lms.example.com/badges/definitions/advanced-python",
    "type": "BadgeClass"
  },
  "verification": {
    "type": "hosted"
  },
  "issuedOn": "2026-02-10",
  "evidence": "https://lms.example.com/courses/python-advanced/completion"
}
```

---

## 5. H5P Interactive Content

### H5P Content Types

| Category | Content Types |
|----------|---------------|
| Interactive | Quiz, Drag and Drop, Fill in the Blanks |
| Media | Video, Audio, Image Hotspots |
| Interactive Video | Branches, Quiz overlays |
| Authoring | Course Presentation, Timeline |

---

## 6. Video Streaming Standards

### HLS vs DASH

| Feature | HLS | DASH |
|---------|-----|------|
| Developed by | Apple | MPEG |
| Browser Support | Safari, Edge | All modern browsers |
| Adaptive | Chunk-based | MPD manifest |
| DRM | FairPlay | Widevine, PlayReady |

---

## Quick Reference

### Content Compliance Checklist

- SCORM 1.2 package validation
- xAPI statement schema validation
- cmi5 conformance testing
- Open Badges verification
- WCAG 2.2 accessibility
- Video caption requirements
