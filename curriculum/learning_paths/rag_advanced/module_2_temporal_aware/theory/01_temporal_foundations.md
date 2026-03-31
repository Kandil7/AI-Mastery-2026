# Theory 1: Temporal Foundations for RAG

## 1.1 Introduction to Temporal Awareness

### 1.1.1 Why Temporal Context Matters

In many domains, the validity and relevance of information changes over time:

| Domain | Temporal Sensitivity | Example |
|--------|---------------------|---------|
| News/Media | Very High | Yesterday's news is outdated |
| Software Docs | High | API changes with versions |
| Financial Data | Very High | Stock prices change constantly |
| Medical Guidelines | Medium | Guidelines update annually |
| Historical Facts | Low | Historical events don't change |

### 1.1.2 Temporal Challenges in RAG

```
Problem 1: Stale Information
Query: "What is the current pricing?"
Retrieved: Document from 2 years ago with old prices
Result: Incorrect answer

Problem 2: Temporal Ambiguity
Query: "Who is the CEO?"
Ambiguity: Current CEO or CEO at a specific time?
Result: Potentially wrong answer

Problem 3: Temporal Reasoning
Query: "How did revenue change after the acquisition?"
Requires: Understanding before/after acquisition date
Result: Complex temporal reasoning needed
```

### 1.1.3 Temporal RAG Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              TEMPORAL-AWARE RAG ARCHITECTURE                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Query ──▶ [Temporal Understanding]                         │
│              │                                              │
│              ├──▶ Intent Classification (current/historical)│
│              ├──▶ Time Range Extraction                     │
│              └──▶ Temporal Entity Recognition               │
│                                                             │
│              ▼                                              │
│         [Temporal Index]                                    │
│              │                                              │
│              ├──▶ Time-partioned chunks                     │
│              ├──▶ Event timelines                           │
│              └──▶ Version tracking                          │
│                                                             │
│              ▼                                              │
│         [Time-Aware Retrieval]                              │
│              │                                              │
│              ├──▶ Recency boosting                          │
│              ├──▶ Time-decay scoring                        │
│              └──▶ Temporal filtering                        │
│                                                             │
│              ▼                                              │
│         [Temporal Grounding]                                │
│              │                                              │
│              ├──▶ Timestamp citation                        │
│              ├──▶ Validity period indication                │
│              └──▶ Temporal confidence scoring               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 1.2 Temporal Entity Recognition

### 1.2.1 Types of Temporal Entities

```python
from enum import Enum

class TemporalEntityType(Enum):
    # Absolute time references
    ABSOLUTE_DATE = "absolute_date"      # "2024-01-15", "March 2024"
    ABSOLUTE_TIME = "absolute_time"      # "3:00 PM", "14:30"
    
    # Relative time references
    RELATIVE_DATE = "relative_date"      # "yesterday", "last week"
    RELATIVE_TIME = "relative_time"      # "in 2 hours", "3 days ago"
    
    # Periods and durations
    TIME_PERIOD = "time_period"          # "Q1 2024", "fiscal year"
    DURATION = "duration"                # "for 3 months", "during 2023"
    
    # Events with temporal significance
    EVENT = "event"                      # "after the merger", "pre-launch"
    FREQUENCY = "frequency"              # "daily", "every Monday"
```

### 1.2.2 Implementation Example

```python
import re
from datetime import datetime, timedelta
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class TemporalEntity:
    text: str
    entity_type: TemporalEntityType
    normalized_value: str  # ISO format when possible
    confidence: float
    start_pos: int
    end_pos: int

class TemporalEntityExtractor:
    """Extract temporal entities from text."""
    
    PATTERNS = {
        TemporalEntityType.ABSOLUTE_DATE: [
            (r'\d{4}-\d{2}-\d{2}', '%Y-%m-%d'),  # 2024-01-15
            (r'\d{2}/\d{2}/\d{4}', '%m/%d/%Y'),  # 01/15/2024
            (r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}', '%B %Y'),
        ],
        TemporalEntityType.RELATIVE_DATE: [
            (r'yesterday', None),
            (r'tomorrow', None),
            (r'last\s+(week|month|year)', None),
            (r'next\s+(week|month|year)', None),
        ],
        TemporalEntityType.TIME_PERIOD: [
            (r'Q[1-4]\s+\d{4}', None),  # Q1 2024
            (r'\d{4}\s+fiscal\s+year', None),
        ],
    }
    
    def extract(self, text: str, reference_time: datetime = None) -> List[TemporalEntity]:
        """Extract all temporal entities from text."""
        reference_time = reference_time or datetime.now()
        entities = []
        
        for entity_type, patterns in self.PATTERNS.items():
            for pattern, date_format in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity = self._create_entity(
                        text, match, entity_type, date_format, reference_time
                    )
                    entities.append(entity)
        
        # Sort by position
        entities.sort(key=lambda e: e.start_pos)
        return entities
    
    def _create_entity(self, text: str, match, entity_type: TemporalEntityType,
                       date_format: str, reference_time: datetime) -> TemporalEntity:
        """Create temporal entity from regex match."""
        matched_text = match.group()
        
        # Normalize based on type
        normalized = self._normalize_temporal(
            matched_text, entity_type, date_format, reference_time
        )
        
        return TemporalEntity(
            text=matched_text,
            entity_type=entity_type,
            normalized_value=normalized,
            confidence=0.9,  # Could be improved with ML model
            start_pos=match.start(),
            end_pos=match.end()
        )
    
    def _normalize_temporal(self, text: str, entity_type: TemporalEntityType,
                            date_format: str, reference_time: datetime) -> str:
        """Normalize temporal expression to ISO format when possible."""
        text_lower = text.lower()
        
        if entity_type == TemporalEntityType.ABSOLUTE_DATE and date_format:
            try:
                dt = datetime.strptime(text, date_format)
                return dt.isoformat()
            except:
                return text
        
        elif entity_type == TemporalEntityType.RELATIVE_DATE:
            if 'yesterday' in text_lower:
                return (reference_time - timedelta(days=1)).strftime('%Y-%m-%d')
            elif 'tomorrow' in text_lower:
                return (reference_time + timedelta(days=1)).strftime('%Y-%m-%d')
            elif 'last week' in text_lower:
                return (reference_time - timedelta(weeks=1)).strftime('%Y-%m-%d')
            elif 'last month' in text_lower:
                return (reference_time - timedelta(days=30)).strftime('%Y-%m-%d')
        
        return text  # Return original if can't normalize
```

## 1.3 Time-Decay Functions

### 1.3.1 Common Decay Functions

```python
import math
from datetime import datetime, timedelta

class TimeDecayFunctions:
    """Common time-decay functions for temporal scoring."""
    
    @staticmethod
    def exponential_decay(doc_time: datetime, query_time: datetime, 
                          half_life_days: float = 30) -> float:
        """
        Exponential decay: score = 2^(-t/half_life)
        
        Most common decay function. Score halves every half_life_days.
        """
        days_old = (query_time - doc_time).total_seconds() / 86400
        return 2 ** (-days_old / half_life_days)
    
    @staticmethod
    def linear_decay(doc_time: datetime, query_time: datetime,
                     max_age_days: float = 365) -> float:
        """
        Linear decay: score = 1 - (t/max_age)
        
        Score decreases linearly to 0 at max_age.
        """
        days_old = (query_time - doc_time).total_seconds() / 86400
        return max(0, 1 - (days_old / max_age_days))
    
    @staticmethod
    def sigmoid_decay(doc_time: datetime, query_time: datetime,
                      midpoint_days: float = 60, steepness: float = 0.1) -> float:
        """
        Sigmoid decay: score = 1 / (1 + e^(steepness * (t - midpoint)))
        
        Gradual decay with sharp drop around midpoint.
        """
        days_old = (query_time - doc_time).total_seconds() / 86400
        return 1 / (1 + math.exp(steepness * (days_old - midpoint_days)))
    
    @staticmethod
    def step_decay(doc_time: datetime, query_time: datetime,
                   thresholds: List[tuple]) -> float:
        """
        Step decay: different scores for different age ranges.
        
        thresholds: [(max_days, score), ...]
        Example: [(7, 1.0), (30, 0.8), (90, 0.5), (365, 0.2)]
        """
        days_old = (query_time - doc_time).total_seconds() / 86400
        
        for max_days, score in sorted(thresholds):
            if days_old <= max_days:
                return score
        
        return 0.0
```

### 1.3.2 Choosing Decay Parameters

| Content Type | Recommended Half-Life | Rationale |
|--------------|----------------------|-----------|
| News articles | 1-7 days | Very time-sensitive |
| Software docs | 30-90 days | Changes with releases |
| Research papers | 180-365 days | Knowledge evolves slowly |
| Historical data | No decay | Timeless information |
| Financial reports | 30-90 days | Quarterly updates |

## 1.4 Summary

Key takeaways:
- Temporal awareness is critical for time-sensitive domains
- Extract temporal entities to understand time references
- Apply appropriate time-decay functions based on content type
- Balance recency with relevance for optimal results
