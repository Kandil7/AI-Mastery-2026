# Module 6: Self-Correcting and Iterative RAG

## 📋 Module Overview

**Duration:** 3-4 weeks (20-25 hours)  
**Difficulty:** Advanced  
**Prerequisites:** Module 1-5 completion, LLM fundamentals

This module teaches you to build RAG systems that can self-evaluate, detect errors, and iteratively improve their retrieval and generation through reflection, verification, and correction loops.

---

## 🎯 Learning Objectives

### Remember
- Define self-correction and iterative refinement
- Identify hallucination types in RAG
- Recall verification strategies

### Understand
- Explain the reflection loop architecture
- Describe confidence estimation techniques
- Summarize error detection approaches

### Apply
- Implement answer verification pipelines
- Build iterative retrieval loops
- Create self-correction mechanisms

### Analyze
- Compare single-pass vs. iterative retrieval
- Diagnose hallucination patterns
- Evaluate correction effectiveness

### Evaluate
- Assess when iteration is beneficial
- Critique confidence calibration
- Judge cost vs. quality trade-offs

### Create
- Design self-correcting RAG architectures
- Develop custom verification prompts
- Build iterative refinement pipelines

---

## 🔄 Self-Correcting Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              SELF-CORRECTING RAG ARCHITECTURE               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Query ──▶ [Initial Retrieval]                              │
│              │                                              │
│              ▼                                              │
│         [First-Pass Generation]                             │
│              │                                              │
│              ▼                                              │
│         [Self-Evaluation] ◀────┐                            │
│              │                 │                            │
│              ├──▶ Faithfulness │                            │
│              ├──▶ Relevance    │                            │
│              ├──▶ Completeness │                            │
│              └──▶ Confidence   │                            │
│              │                 │                            │
│              ▼                 │                            │
│         [Error Detection]      │                            │
│              │                 │                            │
│              ├──▶ Hallucination│                            │
│              ├──▶ Missing info │                            │
│              └──▶ Contradiction│                            │
│              │                 │                            │
│              ▼                 │  Iteration needed          │
│         [Correction Plan] ─────┴─────▶ Yes                  │
│              │                                              │
│              │ No                                           │
│              ▼                                              │
│         [Final Answer]                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Answer Verification

```python
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

class VerificationStatus(Enum):
    VERIFIED = "verified"
    POTENTIAL_HALLUCINATION = "potential_hallucination"
    MISSING_CONTEXT = "missing_context"
    CONTRADICTION = "contradiction"
    UNCERTAIN = "uncertain"

@dataclass
class VerificationResult:
    status: VerificationStatus
    confidence: float
    issues: List[str]
    suggestions: List[str]

class AnswerVerifier:
    """
    Verify generated answers against retrieved context.
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def verify(self, query: str, answer: str, 
                     context: List[str]) -> VerificationResult:
        """
        Verify answer faithfulness to context.
        """
        issues = []
        suggestions = []
        
        # Check 1: Faithfulness (is answer grounded in context?)
        faithfulness = await self._check_faithfulness(answer, context)
        if not faithfulness['faithful']:
            issues.extend(faithfulness['ungrounded_claims'])
            suggestions.append("Revise answer to only include information from context")
        
        # Check 2: Relevance (does answer address query?)
        relevance = await self._check_relevance(query, answer)
        if relevance['score'] < 0.7:
            issues.append("Answer may not fully address the query")
            suggestions.append("Ensure answer directly responds to the query")
        
        # Check 3: Completeness (is all relevant info included?)
        completeness = await self._check_completeness(query, answer, context)
        if not completeness['complete']:
            issues.extend(completeness['missing_info'])
            suggestions.append("Include additional relevant information from context")
        
        # Check 4: Contradiction detection
        contradictions = await self._check_contradictions(answer, context)
        if contradictions:
            issues.extend(contradictions)
            suggestions.append("Resolve contradictory statements")
        
        # Determine overall status
        if contradictions:
            status = VerificationStatus.CONTRADICTION
        elif issues and any("hallucination" in i.lower() for i in issues):
            status = VerificationStatus.POTENTIAL_HALLUCINATION
        elif issues:
            status = VerificationStatus.MISSING_CONTEXT
        elif len(issues) == 0:
            status = VerificationStatus.VERIFIED
        else:
            status = VerificationStatus.UNCERTAIN
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            faithfulness, relevance, completeness, contradictions
        )
        
        return VerificationResult(
            status=status,
            confidence=confidence,
            issues=issues,
            suggestions=suggestions
        )
    
    async def _check_faithfulness(self, answer: str, 
                                   context: List[str]) -> dict:
        """Check if answer claims are grounded in context."""
        prompt = f"""
        Given the following context and answer, identify any claims in the answer
        that are NOT supported by the context (potential hallucinations).
        
        Context:
        {' '.join(context)}
        
        Answer:
        {answer}
        
        List each unsupported claim. If all claims are supported, return empty list.
        
        Format as JSON: {{"faithful": bool, "ungrounded_claims": []}}
        """
        
        response = await self.llm.generate(prompt, response_format='json')
        return response
    
    async def _check_relevance(self, query: str, answer: str) -> dict:
        """Check if answer is relevant to query."""
        prompt = f"""
        Rate how well this answer addresses the query (0-1 scale).
        
        Query: {query}
        Answer: {answer}
        
        Format as JSON: {{"score": float, "reasoning": string}}
        """
        
        response = await self.llm.generate(prompt, response_format='json')
        return response
    
    async def _check_completeness(self, query: str, answer: str,
                                   context: List[str]) -> dict:
        """Check if answer includes all relevant information."""
        prompt = f"""
        Given the query and context, identify any important information
        from the context that is missing from the answer.
        
        Query: {query}
        Context: {' '.join(context)}
        Answer: {answer}
        
        Format as JSON: {{"complete": bool, "missing_info": []}}
        """
        
        response = await self.llm.generate(prompt, response_format='json')
        return response
    
    async def _check_contradictions(self, answer: str, 
                                     context: List[str]) -> List[str]:
        """Detect contradictions between answer and context."""
        prompt = f"""
        Identify any statements in the answer that contradict the context.
        
        Context: {' '.join(context)}
        Answer: {answer}
        
        List contradictions. Return empty list if none.
        
        Format as JSON: {{"contradictions": []}}
        """
        
        response = await self.llm.generate(prompt, response_format='json')
        return response.get('contradictions', [])
    
    def _calculate_confidence(self, faithfulness: dict, relevance: dict,
                              completeness: dict, contradictions: list) -> float:
        """Calculate overall verification confidence."""
        scores = [
            1.0 if faithfulness.get('faithful', False) else 0.3,
            relevance.get('score', 0.5),
            1.0 if completeness.get('complete', False) else 0.5,
            1.0 if not contradictions else 0.2
        ]
        return sum(scores) / len(scores)
```

### Iterative Retrieval

```python
class IterativeRetriever:
    """
    Retriever that iteratively refines search based on intermediate results.
    """
    
    def __init__(self, base_retriever, llm_client, max_iterations: int = 3):
        self.retriever = base_retriever
        self.llm = llm_client
        self.max_iterations = max_iterations
    
    async def retrieve(self, query: str, top_k: int = 10) -> dict:
        """
        Iterative retrieval with query refinement.
        """
        iteration_results = []
        all_documents = []
        
        current_query = query
        
        for iteration in range(self.max_iterations):
            # Step 1: Retrieve with current query
            results = await self.retriever.search(current_query, top_k=top_k)
            iteration_results.append({
                'iteration': iteration,
                'query': current_query,
                'results': results
            })
            
            # Step 2: Evaluate if more retrieval needed
            evaluation = await self._evaluate_retrieval_sufficiency(
                query, results, all_documents
            )
            
            if evaluation['sufficient']:
                break
            
            # Step 3: Generate refined query for next iteration
            current_query = await self._generate_refined_query(
                query, current_query, results, evaluation['gaps']
            )
            
            # Add results to accumulated documents
            all_documents.extend(results)
        
        # Deduplicate and rerank all results
        final_results = self._deduplicate_and_rerank(all_documents, query)
        
        return {
            'query': query,
            'results': final_results[:top_k],
            'iterations': len(iteration_results),
            'iteration_details': iteration_results
        }
    
    async def _evaluate_retrieval_sufficiency(self, original_query: str,
                                               results: List[dict],
                                               all_documents: List[dict]) -> dict:
        """Evaluate if current results are sufficient."""
        prompt = f"""
        Evaluate if the retrieved documents are sufficient to answer the query.
        
        Query: {original_query}
        
        Retrieved documents (summaries):
        {' '.join([r.get('content', '')[:200] for r in results])}
        
        Is this sufficient? What information gaps exist?
        
        Format as JSON: {{"sufficient": bool, "gaps": []}}
        """
        
        response = await self.llm.generate(prompt, response_format='json')
        return response
    
    async def _generate_refined_query(self, original_query: str,
                                       current_query: str,
                                       results: List[dict],
                                       gaps: List[str]) -> str:
        """Generate refined query to address gaps."""
        prompt = f"""
        Generate a refined search query to address the identified gaps.
        
        Original query: {original_query}
        Current query: {current_query}
        
        Information gaps:
        {' '.join(gaps)}
        
        Retrieved so far (summaries):
        {' '.join([r.get('content', '')[:100] for r in results])}
        
        Generate a new query that will help fill the gaps.
        Return only the query string.
        """
        
        refined_query = await self.llm.generate(prompt)
        return refined_query.strip()
    
    def _deduplicate_and_rerank(self, documents: List[dict],
                                 query: str) -> List[dict]:
        """Deduplicate and rerank accumulated documents."""
        # Deduplicate by ID or content hash
        seen = set()
        unique_docs = []
        for doc in documents:
            doc_id = doc.get('id', hash(doc.get('content', '')))
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
        
        # Rerank by relevance to original query
        # (In production, use cross-encoder reranker)
        return unique_docs
```

### Self-Correction Loop

```python
class SelfCorrectingRAG:
    """
    Complete self-correcting RAG system with reflection loops.
    """
    
    def __init__(self, retriever, generator, verifier, llm_client):
        self.retriever = retriever
        self.generator = generator
        self.verifier = verifier
        self.llm = llm_client
        self.max_corrections = 2
    
    async def generate(self, query: str) -> dict:
        """
        Generate answer with self-correction.
        """
        # Initial retrieval and generation
        retrieval_result = await self.retriever.retrieve(query)
        context = [doc['content'] for doc in retrieval_result['results']]
        
        answer = await self.generator.generate(query, context)
        
        # Self-correction loop
        correction_history = []
        
        for correction_round in range(self.max_corrections):
            # Verify current answer
            verification = await self.verifier.verify(query, answer, context)
            
            correction_history.append({
                'round': correction_round,
                'answer': answer,
                'verification': verification
            })
            
            # If verified, return
            if verification.status == VerificationStatus.VERIFIED:
                break
            
            # Generate correction plan
            correction_plan = await self._generate_correction_plan(
                query, answer, context, verification
            )
            
            # Apply corrections
            if correction_plan['needs_revision']:
                answer = await self._apply_corrections(
                    query, answer, context, correction_plan
                )
                
                # If corrections involved new retrieval, get new context
                if correction_plan.get('needs_more_retrieval'):
                    new_query = correction_plan.get('refined_query', query)
                    retrieval_result = await self.retriever.retrieve(new_query)
                    context = [doc['content'] for doc in retrieval_result['results']]
            else:
                break
        
        return {
            'query': query,
            'answer': answer,
            'context': context,
            'correction_history': correction_history,
            'final_verification': verification
        }
    
    async def _generate_correction_plan(self, query: str, answer: str,
                                         context: List[str],
                                         verification: VerificationResult) -> dict:
        """Generate plan for correcting identified issues."""
        prompt = f"""
        Based on the verification results, create a correction plan.
        
        Query: {query}
        Current Answer: {answer}
        
        Verification Issues:
        {verification.issues}
        
        Suggestions:
        {verification.suggestions}
        
        Create a correction plan:
        1. Does the answer need revision?
        2. What specific changes are needed?
        3. Is more retrieval needed? If so, what query?
        
        Format as JSON: {{
            "needs_revision": bool,
            "changes": [],
            "needs_more_retrieval": bool,
            "refined_query": string or null
        }}
        """
        
        plan = await self.llm.generate(prompt, response_format='json')
        return plan
    
    async def _apply_corrections(self, query: str, answer: str,
                                  context: List[str],
                                  correction_plan: dict) -> str:
        """Apply corrections to answer."""
        prompt = f"""
        Revise the answer based on the correction plan.
        
        Query: {query}
        Original Answer: {answer}
        
        Context: {' '.join(context)}
        
        Correction Plan:
        - Changes needed: {correction_plan['changes']}
        
        Generate a revised answer that addresses all issues.
        """
        
        revised_answer = await self.llm.generate(prompt)
        return revised_answer
```

---

## 📚 Module Structure

```
module_6_self_correcting/
├── README.md
├── theory/
│   ├── 01_self_correction_fundamentals.md
│   ├── 02_answer_verification.md
│   ├── 03_iterative_retrieval.md
│   ├── 04_hallucination_detection.md
│   └── 05_reflection_patterns.md
├── labs/
│   ├── lab_1_answer_verification/
│   ├── lab_2_iterative_retrieval/
│   └── lab_3_self_correcting_rag/
├── knowledge_checks/
├── coding_challenges/
├── solutions/
└── further_reading.md
```

---

## Hallucination Types

| Type | Description | Detection Method |
|------|-------------|------------------|
| Fact Fabrication | Inventing facts not in context | Faithfulness check |
| Context Misattribution | Attributing info to wrong source | Source verification |
| Temporal Confusion | Mixing up timelines | Temporal consistency |
| Logical Contradiction | Self-contradictory statements | Logic checking |
| Over-generalization | Making broad claims from limited info | Specificity check |

---

## Cost vs. Quality Trade-offs

```
Single-pass RAG:
- Latency: ~200ms
- Cost: $0.001/query
- Quality: Baseline

2-iteration Self-Correcting:
- Latency: ~600ms
- Cost: $0.003/query
- Quality: +25% accuracy

3-iteration with Verification:
- Latency: ~1000ms
- Cost: $0.005/query
- Quality: +40% accuracy, -60% hallucinations
```

---

*Last Updated: March 30, 2026*
