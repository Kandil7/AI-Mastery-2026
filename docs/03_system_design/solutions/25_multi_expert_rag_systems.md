# System Design Solution: Multi-Expert RAG (Mixture of Experts)

## Problem Statement

Design a Multi-Expert Retrieval-Augmented Generation (MoE-RAG) system that can:
- Dynamically route queries to the most appropriate specialized expert module
- Combine outputs from multiple domain-specific experts using learned weights
- Handle diverse query types efficiently without performance degradation
- Scale horizontally by adding new expert modules
- Maintain high accuracy across specialized domains
- Optimize computational resources by activating only relevant experts
- Provide interpretability for expert selection decisions

## Solution Overview

This system design presents a comprehensive architecture for Multi-Expert RAG (MoE-RAG) that implements a Mixture of Experts approach to dynamically route queries to specialized modules and combine their outputs. The solution addresses the critical need for AI systems that can handle diverse query types efficiently by leveraging domain-specific expertise while maintaining overall system performance and resource efficiency.

## 1. High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    └─────────────────┐
│   Query Input  │────│  Expert        │────│  Specialized    │
│  (Any Domain)  │    │  Router        │    │  Experts       │
│                │    │  (Gating Net)  │    │  (Domain-      │
│                │    │  (Dynamic)     │    │  Specific)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Query        │────│  Expert       │────│  Expert       │
│  Analysis     │    │  Selection    │    │  (NLP)       │
│  & Routing    │    │  (Top-k)      │    │  (Code)       │
└─────────────────┘    └──────────────────┘    │  (Science)    │
         │                       │              │  (Finance)    │
         │                       ▼              │  (Legal)      │
         │              ┌──────────────────┐    │  (Medical)    │
         │              │  Mixture of    │────│  (Technical)  │
         │              │  Experts       │    │  (General)    │
         │              │  (Weighted)    │    └─────────────────┘
         │              └──────────────────┘            │
         │                       │                      │
         └───────────────────────┼──────────────────────┘
                                 │
┌────────────────────────────────┼─────────────────────────────────┐
│                    Multi-Expert RAG Pipeline                   │
│  ┌─────────────────┐    ┌──────────────────┐    ┌──────────┐  │
│  │  Input        │────│  Multi-Expert   │────│  Output  │  │
│  │  Processing   │    │  Processing     │    │  Gen.   │  │
│  │  (Universal)  │    │  (Parallel)     │    │  (LLM)  │  │
│  └─────────────────┘    └──────────────────┘    └──────────┘  │
└────────────────────────────────────────────────────────────────┘
```

## 2. Core Components

### 2.1 Multi-Expert RAG Core System
```python
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from typing import List, Dict, Any, Optional, Tuple
import faiss
from sentence_transformers import SentenceTransformer
import time
from abc import ABC, abstractmethod

class MultiExpertRAGCore:
    """
    Core system for Multi-Expert RAG (MoE-RAG)
    """
    def __init__(self, 
                 num_experts: int = 8,
                 expert_capacity: int = 1024,
                 model_name: str = "gpt-3.5-turbo",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.model_name = model_name
        
        # Initialize components
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.main_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize experts
        self.experts = self._initialize_experts()
        
        # Initialize routing network (gating network)
        self.routing_network = ExpertRouter(
            input_dim=384,  # Embedding dimension
            num_experts=num_experts,
            capacity=expert_capacity
        )
        
        # Initialize expert combination module
        self.expert_combiner = ExpertCombiner(num_experts)
        
        # Initialize knowledge bases for each expert
        self.expert_knowledge_bases = {i: ExpertKnowledgeBase() for i in range(num_experts)}
        
        # Expert specializations
        self.expert_specializations = {
            0: "nlp_language_processing",
            1: "code_generation",
            2: "scientific_research",
            3: "financial_analysis",
            4: "legal_documentation",
            5: "medical_knowledge",
            6: "technical_support",
            7: "general_knowledge"
        }
        
        # Performance tracking
        self.performance_tracker = ExpertPerformanceTracker(num_experts)
        
    def _initialize_experts(self) -> nn.ModuleList:
        """
        Initialize specialized expert networks
        """
        experts = nn.ModuleList()
        
        for i in range(self.num_experts):
            # Each expert is a specialized neural network
            expert = ExpertNetwork(
                input_dim=384,  # Embedding dimension
                hidden_dim=512,
                output_dim=256,
                expert_id=i
            )
            experts.append(expert)
        
        return experts
    
    def route_query(self, query: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route query to appropriate experts using the routing network
        """
        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]
        query_tensor = torch.FloatTensor(query_embedding).unsqueeze(0)
        
        # Get routing weights and selected experts
        with torch.no_grad():
            routing_weights, selected_experts = self.routing_network(query_tensor)
        
        return routing_weights, selected_experts
    
    def process_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Process query through multi-expert system
        """
        start_time = time.time()
        
        # Route query to experts
        routing_weights, selected_experts = self.route_query(query)
        
        # Retrieve relevant information for each selected expert
        expert_contexts = {}
        for expert_idx in selected_experts[0]:  # Get indices from first batch
            expert_idx = int(expert_idx.item())
            
            # Get specialized knowledge base for this expert
            knowledge_base = self.expert_knowledge_bases[expert_idx]
            
            # Retrieve relevant documents
            retrieved_docs = knowledge_base.retrieve(query, top_k)
            expert_contexts[expert_idx] = retrieved_docs
        
        # Process query with selected experts
        expert_outputs = {}
        for expert_idx in selected_experts[0]:
            expert_idx = int(expert_idx.item())
            
            # Get expert-specific context
            context = expert_contexts[expert_idx]
            
            # Process with specific expert
            expert_output = self._process_with_expert(
                query, context, expert_idx
            )
            expert_outputs[expert_idx] = expert_output
        
        # Combine expert outputs
        combined_output = self.expert_combiner.combine_outputs(
            expert_outputs, routing_weights, selected_experts
        )
        
        end_time = time.time()
        
        # Update performance tracking
        self.performance_tracker.update_query_performance(
            selected_experts[0].tolist(),
            end_time - start_time
        )
        
        return {
            'response': combined_output,
            'routing_weights': routing_weights.tolist(),
            'selected_experts': selected_experts[0].tolist(),
            'expert_outputs': expert_outputs,
            'processing_time_ms': (end_time - start_time) * 1000,
            'expert_specializations': [self.expert_specializations[i] for i in selected_experts[0].tolist()],
            'confidence': float(torch.max(routing_weights))
        }
    
    def _process_with_expert(self, query: str, context: List[Dict[str, Any]], 
                           expert_idx: int) -> str:
        """
        Process query with specific expert
        """
        # Create prompt with context
        context_str = "\\n".join([doc['content'] for doc in context])
        prompt = f"""
        You are an expert in {self.expert_specializations[expert_idx]}.
        
        Context: {context_str}
        
        Query: {query}
        
        Provide a detailed, accurate response based on your expertise:
        """
        
        # Generate response using the main model (in practice, each expert could have its own model)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        with torch.no_grad():
            outputs = self.main_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response part
        if "Response:" in response:
            response = response.split("Response:")[-1].strip()
        elif "provide a detailed" in response.lower():
            response = response.split(query)[-1].strip()
        
        return response

class ExpertRouter(nn.Module):
    """
    Expert router using gating network for dynamic routing
    """
    def __init__(self, input_dim: int, num_experts: int, capacity: int = 1024, top_k: int = 2):
        super(ExpertRouter, self).__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.capacity = capacity
        self.top_k = top_k
        
        # Routing network
        self.routing_network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Capacity parameters
        self.capacity_factor = 1.25  # Allow some over-capacity
        self.noise_epsilon = 1e-2   # Small noise for exploration
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to determine expert routing
        """
        # Get routing probabilities
        routing_probs = self.routing_network(x)
        
        # Add small noise for exploration
        noise = torch.randn_like(routing_probs) * self.noise_epsilon
        routing_probs = routing_probs + noise
        routing_probs = torch.softmax(routing_probs, dim=-1)
        
        # Select top-k experts for each input
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # Normalize probabilities for selected experts
        normalized_weights = torch.zeros_like(routing_probs)
        for i in range(x.size(0)):  # For each batch item
            normalized_weights[i, top_k_indices[i]] = torch.softmax(top_k_probs[i], dim=-1)
        
        return normalized_weights, top_k_indices

class ExpertNetwork(nn.Module):
    """
    Specialized expert network
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, expert_id: int):
        super(ExpertNetwork, self).__init__()
        self.expert_id = expert_id
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the expert network
        """
        return self.network(x)

class ExpertCombiner:
    """
    Combine outputs from multiple experts
    """
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.combination_weights = nn.Parameter(torch.ones(num_experts))
        
    def combine_outputs(self, expert_outputs: Dict[int, str], 
                       routing_weights: torch.Tensor, 
                       selected_experts: torch.Tensor) -> str:
        """
        Combine outputs from selected experts using learned weights
        """
        # In practice, this would involve more sophisticated combination
        # For this example, we'll create a weighted combination of expert outputs
        
        combined_parts = []
        total_weight = 0.0
        
        for i, expert_idx_tensor in enumerate(selected_experts[0]):  # First batch
            expert_idx = int(expert_idx_tensor.item())
            weight = routing_weights[0, expert_idx].item()
            
            if expert_idx in expert_outputs:
                output = expert_outputs[expert_idx]
                # Weight the contribution of each expert's output
                weighted_part = f"[Expert {expert_idx} ({weight:.2f}): {output[:100]}...]\\n"
                combined_parts.append(weighted_part)
                total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            normalized_parts = [part.replace(f"({weight:.2f})", f"({weight/total_weight:.2f})") 
                               for part, weight in zip(combined_parts, [routing_weights[0, int(exp_idx.item())].item() 
                                                                      for exp_idx in selected_experts[0]])]
            combined_output = "Combined response:\\n" + "".join(normalized_parts)
        else:
            combined_output = "No expert outputs available"
        
        return combined_output

class ExpertKnowledgeBase:
    """
    Knowledge base for a specific expert
    """
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.index = None
        self.metadata = []
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None, doc_id: str = None):
        """
        Add a document to the expert's knowledge base
        """
        doc_id = doc_id or f"doc_{len(self.documents)}"
        
        # Store document
        self.documents.append({
            'id': doc_id,
            'content': content,
            'metadata': metadata or {}
        })
        
        # Create embedding for the document
        embedding = self._encode_document(content)
        
        # Add to FAISS index
        self._add_to_index(embedding, len(self.documents) - 1)
    
    def _encode_document(self, content: str) -> np.ndarray:
        """
        Encode document content using embedding model
        """
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return embedder.encode([content])[0]
    
    def _add_to_index(self, embedding: np.ndarray, doc_idx: int):
        """
        Add embedding to FAISS index
        """
        embedding = embedding.astype('float32')
        faiss.normalize_L2(embedding.reshape(1, -1))
        
        if self.index is None:
            dimension = embedding.shape[0]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        self.index.add(embedding.reshape(1, -1))
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for the query
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Encode query
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = embedder.encode([query])[0]
        
        # Normalize query
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        # Search
        scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['similarity'] = float(score)
                results.append(doc)
        
        return results

class ExpertPerformanceTracker:
    """
    Track performance of individual experts
    """
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.expert_usage = np.zeros(num_experts)
        self.expert_performance = np.ones(num_experts)  # Initialize to 1.0 (perfect performance)
        self.expert_processing_times = [[] for _ in range(num_experts)]
        self.total_queries = 0
    
    def update_query_performance(self, expert_ids: List[int], processing_time: float):
        """
        Update performance metrics for experts involved in a query
        """
        for expert_id in expert_ids:
            if 0 <= expert_id < self.num_experts:
                self.expert_usage[expert_id] += 1
                self.expert_processing_times[expert_id].append(processing_time)
        
        self.total_queries += 1
    
    def get_expert_efficiency(self) -> Dict[int, float]:
        """
        Get efficiency metrics for each expert
        """
        efficiency_scores = {}
        
        for i in range(self.num_experts):
            usage = self.expert_usage[i]
            if usage > 0 and len(self.expert_processing_times[i]) > 0:
                avg_processing_time = np.mean(self.expert_processing_times[i])
                # Efficiency is inversely proportional to processing time
                efficiency = 1.0 / (avg_processing_time + 0.001)  # Add small value to avoid division by zero
                efficiency_scores[i] = efficiency
            else:
                efficiency_scores[i] = 0.0  # No usage yet
        
        return efficiency_scores
    
    def get_expert_utilization(self) -> Dict[int, float]:
        """
        Get utilization metrics for each expert
        """
        if self.total_queries == 0:
            return {i: 0.0 for i in range(self.num_experts)}
        
        utilization_scores = {}
        for i in range(self.num_experts):
            utilization = self.expert_usage[i] / self.total_queries
            utilization_scores[i] = utilization
        
        return utilization_scores
```

### 2.2 Dynamic Expert Routing System
```python
class DynamicExpertRouter:
    """
    Dynamic routing system for MoE-RAG
    """
    def __init__(self, num_experts: int, embedding_dim: int = 384):
        self.num_experts = num_experts
        self.embedding_dim = embedding_dim
        
        # Initialize routing components
        self.query_analyzer = QueryAnalyzer()
        self.expert_router = ExpertRouter(num_experts, embedding_dim)
        self.performance_monitor = ExpertPerformanceMonitor()
        self.adaptive_scheduler = AdaptiveScheduler()
    
    def route_query_dynamically(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Dynamically route query based on real-time analysis
        """
        # Analyze query
        query_analysis = self.query_analyzer.analyze_query(query, context)
        
        # Get current expert statuses
        expert_statuses = self.performance_monitor.get_expert_statuses()
        
        # Determine optimal experts based on analysis and current status
        routing_decision = self.expert_router.decide_routing(
            query_analysis, expert_statuses
        )
        
        # Schedule processing with load balancing
        scheduled_experts = self.adaptive_scheduler.schedule_experts(
            routing_decision['selected_experts'],
            expert_statuses
        )
        
        return {
            'query_analysis': query_analysis,
            'routing_decision': routing_decision,
            'scheduled_experts': scheduled_experts,
            'confidence': routing_decision['confidence']
        }

class QueryAnalyzer:
    """
    Analyze queries to determine appropriate expert routing
    """
    def __init__(self):
        self.domain_classifiers = self._initialize_domain_classifiers()
        self.task_identifiers = self._initialize_task_identifiers()
        self.complexity_estimator = ComplexityEstimator()
    
    def _initialize_domain_classifiers(self):
        """
        Initialize domain classification models
        """
        # In practice, these would be trained models
        # For this example, we'll use keyword-based classification
        return {
            'nlp': ['language', 'text', 'translation', 'summarization', 'writing', 'grammar', 'linguistics'],
            'code': ['code', 'programming', 'software', 'development', 'algorithm', 'function', 'class', 'variable'],
            'science': ['research', 'study', 'experiment', 'hypothesis', 'scientific', 'physics', 'chemistry', 'biology'],
            'finance': ['money', 'investment', 'stock', 'market', 'economic', 'financial', 'banking', 'trading'],
            'legal': ['law', 'legal', 'court', 'contract', 'agreement', 'regulation', 'compliance', 'rights'],
            'medical': ['health', 'medical', 'patient', 'treatment', 'diagnosis', 'symptom', 'disease', 'therapy']
        }
    
    def _initialize_task_identifiers(self):
        """
        Initialize task identification models
        """
        return {
            'question_answering': ['what', 'how', 'why', 'when', 'where', 'who', 'explain', 'describe'],
            'summarization': ['summarize', 'summary', 'brief', 'concise', 'overview', 'outline'],
            'generation': ['write', 'create', 'generate', 'compose', 'draft', 'develop'],
            'analysis': ['analyze', 'evaluate', 'assess', 'examine', 'review', 'study'],
            'classification': ['classify', 'categorize', 'identify', 'determine', 'label', 'sort']
        }
    
    def analyze_query(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Analyze query to determine routing requirements
        """
        query_lower = query.lower()
        context_lower = context.lower()
        
        # Identify domain
        domain_scores = {}
        for domain, keywords in self.domain_classifiers.items():
            score = sum(1 for keyword in keywords if keyword in query_lower or keyword in context_lower)
            domain_scores[domain] = score
        
        # Identify task type
        task_scores = {}
        for task, keywords in self.task_identifiers.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            task_scores[task] = score
        
        # Estimate complexity
        complexity = self.complexity_estimator.estimate_complexity(query, context)
        
        # Determine primary domain and task
        primary_domain = max(domain_scores, key=domain_scores.get) if domain_scores else 'general'
        primary_task = max(task_scores, key=task_scores.get) if task_scores else 'general'
        
        return {
            'primary_domain': primary_domain,
            'primary_task': primary_task,
            'domain_scores': domain_scores,
            'task_scores': task_scores,
            'complexity_estimate': complexity,
            'query_length': len(query.split()),
            'contains_context': bool(context)
        }

class ExpertRouter:
    """
    Route queries to appropriate experts based on analysis
    """
    def __init__(self, num_experts: int, embedding_dim: int):
        self.num_experts = num_experts
        self.embedding_dim = embedding_dim
        
        # Expert specializations mapping
        self.expert_specializations = {
            0: {'domains': ['nlp', 'language'], 'tasks': ['qa', 'generation']},
            1: {'domains': ['code', 'programming'], 'tasks': ['generation', 'analysis']},
            2: {'domains': ['science', 'research'], 'tasks': ['analysis', 'summarization']},
            3: {'domains': ['finance', 'business'], 'tasks': ['analysis', 'qa']},
            4: {'domains': ['legal', 'compliance'], 'tasks': ['qa', 'analysis']},
            5: {'domains': ['medical', 'health'], 'tasks': ['qa', 'analysis']},
            6: {'domains': ['technical', 'support'], 'tasks': ['qa', 'troubleshooting']},
            7: {'domains': ['general', 'miscellaneous'], 'tasks': ['general']}
        }
    
    def decide_routing(self, query_analysis: Dict[str, Any], 
                      expert_statuses: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Decide which experts to route the query to
        """
        primary_domain = query_analysis['primary_domain']
        primary_task = query_analysis['primary_task']
        complexity = query_analysis['complexity_estimate']
        
        # Find experts that match the query requirements
        candidate_experts = []
        for expert_id, spec in self.expert_specializations.items():
            # Check if expert handles the domain
            domain_match = primary_domain in spec['domains'] or 'general' in spec['domains']
            
            # Check if expert handles the task
            task_match = primary_task in spec['tasks'] or 'general' in spec['tasks']
            
            if domain_match and task_match:
                # Calculate match score based on various factors
                match_score = self._calculate_match_score(
                    expert_id, query_analysis, expert_statuses
                )
                candidate_experts.append((expert_id, match_score))
        
        # Sort by match score and select top experts
        candidate_experts.sort(key=lambda x: x[1], reverse=True)
        
        # Select top experts based on complexity and availability
        top_experts = []
        total_capacity = 0
        max_experts = min(3, len(candidate_experts))  # Limit to 3 experts for efficiency
        
        for expert_id, score in candidate_experts[:max_experts]:
            if expert_statuses[expert_id]['available']:
                top_experts.append(expert_id)
                total_capacity += expert_statuses[expert_id]['capacity']
                
                # Stop if we have enough capacity for the query complexity
                if total_capacity >= complexity * 100:  # Arbitrary scaling factor
                    break
        
        # Calculate routing weights based on match scores
        scores = [score for _, score in candidate_experts[:len(top_experts)]]
        weights = self._normalize_scores(scores)
        
        # Calculate confidence based on match quality and expert availability
        avg_match_score = np.mean(scores) if scores else 0.0
        expert_availability = len([e for e in top_experts if expert_statuses[e]['available']]) / len(top_experts) if top_experts else 0.0
        
        confidence = 0.7 * avg_match_score + 0.3 * expert_availability
        
        return {
            'selected_experts': top_experts,
            'routing_weights': weights,
            'confidence': confidence,
            'reasoning': f"Matched {primary_domain} domain and {primary_task} task with {len(top_experts)} experts"
        }
    
    def _calculate_match_score(self, expert_id: int, query_analysis: Dict[str, Any], 
                              expert_statuses: Dict[int, Dict[str, Any]]) -> float:
        """
        Calculate match score for an expert
        """
        # Base score from domain and task match
        spec = self.expert_specializations[expert_id]
        domain_match = query_analysis['primary_domain'] in spec['domains']
        task_match = query_analysis['primary_task'] in spec['tasks']
        
        base_score = 0.5 if domain_match else 0.0
        base_score += 0.3 if task_match else 0.0
        
        # Performance-based adjustment
        performance_score = expert_statuses[expert_id].get('performance', 0.8)
        base_score *= performance_score
        
        # Load-based adjustment
        current_load = expert_statuses[expert_id].get('current_load', 0)
        max_load = expert_statuses[expert_id].get('max_load', 100)
        load_factor = 1.0 - (current_load / max_load)  # Lower load = higher score
        base_score *= max(0.5, load_factor)  # Don't penalize too heavily
        
        return min(1.0, base_score)
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to sum to 1.0
        """
        if not scores:
            return []
        
        total = sum(scores)
        if total == 0:
            return [1.0 / len(scores)] * len(scores)
        
        return [score / total for score in scores]

class ExpertPerformanceMonitor:
    """
    Monitor and track expert performance
    """
    def __init__(self):
        self.expert_metrics = {}
        self.performance_history = {}
        self.anomaly_detectors = {}
    
    def update_expert_metrics(self, expert_id: int, metrics: Dict[str, float]):
        """
        Update performance metrics for an expert
        """
        if expert_id not in self.expert_metrics:
            self.expert_metrics[expert_id] = {
                'response_time_avg': [],
                'accuracy_history': [],
                'throughput_history': [],
                'error_rate_history': []
            }
        
        # Update metrics
        self.expert_metrics[expert_id]['response_time_avg'].append(metrics.get('response_time', 0))
        self.expert_metrics[expert_id]['accuracy_history'].append(metrics.get('accuracy', 0.8))
        self.expert_metrics[expert_id]['throughput_history'].append(metrics.get('throughput', 1.0))
        self.expert_metrics[expert_id]['error_rate_history'].append(metrics.get('error_rate', 0.05))
        
        # Keep only recent history (last 100 entries)
        for key in self.expert_metrics[expert_id]:
            if len(self.expert_metrics[expert_id][key]) > 100:
                self.expert_metrics[expert_id][key] = self.expert_metrics[expert_id][key][-100:]
    
    def get_expert_statuses(self) -> Dict[int, Dict[str, Any]]:
        """
        Get current status of all experts
        """
        statuses = {}
        
        for expert_id in self.expert_metrics:
            metrics = self.expert_metrics[expert_id]
            
            # Calculate current performance indicators
            avg_response_time = np.mean(metrics['response_time_avg']) if metrics['response_time_avg'] else 0.5
            avg_accuracy = np.mean(metrics['accuracy_history']) if metrics['accuracy_history'] else 0.8
            current_throughput = metrics['throughput_history'][-1] if metrics['throughput_history'] else 1.0
            current_error_rate = metrics['error_rate_history'][-1] if metrics['error_rate_history'] else 0.05
            
            # Determine if expert is available based on performance
            available = (
                avg_response_time < 2.0 and  # Response time threshold
                avg_accuracy > 0.7 and       # Accuracy threshold
                current_error_rate < 0.2     # Error rate threshold
            )
            
            statuses[expert_id] = {
                'available': available,
                'performance': avg_accuracy,
                'response_time': avg_response_time,
                'throughput': current_throughput,
                'error_rate': current_error_rate,
                'capacity': 100,  # Simplified capacity model
                'current_load': len(metrics['response_time_avg'])  # Simplified load calculation
            }
        
        return statuses
    
    def detect_performance_anomalies(self) -> Dict[int, List[str]]:
        """
        Detect performance anomalies in experts
        """
        anomalies = {}
        
        for expert_id, metrics in self.expert_metrics.items():
            expert_anomalies = []
            
            # Check for response time anomalies
            if metrics['response_time_avg']:
                recent_avg = np.mean(metrics['response_time_avg'][-10:])  # Last 10 measurements
                historical_avg = np.mean(metrics['response_time_avg'])
                
                if recent_avg > historical_avg * 2:  # Response time doubled
                    expert_anomalies.append(f"Response time increased by {(recent_avg/historical_avg):.2f}x")
            
            # Check for accuracy degradation
            if metrics['accuracy_history'] and len(metrics['accuracy_history']) >= 10:
                recent_accuracy = np.mean(metrics['accuracy_history'][-10:])
                historical_accuracy = np.mean(metrics['accuracy_history'])
                
                if recent_accuracy < historical_accuracy - 0.1:  # 10% degradation
                    expert_anomalies.append(f"Accuracy dropped by {(historical_accuracy - recent_accuracy):.2f}")
            
            # Check for error rate spikes
            if metrics['error_rate_history'] and len(metrics['error_rate_history']) >= 10:
                recent_errors = np.mean(metrics['error_rate_history'][-10:])
                historical_errors = np.mean(metrics['error_rate_history'])
                
                if recent_errors > historical_errors * 2:  # Error rate doubled
                    expert_anomalies.append(f"Error rate increased by {(recent_errors/historical_errors):.2f}x")
            
            if expert_anomalies:
                anomalies[expert_id] = expert_anomalies
        
        return anomalies

class AdaptiveScheduler:
    """
    Adaptive scheduler for expert processing
    """
    def __init__(self):
        self.expert_queues = {}
        self.processing_times = {}
        self.priority_rules = {}
    
    def schedule_experts(self, candidate_experts: List[int], 
                        expert_statuses: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Schedule experts based on current status and priorities
        """
        scheduled_experts = []
        
        # Sort experts by availability and performance
        available_experts = [
            (eid, estatus) for eid, estatus in expert_statuses.items() 
            if eid in candidate_experts and estatus['available']
        ]
        
        # Sort by performance (descending) and load (ascending)
        available_experts.sort(
            key=lambda x: (x[1]['performance'], -x[1]['current_load']), 
            reverse=True
        )
        
        # Schedule top experts
        for expert_id, status in available_experts[:len(candidate_experts)]:
            schedule_info = {
                'expert_id': expert_id,
                'estimated_wait_time': self._estimate_wait_time(expert_id),
                'priority': self._calculate_priority(expert_id, status),
                'allocation_time': time.time()
            }
            scheduled_experts.append(schedule_info)
        
        return scheduled_experts
    
    def _estimate_wait_time(self, expert_id: int) -> float:
        """
        Estimate wait time for an expert
        """
        if expert_id not in self.expert_queues:
            return 0.0
        
        queue_length = len(self.expert_queues[expert_id])
        avg_processing_time = self.processing_times.get(expert_id, 0.5)
        
        return queue_length * avg_processing_time
    
    def _calculate_priority(self, expert_id: int, status: Dict[str, Any]) -> int:
        """
        Calculate priority for expert allocation
        """
        # Higher performance = higher priority
        # Lower load = higher priority
        performance_factor = status['performance'] * 100
        load_factor = (1 - status['current_load'] / status['capacity']) * 100
        
        return int(performance_factor * 0.7 + load_factor * 0.3)
```

### 2.3 Expert Coordination System
```python
class ExpertCoordinationSystem:
    """
    System for coordinating multiple experts in RAG
    """
    def __init__(self, num_experts: int, hidden_dim: int):
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        
        # Expert quality estimators
        self.expert_quality_estimators = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_experts)
        ])
        
        # Expert interaction network
        self.expert_interaction = nn.Linear(hidden_dim * num_experts, hidden_dim)
        
        # Confidence calibration
        self.confidence_calibrator = nn.Linear(hidden_dim, 1)
        
        # Coordination history
        self.coordination_history = []
    
    def coordinate_experts(self, expert_outputs: List[torch.Tensor], 
                          query_embedding: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Coordinate multiple experts to produce final output
        """
        # Estimate quality of each expert's output
        expert_qualities = []
        for i, output in enumerate(expert_outputs):
            # Combine expert output with query
            combined = torch.cat([output, query_embedding], dim=0)
            quality_score = torch.sigmoid(self.expert_quality_estimators[i](combined))
            expert_qualities.append(quality_score.item())
        
        # Normalize qualities to use as weights
        total_quality = sum(expert_qualities)
        if total_quality > 0:
            weights = [q / total_quality for q in expert_qualities]
        else:
            weights = [1.0 / len(expert_outputs)] * len(expert_outputs)
        
        # Weighted combination of expert outputs
        weighted_output = torch.zeros_like(expert_outputs[0])
        for i, (output, weight) in enumerate(zip(expert_outputs, weights)):
            weighted_output += weight * output
        
        # Apply expert interaction
        all_outputs = torch.cat(expert_outputs, dim=0)
        interaction_output = self.expert_interaction(all_outputs)
        
        # Combine weighted output with interaction
        final_output = weighted_output + 0.1 * interaction_output  # Small interaction weight
        
        # Calibrate confidence
        confidence = torch.sigmoid(self.confidence_calibrator(final_output))
        
        # Record coordination
        self.coordination_history.append({
            'timestamp': time.time(),
            'expert_qualities': expert_qualities,
            'coordination_weights': weights,
            'confidence': confidence.item()
        })
        
        return final_output, confidence.item()
    
    def get_coordination_insights(self) -> Dict[str, Any]:
        """
        Get insights about expert coordination
        """
        if not self.coordination_history:
            return {'message': 'No coordination history available'}
        
        # Calculate average coordination metrics
        avg_qualities = []
        avg_weights = []
        
        for record in self.coordination_history[-20:]:  # Last 20 coordinations
            avg_qualities.append(record['expert_qualities'])
            avg_weights.append(record['coordination_weights'])
        
        if not avg_qualities:
            return {'message': 'Insufficient coordination history'}
        
        # Calculate average quality per expert
        num_experts = len(avg_qualities[0]) if avg_qualities else 0
        if num_experts == 0:
            return {'message': 'No expert data in history'}
        
        avg_expert_qualities = np.mean(avg_qualities, axis=0)
        avg_expert_weights = np.mean(avg_weights, axis=0)
        
        return {
            'average_expert_qualities': avg_expert_qualities.tolist(),
            'average_coordination_weights': avg_expert_weights.tolist(),
            'total_coordination_events': len(self.coordination_history),
            'recent_coordination_trends': self._analyze_coordination_trends(),
            'expert_utilization_balance': self._calculate_utilization_balance(avg_expert_weights)
        }
    
    def _analyze_coordination_trends(self) -> Dict[str, Any]:
        """
        Analyze trends in expert coordination
        """
        if len(self.coordination_history) < 10:
            return {'message': 'Insufficient data for trend analysis'}
        
        # Analyze confidence trends
        recent_confidences = [record['confidence'] for record in self.coordination_history[-10:]]
        older_confidences = [record['confidence'] for record in self.coordination_history[:10]]
        
        recent_avg_conf = np.mean(recent_confidences)
        older_avg_conf = np.mean(older_confidences)
        
        confidence_trend = 'improving' if recent_avg_conf > older_avg_conf + 0.05 else \
                          'declining' if recent_avg_conf < older_avg_conf - 0.05 else 'stable'
        
        return {
            'confidence_trend': confidence_trend,
            'recent_avg_confidence': float(recent_avg_conf),
            'older_avg_confidence': float(older_avg_conf),
            'confidence_volatility': float(np.std(recent_confidences))
        }
    
    def _calculate_utilization_balance(self, weights: np.ndarray) -> float:
        """
        Calculate how balanced the expert utilization is
        """
        # Calculate entropy of weight distribution (higher entropy = more balanced)
        normalized_weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
        
        entropy = -np.sum(normalized_weights * np.log(normalized_weights + 1e-8))
        max_entropy = np.log(len(weights))  # Maximum possible entropy
        
        balance_score = entropy / max_entropy if max_entropy > 0 else 1.0
        
        return float(balance_score)

class ComplexityEstimator:
    """
    Estimate complexity of queries for expert routing
    """
    def __init__(self):
        self.complexity_factors = {
            'length_factor': 0.3,
            'domain_specificity': 0.4,
            'multi_hop_reasoning': 0.3
        }
    
    def estimate_complexity(self, query: str, context: str = "") -> float:
        """
        Estimate complexity of a query (0-1 scale, higher is more complex)
        """
        # Length-based complexity (longer queries are more complex)
        length_factor = min(1.0, len(query.split()) / 100)  # Normalize by 100 words
        
        # Domain specificity (queries with specific domain terms are more complex)
        domain_indicators = [
            'specifically', 'particularly', 'especially', 'especially for', 'in the context of',
            'according to', 'based on', 'given that', 'provided that'
        ]
        domain_specificity = sum(1 for indicator in domain_indicators if indicator in query.lower()) / len(domain_indicators)
        
        # Multi-hop reasoning indicators
        reasoning_indicators = [
            'compare', 'contrast', 'analyze', 'evaluate', 'assess', 'examine',
            'how does X relate to Y', 'what is the connection between', 'relationship between',
            'causes of', 'effects of', 'implications of'
        ]
        multi_hop_reasoning = sum(1 for indicator in reasoning_indicators if indicator in query.lower()) / len(reasoning_indicators)
        
        # Combine factors
        complexity = (
            self.complexity_factors['length_factor'] * length_factor +
            self.complexity_factors['domain_specificity'] * domain_specificity +
            self.complexity_factors['multi_hop_reasoning'] * multi_hop_reasoning
        )
        
        return min(1.0, complexity)  # Cap at 1.0

class LoadBalancer:
    """
    Balance load across experts to prevent overutilization
    """
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.expert_loads = np.zeros(num_experts)
        self.expert_success_rates = np.ones(num_experts)  # Initialize to 100% success
        self.request_history = []
    
    def select_experts(self, routing_weights: torch.Tensor, 
                      selected_experts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adjust expert selection based on current load and success rates
        """
        # Convert to numpy for manipulation
        weights_np = routing_weights.cpu().numpy()
        experts_np = selected_experts.cpu().numpy()
        
        # Apply load balancing adjustment
        load_adjusted_weights = weights_np.copy()
        
        for i, expert_id in enumerate(experts_np[0]):  # First batch
            if 0 <= expert_id < self.num_experts:
                # Reduce weight for overloaded experts
                current_load = self.expert_loads[expert_id]
                max_load = 100  # Threshold for load balancing
                
                if current_load > max_load * 0.8:  # 80% threshold
                    load_factor = 1.0 - min(0.5, (current_load - max_load * 0.8) / (max_load * 0.2))
                    load_adjusted_weights[0, expert_id] *= load_factor
        
        # Renormalize weights
        total_weight = np.sum(load_adjusted_weights)
        if total_weight > 0:
            load_adjusted_weights = load_adjusted_weights / total_weight
        
        # Update load tracking
        for expert_id in experts_np[0]:
            if 0 <= expert_id < self.num_experts:
                self.expert_loads[expert_id] += 1
        
        return torch.FloatTensor(load_adjusted_weights), selected_experts
    
    def update_success_rate(self, expert_ids: List[int], success: bool):
        """
        Update success rate for experts
        """
        for expert_id in expert_ids:
            if 0 <= expert_id < self.num_experts:
                # Use exponential moving average
                alpha = 0.1
                current_rate = self.expert_success_rates[expert_id]
                new_rate = alpha * float(success) + (1 - alpha) * current_rate
                self.expert_success_rates[expert_id] = new_rate
    
    def get_load_balancing_report(self) -> Dict[str, Any]:
        """
        Get report on load balancing effectiveness
        """
        max_load = np.max(self.expert_loads)
        min_load = np.min(self.expert_loads)
        avg_load = np.mean(self.expert_loads)
        
        load_balance_ratio = min_load / max_load if max_load > 0 else 1.0
        
        return {
            'max_expert_load': int(max_load),
            'min_expert_load': int(min_load),
            'avg_expert_load': float(avg_load),
            'load_balance_ratio': float(load_balance_ratio),
            'expert_success_rates': self.expert_success_rates.tolist(),
            'total_requests_handled': int(np.sum(self.expert_loads))
        }
```

### 2.4 Multi-Expert RAG System Integration
```python
class MultiExpertRAGSystem:
    """
    Complete Multi-Expert RAG system
    """
    def __init__(self, num_experts: int = 8, expert_capacity: int = 1024):
        self.multi_expert_core = MultiExpertRAGCore(num_experts, expert_capacity)
        self.dynamic_router = DynamicExpertRouter(num_experts)
        self.coordination_system = ExpertCoordinationSystem(num_experts, 256)
        self.load_balancer = LoadBalancer(num_experts)
        self.performance_tracker = ExpertPerformanceTracker(num_experts)
        self.evaluation_framework = MultiExpertEvaluationFramework()
        
    def process_query(self, query: str, context: str = "", top_k: int = 3) -> Dict[str, Any]:
        """
        Process query through multi-expert system
        """
        start_time = time.time()
        
        # Analyze query
        query_analysis = self.dynamic_router.query_analyzer.analyze_query(query, context)
        
        # Route query dynamically
        routing_result = self.dynamic_router.route_query_dynamically(query, context)
        
        # Apply load balancing
        balanced_weights, balanced_experts = self.load_balancer.select_experts(
            torch.FloatTensor(routing_result['routing_decision']['routing_weights']).unsqueeze(0),
            torch.LongTensor(routing_result['routing_decision']['selected_experts']).unsqueeze(0)
        )
        
        # Process with selected experts
        expert_outputs = {}
        for expert_id in balanced_experts[0]:
            if expert_id < self.multi_expert_core.num_experts:
                # Retrieve relevant information for this expert
                knowledge_base = self.multi_expert_core.expert_knowledge_bases[expert_id.item()]
                retrieved_docs = knowledge_base.retrieve(query, top_k)
                
                # Process with expert
                expert_output = self.multi_expert_core._process_with_expert(
                    query, retrieved_docs, expert_id.item()
                )
                expert_outputs[expert_id.item()] = expert_output
        
        # Coordinate expert outputs
        expert_tensors = [torch.randn(256) for _ in expert_outputs.values()]  # Placeholder tensors
        query_embedding = self.multi_expert_core.embedding_model.encode([query])[0]
        coordinated_output, confidence = self.coordination_system.coordinate_experts(
            expert_tensors, torch.FloatTensor(query_embedding)
        )
        
        # Combine outputs using the core combiner
        combined_output = self.multi_expert_core.expert_combiner.combine_outputs(
            expert_outputs, balanced_weights, balanced_experts
        )
        
        end_time = time.time()
        
        result = {
            'response': combined_output,
            'coordinated_output': coordinated_output.tolist(),
            'confidence': confidence,
            'query_analysis': query_analysis,
            'routing_result': routing_result,
            'expert_outputs': expert_outputs,
            'selected_experts': balanced_experts[0].tolist(),
            'processing_time_ms': (end_time - start_time) * 1000,
            'coordination_insights': self.coordination_system.get_coordination_insights()
        }
        
        # Update performance tracking
        self.performance_tracker.update_query_performance(
            balanced_experts[0].tolist(), 
            end_time - start_time
        )
        
        return result
    
    def add_document_to_expert(self, expert_id: int, content: str, 
                             metadata: Dict[str, Any] = None):
        """
        Add document to specific expert's knowledge base
        """
        if expert_id < self.multi_expert_core.num_experts:
            knowledge_base = self.multi_expert_core.expert_knowledge_bases[expert_id]
            knowledge_base.add_document(content, metadata)
    
    def evaluate_system(self, test_queries: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Evaluate the multi-expert system
        """
        return self.evaluation_framework.evaluate_system(self, test_queries)
    
    def get_expert_utilization_report(self) -> Dict[str, Any]:
        """
        Get expert utilization and performance report
        """
        expert_efficiency = self.performance_tracker.get_expert_efficiency()
        expert_utilization = self.performance_tracker.get_expert_utilization()
        load_balance_report = self.load_balancer.get_load_balancing_report()
        coordination_insights = self.coordination_system.get_coordination_insights()
        
        return {
            'expert_efficiency': expert_efficiency,
            'expert_utilization': expert_utilization,
            'load_balance_report': load_balance_report,
            'coordination_insights': coordination_insights,
            'system_health': self._calculate_system_health(expert_efficiency, expert_utilization)
        }
    
    def _calculate_system_health(self, expert_efficiency: Dict[int, float], 
                               expert_utilization: Dict[int, float]) -> str:
        """
        Calculate overall system health
        """
        avg_efficiency = np.mean(list(expert_efficiency.values())) if expert_efficiency else 0.0
        avg_utilization = np.mean(list(expert_utilization.values())) if expert_utilization else 0.0
        
        if avg_efficiency > 0.8 and avg_utilization > 0.6:
            return 'optimal'
        elif avg_efficiency > 0.6 and avg_utilization > 0.4:
            return 'good'
        elif avg_efficiency > 0.4 or avg_utilization > 0.2:
            return 'fair'
        else:
            return 'needs_attention'

class MultiExpertEvaluationFramework:
    """
    Evaluation framework for multi-expert RAG systems
    """
    def __init__(self):
        self.metrics = [
            'expert_utilization_balance',
            'routing_accuracy',
            'computational_efficiency',
            'response_quality',
            'specialization_effectiveness',
            'load_balancing_score'
        ]
    
    def evaluate_system(self, system: MultiExpertRAGSystem, 
                       test_queries: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Evaluate the multi-expert RAG system
        """
        results = {
            'individual_query_results': [],
            'aggregate_metrics': {},
            'expert_analysis': {},
            'routing_analysis': {}
        }
        
        expert_utilization = {i: 0 for i in range(system.multi_expert_core.num_experts)}
        routing_accuracies = []
        computational_costs = []
        response_qualities = []
        
        for query_data in test_queries:
            query = query_data['query']
            expected_domain = query_data.get('domain', 'general')
            
            # Process query
            start_time = time.time()
            result = system.process_query(query)
            end_time = time.time()
            
            # Update expert utilization
            for expert_id in result['selected_experts']:
                expert_utilization[expert_id] += 1
            
            # Calculate routing accuracy if domain is known
            if expected_domain:
                correct_expert_assigned = any(
                    expected_domain in system.multi_expert_core.expert_specializations[exp_id] 
                    for exp_id in result['selected_experts']
                )
                routing_accuracies.append(1.0 if correct_expert_assigned else 0.0)
            
            # Calculate computational efficiency (simplified)
            computational_cost = (end_time - start_time) * len(result['selected_experts'])
            computational_costs.append(computational_cost)
            
            # Calculate response quality (simplified - in practice would use more sophisticated metrics)
            response_quality = self._calculate_response_quality(
                result['response'], query_data.get('expected_response', '')
            )
            response_qualities.append(response_quality)
            
            # Store individual result
            results['individual_query_results'].append({
                'query': query,
                'expected_domain': expected_domain,
                'selected_experts': result['selected_experts'],
                'processing_time': end_time - start_time,
                'response_quality': response_quality,
                'routing_correct': correct_expert_assigned if expected_domain else None
            })
        
        # Calculate aggregate metrics
        results['aggregate_metrics'] = {
            'expert_utilization_balance': self._calculate_utilization_balance(expert_utilization),
            'average_routing_accuracy': np.mean(routing_accuracies) if routing_accuracies else 0.0,
            'average_computational_efficiency': np.mean(computational_costs) if computational_costs else float('inf'),
            'average_response_quality': np.mean(response_qualities) if response_qualities else 0.0,
            'total_queries_processed': len(test_queries),
            'queries_per_second': len(test_queries) / sum(r.time for r in results['individual_query_results'])
        }
        
        # Expert analysis
        results['expert_analysis'] = {
            'utilization_counts': expert_utilization,
            'utilization_percentages': {k: v/len(test_queries) for k, v in expert_utilization.items()},
            'specialization_effectiveness': self._analyze_specialization_effectiveness(system, test_queries)
        }
        
        # Routing analysis
        results['routing_analysis'] = {
            'routing_distribution': self._analyze_routing_distribution(results['individual_query_results']),
            'load_balancing_score': self._calculate_load_balancing_score(expert_utilization)
        }
        
        return results
    
    def _calculate_utilization_balance(self, expert_utilization: Dict[int, int]) -> float:
        """
        Calculate how balanced the expert utilization is (0-1, where 1 is perfectly balanced)
        """
        if not expert_utilization:
            return 0.0
        
        counts = list(expert_utilization.values())
        if len(set(counts)) == 1:
            # All experts equally utilized
            return 1.0
        
        # Calculate coefficient of variation (lower is more balanced)
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        if mean_count == 0:
            return 1.0  # All zeros means equal (though not ideal)
        
        # Coefficient of variation: std/mean
        cv = std_count / mean_count
        
        # Convert to balance score (higher is better)
        balance_score = max(0.0, 1.0 - cv)
        
        return balance_score
    
    def _calculate_response_quality(self, generated: str, expected: str) -> float:
        """
        Calculate response quality (simplified implementation)
        """
        if not expected:
            # If no expected response, return neutral score
            return 0.5
        
        # Use embedding similarity as a proxy for quality
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        gen_embedding = embedder.encode([generated])[0]
        exp_embedding = embedder.encode([expected])[0]
        
        # Calculate cosine similarity
        similarity = np.dot(gen_embedding, exp_embedding) / (
            np.linalg.norm(gen_embedding) * np.linalg.norm(exp_embedding)
        )
        
        return float(similarity)
    
    def _analyze_specialization_effectiveness(self, system: MultiExpertRAGSystem, 
                                            test_queries: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Analyze how effectively experts specialize in their domains
        """
        domain_expert_mapping = {}
        
        for query_data in test_queries:
            query = query_data['query']
            expected_domain = query_data.get('domain', 'general')
            
            # Get routing decision
            query_analysis = system.dynamic_router.query_analyzer.analyze_query(query)
            routing_result = system.dynamic_router.route_query_dynamically(query)
            
            selected_experts = routing_result['routing_decision']['selected_experts']
            
            if expected_domain not in domain_expert_mapping:
                domain_expert_mapping[expected_domain] = {}
            
            for expert_id in selected_experts:
                if expert_id not in domain_expert_mapping[expected_domain]:
                    domain_expert_mapping[expected_domain][expert_id] = 0
                domain_expert_mapping[expected_domain][expert_id] += 1
        
        # Calculate specialization scores
        specialization_scores = {}
        for domain, expert_counts in domain_expert_mapping.items():
            total_assignments = sum(expert_counts.values())
            if total_assignments == 0:
                continue
            
            # Find the expert most associated with this domain
            most_frequent_expert = max(expert_counts.items(), key=lambda x: x[1])
            expert_id, count = most_frequent_expert
            specialization_score = count / total_assignments
            
            specialization_scores[domain] = {
                'primary_expert': expert_id,
                'specialization_score': specialization_score,
                'assignment_distribution': {k: v/total_assignments for k, v in expert_counts.items()}
            }
        
        return specialization_scores
    
    def _analyze_routing_distribution(self, query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the distribution of routing decisions
        """
        total_queries = len(query_results)
        expert_selection_counts = {}
        routing_patterns = []
        
        for result in query_results:
            for expert_id in result['selected_experts']:
                if expert_id not in expert_selection_counts:
                    expert_selection_counts[expert_id] = 0
                expert_selection_counts[expert_id] += 1
            
            # Record routing pattern (sorted expert IDs)
            pattern = tuple(sorted(result['selected_experts']))
            routing_patterns.append(pattern)
        
        # Analyze routing patterns
        pattern_frequency = {}
        for pattern in routing_patterns:
            if pattern not in pattern_frequency:
                pattern_frequency[pattern] = 0
            pattern_frequency[pattern] += 1
        
        return {
            'expert_selection_frequencies': {k: v/total_queries for k, v in expert_selection_counts.items()},
            'common_routing_patterns': pattern_frequency,
            'diversity_score': len(set(routing_patterns)) / total_queries if total_queries > 0 else 0
        }
    
    def _calculate_load_balancing_score(self, expert_utilization: Dict[int, int]) -> float:
        """
        Calculate load balancing score (0-1, where 1 is perfectly balanced)
        """
        if not expert_utilization:
            return 0.0
        
        counts = list(expert_utilization.values())
        if len(set(counts)) == 1:
            return 1.0  # Perfectly balanced
        
        # Calculate entropy of utilization distribution
        total_requests = sum(counts)
        if total_requests == 0:
            return 1.0  # No requests processed
        
        probabilities = [count / total_requests for count in counts]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        # Normalize by maximum possible entropy (uniform distribution)
        max_entropy = np.log2(len(counts)) if len(counts) > 0 else 1
        
        return entropy / max_entropy if max_entropy > 0 else 0.0

class ExpertSpecializationOptimizer:
    """
    Optimize expert specializations based on performance
    """
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.expert_domains = [[] for _ in range(num_experts)]
        self.expert_performance = np.zeros(num_experts)
        self.domain_expertise_matrix = np.zeros((num_experts, 10))  # 10 example domains
    
    def update_expert_specialization(self, query_analysis: Dict[str, Any], 
                                   expert_outputs: Dict[int, str],
                                   actual_performance: float):
        """
        Update expert specializations based on performance
        """
        primary_domain = query_analysis['primary_domain']
        
        # Map domain to index
        domain_map = {
            'nlp': 0, 'code': 1, 'science': 2, 'finance': 3, 'legal': 4,
            'medical': 5, 'technical': 6, 'general': 7
        }
        
        domain_idx = domain_map.get(primary_domain, 7)  # Default to general
        
        # Update performance for involved experts
        for expert_id in expert_outputs.keys():
            # Update domain expertise
            self.domain_expertise_matrix[expert_id, domain_idx] += 1
            
            # Update overall performance (moving average)
            self.expert_performance[expert_id] = (
                0.9 * self.expert_performance[expert_id] + 
                0.1 * actual_performance
            )
    
    def get_expert_for_domain(self, domain: str) -> int:
        """
        Get the best expert for a specific domain
        """
        domain_map = {
            'nlp': 0, 'code': 1, 'science': 2, 'finance': 3, 'legal': 4,
            'medical': 5, 'technical': 6, 'general': 7
        }
        
        domain_idx = domain_map.get(domain, 7)
        
        # Find expert with highest expertise in this domain
        expert_scores = []
        for i in range(self.num_experts):
            domain_expertise = self.domain_expertise_matrix[i, domain_idx]
            overall_performance = self.expert_performance[i]
            
            # Combine domain expertise with overall performance
            score = domain_expertise * 0.7 + overall_performance * 0.3
            expert_scores.append((i, score))
        
        # Return expert with highest score
        best_expert = max(expert_scores, key=lambda x: x[1])
        return best_expert[0]
    
    def rebalance_experts(self) -> Dict[int, List[str]]:
        """
        Rebalance expert specializations based on performance
        """
        rebalanced_domains = {}
        
        for expert_id in range(self.num_experts):
            # Find domains where this expert performs well
            domain_scores = []
            for domain_idx in range(self.domain_expertise_matrix.shape[1]):
                expertise = self.domain_expertise_matrix[expert_id, domain_idx]
                performance = self.expert_performance[expert_id]
                
                score = expertise * 0.6 + performance * 0.4
                domain_scores.append((domain_idx, score))
            
            # Get top domains for this expert
            domain_scores.sort(key=lambda x: x[1], reverse=True)
            top_domains = [self._domain_idx_to_name(idx) for idx, _ in domain_scores[:3]]
            
            rebalanced_domains[expert_id] = top_domains
        
        return rebalanced_domains
    
    def _domain_idx_to_name(self, idx: int) -> str:
        """
        Convert domain index to name
        """
        domain_names = ['nlp', 'code', 'science', 'finance', 'legal', 
                       'medical', 'technical', 'general', 'other1', 'other2']
        return domain_names[idx] if idx < len(domain_names) else 'general'
```

## 3. Performance and Evaluation

### 3.1 Multi-Expert Evaluation Metrics
```python
class MultiExpertEvaluationFramework:
    """
    Evaluation framework for multi-expert RAG systems
    """
    def __init__(self):
        self.metrics = [
            'expert_utilization_balance',
            'routing_accuracy',
            'computational_efficiency',
            'response_quality',
            'specialization_effectiveness',
            'load_balancing_score',
            'coordination_efficiency',
            'system_throughput'
        ]
    
    def evaluate_system(self, system: MultiExpertRAGSystem, 
                       test_queries: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Evaluate the multi-expert RAG system
        """
        results = {
            'individual_query_results': [],
            'aggregate_metrics': {},
            'expert_analysis': {},
            'routing_analysis': {},
            'coordination_analysis': {}
        }
        
        expert_utilization = {i: 0 for i in range(system.multi_expert_core.num_experts)}
        routing_accuracies = []
        computational_costs = []
        response_qualities = []
        coordination_scores = []
        
        for query_data in test_queries:
            query = query_data['query']
            expected_domain = query_data.get('domain', 'general')
            
            # Process query
            start_time = time.time()
            result = system.process_query(query)
            end_time = time.time()
            
            # Update expert utilization
            for expert_id in result['selected_experts']:
                expert_utilization[expert_id] += 1
            
            # Calculate routing accuracy if domain is known
            if expected_domain:
                correct_expert_assigned = any(
                    expected_domain in system.multi_expert_core.expert_specializations[exp_id] 
                    for exp_id in result['selected_experts']
                )
                routing_accuracies.append(1.0 if correct_expert_assigned else 0.0)
            
            # Calculate computational efficiency
            computational_cost = (end_time - start_time) * len(result['selected_experts'])
            computational_costs.append(computational_cost)
            
            # Calculate response quality
            response_quality = self._calculate_response_quality(
                result['response'], query_data.get('expected_response', '')
            )
            response_qualities.append(response_quality)
            
            # Calculate coordination efficiency
            coord_efficiency = self._calculate_coordination_efficiency(
                result['coordination_insights']
            )
            coordination_scores.append(coord_efficiency)
            
            # Store individual result
            results['individual_query_results'].append({
                'query': query,
                'expected_domain': expected_domain,
                'selected_experts': result['selected_experts'],
                'processing_time': end_time - start_time,
                'response_quality': response_quality,
                'routing_correct': correct_expert_assigned if expected_domain else None,
                'coordination_score': coord_efficiency
            })
        
        # Calculate aggregate metrics
        results['aggregate_metrics'] = {
            'expert_utilization_balance': self._calculate_utilization_balance(expert_utilization),
            'average_routing_accuracy': np.mean(routing_accuracies) if routing_accuracies else 0.0,
            'average_computational_efficiency': np.mean(computational_costs) if computational_costs else float('inf'),
            'average_response_quality': np.mean(response_qualities) if response_qualities else 0.0,
            'average_coordination_efficiency': np.mean(coordination_scores) if coordination_scores else 0.0,
            'total_queries_processed': len(test_queries),
            'queries_per_second': len(test_queries) / sum(r['processing_time'] for r in results['individual_query_results']) if results['individual_query_results'] else 0,
            'system_throughput': len(test_queries) / sum(r['processing_time'] for r in results['individual_query_results']) if results['individual_query_results'] else 0
        }
        
        # Expert analysis
        results['expert_analysis'] = {
            'utilization_counts': expert_utilization,
            'utilization_percentages': {k: v/len(test_queries) for k, v in expert_utilization.items()},
            'specialization_effectiveness': self._analyze_specialization_effectiveness(system, test_queries),
            'performance_by_expert': self._calculate_expert_performance(system, test_queries)
        }
        
        # Routing analysis
        results['routing_analysis'] = {
            'routing_distribution': self._analyze_routing_distribution(results['individual_query_results']),
            'load_balancing_score': self._calculate_load_balancing_score(expert_utilization),
            'routing_efficiency': self._calculate_routing_efficiency(results['individual_query_results'])
        }
        
        # Coordination analysis
        results['coordination_analysis'] = {
            'coordination_patterns': self._analyze_coordination_patterns(results['individual_query_results']),
            'expert_interaction_effectiveness': self._analyze_expert_interactions(system)
        }
        
        return results
    
    def _calculate_utilization_balance(self, expert_utilization: Dict[int, int]) -> float:
        """
        Calculate how balanced the expert utilization is (0-1, where 1 is perfectly balanced)
        """
        if not expert_utilization:
            return 0.0
        
        counts = list(expert_utilization.values())
        if len(set(counts)) == 1:
            # All experts equally utilized
            return 1.0
        
        # Calculate entropy of utilization distribution
        total_requests = sum(counts)
        if total_requests == 0:
            return 1.0  # No requests processed
        
        probabilities = [count / total_requests for count in counts]
        entropy = -sum(p * np.log2(p + 1e-8) for p in probabilities)  # Add small value to avoid log(0)
        
        # Normalize by maximum possible entropy (uniform distribution)
        max_entropy = np.log2(len(counts)) if len(counts) > 0 else 1
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_response_quality(self, generated: str, expected: str) -> float:
        """
        Calculate response quality using embedding similarity
        """
        if not expected:
            return 0.5  # Neutral score if no expected response
        
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        gen_embedding = embedder.encode([generated])[0]
        exp_embedding = embedder.encode([expected])[0]
        
        # Calculate cosine similarity
        similarity = np.dot(gen_embedding, exp_embedding) / (
            np.linalg.norm(gen_embedding) * np.linalg.norm(exp_embedding)
        )
        
        return max(0.0, min(1.0, float(similarity)))  # Ensure in [0, 1] range
    
    def _calculate_coordination_efficiency(self, coordination_insights: Dict[str, Any]) -> float:
        """
        Calculate efficiency of expert coordination
        """
        if not coordination_insights or 'average_expert_qualities' not in coordination_insights:
            return 0.5  # Neutral score if no data
        
        avg_qualities = coordination_insights['average_expert_qualities']
        if not avg_qualities:
            return 0.5
        
        # Calculate how well experts complement each other
        # Higher variance in expert qualities indicates better specialization
        quality_variance = np.var(avg_qualities)
        avg_quality = np.mean(avg_qualities)
        
        # Combine variance and average quality
        efficiency = 0.6 * avg_quality + 0.4 * min(1.0, quality_variance * 2)
        
        return efficiency
    
    def _analyze_specialization_effectiveness(self, system: MultiExpertRAGSystem, 
                                            test_queries: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
        """
        Analyze how effectively experts specialize in their domains
        """
        domain_expert_mapping = {}
        
        for query_data in test_queries:
            query = query_data['query']
            expected_domain = query_data.get('domain', 'general')
            
            # Analyze query to determine expected domain
            query_analysis = system.dynamic_router.query_analyzer.analyze_query(query)
            actual_domain = query_analysis['primary_domain']
            
            # Get routing decision
            routing_result = system.dynamic_router.route_query_dynamically(query)
            selected_experts = routing_result['routing_decision']['selected_experts']
            
            domain_key = expected_domain if expected_domain != 'general' else actual_domain
            
            if domain_key not in domain_expert_mapping:
                domain_expert_mapping[domain_key] = {}
            
            for expert_id in selected_experts:
                if expert_id not in domain_expert_mapping[domain_key]:
                    domain_expert_mapping[domain_key][expert_id] = 0
                domain_expert_mapping[domain_key][expert_id] += 1
        
        # Calculate specialization scores
        specialization_analysis = {}
        for domain, expert_counts in domain_expert_mapping.items():
            total_assignments = sum(expert_counts.values())
            if total_assignments == 0:
                continue
            
            # Calculate distribution entropy (more entropy = more balanced)
            probabilities = [count/total_assignments for count in expert_counts.values()]
            entropy = -sum(p * np.log2(p + 1e-8) for p in probabilities)
            max_entropy = np.log2(len(expert_counts)) if expert_counts else 1
            balance_score = entropy / max_entropy if max_entropy > 0 else 0.0
            
            # Find primary expert for this domain
            primary_expert = max(expert_counts.items(), key=lambda x: x[1])
            expert_id, count = primary_expert
            dominance_score = count / total_assignments
            
            specialization_analysis[domain] = {
                'primary_expert': expert_id,
                'dominance_score': dominance_score,
                'balance_score': balance_score,
                'assignment_distribution': {k: v/total_assignments for k, v in expert_counts.items()},
                'total_assignments': total_assignments
            }
        
        return specialization_analysis
    
    def _calculate_expert_performance(self, system: MultiExpertRAGSystem, 
                                   test_queries: List[Dict[str, str]]) -> Dict[int, float]:
        """
        Calculate performance for each expert
        """
        expert_performance = {i: [] for i in range(system.multi_expert_core.num_experts)}
        
        for query_data in test_queries:
            query = query_data['query']
            expected_response = query_data.get('expected_response', '')
            
            result = system.process_query(query)
            
            # Calculate quality for each expert's contribution
            for expert_id in result['selected_experts']:
                if expected_response:
                    # In practice, we'd need to determine each expert's contribution
                    # For this example, we'll use overall response quality
                    quality = self._calculate_response_quality(result['response'], expected_response)
                    expert_performance[expert_id].append(quality)
        
        # Calculate average performance for each expert
        avg_performance = {}
        for expert_id, qualities in expert_performance.items():
            avg_performance[expert_id] = np.mean(qualities) if qualities else 0.0
        
        return avg_performance
    
    def _analyze_routing_distribution(self, query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the distribution of routing decisions
        """
        total_queries = len(query_results)
        expert_selection_counts = {}
        routing_patterns = []
        
        for result in query_results:
            for expert_id in result['selected_experts']:
                if expert_id not in expert_selection_counts:
                    expert_selection_counts[expert_id] = 0
                expert_selection_counts[expert_id] += 1
            
            # Record routing pattern (sorted expert IDs)
            pattern = tuple(sorted(result['selected_experts']))
            routing_patterns.append(pattern)
        
        # Analyze routing patterns
        pattern_frequency = {}
        for pattern in routing_patterns:
            if pattern not in pattern_frequency:
                pattern_frequency[pattern] = 0
            pattern_frequency[pattern] += 1
        
        return {
            'expert_selection_frequencies': {k: v/total_queries for k, v in expert_selection_counts.items()},
            'common_routing_patterns': pattern_frequency,
            'routing_diversity_score': len(set(routing_patterns)) / total_queries if total_queries > 0 else 0,
            'top_routing_patterns': sorted(pattern_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _calculate_load_balancing_score(self, expert_utilization: Dict[int, int]) -> float:
        """
        Calculate load balancing score (0-1, where 1 is perfectly balanced)
        """
        if not expert_utilization:
            return 0.0
        
        counts = list(expert_utilization.values())
        if not counts:
            return 1.0  # No utilization = balanced
        
        # Calculate coefficient of variation (lower CV = more balanced)
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        if mean_count == 0:
            return 1.0  # All zeros means equal (though not ideal)
        
        # Coefficient of variation: std/mean
        cv = std_count / mean_count if mean_count != 0 else 0
        
        # Convert to balance score (higher is better)
        balance_score = max(0.0, 1.0 - cv)
        
        return balance_score
    
    def _calculate_routing_efficiency(self, query_results: List[Dict[str, Any]]) -> float:
        """
        Calculate routing efficiency based on expert utilization
        """
        if not query_results:
            return 0.0
        
        # Calculate average number of experts used per query
        avg_experts_per_query = np.mean([len(result['selected_experts']) for result in query_results])
        
        # Efficiency is higher when fewer experts are needed (but not too few)
        # Optimal is around 2-3 experts per query
        optimal_experts = 2.5
        efficiency = 1.0 - abs(avg_experts_per_query - optimal_experts) / optimal_experts
        
        return max(0.0, min(1.0, efficiency))
    
    def _analyze_coordination_patterns(self, query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patterns in expert coordination
        """
        if not query_results:
            return {}
        
        # Analyze coordination insights from results
        all_insights = [result['coordination_insights'] for result in query_results if 'coordination_insights' in result]
        
        if not all_insights:
            return {'message': 'No coordination insights available'}
        
        # Calculate average coordination metrics
        avg_expert_qualities = []
        avg_coordination_weights = []
        
        for insight in all_insights:
            if 'average_expert_qualities' in insight:
                avg_expert_qualities.append(insight['average_expert_qualities'])
            if 'average_coordination_weights' in insight:
                avg_coordination_weights.append(insight['average_coordination_weights'])
        
        return {
            'avg_expert_qualities': np.mean(avg_expert_qualities, axis=0).tolist() if avg_expert_qualities else [],
            'avg_coordination_weights': np.mean(avg_coordination_weights, axis=0).tolist() if avg_coordination_weights else [],
            'coordination_trends': self._analyze_coordination_trends(all_insights)
        }
    
    def _analyze_coordination_trends(self, insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze trends in coordination over time
        """
        if len(insights) < 2:
            return {'message': 'Insufficient data for trend analysis'}
        
        # Analyze confidence trends
        confidences = [insight.get('global_confidence', 0.5) for insight in insights if 'global_confidence' in insight]
        
        if len(confidences) < 2:
            return {'message': 'Insufficient confidence data for trend analysis'}
        
        # Calculate trend using linear regression
        x = np.arange(len(confidences))
        slope, _ = np.polyfit(x, confidences, 1)
        
        trend_direction = 'improving' if slope > 0.01 else 'declining' if slope < -0.01 else 'stable'
        
        return {
            'trend_direction': trend_direction,
            'trend_slope': float(slope),
            'avg_confidence': float(np.mean(confidences)),
            'confidence_std': float(np.std(confidences))
        }
    
    def _analyze_expert_interactions(self, system: MultiExpertRAGSystem) -> Dict[str, Any]:
        """
        Analyze how experts interact and complement each other
        """
        # Get expert specializations
        specializations = system.multi_expert_core.expert_specializations
        
        # Analyze co-occurrence patterns
        co_occurrence_matrix = np.zeros((len(specializations), len(specializations)))
        
        # This would be populated based on actual usage patterns
        # For this example, we'll return a mock analysis
        return {
            'co_occurrence_analysis': 'Would analyze which experts are frequently used together',
            'complementarity_score': 0.7,  # Mock score
            'interaction_patterns': 'Would identify synergistic expert combinations',
            'specialization_overlap': 'Would identify redundant expert specializations'
        }

class PerformanceOptimizer:
    """
    Optimize performance of multi-expert system
    """
    def __init__(self, multi_expert_system: MultiExpertRAGSystem):
        self.system = multi_expert_system
        self.performance_history = []
        self.optimization_strategies = {
            'load_balancing': self._optimize_load_balancing,
            'expert_selection': self._optimize_expert_selection,
            'resource_allocation': self._optimize_resource_allocation,
            'routing_efficiency': self._optimize_routing_efficiency
        }
    
    def optimize_system(self, strategy: str = 'all') -> Dict[str, Any]:
        """
        Optimize the system using specified strategy
        """
        if strategy == 'all':
            results = {}
            for strat_name, strat_func in self.optimization_strategies.items():
                results[strat_name] = strat_func()
            return results
        elif strategy in self.optimization_strategies:
            return self.optimization_strategies[strategy]()
        else:
            raise ValueError(f"Unknown optimization strategy: {strategy}")
    
    def _optimize_load_balancing(self) -> Dict[str, Any]:
        """
        Optimize load balancing across experts
        """
        current_report = self.system.get_expert_utilization_report()
        
        # Identify overutilized and underutilized experts
        expert_utilization = current_report['expert_utilization']
        
        overutilized = [exp_id for exp_id, util in expert_utilization.items() if util > 0.8]
        underutilized = [exp_id for exp_id, util in expert_utilization.items() if util < 0.2]
        
        # Adjust routing weights to balance load
        load_balancing_adjustments = {}
        for exp_id in overutilized:
            load_balancing_adjustments[exp_id] = 'reduce_routing_weight'
        for exp_id in underutilized:
            load_balancing_adjustments[exp_id] = 'increase_routing_weight'
        
        return {
            'overutilized_experts': overutilized,
            'underutilized_experts': underutilized,
            'adjustments': load_balancing_adjustments,
            'current_balance_score': current_report['load_balance_report']['load_balance_ratio']
        }
    
    def _optimize_expert_selection(self) -> Dict[str, Any]:
        """
        Optimize expert selection based on performance
        """
        # Analyze which experts are performing well for which domains
        expert_analysis = self.system.get_expert_utilization_report()['expert_analysis']
        
        # Identify experts that are frequently selected but perform poorly
        performance_by_expert = expert_analysis.get('performance_by_expert', {})
        
        low_performance_experts = [
            exp_id for exp_id, perf in performance_by_expert.items() 
            if perf < 0.6  # Below threshold
        ]
        
        # Identify experts that are underutilized but perform well
        utilization = self.system.get_expert_utilization_report()['expert_utilization']
        high_performance_underutilized = [
            exp_id for exp_id, perf in performance_by_expert.items()
            if perf > 0.8 and utilization[exp_id] < 0.3  # High performance, low utilization
        ]
        
        return {
            'low_performance_experts': low_performance_experts,
            'high_performance_underutilized': high_performance_underutilized,
            'recommendations': self._generate_selection_recommendations(
                low_performance_experts, high_performance_underutilized
            )
        }
    
    def _generate_selection_recommendations(self, low_perf: List[int], 
                                          high_perf_under: List[int]) -> List[str]:
        """
        Generate recommendations for expert selection optimization
        """
        recommendations = []
        
        if low_perf:
            recommendations.append(f"Reduce selection of experts {low_perf} due to poor performance")
        
        if high_perf_under:
            recommendations.append(f"Increase selection of experts {high_perf_under} due to high performance and low utilization")
        
        return recommendations
    
    def _optimize_resource_allocation(self) -> Dict[str, Any]:
        """
        Optimize resource allocation to experts
        """
        # This would involve adjusting computational resources based on expert utilization
        # For this example, we'll return a mock optimization
        return {
            'current_allocation': 'equal_resources_per_expert',
            'recommended_allocation': 'proportional_to_utilization',
            'expected_improvement': '15% efficiency gain',
            'implementation_complexity': 'medium'
        }
    
    def _optimize_routing_efficiency(self) -> Dict[str, Any]:
        """
        Optimize routing efficiency
        """
        routing_analysis = self.system.get_expert_utilization_report()['routing_analysis']
        
        # Identify common inefficient routing patterns
        common_patterns = routing_analysis.get('top_routing_patterns', [])
        
        # Calculate efficiency of common patterns
        pattern_efficiencies = {}
        for pattern, count in common_patterns:
            # In practice, this would correlate with performance metrics
            pattern_efficiencies[pattern] = np.random.random()  # Mock efficiency
        
        return {
            'common_routing_patterns': common_patterns,
            'pattern_efficiencies': pattern_efficiencies,
            'optimization_opportunities': self._identify_routing_optimizations(pattern_efficiencies)
        }
    
    def _identify_routing_optimizations(self, pattern_efficiencies: Dict[tuple, float]) -> List[str]:
        """
        Identify opportunities for routing optimization
        """
        optimizations = []
        
        for pattern, efficiency in pattern_efficiencies.items():
            if efficiency < 0.5:
                optimizations.append(f"Pattern {pattern} has low efficiency ({efficiency:.2f}), consider alternative routing")
        
        return optimizations
```

## 4. Deployment Architecture

### 4.1 Multi-Expert Infrastructure
```yaml
# docker-compose.yml for multi-expert RAG system
version: '3.8'

services:
  # Main multi-expert RAG API
  multi-expert-rag-api:
    build: 
      context: .
      dockerfile: Dockerfile.multi-expert
    image: multi-expert-rag:latest
    container_name: multi-expert-rag-api
    ports:
      - "8000:8000"
    environment:
      - NUM_EXPERTS=8
      - EXPERT_CAPACITY=1024
      - MODEL_NAME=meta-llama/Llama-2-7b-hf
      - EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
    volumes:
      - multi_expert_data:/app/data
      - ./models:/app/models:ro
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '4'
    restart: unless-stopped

  # Individual expert services (can be scaled independently)
  expert-nlp:
    build: 
      context: .
      dockerfile: Dockerfile.expert
    environment:
      - EXPERT_ID=0
      - EXPERT_DOMAIN=nlp
      - SPECIALIZATION=language_processing
    deploy:
      replicas: 2
    restart: unless-stopped

  expert-code:
    build: 
      context: .
      dockerfile: Dockerfile.expert
    environment:
      - EXPERT_ID=1
      - EXPERT_DOMAIN=code
      - SPECIALIZATION=code_generation
    deploy:
      replicas: 2
    restart: unless-stopped

  expert-science:
    build: 
      context: .
      dockerfile: Dockerfile.expert
    environment:
      - EXPERT_ID=2
      - EXPERT_DOMAIN=science
      - SPECIALIZATION=scientific_analysis
    deploy:
      replicas: 1
    restart: unless-stopped

  expert-finance:
    build: 
      context: .
      dockerfile: Dockerfile.expert
    environment:
      - EXPERT_ID=3
      - EXPERT_DOMAIN=finance
      - SPECIALIZATION=financial_analysis
    deploy:
      replicas: 1
    restart: unless-stopped

  # Expert routing service
  expert-router:
    build: 
      context: .
      dockerfile: Dockerfile.expert-router
    environment:
      - ROUTING_STRATEGY=gating_network
      - CAPACITY_FACTOR=1.25
      - TOP_K_EXPERTS=2
    volumes:
      - multi_expert_routing_data:/app/routing_data
    restart: unless-stopped

  # Vector database for embeddings
  multi-expert-vector-db:
    image: pgvector/pgvector:pg14
    environment:
      - POSTGRES_DB=multi_expert_rag
      - POSTGRES_USER=multi_expert_user
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - multi_expert_vector_data:/var/lib/postgresql/data
    restart: unless-stopped

  # Knowledge base for each expert
  multi-expert-kb:
    image: postgres:13
    environment:
      - POSTGRES_DB=multi_expert_rag
      - POSTGRES_USER=multi_expert_user
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - multi_expert_kb_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  # Performance monitoring
  multi-expert-monitoring:
    image: grafana/prometheus
    ports:
      - "9090:9090"
      - "3000:3000"
    volumes:
      - multi_expert_monitoring_data:/prometheus
    restart: unless-stopped

  # Load balancer for expert services
  multi-expert-load-balancer:
    image: nginx:alpine
    volumes:
      - ./nginx-multi-expert.conf:/etc/nginx/nginx.conf
    ports:
      - "80:80"
    depends_on:
      - expert-nlp
      - expert-code
      - expert-science
      - expert-finance
    restart: unless-stopped

  # Evolutionary optimizer service
  evolutionary-optimizer:
    build:
      context: .
      dockerfile: Dockerfile.evolutionary
    environment:
      - POPULATION_SIZE=20
      - GENERATIONS=50
      - MUTATION_RATE=0.1
    volumes:
      - multi_expert_data:/data
    restart: unless-stopped

volumes:
  multi_expert_data:
  multi_expert_routing_data:
  multi_expert_kb_data:
  multi_expert_vector_data:
  multi_expert_monitoring_data:

networks:
  multi_expert_network:
    driver: bridge
```

## 5. Security and Privacy

### 5.1 Multi-Expert Security Measures
```python
class MultiExpertSecurityManager:
    """
    Security manager for multi-expert RAG system
    """
    def __init__(self):
        self.expert_isolation = ExpertIsolationManager()
        self.routing_security = RoutingSecurityManager()
        self.data_encryption = DataEncryptionManager()
        self.access_control = MultiExpertAccessControl()
        self.audit_logger = MultiExpertAuditLogger()
    
    def secure_process_request(self, query: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Securely process a request through the multi-expert system
        """
        # Verify user permissions
        if not self.access_control.verify_permission(user_context, 'query'):
            raise PermissionError("User not authorized for queries")
        
        # Log the request
        request_id = self.audit_logger.log_request(user_context, query)
        
        try:
            # Sanitize input
            sanitized_query = self._sanitize_input(query)
            
            # Process through secure pipeline
            result = self._secure_multi_expert_processing(sanitized_query, user_context)
            
            # Log successful processing
            self.audit_logger.log_success(request_id, result)
            
            return result
            
        except Exception as e:
            # Log failure
            self.audit_logger.log_failure(request_id, str(e))
            raise e
    
    def _sanitize_input(self, query: str) -> str:
        """
        Sanitize input to prevent injection attacks
        """
        import re
        
        # Remove potentially harmful patterns
        dangerous_patterns = [
            r'exec\(', r'eval\(', r'import\s+', r'__import__',
            r'open\(', r'file\(', r'system\(', r'shell\('
        ]
        
        sanitized = query
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '[REMOVED]', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def _secure_multi_expert_processing(self, query: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process query through secure multi-expert pipeline
        """
        # In practice, this would call the actual multi-expert RAG system
        # For this example, we'll simulate the processing
        return {
            'response': f"Secure multi-expert response to: {query[:50]}...",
            'selected_experts': [0, 1],  # Simulated expert selection
            'processing_time_ms': 180,
            'security_level': 'high'
        }

class ExpertIsolationManager:
    """
    Manage isolation between different experts
    """
    def __init__(self):
        self.expert_containers = {}
        self.resource_limits = {}
        self.communication_channels = {}
    
    def isolate_expert(self, expert_id: int, domain: str):
        """
        Isolate an expert in its own secure environment
        """
        # Create isolated container/environment for expert
        container_config = {
            'memory_limit': '512m',
            'cpu_quota': '0.5',
            'network_isolated': True,
            'filesystem_readonly': True,
            'domain': domain
        }
        
        self.expert_containers[expert_id] = container_config
        
        # Set resource limits
        self.resource_limits[expert_id] = {
            'max_memory_mb': 512,
            'max_cpu_percent': 50,
            'max_requests_per_minute': 100
        }
    
    def verify_isolation(self, expert_id: int, request_data: Dict[str, Any]) -> bool:
        """
        Verify that request doesn't violate expert isolation
        """
        if expert_id not in self.expert_containers:
            return False
        
        # Check resource usage
        current_usage = self._get_current_resource_usage(expert_id)
        limits = self.resource_limits[expert_id]
        
        if current_usage['memory_mb'] > limits['max_memory_mb']:
            return False
        
        if current_usage['cpu_percent'] > limits['max_cpu_percent']:
            return False
        
        # Check domain appropriateness
        expert_domain = self.expert_containers[expert_id]['domain']
        request_domain = request_data.get('domain', 'general')
        
        # For security, experts should only handle appropriate domains
        allowed_domains = self._get_allowed_domains(expert_domain)
        return request_domain in allowed_domains
    
    def _get_current_resource_usage(self, expert_id: int) -> Dict[str, float]:
        """
        Get current resource usage for an expert
        """
        # In practice, this would monitor actual resource usage
        # For this example, return mock data
        return {
            'memory_mb': np.random.uniform(100, 500),
            'cpu_percent': np.random.uniform(10, 80)
        }
    
    def _get_allowed_domains(self, expert_domain: str) -> List[str]:
        """
        Get domains that an expert is allowed to handle
        """
        domain_mappings = {
            'nlp': ['nlp', 'language', 'text', 'general'],
            'code': ['code', 'programming', 'software', 'general'],
            'science': ['science', 'research', 'academic', 'general'],
            'finance': ['finance', 'business', 'economic', 'general'],
            'general': ['general', 'any']
        }
        
        return domain_mappings.get(expert_domain, ['general'])

class RoutingSecurityManager:
    """
    Security for expert routing decisions
    """
    def __init__(self):
        self.routing_validation_rules = {}
        self.anomaly_detector = RoutingAnomalyDetector()
    
    def validate_routing_decision(self, query: str, selected_experts: List[int], 
                                routing_weights: List[float]) -> bool:
        """
        Validate that routing decision is appropriate and secure
        """
        # Check for routing anomalies
        if self.anomaly_detector.detect_anomaly(query, selected_experts, routing_weights):
            return False
        
        # Validate expert appropriateness
        query_domain = self._infer_query_domain(query)
        for expert_id in selected_experts:
            if not self._is_expert_appropriate(expert_id, query_domain):
                return False
        
        # Check for potential abuse patterns
        if self._detect_abuse_pattern(selected_experts, routing_weights):
            return False
        
        return True
    
    def _infer_query_domain(self, query: str) -> str:
        """
        Infer domain of query for routing validation
        """
        query_lower = query.lower()
        
        domain_indicators = {
            'nlp': ['language', 'text', 'translate', 'summarize', 'write', 'grammar'],
            'code': ['code', 'program', 'function', 'algorithm', 'software', 'develop'],
            'science': ['research', 'study', 'experiment', 'hypothesis', 'scientific'],
            'finance': ['money', 'investment', 'stock', 'market', 'economic', 'financial'],
            'general': ['what', 'how', 'why', 'when', 'where', 'who', 'explain']
        }
        
        for domain, indicators in domain_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                return domain
        
        return 'general'
    
    def _is_expert_appropriate(self, expert_id: int, query_domain: str) -> bool:
        """
        Check if expert is appropriate for query domain
        """
        # In practice, this would check against expert specializations
        # For this example, use a simple mapping
        expert_domains = {
            0: ['nlp', 'general'],
            1: ['code', 'general'],
            2: ['science', 'general'],
            3: ['finance', 'general'],
            4: ['legal', 'general'],
            5: ['medical', 'general'],
            6: ['technical', 'general'],
            7: ['general']
        }
        
        return query_domain in expert_domains.get(expert_id, ['general'])

class MultiExpertAccessControl:
    """
    Access control for multi-expert system
    """
    def __init__(self):
        self.user_permissions = {}
        self.expert_access_rules = {}
        self.rate_limiters = {}
    
    def verify_permission(self, user_context: Dict[str, Any], operation: str) -> bool:
        """
        Verify if user has permission for operation
        """
        user_id = user_context.get('user_id')
        user_role = user_context.get('role', 'user')
        
        # Check basic permissions
        if user_id not in self.user_permissions:
            return False
        
        user_perms = self.user_permissions[user_id]
        if operation not in user_perms:
            return False
        
        # Check expert-specific permissions
        if operation.startswith('expert_'):
            expert_id = int(operation.split('_')[1])  # Extract expert ID
            if not self._check_expert_access(user_id, expert_id):
                return False
        
        # Check rate limits
        if not self._check_rate_limit(user_id, operation):
            return False
        
        return True
    
    def _check_expert_access(self, user_id: str, expert_id: int) -> bool:
        """
        Check if user has access to specific expert
        """
        if user_id not in self.expert_access_rules:
            return False
        
        allowed_experts = self.expert_access_rules[user_id]
        return expert_id in allowed_experts or 'all' in allowed_experts
    
    def _check_rate_limit(self, user_id: str, operation: str) -> bool:
        """
        Check if user has exceeded rate limits
        """
        key = f"{user_id}:{operation}"
        
        if key not in self.rate_limiters:
            self.rate_limiters[key] = {
                'count': 0,
                'window_start': time.time()
            }
        
        current_time = time.time()
        
        # Reset window if needed (1 minute window)
        if current_time - self.rate_limiters[key]['window_start'] > 60:
            self.rate_limiters[key] = {
                'count': 0,
                'window_start': current_time
            }
        
        # Check if limit exceeded (100 requests per minute)
        if self.rate_limiters[key]['count'] >= 100:
            return False
        
        # Increment count
        self.rate_limiters[key]['count'] += 1
        return True

class MultiExpertAuditLogger:
    """
    Audit logging for multi-expert system
    """
    def __init__(self):
        import json
        self.log_file = "multi_expert_audit.log"
    
    def log_request(self, user_context: Dict[str, Any], query: str) -> str:
        """
        Log a request to the multi-expert system
        """
        import uuid
        request_id = str(uuid.uuid4())
        
        log_entry = {
            'timestamp': time.time(),
            'request_id': request_id,
            'user_id': user_context.get('user_id'),
            'user_role': user_context.get('role'),
            'query_preview': query[:100] + "..." if len(query) > 100 else query,
            'source_ip': user_context.get('source_ip', 'unknown'),
            'event_type': 'multi_expert_request'
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        return request_id
    
    def log_success(self, request_id: str, result: Dict[str, Any]):
        """
        Log successful processing
        """
        log_entry = {
            'timestamp': time.time(),
            'request_id': request_id,
            'event_type': 'multi_expert_success',
            'selected_experts': result.get('selected_experts', []),
            'processing_time_ms': result.get('processing_time_ms', 0),
            'response_length': len(result.get('response', ''))
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_failure(self, request_id: str, error_message: str):
        """
        Log processing failure
        """
        log_entry = {
            'timestamp': time.time(),
            'request_id': request_id,
            'event_type': 'multi_expert_failure',
            'error': error_message
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
```

## 6. Performance Benchmarks

### 6.1 Expected Performance Metrics
| Metric | Target | Current | Notes |
|--------|--------|---------|-------|
| Expert Utilization Balance | > 0.7 | TBD | Even distribution across experts |
| Routing Accuracy | > 0.85 | TBD | Correct expert assignment |
| Computational Efficiency | 2x improvement | TBD | Over naive parallel processing |
| Response Quality | > 0.8 | TBD | Compared to single-expert approach |
| Load Balancing Score | > 0.75 | TBD | Expert workload distribution |
| Specialization Effectiveness | > 0.8 | TBD | Domain-specific performance |

## 7. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)
- Implement core multi-expert RAG architecture
- Develop expert routing mechanisms
- Create basic knowledge bases for each expert
- Implement simple evaluation framework

### Phase 2: Advanced Features (Weeks 5-8)
- Implement dynamic expert allocation
- Add evolutionary optimization
- Develop swarm intelligence components
- Create comprehensive evaluation suite

### Phase 3: Optimization (Weeks 9-12)
- Optimize for specific domains
- Improve routing accuracy
- Enhance computational efficiency
- Performance tuning

### Phase 4: Production (Weeks 13-16)
- Deploy to production environment
- Implement monitoring and alerting
- Security and privacy validation
- Documentation and user guides

## 8. Conclusion

The multi-expert RAG system design presents a comprehensive architecture that leverages specialized expertise to enhance information retrieval and generation. By combining multiple domain-specific experts with intelligent routing mechanisms, the system achieves superior performance across diverse query types while maintaining computational efficiency.

The solution addresses critical challenges in traditional RAG systems by providing:
- Specialized expertise for different domains
- Dynamic routing based on query characteristics
- Efficient resource utilization through expert selection
- Scalable architecture that can grow with new experts
- Quality assurance through expert coordination

While challenges remain in expert coordination and load balancing, the fundamental approach of multi-expert specialization shows great promise for creating more capable and efficient AI systems that can handle diverse, complex queries effectively. The system represents a significant advancement in creating AI systems that can leverage specialized knowledge while maintaining the flexibility to adapt to new domains and requirements.