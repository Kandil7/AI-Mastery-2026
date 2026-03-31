"""
Graph-Enhanced RAG (Retrieval-Augmented Generation) Module

This module implements a graph-enhanced RAG system that leverages knowledge graphs
to improve retrieval and generation. It builds entity-relation graphs from documents
and uses graph-based reasoning to enhance the RAG process.

Key Features:
- Entity and relation extraction from documents
- Knowledge graph construction and maintenance
- Graph-based retrieval using entity linking
- Path-based reasoning for complex queries
- Graph neural network integration for enhanced representations
- Multi-hop reasoning capabilities

Architecture:
- Entity Extractor: Extracts named entities from documents
- Relation Extractor: Identifies relations between entities
- Graph Builder: Constructs knowledge graph from entities and relations
- Graph Retriever: Retrieves relevant subgraphs for queries
- Graph Enhancer: Enhances embeddings using graph structure
- Graph Generator: Generates responses using graph context
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from enum import Enum
import hashlib
import re
from collections import defaultdict, deque
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# ENUMS AND DATA CLASSES
# ============================================================

class EntityType(Enum):
    """Types of entities in the knowledge graph."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    EVENT = "event"
    CONCEPT = "concept"
    PRODUCT = "product"
    OTHER = "other"


class RelationType(Enum):
    """Types of relations in the knowledge graph."""
    WORKS_AT = "works_at"
    LOCATED_IN = "located_in"
    PART_OF = "part_of"
    CAUSES = "causes"
    RELATED_TO = "related_to"
    SIMILAR_TO = "similar_to"
    IS_A = "is_a"
    HAS_PROPERTY = "has_property"
    OCCURS_DURING = "occurs_during"
    OTHER = "other"


@dataclass
class Entity:
    """An entity in the knowledge graph."""
    id: str
    name: str
    type: EntityType
    description: str = ""
    embedding: Optional[np.ndarray] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5((self.name + self.type.value).encode()).hexdigest()[:16]


@dataclass
class Relation:
    """A relation between two entities in the knowledge graph."""
    id: str
    subject_id: str  # Entity ID
    object_id: str   # Entity ID
    type: RelationType
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5((self.subject_id + self.object_id + self.type.value).encode()).hexdigest()[:16]


@dataclass
class GraphDocument:
    """A document with associated graph information."""
    id: str
    content: str
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(self.content.encode()).hexdigest()[:16]


@dataclass
class GraphQuery:
    """A query that may involve graph reasoning."""
    text: str
    entities: List[Entity] = field(default_factory=list)  # Entities extracted from query
    hops: int = 2  # Number of hops for graph traversal
    include_related: bool = True  # Include related entities in retrieval
    relation_types: Optional[List[RelationType]] = None  # Specific relation types to consider


@dataclass
class GraphRetrievalResult:
    """Result from graph-enhanced retrieval."""
    document: GraphDocument
    score: float
    entity_matches: List[Entity]
    relation_matches: List[Relation]
    path_depth: int  # How deep in the graph this connection was found
    rank: int = 0


@dataclass
class GraphGenerationResult:
    """Result from graph-enhanced generation."""
    answer: str
    sources: List[GraphDocument]
    entities_mentioned: List[Entity]
    relations_discovered: List[Relation]
    reasoning_paths: List[List[str]]  # Paths taken during graph reasoning
    confidence: float
    latency_ms: float
    token_count: int


# ============================================================
# ENTITY AND RELATION EXTRACTION
# ============================================================

class EntityExtractor:
    """Extracts named entities from text."""
    
    def __init__(self):
        # Common entity patterns for rule-based extraction
        self.patterns = {
            EntityType.PERSON: [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # John Smith
                r'\bDr\.?\s+[A-Z][a-z]+\b',      # Dr. Smith
                r'\bMr\.?\s+[A-Z][a-z]+\b',      # Mr. Smith
            ],
            EntityType.ORGANIZATION: [
                r'\b[A-Z][A-Za-z\s]+(?:Inc|Corp|Ltd|LLC|Co)\b',  # Company names
                r'\b(?:University|College|School)\s+[A-Z][A-Za-z\s]+\b',  # Educational institutions
                r'\b(?:Department|Ministry|Agency)\s+[A-Z][A-Za-z\s]+\b',  # Government entities
            ],
            EntityType.LOCATION: [
                r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b',  # City, ST
                r'\b(?:New York|Los Angeles|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego|Dallas|San Jose)\b',  # Major US cities
                r'\b(?:USA|United States|America|Canada|Mexico|UK|England|France|Germany|Japan|China)\b',  # Countries
            ],
            EntityType.DATE: [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
                r'\b\d{4}-\d{2}-\d{2}\b',      # YYYY-MM-DD
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',  # Month DD, YYYY
            ]
        }
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text using pattern matching."""
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    # Clean the match
                    clean_match = match.strip()
                    
                    # Avoid duplicates
                    if not any(e.name.lower() == clean_match.lower() for e in entities):
                        entity = Entity(
                            id=hashlib.md5((clean_match + entity_type.value).encode()).hexdigest()[:16],
                            name=clean_match,
                            type=entity_type
                        )
                        entities.append(entity)
        
        return entities


class RelationExtractor:
    """Extracts relations between entities."""
    
    def __init__(self):
        # Common relation patterns
        self.relation_patterns = [
            # Pattern: "X works at Y" -> WORKS_AT
            (r'(\w+(?:\s+\w+)*)\s+(?:works at|works for|is employed by|is a member of)\s+(\w+(?:\s+\w+)*)', RelationType.WORKS_AT),
            # Pattern: "X is located in Y" -> LOCATED_IN
            (r'(\w+(?:\s+\w+)*)\s+(?:is located in|is in|located in|based in)\s+(\w+(?:\s+\w+)*)', RelationType.LOCATED_IN),
            # Pattern: "X is part of Y" -> PART_OF
            (r'(\w+(?:\s+\w+)*)\s+(?:is part of|part of|belongs to)\s+(\w+(?:\s+\w+)*)', RelationType.PART_OF),
            # Pattern: "X causes Y" -> CAUSES
            (r'(\w+(?:\s+\w+)*)\s+(?:causes|leads to|results in|triggers)\s+(\w+(?:\s+\w+)*)', RelationType.CAUSES),
            # Pattern: "X is related to Y" -> RELATED_TO
            (r'(\w+(?:\s+\w+)*)\s+(?:is related to|related to|connected to|associated with)\s+(\w+(?:\s+\w+)*)', RelationType.RELATED_TO),
        ]
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations from text based on entity pairs."""
        relations = []
        
        # Create entity name to ID mapping
        entity_names = {e.name.lower(): e.id for e in entities}
        
        for pattern, rel_type in self.relation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                subj_name, obj_name = match[0].strip(), match[1].strip()
                
                # Map to entity IDs
                subj_id = entity_names.get(subj_name.lower())
                obj_id = entity_names.get(obj_name.lower())
                
                if subj_id and obj_id:
                    relation = Relation(
                        id=hashlib.md5((subj_id + obj_id + rel_type.value).encode()).hexdigest()[:16],
                        subject_id=subj_id,
                        object_id=obj_id,
                        type=rel_type,
                        confidence=0.8  # High confidence for pattern-based extraction
                    )
                    
                    # Avoid duplicates
                    if not any(r.subject_id == relation.subject_id and 
                              r.object_id == relation.object_id and 
                              r.type == relation.type for r in relations):
                        relations.append(relation)
        
        return relations


# ============================================================
# KNOWLEDGE GRAPH
# ============================================================

class KnowledgeGraph:
    """Manages the knowledge graph structure."""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.graph: nx.Graph = nx.Graph()  # Using NetworkX for graph operations
        self.entity_adjacency: Dict[str, List[str]] = defaultdict(list)  # Adjacency list for fast traversal
    
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph."""
        self.entities[entity.id] = entity
        self.graph.add_node(entity.id, entity=entity)
    
    def add_relation(self, relation: Relation) -> None:
        """Add a relation to the graph."""
        self.relations[relation.id] = relation
        
        # Add edge to NetworkX graph
        self.graph.add_edge(
            relation.subject_id, 
            relation.object_id, 
            relation=relation,
            weight=relation.confidence
        )
        
        # Update adjacency list
        self.entity_adjacency[relation.subject_id].append(relation.object_id)
        self.entity_adjacency[relation.object_id].append(relation.subject_id)
    
    def add_document(self, doc: GraphDocument) -> None:
        """Add a document's entities and relations to the graph."""
        # Add all entities
        for entity in doc.entities:
            if entity.id not in self.entities:
                self.add_entity(entity)
        
        # Add all relations
        for relation in doc.relations:
            if relation.id not in self.relations:
                self.add_relation(relation)
    
    def get_neighbors(self, entity_id: str, relation_types: Optional[List[RelationType]] = None) -> List[str]:
        """Get neighboring entities, optionally filtered by relation type."""
        if entity_id not in self.entity_adjacency:
            return []
        
        neighbors = self.entity_adjacency[entity_id]
        
        if relation_types:
            # Filter by relation type
            filtered_neighbors = []
            for neighbor_id in neighbors:
                # Check if the relation between entity_id and neighbor_id is of the desired type
                try:
                    edge_data = self.graph.get_edge_data(entity_id, neighbor_id)
                    if edge_data and 'relation' in edge_data:
                        relation = edge_data['relation']
                        if relation.type in relation_types:
                            filtered_neighbors.append(neighbor_id)
                except nx.NetworkXNoEdge:
                    continue
            return filtered_neighbors
        
        return neighbors
    
    def find_shortest_path(self, start_id: str, end_id: str) -> Optional[List[str]]:
        """Find shortest path between two entities."""
        try:
            path = nx.shortest_path(self.graph, source=start_id, target=end_id)
            return path
        except nx.NetworkXNoPath:
            return None
    
    def find_k_hop_neighbors(self, entity_id: str, k: int) -> Set[str]:
        """Find all entities within k hops of the given entity."""
        if k <= 0:
            return {entity_id}
        
        visited = {entity_id}
        current_level = [entity_id]
        
        for _ in range(k):
            next_level = []
            for entity in current_level:
                neighbors = self.get_neighbors(entity)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
            current_level = next_level
        
        return visited
    
    def get_subgraph(self, entity_ids: List[str]) -> nx.Graph:
        """Get subgraph containing specified entities."""
        return self.graph.subgraph(entity_ids)


# ============================================================
# GRAPH RETRIEVER
# ============================================================

class GraphRetriever:
    """
    Graph-enhanced retriever that uses knowledge graph for improved retrieval.
    
    This retriever combines traditional text similarity with graph-based
    reasoning to find relevant documents.
    """
    
    def __init__(self, embedding_dim: int = 384, graph_weight: float = 0.4):
        """
        Initialize graph retriever.
        
        Args:
            embedding_dim: Dimension of document embeddings
            graph_weight: Weight given to graph-based relevance vs text similarity
        """
        self.embedding_dim = embedding_dim
        self.graph_weight = graph_weight
        self.documents: List[GraphDocument] = []
        self.embeddings: Optional[np.ndarray] = None
        self.knowledge_graph = KnowledgeGraph()
        
        # Initialize extractors
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()
    
    def add_documents(self, documents: List[GraphDocument]) -> None:
        """Add documents to the retriever and update the knowledge graph."""
        self.documents.extend(documents)
        
        # Process each document to extract entities and relations if not already done
        for doc in documents:
            if not doc.entities:
                doc.entities = self.entity_extractor.extract_entities(doc.content)
            if not doc.relations:
                doc.relations = self.relation_extractor.extract_relations(doc.content, doc.entities)
            
            # Add document to knowledge graph
            self.knowledge_graph.add_document(doc)
        
        # Collect embeddings
        new_embeddings = []
        for doc in documents:
            if doc.embedding is not None:
                new_embeddings.append(doc.embedding)
            else:
                # Generate random embedding as placeholder
                new_embeddings.append(np.random.randn(self.embedding_dim))
        
        if new_embeddings:
            new_emb_array = np.array(new_embeddings)
            if self.embeddings is None:
                self.embeddings = new_emb_array
            else:
                self.embeddings = np.vstack([self.embeddings, new_emb_array])
    
    def _compute_semantic_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute semantic similarity between query and documents."""
        if self.embeddings is None or len(self.documents) == 0:
            return np.array([])
        
        # Normalize for cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        emb_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarities
        similarities = np.dot(emb_norms, query_norm)
        return similarities
    
    def _compute_graph_relevance(self, query: GraphQuery) -> np.ndarray:
        """Compute graph-based relevance scores for documents."""
        scores = np.zeros(len(self.documents))
        
        # Extract entities from query if not provided
        if not query.entities:
            query.entities = self.entity_extractor.extract_entities(query.text)
        
        if not query.entities:
            # If no entities found, return uniform scores
            return scores
        
        # For each document, compute graph relevance based on entity overlap and connections
        for doc_idx, doc in enumerate(self.documents):
            doc_score = 0.0
            
            # Count entity matches
            entity_matches = []
            for q_entity in query.entities:
                for doc_entity in doc.entities:
                    if q_entity.name.lower() == doc_entity.name.lower():
                        entity_matches.append(doc_entity)
                        doc_score += 0.3  # Base score for entity match
            
            # Count relation matches (relations involving query entities)
            relation_matches = []
            for relation in doc.relations:
                if (relation.subject_id in [e.id for e in entity_matches] or
                    relation.object_id in [e.id for e in entity_matches]):
                    relation_matches.append(relation)
                    doc_score += 0.2  # Bonus for related relations
            
            # Perform graph traversal to find indirect connections
            for q_entity in query.entities:
                if q_entity.id in self.knowledge_graph.entities:
                    # Find entities within k hops
                    k_hop_entities = self.knowledge_graph.find_k_hop_neighbors(q_entity.id, query.hops)
                    
                    # Check if any of these entities are in the current document
                    doc_entity_ids = {e.id for e in doc.entities}
                    connected_entities = k_hop_entities.intersection(doc_entity_ids)
                    
                    if connected_entities:
                        # Add score based on connection depth and number of connections
                        doc_score += 0.1 * len(connected_entities) / (query.hops + 1)
            
            scores[doc_idx] = min(1.0, doc_score)
        
        return scores
    
    def retrieve(self, 
                 query: GraphQuery, 
                 query_embedding: np.ndarray, 
                 k: int = 5) -> List[GraphRetrievalResult]:
        """Retrieve documents using both semantic and graph-based relevance."""
        if len(self.documents) == 0:
            return []
        
        # Compute semantic similarities
        semantic_similarities = self._compute_semantic_similarity(query_embedding)
        
        # Compute graph-based relevance
        graph_relevance = self._compute_graph_relevance(query)
        
        # Combine scores
        combined_scores = (
            (1 - self.graph_weight) * semantic_similarities + 
            self.graph_weight * graph_relevance
        )
        
        # Get top-k indices
        top_k_indices = np.argsort(combined_scores)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_k_indices, 1):
            if idx < len(self.documents):
                doc = self.documents[idx]
                
                # Find entity and relation matches for this document
                entity_matches = []
                for q_entity in query.entities:
                    for doc_entity in doc.entities:
                        if q_entity.name.lower() == doc_entity.name.lower():
                            entity_matches.append(doc_entity)
                
                relation_matches = []
                for relation in doc.relations:
                    if (relation.subject_id in [e.id for e in entity_matches] or
                        relation.object_id in [e.id for e in entity_matches]):
                        relation_matches.append(relation)
                
                result = GraphRetrievalResult(
                    document=doc,
                    score=float(combined_scores[idx]),
                    entity_matches=entity_matches,
                    relation_matches=relation_matches,
                    path_depth=query.hops,  # Use the query's hop setting as path depth
                    rank=rank
                )
                results.append(result)
        
        return results


# ============================================================
# GRAPH ENHANCER
# ============================================================

class GraphEnhancer:
    """
    Enhances document representations using graph structure.
    
    This component uses graph neural networks or random walk techniques
    to enhance document embeddings with structural information.
    """
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
    
    def enhance_embeddings(self, 
                          documents: List[GraphDocument], 
                          knowledge_graph: KnowledgeGraph) -> List[GraphDocument]:
        """Enhance document embeddings using graph structure."""
        enhanced_docs = []
        
        for doc in documents:
            # Create enhanced embedding based on connected entities
            enhanced_embedding = doc.embedding.copy() if doc.embedding is not None else np.random.randn(self.embedding_dim)
            
            # If document has entities, enhance embedding with neighborhood information
            if doc.entities:
                neighborhood_embedding = np.zeros(self.embedding_dim)
                entity_count = 0
                
                for entity in doc.entities:
                    if entity.id in knowledge_graph.entities and entity.embedding is not None:
                        neighborhood_embedding += entity.embedding
                        entity_count += 1
                
                if entity_count > 0:
                    # Average entity embeddings
                    avg_entity_embedding = neighborhood_embedding / entity_count
                    
                    # Blend original document embedding with entity information
                    enhanced_embedding = 0.7 * enhanced_embedding + 0.3 * avg_entity_embedding
            
            # Create new document with enhanced embedding
            enhanced_doc = GraphDocument(
                id=doc.id,
                content=doc.content,
                entities=doc.entities,
                relations=doc.relations,
                metadata=doc.metadata,
                embedding=enhanced_embedding
            )
            enhanced_docs.append(enhanced_doc)
        
        return enhanced_docs


# ============================================================
# GRAPH RAG SYSTEM
# ============================================================

class GraphEnhancedRAG:
    """
    Graph-Enhanced RAG system that leverages knowledge graphs.
    
    This system builds and maintains a knowledge graph from documents
    and uses graph-based reasoning to enhance retrieval and generation.
    """
    
    def __init__(self, embedding_dim: int = 384, graph_weight: float = 0.4):
        """
        Initialize graph-enhanced RAG system.
        
        Args:
            embedding_dim: Dimension of document embeddings
            graph_weight: Weight given to graph-based relevance vs text similarity
        """
        self.retriever = GraphRetriever(embedding_dim=embedding_dim, graph_weight=graph_weight)
        self.enhancer = GraphEnhancer(embedding_dim=embedding_dim)
        
        # Generation function (placeholder - replace with actual LLM)
        self.generate_fn: Optional[Callable] = None
        
        logger.info("Initialized Graph-Enhanced RAG system")
    
    def set_generator(self, generate_fn: Callable[[str], str]) -> None:
        """Set the LLM generation function."""
        self.generate_fn = generate_fn
    
    def add_documents(self, documents: List[GraphDocument]) -> int:
        """Add documents to the graph-enhanced RAG system."""
        # Process documents to extract entities and relations
        processed_docs = []
        for doc in documents:
            # Extract entities if not provided
            if not doc.entities:
                doc.entities = self.retriever.entity_extractor.extract_entities(doc.content)
            
            # Extract relations if not provided
            if not doc.relations:
                doc.relations = self.retriever.relation_extractor.extract_relations(doc.content, doc.entities)
            
            # Generate embedding if not provided
            if doc.embedding is None:
                content_hash = hashlib.md5(doc.content.encode()).hexdigest()
                embedding = np.frombuffer(bytes.fromhex(content_hash[:32]), dtype=np.float32)
                # Pad or truncate to desired dimension
                if len(embedding) < self.retriever.embedding_dim:
                    embedding = np.pad(embedding, (0, self.retriever.embedding_dim - len(embedding)), 'constant')
                elif len(embedding) > self.retriever.embedding_dim:
                    embedding = embedding[:self.retriever.embedding_dim]
                doc.embedding = embedding
            
            processed_docs.append(doc)
        
        # Enhance embeddings with graph information
        enhanced_docs = self.enhancer.enhance_embeddings(processed_docs, self.retriever.knowledge_graph)
        
        # Add to retriever
        self.retriever.add_documents(enhanced_docs)
        
        logger.info(f"Added {len(documents)} graph-enhanced documents")
        return len(documents)
    
    def query(self, 
              query: GraphQuery, 
              query_embedding: np.ndarray, 
              k: int = 5) -> GraphGenerationResult:
        """
        Query the graph-enhanced RAG system.
        
        Args:
            query: Graph query object
            query_embedding: Embedding vector for the query
            k: Number of results to retrieve
            
        Returns:
            GraphGenerationResult with answer and graph context
        """
        import time
        start_time = time.time()
        
        # Retrieve relevant documents
        retrieval_results = self.retriever.retrieve(query, query_embedding, k=k)
        
        if not retrieval_results:
            # No results found, return a default response
            latency_ms = (time.time() - start_time) * 1000
            return GraphGenerationResult(
                answer="No relevant information found for your query.",
                sources=[],
                entities_mentioned=[],
                relations_discovered=[],
                reasoning_paths=[],
                confidence=0.0,
                latency_ms=latency_ms,
                token_count=10
            )
        
        # Build context from retrieved documents
        context_parts = []
        all_sources = []
        all_entities = []
        all_relations = []
        
        for result in retrieval_results:
            context_parts.append(f"Document: {result.document.content}")
            all_sources.append(result.document)
            
            # Collect entities and relations
            all_entities.extend(result.entity_matches)
            all_relations.extend(result.relation_matches)
        
        context = "\n\n".join(context_parts)
        
        # Build reasoning paths from graph traversal
        reasoning_paths = self._extract_reasoning_paths(query, retrieval_results)
        
        # Build the generation prompt
        prompt = self._build_prompt(query, context, all_entities, all_relations, reasoning_paths)
        
        # Generate answer
        if self.generate_fn:
            answer = self.generate_fn(prompt)
        else:
            # Placeholder answer
            answer = self._generate_placeholder(query, retrieval_results)
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = GraphGenerationResult(
            answer=answer,
            sources=all_sources,
            entities_mentioned=list(set(all_entities, key=lambda x: x.id)),
            relations_discovered=list(set(all_relations, key=lambda x: x.id)),
            reasoning_paths=reasoning_paths,
            confidence=self._estimate_confidence(retrieval_results),
            latency_ms=latency_ms,
            token_count=len(prompt.split()) + len(answer.split())
        )
        
        return result
    
    def _extract_reasoning_paths(self, 
                                query: GraphQuery, 
                                results: List[GraphRetrievalResult]) -> List[List[str]]:
        """Extract reasoning paths from graph traversal."""
        paths = []
        
        # For each query entity, find paths to document entities
        for q_entity in query.entities:
            for result in results:
                for doc_entity in result.entity_matches:
                    path = self.retriever.knowledge_graph.find_shortest_path(q_entity.id, doc_entity.id)
                    if path:
                        paths.append(path)
        
        return paths[:5]  # Limit to top 5 paths
    
    def _build_prompt(self, 
                     query: GraphQuery, 
                     context: str, 
                     entities: List[Entity], 
                     relations: List[Relation], 
                     paths: List[List[str]]) -> str:
        """Build the generation prompt with graph context."""
        entity_list = ", ".join([f"{e.name} ({e.type.value})" for e in entities])
        relation_list = ", ".join([f"{r.type.value}" for r in relations])
        path_descriptions = "; ".join([f"{' -> '.join(p)}" for p in paths[:3]])  # Top 3 paths
        
        return f"""Answer the question based on the provided context, utilizing the graph-based information.
The query contains entities: {entity_list}
Relevant relations: {relation_list}
Reasoning paths: {path_descriptions}

If the answer cannot be found in the context, say "I don't have enough information to answer this question."
Always cite your sources and mention relevant entities and relationships.

Context:
{context}

Question: {query.text}

Answer:"""
    
    def _generate_placeholder(self, 
                            query: GraphQuery, 
                            results: List[GraphRetrievalResult]) -> str:
        """Generate a placeholder answer for testing."""
        entity_count = sum(len(r.entity_matches) for r in results)
        relation_count = sum(len(r.relation_matches) for r in results)
        
        return f"Based on the graph-enhanced context, found {len(results)} relevant documents with {entity_count} matching entities and {relation_count} relevant relations. The query '{query.text}' was processed using graph reasoning."
    
    def _estimate_confidence(self, results: List[GraphRetrievalResult]) -> float:
        """Estimate confidence based on retrieval scores and graph connections."""
        if not results:
            return 0.0
        
        # Average of top retrieval scores
        scores = [r.score for r in results[:3]]
        avg_score = np.mean(scores)
        
        # Boost confidence if there are strong graph connections
        graph_boost = 0.0
        for result in results[:3]:  # Top 3 results
            if result.entity_matches or result.relation_matches:
                graph_boost += 0.1
        
        final_score = min(1.0, avg_score + graph_boost)
        return float(final_score)


# ============================================================
# EXAMPLE USAGE AND TESTING
# ============================================================

def example_usage():
    """Demonstrate Graph-Enhanced RAG usage."""
    
    # Create RAG system
    rag = GraphEnhancedRAG(graph_weight=0.5)
    
    # Sample graph documents
    documents = [
        GraphDocument(
            id="doc1",
            content="John Smith works at Microsoft Corporation. He is a software engineer located in Seattle, Washington.",
            metadata={"source": "employee_directory", "department": "engineering"}
        ),
        GraphDocument(
            id="doc2",
            content="Microsoft Corporation is headquartered in Redmond, Washington. The company was founded by Bill Gates and Paul Allen.",
            metadata={"source": "company_info", "category": "about"}
        ),
        GraphDocument(
            id="doc3",
            content="Seattle is a major city in Washington state. It is known for its tech industry and coffee culture.",
            metadata={"source": "city_guide", "category": "location"}
        )
    ]
    
    # Add documents
    num_docs = rag.add_documents(documents)
    print(f"Added {num_docs} graph-enhanced documents")
    
    # Create a graph query
    query = GraphQuery(
        text="Where does John Smith work?",
        hops=2,  # Allow up to 2 hops in the graph
        include_related=True
    )
    
    # Create a simple query embedding (in practice, this would come from an embedding model)
    import hashlib
    query_text_hash = hashlib.md5(query.text.encode()).hexdigest()
    query_embedding = np.frombuffer(bytes.fromhex(query_text_hash[:32]), dtype=np.float32)
    if len(query_embedding) < 384:
        query_embedding = np.pad(query_embedding, (0, 384 - len(query_embedding)), 'constant')
    elif len(query_embedding) > 384:
        query_embedding = query_embedding[:384]
    
    # Query the system
    result = rag.query(query, query_embedding, k=3)
    
    print(f"\nQuery: {query.text}")
    print(f"Entities mentioned: {len(result.entities_mentioned)}")
    print(f"Relations discovered: {len(result.relations_discovered)}")
    print(f"Reasoning paths: {len(result.reasoning_paths)}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Answer: {result.answer}")
    print(f"Sources: {len(result.sources)}")
    
    # Show another example with more complex reasoning
    print("\n" + "="*60)
    print("Complex Graph Query Example:")
    
    complex_query = GraphQuery(
        text="Who founded the company where John Smith works?",
        hops=3,  # Allow more hops for complex reasoning
        include_related=True
    )
    
    complex_result = rag.query(complex_query, query_embedding, k=2)
    print(f"Query: {complex_query.text}")
    print(f"Entities mentioned: {len(complex_result.entities_mentioned)}")
    print(f"Reasoning paths: {complex_result.reasoning_paths}")
    print(f"Answer: {complex_result.answer}")
    
    return rag


if __name__ == "__main__":
    example_usage()