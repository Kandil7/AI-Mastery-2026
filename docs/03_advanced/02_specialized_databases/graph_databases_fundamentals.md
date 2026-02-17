# Graph Databases Fundamentals

Graph databases are specialized database systems designed for storing and querying highly connected data. They excel at representing relationships between entities, making them ideal for AI/ML applications involving knowledge graphs, recommendation systems, and complex relationship analysis.

## Overview

Graph databases use a property graph model with nodes (entities), relationships (edges), and properties (attributes). For senior AI/ML engineers, understanding graph databases is essential for building applications that require complex relationship analysis and traversal.

## Core Concepts

### Property Graph Model
- **Nodes**: Entities (people, products, concepts)
- **Relationships**: Directed connections between nodes (knows, purchases, related_to)
- **Properties**: Key-value pairs on nodes and relationships
- **Labels**: Node types (Person, Product, Category)

### Query Patterns
- **Pattern matching**: Find subgraphs matching specific patterns
- **Path finding**: Shortest path, all paths between nodes
- **Traversal**: Follow relationships to explore connected data
- **Aggregation**: Count relationships, calculate centrality metrics

## Popular Graph Databases

### Neo4j
- **Architecture**: Native graph storage engine
- **Query language**: Cypher
- **Features**: ACID transactions, clustering, ML integration
- **Use cases**: Knowledge graphs, fraud detection, recommendation engines

### Amazon Neptune
- **Architecture**: Fully managed graph database
- **Query languages**: Gremlin, SPARQL, openCypher
- **Features**: High availability, security, integration with AWS ecosystem
- **Use cases**: Enterprise knowledge graphs, social networks

### JanusGraph
- **Architecture**: Open-source, distributed graph database
- **Query language**: Gremlin
- **Features**: Scalable, pluggable storage backends
- **Use cases**: Large-scale graph analytics, IoT relationship analysis

### ArangoDB
- **Architecture**: Multi-model (graph, document, key-value)
- **Query language**: AQL (ArangoDB Query Language)
- **Features**: Single database for multiple data models
- **Use cases**: Mixed workloads, complex application requirements

## Database Design Patterns

### Schema Design
```cypher
// Neo4j: Create nodes and relationships
CREATE (p1:Person {name: "Alice", age: 30, city: "New York"})
CREATE (p2:Person {name: "Bob", age: 35, city: "Boston"})
CREATE (p3:Person {name: "Charlie", age: 28, city: "New York"})
CREATE (c1:Company {name: "TechCorp", industry: "Technology"})
CREATE (p1)-[:WORKS_AT]->(c1)
CREATE (p2)-[:WORKS_AT]->(c1)
CREATE (p1)-[:FRIENDS_WITH]->(p2)
CREATE (p2)-[:FRIENDS_WITH]->(p3)
CREATE (p1)-[:LIVES_IN]->(:City {name: "New York"})
CREATE (p2)-[:LIVES_IN]->(:City {name: "Boston"})
```

### Indexing Strategies
```cypher
// Neo4j: Create indexes for performance
CREATE INDEX FOR (p:Person) ON (p.name);
CREATE INDEX FOR (p:Person) ON (p.city);
CREATE INDEX FOR ()-[r:WORKS_AT]->() ON (r.start_date);

// Composite indexes for common query patterns
CREATE INDEX FOR (p:Person) ON (p.city, p.age);
CREATE INDEX FOR (p:Product) ON (p.category, p.price);
```

### Query Optimization Patterns
```cypher
// Efficient pattern matching
MATCH (p:Person)-[:FRIENDS_WITH*1..3]->(friend:Person)
WHERE p.name = 'Alice'
RETURN friend.name, friend.age

// Path finding with constraints
MATCH path = shortestPath((p1:Person)-[*..5]-(p2:Person))
WHERE p1.name = 'Alice' AND p2.name = 'Charlie'
RETURN path, length(path)

// Aggregation and analytics
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
WITH c.name as company, count(*) as employee_count
ORDER BY employee_count DESC
RETURN company, employee_count

// Centrality calculation (PageRank)
CALL gds.pageRank.stream({
  nodeProjection: 'Person',
  relationshipProjection: 'FRIENDS_WITH',
  relationshipWeightProperty: 'strength'
})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS person, score
```

## AI/ML Integration Patterns

### Knowledge Graph Construction
- **Entity extraction**: NLP for identifying entities from text
- **Relation extraction**: ML models for identifying relationships
- **Graph embedding**: Generate vector representations of nodes/relationships
- **Link prediction**: Predict missing relationships using ML

### Recommendation Systems
- **Collaborative filtering**: User-item interaction graphs
- **Content-based**: Item similarity graphs
- **Hybrid approaches**: Combine multiple graph types
- **Real-time recommendations**: Graph traversal for personalized suggestions

### Anomaly Detection
- **Structural anomalies**: Unusual connection patterns
- **Behavioral anomalies**: Deviations from normal graph patterns
- **Community detection**: Identify unusual communities or clusters
- **Centrality analysis**: Detect abnormal influence patterns

## Performance Optimization

### Indexing Strategies
- **Node labels**: Fast filtering by node type
- **Property indexes**: For frequently queried properties
- **Composite indexes**: Multiple properties for complex filters
- **Full-text indexes**: For text search on node properties

### Query Optimization
- **Early filtering**: Apply WHERE clauses as early as possible
- **Limit results**: Use LIMIT to prevent excessive data retrieval
- **Avoid Cartesian products**: Be explicit about relationship directions
- **Use EXPLAIN**: Analyze query plans for optimization opportunities

### Scalability Patterns
- **Sharding by relationship type**: Separate storage for different relationship types
- **Horizontal scaling**: Clustered deployments for large graphs
- **Read replicas**: Scale query capacity
- **Edge edges across nodes

 partitioning**: Distribute## Implementation Examples

### Neo4j with Python
```python
from neo4j import GraphDatabase
import pandas as pd

class GraphDB:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def create_person(self, name, age, city):
        with self.driver.session() as session:
            result = session.run(
                "CREATE (p:Person {name: $name, age: $age, city: $city}) RETURN p",
                name=name, age=age, city=city
            )
            return result.single()["p"]
    
    def find_friends_of_friends(self, person_name):
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Person)-[:FRIENDS_WITH*2]->(friend:Person)
                WHERE p.name = $name
                RETURN DISTINCT friend.name, friend.age
                """,
                name=person_name
            )
            return [record.data() for record in result]
    
    def recommend_products(self, user_id):
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (u:User {id: $user_id})-[:PURCHASED]->(p:Product)
                WITH u, collect(p) as purchased
                MATCH (p:Product)-[:SIMILAR_TO]->(similar:Product)
                WHERE p IN purchased
                WITH similar, count(*) as score
                ORDER BY score DESC
                RETURN similar.name, similar.category, score
                LIMIT 10
                """,
                user_id=user_id
            )
            return [record.data() for record in result]
```

### Graph Embeddings with PyTorch Geometric
```python
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import pairwise_distances

# Create graph data structure
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 0],  # source -> target
    [1, 0, 2, 1, 0, 2]   # target -> source
], dtype=torch.long)

x = torch.randn(3, 16)  # 3 nodes, 16-dimensional features

data = Data(x=x, edge_index=edge_index)

# Graph Neural Network for embeddings
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GCN(16, 32, 64)
embeddings = model(data.x, data.edge_index)

# Calculate similarity between nodes
similarity_matrix = 1 - pairwise_distances(embeddings.detach().numpy(), metric='cosine')
```

## Best Practices

1. **Start simple**: Begin with basic graph models before adding complexity
2. **Design for traversal patterns**: Optimize schema for expected query patterns
3. **Monitor query performance**: Use EXPLAIN to identify bottlenecks
4. **Implement proper indexing**: Essential for large graphs
5. **Consider hybrid approaches**: Combine graph with relational for complex queries
6. **Test with realistic data**: Synthetic graphs may not reveal performance issues

## Related Resources

- [Graph Database Patterns for AI/ML](./graph_database_patterns_ai_ml.md) - System design and advanced patterns
- [Database Design](./02_core_concepts/database/database_design.md) - General database design principles
- [Index Optimization](./02_intermediate/02_performance_optimization/01_index_optimization.md) - Advanced indexing techniques
- [AI/ML System Design](./03_advanced/01_ai_ml_integration/) - Graph databases in ML system architecture
