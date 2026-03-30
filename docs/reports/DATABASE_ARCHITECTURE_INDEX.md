# Database Architecture Reference Index

This index provides a comprehensive overview of all database case studies and system design solutions in the AI-Mastery-2026 project.

## 1. Case Studies Directory Structure

### `docs/06_case_studies/domain_specific/`

| File | Domain | Key Technologies | Focus |
|------|--------|------------------|-------|
| `01_churn_prediction.md` | Churn Prediction | PostgreSQL, Redis, ML models | Customer retention analysis |
| `02_fraud_detection.md` | Fraud Detection | Neo4j, Cassandra, Redis | Real-time fraud detection |
| `03_recommender_system.md` | Recommender Systems | PostgreSQL, MongoDB, Elasticsearch | Personalized recommendations |
| `04_ab_testing_platform.md` | A/B Testing | TimescaleDB, Redis, PostgreSQL | Experimentation platform |
| `05_computer_vision_quality_control.md` | Computer Vision | PostgreSQL, Redis, OpenCV | Quality control systems |
| `06_financial_nlp_analysis.md` | Financial NLP | PostgreSQL, Elasticsearch, ML | Financial document analysis |
| `07_cybersecurity_anomaly_detection.md` | Cybersecurity | TimescaleDB, Neo4j, Redis | Threat detection systems |
| `08_autonomous_vehicle_perception.md` | Autonomous Vehicles | TimescaleDB, Cassandra, Redis | Sensor data processing |
| `09_genomic_sequence_analysis.md` | Genomics | PostgreSQL, MongoDB, Spark | DNA sequence analysis |
| `10_climate_optimized_energy_grid.md` | Energy Grid | TimescaleDB, PostgreSQL, Redis | Smart grid optimization |
| `12_digital_twins_smart_manufacturing.md` | Digital Twins | PostgreSQL, Neo4j, TimescaleDB | Manufacturing simulation |
| `13_molecular_discovery_platform.md` | Molecular Discovery | PostgreSQL, Redis, ML | Drug discovery platforms |
| `14_federated_learning_healthcare.md` | Federated Learning | PostgreSQL, Redis, Homomorphic encryption | Privacy-preserving healthcare AI |
| `15_computer_vision_quality_control.md` | Computer Vision | PostgreSQL, Redis, OpenCV | Quality control (duplicate) |
| `16_financial_nlp_analysis.md` | Financial NLP | PostgreSQL, Elasticsearch, ML | Financial NLP (duplicate) |
| `17_recommender_systems_advanced.md` | Advanced Recommenders | PostgreSQL, Cassandra, Neo4j | Complex recommendation systems |
| `18_energy_demand_forecasting.md` | Energy Forecasting | TimescaleDB, PostgreSQL, ML | Demand prediction systems |
| `19_cybersecurity_anomaly_detection.md` | Cybersecurity | TimescaleDB, Neo4j, Redis | Anomaly detection (duplicate) |
| `20_database_migration_netflix.md` | Database Migration | MySQL, Cassandra, Redis | Netflix migration case study |
| `21_database_migration_uber.md` | Database Migration | Schemaless, Redis, ScyllaDB | Uber migration case study |
| `22_database_migration_capital_one.md` | Database Migration | PostgreSQL, MongoDB, Redis | Capital One migration case study |
| `23_vector_database_notion_ai.md` | Vector Databases | Weaviate, PostgreSQL | Notion AI search implementation |
| `24_vector_database_github_copilot.md` | Vector Databases | pgvector, Elasticsearch | GitHub Copilot RAG implementation |
| `25_federated_learning_healthcare.md` | Federated Learning | PostgreSQL, Redis, Differential privacy | Healthcare consortium case study |
| `26_fraud_detection_database_architecture.md` | Fraud Detection | Neo4j, TimescaleDB, Redis | Graph + Time-Series hybrid architecture |
| `27_recommender_system_database_architecture.md` | Recommender Systems | PostgreSQL, Cassandra, Neo4j, Qdrant | Multi-model recommendation architecture |
| `28_financial_nlp_database_architecture.md` | Financial NLP | TimescaleDB, Qdrant, PostgreSQL | Time-Series + Vector for financial analysis |
| `29_cybersecurity_anomaly_detection_database_architecture.md` | Cybersecurity | TimescaleDB, Neo4j, Redis | Time-Series + Graph for threat detection |
| `30_autonomous_vehicle_perception_database_architecture.md` | Autonomous Vehicles | TimescaleDB, Cassandra, Redis | High-throughput sensor data processing |
| `31_acid_properties_in_practice.md` | Database Fundamentals | PostgreSQL, MySQL, WAL | ACID properties in real-world systems |
| `32_cap_theorem_deep_dive.md` | Database Fundamentals | Cassandra, MongoDB, PostgreSQL | CAP theorem trade-offs for ML systems |
| `33_normalization_denormalization.md` | Database Fundamentals | PostgreSQL, MongoDB, Redis | Data modeling patterns for AI/ML |
| `34_indexing_fundamentals.md` | Database Fundamentals | PostgreSQL, Cassandra, Redis | B-tree, hash, and LSM-tree indexing |

## 2. System Design Solutions Directory Structure

### `docs/03_system_design/solutions/`

| File | Pattern Category | Key Technologies | Use Cases |
|------|------------------|------------------|-----------|
| `database_architecture_patterns_ai.md` | Architecture Patterns | PostgreSQL, Neo4j, Qdrant | AI system database patterns |
| `database_migration_implementation_guide.md` | Migration Strategies | Kafka, PostgreSQL, TimescaleDB | Strangler Fig, blue-green deployment |
| `federated_learning_database_architecture.md` | Federated Learning | PostgreSQL, Redis, Cryptography | Privacy-preserving AI systems |
| `vector_database_integration_rag.md` | Vector Databases | Qdrant, Milvus, PostgreSQL | RAG implementation patterns |
| `time_series_database_patterns_ai.md` | Time-Series Databases | TimescaleDB, InfluxDB, Redis | Sensor data, financial time-series |
| `graph_database_patterns_ai.md` | Graph Databases | Neo4j, JanusGraph, Qdrant | Knowledge graphs, relationship analysis |
| `database_security_compliance_patterns.md` | Security & Compliance | PostgreSQL, Redis, HSM | Zero-trust, multi-tenant isolation |
| `database_cost_optimization_patterns.md` | Cost Optimization | All databases, Cloud services | Storage tiering, query optimization |
| `relational_database_internals_fundamentals.md` | Database Fundamentals | PostgreSQL, MySQL | Relational database internals and ACID |
| `nosql_database_internals_fundamentals.md` | Database Fundamentals | Cassandra, MongoDB | NoSQL distributed architecture fundamentals |
| `time_series_database_architecture_fundamentals.md` | Database Fundamentals | TimescaleDB, InfluxDB | Time-series database internals |
| `vector_database_fundamentals.md` | Database Fundamentals | Qdrant, Milvus, pgvector | Vector search algorithms (HNSW, IVF, PQ) |

## 3. Foundational Database Concepts (Educational Series)

### Core Database Fundamentals Case Studies
- **ACID Properties** (`31_acid_properties_in_practice.md`): Real-world examples of Atomicity, Consistency, Isolation, Durability with failure scenarios
- **CAP Theorem** (`32_cap_theorem_deep_dive.md`): Practical trade-offs between Consistency, Availability, Partition tolerance in ML systems
- **Normalization vs Denormalization** (`33_normalization_denormalization.md`): When to use each pattern with concrete AI/ML examples
- **Indexing Fundamentals** (`34_indexing_fundamentals.md`): B-tree, hash, and LSM-tree indexing strategies with performance characteristics

### Database Internals System Designs
- **Relational Database Internals** (`relational_database_internals_fundamentals.md`): How PostgreSQL/MySQL achieve ACID compliance and handle concurrency
- **NoSQL Database Internals** (`nosql_database_internals_fundamentals.md`): How Cassandra/MongoDB handle distributed operations and consistency
- **Time-Series Database Architecture** (`time_series_database_architecture_fundamentals.md`): How TimescaleDB/InfluxDB optimize for time-based queries
- **Vector Database Fundamentals** (`vector_database_fundamentals.md`): How HNSW, IVF, and PQ algorithms work for approximate nearest neighbor search

## 4. Key Technical Themes Across All Documents

### Architecture Patterns
- **Polyglot Persistence**: Using multiple database types for different workloads
- **Hybrid Architectures**: Combining traditional databases with modern specialized databases
- **Event-Driven Design**: Kafka-based decoupling and real-time processing
- **Multi-Tier Storage**: Hot/warm/cold storage tiering for cost optimization

### Performance Optimization
- **Time-Series Optimization**: Continuous aggregates, compression, retention policies
- **Graph Optimization**: Indexing, path limiting, materialized views
- **Vector Search Optimization**: HNSW parameters, quantization, hybrid search
- **Query Optimization**: Indexing strategies, materialized views, sampling

### AI/ML Integration
- **Feature Engineering**: Database-backed feature stores
- **Real-time Inference**: Low-latency database access for model serving
- **Training Data Preparation**: Efficient extraction of training datasets
- **Feedback Loops**: Database-backed model monitoring and retraining

## 5. Recommended Learning Path

1. **Start with fundamentals**: Read the foundational case studies (31-34) and system designs (relational/nosql internals)
2. **Study specific domains**: Choose case studies relevant to your domain
3. **Implement patterns**: Use system design solutions as templates
4. **Validate with benchmarks**: Compare performance metrics across implementations
5. **Build your own**: Apply patterns to your specific use cases

> 💡 **Pro Tip**: The most valuable learning comes from comparing similar architectures across different domains. Notice how fraud detection (case 26) and cybersecurity (case 29) both use graph+time-series patterns but with different optimizations for their specific requirements.