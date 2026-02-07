# Domain-Specific AI Applications

Industry-specific AI implementations with compliance and best practices.

---

## 1. Healthcare AI

### 1.1 Medical Image Classification

**Use Case:** Classify X-ray images for pneumonia detection

**Compliance Requirements:**
- HIPAA (Health Insurance Portability and Accountability Act)
- FDA 510(k) for medical device software
- Data anonymization requirements

**Architecture:**
```
DICOM Image → Anonymization → Preprocessing → CNN Model → Radiologist Review
                    ↓
              Audit Log (HIPAA)
```

**Implementation:**
```python
from src.ml.deep_learning import Conv2D, MaxPool2D, Dense, NeuralNetwork

class MedicalImageClassifier:
    """HIPAA-compliant medical image classifier"""
    
    def __init__(self):
        self.model = self._build_model()
        self.audit_logger = HIPAALogger()
    
    def _build_model(self):
        model = NeuralNetwork()
        model.add(Conv2D(1, 32, kernel_size=3, padding=1))
        model.add(MaxPool2D(2))
        model.add(Conv2D(32, 64, kernel_size=3, padding=1))
        model.add(MaxPool2D(2))
        model.add(Flatten())
        model.add(Dense(64 * 56 * 56, 512))
        model.add(Dense(512, 2))  # Normal vs Pneumonia
        return model
    
    def predict(self, image, patient_id, physician_id):
        # Anonymize before processing
        anon_image = self._anonymize(image)
        
        # Log access (HIPAA requirement)
        self.audit_logger.log_access(
            patient_id=hash(patient_id),
            physician_id=physician_id,
            action="pneumonia_screening"
        )
        
        # Predict with confidence
        result = self.model.predict_proba(anon_image)
        
        # Return with clinical decision support disclaimer
        return {
            'prediction': 'pneumonia' if result[0, 1] > 0.5 else 'normal',
            'confidence': float(max(result[0])),
            'disclaimer': 'This is a clinical decision support tool. '
                         'Final diagnosis must be made by a qualified physician.'
        }
```

**Key Compliance Features:**
```python
class HIPAALogger:
    """Audit logging for HIPAA compliance"""
    
    def log_access(self, patient_id, physician_id, action):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'patient_id_hash': patient_id,  # Never store actual ID
            'physician_id': physician_id,
            'action': action,
            'ip_address': self._get_requester_ip(),
            'retention_days': 2190  # 6 years per HIPAA
        }
        self._store_encrypted(log_entry)
```

---

### 1.2 Clinical NLP

**Use Case:** Extract medical entities from clinical notes

```python
from src.llm.rag import RAGModel

class ClinicalNER:
    """Extract medical entities from clinical text"""
    
    ENTITY_TYPES = [
        'MEDICATION', 'DOSAGE', 'CONDITION', 
        'PROCEDURE', 'ANATOMY', 'LAB_VALUE'
    ]
    
    def extract_entities(self, clinical_note):
        # Use RAG for context-aware extraction
        rag = RAGModel(retriever_strategy='dense')
        rag.add_documents(self.medical_knowledge_base)
        
        prompt = f"""
        Extract medical entities from this clinical note.
        Categories: {', '.join(self.ENTITY_TYPES)}
        
        Note: {clinical_note}
        
        Format: ENTITY_TYPE: value
        """
        
        result = rag.query(prompt)
        return self._parse_entities(result['response'])
```

---

## 2. Financial Services AI

### 2.1 Real-Time Fraud Detection

**Compliance Requirements:**
- PCI DSS for payment data
- SOX for financial reporting
- AML/KYC regulations

**Architecture:**
```
Transaction → Feature Engineering → Fraud Model → Risk Score
      ↓                                    ↓
   Encrypt                          Decision Engine
      ↓                                    ↓
  Token Vault                    Approve/Flag/Block
```

**Implementation:**
```python
from src.ml.classical import SVMScratch, RandomForestScratch

class FraudDetectionSystem:
    """PCI-DSS compliant fraud detection"""
    
    def __init__(self):
        self.models = {
            'svm': SVMScratch(C=10.0, kernel='rbf'),
            'rf': RandomForestScratch(n_estimators=100)
        }
        self.feature_store = FeatureStore()
    
    def score_transaction(self, transaction):
        # Never log raw card numbers (PCI-DSS)
        masked_tx = self._mask_pci_data(transaction)
        
        # Extract features
        features = self.feature_store.get_features([
            'tx_amount_zscore',
            'merchant_risk_score',
            'velocity_24h',
            'distance_from_home',
            'time_since_last_tx'
        ], transaction['user_id'])
        
        # Ensemble prediction
        scores = []
        for name, model in self.models.items():
            score = model.predict_proba(features)[0, 1]
            scores.append(score)
        
        ensemble_score = np.mean(scores)
        
        return {
            'risk_score': ensemble_score,
            'decision': self._make_decision(ensemble_score),
            'explanation': self._explain_decision(features, scores)
        }
    
    def _make_decision(self, score):
        if score > 0.9:
            return 'BLOCK'
        elif score > 0.7:
            return 'FLAG_FOR_REVIEW'
        elif score > 0.5:
            return 'STEP_UP_AUTH'
        else:
            return 'APPROVE'
```

### 2.2 Credit Risk Modeling

```python
class CreditRiskModel:
    """Explainable credit risk model with regulatory compliance"""
    
    def __init__(self):
        self.model = LogisticRegressionScratch()  # Interpretable
        self.feature_importance = {}
    
    def train(self, X, y, feature_names):
        self.model.fit(X, y)
        
        # Store feature importance for explainability
        self.feature_importance = dict(zip(
            feature_names,
            self.model.weights
        ))
    
    def predict_with_explanation(self, applicant_features):
        score = self.model.predict_proba(applicant_features)[0, 1]
        
        # Generate adverse action reasons (required by ECOA)
        adverse_reasons = self._get_adverse_reasons(applicant_features)
        
        return {
            'probability_of_default': score,
            'credit_decision': 'APPROVE' if score < 0.3 else 'DECLINE',
            'adverse_action_reasons': adverse_reasons[:4]  # Top 4 reasons
        }
    
    def _get_adverse_reasons(self, features):
        """Generate explainable reasons for credit decision"""
        contributions = features * self.model.weights
        reason_indices = np.argsort(contributions)[::-1]
        
        reasons = []
        for idx in reason_indices:
            if contributions[idx] > 0:
                reasons.append({
                    'factor': self.feature_names[idx],
                    'impact': 'negative',
                    'contribution': float(contributions[idx])
                })
        return reasons
```

---

## 3. E-Commerce AI

### 3.1 Product Recommendation System

**Architecture:**
```
User Activity → Embedding → Similarity Search → Ranking → Recommendations
      ↓              ↓              ↓
   Clickstream   Item2Vec       HNSW Index
```

**Implementation:**
```python
from src.production.vector_db import HNSWIndex

class RecommendationEngine:
    """Scalable product recommendation system"""
    
    def __init__(self, embedding_dim=128):
        self.item_index = HNSWIndex(dim=embedding_dim)
        self.user_embeddings = {}
        self.item_embeddings = {}
    
    def train_embeddings(self, interactions):
        """Train item and user embeddings using matrix factorization"""
        from src.core.math_operations import svd
        
        # Build interaction matrix
        user_item_matrix = self._build_matrix(interactions)
        
        # SVD decomposition
        U, S, Vt = svd(user_item_matrix, k=128)
        
        # User embeddings: U @ sqrt(S)
        # Item embeddings: sqrt(S) @ Vt
        sqrt_S = np.diag(np.sqrt(S))
        self.user_embeddings = U @ sqrt_S
        self.item_embeddings = (sqrt_S @ Vt).T
        
        # Index items for fast retrieval
        for item_id, embedding in enumerate(self.item_embeddings):
            self.item_index.add(item_id, embedding)
    
    def recommend(self, user_id, n=10, exclude_purchased=True):
        user_embedding = self.user_embeddings[user_id]
        
        # Find similar items
        candidates = self.item_index.search(user_embedding, k=n * 2)
        
        # Filter and rank
        if exclude_purchased:
            purchased = self.get_user_purchases(user_id)
            candidates = [c for c in candidates if c['id'] not in purchased]
        
        return candidates[:n]
```

### 3.2 Dynamic Pricing

```python
class DynamicPricingEngine:
    """ML-based dynamic pricing with business constraints"""
    
    def __init__(self):
        self.demand_model = NeuralNetwork()
        self.elasticity_model = LinearRegressionScratch()
    
    def optimize_price(self, product_id, context):
        """Find optimal price given demand elasticity"""
        
        # Get product constraints
        min_price = self.get_min_price(product_id)  # Cost + margin
        max_price = self.get_max_price(product_id)  # Competitor ceiling
        
        # Estimate demand at different price points
        prices = np.linspace(min_price, max_price, 100)
        revenues = []
        
        for price in prices:
            features = self._build_features(product_id, price, context)
            demand = self.demand_model.predict(features)
            revenue = price * demand
            revenues.append(revenue)
        
        optimal_price = prices[np.argmax(revenues)]
        
        return {
            'recommended_price': optimal_price,
            'expected_demand': self.demand_model.predict(
                self._build_features(product_id, optimal_price, context)
            ),
            'expected_revenue': max(revenues)
        }
```

---

## 4. Compliance Checklist by Industry

### Healthcare (HIPAA)
- [ ] Data encryption at rest and in transit
- [ ] Access audit logging (6-year retention)
- [ ] Minimum necessary data principle
- [ ] Business Associate Agreements
- [ ] Breach notification procedures

### Financial Services (PCI-DSS)
- [ ] Never store full card numbers
- [ ] Encrypt cardholder data
- [ ] Quarterly vulnerability scans
- [ ] Annual penetration testing
- [ ] Access control by role

### E-Commerce (GDPR/CCPA)
- [ ] Consent management
- [ ] Right to deletion
- [ ] Data portability
- [ ] Privacy by design
- [ ] Cross-border transfer compliance

---

## Quick Reference

| Industry | Key Regulation | AI Considerations |
|----------|----------------|-------------------|
| Healthcare | HIPAA, FDA 510(k) | Explainability, audit trails |
| Finance | PCI-DSS, SOX, ECOA | Fairness, adverse action reasons |
| E-Commerce | GDPR, CCPA | Consent, data minimization |
| Insurance | State regulations | Rate fairness, discrimination |
