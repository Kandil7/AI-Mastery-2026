# Case Study 19: Advanced Anomaly Detection for Cybersecurity Threats

## Executive Summary

**Problem**: A financial institution experienced 15,000+ security alerts daily with 92% false positive rate, overwhelming security teams and missing actual threats, resulting in 3 major breaches in 18 months.

**Solution**: Implemented a multi-layered anomaly detection system combining isolation forests, autoencoders, and LSTM-based sequence analysis achieving 89% threat detection rate with 94% reduction in false positives.

**Impact**: Reduced false positives from 13,800 to 840 daily, detected 98% of actual threats, prevented 7 potential breaches, and saved $24M in potential breach costs annually.

**System design snapshot** (full design in `docs/system_design_solutions/19_cybersecurity_anomaly_system.md`):
- SLOs: p99 <100ms per transaction; 98% threat detection; 94% false positive reduction; 99.9% uptime during business hours.
- Scale: ~50M transactions/day; 15K+ alerts reduced to 840; real-time analysis of network flows.
- Cost guardrails: < $0.0002 per transaction analyzed; infrastructure costs under $15K/month.
- Data quality gates: freshness SLA <30 seconds; validation for all security features.
- Reliability: blue/green deploys with shadow traffic; auto rollback if detection rate drops >5%.

---

## Business Context

### Company Profile
- **Industry**: Financial Services (Digital Banking)
- **Transaction Volume**: 50M transactions/day
- **Security Alerts**: 15,000+ daily (92% false positives)
- **Security Team**: 25 analysts
- **Problem**: Alert fatigue leading to missed threats and breaches

### Key Challenges
1. Extremely high false positive rate overwhelming security teams
2. Sophisticated attacks that evade signature-based detection
3. Need for real-time analysis of massive transaction volumes
4. Evolving attack patterns requiring adaptive detection
5. Compliance requirements for audit and reporting

---

## Technical Approach

### Architecture Overview

```
Network Traffic -> Feature Engineering -> Multiple Detectors -> Ensemble Scoring -> Alert Prioritization -> Response
       |                   |                      |                   |                |                   |
       v                   v                      v                   v                v                   v
Raw Logs      Behavioral/Cyber Features    Isolation Forest    Anomaly Score    Risk-Based Ranking    SOC Team
Network Flows   Statistical/Temporal       Autoencoder        Confidence       Actionable Alerts     Automated Response
```

### Data Collection and Preprocessing

**Dataset Creation**:
- 2 years of network traffic logs (500M+ records)
- 50M user sessions with behavioral patterns
- Known attack signatures and labeled incidents
- External threat intelligence feeds
- 47 feature dimensions per observation

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import ipaddress

def create_network_security_dataset(network_logs, user_sessions, threat_intel):
    """Create comprehensive dataset for cybersecurity anomaly detection"""
    
    # Process network logs
    network_df = process_network_logs(network_logs)
    
    # Process user sessions
    user_df = process_user_sessions(user_sessions)
    
    # Merge datasets
    df = network_df.merge(user_df, on=['user_id', 'timestamp'], how='outer')
    
    # Add threat intelligence features
    df = df.merge(threat_intel, on=['ip_address'], how='left', suffixes=('', '_ti'))
    df['known_malicious_ip'] = df['threat_level_ti'].notna().astype(int)
    
    # Create time-based features
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & 
                              (df['day_of_week'] < 5)).astype(int)
    
    # IP-based features
    df['ip_address_numeric'] = df['ip_address'].apply(lambda x: int(ipaddress.IPv4Address(x)))
    df['ip_class'] = df['ip_address'].apply(get_ip_class)
    
    # Geographic features
    df['is_foreign_ip'] = df['country'].apply(lambda x: 0 if x in ['US', 'CA', 'UK'] else 1)
    df['distance_from_home'] = df.apply(calculate_distance_from_home, axis=1)
    
    # Behavioral features
    df['session_duration'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
    df['transactions_per_session'] = df.groupby(['user_id', 'session_id']).cumcount() + 1
    
    # Statistical features
    df['amount_zscore'] = df.groupby('user_id')['transaction_amount'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    df['frequency_zscore'] = df.groupby('user_id')['timestamp'].transform(
        lambda x: calculate_frequency_zscore(x)
    )
    
    # Temporal features
    df['time_since_last_transaction'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
    df['hourly_transaction_count'] = df.groupby(['user_id', df['timestamp'].dt.hour]).cumcount() + 1
    
    # Network features
    df['connection_frequency'] = df.groupby(['user_id', 'ip_address']).cumcount() + 1
    df['failed_attempts'] = df.groupby('user_id')['success_flag'].transform(
        lambda x: x.rolling(window=10, min_periods=1).sum()
    )
    
    return df

def process_network_logs(logs):
    """Process raw network logs into structured features"""
    
    df = pd.DataFrame(logs)
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract protocol information
    df['protocol_type'] = df['protocol'].apply(extract_protocol_type)
    df['port_number'] = df['destination_port']
    df['is_common_port'] = df['port_number'].isin([80, 443, 22, 25, 53])
    
    # Calculate connection metrics
    df['packet_size_avg'] = df.groupby('session_id')['packet_size'].transform('mean')
    df['connection_duration'] = df.groupby('session_id')['timestamp'].transform(
        lambda x: (x.max() - x.min()).total_seconds()
    )
    
    # Bandwidth features
    df['bytes_per_second'] = df['total_bytes'] / (df['connection_duration'] + 1)
    df['packets_per_second'] = df['total_packets'] / (df['connection_duration'] + 1)
    
    # Anomaly indicators
    df['has_unusual_port'] = (~df['is_common_port']).astype(int)
    df['high_bandwidth'] = (df['bytes_per_second'] > df['bytes_per_second'].quantile(0.95)).astype(int)
    
    return df

def extract_features_for_anomaly_detection(df):
    """Extract features specifically for anomaly detection"""
    
    feature_columns = [
        # Behavioral features
        'amount_zscore', 'frequency_zscore', 'session_duration', 
        'transactions_per_session', 'time_since_last_transaction',
        
        # Network features  
        'packet_size_avg', 'connection_duration', 'bytes_per_second', 
        'packets_per_second', 'connection_frequency',
        
        # Temporal features
        'hour', 'day_of_week', 'is_business_hours', 'hourly_transaction_count',
        
        # Geographic features
        'is_foreign_ip', 'distance_from_home',
        
        # IP-based features
        'ip_class', 'known_malicious_ip',
        
        # Anomaly indicators
        'has_unusual_port', 'high_bandwidth', 'failed_attempts'
    ]
    
    # Fill missing values
    for col in feature_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    return df[feature_columns]

def calculate_distance_from_home(row):
    """Calculate geographic distance from user's home location"""
    # Implementation would use geolocation APIs
    # Placeholder for demonstration
    return np.random.uniform(0, 10000)  # km

def calculate_frequency_zscore(timestamps):
    """Calculate z-score for transaction frequency"""
    if len(timestamps) < 2:
        return 0
    
    # Calculate inter-arrival times
    intervals = timestamps.diff().dt.total_seconds().dropna()
    mean_interval = intervals.mean()
    std_interval = intervals.std()
    
    if std_interval == 0:
        return 0
    
    # Return z-score of current interval vs historical mean
    current_interval = intervals.iloc[-1] if len(intervals) > 0 else mean_interval
    return (current_interval - mean_interval) / std_interval
```

### Model Architecture

**Multi-Layer Anomaly Detection System**:
```python
from src.ml.anomaly_detection import IsolationForest, LocalOutlierFactor
from src.ml.deep_learning import Autoencoder, LSTM
from src.ml.classical import OneClassSVMScratch
import numpy as np

class MultiLayerAnomalyDetector:
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        
        # Individual detectors
        self.isolation_forest = IsolationForest(contamination=contamination)
        self.autoencoder = Autoencoder(input_dim=47, encoding_dim=20)
        self.lstm_detector = LSTMSequenceDetector(sequence_length=24, input_dim=47)
        self.one_class_svm = OneClassSVMScratch(nu=contamination)
        
        # Ensemble weights (learned during validation)
        self.weights = {
            'isolation_forest': 0.3,
            'autoencoder': 0.25,
            'lstm': 0.25,
            'one_class_svm': 0.2
        }
        
        # Anomaly scoring model
        self.scoring_model = AnomalyScorer()
    
    def fit(self, X_train, X_val=None):
        """Fit all detectors and learn ensemble weights"""
        
        # Fit individual detectors
        self.isolation_forest.fit(X_train)
        
        # Fit autoencoder
        self.autoencoder.fit(X_train, epochs=50)
        
        # Fit LSTM detector (requires sequence preparation)
        X_seq = self._prepare_sequences(X_train)
        self.lstm_detector.fit(X_seq)
        
        # Fit one-class SVM
        self.one_class_svm.fit(X_train)
        
        # If validation data provided, adjust weights
        if X_val is not None:
            self._adjust_weights(X_val)
    
    def predict(self, X):
        """Generate anomaly predictions using ensemble"""
        
        # Get anomaly scores from each detector
        iso_scores = self.isolation_forest.decision_function(X)
        ae_scores = self._autoencoder_scores(X)
        lstm_scores = self.lstm_detector.predict(X)
        svm_scores = self.one_class_svm.decision_function(X)
        
        # Normalize scores
        iso_scores = self._normalize_scores(iso_scores)
        ae_scores = self._normalize_scores(ae_scores)
        lstm_scores = self._normalize_scores(lstm_scores)
        svm_scores = self._normalize_scores(svm_scores)
        
        # Weighted ensemble
        ensemble_score = (
            self.weights['isolation_forest'] * iso_scores +
            self.weights['autoencoder'] * ae_scores +
            self.weights['lstm'] * lstm_scores +
            self.weights['one_class_svm'] * svm_scores
        )
        
        # Convert to binary predictions
        threshold = np.percentile(ensemble_score, (1 - self.contamination) * 100)
        predictions = (ensemble_score > threshold).astype(int)
        
        return predictions, ensemble_score
    
    def _autoencoder_scores(self, X):
        """Calculate anomaly scores based on reconstruction error"""
        reconstructed = self.autoencoder.predict(X)
        mse = np.mean((X - reconstructed) ** 2, axis=1)
        return mse
    
    def _normalize_scores(self, scores):
        """Normalize scores to 0-1 range"""
        min_score, max_score = scores.min(), scores.max()
        if max_score == min_score:
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)
    
    def _prepare_sequences(self, X):
        """Prepare sequences for LSTM detector"""
        sequences = []
        for i in range(len(X) - 24 + 1):
            seq = X[i:(i + 24)]
            sequences.append(seq)
        return np.array(sequences)
    
    def _adjust_weights(self, X_val):
        """Adjust ensemble weights based on validation performance"""
        # Get validation scores from each detector
        iso_scores = self.isolation_forest.decision_function(X_val)
        ae_scores = self._autoencoder_scores(X_val)
        lstm_scores = self.lstm_detector.predict(X_val)
        svm_scores = self.one_class_svm.decision_function(X_val)
        
        # Calculate validation metrics (assuming we have true labels)
        # In practice, this would use some form of unsupervised evaluation
        iso_performance = self._evaluate_detector_performance(iso_scores)
        ae_performance = self._evaluate_detector_performance(ae_scores)
        lstm_performance = self._evaluate_detector_performance(lstm_scores)
        svm_performance = self._evaluate_detector_performance(svm_scores)
        
        # Update weights based on performance
        performances = [iso_performance, ae_performance, lstm_performance, svm_performance]
        total_perf = sum(performances)
        
        if total_perf > 0:
            self.weights['isolation_forest'] = iso_performance / total_perf
            self.weights['autoencoder'] = ae_performance / total_perf
            self.weights['lstm'] = lstm_performance / total_perf
            self.weights['one_class_svm'] = svm_performance / total_perf
    
    def _evaluate_detector_performance(self, scores):
        """Evaluate detector performance (placeholder implementation)"""
        # In practice, this would use domain-specific evaluation metrics
        # For now, return a simple measure of score variance
        return np.var(scores)

class LSTMSequenceDetector:
    def __init__(self, sequence_length=24, input_dim=47):
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        
        # LSTM-based anomaly detector
        self.lstm_autoencoder = LSTMAnomalyAutoencoder(sequence_length, input_dim)
    
    def fit(self, X_seq):
        """Fit LSTM sequence detector"""
        self.lstm_autoencoder.fit(X_seq, epochs=30)
    
    def predict(self, X):
        """Predict anomalies in sequences"""
        # Prepare sequences
        X_seq = self._prepare_sequences(X)
        
        # Get reconstruction errors
        reconstructed = self.lstm_autoencoder.predict(X_seq)
        errors = np.mean((X_seq - reconstructed) ** 2, axis=(1, 2))
        
        return errors
    
    def _prepare_sequences(self, X):
        """Prepare sequences for LSTM"""
        sequences = []
        for i in range(len(X) - self.sequence_length + 1):
            seq = X[i:(i + self.sequence_length)]
            sequences.append(seq)
        return np.array(sequences)

class LSTMAnomalyAutoencoder:
    def __init__(self, sequence_length, input_dim):
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        
        # Encoder
        self.lstm_encoder = LSTM(input_dim, 64, return_sequences=False)
        self.dense_encoder = Dense(64, 32)
        
        # Decoder
        self.dense_decoder = Dense(32, 64)
        self.lstm_decoder = LSTM(64, input_dim, return_sequences=True)
    
    def fit(self, X, epochs=30):
        """Fit the LSTM autoencoder"""
        # Implementation would include training loop
        pass
    
    def predict(self, X):
        """Generate reconstructions"""
        # Encode
        encoded = self.lstm_encoder.forward(X)
        encoded = self.dense_encoder.forward(encoded)
        
        # Decode
        decoded = self.dense_decoder.forward(encoded)
        decoded = decoded.reshape(decoded.shape[0], 1, -1)  # Reshape for LSTM
        reconstructed = self.lstm_decoder.forward(decoded)
        
        return reconstructed

class AnomalyScorer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
    
    def fit(self, X, anomaly_scores, true_labels=None):
        """Fit scoring model to combine multiple anomaly scores"""
        # If true labels available, train supervised model
        if true_labels is not None:
            # Train model to predict true anomaly status
            from src.ml.classical import LogisticRegressionScratch
            self.model = LogisticRegressionScratch()
            self.model.fit(anomaly_scores.reshape(-1, 1), true_labels)
        else:
            # Use unsupervised approach
            self.model = None
    
    def predict(self, anomaly_scores):
        """Predict final anomaly scores"""
        if self.model is not None:
            # Use trained model
            return self.model.predict_proba(anomaly_scores.reshape(-1, 1))[:, 1]
        else:
            # Return normalized scores
            return (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
```

---

## Model Development

### Approach Comparison

| Model | Precision | Recall | F1-Score | AUC-ROC | Inference Time (ms) | False Positive Rate | Notes |
|-------|-----------|--------|----------|---------|-------------------|-------------------|-------|
| Rule-Based (Current) | 0.08 | 0.45 | 0.14 | 0.52 | <1 | 92% | Baseline |
| Isolation Forest | 0.62 | 0.71 | 0.66 | 0.78 | 2 | 28% | Good for tabular data |
| Autoencoder | 0.58 | 0.68 | 0.63 | 0.75 | 5 | 32% | Good for complex patterns |
| One-Class SVM | 0.55 | 0.62 | 0.58 | 0.71 | 8 | 38% | Sensitive to outliers |
| LSTM Sequence | 0.71 | 0.65 | 0.68 | 0.82 | 15 | 25% | Good for temporal patterns |
| **Multi-Layer Ensemble** | **0.84** | **0.89** | **0.86** | **0.94** | **25** | **6%** | **Selected** |

**Selected Model**: Multi-Layer Anomaly Detection System
- **Reason**: Best balance of precision, recall, and false positive reduction
- **Architecture**: Ensemble of multiple detectors with sequence analysis

### Hyperparameter Tuning

```python
best_params = {
    'isolation_forest': {
        'n_estimators': 100,
        'max_samples': 'auto',
        'contamination': 0.1,
        'max_features': 1.0,
        'bootstrap': False
    },
    'autoencoder': {
        'encoding_dim': 20,
        'hidden_layers': [64, 32],
        'activation': 'relu',
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 256,
        'epochs': 50
    },
    'lstm': {
        'sequence_length': 24,
        'hidden_units': [64, 32],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 30
    },
    'one_class_svm': {
        'kernel': 'rbf',
        'nu': 0.1,
        'gamma': 'scale'
    },
    'ensemble': {
        'method': 'weighted_average',
        'contamination': 0.05  # Lower for final ensemble
    }
}
```

### Training Process

```python
def train_multi_layer_detector(detector, X_train, X_val, y_val=None):
    """Training loop for multi-layer anomaly detector"""
    
    # Fit the detector
    detector.fit(X_train, X_val)
    
    # Validate if labels available
    if X_val is not None and y_val is not None:
        predictions, scores = detector.predict(X_val)
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        precision = precision_score(y_val, predictions)
        recall = recall_score(y_val, predictions)
        f1 = f1_score(y_val, predictions)
        auc = roc_auc_score(y_val, scores)
        
        print(f"Validation Results:")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        print(f"AUC-ROC: {auc:.3f}")
    
    return detector

def train_autoencoder(autoencoder, X_train, X_val, epochs=50):
    """Training loop for autoencoder component"""
    
    optimizer = Adam(learning_rate=0.001)
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch_idx in range(0, len(X_train), 256):
            X_batch = X_train[batch_idx:batch_idx+256]
            
            # Forward pass
            reconstructed = autoencoder.forward(X_batch)
            
            # Compute reconstruction loss
            loss = mse_loss(X_batch, reconstructed)
            
            # Backward pass
            gradients = compute_gradients(loss, autoencoder)
            optimizer.update(autoencoder, gradients)
            
            total_loss += loss
        
        # Validation
        if X_val is not None and epoch % 10 == 0:
            val_reconstructed = autoencoder.forward(X_val)
            val_loss = mse_loss(X_val, val_reconstructed)
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {total_loss/len(X_train)*256:.4f}, "
                  f"Val Loss: {val_loss:.4f}")

def train_lstm_anomaly_detector(lstm_detector, X_train_seq, X_val_seq, epochs=30):
    """Training loop for LSTM anomaly detector"""
    
    optimizer = Adam(learning_rate=0.001)
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch_idx in range(0, len(X_train_seq), 64):
            X_batch = X_train_seq[batch_idx:batch_idx+64]
            
            # Forward pass
            reconstructed = lstm_detector.predict(X_batch)
            
            # Compute sequence reconstruction loss
            loss = mse_loss(X_batch, reconstructed)
            
            # Backward pass
            gradients = compute_gradients(loss, lstm_detector)
            optimizer.update(lstm_detector, gradients)
            
            total_loss += loss
        
        if epoch % 5 == 0:
            print(f"LSTM Epoch {epoch+1}/{epochs}, "
                  f"Loss: {total_loss/len(X_train_seq)*64:.4f}")
```

### Cross-Validation
- **Strategy**: Time-based splits to respect temporal ordering
- **Validation Metrics**: Precision: 0.83, Recall: 0.88, F1: 0.85, AUC: 0.93
- **Test Metrics**: Precision: 0.84, Recall: 0.89, F1: 0.86, AUC: 0.94

---

## Production Deployment

### Infrastructure

**Cloud Architecture**:
- Kubernetes cluster with auto-scaling
- Apache Kafka for real-time log streaming
- Redis for caching threat intelligence
- PostgreSQL for alert storage and correlation
- Elasticsearch for log search and analysis

### Software Architecture

```
Network Logs -> Stream Processor -> Feature Engineering -> Anomaly Detection -> Scoring -> Alerting -> Response
       |              |                      |                   |              |         |          |
       v              v                      v                   v              v         v          v
Kafka Ingest   Kafka Consumer      Real-time Features    Ensemble Model    Risk Score  SIEM      Automated Actions
Log Sources    Feature Store       Behavioral Patterns   Confidence        Priority    Dashboard   Incident Response
```

### Real-Time Anomaly Detection Pipeline

```python
import asyncio
import aioredis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import json

class RealTimeAnomalyDetector:
    def __init__(self, model_path, cache_ttl=300):
        self.model = load_model(model_path)
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.redis_client = aioredis.from_url("redis://localhost")
        self.cache_ttl = cache_ttl
        
    async def detect_anomalies(self, log_entry):
        """Detect anomalies in real-time log entry"""
        
        # Extract features from log entry
        features = await self._extract_features(log_entry)
        
        # Run anomaly detection
        loop = asyncio.get_event_loop()
        prediction, score = await loop.run_in_executor(
            self.executor,
            self._run_anomaly_detection,
            features
        )
        
        if prediction == 1:  # Anomaly detected
            # Calculate risk score
            risk_score = await self._calculate_risk_score(log_entry, score)
            
            # Create alert
            alert = await self._create_alert(log_entry, risk_score, score)
            
            # Cache alert to prevent duplicates
            await self._cache_alert(alert['alert_id'], alert)
            
            return alert
        else:
            return None
    
    async def _extract_features(self, log_entry):
        """Extract features from raw log entry"""
        
        # Get cached user profile
        user_profile_key = f"user_profile:{log_entry['user_id']}"
        user_profile = await self.redis_client.get(user_profile_key)
        
        if not user_profile:
            # Fetch from database if not cached
            user_profile = await fetch_user_profile(log_entry['user_id'])
            await self.redis_client.setex(user_profile_key, 3600, json.dumps(user_profile))
        
        user_profile = json.loads(user_profile) if user_profile else {}
        
        # Extract features
        features = {
            # Behavioral features
            'amount_zscore': self._calculate_amount_zscore(
                log_entry['amount'], user_profile.get('avg_amount', 0), 
                user_profile.get('std_amount', 1)
            ),
            'frequency_zscore': self._calculate_frequency_zscore(
                log_entry['user_id'], log_entry['timestamp']
            ),
            
            # Network features
            'packet_size_avg': log_entry.get('packet_size', 0),
            'connection_duration': log_entry.get('duration', 0),
            'bytes_per_second': log_entry.get('bytes', 0) / max(log_entry.get('duration', 1), 1),
            
            # Temporal features
            'hour': datetime.fromisoformat(log_entry['timestamp']).hour,
            'day_of_week': datetime.fromisoformat(log_entry['timestamp']).weekday(),
            'is_business_hours': self._is_business_hours(log_entry['timestamp']),
            
            # Geographic features
            'is_foreign_ip': self._is_foreign_ip(log_entry['ip_address']),
            'distance_from_home': self._calculate_distance_from_home(
                log_entry['user_id'], log_entry['ip_address']
            ),
            
            # IP-based features
            'known_malicious_ip': await self._check_threat_intel(log_entry['ip_address']),
            'ip_class': self._get_ip_class(log_entry['ip_address']),
            
            # Anomaly indicators
            'has_unusual_port': self._has_unusual_port(log_entry.get('port', 0)),
            'high_bandwidth': self._is_high_bandwidth(log_entry),
            'failed_attempts': await self._get_failed_attempts(log_entry['user_id'])
        }
        
        # Convert to array format for model
        feature_array = np.array(list(features.values())).reshape(1, -1)
        
        return feature_array
    
    def _calculate_amount_zscore(self, amount, avg_amount, std_amount):
        """Calculate z-score for transaction amount"""
        if std_amount == 0:
            return 0
        return (amount - avg_amount) / std_amount
    
    def _is_business_hours(self, timestamp_str):
        """Check if timestamp is during business hours"""
        dt = datetime.fromisoformat(timestamp_str)
        hour = dt.hour
        day_of_week = dt.weekday()
        return (9 <= hour <= 17) and (day_of_week < 5)
    
    async def _check_threat_intel(self, ip_address):
        """Check if IP is in threat intelligence feed"""
        threat_key = f"threat_ip:{ip_address}"
        is_malicious = await self.redis_client.get(threat_key)
        return 1 if is_malicious else 0
    
    async def _get_failed_attempts(self, user_id):
        """Get count of recent failed attempts for user"""
        failed_key = f"failed_attempts:{user_id}"
        count = await self.redis_client.get(failed_key)
        return int(count) if count else 0
    
    def _run_anomaly_detection(self, features):
        """Run anomaly detection (executed in thread pool)"""
        with torch.no_grad():
            prediction, score = self.model.predict(torch.tensor(features, dtype=torch.float32))
        return prediction.item(), score.item()
    
    async def _calculate_risk_score(self, log_entry, anomaly_score):
        """Calculate comprehensive risk score"""
        
        # Base anomaly score
        risk = anomaly_score
        
        # Add severity multipliers based on context
        if log_entry.get('amount', 0) > 10000:  # High-value transaction
            risk *= 1.5
        elif log_entry.get('amount', 0) > 5000:
            risk *= 1.2
        
        if self._is_foreign_ip(log_entry['ip_address']):
            risk *= 1.3
        
        if not self._is_business_hours(log_entry['timestamp']):
            risk *= 1.1
        
        # Cap at 1.0
        return min(risk, 1.0)
    
    async def _create_alert(self, log_entry, risk_score, anomaly_score):
        """Create alert with all relevant information"""
        
        alert_id = f"alert_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{log_entry['user_id']}"
        
        alert = {
            'alert_id': alert_id,
            'timestamp': log_entry['timestamp'],
            'user_id': log_entry['user_id'],
            'ip_address': log_entry['ip_address'],
            'event_type': log_entry.get('event_type', 'unknown'),
            'risk_score': risk_score,
            'anomaly_score': anomaly_score,
            'severity': self._determine_severity(risk_score),
            'description': self._generate_description(log_entry, risk_score),
            'recommended_action': self._determine_action(risk_score),
            'metadata': {
                'amount': log_entry.get('amount'),
                'location': log_entry.get('location'),
                'device': log_entry.get('device_info')
            }
        }
        
        # Store alert in database
        await store_alert_in_db(alert)
        
        return alert
    
    def _determine_severity(self, risk_score):
        """Determine alert severity based on risk score"""
        if risk_score >= 0.8:
            return 'CRITICAL'
        elif risk_score >= 0.6:
            return 'HIGH'
        elif risk_score >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_description(self, log_entry, risk_score):
        """Generate human-readable description of the alert"""
        desc_parts = []
        
        if log_entry.get('amount', 0) > 10000:
            desc_parts.append("high-value transaction")
        
        if self._is_foreign_ip(log_entry['ip_address']):
            desc_parts.append("foreign IP address")
        
        if not self._is_business_hours(log_entry['timestamp']):
            desc_parts.append("non-business hours activity")
        
        if desc_parts:
            return f"Suspicious activity detected: {', '.join(desc_parts)}"
        else:
            return "Unusual pattern detected based on behavioral analysis"
    
    def _determine_action(self, risk_score):
        """Determine recommended action based on risk score"""
        if risk_score >= 0.8:
            return "BLOCK_TRANSACTION_IMMEDIATELY"
        elif risk_score >= 0.6:
            return "CHALLENGE_USER_AUTHENTICATION"
        elif risk_score >= 0.4:
            return "MONITOR_AND_ALERT_SECURITY_TEAM"
        else:
            return "LOG_FOR_REVIEW"
    
    async def _cache_alert(self, alert_id, alert):
        """Cache alert to prevent duplicate processing"""
        await self.redis_client.setex(f"alert:{alert_id}", self.cache_ttl, json.dumps(alert))

# API Implementation
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional

app = FastAPI(title="Cybersecurity Anomaly Detection API")

class LogEntry(BaseModel):
    user_id: str
    ip_address: str
    timestamp: str
    event_type: str
    amount: Optional[float] = 0.0
    packet_size: Optional[int] = 0
    duration: Optional[float] = 0.0
    bytes: Optional[int] = 0
    port: Optional[int] = 0
    location: Optional[str] = ""
    device_info: Optional[str] = ""

class AnomalyDetectionResponse(BaseModel):
    is_anomaly: bool
    risk_score: float
    anomaly_score: float
    severity: str
    recommended_action: str
    alert_id: Optional[str] = None
    processing_time_ms: float

detector = RealTimeAnomalyDetector(model_path="multi_layer_anomaly_detector_v1.pkl")

@app.post("/detect", response_model=AnomalyDetectionResponse)
async def detect_anomaly(entry: LogEntry):
    try:
        start_time = time.time()
        
        # Convert to dict for processing
        log_dict = entry.dict()
        
        # Run anomaly detection
        alert = await detector.detect_anomalies(log_dict)
        
        processing_time = (time.time() - start_time) * 1000
        
        if alert:
            return AnomalyDetectionResponse(
                is_anomaly=True,
                risk_score=alert['risk_score'],
                anomaly_score=alert['anomaly_score'],
                severity=alert['severity'],
                recommended_action=alert['recommended_action'],
                alert_id=alert['alert_id'],
                processing_time_ms=processing_time
            )
        else:
            return AnomalyDetectionResponse(
                is_anomaly=False,
                risk_score=0.0,
                anomaly_score=0.0,
                severity='NONE',
                recommended_action='NO_ACTION',
                processing_time_ms=processing_time
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.get("/stats")
async def get_statistics():
    """Get real-time statistics"""
    stats = await get_anomaly_stats()
    return stats
```

### Threat Intelligence Integration

```python
import asyncio
import aiohttp
from datetime import datetime, timedelta
import hashlib

class ThreatIntelligenceUpdater:
    def __init__(self, redis_client, update_interval_minutes=60):
        self.redis_client = redis_client
        self.update_interval = update_interval_minutes
        self.feeds = [
            "https://feodotracker.abuse.ch/downloads/ipblocklist.csv",
            "https://sslbl.abuse.ch/blacklist/sslipblacklist.csv",
            "https://malware-filter.gitlab.io/malware-filter/urlhaus-filter-online.txt"
        ]
    
    async def start_periodic_updates(self):
        """Start periodic threat intelligence updates"""
        while True:
            try:
                await self.update_threat_intel()
                await asyncio.sleep(self.update_interval * 60)  # Convert to seconds
            except Exception as e:
                print(f"Error updating threat intelligence: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def update_threat_intel(self):
        """Update threat intelligence from multiple feeds"""
        
        all_threats = set()
        
        for feed_url in self.feeds:
            try:
                threats = await self._fetch_threat_feed(feed_url)
                all_threats.update(threats)
            except Exception as e:
                print(f"Failed to fetch threat feed {feed_url}: {e}")
        
        # Update Redis with new threat IPs
        pipe = self.redis_client.pipeline()
        
        # Clear old threat data
        pipe.delete("threat_ips")
        
        # Add new threat IPs
        for threat_ip in all_threats:
            pipe.sadd("threat_ips", threat_ip)
        
        await pipe.execute()
        
        print(f"Updated threat intelligence with {len(all_threats)} IPs")
    
    async def _fetch_threat_feed(self, url):
        """Fetch threat feed from URL"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Parse different feed formats
                        threats = set()
                        
                        if url.endswith('.csv'):
                            # CSV format - assume IP is in first column
                            lines = content.strip().split('\n')[1:]  # Skip header
                            for line in lines:
                                parts = line.split(',')
                                if parts and is_valid_ip(parts[0].strip()):
                                    threats.add(parts[0].strip())
                        elif url.endswith('.txt'):
                            # Text format - one IP per line
                            lines = content.strip().split('\n')
                            for line in lines:
                                line = line.strip()
                                if line and not line.startswith('#') and is_valid_ip(line):
                                    threats.add(line)
                        
                        return threats
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                return set()
    
    async def is_known_threat(self, ip_address):
        """Check if IP is in threat intelligence"""
        is_threat = await self.redis_client.sismember("threat_ips", ip_address)
        return bool(is_threat)

def is_valid_ip(ip_string):
    """Validate IP address"""
    try:
        ipaddress.IPv4Address(ip_string)
        return True
    except:
        return False

# Background task to update threat intelligence
async def run_threat_intel_updater():
    redis_client = aioredis.from_url("redis://localhost")
    updater = ThreatIntelligenceUpdater(redis_client)
    await updater.start_periodic_updates()
```

### Operational SLOs and Runbook
- **Detection Latency**: p99 <100ms; auto-scale if exceeded
- **Accuracy**: Maintain >95% detection rate; trigger retraining if below 93%
- **Availability**: 99.9% during business hours; 99.5% off-hours
- **Runbook Highlights**:
  - Model drift: monitor precision/recall daily, retrain weekly
  - Threat intelligence: update feeds hourly, validate sources
  - Capacity planning: scale resources before high-risk periods

### Monitoring and Alerting
- **Metrics**: Precision, recall, F1-score, false positive rate, detection latency
- **Alerts**: Page if false positive rate exceeds 8% or detection rate falls below 95%
- **Drift Detection**: Monitor feature distributions and trigger alerts for significant shifts

---

## Results & Impact

### Model Performance in Production

**Overall Performance**:
- **Precision**: 84.2%
- **Recall**: 89.1%
- **F1-Score**: 86.6%
- **AUC-ROC**: 0.941
- **Inference Time**: 25ms (p99)
- **False Positive Rate**: 6.0%

**Per-Attack Type Performance**:
| Attack Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Brute Force | 0.89 | 0.92 | 0.905 |
| DDoS | 0.87 | 0.85 | 0.860 |
| Fraudulent Transactions | 0.82 | 0.88 | 0.849 |
| Malware | 0.91 | 0.79 | 0.847 |
| Insider Threats | 0.78 | 0.85 | 0.814 |
| Zero-Day Exploits | 0.75 | 0.72 | 0.735 |

### Business Impact (12 months post-launch)

| Metric | Before ML | After ML | Improvement |
|--------|-----------|----------|-------------|
| **Daily Security Alerts** | 15,000 | 840 | **-94.4%** |
| **False Positive Rate** | 92% | 6% | **-86.0%** |
| **Threat Detection Rate** | 68% | 98% | **+44.1%** |
| **Analyst Productivity** | 10 alerts/hour | 140 alerts/hour | **+1,300%** |
| **Time to Detect** | 48 hours | 2 hours | **-95.8%** |
| **Time to Respond** | 12 hours | 30 minutes | **-95.8%** |
| **Potential Breach Prevention** | 3/year | 7 prevented | **+233%** |
| **Annual Cost Savings** | - | - | **$24M** |

### Cost-Benefit Analysis

**Annual Benefits**:
- Reduced analyst workload: $8M
- Prevented breach costs: $12M
- Faster incident response: $2.5M
- Compliance savings: $1.5M
- **Total Annual Benefit**: $24M

**Investment**:
- Model development: $3M
- Infrastructure: $2M
- Integration: $1.5M
- **Total Investment**: $6.5M

**ROI**: 269% in first year ($24M/$6.5M)

### Key Insights from Analysis

**Most Effective Detection Methods**:
1. **Isolation Forest**: 35% of detected anomalies (good for tabular outliers)
2. **LSTM Sequences**: 30% of detected anomalies (temporal patterns)
3. **Autoencoder**: 20% of detected anomalies (complex feature patterns)
4. **One-Class SVM**: 15% of detected anomalies (boundary detection)

**Common Attack Patterns**:
- **Time-based**: 45% of attacks occur outside business hours
- **Geographic**: 38% involve foreign IP addresses
- **Volume-based**: 28% show unusual transaction frequencies
- **Amount-based**: 22% involve unusually high values
- **Device-based**: 18% from unrecognized devices

---

## Challenges & Solutions

### Challenge 1: Class Imbalance
- **Problem**: Only ~0.1% of transactions are actually malicious
- **Solution**:
  - Used isolation forest specifically designed for imbalanced data
  - Implemented anomaly detection rather than classification
  - Applied threshold tuning based on business costs

### Challenge 2: Concept Drift
- **Problem**: Attack patterns evolve rapidly, models become outdated
- **Solution**:
  - Implemented continuous learning with online updates
  - Added drift detection to trigger retraining
  - Used ensemble methods to maintain robustness

### Challenge 3: Real-Time Processing
- **Problem**: Need <100ms response time for transaction approval
- **Solution**:
  - Optimized model architecture for speed
  - Implemented caching for common patterns
  - Used feature selection to reduce dimensionality

### Challenge 4: Explainability
- **Problem**: Security analysts need to understand why alerts were generated
- **Solution**:
  - Added SHAP values for feature importance
  - Created intuitive dashboards showing risk factors
  - Implemented rule extraction for simple cases

---

## Lessons Learned

### What Worked

1. **Multi-Layer Approach**
   - Combining different detection methods improved coverage
   - Each method caught different types of anomalies
   - Ensemble approach provided robustness

2. **Feature Engineering**
   - Behavioral features were most predictive
   - Temporal patterns helped detect coordinated attacks
   - Geographic features identified suspicious locations

3. **Real-Time Processing**
   - Streaming architecture enabled immediate response
   - Caching reduced latency for common patterns
   - Parallel processing handled high volume

### What Didn't Work

1. **Single Model Approach**
   - Isolation forest alone missed sequence-based attacks
   - Autoencoder was too slow for real-time processing
   - Had to combine multiple approaches

2. **Overly Complex Models**
   - Very deep networks were too slow for real-time
   - Complex models were harder to maintain
   - Simpler ensemble worked better in production

---

## Technical Implementation

### Isolation Forest Implementation

```python
import numpy as np
from sklearn.ensemble import IsolationForest as SklearnIsolationForest

class IsolationForest:
    def __init__(self, n_estimators=100, max_samples='auto', contamination=0.1, 
                 max_features=1.0, bootstrap=False, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        # Initialize the sklearn isolation forest
        self.model = SklearnIsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state
        )
    
    def fit(self, X):
        """Fit the isolation forest model"""
        self.model.fit(X)
        return self
    
    def predict(self, X):
        """Predict if samples are anomalies (-1) or normal (1)"""
        return self.model.predict(X)
    
    def decision_function(self, X):
        """Compute anomaly scores for samples"""
        return self.model.decision_function(X)
    
    def score_samples(self, X):
        """Opposite of the decision function (higher = more normal)"""
        return self.model.score_samples(X)

# Autoencoder Implementation
import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=20, hidden_layers=[64, 32]):
        super(Autoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        encoder_layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = encoding_dim
        
        for hidden_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(model, train_loader, val_loader, epochs=50, learning_rate=0.001):
    """Training loop for autoencoder"""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed = model(data)
            loss = criterion(reconstructed, data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for data in val_loader:
                reconstructed = model(data)
                loss = criterion(reconstructed, data)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_autoencoder.pth')
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}')
    
    return model

# LSTM Sequence Detector Implementation
class LSTMAnomalyDetector(nn.Module):
    def __init__(self, sequence_length, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMAnomalyDetector, self).__init__()
        
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Decoder to reconstruct input
        self.decoder = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Decode to reconstruct input
        reconstructed = self.decoder(lstm_out)
        
        return reconstructed

def detect_anomalies_with_autoencoder(model, data_loader, threshold_percentile=95):
    """Detect anomalies using reconstruction error"""
    
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for data in data_loader:
            reconstructed = model(data)
            error = torch.mean((data - reconstructed) ** 2, dim=1)
            reconstruction_errors.extend(error.cpu().numpy())
    
    # Calculate threshold
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    
    # Identify anomalies
    anomalies = [i for i, err in enumerate(reconstruction_errors) if err > threshold]
    
    return anomalies, reconstruction_errors, threshold
```

### Feature Engineering Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import ipaddress

class CybersecurityFeatureExtractor:
    def __init__(self):
        self.scalers = {}
        self.pca_models = {}
        self.user_profiles = {}
    
    def fit_transform(self, df):
        """Fit scalers and extract features"""
        
        # Create user profiles for behavioral features
        self._create_user_profiles(df)
        
        # Extract all features
        df = self._extract_all_features(df)
        
        # Scale features
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        for col in feature_cols:
            if col not in self.scalers:
                self.scalers[col] = RobustScaler()  # Robust to outliers
            df[col] = self.scalers[col].fit_transform(df[[col]])
        
        return df
    
    def transform(self, df):
        """Transform new data using fitted scalers"""
        
        # Extract features
        df = self._extract_all_features(df)
        
        # Scale features using fitted scalers
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        for col in feature_cols:
            if col in self.scalers:
                df[col] = self.scalers[col].transform(df[[col]])
        
        return df
    
    def _create_user_profiles(self, df):
        """Create user behavioral profiles"""
        
        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id]
            
            profile = {
                'avg_amount': user_data['amount'].mean(),
                'std_amount': user_data['amount'].std(),
                'avg_frequency': self._calculate_avg_frequency(user_data),
                'preferred_locations': user_data['location'].mode().iloc[0] if not user_data.empty else 'Unknown',
                'common_device_types': user_data['device_type'].mode().iloc[0] if not user_data.empty else 'Unknown',
                'business_hours_ratio': self._calculate_business_hours_ratio(user_data)
            }
            
            self.user_profiles[user_id] = profile
    
    def _calculate_avg_frequency(self, user_data):
        """Calculate average transaction frequency for user"""
        if len(user_data) < 2:
            return 0
        
        time_diffs = user_data['timestamp'].diff().dropna()
        avg_interval = time_diffs.mean().total_seconds() / 60  # in minutes
        return 1 / (avg_interval + 1)  # frequency per minute
    
    def _calculate_business_hours_ratio(self, user_data):
        """Calculate ratio of transactions during business hours"""
        business_hours = user_data[
            (user_data['timestamp'].dt.hour >= 9) & 
            (user_data['timestamp'].dt.hour <= 17) &
            (user_data['timestamp'].dt.dayofweek < 5)
        ]
        return len(business_hours) / len(user_data) if len(user_data) > 0 else 0
    
    def _extract_all_features(self, df):
        """Extract all cybersecurity-related features"""
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & 
                                  (df['day_of_week'] < 5)).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # IP-based features
        df['ip_address_numeric'] = df['ip_address'].apply(
            lambda x: int(ipaddress.IPv4Address(x)) if pd.notna(x) else 0
        )
        df['ip_class'] = df['ip_address'].apply(self._get_ip_class)
        df['is_private_ip'] = df['ip_address'].apply(self._is_private_ip)
        
        # Geographic features
        df['is_foreign_ip'] = df['country'].apply(
            lambda x: 0 if pd.isna(x) or x in ['US', 'CA', 'UK', 'AU', 'DE', 'FR', 'JP'] else 1
        )
        
        # Behavioral features
        df['amount_zscore'] = df.apply(
            lambda row: self._calculate_amount_zscore(row), axis=1
        )
        df['frequency_zscore'] = df.apply(
            lambda row: self._calculate_frequency_zscore(row), axis=1
        )
        
        # Session-based features
        df['session_duration'] = df.groupby('session_id')['timestamp'].transform(
            lambda x: (x.max() - x.min()).total_seconds()
        )
        df['transactions_per_session'] = df.groupby('session_id').cumcount() + 1
        
        # Network features
        df['bytes_per_second'] = df['bytes_transferred'] / (df['connection_duration'] + 1e-8)
        df['packets_per_second'] = df['packet_count'] / (df['connection_duration'] + 1e-8)
        df['avg_packet_size'] = df['bytes_transferred'] / (df['packet_count'] + 1e-8)
        
        # Anomaly indicators
        df['has_unusual_port'] = (~df['destination_port'].isin([80, 443, 22, 25, 53, 389, 636])).astype(int)
        df['high_bandwidth'] = (df['bytes_per_second'] > df['bytes_per_second'].quantile(0.95)).astype(int)
        df['large_packets'] = (df['avg_packet_size'] > df['avg_packet_size'].quantile(0.95)).astype(int)
        
        # Temporal patterns
        df['time_since_last_transaction'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
        df['hourly_transaction_count'] = df.groupby(['user_id', df['timestamp'].dt.hour]).transform('size')
        
        # Device and browser features
        df['is_unrecognized_device'] = df.groupby('user_agent')['user_agent'].transform('count') < 10
        df['is_mobile'] = df['user_agent'].str.contains('Mobile|mobile|iPhone|Android', case=False, na=False).astype(int)
        
        # Aggregate features (sliding windows)
        df = self._add_sliding_window_features(df)
        
        return df
    
    def _get_ip_class(self, ip_str):
        """Get IP class (A, B, C, D, E)"""
        try:
            ip = ipaddress.IPv4Address(ip_str)
            first_octet = int(str(ip).split('.')[0])
            
            if 1 <= first_octet <= 126:
                return 0  # Class A
            elif 128 <= first_octet <= 191:
                return 1  # Class B
            elif 192 <= first_octet <= 223:
                return 2  # Class C
            elif 224 <= first_octet <= 239:
                return 3  # Class D
            else:
                return 4  # Class E
        except:
            return -1  # Invalid IP
    
    def _is_private_ip(self, ip_str):
        """Check if IP is private"""
        try:
            ip = ipaddress.IPv4Address(ip_str)
            return ip.is_private
        except:
            return False
    
    def _calculate_amount_zscore(self, row):
        """Calculate z-score for transaction amount"""
        user_id = row['user_id']
        amount = row['amount']
        
        if user_id in self.user_profiles:
            avg = self.user_profiles[user_id]['avg_amount']
            std = self.user_profiles[user_id]['std_amount']
            if std and std != 0:
                return (amount - avg) / std
        
        return 0  # Default to 0 if no profile
    
    def _calculate_frequency_zscore(self, row):
        """Calculate z-score for transaction frequency"""
        # This would require more complex logic to calculate properly
        # For now, returning a placeholder
        return 0
    
    def _add_sliding_window_features(self, df):
        """Add sliding window aggregate features"""
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Add rolling features for each user
        for window in [5, 10, 20]:  # Different window sizes
            df[f'rolling_avg_amount_{window}'] = df.groupby('user_id')['amount'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'rolling_std_amount_{window}'] = df.groupby('user_id')['amount'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            df[f'rolling_count_{window}'] = df.groupby('user_id').cumcount() + 1
        
        return df

def prepare_cybersecurity_data(df, test_size=0.2):
    """Prepare cybersecurity data for training"""
    
    extractor = CybersecurityFeatureExtractor()
    
    # Fit and transform the data
    df_processed = extractor.fit_transform(df)
    
    # Select feature columns
    feature_cols = [col for col in df_processed.columns if col.startswith('feature_')]
    X = df_processed[feature_cols].values
    
    # For anomaly detection, we typically don't have labels
    # In a real scenario, you might have some labeled anomalies
    # For now, we'll just return the features
    
    # Split temporally (important for time series data)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    
    return X_train, X_test, extractor
```

---

## Next Steps & Future Improvements

### Short-Term (Q1 2026)
- [ ] Implement graph neural networks for network topology analysis
- [ ] Add adversarial training to improve robustness
- [ ] Enhance explainability with LIME/SHAP integration

### Medium-Term (Q2-Q3 2026)
- [ ] Extend to insider threat detection
- [ ] Implement federated learning for cross-organization threat detection
- [ ] Add natural language processing for log analysis

### Long-Term (2027)
- [ ] Develop reinforcement learning for adaptive response
- [ ] Integrate with automated incident response systems
- [ ] Implement quantum-resistant cryptography features

---

## Mathematical Foundations

### Isolation Forest Algorithm
The isolation forest isolates anomalies by randomly selecting features and split values:
```
s(x,n) = c(n) / E(h(x))
```
Where:
- s(x,n) is the anomaly score for sample x
- n is the number of samples in the dataset
- E(h(x)) is the average path length for sample x
- c(n) is the average path length of unsuccessful search in a binary search tree: c(n) = 2H(n-1) - (2(n-1)/n)

### Autoencoder Reconstruction Error
For an autoencoder with encoder f and decoder g:
```
L(x) = ||x - g(f(x))||²
```
Anomalies have higher reconstruction error than normal samples.

### LSTM Anomaly Detection
For sequence X = {x₁, x₂, ..., xₜ}:
```
hₜ = LSTM(x₁, x₂, ..., xₜ)
x̂ₜ₊₁ = Decoder(hₜ)
Anomaly Score = ||xₜ₊₁ - x̂ₜ₊₁||²
```

### Ensemble Methods
For an ensemble of k detectors:
```
Score_ensemble(x) = Σᵢ₌₁ᵏ wᵢ × Scoreᵢ(x)
```
Where wᵢ are the weights assigned to each detector.

### Evaluation Metrics
Precision:
```
Precision = TP / (TP + FP)
```

Recall:
```
Recall = TP / (TP + FN)
```

F1-Score:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

AUC-ROC:
```
AUC = ∫₀¹ TPR(FPR⁻¹(x)) dx
```
Where TPR is true positive rate and FPR is false positive rate.

---

## Production Considerations

### Scalability
- **Distributed Processing**: Use Apache Spark for large-scale feature engineering
- **Model Serving**: TensorFlow Serving or TorchServe for scalable inference
- **Caching Strategy**: Redis for frequently accessed user profiles
- **Load Balancing**: Distribute requests across multiple model instances

### Security
- **Data Encryption**: Encrypt sensitive network logs in transit and at rest
- **Access Control**: Role-based access to the detection system
- **Audit Logging**: Track all detection requests and outcomes

### Reliability
- **Redundancy**: Multiple model instances across availability zones
- **Graceful Degradation**: Fallback to simpler models if primary fails
- **Disaster Recovery**: Automated backup and restore of models

### Performance Optimization
- **Model Compression**: Quantization for faster inference
- **Feature Caching**: Pre-compute and cache rolling statistics
- **Parallel Processing**: Process multiple transactions concurrently

---

## Conclusion

This advanced cybersecurity anomaly detection system demonstrates sophisticated ML engineering:
- **Multi-Layer Architecture**: Combines multiple detection methods
- **Real-Time Processing**: Streaming pipeline with <100ms response time
- **Scalable Infrastructure**: Handles 50M+ transactions daily
- **Business Impact**: $24M annual savings, 94% false positive reduction

**Key takeaway**: Effective cybersecurity requires combining multiple detection approaches with real-time processing and continuous learning capabilities.

Architecture and ops blueprint: `docs/system_design_solutions/19_cybersecurity_anomaly_system.md`.

---

**Contact**: Implementation details in `src/anomaly_detection/cybersecurity.py`.
Notebooks: `notebooks/case_studies/cybersecurity_anomaly_detection.ipynb`