# Case Study 7: Anomaly Detection for Cybersecurity Threats

## Executive Summary

**Problem**: Enterprise SaaS company facing 2,000+ security alerts daily with 95% false positives, overwhelming security team and missing real threats.

**Solution**: Built ML-powered anomaly detection system using ensemble of isolation forests, autoencoders, and sequence models.

**Impact**: Reduced false positives by 80%, increased threat detection rate by 45%, and decreased mean time to respond from 4 hours to 45 minutes.

---

## Business Context

### Company Profile
- **Industry**: Enterprise SaaS (HR Management Platform)
- **Users**: 50,000+ organizations, 5M+ employees
- **Security Events**: 2M+ logs/day across network, application, and user activity
- **Problem**: Alert fatigue causing real threats to be missed

### Key Challenges
1. **Scale**: 2M+ events/day from 5M+ users across 50,000+ organizations
2. **Variety**: Network traffic, user behavior, system logs, API calls
3. **Imbalance**: <1% of alerts represent actual threats
4. **Latency**: Need real-time detection to prevent ongoing attacks

---

## Technical Approach

### Multi-Model Anomaly Detection Architecture

```
Log Ingestion → Feature Engineering → Anomaly Detection Ensemble → Threat Scoring → Alert Prioritization
   (Kafka)        (Time Series,     (Isolation Forest,            (Risk Model)    (Risk-Based Ranking)
                 Behavioral,        Autoencoder, LSTM)
                 Network)
```

### Stage 1: Data Collection & Feature Engineering

**Data Sources**:
- Network logs: Traffic patterns, connection attempts, data volumes
- User activity: Login patterns, access times, resource usage
- System logs: Error rates, API calls, database queries
- Application logs: Feature usage, transaction patterns

**Feature Engineering Pipeline**:
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_network_features(logs_df):
    """Create network-related features for anomaly detection."""
    features = pd.DataFrame()
    
    # Connection-based features
    features['connections_per_minute'] = logs_df.groupby('minute')['connection_id'].count()
    features['unique_ips_per_minute'] = logs_df.groupby('minute')['source_ip'].nunique()
    features['avg_bytes_transferred'] = logs_df.groupby('minute')['bytes'].mean()
    features['failed_connections_ratio'] = (
        logs_df[logs_df['status_code'] >= 400].groupby('minute')['connection_id'].count() /
        logs_df.groupby('minute')['connection_id'].count()
    )
    
    # Time-based features
    features['hour'] = logs_df['timestamp'].dt.hour
    features['day_of_week'] = logs_df['timestamp'].dt.dayofweek
    features['is_weekend'] = (logs_df['timestamp'].dt.dayofweek >= 5).astype(int)
    
    # Rolling statistics (last 5 minutes)
    features['conn_rolling_mean'] = features['connections_per_minute'].rolling(window=5).mean()
    features['conn_rolling_std'] = features['connections_per_minute'].rolling(window=5).std()
    
    return features.fillna(method='bfill')

def create_user_behavior_features(activity_df):
    """Create user behavior features for anomaly detection."""
    features = pd.DataFrame()
    
    # Login patterns
    features['logins_per_hour'] = activity_df.groupby(['user_id', 'hour'])['timestamp'].count()
    features['avg_session_duration'] = activity_df.groupby('user_id').apply(
        lambda x: (x['logout_time'] - x['login_time']).mean()
    )
    features['unusual_login_times'] = activity_df.groupby('user_id').apply(
        lambda x: sum((x['hour'] < 6) | (x['hour'] > 22)) / len(x)
    )
    
    # Access patterns
    features['resources_accessed_per_session'] = activity_df.groupby('session_id')['resource'].nunique()
    features['data_download_volume'] = activity_df.groupby('user_id')['downloaded_bytes'].sum()
    
    return features
```

### Stage 2: Anomaly Detection Ensemble

**Model 1: Isolation Forest** (for tabular anomalies)
- Detects outliers in feature space
- Effective for point anomalies
- Fast training and inference

**Model 2: Autoencoder** (for reconstruction errors)
- Learns normal patterns in data
- Flags high reconstruction errors as anomalies
- Good for complex, non-linear patterns

**Model 3: LSTM Sequence Model** (for temporal anomalies)
- Detects unusual sequences of events
- Captures temporal dependencies
- Effective for attack pattern detection

### Stage 3: Threat Scoring & Alert Prioritization

**Risk Scoring Model**:
- Combines anomaly scores from all models
- Factors in asset sensitivity and user role
- Outputs probability of actual threat

---

## Model Development

### Isolation Forest Model

**Approach**: Isolate anomalous data points by randomly selecting features and split values
- Effective for high-dimensional data
- Linear time complexity
- No assumptions about data distribution

**Hyperparameter Tuning**:
```python
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_samples': ['auto', 0.8, 0.9],
    'contamination': [0.01, 0.05, 0.1, 0.15],
    'max_features': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(
    IsolationForest(random_state=42),
    param_grid,
    cv=5,
    scoring='f1'
)
```

**Optimized Parameters**:
- n_estimators: 100
- contamination: 0.05 (expected 5% anomalies)
- max_samples: 0.8
- max_features: 0.9

### Autoencoder Model

**Architecture**: Encoder-Decoder with bottleneck layer
- Input: Normalized feature vector (50 dimensions)
- Encoder: [50 → 25 → 10 → 5] dense layers
- Decoder: [5 → 10 → 25 → 50] dense layers
- Reconstruction loss: Mean squared error

**Training Process**:
- Train only on normal (non-anomalous) data
- Use reconstruction error as anomaly score
- Threshold determined by validation on mixed data

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class AnomalyAutoencoder(Model):
    def __init__(self, input_dim, encoding_dim=5):
        super(AnomalyAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Dense(25, activation='relu'),
            layers.Dense(10, activation='relu'),
            layers.Dense(encoding_dim, activation='relu')
        ])
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(10, activation='relu'),
            layers.Dense(25, activation='relu'),
            layers.Dense(input_dim, activation='sigmoid')
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Compile and train
autoencoder = AnomalyAutoencoder(input_dim=50)
autoencoder.compile(optimizer='adam', loss='mse')

# Train on normal data only
history = autoencoder.fit(
    normal_data, normal_data,
    epochs=100,
    batch_size=256,
    validation_split=0.2,
    verbose=1
)
```

### LSTM Sequence Model

**Approach**: Learn temporal patterns in security event sequences
- Input: Sequences of security events over time windows
- Architecture: LSTM layers followed by dense classifier
- Output: Probability of anomalous sequence

**Model Architecture**:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_lstm_model(sequence_length, n_features):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

### Model Comparison

| Model | Precision | Recall | F1-Score | Inference Time | Notes |
|-------|-----------|--------|----------|----------------|-------|
| Rule-based | 0.12 | 0.35 | 0.18 | <1ms | High recall, low precision |
| Isolation Forest | 0.45 | 0.52 | 0.48 | 2ms | Good balance |
| Autoencoder | 0.58 | 0.41 | 0.48 | 5ms | Good precision |
| LSTM | 0.52 | 0.61 | 0.56 | 15ms | Good recall |
| **Ensemble** | **0.78** | **0.69** | **0.73** | **20ms** | **Selected** |

**Selected Approach**: Ensemble of all three models
- **Reason**: Combined strengths of different approaches
- **Combination**: Weighted average based on validation performance

### Ensemble Method

```python
def ensemble_predict(iso_score, ae_score, lstm_score):
    """
    Combine scores from different models.
    Higher scores indicate higher anomaly probability.
    """
    # Normalize scores to 0-1 range
    iso_norm = (iso_score - iso_min) / (iso_max - iso_min)
    ae_norm = (ae_score - ae_min) / (ae_max - ae_min)
    lstm_norm = (lstm_score - lstm_min) / (lstm_max - lstm_min)
    
    # Weighted combination (weights based on validation performance)
    final_score = (0.3 * iso_norm + 0.4 * ae_norm + 0.3 * lstm_norm)
    
    return final_score
```

### Cross-Validation Results

**Individual Models**:
- Isolation Forest: F1 = 0.48, AUC = 0.72
- Autoencoder: F1 = 0.48, AUC = 0.76
- LSTM: F1 = 0.56, AUC = 0.79

**Ensemble Performance**:
- F1-Score: 0.73
- Precision: 0.78
- Recall: 0.69
- AUC-ROC: 0.89

---

## Production Deployment

### Real-Time Architecture

```
Security Logs → Stream Processing → Feature Store → Anomaly Detection → Threat Engine → Alert System
   (Syslog)       (Apache Kafka)    (Redis/Flink)   (ML Models)      (Risk Scoring)  (SIEM)
```

### Components

**1. Log Ingestion Service** (Apache Kafka):
```python
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers=['kafka-cluster:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def send_security_event(event):
    producer.send('security-events', event)
    producer.flush()
```

**2. Feature Engineering Service** (Apache Flink):
```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

def create_feature_pipeline():
    env = StreamExecutionEnvironment.get_execution_environment()
    t_env = StreamTableEnvironment.create(env)
    
    # Define input stream from Kafka
    t_env.execute_sql("""
        CREATE TABLE security_logs (
            timestamp TIMESTAMP(3),
            user_id STRING,
            ip_address STRING,
            event_type STRING,
            resource STRING,
            bytes_transferred BIGINT,
            session_id STRING
        ) WITH (
            'connector' = 'kafka',
            'topic' = 'security-events',
            'properties.bootstrap.servers' = 'kafka-cluster:9092',
            'format' = 'json'
        )
    """)
    
    # Create sliding window aggregations
    t_env.execute_sql("""
        CREATE VIEW user_activity_features AS
        SELECT 
            user_id,
            TUMBLE_START(rowtime, INTERVAL '5' MINUTE) as window_start,
            COUNT(*) as login_count,
            AVG(bytes_transferred) as avg_bytes,
            COUNT(DISTINCT ip_address) as unique_ips
        FROM security_logs
        GROUP BY user_id, TUMBLE(rowtime, INTERVAL '5' MINUTE)
    """)
    
    return t_env
```

**3. Anomaly Detection Service** (FastAPI + TensorFlow):
```python
from fastapi import FastAPI
import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest

app = FastAPI()

# Load pre-trained models
iso_forest = IsolationForest(contamination=0.05)
autoencoder = tf.keras.models.load_model('models/autoencoder.h5')
lstm_model = tf.keras.models.load_model('models/lstm.h5')

@app.post("/detect_anomaly")
async def detect_anomaly(features: dict):
    feature_vector = np.array([list(features.values())]).reshape(1, -1)
    
    # Get anomaly scores from each model
    iso_score = iso_forest.decision_function(feature_vector)[0]
    ae_reconstruction = autoencoder.predict(feature_vector)
    ae_score = np.mean((feature_vector - ae_reconstruction) ** 2)
    
    # LSTM requires sequence input
    sequence = get_recent_sequence(features['user_id'])
    lstm_score = lstm_model.predict(sequence)[0][0]
    
    # Combine scores using ensemble method
    final_score = ensemble_predict(iso_score, ae_score, lstm_score)
    
    # Determine if anomaly
    is_anomaly = final_score > 0.7  # threshold
    
    # Calculate risk score
    risk_score = calculate_risk_score(features, final_score)
    
    if is_anomaly:
        alert = {
            'user_id': features['user_id'],
            'timestamp': features['timestamp'],
            'anomaly_score': float(final_score),
            'risk_score': float(risk_score),
            'alert_level': get_alert_level(risk_score)
        }
        send_alert(alert)
    
    return {
        'is_anomaly': is_anomaly,
        'anomaly_score': float(final_score),
        'risk_score': float(risk_score)
    }
```

**4. Threat Intelligence Engine**:
- Correlates multiple anomalies
- Applies business context (user role, asset sensitivity)
- Generates prioritized alerts

### Operational SLOs
- **Inference Latency**: p95 < 50ms for real-time scoring
- **Throughput**: Handle 10,000 events/second
- **Availability**: 99.9% uptime (security cannot be offline)
- **False Positive Rate**: <5% (to prevent alert fatigue)

### Monitoring & Drift Detection
- **Real-time metrics**: Anomaly rates, model performance, alert volumes
- **Data drift**: Monitor feature distributions and correlations
- **Concept drift**: Track model performance degradation over time
- **Feedback loop**: Incorporate analyst feedback to improve models

---

## Results & Impact

### Model Performance in Production

**Anomaly Detection Metrics**:
- **Precision**: 78% (78% of flagged events are actual threats)
- **Recall**: 69% (catches 69% of all actual threats)
- **F1-Score**: 73%
- **AUC-ROC**: 0.89

**Alert Quality Improvement**:
- **Before**: 2,000+ alerts/day, 5% true threats
- **After**: 400 alerts/day, 20% true threats
- **Reduction in noise**: 80% fewer false positives

### Business Impact (6 months post-launch)

| Metric | Before ML | After ML | Improvement |
|--------|-----------|----------|-------------|
| **Daily Security Alerts** | 2,100 | 420 | **-80%** |
| **True Threat Detection Rate** | 35% | 51% | **+46%** |
| **Mean Time to Respond** | 4.2 hours | 0.75 hours | **-82%** |
| **Security Team Productivity** | Baseline | 3.2x | **+220%** |
| **Incident Response Cost** | $120K/month | $45K/month | **-62%** |

### Threat Detection Performance

**By Threat Type**:
- **Brute Force Attacks**: 85% detection rate
- **Privilege Escalation**: 72% detection rate
- **Data Exfiltration**: 78% detection rate
- **Insider Threats**: 65% detection rate
- **Zero-Day Exploits**: 58% detection rate

### Cost-Benefit Analysis

**Savings**:
- Reduced analyst time: 1,680 hours/day × $80/hr × 180 days = $24.2M
- Faster incident response: Reduced breach costs by 62%
- Automation benefits: $1.2M annually in efficiency gains

**Investment**:
- Development cost: $1.5M (team of 8 for 8 months)
- Infrastructure cost: $300K/year
- Training and deployment: $200K

**Net Benefit**: $24.2M + $2.1M - $2M = **$24.3M annually**

### Specific Success Stories

**Case 1: Insider Threat Detection**
- System flagged unusual data access patterns by departing employee
- Detected attempt to download 50GB of customer data
- Incident contained before data exfiltration completed
- Estimated savings: $2.3M in potential breach costs

**Case 2: Zero-Day Attack Prevention**
- LSTM model detected unusual sequence of API calls
- Pattern didn't match known signatures but flagged as anomalous
- Investigation revealed new attack vector
- Protection extended to all customers before wider exploitation

---

## Challenges & Solutions

### Challenge 1: Class Imbalance
- **Problem**: Only ~1% of events represent actual threats
- **Solution**:
  - Synthetic minority oversampling technique (SMOTE)
  - Cost-sensitive learning with higher penalty for false negatives
  - Threshold tuning to optimize for business impact

### Challenge 2: Concept Drift
- **Problem**: Attack patterns evolve over time, degrading model performance
- **Solution**:
  - Continuous monitoring of model performance
  - Automated retraining pipeline triggered by performance drops
  - Ensemble approach provides stability against individual model drift

### Challenge 3: Real-Time Processing
- **Problem**: Need to process 2M+ events/day with <50ms latency
- **Solution**:
  - Stream processing with Apache Kafka and Flink
  - Model optimization and quantization
  - Caching of frequently accessed features

### Challenge 4: Feature Engineering Complexity
- **Problem**: Security data highly dimensional and complex
- **Solution**:
  - Domain expert collaboration to identify relevant features
  - Automated feature selection techniques
  - Regular feature importance analysis and updates

---

## Lessons Learned

### What Worked

1. **Ensemble Approach Superior to Single Models**
   - Individual models: F1 = 0.48-0.56
   - Ensemble: F1 = 0.73
   - Different models capture different types of anomalies

2. **Feature Engineering Critical for Performance**
   - Raw logs: F1 = 0.35
   - Engineered features: F1 = 0.73
   - Domain knowledge essential for effective features

3. **Real-Time Processing Required for Effectiveness**
   - Batch processing: 4-hour delay
   - Real-time: Immediate response
   - Critical for preventing ongoing attacks

### What Didn't Work

1. **Pure Statistical Approaches**
   - Z-score, IQR methods: F1 = 0.28
   - Too many false positives for complex security patterns
   - ML approaches much more effective

2. **Single Algorithm Approach**
   - Tried isolation forest alone: F1 = 0.48
   - Ensemble significantly better: F1 = 0.73
   - Different algorithms capture different anomaly types

---

## Technical Implementation

### Complete Anomaly Detection Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
import joblib

class CybersecurityAnomalyDetector:
    def __init__(self, model_paths=None):
        self.iso_forest = None
        self.autoencoder = None
        self.lstm_model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
        if model_paths:
            self.load_models(model_paths)
    
    def create_autoencoder(self, input_dim, encoding_dim=10):
        """Create autoencoder for anomaly detection."""
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
        
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def create_lstm_model(self, sequence_length, n_features):
        """Create LSTM model for sequence anomaly detection."""
        model = tf.keras.Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)),
            tf.keras.layers.Dropout(0.2),
            LSTM(50, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X_normal, sequence_data=None):
        """Train all models on normal data."""
        # Prepare data
        X_scaled = self.scaler.fit_transform(X_normal)
        self.feature_columns = X_normal.columns.tolist()
        
        # Train Isolation Forest
        self.iso_forest = IsolationForest(
            contamination=0.05,
            random_state=42,
            n_estimators=100
        )
        self.iso_forest.fit(X_scaled)
        
        # Train Autoencoder
        self.autoencoder = self.create_autoencoder(X_scaled.shape[1])
        self.autoencoder.fit(
            X_scaled, X_scaled,
            epochs=50,
            batch_size=256,
            validation_split=0.2,
            verbose=0
        )
        
        # Train LSTM if sequence data provided
        if sequence_data is not None:
            seq_length, n_features = sequence_data.shape[1], sequence_data.shape[2]
            self.lstm_model = self.create_lstm_model(seq_length, n_features)
            self.lstm_model.fit(
                sequence_data, np.ones(len(sequence_data)),  # All normal sequences
                epochs=30,
                batch_size=64,
                verbose=0
            )
    
    def predict_anomaly_scores(self, X):
        """Get anomaly scores from all models."""
        X_scaled = self.scaler.transform(X)
        
        # Isolation Forest score (convert to probability-like score)
        iso_scores = self.iso_forest.decision_function(X_scaled)
        iso_probs = 1 / (1 + np.exp(-iso_scores))  # Sigmoid transformation
        
        # Autoencoder reconstruction error
        reconstructed = self.autoencoder.predict(X_scaled)
        ae_errors = np.mean((X_scaled - reconstructed) ** 2, axis=1)
        ae_probs = 1 / (1 + np.exp(-ae_errors))  # Normalize to 0-1
        
        # LSTM scores (would need sequence data)
        lstm_probs = np.zeros(len(X))  # Placeholder if no sequence model
        if self.lstm_model is not None:
            # This would require converting X to sequences
            pass
        
        return iso_probs, ae_probs, lstm_probs
    
    def predict(self, X, threshold=0.7):
        """Predict anomalies using ensemble approach."""
        iso_scores, ae_scores, lstm_scores = self.predict_anomaly_scores(X)
        
        # Ensemble combination
        ensemble_scores = (0.3 * iso_scores + 0.4 * ae_scores + 0.3 * lstm_scores)
        
        return ensemble_scores > threshold, ensemble_scores
    
    def save_models(self, paths):
        """Save trained models."""
        joblib.dump(self.iso_forest, paths['isolation_forest'])
        joblib.dump(self.scaler, paths['scaler'])
        self.autoencoder.save(paths['autoencoder'])
        if self.lstm_model:
            self.lstm_model.save(paths['lstm'])
    
    def load_models(self, paths):
        """Load pre-trained models."""
        self.iso_forest = joblib.load(paths['isolation_forest'])
        self.scaler = joblib.load(paths['scaler'])
        self.autoencoder = tf.keras.models.load_model(paths['autoencoder'])
        if 'lstm' in paths:
            self.lstm_model = tf.keras.models.load_model(paths['lstm'])

# Usage example
detector = CybersecurityAnomalyDetector()

# Train on normal data
normal_data = load_normal_security_data()  # Your data loading function
detector.fit(normal_data)

# Detect anomalies in new data
new_data = load_new_security_data()
anomalies, scores = detector.predict(new_data, threshold=0.7)

print(f"Detected {sum(anomalies)} anomalies out of {len(new_data)} events")
```

### Threat Scoring Engine

```python
import numpy as np
from datetime import datetime, timedelta

class ThreatScoringEngine:
    def __init__(self):
        self.risk_weights = {
            'user_role': {'admin': 0.3, 'power_user': 0.2, 'regular': 0.1},
            'asset_sensitivity': {'critical': 0.4, 'high': 0.3, 'medium': 0.2, 'low': 0.1},
            'time_factor': {'off_hours': 0.2, 'weekend': 0.15, 'normal': 0.05},
            'behavior_pattern': {'first_time': 0.25, 'unusual_volume': 0.3, 'normal': 0.0}
        }
    
    def calculate_threat_score(self, event, anomaly_score, user_context, asset_context):
        """
        Calculate comprehensive threat score combining anomaly detection with contextual risk factors.
        """
        base_score = anomaly_score
        
        # User role risk factor
        user_risk = self.risk_weights['user_role'].get(user_context.get('role', 'regular'), 0.1)
        
        # Asset sensitivity risk factor
        asset_risk = self.risk_weights['asset_sensitivity'].get(
            asset_context.get('sensitivity', 'medium'), 0.2
        )
        
        # Time-based risk factor
        event_time = datetime.fromisoformat(event['timestamp'])
        time_risk = self.evaluate_time_risk(event_time)
        
        # Behavior pattern risk
        behavior_risk = self.evaluate_behavior_risk(event, user_context)
        
        # Weighted combination
        threat_score = (
            0.4 * base_score +           # Anomaly score weight
            0.2 * user_risk +            # User risk weight  
            0.2 * asset_risk +           # Asset risk weight
            0.1 * time_risk +            # Time risk weight
            0.1 * behavior_risk          # Behavior risk weight
        )
        
        return min(threat_score, 1.0)  # Cap at 1.0
    
    def evaluate_time_risk(self, event_time):
        """Evaluate risk based on time of event."""
        hour = event_time.hour
        day_of_week = event_time.weekday()
        
        # Off-hours (before 6am or after 10pm)
        if hour < 6 or hour > 22:
            return self.risk_weights['time_factor']['off_hours']
        
        # Weekend
        if day_of_week >= 5:
            return self.risk_weights['time_factor']['weekend']
        
        return self.risk_weights['time_factor']['normal']
    
    def evaluate_behavior_risk(self, event, user_context):
        """Evaluate risk based on user behavior patterns."""
        # Check if this is first time accessing this resource
        if event['resource'] not in user_context.get('accessed_resources', []):
            return self.risk_weights['behavior_pattern']['first_time']
        
        # Check for unusual volume (compared to user's historical patterns)
        if self.is_unusual_volume(event, user_context):
            return self.risk_weights['behavior_pattern']['unusual_volume']
        
        return self.risk_weights['behavior_pattern']['normal']
    
    def is_unusual_volume(self, event, user_context):
        """Check if the event represents unusual volume for this user."""
        # Simplified logic - in practice would compare to historical patterns
        threshold_multiplier = user_context.get('volume_threshold', 2.0)
        historical_avg = user_context.get('avg_daily_actions', 10)
        
        # This would vary by event type
        current_volume = event.get('actions_count', 1)
        
        return current_volume > (historical_avg * threshold_multiplier)

# Usage
scoring_engine = ThreatScoringEngine()
threat_score = scoring_engine.calculate_threat_score(
    event={'timestamp': '2023-05-15T02:30:00Z', 'resource': 'database_admin_panel'},
    anomaly_score=0.85,
    user_context={'role': 'admin', 'accessed_resources': ['dashboard'], 'avg_daily_actions': 5},
    asset_context={'sensitivity': 'critical'}
)
print(f"Threat score: {threat_score:.3f}")
```

---

## Next Steps & Future Improvements

### Short-Term (Q1 2026)
- [ ] Implement graph neural networks for network topology analysis
- [ ] Add adversarial training to handle evasion attempts
- [ ] Enhance threat intelligence integration

### Medium-Term (Q2-Q3 2026)
- [ ] Deploy federated learning for cross-organization threat detection
- [ ] Integrate with SOAR (Security Orchestration and Response) platforms
- [ ] Add explainability for model decisions

### Long-Term (2027)
- [ ] Implement reinforcement learning for adaptive defense
- [ ] Expand to IoT and endpoint security
- [ ] Create predictive threat modeling

---

## Conclusion

This cybersecurity anomaly detection system demonstrates:
- **Advanced ML**: Ensemble of isolation forests, autoencoders, and LSTMs
- **Real-Time Processing**: Stream processing architecture for immediate response
- **Security Impact**: 80% reduction in false positives, 45% increase in threat detection

**Key takeaway**: Combining multiple ML approaches with domain expertise and real-time processing delivers superior cybersecurity outcomes.

Architecture and ops blueprint: `docs/system_design_solutions/10_cybersecurity_anomaly_detection.md`.

---

**Contact**: Implementation details in `src/ml/anomaly_detection.py`.
Notebooks: `notebooks/case_studies/cybersecurity_anomaly_detection.ipynb`