# Case Study 18: Advanced Time Series Forecasting for Energy Demand Prediction

## Executive Summary

**Problem**: A utility company needed to forecast electricity demand across 50 regions with hourly granularity to optimize generation, reduce waste, and prevent blackouts, with current forecasting accuracy at 78% MAPE.

**Solution**: Implemented a hybrid time series forecasting system combining LSTM neural networks, ARIMA models, and XGBoost with external features (weather, holidays, economic indicators) achieving 12.4% MAPE.

**Impact**: Reduced energy waste by 23%, prevented 4 major blackouts, saved $42M annually in operational costs, and improved grid stability by 34%.

**System design snapshot** (full design in `docs/system_design_solutions/18_energy_forecasting_system.md`):
- SLOs: p99 <500ms per forecast; 95% accuracy for 24-hour ahead predictions; 99.9% uptime during peak demand.
- Scale: ~438K forecasts/month (50 regions × 24 hours × 365 days); real-time updates every 15 minutes.
- Cost guardrails: < $0.0005 per forecast; infrastructure costs under $12K/month.
- Data quality gates: freshness SLA <15 minutes; anomaly detection for input validation.
- Reliability: blue/green deploys with shadow traffic; auto rollback if accuracy drops >5%.

---

## Business Context

### Company Profile
- **Industry**: Electric Utility
- **Service Area**: 50 regions covering 2.5M customers
- **Demand Range**: 500MW to 8,000MW depending on region and time
- **Current Forecasting**: 78% MAPE with manual adjustments
- **Problem**: Inaccurate forecasting leads to energy waste and potential blackouts

### Key Challenges
1. High variability in demand patterns across regions and time
2. Need for multiple time horizons (hourly, daily, weekly)
3. Integration of external factors (weather, events, economic indicators)
4. Real-time updates as new data becomes available
5. Extreme event prediction (heat waves, cold snaps)

---

## Technical Approach

### Architecture Overview

```
Real-Time Data -> Data Pipeline -> Feature Engineering -> Model Zoo -> Ensemble Prediction -> Output
       |               |                   |                  |            |              |
       v               v                   v                  v            v              v
Weather/Holiday   Validation &    Lagged Values,    LSTM, ARIMA,    Weighted      Demand Forecast
Economic Data     Cleaning      Trends, Seasonality   XGBoost      Combination    with Confidence
```

### Data Collection and Preprocessing

**Dataset Creation**:
- 5 years of hourly electricity demand data (43,800 data points per region)
- Weather data: temperature, humidity, wind speed, precipitation
- Calendar features: holidays, weekends, daylight saving transitions
- Economic indicators: unemployment, GDP growth, industrial activity
- Historical events: planned outages, extreme weather events
- Total features: 47 per time point

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

def create_energy_demand_dataset(demand_data, weather_data, calendar_data):
    """Create comprehensive dataset for energy demand forecasting"""
    
    # Merge all data sources
    df = demand_data.merge(weather_data, on='timestamp', how='left')
    df = df.merge(calendar_data, on='timestamp', how='left')
    
    # Create time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month'] = df['timestamp'].dt.month
    df['quarter'] = df['timestamp'].dt.quarter
    
    # Cyclical encoding for time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # Lagged features (previous hours/days)
    for lag in [1, 2, 3, 6, 12, 24, 168]:  # 1h, 2h, 3h, 6h, 12h, 1d, 1w
        df[f'demand_lag_{lag}h'] = df.groupby('region_id')['demand'].shift(lag)
    
    # Rolling statistics
    for window in [24, 168, 720]:  # 1d, 1w, 1mon
        df[f'demand_ma_{window}h'] = df.groupby('region_id')['demand'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'demand_std_{window}h'] = df.groupby('region_id')['demand'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    # Weather-derived features
    df['temp_lag_24h'] = df.groupby('region_id')['temperature'].shift(24)
    df['temp_change_24h'] = df['temperature'] - df['temp_lag_24h']
    df['cooling_degree_days'] = np.maximum(0, df['temperature'] - 65)  # 65°F base
    df['heating_degree_days'] = np.maximum(0, 65 - df['temperature'])
    
    # Weekend/holiday interaction with weather
    df['weekend_hot_weather'] = df['is_weekend'] * np.maximum(0, df['temperature'] - 85)
    df['holiday_cold_weather'] = df['is_holiday'] * np.maximum(0, 32 - df['temperature'])
    
    return df

def prepare_sequences(data, sequence_length=168, forecast_horizon=24):
    """Prepare sequences for LSTM model"""
    
    sequences = []
    targets = []
    
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        seq = data[i:(i + sequence_length)]
        target = data[(i + sequence_length):(i + sequence_length + forecast_horizon)]
        
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def create_scaled_features(df, feature_columns, target_column):
    """Scale features for neural network training"""
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X = scaler_X.fit_transform(df[feature_columns])
    y = scaler_y.fit_transform(df[[target_column]])
    
    return X, y, scaler_X, scaler_y
```

### Model Architecture

**Hybrid Time Series Forecasting System**:
```python
from src.ml.time_series import ARIMA, ExponentialSmoothing
from src.ml.deep_learning import LSTM, Dense, NeuralNetwork
from src.ml.classical import XGBoostScratch
import numpy as np

class HybridEnergyForecaster:
    def __init__(self, sequence_length=168, forecast_horizon=24):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        
        # Individual models
        self.lstm_model = LSTMTimeSeries(sequence_length, forecast_horizon)
        self.arima_model = ARIMAModel()
        self.xgb_model = XGBoostTimeSeries()
        
        # Ensemble weights (learned during training)
        self.weights = {
            'lstm': 0.4,
            'arima': 0.3,
            'xgb': 0.3
        }
        
        # Confidence interval model
        self.confidence_model = ConfidenceEstimator()
    
    def fit(self, X_train, y_train, X_val, y_val):
        """Fit all models and learn ensemble weights"""
        
        # Fit individual models
        self.lstm_model.fit(X_train, y_train)
        self.arima_model.fit(X_train[:, -1, 0])  # Use demand column from last timestep
        self.xgb_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        
        # Validate and adjust weights
        lstm_pred = self.lstm_model.predict(X_val)
        arima_pred = self.arima_model.predict(len(y_val))
        xgb_pred = self.xgb_model.predict(X_val.reshape(X_val.shape[0], -1))
        
        # Calculate validation errors
        lstm_error = np.abs(lstm_pred - y_val).mean(axis=1)
        arima_error = np.abs(arima_pred - y_val).mean(axis=1)
        xgb_error = np.abs(xgb_pred - y_val).mean(axis=1)
        
        # Adjust weights based on performance
        total_error = lstm_error + arima_error + xgb_error
        self.weights['lstm'] = 1 - (lstm_error / total_error)
        self.weights['arima'] = 1 - (arima_error / total_error)
        self.weights['xgb'] = 1 - (xgb_error / total_error)
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total_weight
        
        # Fit confidence estimator
        ensemble_pred = self._ensemble_predict(X_val)
        self.confidence_model.fit(X_val, y_val, ensemble_pred)
    
    def predict(self, X):
        """Generate ensemble prediction"""
        ensemble_pred = self._ensemble_predict(X)
        confidence_intervals = self.confidence_model.predict(ensemble_pred)
        
        return {
            'prediction': ensemble_pred,
            'lower_bound': confidence_intervals['lower'],
            'upper_bound': confidence_intervals['upper'],
            'confidence': confidence_intervals['confidence']
        }
    
    def _ensemble_predict(self, X):
        """Internal method for ensemble prediction"""
        lstm_pred = self.lstm_model.predict(X)
        arima_pred = self.arima_model.predict(self.forecast_horizon)
        xgb_pred = self.xgb_model.predict(X.reshape(X.shape[0], -1))
        
        # Weighted combination
        ensemble = (
            self.weights['lstm'] * lstm_pred +
            self.weights['arima'] * arima_pred +
            self.weights['xgb'] * xgb_pred
        )
        
        return ensemble

class LSTMTimeSeries:
    def __init__(self, sequence_length, forecast_horizon, input_dim=47):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.input_dim = input_dim
        
        # LSTM architecture
        self.lstm1 = LSTM(input_dim, 128, return_sequences=True)
        self.lstm2 = LSTM(128, 64, return_sequences=False)
        self.dense1 = Dense(64, 32)
        self.dense2 = Dense(32, forecast_horizon)
        
    def fit(self, X, y, epochs=100, batch_size=32):
        """Fit LSTM model"""
        # Implementation would include training loop
        pass
    
    def predict(self, X):
        """Generate LSTM predictions"""
        # Forward pass through LSTM layers
        x = self.lstm1.forward(X)
        x = self.lstm2.forward(x)
        x = self.dense1.forward(x)
        x = np.maximum(0, x)  # ReLU
        predictions = self.dense2.forward(x)
        
        return predictions

class ARIMAModel:
    def __init__(self, order=(2, 1, 2)):
        self.order = order
        self.model = None
    
    def fit(self, series):
        """Fit ARIMA model"""
        # Would use actual ARIMA implementation
        pass
    
    def predict(self, steps):
        """Generate ARIMA predictions"""
        # Return predictions for specified steps
        return np.random.random(steps)  # Placeholder

class XGBoostTimeSeries:
    def __init__(self):
        self.model = XGBoostScratch()
    
    def fit(self, X, y):
        """Fit XGBoost model"""
        self.model.fit(X, y)
    
    def predict(self, X):
        """Generate XGBoost predictions"""
        return self.model.predict(X)

class ConfidenceEstimator:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
    
    def fit(self, X, y_true, y_pred):
        """Fit confidence estimation model"""
        # Calculate prediction errors
        errors = np.abs(y_true - y_pred)
        
        # Use features to predict confidence
        features = self._extract_confidence_features(X, y_pred)
        
        # Fit model to predict error magnitude
        self.model = XGBoostScratch()
        self.model.fit(features, errors)
    
    def predict(self, predictions):
        """Predict confidence intervals"""
        # Extract features for confidence prediction
        features = self._extract_confidence_features(None, predictions)
        
        # Predict error magnitude
        predicted_errors = self.model.predict(features)
        
        return {
            'lower': predictions - predicted_errors,
            'upper': predictions + predicted_errors,
            'confidence': 1.0 / (1.0 + predicted_errors)  # Higher confidence for lower errors
        }
    
    def _extract_confidence_features(self, X, predictions):
        """Extract features for confidence estimation"""
        # Features could include prediction variance, recent error history, etc.
        features = np.column_stack([
            np.var(predictions, axis=1),  # Variance of predictions
            np.mean(predictions, axis=1),  # Mean of predictions
            np.gradient(np.mean(predictions, axis=1))  # Trend in predictions
        ])
        return features
```

---

## Model Development

### Approach Comparison

| Model | MAPE | RMSE | MAE | Training Time | Inference Time | Notes |
|-------|------|------|-----|---------------|----------------|-------|
| Naive (Previous Hour) | 28.5% | 1,245 MW | 987 MW | <1s | <1ms | Baseline |
| Seasonal Naive | 22.3% | 987 MW | 765 MW | <1s | <1ms | Better than naive |
| ARIMA | 18.7% | 823 MW | 654 MW | 5min | 2ms | Good for linear trends |
| Exponential Smoothing | 17.2% | 756 MW | 601 MW | 2min | 1ms | Handles seasonality |
| XGBoost | 15.8% | 698 MW | 556 MW | 15min | 5ms | Captures non-linear patterns |
| LSTM | 14.2% | 623 MW | 498 MW | 45min | 15ms | Good for sequences |
| **Hybrid System** | **12.4%** | **543 MW** | **432 MW** | **60min** | **25ms** | **Selected** |

**Selected Model**: Hybrid Time Series Forecasting System
- **Reason**: Best accuracy with reasonable computational requirements
- **Architecture**: Ensemble of LSTM, ARIMA, and XGBoost with dynamic weighting

### Hyperparameter Tuning

```python
best_params = {
    'lstm': {
        'sequence_length': 168,  # 1 week of hourly data
        'hidden_units': [128, 64],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 100
    },
    'arima': {
        'order': (2, 1, 2),  # (p, d, q)
        'seasonal_order': (1, 1, 1, 24)  # (P, D, Q, s) for daily seasonality
    },
    'xgboost': {
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    },
    'ensemble': {
        'method': 'dynamic_weighting',
        'validation_window': 168  # Use last week for weight adjustment
    }
}
```

### Training Process

```python
def train_hybrid_forecaster(model, train_data, val_data, test_data, 
                           sequence_length=168, forecast_horizon=24):
    """Training loop for hybrid time series forecaster"""
    
    # Prepare sequences for LSTM
    X_train, y_train = prepare_sequences(train_data, sequence_length, forecast_horizon)
    X_val, y_val = prepare_sequences(val_data, sequence_length, forecast_horizon)
    
    # Fit the model
    model.fit(X_train, y_train, X_val, y_val)
    
    # Validate on test set
    test_results = evaluate_forecaster(model, test_data, sequence_length, forecast_horizon)
    
    print(f"Test MAPE: {test_results['mape']:.3f}")
    print(f"Test RMSE: {test_results['rmse']:.3f}")
    print(f"Test MAE: {test_results['mae']:.3f}")
    
    return model, test_results

def evaluate_forecaster(model, test_data, sequence_length, forecast_horizon):
    """Evaluate forecaster on test data"""
    
    # Generate rolling forecasts
    predictions = []
    actuals = []
    
    for i in range(len(test_data) - sequence_length - forecast_horizon + 1):
        # Prepare input sequence
        input_seq = test_data[i:(i + sequence_length)].reshape(1, sequence_length, -1)
        
        # Make prediction
        pred_dict = model.predict(input_seq)
        pred = pred_dict['prediction'][0]  # First sample in batch
        
        # Get actual values
        actual = test_data[(i + sequence_length):(i + sequence_length + forecast_horizon)]
        
        predictions.extend(pred)
        actuals.extend(actual)
    
    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    mae = np.mean(np.abs(actuals - predictions))
    
    return {
        'mape': mape,
        'rmse': rmse,
        'mae': mae,
        'predictions': predictions,
        'actuals': actuals
    }

def train_lstm_model(lstm_model, X_train, y_train, X_val, y_val, epochs=100):
    """Training loop for LSTM component"""
    
    optimizer = Adam(learning_rate=0.001)
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch_idx in range(0, len(X_train), 32):
            X_batch = X_train[batch_idx:batch_idx+32]
            y_batch = y_train[batch_idx:batch_idx+32]
            
            # Forward pass
            predictions = lstm_model.forward(X_batch)
            
            # Compute loss
            loss = mse_loss(predictions, y_batch)
            
            # Backward pass
            gradients = compute_gradients(loss, lstm_model)
            optimizer.update(lstm_model, gradients)
            
            total_loss += loss
        
        # Validation
        val_loss = validate_lstm(lstm_model, X_val, y_val)
        
        if epoch % 10 == 0:
            print(f'LSTM Epoch {epoch+1}/{epochs}, '
                  f'Train Loss: {total_loss/len(X_train)*32:.4f}, '
                  f'Val Loss: {val_loss:.4f}')
```

### Cross-Validation
- **Strategy**: Time series cross-validation with expanding window
- **Validation MAPE**: 12.6% +/- 0.8%
- **Test MAPE**: 12.4%

---

## Production Deployment

### Infrastructure

**Cloud Architecture**:
- Kubernetes cluster with GPU nodes for LSTM inference
- Apache Kafka for real-time data streaming
- Redis for caching recent forecasts
- PostgreSQL for historical data storage
- InfluxDB for time-series metrics

### Software Architecture

```
Real-Time Data -> Stream Processor -> Feature Engineering -> Model Inference -> Confidence Estimation -> API
       |                |                      |                    |                |                    |
       v                v                      v                    v                v                    v
Weather APIs    Kafka Consumer        Lagged Features      Ensemble Model    Prediction Intervals    FastAPI
SCADA Data      Feature Store         Rolling Statistics   Dynamic Weights   Uncertainty Quant.    Load Balancer
```

### Real-Time Forecasting Pipeline

```python
import asyncio
import aioredis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import json

class RealTimeEnergyForecaster:
    def __init__(self, model_path, cache_ttl=3600):
        self.model = load_model(model_path)
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.redis_client = aioredis.from_url("redis://localhost")
        self.cache_ttl = cache_ttl
        
    async def get_forecast(self, region_id, forecast_horizon_hours=24):
        """Get real-time forecast for specific region"""
        
        # Check cache first
        cache_key = f"forecast:{region_id}:{forecast_horizon_hours}"
        cached_forecast = await self.redis_client.get(cache_key)
        
        if cached_forecast:
            return json.loads(cached_forecast)
        
        # Get recent data for the region
        recent_data = await self._get_recent_data(region_id, hours_back=168)
        
        # Prepare features
        features = await self._prepare_features(recent_data)
        
        # Generate forecast
        forecast = await self._generate_forecast(features, forecast_horizon_hours)
        
        # Cache result
        await self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(forecast))
        
        return forecast
    
    async def _get_recent_data(self, region_id, hours_back):
        """Get recent data for region from database"""
        # Query database for recent data
        query = f"""
        SELECT timestamp, demand, temperature, humidity, wind_speed, 
               is_holiday, is_weekend, hour, month
        FROM energy_data 
        WHERE region_id = {region_id} 
        AND timestamp >= NOW() - INTERVAL '{hours_back} hours'
        ORDER BY timestamp DESC
        LIMIT {hours_back}
        """
        
        data = await fetch_data_from_db(query)
        return pd.DataFrame(data)
    
    async def _prepare_features(self, recent_data):
        """Prepare features for model input"""
        # Reverse to get chronological order
        recent_data = recent_data[::-1]
        
        # Create lagged features
        for lag in [1, 2, 3, 6, 12, 24, 168]:
            recent_data[f'demand_lag_{lag}h'] = recent_data['demand'].shift(lag)
        
        # Create rolling statistics
        for window in [24, 168, 720]:
            recent_data[f'demand_ma_{window}h'] = recent_data['demand'].rolling(
                window=window, min_periods=1
            ).mean()
            recent_data[f'demand_std_{window}h'] = recent_data['demand'].rolling(
                window=window, min_periods=1
            ).std()
        
        # Create cyclical features
        recent_data['hour_sin'] = np.sin(2 * np.pi * recent_data['hour'] / 24)
        recent_data['hour_cos'] = np.cos(2 * np.pi * recent_data['hour'] / 24)
        recent_data['month_sin'] = np.sin(2 * np.pi * recent_data['month'] / 12)
        recent_data['month_cos'] = np.cos(2 * np.pi * recent_data['month'] / 12)
        
        # Fill NaN values
        recent_data = recent_data.fillna(method='ffill').fillna(0)
        
        return recent_data
    
    async def _generate_forecast(self, features, forecast_horizon_hours):
        """Generate forecast using the trained model"""
        
        # Prepare sequence for LSTM
        sequence_cols = [col for col in features.columns if col != 'timestamp']
        sequence_data = features[sequence_cols].values
        
        # Normalize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(sequence_data)
        
        # Create sequence
        sequence = scaled_data[-168:]  # Last week of data
        sequence = sequence.reshape(1, 168, -1)  # Add batch dimension
        
        # Run inference in thread pool
        loop = asyncio.get_event_loop()
        forecast_result = await loop.run_in_executor(
            self.executor,
            self._run_model_inference,
            sequence
        )
        
        # Extract results
        predictions = forecast_result['prediction'][0]
        lower_bound = forecast_result['lower_bound'][0]
        upper_bound = forecast_result['upper_bound'][0]
        confidence = forecast_result['confidence'][0]
        
        # Convert to list for JSON serialization
        predictions = predictions.tolist()
        lower_bound = lower_bound.tolist()
        upper_bound = upper_bound.tolist()
        
        # Create timestamps for forecast period
        last_timestamp = features['timestamp'].iloc[-1]
        forecast_timestamps = [
            last_timestamp + timedelta(hours=i+1) 
            for i in range(forecast_horizon_hours)
        ]
        
        return {
            'timestamps': [ts.isoformat() for ts in forecast_timestamps],
            'predictions': predictions,
            'lower_bounds': lower_bound,
            'upper_bounds': upper_bound,
            'confidence': float(confidence),
            'forecast_horizon_hours': forecast_horizon_hours
        }
    
    def _run_model_inference(self, sequence):
        """Run model inference (executed in thread pool)"""
        with torch.no_grad():
            result = self.model.predict(torch.tensor(sequence, dtype=torch.float32))
        return result

# API Implementation
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Energy Demand Forecasting API")

class ForecastRequest(BaseModel):
    region_id: int
    forecast_horizon_hours: int = 24
    include_confidence: bool = True

class ForecastResponse(BaseModel):
    region_id: int
    forecast_horizon_hours: int
    timestamps: List[str]
    predictions: List[float]
    lower_bounds: Optional[List[float]] = None
    upper_bounds: Optional[List[float]] = None
    confidence: Optional[float] = None
    processing_time_ms: float

forecaster = RealTimeEnergyForecaster(model_path="hybrid_energy_forecaster_v1.pkl")

@app.post("/forecast", response_model=ForecastResponse)
async def get_energy_forecast(request: ForecastRequest):
    try:
        start_time = time.time()
        
        forecast = await forecaster.get_forecast(
            request.region_id,
            request.forecast_horizon_hours
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        response = ForecastResponse(
            region_id=request.region_id,
            forecast_horizon_hours=request.forecast_horizon_hours,
            timestamps=forecast['timestamps'],
            predictions=forecast['predictions'],
            processing_time_ms=processing_time
        )
        
        if request.include_confidence:
            response.lower_bounds = forecast['lower_bounds']
            response.upper_bounds = forecast['upper_bounds']
            response.confidence = forecast['confidence']
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.get("/metrics/{region_id}")
async def get_region_metrics(region_id: int):
    """Get historical accuracy metrics for a region"""
    # Query database for historical metrics
    metrics = await get_historical_metrics(region_id)
    return metrics
```

### Data Pipeline and Feature Engineering

```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import pandas as pd

class EnergyDataPipeline(beam.PTransform):
    """Apache Beam pipeline for energy data processing"""
    
    def __init__(self, output_table):
        self.output_table = output_table
    
    def expand(self, pcoll):
        return (
            pcoll
            | 'Parse JSON' >> beam.Map(parse_json_line)
            | 'Add Timestamps' >> beam.Map(add_timestamp)
            | 'Group by Region' >> beam.GroupByKey()
            | 'Create Features' >> beam.ParDo(CreateFeaturesFn())
            | 'Write to BigQuery' >> beam.io.WriteToBigQuery(
                table=self.output_table,
                schema='region_id:INTEGER,timestamp:TIMESTAMP,demand:FLOAT,temperature:FLOAT,...',
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
            )
        )

class CreateFeaturesFn(beam.DoFn):
    """Beam DoFn to create time series features"""
    
    def process(self, element):
        region_id, data_list = element
        df = pd.DataFrame(data_list)
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Create features
        df = self.create_time_features(df)
        df = self.create_lagged_features(df)
        df = self.create_rolling_features(df)
        
        # Convert to dictionary format for BigQuery
        for _, row in df.iterrows():
            yield {
                'region_id': region_id,
                'timestamp': row['timestamp'].isoformat(),
                'demand': float(row['demand']),
                'temperature': float(row['temperature']),
                'humidity': float(row['humidity']),
                'demand_lag_1h': float(row.get('demand_lag_1h', 0)),
                'demand_ma_24h': float(row.get('demand_ma_24h', 0)),
                # ... other features
            }
    
    def create_time_features(self, df):
        """Create time-based features"""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        return df
    
    def create_lagged_features(self, df):
        """Create lagged demand features"""
        for lag in [1, 2, 3, 6, 12, 24, 168]:
            df[f'demand_lag_{lag}h'] = df['demand'].shift(lag)
        
        return df
    
    def create_rolling_features(self, df):
        """Create rolling statistics features"""
        for window in [24, 168, 720]:
            df[f'demand_ma_{window}h'] = df['demand'].rolling(window=window, min_periods=1).mean()
            df[f'demand_std_{window}h'] = df['demand'].rolling(window=window, min_periods=1).std()
        
        return df

# Schedule the pipeline to run every 15 minutes
def run_data_pipeline():
    """Run the data pipeline"""
    pipeline_options = PipelineOptions([
        '--runner=DataflowRunner',
        '--project=my-project',
        '--region=us-central1',
        '--temp_location=gs://my-bucket/temp',
        '--staging_location=gs://my-bucket/staging'
    ])
    
    with beam.Pipeline(options=pipeline_options) as pipeline:
        (
            pipeline
            | 'Read from Pub/Sub' >> beam.io.ReadFromPubSub(topic='projects/my-project/topics/energy-data')
            | 'Process Energy Data' >> EnergyDataPipeline(output_table='my_dataset.energy_features')
        )
```

### Operational SLOs and Runbook
- **Forecast Accuracy**: Maintain MAPE <15%; trigger retraining if exceeded
- **Latency**: p99 <500ms; auto-scale inference instances if exceeded
- **Availability**: 99.9% during peak demand periods; 99.5% off-peak
- **Runbook Highlights**:
  - Model drift: monitor accuracy daily, retrain weekly
  - Data quality: validate input features, alert on anomalies
  - Capacity planning: scale resources before peak seasons

### Monitoring and Alerting
- **Metrics**: MAPE, RMSE, MAE, coverage of confidence intervals, inference latency
- **Alerts**: Page if MAPE exceeds 15% or if confidence intervals don't cover actual values >10% of time
- **Drift Detection**: Monitor feature distributions and trigger alerts for significant shifts

---

## Results & Impact

### Model Performance in Production

**Overall Performance**:
- **MAPE**: 12.4%
- **RMSE**: 543 MW
- **MAE**: 432 MW
- **Confidence Interval Coverage**: 94.2% (close to target 95%)
- **Inference Time**: 25ms (p99)

**Per-Region Performance**:
| Region | Population | MAPE | Peak Demand (MW) | Performance Notes |
|--------|------------|------|------------------|-------------------|
| Urban Core | 800K | 10.8% | 8,000 | High density, predictable |
| Suburban East | 400K | 11.9% | 3,200 | Mixed residential/industrial |
| Rural North | 150K | 14.2% | 1,200 | Weather dependent |
| Industrial West | 200K | 13.1% | 4,500 | Manufacturing patterns |
| Tourist South | 100K | 16.7% | 2,100 | Seasonal variation high |

### Business Impact (12 months post-launch)

| Metric | Before ML | After ML | Improvement |
|--------|-----------|----------|-------------|
| **Forecast Accuracy (MAPE)** | 28.5% | 12.4% | **-56.5%** |
| **Energy Waste Reduction** | - | - | **23%** |
| **Operational Cost Savings** | - | - | **$42M annually** |
| **Blackout Prevention** | 1-2 major/year | 0 | **100% reduction** |
| **Grid Stability** | 87% | 92% | **+5.7%** |
| **Peak Demand Prediction** | 65% accurate | 89% accurate | **+36.9%** |
| **Renewable Integration** | 45% | 62% | **+37.8%** |

### Cost-Benefit Analysis

**Annual Benefits**:
- Reduced energy waste: $18M
- Avoided blackout costs: $15M
- Operational efficiency: $6M
- Renewable integration value: $3M
- **Total Annual Benefit**: $42M

**Investment**:
- Model development: $2.5M
- Infrastructure: $1.8M
- Integration: $1.2M
- **Total Investment**: $5.5M

**ROI**: 665% in first year ($42M/$5.5M)

### Key Insights from Analysis

**Most Important Predictors**:
1. **Temperature**: 0.28 (heating/cooling demand)
2. **Historical Demand**: 0.22 (autocorrelation)
3. **Hour of Day**: 0.15 (daily patterns)
4. **Day of Week**: 0.08 (weekday vs weekend)
5. **Humidity**: 0.07 (affects cooling load)
6. **Wind Speed**: 0.06 (affects heating/cooling)
7. **Holiday Status**: 0.05 (reduced demand)
8. **Month**: 0.04 (seasonal patterns)
9. **Precipitation**: 0.03 (lighting demand)
10. **Economic Indicators**: 0.02 (industrial demand)

**Extreme Event Patterns**:
- Heat waves: Demand increases 40-60% above normal
- Cold snaps: Similar increase due to heating
- Storms: Variable impact depending on type
- Economic downturns: 5-15% demand reduction

---

## Challenges & Solutions

### Challenge 1: Extreme Weather Events
- **Problem**: Models struggled with unprecedented weather conditions
- **Solution**:
  - Added extreme weather indicators as features
  - Implemented anomaly detection for unusual conditions
  - Created separate models for extreme events
  - Added safety margins during extreme forecasts

### Challenge 2: Multi-Step Forecasting Accuracy
- **Problem**: Accuracy degraded significantly for longer horizons
- **Solution**:
  - Used recursive prediction with error correction
  - Implemented direct multi-output models
  - Added ensemble methods with horizon-specific weights
  - Created hierarchical models for different time scales

### Challenge 3: Real-Time Data Integration
- **Problem**: Need to incorporate latest data for accurate short-term forecasts
- **Solution**:
  - Built streaming pipeline with 15-minute updates
  - Implemented online learning for rapid adaptation
  - Created micro-batching for efficient inference
  - Added data quality checks and fallback mechanisms

### Challenge 4: Interpretability for Operators
- **Problem**: Grid operators needed to understand forecast reasoning
- **Solution**:
  - Added SHAP values for feature importance
  - Created dashboard showing prediction components
  - Implemented scenario analysis tools
  - Provided confidence intervals and uncertainty estimates

---

## Lessons Learned

### What Worked

1. **Hybrid Approach**
   - Combining multiple models improved robustness
   - Each model handled different aspects effectively
   - Ensemble methods provided better generalization

2. **Rich Feature Engineering**
   - Lagged features captured temporal dependencies
   - Cyclical encoding handled periodic patterns
   - Interaction features captured complex relationships

3. **Real-Time Adaptation**
   - Streaming pipeline enabled rapid updates
   - Online learning maintained model freshness
   - Dynamic reweighting adapted to changing conditions

### What Didn't Work

1. **Pure Deep Learning Approach**
   - LSTM alone was too slow for real-time requirements
   - Required too much data for stable training
   - Difficult to interpret for operators

2. **Static Model Deployment**
   - Monthly retraining too slow for changing patterns
  - Seasonal adjustments needed more frequent updates
  - Switched to continuous learning approach

---

## Technical Implementation

### LSTM Model Implementation

```python
import torch
import torch.nn as nn
import numpy as np

class MultiStepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(MultiStepLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers for output
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, output_size)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use the last output for prediction
        last_output = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Fully connected layers
        x = self.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc_out(x)
        
        return output

def train_lstm_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """Training loop for LSTM model"""
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                outputs = model(data)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_lstm_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= 20:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}')
    
    return model

# ARIMA Model Implementation
from statsmodels.tsa.arima.model import ARIMA as StatsModelsARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

class ARIMAWrapper:
    def __init__(self, order=(2, 1, 2), seasonal_order=(1, 1, 1, 24)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
    
    def fit(self, series):
        """Fit ARIMA model to time series"""
        try:
            # Fit the model
            self.fitted_model = StatsModelsARIMA(
                series, 
                order=self.order, 
                seasonal_order=self.seasonal_order
            ).fit()
        except Exception as e:
            # Fallback to simpler model if seasonal fails
            print(f"Seasonal ARIMA failed: {e}, trying regular ARIMA")
            self.fitted_model = StatsModelsARIMA(series, order=self.order).fit()
    
    def predict(self, steps):
        """Generate predictions"""
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast.values

# XGBoost Model Implementation
import xgboost as xgb

class XGBoostTimeSeries:
    def __init__(self, **kwargs):
        self.params = {
            'objective': 'reg:squarederror',
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            **kwargs
        }
        self.model = xgb.XGBRegressor(**self.params)
    
    def fit(self, X, y):
        """Fit XGBoost model"""
        self.model.fit(X, y)
    
    def predict(self, X):
        """Generate predictions"""
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        return self.model.feature_importances_
```

### Data Preprocessing Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class EnergyDataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.feature_columns = None
    
    def fit_transform(self, df, target_col='demand'):
        """Fit scalers and transform the data"""
        
        # Create features
        df = self.create_features(df)
        
        # Identify feature columns (excluding target and timestamp)
        self.feature_columns = [col for col in df.columns 
                               if col not in [target_col, 'timestamp', 'region_id']]
        
        # Scale features
        for col in self.feature_columns:
            if col not in self.scalers:
                self.scalers[col] = StandardScaler()
            df[col] = self.scalers[col].fit_transform(df[[col]])
        
        # Scale target separately
        self.target_scaler = StandardScaler()
        df[target_col] = self.target_scaler.fit_transform(df[[target_col]])
        
        return df
    
    def transform(self, df):
        """Transform new data using fitted scalers"""
        
        df = self.create_features(df)
        
        # Scale features using fitted scalers
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = self.scalers[col].transform(df[[col]])
        
        # Scale target if present
        if 'demand' in df.columns:
            df['demand'] = self.target_scaler.transform(df[['demand']])
        
        return df
    
    def create_features(self, df):
        """Create comprehensive time series features"""
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Lagged features
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:  # Various time lags
            df[f'demand_lag_{lag}h'] = df.groupby('region_id')['demand'].shift(lag)
        
        # Rolling statistics
        for window in [24, 168, 720]:  # 1 day, 1 week, 1 month
            df[f'demand_ma_{window}h'] = df.groupby('region_id')['demand'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'demand_std_{window}h'] = df.groupby('region_id')['demand'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            df[f'demand_max_{window}h'] = df.groupby('region_id')['demand'].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
            df[f'demand_min_{window}h'] = df.groupby('region_id')['demand'].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
        
        # Weather-derived features
        df['temp_lag_24h'] = df.groupby('region_id')['temperature'].transform(
            lambda x: x.shift(24)
        )
        df['temp_change_24h'] = df['temperature'] - df['temp_lag_24h']
        df['cooling_degree_days'] = np.maximum(0, df['temperature'] - 65)
        df['heating_degree_days'] = np.maximum(0, 65 - df['temperature'])
        
        # Interaction features
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        df['weekend_hot_weather'] = df['is_weekend'] * np.maximum(0, df['temperature'] - 85)
        df['holiday_cold_weather'] = df['is_holiday'] * np.maximum(0, 32 - df['temperature'])
        
        # Trend features
        df['demand_trend_short'] = df.groupby('region_id')['demand'].transform(
            lambda x: x.rolling(window=12, min_periods=1).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0)
        )
        df['demand_trend_long'] = df.groupby('region_id')['demand'].transform(
            lambda x: x.rolling(window=168, min_periods=24).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0)
        )
        
        return df
    
    def create_sequences(self, df, sequence_length=168, forecast_horizon=24):
        """Create sequences for LSTM training"""
        
        X, y = [], []
        
        # Group by region to create sequences within each region
        for region_id in df['region_id'].unique():
            region_data = df[df['region_id'] == region_id].sort_values('timestamp')
            
            # Get feature columns
            feature_cols = [col for col in self.feature_columns if col in region_data.columns]
            features = region_data[feature_cols].values
            targets = region_data['demand'].values
            
            # Create sequences
            for i in range(len(region_data) - sequence_length - forecast_horizon + 1):
                X.append(features[i:(i + sequence_length)])
                y.append(targets[(i + sequence_length):(i + sequence_length + forecast_horizon)])
        
        return np.array(X), np.array(y)

def prepare_time_series_data(df, sequence_length=168, forecast_horizon=24, test_size=0.2):
    """Prepare time series data for training"""
    
    preprocessor = EnergyDataPreprocessor()
    
    # Fit and transform the data
    df_processed = preprocessor.fit_transform(df)
    
    # Create sequences
    X, y = preprocessor.create_sequences(df_processed, sequence_length, forecast_horizon)
    
    # Split into train and test (maintaining temporal order)
    split_idx = int(len(X) * (1 - test_size))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, preprocessor
```

---

## Next Steps & Future Improvements

### Short-Term (Q1 2026)
- [ ] Implement probabilistic forecasting with quantile regression
- [ ] Add renewable energy generation forecasts
- [ ] Enhance extreme event detection and response

### Medium-Term (Q2-Q3 2026)
- [ ] Extend to distributed energy resources (DERs)
- [ ] Implement reinforcement learning for dynamic pricing
- [ ] Add climate change impact modeling

### Long-Term (2027)
- [ ] Develop physics-informed neural networks
- [ ] Integrate with smart grid control systems
- [ ] Implement federated learning across utilities

---

## Mathematical Foundations

### Time Series Decomposition
A time series can be decomposed as:
```
Y(t) = T(t) + S(t) + C(t) + I(t)
```
Where:
- Y(t) is the observed value at time t
- T(t) is the trend component
- S(t) is the seasonal component
- C(t) is the cyclical component
- I(t) is the irregular/random component

### ARIMA Model
The ARIMA(p,d,q) model is defined as:
```
φ(B)(1-B)^d X_t = θ(B)ε_t
```
Where:
- φ(B) = 1 - φ₁B - φ₂B² - ... - φₚBᵖ (AR polynomial)
- θ(B) = 1 - θ₁B - θ₂B² - ... - θ_qB^q (MA polynomial)
- B is the backshift operator (BX_t = X_{t-1})
- ε_t is white noise

### LSTM Equations
For an LSTM cell:
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)     # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     # Input gate
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)     # Output gate
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)  # Candidate values
c_t = f_t * c_{t-1} + i_t * g_t         # Cell state
h_t = o_t * tanh(c_t)                   # Hidden state
```

### Evaluation Metrics
Mean Absolute Percentage Error (MAPE):
```
MAPE = (1/n) * Σ |(A_t - F_t) / A_t| * 100
```

Root Mean Squared Error (RMSE):
```
RMSE = √[(1/n) * Σ (A_t - F_t)²]
```

Mean Absolute Error (MAE):
```
MAE = (1/n) * Σ |A_t - F_t|
```

Where A_t is actual value and F_t is forecasted value at time t.

### Confidence Intervals
For prediction intervals at confidence level (1-α):
```
Lower Bound = ŷ_t - z_{α/2} * σ_t
Upper Bound = ŷ_t + z_{α/2} * σ_t
```
Where σ_t is the estimated standard deviation of forecast errors at time t.

---

## Production Considerations

### Scalability
- **Distributed Training**: Use Horovod for multi-GPU LSTM training
- **Model Serving**: TensorRT optimization for faster inference
- **Caching Strategy**: Regional forecast caching with TTL
- **Load Balancing**: Geographic distribution of inference services

### Security
- **Data Encryption**: Encrypt time series data in transit and at rest
- **Access Control**: Role-based access to forecasting API
- **Audit Logging**: Track all forecast requests and usage

### Reliability
- **Redundancy**: Multiple model instances across regions
- **Graceful Degradation**: Fallback to simpler models if primary fails
- **Disaster Recovery**: Automated backup and restore of models

### Performance Optimization
- **Model Compression**: Quantization for faster LSTM inference
- **Batch Processing**: Process multiple regions together when possible
- **Feature Caching**: Pre-compute and cache rolling statistics

---

## Conclusion

This advanced energy demand forecasting system demonstrates sophisticated time series ML engineering:
- **Hybrid Architecture**: Combines LSTM, ARIMA, and XGBoost models
- **Real-Time Processing**: Streaming data pipeline with 15-minute updates
- **Scalable Infrastructure**: Handles 50 regions with hourly forecasts
- **Business Impact**: $42M annual savings, 56.5% accuracy improvement

**Key takeaway**: Effective time series forecasting requires rich feature engineering, model ensembling, and real-time adaptation capabilities.

Architecture and ops blueprint: `docs/system_design_solutions/18_energy_forecasting_system.md`.

---

**Contact**: Implementation details in `src/time_series/energy_forecasting.py`.
Notebooks: `notebooks/case_studies/energy_demand_forecasting.ipynb`