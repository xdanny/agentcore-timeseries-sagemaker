#!/usr/bin/env python3
"""
SageMaker Inference Script for LSTM Model
Handles model loading and prediction requests
"""
import json
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras


def model_fn(model_dir):
    """Load LSTM model from model_dir"""
    model_path = os.path.join(model_dir, 'lstm_model.h5')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    metadata_path = os.path.join(model_dir, 'metadata.pkl')
    
    model = keras.models.load_model(model_path)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"LSTM model loaded successfully")
    print(f"Hyperparameters: {metadata['hyperparameters']}")
    
    return {
        'model': model,
        'scaler': scaler,
        'metadata': metadata
    }


def input_fn(request_body, content_type='application/json'):
    """Deserialize input data"""
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model_dict):
    """Make predictions with LSTM"""
    model = model_dict['model']
    scaler = model_dict['scaler']
    metadata = model_dict['metadata']
    
    lookback = metadata['hyperparameters']['lookback']
    steps = input_data.get('steps', 7)
    
    # Get historical data for forecasting
    historical_data = np.array(input_data.get('historical_data', []))
    
    if len(historical_data) < lookback:
        raise ValueError(f"Need at least {lookback} historical data points, got {len(historical_data)}")
    
    print(f"Generating {steps}-step forecast with lookback={lookback}")
    
    # Scale historical data
    historical_scaled = scaler.transform(historical_data.reshape(-1, 1))
    
    # Generate multi-step forecast
    forecasts = []
    current_sequence = historical_scaled[-lookback:].copy()
    
    for _ in range(steps):
        # Reshape for LSTM input
        X = current_sequence.reshape(1, lookback, 1)
        
        # Predict next value
        pred_scaled = model.predict(X, verbose=0)
        forecasts.append(pred_scaled[0, 0])
        
        # Update sequence (rolling window)
        current_sequence = np.append(current_sequence[1:], pred_scaled[0, 0])
    
    # Inverse transform predictions
    forecasts_array = np.array(forecasts).reshape(-1, 1)
    forecasts_original = scaler.inverse_transform(forecasts_array).flatten()
    
    predictions = {
        'forecast': forecasts_original.tolist(),
        'steps': steps,
        'model_type': 'LSTM',
        'hyperparameters': metadata['hyperparameters'],
        'metrics': metadata.get('metrics', {})
    }
    
    return predictions


def output_fn(prediction, accept='application/json'):
    """Serialize prediction output"""
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
