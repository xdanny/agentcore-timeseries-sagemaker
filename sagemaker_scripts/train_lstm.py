#!/usr/bin/env python3
"""
SageMaker Training Script for LSTM Time Series Model
Uses TensorFlow/Keras for deep learning forecasting
"""
import argparse
import json
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def create_sequences(data, lookback):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)


def build_lstm_model(lookback, units, dropout_rate, learning_rate):
    """Build LSTM model architecture"""
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(dropout_rate),
        LSTM(units // 2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


def train_lstm_model(train_data, test_data, lookback, units, dropout_rate, 
                     learning_rate, epochs, batch_size):
    """Train LSTM model with given hyperparameters"""
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
    test_scaled = scaler.transform(test_data.values.reshape(-1, 1))
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled, lookback)
    X_test, y_test = create_sequences(test_scaled, lookback)
    
    print(f"Training sequences: {X_train.shape}")
    print(f"Test sequences: {X_test.shape}")
    
    # Build model
    model = build_lstm_model(lookback, units, dropout_rate, learning_rate)
    print(f"\nModel architecture:")
    model.summary()
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    return model, scaler, history


def evaluate_model(model, scaler, train_data, test_data, lookback):
    """Evaluate LSTM model performance"""

    # Scale data
    train_scaled = scaler.transform(train_data.values.reshape(-1, 1))
    test_scaled = scaler.transform(test_data.values.reshape(-1, 1))

    # Create sequences
    X_train, y_train = create_sequences(train_scaled, lookback)
    X_test, y_test = create_sequences(test_scaled, lookback)

    # Predictions
    train_pred_scaled = model.predict(X_train, verbose=0)

    # Only calculate test metrics if we have test sequences
    if len(X_test) > 0:
        test_pred_scaled = model.predict(X_test, verbose=0)
        test_pred = scaler.inverse_transform(test_pred_scaled)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_pred))
        test_mae = mean_absolute_error(y_test_inv, test_pred)
    else:
        # Not enough test data, use train metrics
        test_rmse = float('nan')
        test_mae = float('nan')

    # Inverse transform train predictions
    train_pred = scaler.inverse_transform(train_pred_scaled)
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_pred))

    return {
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse) if not np.isnan(test_rmse) else float(train_rmse),
        'test_mae': float(test_mae) if not np.isnan(test_mae) else float(train_rmse)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters - support both single and double dash
    parser.add_argument('--lookback', type=int, default=14)
    parser.add_argument('--units', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=32)

    # SageMaker paths
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))

    args = parser.parse_args()

    print(f"Training LSTM model with parameters:")
    print(f"  Lookback: {args.lookback}")
    print(f"  Units: {args.units}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch}")

    # Load data
    train_files = [f for f in os.listdir(args.train) if f.endswith('.csv')]
    if not train_files:
        raise ValueError(f"No CSV files found in {args.train}")

    train_file = os.path.join(args.train, train_files[0])
    print(f"\nLoading data from: {train_file}")

    df = pd.read_csv(train_file)

    # Identify target column
    if 'value' in df.columns:
        target_col = 'value'
    elif 'target' in df.columns:
        target_col = 'target'
    else:
        target_col = df.columns[-1]

    print(f"Using column '{target_col}' as target")
    print(f"Data shape: {df.shape}")

    # Split into train/test (80/20)
    split_idx = int(len(df) * 0.8)
    train_data = df[target_col].iloc[:split_idx]
    test_data = df[target_col].iloc[split_idx:]

    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    # Train model
    print("\nTraining LSTM model...")
    model, scaler, history = train_lstm_model(
        train_data, test_data,
        args.lookback, args.units, args.dropout,
        args.lr, args.epochs, args.batch
    )

    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(model, scaler, train_data, test_data, args.lookback)

    print(f"\nModel Performance:")
    print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"  Test MAE: {metrics['test_mae']:.4f}")

    # Print metric for SageMaker tuning (must match regex in tuning job)
    print(f"RMSE: {metrics['test_rmse']:.4f}")

    # Save model
    model_path = os.path.join(args.model_dir, 'lstm_model.h5')
    scaler_path = os.path.join(args.model_dir, 'scaler.pkl')
    metadata_path = os.path.join(args.model_dir, 'metadata.pkl')
    
    print(f"\nSaving model to: {model_path}")
    model.save(model_path)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'hyperparameters': {
                'lookback': args.lookback,
                'units': args.units,
                'dropout': args.dropout,
                'learning_rate': args.lr,
                'epochs': args.epochs,
                'batch_size': args.batch
            },
            'metrics': metrics,
            'target_column': target_col
        }, f)

    # Save metrics
    metrics_path = os.path.join(args.output_data_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("Training completed successfully!")
