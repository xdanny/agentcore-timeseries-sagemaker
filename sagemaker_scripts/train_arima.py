#!/usr/bin/env python3
"""
SageMaker Training Script for ARIMA Time Series Model
This script runs inside SageMaker training containers
"""
import argparse
import json
import os
import pandas as pd
import numpy as np
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def train_arima_model(train_data, p, d, q, P, D, Q, m):
    """
    Train SARIMAX model with given hyperparameters

    Args:
        train_data: Training time series data
        p, d, q: Non-seasonal ARIMA orders
        P, D, Q: Seasonal ARIMA orders
        m: Seasonal period

    Returns:
        Fitted model
    """
    model = SARIMAX(
        train_data,
        order=(p, d, q),
        seasonal_order=(P, D, Q, m),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_model = model.fit(disp=False, maxiter=200)
    return fitted_model


def evaluate_model(model, train_data, test_data):
    """
    Evaluate model performance

    Returns:
        Dictionary with metrics
    """
    # In-sample predictions
    train_pred = model.fittedvalues

    # Out-of-sample forecast
    forecast_steps = len(test_data)
    forecast = model.forecast(steps=forecast_steps)

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(train_data[len(train_data)-len(train_pred):], train_pred))
    test_rmse = np.sqrt(mean_squared_error(test_data, forecast))
    test_mae = mean_absolute_error(test_data, forecast)

    return {
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'aic': float(model.aic),
        'bic': float(model.bic)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters - SageMaker uses single dash -p not --p
    parser.add_argument('-p', '--p', dest='p', type=int, default=2)
    parser.add_argument('-d', '--d', dest='d', type=int, default=1)
    parser.add_argument('-q', '--q', dest='q', type=int, default=2)
    parser.add_argument('--seasonal-p', dest='P', type=int, default=1)
    parser.add_argument('--seasonal-d', dest='D', type=int, default=1)
    parser.add_argument('--seasonal-q', dest='Q', type=int, default=1)
    parser.add_argument('-m', '--m', dest='m', type=int, default=7)

    # SageMaker paths
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))

    args = parser.parse_args()

    print(f"Training ARIMA model with parameters:")
    print(f"  Non-seasonal: ({args.p}, {args.d}, {args.q})")
    print(f"  Seasonal: ({args.P}, {args.D}, {args.Q}, {args.m})")

    # Load data
    train_files = [f for f in os.listdir(args.train) if f.endswith('.csv')]
    if not train_files:
        raise ValueError(f"No CSV files found in {args.train}")

    train_file = os.path.join(args.train, train_files[0])
    print(f"Loading data from: {train_file}")

    df = pd.read_csv(train_file)

    # Assume last column is target, or look for 'value' column
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
    print("Training ARIMA model...")
    model = train_arima_model(
        train_data,
        args.p, args.d, args.q,
        args.P, args.D, args.Q, args.m
    )

    # Evaluate
    print("Evaluating model...")
    metrics = evaluate_model(model, train_data, test_data)

    print(f"\nModel Performance:")
    print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"  Test MAE: {metrics['test_mae']:.4f}")
    print(f"  AIC: {metrics['aic']:.2f}")
    print(f"  BIC: {metrics['bic']:.2f}")

    # Print metric for SageMaker tuning (must match regex in tuning job)
    print(f"RMSE: {metrics['test_rmse']:.4f}")

    # Save model
    model_path = os.path.join(args.model_dir, 'model.pkl')
    print(f"\nSaving model to: {model_path}")

    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'hyperparameters': {
                'p': args.p, 'd': args.d, 'q': args.q,
                'P': args.P, 'D': args.D, 'Q': args.Q, 'm': args.m
            },
            'metrics': metrics,
            'target_column': target_col
        }, f)

    # Save metrics
    metrics_path = os.path.join(args.output_data_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("Training completed successfully!")
