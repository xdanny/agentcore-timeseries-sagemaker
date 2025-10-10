#!/usr/bin/env python3
"""
Working SageMaker inference script for ARIMA models
Handles the model.pkl format from training
"""
import json
import os
import pickle


def model_fn(model_dir):
    """Load the trained model from model.pkl"""
    model_path = os.path.join(model_dir, 'model.pkl')

    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)

    # Return the actual ARIMA model from the dict
    return model_dict['model']


def input_fn(request_body, content_type='application/json'):
    """Parse input - expects {"steps": N}"""
    if content_type == 'application/json':
        data = json.loads(request_body)
        return data.get('steps', 7)
    elif content_type == 'text/csv':
        return int(request_body.strip())
    return 7


def predict_fn(steps, model):
    """Generate forecast using the trained ARIMA model"""
    try:
        # Generate forecast
        forecast = model.forecast(steps=steps)

        # Get confidence intervals
        forecast_obj = model.get_forecast(steps=steps)
        conf_int = forecast_obj.conf_int()

        return {
            'forecast': forecast.tolist(),
            'steps': steps,
            'lower_bound': conf_int.iloc[:, 0].tolist(),
            'upper_bound': conf_int.iloc[:, 1].tolist()
        }
    except Exception as e:
        # Return error in a way that won't crash the endpoint
        return {
            'forecast': [0.0] * steps,
            'steps': steps,
            'error': str(e),
            'lower_bound': [0.0] * steps,
            'upper_bound': [0.0] * steps
        }


def output_fn(prediction, accept='application/json'):
    """Format output as JSON"""
    return json.dumps(prediction), accept
