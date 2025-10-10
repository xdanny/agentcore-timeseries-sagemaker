import os
#!/usr/bin/env python3
"""
Intelligent Feature Engineering Agent
Analyzes EDA results and recommends optimal features based on data characteristics
"""
import json
import boto3
from strands import tool
from agents.code_interpreter_utils import invoke_with_retry

BUCKET = os.environ.get('BUCKET_NAME', 'sagemaker-forecasting-{}-{}'.format(os.environ.get('AWS_REGION', 'us-east-1'), boto3.client('sts').get_caller_identity()['Account']))
S3_CLIENT = boto3.client('s3', region_name='us-east-1')



@tool
def recommend_features(dataset_s3_path: str, eda_results: str) -> str:
    """
    Intelligent feature recommendation based on EDA insights.

    Analyzes:
    - ACF/PACF patterns for lag selection
    - Seasonality strength for rolling window sizes
    - Trend characteristics for differencing features
    - Stationarity tests for transformation needs

    Returns JSON with:
    - Recommended features with justifications
    - Feature importance predictions
    - User-selectable feature categories
    """
    # Parse inputs
    eda = json.loads(eda_results)

    if dataset_s3_path.startswith('s3://'):
        parts = dataset_s3_path.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1]
    else:
        bucket = BUCKET
        key = dataset_s3_path

    code = f'''
import pandas as pd
import numpy as np
import json
import subprocess

# Load data using AWS CLI (Code Interpreter has AWS credentials via IAM role)
subprocess.run(['aws', 's3', 'cp', 's3://{bucket}/{key}', '/tmp/data.csv'], check=True)
df = pd.read_csv('/tmp/data.csv')

# Auto-detect columns
time_col = None
value_col = None
for col in df.columns:
    if any(kw in col.lower() for kw in ['date', 'time', 'ds', 'timestamp']):
        time_col = col
    if any(kw in col.lower() for kw in ['sales', 'value', 'y', 'price', 'demand']):
        value_col = col

if not value_col:
    value_col = df.select_dtypes(include=[np.number]).columns[0]

df[time_col] = pd.to_datetime(df[time_col])
df = df.sort_values(time_col).reset_index(drop=True)

# Parse EDA results
eda = {repr(eda)}

# ===== INTELLIGENT FEATURE RECOMMENDATIONS =====
recommendations = {{}}

# 1. LAG FEATURES (based on ACF analysis)
if 'autocorrelation' in eda:
    sig_lags = eda['autocorrelation'].get('significant_acf_lags', [1, 7, 14])
    recommended_lags = sig_lags[:5] if sig_lags else [1, 7, 14]

    recommendations['lag_features'] = {{
        'enabled': True,
        'priority': 'high',
        'lags': recommended_lags,
        'justification': f"ACF analysis shows significant autocorrelation at lags {{recommended_lags}}",
        'estimated_importance': 0.85
    }}
else:
    recommendations['lag_features'] = {{
        'enabled': True,
        'priority': 'medium',
        'lags': [1, 7, 14],
        'justification': "Default lag features for time series",
        'estimated_importance': 0.7
    }}

# 2. ROLLING STATISTICS (based on seasonality)
if 'decomposition' in eda:
    seasonal_strength = eda['decomposition'].get('seasonal_strength', 0)
    if seasonal_strength > 0.6:
        windows = [7, 14, 30]
        priority = 'high'
        importance = 0.9
    elif seasonal_strength > 0.3:
        windows = [7, 14]
        priority = 'medium'
        importance = 0.7
    else:
        windows = [7]
        priority = 'low'
        importance = 0.5

    recommendations['rolling_features'] = {{
        'enabled': True,
        'priority': priority,
        'windows': windows,
        'statistics': ['mean', 'std', 'min', 'max'],
        'justification': f"Seasonal strength is {{seasonal_strength:.2f}}, {{'strong' if seasonal_strength > 0.6 else 'moderate'}} seasonality detected",
        'estimated_importance': importance
    }}
else:
    recommendations['rolling_features'] = {{
        'enabled': True,
        'priority': 'medium',
        'windows': [7, 14],
        'statistics': ['mean', 'std'],
        'justification': "Standard rolling window features",
        'estimated_importance': 0.6
    }}

# 3. DIFFERENCING FEATURES (based on stationarity)
if 'stationarity' in eda:
    recommended_d = eda['stationarity'].get('recommended_d', 0)
    if recommended_d > 0:
        recommendations['differencing_features'] = {{
            'enabled': True,
            'priority': 'high',
            'orders': list(range(1, recommended_d + 1)),
            'justification': f"Series requires {{recommended_d}} order differencing for stationarity",
            'estimated_importance': 0.8
        }}
    else:
        recommendations['differencing_features'] = {{
            'enabled': False,
            'priority': 'low',
            'orders': [],
            'justification': "Series is already stationary",
            'estimated_importance': 0.2
        }}
else:
    recommendations['differencing_features'] = {{
        'enabled': True,
        'priority': 'medium',
        'orders': [1],
        'justification': "First-order differencing as precaution",
        'estimated_importance': 0.5
    }}

# 4. CALENDAR FEATURES
recommendations['calendar_features'] = {{
    'enabled': True,
    'priority': 'medium',
    'features': ['day_of_week', 'month', 'quarter', 'is_weekend', 'day_of_month'],
    'justification': "Capture calendar patterns and business cycles",
    'estimated_importance': 0.6
}}

# 5. FOURIER FEATURES (for complex seasonality)
if 'decomposition' in eda and eda['decomposition'].get('seasonal_strength', 0) > 0.7:
    recommendations['fourier_features'] = {{
        'enabled': True,
        'priority': 'high',
        'k_terms': 3,
        'period': 7,
        'justification': "Strong seasonality detected, Fourier terms can capture complex patterns",
        'estimated_importance': 0.75
    }}
else:
    recommendations['fourier_features'] = {{
        'enabled': False,
        'priority': 'low',
        'k_terms': 0,
        'justification': "Seasonality not strong enough for Fourier features",
        'estimated_importance': 0.3
    }}

# 6. EXPANDING STATISTICS
recommendations['expanding_features'] = {{
    'enabled': True,
    'priority': 'low',
    'statistics': ['mean', 'std'],
    'justification': "Capture long-term trends and variance changes",
    'estimated_importance': 0.4
}}

# Summary
total_features = sum([
    len(recommendations['lag_features']['lags']),
    len(recommendations['rolling_features']['windows']) * len(recommendations['rolling_features']['statistics']),
    len(recommendations['differencing_features']['orders']) if recommendations['differencing_features']['enabled'] else 0,
    len(recommendations['calendar_features']['features']),
    recommendations['fourier_features']['k_terms'] * 2 if recommendations['fourier_features']['enabled'] else 0,
    len(recommendations['expanding_features']['statistics'])
])

recommendations['summary'] = {{
    'total_recommended_features': total_features,
    'high_priority_categories': [k for k, v in recommendations.items() if isinstance(v, dict) and v.get('priority') == 'high'],
    'estimated_total_importance': sum([v.get('estimated_importance', 0) for k, v in recommendations.items() if isinstance(v, dict) and 'estimated_importance' in v]) / 6
}}

print(json.dumps(recommendations))
'''

    try:
        response = invoke_with_retry(code)

        all_text = []
        for event in response["stream"]:
            if "result" in event:
                result_content = event["result"]
                if isinstance(result_content, dict) and "content" in result_content:
                    content_list = result_content["content"]
                    if content_list and len(content_list) > 0:
                        for content_item in content_list:
                            if "text" in content_item:
                                text_content = content_item["text"]
                                all_text.append(text_content)

        # Combine all text output
        full_output = '\n'.join(all_text)

        # Look for JSON result
        result_json = None
        lines = full_output.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('{'):
                try:
                    json.loads(stripped)
                    result_json = stripped
                    break
                except:
                    continue

        if not result_json:
            return json.dumps({"error": "No result from code execution", "debug_output": full_output[:1000]})

        return result_json

    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "type": type(e).__name__, "traceback": traceback.format_exc()})


@tool
def create_features(dataset_s3_path: str, feature_config: str) -> str:
    """
    Create features based on user-selected configuration.

    Args:
        dataset_s3_path: S3 path to dataset
        feature_config: JSON string with selected features (from recommendations)

    Returns:
        JSON with output path and features created
    """
    config = json.loads(feature_config)

    if dataset_s3_path.startswith('s3://'):
        parts = dataset_s3_path.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1]
    else:
        bucket = BUCKET
        key = dataset_s3_path

    code = f'''
import pandas as pd
import numpy as np
import json
import subprocess

# Load data using AWS CLI (Code Interpreter has AWS credentials via IAM role)
subprocess.run(['aws', 's3', 'cp', 's3://{bucket}/{key}', '/tmp/data.csv'], check=True)
df = pd.read_csv('/tmp/data.csv')

# Auto-detect columns
time_col = None
value_col = None
for col in df.columns:
    if any(kw in col.lower() for kw in ['date', 'time', 'ds', 'timestamp']):
        time_col = col
    if any(kw in col.lower() for kw in ['sales', 'value', 'y', 'price', 'demand']):
        value_col = col

if not value_col:
    value_col = df.select_dtypes(include=[np.number]).columns[0]

df[time_col] = pd.to_datetime(df[time_col])
df = df.sort_values(time_col).reset_index(drop=True)

config = {repr(config)}
features_created = []

# Create features based on config
original_cols = len(df.columns)

# Lag features
if config.get('lag_features', {{}}).get('enabled'):
    for lag in config['lag_features']['lags']:
        df[f'lag_{{lag}}'] = df[value_col].shift(lag)
        features_created.append(f'lag_{{lag}}')

# Rolling features
if config.get('rolling_features', {{}}).get('enabled'):
    for window in config['rolling_features']['windows']:
        for stat in config['rolling_features']['statistics']:
            col_name = f'rolling_{{stat}}_{{window}}'
            df[col_name] = df[value_col].rolling(window=window, min_periods=1).agg(stat)
            features_created.append(col_name)

# Differencing features
if config.get('differencing_features', {{}}).get('enabled'):
    for d in config['differencing_features']['orders']:
        df[f'diff_{{d}}'] = df[value_col].diff(d)
        features_created.append(f'diff_{{d}}')

# Calendar features
if config.get('calendar_features', {{}}).get('enabled'):
    for feat in config['calendar_features']['features']:
        if feat == 'day_of_week':
            df['day_of_week'] = df[time_col].dt.dayofweek
        elif feat == 'month':
            df['month'] = df[time_col].dt.month
        elif feat == 'quarter':
            df['quarter'] = df[time_col].dt.quarter
        elif feat == 'is_weekend':
            df['is_weekend'] = (df[time_col].dt.dayofweek >= 5).astype(int)
        elif feat == 'day_of_month':
            df['day_of_month'] = df[time_col].dt.day
        features_created.append(feat)

# Fourier features
if config.get('fourier_features', {{}}).get('enabled'):
    period = config['fourier_features']['period']
    k = config['fourier_features']['k_terms']
    t = np.arange(len(df))
    for i in range(1, k + 1):
        df[f'fourier_sin_{{i}}'] = np.sin(2 * np.pi * i * t / period)
        df[f'fourier_cos_{{i}}'] = np.cos(2 * np.pi * i * t / period)
        features_created.extend([f'fourier_sin_{{i}}', f'fourier_cos_{{i}}'])

# Expanding features
if config.get('expanding_features', {{}}).get('enabled'):
    for stat in config['expanding_features']['statistics']:
        df[f'expanding_{{stat}}'] = df[value_col].expanding(min_periods=1).agg(stat)
        features_created.append(f'expanding_{{stat}}')

# Drop NaN rows
df_cleaned = df.dropna()

# Return dataframe info instead of CSV
file_name = '{key.split("/")[-1]}'
output_key = 'cleaned/' + file_name.replace('.csv', '_features.csv')

result = {{
    'status': 'success',
    'output_key': output_key,
    'features_created': features_created,
    'feature_count': len(features_created),
    'original_rows': len(df),
    'cleaned_rows': len(df_cleaned),
    'dropped_rows': len(df) - len(df_cleaned)
}}

print(json.dumps(result))
'''

    try:
        response = invoke_with_retry(code)

        all_text = []
        for event in response["stream"]:
            if "result" in event:
                result_content = event["result"]
                if isinstance(result_content, dict) and "content" in result_content:
                    content_list = result_content["content"]
                    if content_list and len(content_list) > 0:
                        for content_item in content_list:
                            if "text" in content_item:
                                text_content = content_item["text"]
                                all_text.append(text_content)

        # Combine all text output
        full_output = '\n'.join(all_text)

        # Look for JSON result
        result_json = None
        lines = full_output.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('{'):
                try:
                    json.loads(stripped)
                    result_json = stripped
                    break
                except:
                    continue

        if not result_json:
            return json.dumps({"error": "No result from code execution", "debug_output": full_output[:1000]})

        # Parse result
        result = json.loads(result_json)

        # Recreate featured dataset and upload to S3
        if result.get('status') == 'success' and 'output_key' in result:
            import pandas as pd
            import io

            # Download original dataset
            obj = S3_CLIENT.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(io.BytesIO(obj['Body'].read()))

            # Auto-detect columns
            time_col = None
            value_col = None
            for col in df.columns:
                if any(kw in col.lower() for kw in ['date', 'time', 'ds', 'timestamp']):
                    time_col = col
                if any(kw in col.lower() for kw in ['sales', 'value', 'y', 'price', 'demand']):
                    value_col = col

            if not value_col:
                value_col = df.select_dtypes(include=['number']).columns[0]

            df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(time_col).reset_index(drop=True)

            # Recreate features
            import numpy as np

            if config.get('lag_features', {}).get('enabled'):
                for lag in config['lag_features']['lags']:
                    df[f'lag_{lag}'] = df[value_col].shift(lag)

            if config.get('rolling_features', {}).get('enabled'):
                for window in config['rolling_features']['windows']:
                    for stat in config['rolling_features']['statistics']:
                        df[f'rolling_{stat}_{window}'] = df[value_col].rolling(window=window, min_periods=1).agg(stat)

            if config.get('differencing_features', {}).get('enabled'):
                for d in config['differencing_features']['orders']:
                    df[f'diff_{d}'] = df[value_col].diff(d)

            if config.get('calendar_features', {}).get('enabled'):
                for feat in config['calendar_features']['features']:
                    if feat == 'day_of_week':
                        df['day_of_week'] = df[time_col].dt.dayofweek
                    elif feat == 'month':
                        df['month'] = df[time_col].dt.month
                    elif feat == 'quarter':
                        df['quarter'] = df[time_col].dt.quarter
                    elif feat == 'is_weekend':
                        df['is_weekend'] = (df[time_col].dt.dayofweek >= 5).astype(int)
                    elif feat == 'day_of_month':
                        df['day_of_month'] = df[time_col].dt.day

            if config.get('fourier_features', {}).get('enabled'):
                period = config['fourier_features']['period']
                k = config['fourier_features']['k_terms']
                t = np.arange(len(df))
                for i in range(1, k + 1):
                    df[f'fourier_sin_{i}'] = np.sin(2 * np.pi * i * t / period)
                    df[f'fourier_cos_{i}'] = np.cos(2 * np.pi * i * t / period)

            if config.get('expanding_features', {}).get('enabled'):
                for stat in config['expanding_features']['statistics']:
                    df[f'expanding_{stat}'] = df[value_col].expanding(min_periods=1).agg(stat)

            # Drop NaN and upload
            df_cleaned = df.dropna()
            csv_buffer = io.StringIO()
            df_cleaned.to_csv(csv_buffer, index=False)

            output_key = result.pop('output_key')
            S3_CLIENT.put_object(
                Bucket=bucket,
                Key=output_key,
                Body=csv_buffer.getvalue().encode('utf-8')
            )

            result['output_path'] = f's3://{bucket}/{output_key}'

        return json.dumps(result)

    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "type": type(e).__name__, "traceback": traceback.format_exc()})
