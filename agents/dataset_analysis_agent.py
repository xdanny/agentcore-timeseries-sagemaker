import os
#!/usr/bin/env python3
"""
Dataset Analysis Agent - Intelligent column detection and recommendations
Uses Code Interpreter to analyze dataset structure and suggest optimal configuration
"""
import json
from strands import tool
from agents.code_interpreter_utils import invoke_with_retry

BUCKET = os.environ.get('BUCKET_NAME', 'sagemaker-forecasting-{}-{}'.format(os.environ.get('AWS_REGION', 'us-east-1'), boto3.client('sts').get_caller_identity()['Account']))


@tool
def analyze_dataset_structure(s3_key: str) -> str:
    """
    Intelligently analyze dataset structure and recommend configuration.

    Uses Code Interpreter to:
    1. Detect time column
    2. Identify potential target variables
    3. Detect exogenous variables vs additional targets
    4. Recommend univariate vs multivariate approach
    5. Calculate correlations and data quality metrics

    Args:
        s3_key: S3 key to dataset (e.g., 'uploads/data.csv')

    Returns:
        JSON with intelligent recommendations
    """

    code = f'''
import pandas as pd
import numpy as np
import json
import subprocess
from scipy import stats

# Load data
subprocess.run(['aws', 's3', 'cp', 's3://{BUCKET}/{s3_key}', '/tmp/data.csv'], check=True)
df = pd.read_csv('/tmp/data.csv')

# 1. Detect time column
time_column = None
time_candidates = []
for col in df.columns:
    col_lower = col.lower()
    # Check name
    if any(kw in col_lower for kw in ['date', 'time', 'ds', 'timestamp', 'day', 'month', 'year', 'period']):
        time_candidates.append({{'name': col, 'reason': 'column name', 'confidence': 'high'}})
        if not time_column:
            time_column = col
    # Check if parseable as datetime
    elif df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].head(10))
            time_candidates.append({{'name': col, 'reason': 'parseable as datetime', 'confidence': 'medium'}})
            if not time_column:
                time_column = col
        except:
            pass

if not time_column and len(time_candidates) == 0:
    # Default to first column
    time_column = df.columns[0]
    time_candidates.append({{'name': time_column, 'reason': 'default (first column)', 'confidence': 'low'}})

# 2. Analyze numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != time_column]

column_analysis = []
for col in numeric_cols:
    series = df[col].dropna()

    analysis = {{
        'name': col,
        'dtype': str(df[col].dtype),
        'missing_pct': float((df[col].isna().sum() / len(df)) * 100),
        'mean': float(series.mean()),
        'std': float(series.std()),
        'min': float(series.min()),
        'max': float(series.max()),
        'unique_values': int(series.nunique()),
        'unique_ratio': float(series.nunique() / len(series))
    }}

    # Detect column type based on name
    col_lower = col.lower()

    # Target variable indicators
    target_keywords = ['sales', 'revenue', 'price', 'demand', 'quantity', 'volume', 'count', 'value', 'y', 'target']
    is_target_like = any(kw in col_lower for kw in target_keywords)

    # Exogenous variable indicators
    exog_keywords = ['temp', 'weather', 'holiday', 'promo', 'ad', 'marketing', 'gdp', 'index', 'rate', 'sentiment']
    is_exog_like = any(kw in col_lower for kw in exog_keywords)

    analysis['is_target_like'] = is_target_like
    analysis['is_exog_like'] = is_exog_like

    column_analysis.append(analysis)

# 3. Detect metric types and entity prefixes
# Group by entity (STOCK_A, STOCK_B) and metric type (price, volume)
entity_metrics = {{}}
for col_info in column_analysis:
    col = col_info['name']
    col_lower = col.lower()

    # Extract entity prefix (e.g., STOCK_A, STOCK_B, product_1)
    entity = None
    parts = col.split('_')
    if len(parts) >= 2:
        # Entity is usually first 2 parts (e.g., STOCK_A from STOCK_A_price)
        entity = '_'.join(parts[:-1])

    # Extract metric type
    metric = None
    for m in ['price', 'volume', 'sales', 'revenue', 'quantity', 'demand', 'count', 'cost', 'profit']:
        if m in col_lower:
            metric = m
            break

    if entity and metric:
        if entity not in entity_metrics:
            entity_metrics[entity] = {{}}
        entity_metrics[entity][metric] = col

# 4. Identify primary forecast targets (e.g., prices) vs exogenous variables (e.g., volumes)
# Primary targets: price, sales, revenue, demand
# Exogenous: volume, cost, quantity
primary_metrics = ['price', 'sales', 'revenue', 'demand', 'profit']
exogenous_metrics = ['volume', 'quantity', 'cost', 'count']

target_recommendations = []
for entity, metrics in entity_metrics.items():
    for metric, col in metrics.items():
        if metric in primary_metrics:
            # This is a forecast target
            num_entities = len([e for e in entity_metrics.keys() if metric in entity_metrics[e]])
            if num_entities > 1:
                target_recommendations.append({{
                    'column': col,
                    'reason': f'One of {{num_entities}} {{metric}} columns (entity: {{entity}}) - suggests separate forecasts',
                    'confidence': 'high',
                    'type': 'independent_target'
                }})
            else:
                target_recommendations.append({{
                    'column': col,
                    'reason': f'Primary {{metric}} column for {{entity}}',
                    'confidence': 'high',
                    'type': 'primary_target'
                }})

# If no structured entity metrics found, use simple metric grouping
if len(target_recommendations) == 0:
    metric_groups = {{}}
    for col_info in column_analysis:
        col = col_info['name']
        col_lower = col.lower()

        metric = None
        for m in ['price', 'sales', 'revenue', 'quantity', 'demand', 'count', 'volume']:
            if m in col_lower:
                metric = m
                break

        if metric and metric in primary_metrics:
            if metric not in metric_groups:
                metric_groups[metric] = []
            metric_groups[metric].append(col)

    for metric, cols in metric_groups.items():
        for col in cols:
            target_recommendations.append({{
                'column': col,
                'reason': f'{{metric.capitalize()}} column',
                'confidence': 'medium',
                'type': 'primary_target'
            }})

# Fallback: highest coefficient of variation
if len(target_recommendations) == 0:
    cv_scores = []
    for col_info in column_analysis:
        if col_info['std'] > 0:
            cv = col_info['std'] / abs(col_info['mean']) if col_info['mean'] != 0 else 0
            cv_scores.append({{'column': col_info['name'], 'cv': cv}})

    cv_scores.sort(key=lambda x: x['cv'], reverse=True)
    if cv_scores:
        target_recommendations.append({{
            'column': cv_scores[0]['column'],
            'reason': 'Highest variability (good for forecasting)',
            'confidence': 'medium',
            'type': 'primary_target'
        }})

# 5. Recommend exogenous variables for each target (entity-aware)
exog_recommendations = {{}}
for target_rec in target_recommendations:
    target = target_rec['column']
    exog_candidates = []

    # Extract target entity and metric
    target_entity = None
    target_metric = None
    target_parts = target.split('_')
    if len(target_parts) >= 2:
        target_entity = '_'.join(target_parts[:-1])
        target_metric = target_parts[-1]

    for col_info in column_analysis:
        col = col_info['name']
        if col == target:
            continue

        # Extract candidate entity and metric
        col_entity = None
        col_metric = None
        col_parts = col.split('_')
        if len(col_parts) >= 2:
            col_entity = '_'.join(col_parts[:-1])
            col_metric = col_parts[-1]

        # Calculate correlation
        try:
            correlation = float(df[[target, col]].corr().iloc[0, 1])
        except:
            correlation = 0.0

        # Recommendation logic:
        # 1. Same entity, different metric type (e.g., STOCK_A_volume for STOCK_A_price) = HIGH priority exogenous
        # 2. Different entity, exogenous metric (e.g., STOCK_B_volume for STOCK_A_price) = MEDIUM priority
        # 3. Same entity, same primary metric = SKIP (another target, not exogenous)

        if col_entity == target_entity:
            # Same entity
            col_metric_lower = col_metric.lower() if col_metric else ''
            target_metric_lower = target_metric.lower() if target_metric else ''

            # Check if col is exogenous metric type
            is_exog_metric = any(m in col_metric_lower for m in exogenous_metrics)
            is_target_metric = any(m in col_metric_lower for m in primary_metrics)

            if is_exog_metric:
                # Same entity, exogenous metric (e.g., STOCK_A_volume for STOCK_A_price)
                exog_candidates.append({{
                    'column': col,
                    'correlation': correlation,
                    'reason': f'Same entity ({{col_entity}}) - {{col_metric}} affects {{target_metric}}',
                    'confidence': 'high'
                }})
            elif is_target_metric and col_metric_lower != target_metric_lower:
                # Same entity, different primary metric (e.g., STOCK_A_revenue for STOCK_A_price)
                exog_candidates.append({{
                    'column': col,
                    'correlation': correlation,
                    'reason': f'Same entity ({{col_entity}}) - related metric',
                    'confidence': 'medium'
                }})
        else:
            # Different entity
            col_metric_lower = col_metric.lower() if col_metric else col.lower()

            # Only recommend if it's exogenous-type metric or has strong correlation
            is_exog_metric = any(m in col_metric_lower for m in exogenous_metrics)

            if is_exog_metric or abs(correlation) > 0.5:
                exog_candidates.append({{
                    'column': col,
                    'correlation': correlation,
                    'reason': f'Different entity ({{col_entity}}) - {{col_metric if col_metric else col}}' + (f' (high correlation)' if abs(correlation) > 0.5 else ''),
                    'confidence': 'high' if abs(correlation) > 0.5 else 'low'
                }})

        # Also check for general exogenous keywords
        if col_info['is_exog_like'] and col not in [c['column'] for c in exog_candidates]:
            exog_candidates.append({{
                'column': col,
                'correlation': correlation,
                'reason': 'External factor (weather, holiday, etc.)',
                'confidence': 'high'
            }})

    # Sort by confidence and correlation
    confidence_order = {{'high': 3, 'medium': 2, 'low': 1}}
    exog_candidates.sort(key=lambda x: (confidence_order.get(x['confidence'], 0), abs(x['correlation'])), reverse=True)

    exog_recommendations[target] = exog_candidates

# 6. Calculate correlations for visualization
correlation_matrix = {{}}
if len(numeric_cols) > 0:
    corr = df[numeric_cols].corr()
    correlation_matrix = corr.to_dict()

# 7. Recommend model type
model_recommendation = {{}}
if len(target_recommendations) > 1 and any(rec['type'] == 'independent_target' for rec in target_recommendations):
    model_recommendation = {{
        'type': 'multiple_univariate',
        'reason': 'Multiple similar metrics detected - recommend separate univariate forecasts for each',
        'suggested_approach': 'Run pipeline once per target variable'
    }}
elif len(target_recommendations) == 1 and len(exog_recommendations.get(target_recommendations[0]['column'], [])) > 0:
    model_recommendation = {{
        'type': 'multivariate',
        'reason': 'Single target with exogenous variables available',
        'suggested_approach': 'Use ARIMAX with selected exogenous variables'
    }}
else:
    model_recommendation = {{
        'type': 'univariate',
        'reason': 'Single target, no clear exogenous variables',
        'suggested_approach': 'Use standard ARIMA model'
    }}

# 8. Data quality assessment
quality_metrics = {{
    'total_rows': len(df),
    'date_range': {{
        'start': str(df[time_column].iloc[0]) if time_column else None,
        'end': str(df[time_column].iloc[-1]) if time_column else None
    }},
    'missing_data_pct': float((df.isna().sum().sum() / (len(df) * len(df.columns))) * 100),
    'numeric_columns': len(numeric_cols),
    'total_columns': len(df.columns)
}}

result = {{
    'time_column': {{
        'recommended': time_column,
        'candidates': time_candidates
    }},
    'target_recommendations': target_recommendations,
    'exogenous_recommendations': exog_recommendations,
    'model_recommendation': model_recommendation,
    'column_analysis': column_analysis,
    'correlation_matrix': correlation_matrix,
    'quality_metrics': quality_metrics
}}

print(json.dumps(result))
'''

    try:
        # Invoke with automatic session retry
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
                                all_text.append(content_item["text"])

        full_output = '\n'.join(all_text)

        # Parse result - look for JSON object (may span multiple lines)
        lines = full_output.split('\n')

        # Try to find complete JSON object
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('{'):
                # Try to parse from this line onwards
                json_text = '\n'.join(lines[i:])
                try:
                    result = json.loads(json_text)
                    # Verify it has required keys
                    if 'time_column' in result and 'target_recommendations' in result:
                        return json.dumps(result)
                except:
                    # Try single line
                    try:
                        result = json.loads(stripped)
                        if 'time_column' in result and 'target_recommendations' in result:
                            return json.dumps(result)
                    except:
                        continue

        return json.dumps({
            "error": "No valid JSON in output",
            "debug": full_output[:2000],
            "hint": "Check Code Interpreter logs for Python errors"
        })

    except Exception as e:
        import traceback
        return json.dumps({
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        })
