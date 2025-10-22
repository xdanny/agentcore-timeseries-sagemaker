import os
#!/usr/bin/env python3
"""
Comprehensive Forecast Report Generator
Uses SageMaker endpoint for predictions and Code Interpreter for visualizations
"""
import json
import boto3
from strands import tool
from agents.code_interpreter_utils import invoke_with_retry

BUCKET = os.environ.get('BUCKET_NAME', 'sagemaker-forecasting-{}-{}'.format(os.environ.get('AWS_REGION', 'us-east-1'), boto3.client('sts').get_caller_identity()['Account']))
S3_CLIENT = boto3.client('s3', region_name='us-east-1')
SAGEMAKER_RUNTIME = boto3.client('sagemaker-runtime', region_name='us-east-1')


@tool
def generate_comprehensive_report(
    dataset_s3_path: str,
    endpoint_name: str,
    eda_results: str,
    forecast_horizon: int = 7
) -> str:
    """
    Generate comprehensive forecast report using SageMaker endpoint for predictions.

    Workflow:
    1. Load test data and prepare for inference
    2. Call SageMaker endpoint for predictions
    3. Generate visualizations with Code Interpreter
    4. Upload HTML report to S3

    Args:
        dataset_s3_path: S3 path to featured dataset
        endpoint_name: SageMaker endpoint name for inference
        eda_results: JSON string with EDA insights
        forecast_horizon: Number of days to forecast

    Returns:
        JSON with report S3 path and forecast metrics
    """
    # Parse inputs
    eda = json.loads(eda_results)

    # Get dataset path
    if dataset_s3_path.startswith('s3://'):
        parts = dataset_s3_path.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1]
    else:
        bucket = BUCKET
        key = dataset_s3_path

    # Step 1: Prepare test data using Code Interpreter
    prep_code = f'''
import pandas as pd
import numpy as np
import json
import subprocess

# Load data from S3
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
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    value_col = [c for c in numeric_cols if not any(kw in c for kw in ['lag_', 'rolling_', 'diff_', 'day_', 'month', 'quarter', 'weekend', 'fourier', 'expanding'])][0]

df[time_col] = pd.to_datetime(df[time_col])
df = df.sort_values(time_col).reset_index(drop=True)

# Split data (80/20) for backtesting
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

# Prepare test data
test_actual = test_df[value_col].values.tolist()
test_dates = test_df[time_col].dt.strftime('%Y-%m-%d').tolist()

# Prepare data for SageMaker endpoint (CSV format)
# SageMaker sklearn container expects CSV input
test_features = test_df.drop(columns=[time_col, value_col])
test_csv = test_features.to_csv(index=False, header=False)

result = {{
    'test_actual': test_actual,
    'test_dates': test_dates,
    'test_csv': test_csv,
    'n_samples': len(test_df)
}}

print(json.dumps(result))
'''

    try:
        # Prepare test data
        response = invoke_with_retry(prep_code)

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

        # Parse preparation result
        prep_result = None
        for line in full_output.split('\n'):
            stripped = line.strip()
            if stripped.startswith('{'):
                try:
                    prep_result = json.loads(stripped)
                    break
                except:
                    continue

        if not prep_result:
            return json.dumps({
                "error": "Failed to prepare test data",
                "debug_output": full_output[:2000]
            })

        # Step 2: Call SageMaker endpoint for predictions
        try:
            # ARIMA endpoint expects number of steps to forecast
            forecast_request = {
                'steps': prep_result['n_samples']
            }

            response = SAGEMAKER_RUNTIME.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(forecast_request)
            )

            predictions_raw = response['Body'].read().decode('utf-8')
            # Parse predictions from JSON response
            predictions_data = json.loads(predictions_raw)

            # Handle different response formats
            if 'forecast' in predictions_data:
                predictions = predictions_data['forecast']
            elif 'predictions' in predictions_data:
                predictions = predictions_data['predictions']
            elif isinstance(predictions_data, list):
                predictions = predictions_data
            else:
                predictions = []

            # Verify predictions were parsed
            if not predictions:
                return json.dumps({
                    "error": "Endpoint returned empty predictions",
                    "raw_response": predictions_raw[:500],
                    "hint": "Check inference script output format"
                })

        except Exception as e:
            return json.dumps({
                "error": f"SageMaker endpoint error: {str(e)}",
                "type": type(e).__name__,
                "hint": "Ensure the endpoint is deployed and InService"
            })

        # Step 3: Generate visualizations with Code Interpreter
        viz_code = f'''
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load test data and predictions
test_dates = pd.to_datetime({repr(prep_result['test_dates'])})
test_actual = np.array({repr(prep_result['test_actual'])})
test_pred = np.array({repr(predictions)})

# Calculate metrics
mae = float(np.mean(np.abs(test_actual - test_pred)))
rmse = float(np.sqrt(np.mean((test_actual - test_pred)**2)))
mape = float(np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100)

# Create visualizations
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Backtest: Actual vs Predicted',
        'Prediction Errors',
        'Error Distribution',
        'Forecast Metrics'
    ),
    specs=[
        [{{"type": "scatter"}}, {{"type": "scatter"}}],
        [{{"type": "histogram"}}, {{"type": "table"}}]
    ]
)

# Row 1, Col 1: Actual vs Predicted
fig.add_trace(go.Scatter(
    x=test_dates, y=test_actual,
    mode='lines+markers', name='Actual',
    line=dict(color='blue', width=2)
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=test_dates, y=test_pred,
    mode='lines+markers', name='Predicted',
    line=dict(color='red', width=2, dash='dot')
), row=1, col=1)

# Row 1, Col 2: Prediction Errors
errors = test_actual - test_pred
fig.add_trace(go.Scatter(
    x=test_dates, y=errors,
    mode='markers', name='Errors',
    marker=dict(color='orange', size=8)
), row=1, col=2)
fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)

# Row 2, Col 1: Error Distribution
fig.add_trace(go.Histogram(
    x=errors, name='Error Distribution',
    marker=dict(color='purple'), nbinsx=20
), row=2, col=1)

# Row 2, Col 2: Metrics Table
fig.add_trace(go.Table(
    header=dict(values=['<b>Metric</b>', '<b>Value</b>'],
               fill_color='paleturquoise', align='left'),
    cells=dict(values=[
        ['MAE', 'RMSE', 'MAPE', 'Test Samples'],
        [f'{{mae:.2f}}', f'{{rmse:.2f}}', f'{{mape:.2f}}%', str(len(test_actual))]
    ],
    fill_color='lavender', align='left')
), row=2, col=2)

fig.update_layout(
    height=800,
    showlegend=True,
    title_text="<b>Forecasting Report - SageMaker Endpoint Results</b>",
    title_font_size=18
)

html_str = fig.to_html(full_html=True, include_plotlyjs='cdn')

# Add executive summary
eda_summary = {repr(eda)}
executive_summary = f"""
<div style="padding: 20px; background-color: #f0f8ff; margin: 20px; border-radius: 10px;">
    <h2>📊 Executive Summary</h2>
    <ul>
        <li><b>Model Performance</b>: MAE={{mae:.2f}}, RMSE={{rmse:.2f}}, MAPE={{mape:.2f}}%</li>
        <li><b>Test Samples</b>: {{len(test_actual)}}</li>
        <li><b>Endpoint</b>: {endpoint_name}</li>
        <li><b>Data Characteristics</b>:
            <ul>
                <li>Seasonal Strength: {{eda_summary.get('decomposition', {{}}).get('seasonal_strength', 0):.2f}}</li>
                <li>Trend: {{eda_summary.get('trend', {{}}).get('trend_direction', 'unknown')}}</li>
                <li>Stationarity: Required {{eda_summary.get('stationarity', {{}}).get('recommended_d', 0)}} differencing</li>
            </ul>
        </li>
    </ul>
    <h3>🔍 Key Insights</h3>
    <ul>
        <li>Model predictions show {{f"{{mape:.1f}}%"}} average error</li>
        <li>Error distribution appears {{{{('normal' if abs(np.mean(errors)) < np.std(errors) * 0.1 else 'skewed')}}}}</li>
        <li>RMSE indicates typical prediction error of {{{{rmse:.2f}}}} units</li>
    </ul>
</div>
"""

html_with_summary = html_str.replace('<body>', '<body>' + executive_summary)

result = {{
    'status': 'success',
    'metrics': {{
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }},
    'test_samples': int(len(test_actual))
}}

print(json.dumps(result))
print("===HTML_REPORT_START===")
print(html_with_summary)
print("===HTML_REPORT_END===")
'''

        response = invoke_with_retry(viz_code)

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

        # Extract JSON result
        result_json = None
        for line in full_output.split('\n'):
            stripped = line.strip()
            if stripped.startswith('{'):
                try:
                    result_json = json.loads(stripped)
                    break
                except:
                    continue

        if not result_json:
            return json.dumps({
                "error": "No result from visualization",
                "debug_output": full_output[:2000]
            })

        result = result_json

        # Extract and upload HTML
        if "===HTML_REPORT_START===" in full_output:
            html_start = full_output.find("===HTML_REPORT_START===") + len("===HTML_REPORT_START===")
            html_end = full_output.find("===HTML_REPORT_END===")
            if html_end > html_start:
                html_content = full_output[html_start:html_end].strip()

                import datetime
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                report_key = f'reports/comprehensive_forecast_report_{timestamp}.html'

                S3_CLIENT.put_object(
                    Bucket=bucket,
                    Key=report_key,
                    Body=html_content.encode('utf-8'),
                    ContentType='text/html'
                )

                # Generate presigned URL
                presigned_url = S3_CLIENT.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket, 'Key': report_key},
                    ExpiresIn=86400  # 24 hours
                )

                result['report_url'] = presigned_url
                result['report_s3_path'] = f's3://{bucket}/{report_key}'

        return json.dumps(result)

    except Exception as e:
        import traceback
        return json.dumps({
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        })
