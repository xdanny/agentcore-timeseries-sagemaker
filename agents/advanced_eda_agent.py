import os
#!/usr/bin/env python3
"""
Advanced EDA Agent with Stationarity Tests, Differencing Analysis, and Plotly Visualizations
"""
import json
import boto3
from strands import tool
from agents.code_interpreter_utils import invoke_with_retry

BUCKET = os.environ.get('BUCKET_NAME', 'sagemaker-forecasting-{}-{}'.format(os.environ.get('AWS_REGION', 'us-east-1'), boto3.client('sts').get_caller_identity()['Account']))
S3_CLIENT = boto3.client('s3', region_name='us-east-1')


@tool
def run_advanced_eda(dataset_s3_path: str, time_column: str = None, value_column: str = None) -> str:
    """
    Advanced EDA with stationarity tests, differencing recommendations, and interactive visualizations.

    Analysis includes:
    - Stationarity tests (ADF, KPSS) at multiple differencing levels
    - Seasonal decomposition with strength metrics
    - ACF/PACF analysis for lag recommendations
    - Trend detection and removal suggestions
    - Interactive Plotly visualizations
    - Optimal differencing order recommendation

    Returns JSON with complete analysis + generates HTML report with graphs
    """
    # Parse S3 path
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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import acf, pacf
import subprocess
import warnings
warnings.filterwarnings('ignore')

# Load data from S3 using AWS CLI (Code Interpreter has AWS credentials via agent execution role)
subprocess.run(['aws', 's3', 'cp', 's3://{bucket}/{key}', '/tmp/data.csv'], check=True)
df = pd.read_csv('/tmp/data.csv')

# Auto-detect columns
time_keywords = ['date', 'time', 'ds', 'timestamp', 'datetime', 'period']
value_keywords = ['sales', 'value', 'y', 'price', 'demand', 'revenue', 'quantity']

time_col = {repr(time_column)}
value_col = {repr(value_column)}

if not time_col:
    for col in df.columns:
        if any(kw in col.lower() for kw in time_keywords):
            time_col = col
            break

if not value_col:
    for col in df.columns:
        if any(kw in col.lower() for kw in value_keywords):
            value_col = col
            break
    if not value_col:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        value_col = numeric_cols[0] if len(numeric_cols) > 0 else None

# Prepare time series
df[time_col] = pd.to_datetime(df[time_col])
df = df.sort_values(time_col).reset_index(drop=True)
series = df[value_col].dropna()

results = {{}}

# ===== 1. STATIONARITY ANALYSIS AT MULTIPLE LEVELS =====
stationarity_tests = []

for d in range(0, 3):  # Test original, 1st diff, 2nd diff
    if d == 0:
        test_series = series
        label = "Original"
    elif d == 1:
        test_series = series.diff().dropna()
        label = "1st Difference"
    else:
        test_series = series.diff().diff().dropna()
        label = "2nd Difference"

    # ADF Test (H0: non-stationary)
    adf_result = adfuller(test_series, autolag='AIC')
    adf_stationary = adf_result[1] < 0.05

    # KPSS Test (H0: stationary)
    try:
        kpss_result = kpss(test_series, regression='ct')
        kpss_stationary = kpss_result[1] > 0.05
    except:
        kpss_stationary = None

    stationarity_tests.append({{
        'level': label,
        'd': d,
        'adf_statistic': float(adf_result[0]),
        'adf_pvalue': float(adf_result[1]),
        'adf_stationary': bool(adf_stationary),
        'kpss_statistic': float(kpss_result[0]) if kpss_stationary is not None else None,
        'kpss_pvalue': float(kpss_result[1]) if kpss_stationary is not None else None,
        'kpss_stationary': bool(kpss_stationary) if kpss_stationary is not None else None,
        'both_agree': bool(adf_stationary and kpss_stationary) if kpss_stationary is not None else bool(adf_stationary)
    }})

# Recommend optimal differencing
optimal_d = next((t['d'] for t in stationarity_tests if t['both_agree']), 1)

results['stationarity'] = {{
    'tests': stationarity_tests,
    'recommended_d': int(optimal_d),
    'interpretation': f"Series becomes stationary after {{optimal_d}} differencing" if optimal_d > 0 else "Series is already stationary"
}}

# ===== 2. SEASONAL DECOMPOSITION WITH STRENGTH =====
if len(series) >= 14:
    try:
        ts_series = df.set_index(time_col)[value_col].dropna()
        decomp = seasonal_decompose(ts_series, model='additive', period=7, extrapolate_trend='freq')

        # Calculate component strengths
        seasonal_strength = 1 - (np.var(decomp.resid.dropna()) / np.var((decomp.seasonal + decomp.resid).dropna()))
        trend_strength = 1 - (np.var(decomp.resid.dropna()) / np.var((decomp.trend.dropna() + decomp.resid.dropna())))

        results['decomposition'] = {{
            'period': 7,
            'seasonal_strength': float(seasonal_strength),
            'trend_strength': float(trend_strength),
            'has_strong_seasonality': bool(seasonal_strength > 0.6),
            'has_strong_trend': bool(trend_strength > 0.6),
            'residual_variance': float(np.var(decomp.resid.dropna()))
        }}
    except Exception as e:
        results['decomposition'] = {{'error': str(e)}}

# ===== 3. ACF/PACF ANALYSIS FOR LAG SELECTION =====
max_lags = min(40, len(series) // 2)
acf_vals = acf(series, nlags=max_lags)
pacf_vals = pacf(series, nlags=max_lags)

# Confidence interval
conf_int = 1.96 / np.sqrt(len(series))
sig_acf = [i for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > conf_int]
sig_pacf = [i for i in range(1, len(pacf_vals)) if abs(pacf_vals[i]) > conf_int]

# Recommend p, q based on cutoffs
p_recommend = min(sig_pacf[:3]) if sig_pacf else 1
q_recommend = min(sig_acf[:3]) if sig_acf else 1

results['autocorrelation'] = {{
    'significant_acf_lags': list(sig_acf[:10]),
    'significant_pacf_lags': list(sig_pacf[:10]),
    'recommended_p': int(p_recommend),
    'recommended_q': int(q_recommend),
    'recommended_lags_for_features': list(sig_acf[:5])
}}

# ===== 4. TREND ANALYSIS =====
from scipy.stats import linregress
x = np.arange(len(series))
slope, intercept, r_value, p_value, std_err = linregress(x, series)

results['trend'] = {{
    'slope': float(slope),
    'r_squared': float(r_value**2),
    'p_value': float(p_value),
    'has_significant_trend': bool(p_value < 0.05),
    'trend_direction': 'increasing' if slope > 0 else 'decreasing',
    'detrend_recommended': bool(abs(slope) > series.std() * 0.01)
}}

# ===== 5. GENERATE INTERACTIVE VISUALIZATIONS =====
fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=('Time Series', 'Distribution',
                    'ACF', 'PACF',
                    'Seasonal Decomposition: Trend', 'Seasonal Decomposition: Seasonal',
                    'Stationarity Tests', 'Differenced Series (d=1)'),
    vertical_spacing=0.08,
    horizontal_spacing=0.1
)

# Row 1: Time series and distribution
fig.add_trace(go.Scatter(x=df[time_col], y=df[value_col], mode='lines', name='Original'), row=1, col=1)
fig.add_trace(go.Histogram(x=series, name='Distribution', nbinsx=50), row=1, col=2)

# Row 2: ACF and PACF
fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name='ACF'), row=2, col=1)
fig.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name='PACF'), row=2, col=2)
fig.add_hline(y=conf_int, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=-conf_int, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=conf_int, line_dash="dash", line_color="red", row=2, col=2)
fig.add_hline(y=-conf_int, line_dash="dash", line_color="red", row=2, col=2)

# Row 3: Decomposition
if 'decomposition' in results and 'error' not in results['decomposition']:
    fig.add_trace(go.Scatter(y=decomp.trend, mode='lines', name='Trend'), row=3, col=1)
    fig.add_trace(go.Scatter(y=decomp.seasonal, mode='lines', name='Seasonal'), row=3, col=2)

# Row 4: Stationarity comparison
adf_pvals = [t['adf_pvalue'] for t in stationarity_tests]
fig.add_trace(go.Bar(x=['Original', '1st Diff', '2nd Diff'], y=adf_pvals, name='ADF p-value'), row=4, col=1)
fig.add_hline(y=0.05, line_dash="dash", line_color="green", row=4, col=1, annotation_text="Stationary threshold")

# Differenced series
diff_series = series.diff().dropna()
fig.add_trace(go.Scatter(y=diff_series, mode='lines', name='1st Difference'), row=4, col=2)

fig.update_layout(height=1600, showlegend=False, title_text="Advanced Time Series EDA")

# Generate HTML
html_str = fig.to_html()

# Print results first
print(json.dumps(results))

# Print HTML marker and content separately
print("===HTML_REPORT_START===")
print(html_str)
print("===HTML_REPORT_END===")
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

        # Extract HTML if present in output
        if "===HTML_REPORT_START===" in full_output:
            html_start = full_output.find("===HTML_REPORT_START===") + len("===HTML_REPORT_START===")
            html_end = full_output.find("===HTML_REPORT_END===")
            if html_end > html_start:
                html_content = full_output[html_start:html_end].strip()

                # Upload to S3 with public-read ACL
                import datetime
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                report_key = f'reports/eda_advanced_report_{timestamp}.html'
                S3_CLIENT.put_object(
                    Bucket=bucket,
                    Key=report_key,
                    Body=html_content.encode('utf-8'),
                    ContentType='text/html'
                )

                # Generate presigned URL for viewing
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
        return json.dumps({"error": str(e), "type": type(e).__name__, "traceback": traceback.format_exc()})
