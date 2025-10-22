#!/usr/bin/env python3
"""
Comprehensive End-to-End Test for Intelligent Forecasting System
Tests the complete pipeline: EDA → Feature Engineering → Tuning → Training → Report
"""
import os
import sys
import json
import time
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.advanced_eda_agent import run_advanced_eda
from agents.intelligent_feature_engineering_agent import recommend_features, create_features
from agents.sagemaker_tuning import create_sagemaker_tuning_job, get_tuning_job_status
from agents.sagemaker_simple import create_sagemaker_training_job, get_training_job_status, deploy_sagemaker_model, invoke_sagemaker_endpoint
from agents.sagemaker_lstm import create_lstm_training_job, get_lstm_training_status
from agents.comprehensive_report_agent import generate_comprehensive_report

# Configuration - Auto-detect from environment or AWS account
REGION = os.environ.get('AWS_REGION', 'us-east-1')
ACCOUNT_ID = boto3.client('sts').get_caller_identity()['Account']
BUCKET = os.environ.get('BUCKET_NAME', f'sagemaker-forecasting-{REGION}-{ACCOUNT_ID}')
ROLE_ARN = os.environ.get('SAGEMAKER_ROLE_ARN', f'arn:aws:iam::{ACCOUNT_ID}:role/SageMaker-ForecastingPipeline-ExecutionRole')

s3 = boto3.client('s3', region_name=REGION)
sagemaker = boto3.client('sagemaker', region_name=REGION)

class TestResults:
    def __init__(self):
        self.results = []
        self.start_time = time.time()

    def add(self, step, status, details=None, error=None):
        self.results.append({
            'step': step,
            'status': status,
            'details': details,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })

        icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⏳"
        print(f"{icon} {step}: {status}")
        if details:
            print(f"   Details: {details}")
        if error:
            print(f"   Error: {error}")

    def summary(self):
        elapsed = time.time() - self.start_time
        passed = sum(1 for r in self.results if r['status'] == 'PASS')
        failed = sum(1 for r in self.results if r['status'] == 'FAIL')

        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {len(self.results)}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⏱️  Duration: {elapsed:.1f}s")
        print("="*60)

        return failed == 0

def create_test_dataset():
    """Create synthetic time series dataset for testing"""
    print("\n📁 Creating test dataset...")

    # Generate 90 days of daily data with trend + seasonality
    dates = pd.date_range(start='2024-01-01', periods=90, freq='D')

    # Trend component
    trend = np.linspace(100, 150, 90)

    # Weekly seasonality (7-day cycle)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(90) / 7)

    # Random noise
    noise = np.random.normal(0, 5, 90)

    # Combine components
    values = trend + seasonality + noise

    df = pd.DataFrame({
        'date': dates,
        'sales': values
    })

    # Upload to S3
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    key = f'test_data/synthetic_sales_{timestamp}.csv'
    csv_data = df.to_csv(index=False)
    s3.put_object(Bucket=BUCKET, Key=key, Body=csv_data.encode('utf-8'))

    s3_path = f's3://{BUCKET}/{key}'
    print(f"✅ Test dataset created: {s3_path}")
    return s3_path

def validate_eda_output(eda_result):
    """Validate EDA output has proper time series analysis"""
    eda = json.loads(eda_result)

    checks = []

    # Check stationarity analysis
    if 'stationarity' in eda:
        stat = eda['stationarity']
        checks.append(('Stationarity tests', 'tests' in stat and len(stat['tests']) > 0))
        checks.append(('Differencing recommendation', 'recommended_d' in stat))

    # Check decomposition
    if 'decomposition' in eda:
        decomp = eda['decomposition']
        checks.append(('Seasonal strength', 'seasonal_strength' in decomp))
        checks.append(('Trend strength', 'trend_strength' in decomp))

    # Check autocorrelation
    if 'autocorrelation' in eda:
        acf = eda['autocorrelation']
        checks.append(('ACF lags', 'significant_acf_lags' in acf))
        checks.append(('PACF lags', 'significant_pacf_lags' in acf))
        checks.append(('Recommended p,q', 'recommended_p' in acf and 'recommended_q' in acf))

    # Check trend analysis
    if 'trend' in eda:
        trend = eda['trend']
        checks.append(('Trend slope', 'slope' in trend))
        checks.append(('Trend significance', 'has_significant_trend' in trend))

    # Check report
    checks.append(('HTML report', 'report_url' in eda))

    passed = sum(1 for _, status in checks if status)
    total = len(checks)

    return passed == total, f"{passed}/{total} checks passed", checks

def validate_feature_recommendations(rec_result, eda_result):
    """Validate feature recommendations are intelligent and not excessive"""
    recs = json.loads(rec_result)
    eda = json.loads(eda_result)

    checks = []

    # Check summary
    if 'summary' in recs:
        feature_count = recs['summary'].get('total_recommended_features', 0)
        checks.append(('Feature count reasonable', 10 <= feature_count <= 40))
        checks.append(('Has summary', True))

    # Check lag features (should be based on ACF)
    if 'lag_features' in recs and 'autocorrelation' in eda:
        lag_recs = recs['lag_features']
        acf_lags = eda['autocorrelation'].get('significant_acf_lags', [])
        checks.append(('Lag recommendations', len(lag_recs.get('lags', [])) > 0))
        checks.append(('Has justification', 'justification' in lag_recs))
        checks.append(('Has priority', 'priority' in lag_recs))

    # Check rolling features (should be based on seasonality)
    if 'rolling_features' in recs:
        roll_recs = recs['rolling_features']
        checks.append(('Rolling windows defined', len(roll_recs.get('windows', [])) > 0))
        checks.append(('Statistics defined', len(roll_recs.get('statistics', [])) > 0))

    # Check calendar features
    checks.append(('Calendar features', 'calendar_features' in recs))

    # Check that high priority features are justified
    high_priority = [k for k, v in recs.items() if isinstance(v, dict) and v.get('priority') == 'high']
    checks.append(('High priority features exist', len(high_priority) > 0))

    passed = sum(1 for _, status in checks if status)
    total = len(checks)

    return passed == total, f"{passed}/{total} checks passed", checks

def validate_feature_creation(feature_result):
    """Validate feature creation output"""
    result = json.loads(feature_result)

    checks = []
    checks.append(('Status success', result.get('status') == 'success'))
    checks.append(('Output path exists', 'output_path' in result))
    checks.append(('Features created', result.get('feature_count', 0) > 0))
    checks.append(('Feature list', len(result.get('features_created', [])) > 0))
    checks.append(('Rows cleaned', 'cleaned_rows' in result))

    # Check feature count not excessive
    feature_count = result.get('feature_count', 0)
    checks.append(('Feature count reasonable', feature_count < 50))

    passed = sum(1 for _, status in checks if status)
    total = len(checks)

    return passed == total, f"{passed}/{total} checks passed", checks

def validate_report(report_result):
    """Validate comprehensive report has all required elements"""
    result = json.loads(report_result)

    checks = []

    checks.append(('Status success', result.get('status') == 'success'))
    checks.append(('Report URL', 'report_url' in result))

    # Check metrics
    if 'metrics' in result:
        metrics = result['metrics']
        checks.append(('MAE metric', 'mae' in metrics))
        checks.append(('RMSE metric', 'rmse' in metrics))
        checks.append(('MAPE metric', 'mape' in metrics))
        checks.append(('AIC metric', 'aic' in metrics))
        checks.append(('BIC metric', 'bic' in metrics))

    # Check forecast
    if 'forecast' in result:
        forecast = result['forecast']
        checks.append(('Forecast data', isinstance(forecast, list) and len(forecast) > 0))
        if forecast:
            checks.append(('Forecast with CI', 'lower_ci' in forecast[0] and 'upper_ci' in forecast[0]))

    # Check model info
    checks.append(('Model order', 'model_order' in result))
    checks.append(('Seasonal order', 'seasonal_order' in result))

    passed = sum(1 for _, status in checks if status)
    total = len(checks)

    return passed == total, f"{passed}/{total} checks passed", checks

def run_end_to_end_test():
    """Run complete end-to-end test"""
    results = TestResults()

    print("\n" + "="*60)
    print("🧪 INTELLIGENT FORECASTING SYSTEM - END-TO-END TEST")
    print("="*60)

    # Step 1: Create test dataset
    try:
        dataset_path = create_test_dataset()
        results.add("Test Dataset Creation", "PASS", dataset_path)
    except Exception as e:
        results.add("Test Dataset Creation", "FAIL", error=str(e))
        return results.summary()

    # Step 2: Advanced EDA
    print("\n📊 Step 1/7: Running Advanced EDA...")
    try:
        eda_result = run_advanced_eda(dataset_path)
        eda_data = json.loads(eda_result)

        if 'error' in eda_data:
            results.add("Advanced EDA", "FAIL", error=eda_data['error'])
            return results.summary()

        passed, summary, checks = validate_eda_output(eda_result)
        status = "PASS" if passed else "FAIL"
        results.add("Advanced EDA", status, summary)

        for check_name, check_passed in checks:
            if not check_passed:
                print(f"   ⚠️  {check_name}: FAILED")

    except Exception as e:
        results.add("Advanced EDA", "FAIL", error=str(e))
        return results.summary()

    # Step 3: Feature Recommendations
    print("\n🧠 Step 2/7: Getting Feature Recommendations...")
    try:
        rec_result = recommend_features(dataset_path, eda_result)
        rec_data = json.loads(rec_result)

        if 'error' in rec_data:
            results.add("Feature Recommendations", "FAIL", error=rec_data['error'])
            return results.summary()

        passed, summary, checks = validate_feature_recommendations(rec_result, eda_result)
        status = "PASS" if passed else "FAIL"
        results.add("Feature Recommendations", status, summary)

        # Create simplified config (not too many features)
        feature_config = {
            'lag_features': {
                'enabled': True,
                'lags': rec_data.get('lag_features', {}).get('lags', [1, 7])[:3]  # Max 3 lags
            },
            'rolling_features': {
                'enabled': True,
                'windows': rec_data.get('rolling_features', {}).get('windows', [7])[:2],  # Max 2 windows
                'statistics': ['mean', 'std']  # Only 2 stats
            },
            'calendar_features': {
                'enabled': True,
                'features': ['day_of_week', 'is_weekend']  # Only 2 calendar features
            }
        }

    except Exception as e:
        results.add("Feature Recommendations", "FAIL", error=str(e))
        return results.summary()

    # Step 4: Feature Creation
    print("\n⚙️  Step 3/7: Creating Features...")
    try:
        feature_result = create_features(dataset_path, json.dumps(feature_config))
        feature_data = json.loads(feature_result)

        if 'error' in feature_data or feature_data.get('status') != 'success':
            results.add("Feature Creation", "FAIL", error=feature_data.get('error', 'Unknown error'))
            return results.summary()

        passed, summary, checks = validate_feature_creation(feature_result)
        status = "PASS" if passed else "FAIL"
        results.add("Feature Creation", status, f"{summary}, {feature_data.get('feature_count', 0)} features")

        feature_dataset_path = feature_data['output_path']

    except Exception as e:
        results.add("Feature Creation", "FAIL", error=str(e))
        return results.summary()

    # Step 5: Hyperparameter Tuning
    print("\n🎯 Step 4/7: Starting Hyperparameter Tuning...")
    try:
        tuning_job_name = f"test-tuning-{int(time.time())}"
        tuning_result = create_sagemaker_tuning_job(
            job_name=tuning_job_name,
            dataset_s3_path=feature_dataset_path,
            role_arn=ROLE_ARN,
            max_jobs=3,  # Minimal for testing
            max_parallel_jobs=1
        )
        tuning_data = json.loads(tuning_result)

        if not tuning_data.get('success'):
            results.add("Hyperparameter Tuning", "FAIL", error=tuning_data.get('error'))
            return results.summary()

        results.add("Hyperparameter Tuning Job Created", "PASS", tuning_job_name)

        # Monitor tuning (with timeout)
        print("   Monitoring tuning progress (max 10 minutes)...")
        max_wait = 600  # 10 minutes
        start_wait = time.time()

        while time.time() - start_wait < max_wait:
            status_result = get_tuning_job_status(tuning_job_name)
            status_data = json.loads(status_result)

            if not status_data.get('success'):
                results.add("Tuning Job Monitoring", "FAIL", error=status_data.get('error'))
                return results.summary()

            status = status_data.get('status')
            completed = status_data.get('training_job_count', 0)
            print(f"   Status: {status}, Completed Jobs: {completed}/3")

            if status == 'Completed':
                best_params = status_data.get('best_hyperparameters', {})
                results.add("Hyperparameter Tuning", "PASS", f"Best params: p={best_params.get('p')}, d={best_params.get('d')}, q={best_params.get('q')}")
                break
            elif status in ['Failed', 'Stopped']:
                results.add("Hyperparameter Tuning", "FAIL", error=f"Job {status}")
                return results.summary()

            time.sleep(30)  # Check every 30 seconds
        else:
            results.add("Hyperparameter Tuning", "FAIL", error="Timeout after 10 minutes")
            return results.summary()

    except Exception as e:
        results.add("Hyperparameter Tuning", "FAIL", error=str(e))
        return results.summary()

    # Step 6: Model Training
    print("\n🚀 Step 5/7: Training ARIMA Model...")
    try:
        training_job_name = f"test-training-{int(time.time())}"
        training_result = create_sagemaker_training_job(
            job_name=training_job_name,
            dataset_s3_path=feature_dataset_path,
            role_arn=ROLE_ARN
        )
        training_data = json.loads(training_result)

        if not training_data.get('success'):
            results.add("ARIMA Training", "FAIL", error=training_data.get('error'))
            return results.summary()

        results.add("ARIMA Training Job Created", "PASS", training_job_name)

        # Monitor training (with timeout)
        print("   Monitoring training progress (max 10 minutes)...")
        max_wait = 600
        start_wait = time.time()

        while time.time() - start_wait < max_wait:
            status_result = get_training_job_status(training_job_name)
            status_data = json.loads(status_result)

            if not status_data.get('success'):
                results.add("Training Job Monitoring", "FAIL", error=status_data.get('error'))
                return results.summary()

            status = status_data.get('status')
            print(f"   Status: {status}")

            if status == 'Completed':
                model_artifacts = status_data.get('model_artifacts')
                results.add("ARIMA Training", "PASS", f"Model: {model_artifacts}")
                break
            elif status in ['Failed', 'Stopped']:
                results.add("ARIMA Training", "FAIL", error=f"Job {status}")
                return results.summary()

            time.sleep(20)
        else:
            results.add("ARIMA Training", "FAIL", error="Timeout after 10 minutes")
            return results.summary()

    except Exception as e:
        results.add("ARIMA Training", "FAIL", error=str(e))
        return results.summary()

    # Step 7: Comprehensive Report
    print("\n📋 Step 6/7: Generating Comprehensive Report...")
    try:
        report_result = generate_comprehensive_report(
            dataset_s3_path=feature_dataset_path,
            eda_results=eda_result,
            feature_config=json.dumps(feature_config),
            best_hyperparameters=json.dumps(best_params),
            forecast_horizon=7
        )

        report_data = json.loads(report_result)

        if 'error' in report_data or report_data.get('status') != 'success':
            results.add("Comprehensive Report", "FAIL", error=report_data.get('error', 'Unknown error'))
            return results.summary()

        passed, summary, checks = validate_report(report_result)
        status = "PASS" if passed else "FAIL"

        metrics = report_data.get('metrics', {})
        details = f"{summary}, RMSE={metrics.get('rmse', 0):.2f}, MAPE={metrics.get('mape', 0):.1f}%"
        results.add("Comprehensive Report", status, details)

        print(f"   📄 Report: {report_data.get('report_url')}")

    except Exception as e:
        results.add("Comprehensive Report", "FAIL", error=str(e))
        return results.summary()

    # Final Summary
    return results.summary()

if __name__ == "__main__":
    success = run_end_to_end_test()
    sys.exit(0 if success else 1)
