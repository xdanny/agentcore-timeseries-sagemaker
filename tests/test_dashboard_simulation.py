#!/usr/bin/env python3
"""
Dashboard Simulation Test
Simulates the Streamlit dashboard workflow by invoking AgentCore directly
Tests the 7-step user-driven workflow
"""
import os
import sys
import json
import time
import boto3
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration - Auto-detect from environment or AWS account
REGION = os.environ.get('AWS_REGION', 'us-east-1')
ACCOUNT_ID = boto3.client('sts').get_caller_identity()['Account']
BUCKET = os.environ.get('BUCKET_NAME', f'sagemaker-forecasting-{REGION}-{ACCOUNT_ID}')
AGENTCORE_ARN = os.environ.get(
    'AGENTCORE_ENDPOINT_ARN',
    f'arn:aws:bedrock-agentcore:{REGION}:{ACCOUNT_ID}:runtime/agent-PLACEHOLDER'
)

s3 = boto3.client('s3', region_name=REGION)
agentcore = boto3.client('bedrock-agentcore', region_name=REGION)
sagemaker = boto3.client('sagemaker', region_name=REGION)

class DashboardSimulation:
    def __init__(self):
        self.session_id = f"dashboard-test-{int(time.time())}"
        self.results = {}
        self.errors = []

    def invoke_agent(self, prompt: str):
        """Invoke AgentCore with prompt"""
        print(f"\n💬 Prompt: {prompt}")

        payload = {
            "prompt": prompt,
            "sessionId": self.session_id
        }

        try:
            response = agentcore.invoke_agent_runtime(
                agentRuntimeArn=AGENTCORE_ARN,
                contentType='application/json',
                accept='application/json',
                payload=json.dumps(payload).encode('utf-8')
            )

            # Parse streaming response
            result_text = ""
            if 'body' in response:
                for event in response['body']:
                    if 'chunk' in event and 'bytes' in event['chunk']:
                        result_text += event['chunk']['bytes'].decode('utf-8')

            print(f"📤 Response: {result_text[:200]}...")
            return result_text

        except Exception as e:
            error_msg = f"AgentCore invocation failed: {str(e)}"
            print(f"❌ {error_msg}")
            self.errors.append(error_msg)
            return None

    def create_test_data(self):
        """Create synthetic time series data"""
        print("\n" + "="*60)
        print("📁 STEP 1: Upload Data")
        print("="*60)

        # Generate 90 days of sales data with trend and weekly seasonality
        dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
        trend = np.linspace(100, 150, 90)
        seasonality = 10 * np.sin(2 * np.pi * np.arange(90) / 7)
        noise = np.random.normal(0, 5, 90)
        values = trend + seasonality + noise

        df = pd.DataFrame({
            'date': dates,
            'sales': values
        })

        # Upload to S3
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        key = f'test_data/dashboard_test_{timestamp}.csv'
        csv_data = df.to_csv(index=False)
        s3.put_object(Bucket=BUCKET, Key=key, Body=csv_data.encode('utf-8'))

        self.dataset_path = f's3://{BUCKET}/{key}'
        print(f"✅ Uploaded test data: {self.dataset_path}")
        return self.dataset_path

    def run_advanced_eda(self):
        """Step 2: Run Advanced EDA"""
        print("\n" + "="*60)
        print("📊 STEP 2: Run Advanced EDA")
        print("="*60)

        prompt = f"run advanced eda on {self.dataset_path}"
        result = self.invoke_agent(prompt)

        if not result:
            return False

        try:
            eda_data = json.loads(result)
            self.results['eda'] = eda_data

            # Validate EDA results
            checks = []
            checks.append(('Stationarity analysis', 'stationarity' in eda_data))
            checks.append(('Decomposition', 'decomposition' in eda_data))
            checks.append(('Autocorrelation', 'autocorrelation' in eda_data))
            checks.append(('Trend analysis', 'trend' in eda_data))
            checks.append(('HTML report', 'report_url' in eda_data))

            print("\n📋 EDA Validation:")
            for check_name, passed in checks:
                icon = "✅" if passed else "❌"
                print(f"   {icon} {check_name}")

            if 'stationarity' in eda_data:
                print(f"\n   Recommended differencing: {eda_data['stationarity'].get('recommended_d', 'N/A')}")

            if 'decomposition' in eda_data:
                seasonal_strength = eda_data['decomposition'].get('seasonal_strength', 0)
                print(f"   Seasonal strength: {seasonal_strength:.2f}")

            return all(passed for _, passed in checks)

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse EDA result: {e}")
            return False

    def get_feature_recommendations(self):
        """Step 3: Get Feature Recommendations"""
        print("\n" + "="*60)
        print("🧠 STEP 3: Get Feature Recommendations")
        print("="*60)

        if 'eda' not in self.results:
            print("❌ EDA results not available")
            return False

        prompt = f"recommend features for {self.dataset_path} with eda results {json.dumps(self.results['eda'])}"
        result = self.invoke_agent(prompt)

        if not result:
            return False

        try:
            rec_data = json.loads(result)
            self.results['recommendations'] = rec_data

            # Validate recommendations
            print("\n💡 Feature Recommendations:")

            if 'summary' in rec_data:
                total_features = rec_data['summary'].get('total_recommended_features', 0)
                print(f"   Total recommended features: {total_features}")

            # Show recommendations by priority
            for feat_type, config in rec_data.items():
                if isinstance(config, dict) and 'priority' in config:
                    priority = config.get('priority', '').upper()
                    justification = config.get('justification', 'N/A')
                    print(f"\n   {feat_type} [{priority}]:")
                    print(f"      {justification}")

            # Check feature count is reasonable
            feature_count = rec_data.get('summary', {}).get('total_recommended_features', 0)
            if feature_count > 50:
                print(f"\n   ⚠️  WARNING: {feature_count} features recommended (may overwhelm SageMaker)")
                return False

            return True

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse recommendations: {e}")
            return False

    def create_selected_features(self):
        """Step 4: Create Features (User Selection Simulation)"""
        print("\n" + "="*60)
        print("⚙️  STEP 4: Create Selected Features")
        print("="*60)

        if 'recommendations' not in self.results:
            print("❌ Recommendations not available")
            return False

        recs = self.results['recommendations']

        # Simulate user selection (limit features to avoid SageMaker overload)
        selected_config = {}

        # Lag features (max 3)
        if 'lag_features' in recs and recs['lag_features'].get('enabled'):
            lags = recs['lag_features']['lags'][:3]
            selected_config['lag_features'] = {'enabled': True, 'lags': lags}
            print(f"   ✅ Selected lag features: {lags}")

        # Rolling features (max 2 windows, 2 stats)
        if 'rolling_features' in recs and recs['rolling_features'].get('enabled'):
            windows = recs['rolling_features']['windows'][:2]
            stats = recs['rolling_features']['statistics'][:2]
            selected_config['rolling_features'] = {'enabled': True, 'windows': windows, 'statistics': stats}
            print(f"   ✅ Selected rolling features: windows={windows}, stats={stats}")

        # Calendar features (selective)
        if 'calendar_features' in recs:
            cal_features = ['day_of_week', 'is_weekend']
            selected_config['calendar_features'] = {'enabled': True, 'features': cal_features}
            print(f"   ✅ Selected calendar features: {cal_features}")

        # Skip Fourier and expanding features for simplicity
        print("\n   Creating features...")

        prompt = f"create features for {self.dataset_path} with config {json.dumps(selected_config)}"
        result = self.invoke_agent(prompt)

        if not result:
            return False

        try:
            feature_data = json.loads(result)
            self.results['features'] = feature_data

            if feature_data.get('status') == 'success':
                feature_count = feature_data.get('feature_count', 0)
                output_path = feature_data.get('output_path')
                self.feature_dataset_path = output_path

                print(f"\n   ✅ Created {feature_count} features")
                print(f"   📁 Output: {output_path}")

                # Validate feature count
                if feature_count > 30:
                    print(f"   ⚠️  WARNING: {feature_count} features may slow SageMaker")

                return True
            else:
                print(f"   ❌ Feature creation failed: {feature_data.get('error', 'Unknown error')}")
                return False

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse feature result: {e}")
            return False

    def start_tuning(self):
        """Step 5: Start Hyperparameter Tuning"""
        print("\n" + "="*60)
        print("🎯 STEP 5: Start Hyperparameter Tuning")
        print("="*60)

        if not hasattr(self, 'feature_dataset_path'):
            print("❌ Feature dataset not available")
            return False

        max_jobs = 5
        parallel_jobs = 2

        prompt = f"create sagemaker tuning job for {self.feature_dataset_path} with max_jobs {max_jobs}"
        result = self.invoke_agent(prompt)

        if not result:
            return False

        try:
            tuning_data = json.loads(result)

            if tuning_data.get('success'):
                self.tuning_job_name = tuning_data.get('job_name')
                print(f"\n   ✅ Tuning job started: {self.tuning_job_name}")

                # Monitor progress
                print("\n   Monitoring tuning progress...")
                return self.monitor_tuning_job()
            else:
                print(f"   ❌ Tuning job creation failed: {tuning_data.get('error')}")
                return False

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse tuning result: {e}")
            return False

    def monitor_tuning_job(self):
        """Monitor tuning job progress"""
        max_wait = 600  # 10 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                response = sagemaker.describe_hyper_parameter_tuning_job(
                    HyperParameterTuningJobName=self.tuning_job_name
                )

                status = response['HyperParameterTuningJobStatus']
                counters = response.get('TrainingJobStatusCounters', {})
                completed = counters.get('Completed', 0)
                in_progress = counters.get('InProgress', 0)

                print(f"   Status: {status}, Completed: {completed}, In Progress: {in_progress}")

                if status == 'Completed':
                    best_job = response.get('BestTrainingJob', {})
                    best_params = best_job.get('TunedHyperParameters', {})
                    self.results['best_hyperparameters'] = best_params

                    print(f"\n   ✅ Tuning completed!")
                    print(f"   Best hyperparameters: p={best_params.get('p')}, d={best_params.get('d')}, q={best_params.get('q')}")

                    return True

                elif status in ['Failed', 'Stopped']:
                    print(f"\n   ❌ Tuning job {status}")
                    return False

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                print(f"   ❌ Error monitoring tuning: {e}")
                return False

        print(f"\n   ⏰ Timeout after {max_wait}s")
        return False

    def train_models(self):
        """Step 6: Train ARIMA and LSTM Models"""
        print("\n" + "="*60)
        print("🚀 STEP 6: Train Models")
        print("="*60)

        if 'best_hyperparameters' not in self.results:
            print("❌ Best hyperparameters not available")
            return False

        # For this test, we'll only train ARIMA (LSTM follows same pattern)
        best_params = self.results['best_hyperparameters']

        print("\n   Training ARIMA model with best hyperparameters...")
        prompt = f"train sagemaker model on {self.feature_dataset_path} with hyperparameters {json.dumps(best_params)}"
        result = self.invoke_agent(prompt)

        if not result:
            return False

        try:
            training_data = json.loads(result)

            if training_data.get('success'):
                self.arima_training_job = training_data.get('job_name')
                print(f"\n   ✅ ARIMA training started: {self.arima_training_job}")

                # Monitor training (simplified)
                return self.monitor_training_job(self.arima_training_job)
            else:
                print(f"   ❌ Training job creation failed: {training_data.get('error')}")
                return False

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse training result: {e}")
            return False

    def monitor_training_job(self, job_name: str):
        """Monitor training job progress"""
        max_wait = 600  # 10 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                response = sagemaker.describe_training_job(TrainingJobName=job_name)
                status = response['TrainingJobStatus']

                print(f"   Training status: {status}")

                if status == 'Completed':
                    model_artifacts = response.get('ModelArtifacts', {}).get('S3ModelArtifacts')
                    print(f"\n   ✅ Training completed!")
                    print(f"   Model artifacts: {model_artifacts}")
                    return True

                elif status in ['Failed', 'Stopped']:
                    print(f"\n   ❌ Training {status}")
                    return False

                time.sleep(20)

            except Exception as e:
                print(f"   ❌ Error monitoring training: {e}")
                return False

        print(f"\n   ⏰ Timeout after {max_wait}s")
        return False

    def generate_report(self):
        """Step 7: Generate Comprehensive Report"""
        print("\n" + "="*60)
        print("📋 STEP 7: Generate Comprehensive Report")
        print("="*60)

        if not hasattr(self, 'feature_dataset_path'):
            print("❌ Feature dataset not available")
            return False

        # Get selected feature config from results
        recs = self.results.get('recommendations', {})
        feature_config = {
            'lag_features': {
                'enabled': True,
                'lags': recs.get('lag_features', {}).get('lags', [1, 7])[:3]
            },
            'rolling_features': {
                'enabled': True,
                'windows': recs.get('rolling_features', {}).get('windows', [7])[:2],
                'statistics': ['mean', 'std']
            },
            'calendar_features': {
                'enabled': True,
                'features': ['day_of_week', 'is_weekend']
            }
        }

        best_params = self.results.get('best_hyperparameters', {'p': 2, 'd': 1, 'q': 2})

        prompt = f"""generate comprehensive report for {self.feature_dataset_path}
        with eda results {json.dumps(self.results.get('eda', {}))}
        and feature config {json.dumps(feature_config)}
        and best hyperparameters {json.dumps(best_params)}
        and forecast horizon 7"""

        result = self.invoke_agent(prompt)

        if not result:
            return False

        try:
            report_data = json.loads(result)

            if report_data.get('status') == 'success':
                report_url = report_data.get('report_url')
                metrics = report_data.get('metrics', {})

                print(f"\n   ✅ Report generated!")
                print(f"   📄 Report URL: {report_url}")
                print(f"\n   📊 Model Metrics:")
                print(f"      MAE:  {metrics.get('mae', 0):.2f}")
                print(f"      RMSE: {metrics.get('rmse', 0):.2f}")
                print(f"      MAPE: {metrics.get('mape', 0):.1f}%")
                print(f"      AIC:  {metrics.get('aic', 0):.2f}")
                print(f"      BIC:  {metrics.get('bic', 0):.2f}")

                # Validate forecast
                forecast = report_data.get('forecast', [])
                if forecast:
                    print(f"\n   📈 7-day Forecast:")
                    for i, point in enumerate(forecast[:3]):
                        print(f"      Day {i+1}: {point.get('forecast', 0):.2f} (CI: [{point.get('lower_ci', 0):.2f}, {point.get('upper_ci', 0):.2f}])")
                    if len(forecast) > 3:
                        print(f"      ... and {len(forecast)-3} more days")

                return True
            else:
                print(f"   ❌ Report generation failed: {report_data.get('error', 'Unknown error')}")
                return False

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse report result: {e}")
            return False

    def run_full_workflow(self):
        """Run complete 7-step workflow"""
        print("\n" + "="*70)
        print("🤖 INTELLIGENT FORECASTING DASHBOARD - SIMULATION TEST")
        print("="*70)

        steps = [
            ("Create Test Data", self.create_test_data),
            ("Run Advanced EDA", self.run_advanced_eda),
            ("Get Feature Recommendations", self.get_feature_recommendations),
            ("Create Selected Features", self.create_selected_features),
            ("Start Hyperparameter Tuning", self.start_tuning),
            ("Train Models", self.train_models),
            ("Generate Comprehensive Report", self.generate_report)
        ]

        passed = 0
        failed = 0

        for step_name, step_func in steps:
            try:
                success = step_func()
                if success:
                    passed += 1
                    print(f"\n✅ {step_name}: PASSED")
                else:
                    failed += 1
                    print(f"\n❌ {step_name}: FAILED")
                    # Don't continue if critical step fails
                    if step_name in ["Create Test Data", "Run Advanced EDA"]:
                        break
            except Exception as e:
                failed += 1
                print(f"\n❌ {step_name}: FAILED - {str(e)}")
                break

        # Summary
        print("\n" + "="*70)
        print("📊 SIMULATION TEST SUMMARY")
        print("="*70)
        print(f"Total Steps: {len(steps)}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")

        if self.errors:
            print(f"\n⚠️  Errors encountered:")
            for error in self.errors:
                print(f"   - {error}")

        print("="*70)

        return failed == 0

if __name__ == "__main__":
    sim = DashboardSimulation()
    success = sim.run_full_workflow()
    sys.exit(0 if success else 1)
