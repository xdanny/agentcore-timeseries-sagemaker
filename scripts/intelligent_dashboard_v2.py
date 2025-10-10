#!/usr/bin/env python3
"""
Intelligent Forecasting Dashboard - Direct Tool Calls
Clean workflow with proper status tracking
"""
import streamlit as st
import boto3
import json
import pandas as pd
import time
from datetime import datetime
import sys
import os

# Add parent directory to path for agent imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Intelligent Forecasting System", layout="wide")

# AWS Clients
s3 = boto3.client('s3', region_name='us-east-1')
sagemaker = boto3.client('sagemaker', region_name='us-east-1')

BUCKET = os.environ.get('BUCKET_NAME', 'sagemaker-forecasting-{}-{}'.format(os.environ.get('AWS_REGION', 'us-east-1'), boto3.client('sts').get_caller_identity()['Account']))
SAGEMAKER_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", f"arn:aws:iam::{boto3.client('sts').get_caller_identity()['Account']}:role/SageMaker-ForecastingPipeline-ExecutionRole")

st.title("🤖 Intelligent Time Series Forecasting")
st.markdown("**Powered by AgentCore + SageMaker**")

# Session state initialization
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = None

# === STEP 1: DATA UPLOAD & COLUMN SELECTION ===
st.header("📊 Step 1: Upload Data & Select Columns")
uploaded_file = st.file_uploader("Upload time series CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview Data")
    st.dataframe(df.head())

    # First upload to S3 for agent analysis
    if 'temp_dataset_path' not in st.session_state:
        if st.button("🧠 Analyze Dataset Structure"):
            with st.spinner("Agent is analyzing your dataset..."):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                s3_key = f'uploads/{timestamp}_{uploaded_file.name}'
                s3.put_object(Bucket=BUCKET, Key=s3_key, Body=uploaded_file.getvalue())
                st.session_state.temp_dataset_path = s3_key

                # Call intelligent agent
                try:
                    from agents.dataset_analysis_agent import analyze_dataset_structure

                    analysis_text = analyze_dataset_structure(s3_key)
                    analysis_result = json.loads(analysis_text)

                    # Check for errors
                    if 'error' in analysis_result:
                        st.error(f"❌ Agent Error: {analysis_result['error']}")
                        if 'debug' in analysis_result:
                            with st.expander("🔍 Debug Output"):
                                st.code(analysis_result['debug'])
                        if 'hint' in analysis_result:
                            st.info(f"💡 {analysis_result['hint']}")
                    else:
                        st.session_state.dataset_analysis = analysis_result
                        st.success("✅ Analysis complete!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # Show agent recommendations
    if 'dataset_analysis' in st.session_state:
        analysis = st.session_state.dataset_analysis

        st.subheader("🤖 Agent Recommendations")

        # Show model recommendation (if available)
        if 'model_recommendation' in analysis:
            model_rec = analysis['model_recommendation']
            if model_rec['type'] == 'multiple_univariate':
                st.info(f"📊 **{model_rec['reason']}**")
                st.markdown(f"💡 Suggested approach: {model_rec['suggested_approach']}")
            elif model_rec['type'] == 'multivariate':
                st.success(f"✅ **{model_rec['reason']}**")
                st.markdown(f"💡 Suggested approach: {model_rec['suggested_approach']}")
            else:
                st.info(f"📈 **{model_rec['reason']}**")
                st.markdown(f"💡 Suggested approach: {model_rec['suggested_approach']}")
        else:
            st.warning("⚠️ Model recommendation not available - agent analysis incomplete")

        # Time column
        st.subheader("📋 Column Selection")
        time_rec = analysis['time_column']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🕐 Time Column**")
            time_column = st.selectbox(
                "Select time column:",
                options=df.columns,
                index=list(df.columns).index(time_rec['recommended']),
                help=f"Agent confidence: {time_rec['candidates'][0]['confidence']}"
            )
            st.session_state.time_column = time_column
            st.caption(f"✨ Agent says: {time_rec['candidates'][0]['reason']}")

        with col2:
            st.markdown("**🎯 Target Variable**")
            target_recs = analysis['target_recommendations']

            # Get primary target recommendation
            primary_target = target_recs[0]['column'] if target_recs else df.columns[1]

            value_column = st.selectbox(
                "Select target to forecast:",
                options=[rec['column'] for rec in target_recs] if target_recs else df.columns,
                index=0,
                help="Choose which variable to forecast"
            )
            st.session_state.value_column = value_column

            # Show agent reasoning
            for rec in target_recs:
                if rec['column'] == value_column:
                    st.caption(f"✨ Agent says: {rec['reason']}")
                    break

        # Show all potential targets if multiple detected
        if len(target_recs) > 1:
            with st.expander("👀 All Potential Targets Detected"):
                for rec in target_recs:
                    st.markdown(f"- **{rec['column']}** - {rec['reason']} (confidence: {rec['confidence']})")

        # Exogenous variables
        st.subheader("🔍 Exogenous Variables")

        exog_for_target = analysis['exogenous_recommendations'].get(value_column, [])

        if len(exog_for_target) > 0:
            st.info(f"🎯 Agent found {len(exog_for_target)} potential exogenous variables for `{value_column}`")

            use_exog = st.checkbox(
                "Use exogenous variables for multivariate forecasting?",
                value=len(exog_for_target) > 0,
                help="Exogenous variables are external factors that influence the target"
            )

            if use_exog:
                # Show agent recommendations
                with st.expander("💡 Agent Recommendations"):
                    for exog in exog_for_target:
                        st.markdown(f"- **{exog['column']}** (correlation: {exog['correlation']:.2f}) - {exog['reason']}")

                selected_exog = st.multiselect(
                    "Select exogenous variables:",
                    options=[e['column'] for e in exog_for_target],
                    default=[e['column'] for e in exog_for_target if e['confidence'] == 'high'],
                    help="Agent pre-selected high-confidence variables"
                )
                st.session_state.feature_columns = selected_exog
                st.session_state.is_multivariate = len(selected_exog) > 0

                if len(selected_exog) > 0:
                    st.success(f"✅ Multivariate ARIMAX with {len(selected_exog)} exogenous variables")
            else:
                st.session_state.feature_columns = []
                st.session_state.is_multivariate = False
                st.info("📈 Using univariate ARIMA model")
        else:
            st.info("📈 No exogenous variables detected - using univariate ARIMA")
            st.session_state.feature_columns = []
            st.session_state.is_multivariate = False

        # Data quality summary
        with st.expander("📊 Data Quality Report"):
            quality = analysis['quality_metrics']
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Rows", f"{quality['total_rows']:,}")
            with col_b:
                st.metric("Numeric Columns", quality['numeric_columns'])
            with col_c:
                st.metric("Missing Data", f"{quality['missing_data_pct']:.1f}%")

            st.markdown(f"**Date Range**: {quality['date_range']['start']} to {quality['date_range']['end']}")

        if st.button("✅ Confirm Selection & Continue"):
            st.session_state.dataset_path = st.session_state.temp_dataset_path
            st.success(f"✅ Configuration saved! Proceeding to EDA...")

# === STEP 2: INTELLIGENT EDA ===
if 'dataset_path' in st.session_state:
    st.header("🔍 Step 2: Advanced EDA")

    # Show selected columns summary
    st.info(f"📊 Analyzing **{st.session_state.value_column}** over **{st.session_state.time_column}** "
            f"({'Multivariate' if st.session_state.is_multivariate else 'Univariate'} model)")

    if st.button("🚀 Run Advanced EDA"):
        with st.spinner("Running intelligent EDA analysis..."):
            try:
                from agents.advanced_eda_agent import run_advanced_eda

                # Pass column selections to EDA
                result_text = run_advanced_eda(
                    st.session_state.dataset_path,
                    time_column=st.session_state.time_column,
                    value_column=st.session_state.value_column
                )
                st.session_state.eda_results = json.loads(result_text)
                st.success("✅ EDA Complete!")
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

# Display EDA Results
if 'eda_results' in st.session_state and st.session_state.eda_results:
    st.subheader("📈 EDA Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Recommended Differencing",
                  st.session_state.eda_results.get('stationarity', {}).get('recommended_d', 'N/A'))
    with col2:
        seasonal_strength = st.session_state.eda_results.get('decomposition', {}).get('seasonal_strength', 0)
        st.metric("Seasonal Strength", f"{seasonal_strength:.2f}")
    with col3:
        trend_slope = st.session_state.eda_results.get('trend', {}).get('slope', 0)
        st.metric("Trend Slope", f"{trend_slope:.4f}")

    # Show interactive report (embedded)
    if 'report_url' in st.session_state.eda_results:
        report_url = st.session_state.eda_results['report_url']
        st.markdown(f"### 📊 Interactive EDA Report")
        st.markdown(f"[🔗 Open in New Tab]({report_url})")

        # Embed the report
        st.components.v1.iframe(report_url, height=1200, scrolling=True)

    # === STEP 3: FEATURE ENGINEERING ===
    st.header("⚙️ Step 3: Feature Engineering")

    if st.button("🧠 Get Feature Recommendations"):
        with st.spinner("Agent is analyzing data..."):
            try:
                from agents.intelligent_feature_engineering_agent import recommend_features

                result_text = recommend_features(st.session_state.dataset_path,
                                                json.dumps(st.session_state.eda_results))
                st.session_state.feature_recommendations = json.loads(result_text)
                st.success("✅ Recommendations Ready!")
            except Exception as e:
                st.error(f"Error: {e}")

# Display Feature Recommendations
if 'feature_recommendations' in st.session_state and st.session_state.feature_recommendations:
    st.subheader("💡 Recommended Features")

    recs = st.session_state.feature_recommendations
    selected_config = {}

    # Lag features
    if 'lag_features' in recs and recs['lag_features']['enabled']:
        with st.expander(f"🔄 Lag Features"):
            st.markdown(f"**Justification**: {recs['lag_features']['justification']}")
            enable_lag = st.checkbox("Enable Lag Features", value=True, key='lag')
            if enable_lag:
                selected_lags = st.multiselect("Select Lags", recs['lag_features']['lags'],
                                              default=recs['lag_features']['lags'])
                selected_config['lag_features'] = {'enabled': True, 'lags': selected_lags}

    # Rolling features
    if 'rolling_features' in recs and recs['rolling_features']['enabled']:
        with st.expander(f"📊 Rolling Features"):
            enable_roll = st.checkbox("Enable Rolling Features", value=True, key='roll')
            if enable_roll:
                windows = st.multiselect("Windows", recs['rolling_features']['windows'],
                                        default=recs['rolling_features']['windows'])
                stats = st.multiselect("Statistics", recs['rolling_features']['statistics'],
                                      default=recs['rolling_features']['statistics'])
                selected_config['rolling_features'] = {'enabled': True, 'windows': windows, 'statistics': stats}

    # Calendar features
    if 'calendar_features' in recs:
        with st.expander(f"📅 Calendar Features"):
            enable_cal = st.checkbox("Enable Calendar Features", value=True, key='cal')
            if enable_cal:
                cal_feats = st.multiselect("Select Calendar Features",
                                           recs['calendar_features']['features'],
                                           default=recs['calendar_features']['features'])
                selected_config['calendar_features'] = {'enabled': True, 'features': cal_feats}

    if st.button("✨ Create Selected Features"):
        with st.spinner("Creating features..."):
            try:
                from agents.intelligent_feature_engineering_agent import create_features

                result_text = create_features(st.session_state.dataset_path,
                                             json.dumps(selected_config))
                feature_result = json.loads(result_text)
                st.session_state.feature_dataset_path = feature_result['output_path']
                st.session_state.selected_config = selected_config
                st.success(f"✅ Created {feature_result['feature_count']} features!")
                st.info(f"Output: {feature_result['output_path']}")
            except Exception as e:
                st.error(f"Error: {e}")

# === STEP 4: MODEL SELECTION ===
if 'feature_dataset_path' in st.session_state:
    st.header("🎯 Step 4: Model Selection & Tuning")

    model_choice = st.radio(
        "Select Model(s) to Train:",
        ["ARIMA Only", "LSTM Only", "Both ARIMA and LSTM"],
        key='model_selection'
    )
    st.session_state.model_choice = model_choice

    col1, col2 = st.columns(2)
    with col1:
        max_jobs = st.number_input("Max Tuning Jobs", 5, 20, 10)
    with col2:
        parallel_jobs = st.number_input("Parallel Jobs", 1, 5, 2)

    # Single button to start tuning based on selection
    if st.button("🚀 Start Hyperparameter Tuning"):
        # Fix S3 path - remove duplicate bucket prefix
        dataset_path = st.session_state.feature_dataset_path
        if dataset_path.startswith('s3://'):
            # Already has s3:// prefix, use as-is
            full_path = dataset_path
        else:
            # Add bucket prefix
            full_path = f"s3://{BUCKET}/{dataset_path}"

        # Start ARIMA tuning if selected
        if model_choice in ["ARIMA Only", "Both ARIMA and LSTM"]:
            with st.spinner("Creating ARIMA tuning job..."):
                try:
                    from agents.sagemaker_tuning import create_sagemaker_tuning_job

                    job_name = f"arima-tuning-{int(time.time())}"
                    result_text = create_sagemaker_tuning_job(
                        dataset_s3_path=full_path,
                        role_arn=SAGEMAKER_ROLE_ARN,
                        job_name=job_name,
                        max_jobs=max_jobs,
                        max_parallel_jobs=parallel_jobs
                    )
                    tuning_result = json.loads(result_text)

                    if tuning_result.get('success'):
                        st.session_state.arima_tuning_job = tuning_result['job_name']
                        st.success(f"✅ ARIMA tuning started: {st.session_state.arima_tuning_job}")
                    else:
                        st.error(f"ARIMA Error: {tuning_result.get('error')}")
                except Exception as e:
                    st.error(f"ARIMA Error: {e}")

        # Start LSTM tuning if selected
        if model_choice in ["LSTM Only", "Both ARIMA and LSTM"]:
            with st.spinner("Creating LSTM tuning job..."):
                try:
                    from agents.sagemaker_lstm import create_lstm_tuning_job

                    job_name = f"lstm-tuning-{int(time.time())}"
                    result_text = create_lstm_tuning_job(
                        dataset_s3_path=full_path,
                        role_arn=SAGEMAKER_ROLE_ARN,
                        job_name=job_name,
                        max_jobs=max_jobs,
                        max_parallel_jobs=parallel_jobs
                    )
                    tuning_result = json.loads(result_text)

                    if tuning_result.get('success'):
                        st.session_state.lstm_tuning_job = tuning_result['job_name']
                        st.success(f"✅ LSTM tuning started: {st.session_state.lstm_tuning_job}")
                    else:
                        st.error(f"LSTM Error: {tuning_result.get('error')}")
                except Exception as e:
                    st.error(f"LSTM Error: {e}")

        if model_choice == "Both ARIMA and LSTM":
            st.info("💡 Both tuning jobs are running in parallel")

# === STEP 5: TUNING PROGRESS ===
if 'arima_tuning_job' in st.session_state or 'lstm_tuning_job' in st.session_state:
    st.header("📊 Step 5: Tuning Progress")

    # Check if any jobs are still in progress
    any_in_progress = False

    col1, col2 = st.columns(2)

    # ARIMA Progress
    if 'arima_tuning_job' in st.session_state:
        with col1:
            st.subheader("ARIMA Tuning")
            try:
                job_desc = sagemaker.describe_hyper_parameter_tuning_job(
                    HyperParameterTuningJobName=st.session_state.arima_tuning_job
                )
                status = job_desc['HyperParameterTuningJobStatus']
                counters = job_desc.get('TrainingJobStatusCounters', {})

                # Status indicator with color
                if status == 'InProgress':
                    st.info(f"⏳ Status: {status}")
                    any_in_progress = True
                elif status == 'Completed':
                    st.success(f"✅ Status: {status}")
                else:
                    st.error(f"❌ Status: {status}")

                # Progress metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Completed", counters.get('Completed', 0))
                with col_b:
                    st.metric("In Progress", counters.get('InProgress', 0))
                with col_c:
                    st.metric("Failed", counters.get('Failed', 0))

                # Progress bar
                max_jobs = counters.get('Completed', 0) + counters.get('InProgress', 0) + counters.get('Failed', 0)
                if max_jobs > 0:
                    progress = counters.get('Completed', 0) / max_jobs
                    st.progress(progress)
                    st.caption(f"{counters.get('Completed', 0)}/{max_jobs} jobs completed")

                if status == 'Completed':
                    best_params = job_desc.get('BestTrainingJob', {}).get('TunedHyperParameters', {})
                    st.session_state.arima_best_params = best_params

                    # Show best params
                    with st.expander("🏆 Best Hyperparameters"):
                        st.json(best_params)

                        best_metric = job_desc.get('BestTrainingJob', {}).get('FinalHyperParameterTuningJobObjectiveMetric', {})
                        if best_metric:
                            st.metric("Best Objective Metric", f"{best_metric.get('Value', 0):.4f}")

            except Exception as e:
                st.error(f"Error: {e}")

    # LSTM Progress
    if 'lstm_tuning_job' in st.session_state:
        with col2:
            st.subheader("LSTM Tuning")
            try:
                job_desc = sagemaker.describe_hyper_parameter_tuning_job(
                    HyperParameterTuningJobName=st.session_state.lstm_tuning_job
                )
                status = job_desc['HyperParameterTuningJobStatus']
                counters = job_desc.get('TrainingJobStatusCounters', {})

                # Status indicator with color
                if status == 'InProgress':
                    st.info(f"⏳ Status: {status}")
                    any_in_progress = True
                elif status == 'Completed':
                    st.success(f"✅ Status: {status}")
                elif status == 'Failed':
                    st.error(f"❌ Status: {status}")

                    # Get failure reason
                    failure_reason = job_desc.get('FailureReason', 'No failure reason provided')
                    with st.expander("🔍 View Failure Reason"):
                        st.error(failure_reason)

                        st.warning("""
                        **LSTM Training Requirements:**
                        - Requires custom training script (`train_lstm.py`)
                        - Script must be uploaded to S3 as `lstm_sourcedir.tar.gz`
                        - TensorFlow/Keras dependencies must be configured

                        💡 **Recommendation**: Continue with ARIMA model, which works out-of-the-box.
                        LSTM can be added later with custom training scripts.
                        """)
                else:
                    st.warning(f"⚠️ Status: {status}")

                # Progress metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Completed", counters.get('Completed', 0))
                with col_b:
                    st.metric("In Progress", counters.get('InProgress', 0))
                with col_c:
                    st.metric("Failed", counters.get('Failed', 0))

                # Progress bar
                max_jobs = counters.get('Completed', 0) + counters.get('InProgress', 0) + counters.get('Failed', 0)
                if max_jobs > 0:
                    progress = counters.get('Completed', 0) / max_jobs
                    st.progress(progress)
                    st.caption(f"{counters.get('Completed', 0)}/{max_jobs} jobs completed")

                if status == 'Completed':
                    best_params = job_desc.get('BestTrainingJob', {}).get('TunedHyperParameters', {})
                    st.session_state.lstm_best_params = best_params

                    # Show best params
                    with st.expander("🏆 Best Hyperparameters"):
                        st.json(best_params)

                        best_metric = job_desc.get('BestTrainingJob', {}).get('FinalHyperParameterTuningJobObjectiveMetric', {})
                        if best_metric:
                            st.metric("Best Objective Metric", f"{best_metric.get('Value', 0):.4f}")

            except Exception as e:
                st.error(f"Error: {e}")

    # Auto-refresh if jobs are in progress
    if any_in_progress:
        st.info("💡 Page auto-refreshing every 30 seconds while jobs are in progress...")
        time.sleep(30)
        st.rerun()
    else:
        if st.button("🔄 Refresh Status"):
            st.rerun()

# === STEP 6: TRAINING ===
if ('arima_best_params' in st.session_state or 'lstm_best_params' in st.session_state):
    st.header("🎓 Step 6: Model Training")

    col1, col2 = st.columns(2)

    # Fix S3 path - reuse logic from tuning
    dataset_path = st.session_state.feature_dataset_path
    if dataset_path.startswith('s3://'):
        full_path = dataset_path
    else:
        full_path = f"s3://{BUCKET}/{dataset_path}"

    # ARIMA Training
    if 'arima_best_params' in st.session_state:
        with col1:
            st.subheader("ARIMA Training")
            if 'arima_training_job' not in st.session_state:
                if st.button("🚀 Train ARIMA Model"):
                    with st.spinner("Starting ARIMA training..."):
                        try:
                            from agents.sagemaker_simple import create_sagemaker_training_job

                            job_name = f"arima-training-{int(time.time())}"
                            result_text = create_sagemaker_training_job(
                                job_name=job_name,
                                dataset_s3_path=full_path,
                                role_arn=SAGEMAKER_ROLE_ARN
                            )
                            training_result = json.loads(result_text)

                            if training_result.get('success'):
                                st.session_state.arima_training_job = job_name
                                st.success(f"✅ Training started: {job_name}")
                                st.rerun()
                            else:
                                st.error(f"Error: {training_result.get('error')}")
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                # Show training status
                try:
                    from agents.sagemaker_simple import get_training_job_status

                    status_text = get_training_job_status(st.session_state.arima_training_job)
                    status = json.loads(status_text)

                    if status.get('success'):
                        st.metric("Training Status", status['status'])
                        if status['status'] == 'Completed':
                            st.session_state.arima_model_artifacts = status['model_artifacts']
                            st.success("✅ Training Complete!")
                        elif status['status'] in ['Failed', 'Stopped']:
                            st.error(f"❌ Training {status['status']}")
                        else:
                            if st.button("🔄 Check ARIMA Status"):
                                st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # LSTM Training
    if 'lstm_best_params' in st.session_state:
        with col2:
            st.subheader("LSTM Training")
            if 'lstm_training_job' not in st.session_state:
                if st.button("🚀 Train LSTM Model"):
                    with st.spinner("Starting LSTM training..."):
                        try:
                            from agents.sagemaker_lstm import create_lstm_training_job

                            job_name = f"lstm-training-{int(time.time())}"
                            result_text = create_lstm_training_job(
                                dataset_s3_path=full_path,
                                role_arn=SAGEMAKER_ROLE_ARN,
                                job_name=job_name,
                                hyperparameters=st.session_state.lstm_best_params
                            )
                            training_result = json.loads(result_text)

                            if training_result.get('success'):
                                st.session_state.lstm_training_job = job_name
                                st.success(f"✅ Training started: {job_name}")
                                st.rerun()
                            else:
                                st.error(f"Error: {training_result.get('error')}")
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                # Show training status
                try:
                    from agents.sagemaker_lstm import get_lstm_training_status

                    status_text = get_lstm_training_status(st.session_state.lstm_training_job)
                    status = json.loads(status_text)

                    if status.get('success'):
                        st.metric("Training Status", status['status'])
                        if status['status'] == 'Completed':
                            st.session_state.lstm_model_artifacts = status['model_artifacts']
                            st.success("✅ Training Complete!")
                        elif status['status'] in ['Failed', 'Stopped']:
                            st.error(f"❌ Training {status['status']}")
                        else:
                            if st.button("🔄 Check LSTM Status"):
                                st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

# === STEP 7: DEPLOYMENT ===
if ('arima_model_artifacts' in st.session_state or 'lstm_model_artifacts' in st.session_state):
    st.header("🚀 Step 7: Deployment")

    col1, col2 = st.columns(2)

    # ARIMA Deployment
    if 'arima_model_artifacts' in st.session_state:
        with col1:
            st.subheader("ARIMA Deployment")
            if 'arima_endpoint' not in st.session_state:
                if st.button("📡 Deploy ARIMA Endpoint"):
                    with st.spinner("Deploying ARIMA..."):
                        try:
                            from agents.sagemaker_simple import deploy_sagemaker_model

                            endpoint_name = f"arima-endpoint-{int(time.time())}"
                            model_name = f"arima-model-{int(time.time())}"

                            result_text = deploy_sagemaker_model(
                                model_name=model_name,
                                endpoint_name=endpoint_name,
                                model_data_url=st.session_state.arima_model_artifacts,
                                role_arn=SAGEMAKER_ROLE_ARN
                            )
                            deploy_result = json.loads(result_text)

                            if deploy_result.get('success'):
                                st.session_state.arima_endpoint = endpoint_name
                                st.success(f"✅ Deployed: {endpoint_name}")
                                st.rerun()
                            else:
                                st.error(f"Error: {deploy_result.get('error')}")
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                st.success(f"✅ Endpoint: {st.session_state.arima_endpoint}")

    # LSTM Deployment
    if 'lstm_model_artifacts' in st.session_state:
        with col2:
            st.subheader("LSTM Deployment")
            if 'lstm_endpoint' not in st.session_state:
                if st.button("📡 Deploy LSTM Endpoint"):
                    with st.spinner("Deploying LSTM..."):
                        try:
                            from agents.sagemaker_lstm import deploy_lstm_model

                            endpoint_name = f"lstm-endpoint-{int(time.time())}"
                            model_name = f"lstm-model-{int(time.time())}"

                            result_text = deploy_lstm_model(
                                model_name=model_name,
                                endpoint_name=endpoint_name,
                                model_data_url=st.session_state.lstm_model_artifacts,
                                role_arn=SAGEMAKER_ROLE_ARN
                            )
                            deploy_result = json.loads(result_text)

                            if deploy_result.get('success'):
                                st.session_state.lstm_endpoint = endpoint_name
                                st.success(f"✅ Deployed: {endpoint_name}")
                                st.rerun()
                            else:
                                st.error(f"Error: {deploy_result.get('error')}")
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                st.success(f"✅ Endpoint: {st.session_state.lstm_endpoint}")

# === STEP 8: COMPREHENSIVE REPORT ===
if ('arima_endpoint' in st.session_state or 'lstm_endpoint' in st.session_state):
    st.header("📊 Step 8: Generate Comprehensive Report")

    # Show endpoint status with auto-refresh
    endpoint = st.session_state.get('arima_endpoint') or st.session_state.get('lstm_endpoint')

    endpoint_ready = False
    try:
        endpoint_desc = sagemaker.describe_endpoint(EndpointName=endpoint)
        endpoint_status = endpoint_desc['EndpointStatus']

        if endpoint_status == 'InService':
            st.success(f"✅ Endpoint Ready: {endpoint} ({endpoint_status})")
            endpoint_ready = True
        elif endpoint_status == 'Creating' or endpoint_status == 'Updating':
            st.info(f"⏳ Endpoint Status: {endpoint_status}")
            st.warning("🔄 Endpoint is deploying... Page will auto-refresh every 30 seconds")

            # Show deployment progress
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Endpoint Name", endpoint)
            with col2:
                st.metric("Status", endpoint_status)

            # Auto-refresh while deploying
            time.sleep(30)
            st.rerun()
        else:
            st.warning(f"⚠️ Endpoint Status: {endpoint_status}")
            if 'FailureReason' in endpoint_desc:
                with st.expander("🔍 View Failure Reason"):
                    st.error(endpoint_desc['FailureReason'])
    except Exception as e:
        st.error(f"❌ Endpoint Error: {e}")

    if st.button("📋 Generate Report", disabled=not endpoint_ready):
        with st.spinner("Generating comprehensive report..."):
            try:
                from agents.comprehensive_report_agent import generate_comprehensive_report

                # Use whichever endpoint is available
                endpoint = st.session_state.get('arima_endpoint') or st.session_state.get('lstm_endpoint')

                result_text = generate_comprehensive_report(
                    dataset_s3_path=st.session_state.feature_dataset_path,
                    endpoint_name=endpoint,
                    eda_results=json.dumps(st.session_state.eda_results),
                    forecast_horizon=7
                )
                report_result = json.loads(result_text)

                if 'error' in report_result:
                    st.error(f"❌ Error: {report_result['error']}")
                    if 'debug_output' in report_result:
                        with st.expander("🔍 Debug Output"):
                            st.code(report_result['debug_output'])
                    if 'hint' in report_result:
                        st.info(f"💡 {report_result['hint']}")
                    if 'traceback' in report_result:
                        with st.expander("📋 Full Traceback"):
                            st.code(report_result['traceback'])
                else:
                    st.session_state.report_result = report_result
                    st.success("✅ Report generated!")

                    # Display metrics
                    metrics = report_result.get('metrics', {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MAE", f"{metrics.get('mae', 0):.2f}")
                    with col2:
                        st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
                    with col3:
                        st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")

                    # Show report link
                    if 'report_url' in report_result:
                        st.markdown(f"### 📊 Forecasting Report")
                        st.markdown(f"[🔗 Open in New Tab]({report_result['report_url']})")

                        # Embed the report
                        st.markdown("---")
                        st.components.v1.iframe(report_result['report_url'], height=1400, scrolling=True)

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

st.sidebar.markdown("---")
st.sidebar.info("💡 Direct tool calls for clean, predictable workflow")
