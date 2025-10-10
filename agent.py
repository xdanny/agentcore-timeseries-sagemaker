"""
Autonomous Time Series Forecasting Agent with SageMaker Integration
Direct tool orchestration for complete pipeline execution
"""
import json
import logging
import re
import os
from datetime import datetime
from strands import Agent
from agents.advanced_eda_agent import run_advanced_eda
from agents.intelligent_feature_engineering_agent import recommend_features, create_features
from agents.comprehensive_report_agent import generate_comprehensive_report
from agents.sagemaker_simple import (
    create_sagemaker_training_job,
    get_training_job_status,
    deploy_sagemaker_model,
    invoke_sagemaker_endpoint
)
from agents.sagemaker_tuning import (
    create_sagemaker_tuning_job,
    get_tuning_job_status
)
from agents.sagemaker_lstm import (
    create_lstm_training_job,
    get_lstm_training_status,
    create_lstm_tuning_job,
    get_lstm_tuning_status,
    deploy_lstm_model,
    invoke_lstm_endpoint
)
from fastapi import FastAPI
import time
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded SageMaker role ARN
SAGEMAKER_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", f"arn:aws:iam::{boto3.client('sts').get_caller_identity()['Account']}:role/SageMaker-ForecastingPipeline-ExecutionRole")

logger.info(f"SageMaker Role: {SAGEMAKER_ROLE_ARN}")

# Create autonomous forecasting agent with graph workflow
agent = Agent(
    name="IntelligentForecastingAgent",
    description="""Intelligent time series forecasting with ARIMA and LSTM model comparison.

Agent performs advanced statistical analysis and makes intelligent recommendations:

1. **Advanced EDA**: Stationarity tests (ADF, KPSS), seasonal decomposition, ACF/PACF analysis, trend detection
2. **Feature Recommendations**: Analyzes data characteristics to suggest optimal features with justifications
3. **Feature Engineering**: Creates user-selected features (lags, rolling stats, calendar, Fourier, etc.)
4. **Dual Model Training**:
   - ARIMA: SageMaker hyperparameter tuning for p, q parameters
   - LSTM: Deep learning model for sequence prediction
5. **Model Comparison**: Evaluate both models on same dataset
6. **Comprehensive Report**: Model diagnostics, backtesting, forecast with CI, residual analysis

Agent provides intelligence and recommendations. User controls workflow via UI buttons.""",
    tools=[
        # Intelligent analysis and feature engineering
        run_advanced_eda,
        recommend_features,
        create_features,
        # ARIMA pipeline (SageMaker)
        create_sagemaker_tuning_job,
        get_tuning_job_status,
        create_sagemaker_training_job,
        get_training_job_status,
        deploy_sagemaker_model,
        invoke_sagemaker_endpoint,
        # LSTM pipeline (SageMaker)
        create_lstm_tuning_job,
        get_lstm_tuning_status,
        create_lstm_training_job,
        get_lstm_training_status,
        deploy_lstm_model,
        invoke_lstm_endpoint,
        # Comprehensive reporting
        generate_comprehensive_report
    ]
)

logger.info("Autonomous forecasting agent initialized")

# FastAPI app
app = FastAPI()

@app.get("/ping")
@app.post("/ping")
async def ping():
    """Health check endpoint required by AgentCore"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

def extract_params(prompt: str):
    """Extract S3 path and column names from prompt"""
    s3_match = re.search(r's3://[a-zA-Z0-9\-._/]+', prompt)

    # Extract column names
    time_col_match = re.search(r'time_column=(\w+)', prompt)
    value_col_match = re.search(r'value_column=(\w+)', prompt)

    # Try to extract role from prompt, but prefer environment variable
    role_match = re.search(r'arn:aws:iam::[0-9]+:role/[a-zA-Z0-9\-]+', prompt)
    role_arn = role_match.group(0) if role_match else SAGEMAKER_ROLE_ARN

    return {
        'dataset_s3_path': s3_match.group(0) if s3_match else None,
        'role_arn': role_arn,
        'time_column': time_col_match.group(1) if time_col_match else None,
        'value_column': value_col_match.group(1) if value_col_match else None
    }

def execute_pipeline(dataset_s3_path: str, role_arn: str, workflow_id: str, time_column: str = None, value_column: str = None):
    """Execute complete forecasting pipeline with direct tool orchestration and status tracking"""
    results = {}
    state_manager = WorkflowStateManager(workflow_id)
    hook = create_strands_hook(workflow_id)

    try:
        # Step 1: EDA
        logger.info("📊 Step 1/8: Running EDA...")
        mark_step_running(workflow_id, "eda")
        hook.before_tool_call("run_comprehensive_eda", {"dataset_s3_path": dataset_s3_path, "time_column": time_column, "value_column": value_column})
        try:
            eda_result = run_comprehensive_eda(dataset_s3_path, time_column=time_column, value_column=value_column)
            results['eda'] = json.loads(eda_result)
            update_eda_status(workflow_id, eda_result)
            hook.after_tool_call("run_comprehensive_eda", eda_result, success=True)
            logger.info(f"✅ EDA complete")
        except Exception as e:
            hook.after_tool_call("run_comprehensive_eda", None, success=False, error=str(e))
            raise

        # Step 2: Cleaning
        logger.info("🧹 Step 2/8: Cleaning data...")
        mark_step_running(workflow_id, "cleaning")
        # Extract bucket and create cleaned path
        bucket = dataset_s3_path.split('/')[2]
        timestamp = int(time.time())
        cleaned_path = f"s3://{bucket}/cleaned/cleaned_{timestamp}.csv"

        hook.before_tool_call("clean_timeseries_data", {"dataset_s3_path": dataset_s3_path, "output_path": cleaned_path})
        try:
            clean_result = clean_timeseries_data(dataset_s3_path, cleaned_path)
            clean_data = json.loads(clean_result)
            results['cleaning'] = clean_data
            update_cleaning_status(workflow_id, clean_result)
            cleaned_path = clean_data.get('output_path', cleaned_path)
            hook.after_tool_call("clean_timeseries_data", clean_result, success=True)
            logger.info(f"✅ Cleaned data: {cleaned_path}")
        except Exception as e:
            hook.after_tool_call("clean_timeseries_data", None, success=False, error=str(e))
            raise

        # Step 3: Feature Engineering
        logger.info("🔧 Step 3/8: Generating features...")
        mark_step_running(workflow_id, "features")

        hook.before_tool_call("auto_generate_features", {"dataset_s3_path": cleaned_path})
        try:
            feature_result = auto_generate_features(cleaned_path)
            feature_data = json.loads(feature_result)
            features_path = feature_data.get('output_path', cleaned_path)
            results['features'] = feature_data
            update_features_status(workflow_id, feature_result)
            hook.after_tool_call("auto_generate_features", feature_result, success=True)
            logger.info(f"✅ Features generated: {features_path}")
        except Exception as e:
            hook.after_tool_call("auto_generate_features", None, success=False, error=str(e))
            raise

        # Step 4: ARIMA Hyperparameter Tuning
        logger.info("🎯 Step 4/9: ARIMA Hyperparameter Tuning...")
        mark_step_running(workflow_id, "arima_tuning")

        hook.before_tool_call("create_sagemaker_tuning_job", {"dataset_s3_path": cleaned_path, "role_arn": role_arn})
        try:
            arima_tuning_result = create_sagemaker_tuning_job(
                dataset_s3_path=cleaned_path,
                role_arn=role_arn,
                job_name=f"arima-tuning-{int(time.time())}"
            )
            arima_tuning_data = json.loads(arima_tuning_result)

            if not arima_tuning_data.get('success'):
                hook.after_tool_call("create_sagemaker_tuning_job", arima_tuning_result, success=False, error=arima_tuning_data.get('error'))
                raise Exception(f"ARIMA tuning job creation failed: {arima_tuning_data.get('error')}")

            arima_tuning_job_name = arima_tuning_data['job_name']
            hook.after_tool_call("create_sagemaker_tuning_job", arima_tuning_result, success=True)
            logger.info(f"✅ ARIMA tuning started: {arima_tuning_job_name}")
        except Exception as e:
            if 'hook.after_tool_call' not in str(e):
                hook.after_tool_call("create_sagemaker_tuning_job", None, success=False, error=str(e))
            raise

        # Phase 1: Create tuning jobs, don't wait (EventBridge will trigger Phase 2)
        results['arima_tuning'] = {
            'job_name': arima_tuning_job_name,
            'status': 'InProgress',
            'message': 'Tuning job created. EventBridge will trigger continuation when complete.'
        }
        update_tuning_status(workflow_id, json.dumps(results['arima_tuning']), "arima")
        logger.info(f"✅ ARIMA tuning job created: {arima_tuning_job_name}")

        # Phase 1 complete - EventBridge will trigger Phase 2 when tuning job completes
        logger.info(f"✅ Phase 1 complete. Tuning job running in background.")
        logger.info(f"   ARIMA: {arima_tuning_job_name}")
        logger.info(f"   EventBridge will automatically continue pipeline when tuning completes.")

        # Write Phase 1 completion to journal using hook
        phase1_result = {
            "status": "phase1_complete",
            "workflow_id": workflow_id,
            "tuning_job": arima_tuning_job_name,
            "message": "Phase 1 complete. EventBridge will trigger Phase 2 when tuning completes."
        }
        hook.after_tool_call("phase1_pipeline", json.dumps(phase1_result), success=True)

        return {
            "status": "phase1_complete",
            "workflow_id": workflow_id,
            "tuning_job": arima_tuning_job_name,
            "message": "Phase 1 complete. ARIMA tuning job running. Pipeline will auto-continue via EventBridge."
        }

        # Step 7: LSTM Training with tuned hyperparameters
        logger.info("🧠 Step 7/9: Training LSTM model with tuned hyperparameters...")
        mark_step_running(workflow_id, "lstm_training")

        hook.before_tool_call("create_lstm_training_job", {"dataset_s3_path": features_path, "role_arn": role_arn, "hyperparameters": best_lstm_params})
        try:
            lstm_train_result = create_lstm_training_job(
                dataset_s3_path=features_path,
                role_arn=role_arn,
                job_name=f"lstm-pipeline-{int(time.time())}",
                hyperparameters=best_lstm_params
            )
            lstm_train_data = json.loads(lstm_train_result)

            # Check for errors
            if not lstm_train_data.get('success'):
                hook.after_tool_call("create_lstm_training_job", lstm_train_result, success=False, error=lstm_train_data.get('error'))
                raise Exception(f"LSTM training job creation failed: {lstm_train_data.get('error')}")

            lstm_job_name = lstm_train_data['job_name']
            hook.after_tool_call("create_lstm_training_job", lstm_train_result, success=True)
            logger.info(f"✅ LSTM training started: {lstm_job_name}")
        except Exception as e:
            if 'hook.after_tool_call' not in str(e):
                hook.after_tool_call("create_lstm_training_job", None, success=False, error=str(e))
            raise

        # Monitor LSTM training with periodic status checks (non-blocking)
        logger.info("⏳ Monitoring LSTM training progress...")
        max_training_checks = 60  # 10 minutes max (10 sec intervals)

        for check_num in range(max_training_checks):
            # Log each status check to journal for real-time visibility
            hook.before_tool_call("check_lstm_training_status", {
                "job_name": lstm_job_name,
                "check": f"{check_num + 1}/{max_training_checks}"
            })

            status_result = get_lstm_training_status(lstm_job_name)
            status_data = json.loads(status_result)

            hook.after_tool_call("check_lstm_training_status", status_result, success=True)

            if not status_data.get('success'):
                error_msg = f"Failed to get training status: {status_data.get('error')}"
                hook.after_tool_call("check_lstm_training_status", None, success=False, error=error_msg)
                raise Exception(error_msg)

            if status_data['status'] == 'Completed':
                results['lstm_training'] = status_data
                update_training_status(workflow_id, json.dumps(status_data), "lstm")
                logger.info(f"✅ LSTM training completed")
                break
            elif status_data['status'] in ['Failed', 'Stopped']:
                error_msg = f"LSTM training failed: {status_data.get('secondary_status')}"
                raise Exception(error_msg)

            # Short sleep between checks
            time.sleep(10)

        # Step 6: Deploy ARIMA model
        logger.info("🚀 Step 6/8: Deploying ARIMA model...")
        mark_step_running(workflow_id, "arima_deployment")
        arima_model_path = results['arima_training']['model_artifacts']
        arima_deploy_result = deploy_sagemaker_model(
            model_name=f"arima-model-{int(time.time())}",
            endpoint_name=f"arima-endpoint-{int(time.time())}",
            model_data_url=arima_model_path,
            role_arn=role_arn
        )
        arima_deploy_data = json.loads(arima_deploy_result)

        if not arima_deploy_data.get('success'):
            raise Exception(f"ARIMA deployment failed: {arima_deploy_data.get('error')}")

        arima_endpoint = arima_deploy_data['endpoint_name']
        results['arima_deployment'] = arima_deploy_data
        update_deployment_status(workflow_id, arima_deploy_result, "arima")
        logger.info(f"✅ ARIMA endpoint: {arima_endpoint}")

        # Step 7: Deploy LSTM model
        logger.info("🚀 Step 7/8: Deploying LSTM model...")
        mark_step_running(workflow_id, "lstm_deployment")
        lstm_model_path = results['lstm_training']['model_artifacts']
        lstm_deploy_result = deploy_lstm_model(
            model_name=f"lstm-model-{int(time.time())}",
            endpoint_name=f"lstm-endpoint-{int(time.time())}",
            model_data_url=lstm_model_path,
            role_arn=role_arn
        )
        lstm_deploy_data = json.loads(lstm_deploy_result)

        if not lstm_deploy_data.get('success'):
            raise Exception(f"LSTM deployment failed: {lstm_deploy_data.get('error')}")

        lstm_endpoint = lstm_deploy_data['endpoint_name']
        results['lstm_deployment'] = lstm_deploy_data
        update_deployment_status(workflow_id, lstm_deploy_result, "lstm")
        logger.info(f"✅ LSTM endpoint: {lstm_endpoint}")

        # Step 8: Generate forecasts
        logger.info("🔮 Step 8/8: Generating forecasts...")
        mark_step_running(workflow_id, "inference")

        # ARIMA forecast
        arima_forecast_result = invoke_sagemaker_endpoint(
            endpoint_name=arima_endpoint,
            input_data=json.dumps({"steps": 7})
        )
        arima_forecast_data = json.loads(arima_forecast_result)

        if not arima_forecast_data.get('success'):
            raise Exception(f"ARIMA inference failed: {arima_forecast_data.get('error')}")

        results['arima_forecast'] = arima_forecast_data

        # LSTM forecast (needs historical data - use last 30 points from cleaned data)
        lstm_forecast_result = invoke_lstm_endpoint(
            endpoint_name=lstm_endpoint,
            historical_data=[],  # TODO: Extract from cleaned CSV
            steps=7
        )
        lstm_forecast_data = json.loads(lstm_forecast_result)

        if not lstm_forecast_data.get('success'):
            raise Exception(f"LSTM inference failed: {lstm_forecast_data.get('error')}")

        results['lstm_forecast'] = lstm_forecast_data

        # Update inference status
        inference_result = {
            "forecast_horizon": 7,
            "arima_predictions": arima_forecast_data.get('predictions', []),
            "lstm_predictions": lstm_forecast_data.get('predictions', [])
        }
        update_inference_status(workflow_id, json.dumps(inference_result))
        logger.info(f"✅ Forecasts generated")

        # Generate report
        logger.info("📝 Generating report...")
        mark_step_running(workflow_id, "report")
        report_result = generate_simple_report(
            eda_results=json.dumps(results['eda']),
            training_results=json.dumps({
                'arima': results['arima_training'],
                'lstm': results['lstm_training']
            }),
            forecast_results=json.dumps({
                'arima': results['arima_forecast'],
                'lstm': results['lstm_forecast']
            })
        )
        results['report'] = json.loads(report_result)
        report_s3_path = results['report'].get('report_s3_path', 'N/A')
        update_report_status(workflow_id, report_s3_path)
        logger.info(f"✅ Report generated: {report_s3_path}")

        return results

    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.post("/invocations")
async def handle_invocation(request: dict):
    """Handle AgentCore invocations - triggers complete forecasting workflow"""
    try:
        prompt = request.get("prompt", "help")
        logger.info(f"📥 Request: {prompt[:100]}...")

        # Check for Phase 2 continuation (from Lambda/EventBridge)
        if "continue workflow" in prompt.lower():
            workflow_id_match = re.search(r'workflow-[\w-]+', prompt)
            tuning_job_match = re.search(r'arima-tuning-\d+', prompt)

            if workflow_id_match and tuning_job_match:
                workflow_id = workflow_id_match.group(0)
                tuning_job_name = tuning_job_match.group(0)
                logger.info(f"🔄 Phase 2 triggered by EventBridge for {workflow_id}")
                logger.info(f"   Tuning job: {tuning_job_name}")

                # Start Phase 2 in background thread
                import threading
                def run_phase2():
                    try:
                        hook = create_strands_hook(workflow_id)

                        # Get best hyperparameters
                        logger.info("📊 Getting best hyperparameters from tuning job...")
                        tuning_status_result = get_tuning_job_status(tuning_job_name)
                        tuning_data = json.loads(tuning_status_result)

                        if tuning_data.get('status') != 'Completed':
                            logger.error(f"Tuning job not completed: {tuning_data.get('status')}")
                            return

                        best_params = tuning_data.get('best_hyperparameters', {})
                        logger.info(f"   Best params: {best_params}")

                        # Get dataset path from workflow journal
                        import boto3
                        import os
                        s3_client = boto3.client('s3', region_name='us-east-1')
                        aws_account_id = boto3.client('sts').get_caller_identity()['Account']
                        bucket_name = os.environ.get('BUCKET_NAME', f'sagemaker-forecasting-us-east-1-{aws_account_id}')
                        journal_key = f"workflows/{workflow_id}/journal.jsonl"
                        journal_obj = s3_client.get_object(Bucket=bucket_name, Key=journal_key)
                        journal_lines = journal_obj['Body'].read().decode('utf-8').split('\n')

                        dataset_path = None
                        for line in journal_lines:
                            if line and 'auto_generate_features' in line and 'tool_complete' in line:
                                event = json.loads(line)
                                output = json.loads(event.get('tool_output', '{}'))
                                dataset_path = output.get('output_path')
                                break

                        if not dataset_path:
                            logger.error("Could not find dataset path in journal")
                            return

                        logger.info(f"   Dataset: {dataset_path}")

                        # Training
                        logger.info("🧠 Training ARIMA model with best hyperparameters...")
                        training_result = create_sagemaker_training_job(
                            dataset_s3_path=dataset_path,
                            role_arn=SAGEMAKER_ROLE_ARN,
                            job_name=f"arima-final-{int(time.time())}",
                            hyperparameters=best_params
                        )
                        training_data = json.loads(training_result)

                        if not training_data.get('success'):
                            logger.error(f"Training failed: {training_data.get('error')}")
                            return

                        training_job_name = training_data['job_name']
                        logger.info(f"   Training job: {training_job_name}")

                        # Wait for training
                        for i in range(60):
                            status_result = get_training_job_status(training_job_name)
                            status_data = json.loads(status_result)
                            if status_data.get('status') == 'Completed':
                                logger.info("   Training complete!")
                                break
                            elif status_data.get('status') in ['Failed', 'Stopped']:
                                logger.error(f"Training {status_data.get('status')}")
                                return
                            time.sleep(10)

                        # Deployment
                        logger.info("🚀 Deploying model to endpoint...")
                        deploy_result = deploy_sagemaker_model(
                            training_job_name=training_job_name,
                            endpoint_name=f"arima-endpoint-{workflow_id[:8]}"
                        )
                        deploy_data = json.loads(deploy_result)

                        if not deploy_data.get('success'):
                            logger.error(f"Deployment failed: {deploy_data.get('error')}")
                            return

                        endpoint_name = deploy_data['endpoint_name']
                        logger.info(f"   Endpoint: {endpoint_name}")

                        # Inference
                        logger.info("📈 Generating 7-day forecast...")
                        forecast_result = invoke_sagemaker_endpoint(
                            endpoint_name=endpoint_name,
                            input_data={"forecast_horizon": 7}
                        )
                        forecast_data = json.loads(forecast_result)

                        # Report
                        logger.info("📝 Generating report...")
                        report_result = generate_simple_report(
                            eda_results="{}",
                            training_results=json.dumps({"arima": training_data}),
                            forecast_results=json.dumps({"arima": forecast_data})
                        )

                        logger.info(f"✅ Phase 2 complete for {workflow_id}")

                    except Exception as e:
                        logger.error(f"❌ Phase 2 failed for {workflow_id}: {e}")

                thread = threading.Thread(target=run_phase2, daemon=True)
                thread.start()

                return {"response": json.dumps({
                    "status": "phase2_started",
                    "workflow_id": workflow_id,
                    "tuning_job": tuning_job_name,
                    "message": "Phase 2 (training/deployment/inference) started in background",
                    "timestamp": datetime.now().isoformat()
                })}

        # Check for status request (must have "status" keyword)
        if "status" in prompt.lower():
            workflow_id_match = re.search(r'workflow-[\w-]+', prompt)
            if workflow_id_match:
                workflow_id = workflow_id_match.group(0)
                status_result = get_workflow_status(workflow_id)
                return {"response": status_result}

        # Check if this is a workflow execution request
        workflow_keywords = ["run workflow", "pipeline", "forecast", "s3://"]
        is_workflow_request = any(kw in prompt.lower() for kw in workflow_keywords)

        if is_workflow_request:
            # Extract parameters
            params = extract_params(prompt)

            if not params['dataset_s3_path']:
                return {"response": json.dumps({
                    "status": "error",
                    "error": "Missing S3 path. Please provide S3 dataset path.",
                    "example": "Run workflow on s3://bucket/data.csv",
                    "timestamp": datetime.now().isoformat()
                })}

            if not params['role_arn']:
                return {"response": json.dumps({
                    "status": "error",
                    "error": "SageMaker role not configured. Set SAGEMAKER_ROLE_ARN environment variable in .bedrock_agentcore.yaml",
                    "timestamp": datetime.now().isoformat()
                })}

            # Extract workflow ID from prompt or generate one
            workflow_id_match = re.search(r'workflow-[\w-]+', prompt)
            if workflow_id_match:
                workflow_id = workflow_id_match.group(0)
            else:
                workflow_id = f"workflow-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"

            logger.info(f"🔄 Starting pipeline {workflow_id}:")
            logger.info(f"   Dataset: {params['dataset_s3_path']}")
            logger.info(f"   Role: {params['role_arn']}")

            # Start pipeline asynchronously in background thread
            import threading
            def run_pipeline():
                try:
                    start_time = time.time()
                    results = execute_pipeline(
                        params['dataset_s3_path'],
                        params['role_arn'],
                        workflow_id,
                        time_column=params.get('time_column'),
                        value_column=params.get('value_column')
                    )
                    execution_time = time.time() - start_time
                    logger.info(f"✅ Pipeline {workflow_id} completed in {execution_time:.0f}s")
                    logger.info(f"   Status: {results.get('status')}")
                    logger.info(f"   Message: {results.get('message')}")
                except Exception as e:
                    logger.error(f"❌ Pipeline {workflow_id} failed: {e}")

            thread = threading.Thread(target=run_pipeline, daemon=True)
            thread.start()

            # Return immediately with workflow ID
            return {"response": json.dumps({
                "status": "started",
                "workflow_id": workflow_id,
                "message": "Pipeline started successfully. Use workflow status endpoint to monitor progress.",
                "dataset": params['dataset_s3_path'],
                "timestamp": datetime.now().isoformat()
            })}
        else:
            # Use regular agent for help/info queries
            result = agent(prompt)

            return {"response": json.dumps({
                "status": "success",
                "result": str(result),
                "timestamp": datetime.now().isoformat()
            })}

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"response": json.dumps({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)