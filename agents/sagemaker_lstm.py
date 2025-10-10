import os
#!/usr/bin/env python3
"""
SageMaker LSTM Training - Tools for LSTM model training and tuning
"""
import json
import boto3
from strands import tool

BUCKET = os.environ.get('BUCKET_NAME', 'sagemaker-forecasting-{}-{}'.format(os.environ.get('AWS_REGION', 'us-east-1'), boto3.client('sts').get_caller_identity()['Account']))
REGION = 'us-east-1'
SAGEMAKER = boto3.client('sagemaker', region_name=REGION)
SAGEMAKER_RUNTIME = boto3.client('sagemaker-runtime', region_name=REGION)


@tool
def create_lstm_tuning_job(
    job_name: str,
    dataset_s3_path: str,
    role_arn: str,
    max_jobs: int = 10,
    max_parallel_jobs: int = 2
) -> str:
    """
    Create SageMaker hyperparameter tuning job for LSTM

    Args:
        job_name: Tuning job name
        dataset_s3_path: S3 path to training data
        role_arn: SageMaker execution role ARN
        max_jobs: Maximum tuning jobs to run
        max_parallel_jobs: Parallel jobs

    Returns:
        JSON with tuning job details
    """
    try:
        response = SAGEMAKER.create_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=job_name,
            HyperParameterTuningJobConfig={
                'Strategy': 'Bayesian',
                'HyperParameterTuningJobObjective': {
                    'Type': 'Minimize',
                    'MetricName': 'validation:rmse'
                },
                'ResourceLimits': {
                    'MaxNumberOfTrainingJobs': max_jobs,
                    'MaxParallelTrainingJobs': max_parallel_jobs
                },
                'ParameterRanges': {
                    'IntegerParameterRanges': [
                        {'Name': 'lookback', 'MinValue': '7', 'MaxValue': '30'},
                        {'Name': 'units', 'MinValue': '32', 'MaxValue': '128'},
                        {'Name': 'epochs', 'MinValue': '50', 'MaxValue': '200'},
                        {'Name': 'batch', 'MinValue': '16', 'MaxValue': '64'}
                    ],
                    'ContinuousParameterRanges': [
                        {'Name': 'dropout', 'MinValue': '0.1', 'MaxValue': '0.5'},
                        {'Name': 'lr', 'MinValue': '0.0001', 'MaxValue': '0.01', 'ScalingType': 'Logarithmic'}
                    ]
                }
            },
            TrainingJobDefinition={
                'AlgorithmSpecification': {
                    'TrainingImage': '763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.13-cpu-py310',
                    'TrainingInputMode': 'File',
                    'MetricDefinitions': [
                        {
                            'Name': 'validation:rmse',
                            'Regex': 'RMSE: ([0-9\\.]+)'
                        }
                    ]
                },
                'RoleArn': role_arn,
                'InputDataConfig': [{
                    'ChannelName': 'train',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': dataset_s3_path,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'text/csv'
                }],
                'OutputDataConfig': {
                    'S3OutputPath': f's3://{BUCKET}/sagemaker/lstm-tuning-output/'
                },
                'ResourceConfig': {
                    'InstanceType': 'ml.m5.xlarge',
                    'InstanceCount': 1,
                    'VolumeSizeInGB': 10
                },
                'StoppingCondition': {'MaxRuntimeInSeconds': 7200},
                'StaticHyperParameters': {
                    'sagemaker_program': 'train_lstm.py',
                    'sagemaker_submit_directory': f's3://{BUCKET}/sagemaker/code/lstm_sourcedir.tar.gz'
                }
            }
        )

        return json.dumps({
            'success': True,
            'tuning_job_arn': response['HyperParameterTuningJobArn'],
            'job_name': job_name
        })

    except Exception as e:
        return json.dumps({
            'success': False,
            'error': str(e)
        })


@tool
def get_lstm_tuning_status(job_name: str) -> str:
    """
    Get LSTM tuning job status and best parameters

    Args:
        job_name: Tuning job name

    Returns:
        JSON with status and best hyperparameters
    """
    try:
        response = SAGEMAKER.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=job_name
        )

        result = {
            'success': True,
            'status': response['HyperParameterTuningJobStatus'],
            'training_job_count': response['TrainingJobStatusCounters']['Completed'],
            'best_training_job': response.get('BestTrainingJob', {}).get('TrainingJobName'),
            'objective_metric': response.get('BestTrainingJob', {}).get('FinalHyperParameterTuningJobObjectiveMetric')
        }

        # Get best hyperparameters if available
        if result['best_training_job']:
            training_response = SAGEMAKER.describe_training_job(
                TrainingJobName=result['best_training_job']
            )
            result['best_hyperparameters'] = training_response['HyperParameters']

        return json.dumps(result)

    except Exception as e:
        return json.dumps({
            'success': False,
            'error': str(e)
        })


@tool
def create_lstm_training_job(
    job_name: str,
    dataset_s3_path: str,
    role_arn: str,
    lookback: int = 14,
    units: int = 50,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 32
) -> str:
    """
    Create SageMaker LSTM training job

    Args:
        job_name: Training job name
        dataset_s3_path: S3 path to training data
        role_arn: SageMaker execution role ARN
        lookback: Sequence length for LSTM
        units: Number of LSTM units
        dropout: Dropout rate
        learning_rate: Learning rate
        epochs: Training epochs
        batch_size: Batch size

    Returns:
        JSON with job status
    """
    try:
        response = SAGEMAKER.create_training_job(
            TrainingJobName=job_name,
            AlgorithmSpecification={
                'TrainingImage': '763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.13-cpu-py310',
                'TrainingInputMode': 'File',
                'EnableSageMakerMetricsTimeSeries': False
            },
            RoleArn=role_arn,
            InputDataConfig=[{
                'ChannelName': 'train',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': dataset_s3_path,
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'ContentType': 'text/csv'
            }],
            OutputDataConfig={
                'S3OutputPath': f's3://{BUCKET}/sagemaker/lstm-models/'
            },
            ResourceConfig={
                'InstanceType': 'ml.m5.xlarge',
                'InstanceCount': 1,
                'VolumeSizeInGB': 10
            },
            StoppingCondition={'MaxRuntimeInSeconds': 7200},
            HyperParameters={
                'lookback': str(lookback),
                'units': str(units),
                'dropout': str(dropout),
                'lr': str(learning_rate),
                'epochs': str(epochs),
                'batch': str(batch_size),
                'sagemaker_program': 'train_lstm.py',
                'sagemaker_submit_directory': f's3://{BUCKET}/sagemaker/code/lstm_sourcedir.tar.gz'
            }
        )

        return json.dumps({
            'success': True,
            'job_arn': response['TrainingJobArn'],
            'job_name': job_name
        })

    except Exception as e:
        return json.dumps({
            'success': False,
            'error': str(e)
        })


@tool
def get_lstm_training_status(job_name: str) -> str:
    """
    Get LSTM training job status

    Args:
        job_name: Training job name

    Returns:
        JSON with status and details
    """
    try:
        response = SAGEMAKER.describe_training_job(TrainingJobName=job_name)

        return json.dumps({
            'success': True,
            'status': response['TrainingJobStatus'],
            'secondary_status': response.get('SecondaryStatus', ''),
            'model_artifacts': response.get('ModelArtifacts', {}).get('S3ModelArtifacts'),
            'training_time': response.get('TrainingTimeInSeconds', 0)
        })

    except Exception as e:
        return json.dumps({
            'success': False,
            'error': str(e)
        })


@tool
def deploy_lstm_model(
    model_name: str,
    endpoint_name: str,
    model_data_url: str,
    role_arn: str
) -> str:
    """
    Deploy LSTM model to SageMaker endpoint

    Args:
        model_name: Model name
        endpoint_name: Endpoint name
        model_data_url: S3 URL to model.tar.gz
        role_arn: SageMaker execution role ARN

    Returns:
        JSON with deployment status
    """
    try:
        # Create model
        SAGEMAKER.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': '763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.13-cpu',
                'ModelDataUrl': model_data_url,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference_lstm.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': f's3://{BUCKET}/sagemaker/code/lstm_inference.tar.gz'
                }
            },
            ExecutionRoleArn=role_arn
        )

        # Create endpoint config
        config_name = f'{model_name}-config'
        SAGEMAKER.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[{
                'VariantName': 'variant-1',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.m5.xlarge'
            }]
        )

        # Create endpoint
        response = SAGEMAKER.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )

        return json.dumps({
            'success': True,
            'endpoint_arn': response['EndpointArn'],
            'endpoint_name': endpoint_name
        })

    except Exception as e:
        return json.dumps({
            'success': False,
            'error': str(e)
        })


@tool
def invoke_lstm_endpoint(endpoint_name: str, historical_data: list, steps: int = 7) -> str:
    """
    Invoke LSTM endpoint for predictions

    Args:
        endpoint_name: Endpoint name
        historical_data: List of historical values
        steps: Number of forecast steps

    Returns:
        JSON with predictions
    """
    try:
        payload = {
            'historical_data': historical_data,
            'steps': steps
        }

        response = SAGEMAKER_RUNTIME.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )

        predictions = response['Body'].read().decode('utf-8')

        return json.dumps({
            'success': True,
            'predictions': json.loads(predictions)
        })

    except Exception as e:
        return json.dumps({
            'success': False,
            'error': str(e)
        })
