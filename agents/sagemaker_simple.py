import os
#!/usr/bin/env python3
"""
Simple SageMaker Integration - Direct boto3 calls (agent has credentials)
No Code Interpreter needed for SageMaker API operations
"""
import json
import boto3
import time
from strands import tool

BUCKET = os.environ.get('BUCKET_NAME', 'sagemaker-forecasting-{}-{}'.format(os.environ.get('AWS_REGION', 'us-east-1'), boto3.client('sts').get_caller_identity()['Account']))
REGION = 'us-east-1'

# Agent environment HAS boto3 and credentials
SAGEMAKER = boto3.client('sagemaker', region_name=REGION)
SAGEMAKER_RUNTIME = boto3.client('sagemaker-runtime', region_name=REGION)
S3 = boto3.client('s3', region_name=REGION)


@tool
def get_training_job_status(job_name: str) -> str:
    """
    Get SageMaker training job status

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
def create_sagemaker_training_job(
    job_name: str,
    dataset_s3_path: str,
    role_arn: str
) -> str:
    """
    Create SageMaker training job - Direct boto3 API call

    Args:
        job_name: Training job name
        dataset_s3_path: S3 path to training data
        role_arn: SageMaker execution role ARN

    Returns:
        JSON with job status
    """
    try:
        response = SAGEMAKER.create_training_job(
            TrainingJobName=job_name,
            AlgorithmSpecification={
                'TrainingImage': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3',
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
                'S3OutputPath': f's3://{BUCKET}/sagemaker/models/'
            },
            ResourceConfig={
                'InstanceType': 'ml.m5.xlarge',
                'InstanceCount': 1,
                'VolumeSizeInGB': 10
            },
            StoppingCondition={'MaxRuntimeInSeconds': 3600},
            HyperParameters={
                'p': '2',
                'd': '1',
                'q': '3',
                'seasonal-p': '1',
                'seasonal-d': '1',
                'seasonal-q': '1',
                'm': '7',
                'sagemaker_program': 'train_arima.py',
                'sagemaker_submit_directory': f's3://{BUCKET}/sagemaker/code/sourcedir.tar.gz'
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
def get_training_job_status(job_name: str) -> str:
    """
    Get SageMaker training job status

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
def deploy_sagemaker_model(
    model_name: str,
    endpoint_name: str,
    model_data_url: str,
    role_arn: str
) -> str:
    """
    Deploy model to SageMaker endpoint

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
                'Image': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3',
                'ModelDataUrl': model_data_url,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'fixed_inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': f's3://{BUCKET}/sagemaker/code/inference_working.tar.gz'
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
def invoke_sagemaker_endpoint(endpoint_name: str, input_data: str) -> str:
    """
    Invoke SageMaker endpoint for predictions

    Args:
        endpoint_name: Endpoint name
        input_data: CSV input data

    Returns:
        JSON with predictions
    """
    try:
        response = SAGEMAKER_RUNTIME.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=input_data if isinstance(input_data, (bytes, str)) else json.dumps(input_data)
        )

        predictions = response['Body'].read().decode('utf-8')

        return json.dumps({
            'success': True,
            'predictions': predictions
        })

    except Exception as e:
        return json.dumps({
            'success': False,
            'error': str(e)
        })
