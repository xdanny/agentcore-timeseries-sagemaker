import os
#!/usr/bin/env python3
"""
SageMaker Hyperparameter Tuning - Use SageMaker for hyperparameter optimization
"""
import json
import boto3
import time
from strands import tool

BUCKET = os.environ.get('BUCKET_NAME', 'sagemaker-forecasting-{}-{}'.format(os.environ.get('AWS_REGION', 'us-east-1'), boto3.client('sts').get_caller_identity()['Account']))
REGION = 'us-east-1'
SAGEMAKER = boto3.client('sagemaker', region_name=REGION)


@tool
def create_sagemaker_tuning_job(
    job_name: str,
    dataset_s3_path: str,
    role_arn: str,
    max_jobs: int = 10,
    max_parallel_jobs: int = 2
) -> str:
    """
    Create SageMaker hyperparameter tuning job for ARIMA

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
                        {'Name': 'p', 'MinValue': '0', 'MaxValue': '3'},
                        {'Name': 'd', 'MinValue': '0', 'MaxValue': '2'},
                        {'Name': 'q', 'MinValue': '0', 'MaxValue': '3'},
                        {'Name': 'seasonal-p', 'MinValue': '0', 'MaxValue': '2'},
                        {'Name': 'seasonal-d', 'MinValue': '0', 'MaxValue': '1'},
                        {'Name': 'seasonal-q', 'MinValue': '0', 'MaxValue': '2'}
                    ]
                }
            },
            TrainingJobDefinition={
                'AlgorithmSpecification': {
                    'TrainingImage': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3',
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
                    'S3OutputPath': f's3://{BUCKET}/sagemaker/tuning-output/'
                },
                'ResourceConfig': {
                    'InstanceType': 'ml.m5.xlarge',
                    'InstanceCount': 1,
                    'VolumeSizeInGB': 10
                },
                'StoppingCondition': {'MaxRuntimeInSeconds': 3600},
                'StaticHyperParameters': {
                    'sagemaker_program': 'train_arima.py',
                    'sagemaker_submit_directory': f's3://{BUCKET}/sagemaker/code/sourcedir.tar.gz',
                    'm': '7'
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
def get_tuning_job_status(job_name: str) -> str:
    """
    Get SageMaker tuning job status and best parameters

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
