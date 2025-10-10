import os
#!/usr/bin/env python3
"""
Update SageMaker endpoint with new inference script
"""
import boto3
import time
from datetime import datetime

REGION = os.environ.get('AWS_REGION', 'us-east-1')
BUCKET = os.environ.get('BUCKET_NAME', f'sagemaker-forecasting-{REGION}-{boto3.client("sts").get_caller_identity()["Account"]}')
ENDPOINT_NAME = 'arima-endpoint-1760093864'
ROLE_ARN = os.environ.get('SAGEMAKER_ROLE_ARN', f'arn:aws:iam::{boto3.client("sts").get_caller_identity()["Account"]}:role/SageMaker-ForecastingPipeline-ExecutionRole')

# Initialize clients
sagemaker = boto3.client('sagemaker', region_name=REGION)

def update_endpoint():
    """Update endpoint with new inference script"""

    print(f"🔍 Checking current endpoint: {ENDPOINT_NAME}")

    # Get current endpoint config
    try:
        endpoint_desc = sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
        current_config = endpoint_desc['EndpointConfigName']
        print(f"✓ Current config: {current_config}")

        # Get model details from current config
        config_desc = sagemaker.describe_endpoint_config(EndpointConfigName=current_config)
        model_name = config_desc['ProductionVariants'][0]['ModelName']
        instance_type = config_desc['ProductionVariants'][0]['InstanceType']

        print(f"✓ Current model: {model_name}")
        print(f"✓ Instance type: {instance_type}")

        # Get model details
        model_desc = sagemaker.describe_model(ModelName=model_name)
        model_data_url = model_desc['PrimaryContainer']['ModelDataUrl']
        image_uri = model_desc['PrimaryContainer']['Image']

        print(f"✓ Model data: {model_data_url}")

    except Exception as e:
        print(f"❌ Error getting endpoint info: {e}")
        return

    # Create new model with updated inference script
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    new_model_name = f'arima-model-{timestamp}'
    new_config_name = f'arima-config-{timestamp}'

    print(f"\n📦 Creating new model: {new_model_name}")

    try:
        sagemaker.create_model(
            ModelName=new_model_name,
            PrimaryContainer={
                'Image': image_uri,
                'ModelDataUrl': model_data_url,
                'Mode': 'SingleModel',
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference_working.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': f's3://{BUCKET}/sagemaker/code/inference_working.tar.gz',
                    'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                    'SAGEMAKER_REGION': REGION
                }
            },
            ExecutionRoleArn=ROLE_ARN
        )
        print(f"✓ Model created: {new_model_name}")

    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return

    # Create new endpoint config
    print(f"\n⚙️ Creating new endpoint config: {new_config_name}")

    try:
        sagemaker.create_endpoint_config(
            EndpointConfigName=new_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': new_model_name,
                    'InstanceType': instance_type,
                    'InitialInstanceCount': 1,
                    'InitialVariantWeight': 1.0
                }
            ]
        )
        print(f"✓ Endpoint config created: {new_config_name}")

    except Exception as e:
        print(f"❌ Error creating endpoint config: {e}")
        return

    # Update endpoint
    print(f"\n🔄 Updating endpoint: {ENDPOINT_NAME}")

    try:
        sagemaker.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=new_config_name
        )
        print(f"✓ Update initiated")

    except Exception as e:
        print(f"❌ Error updating endpoint: {e}")
        return

    # Wait for update to complete
    print("\n⏳ Waiting for endpoint update to complete...")
    print("This will take 5-10 minutes...")

    while True:
        try:
            status = sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
            current_status = status['EndpointStatus']

            print(f"   Status: {current_status}", end='\r')

            if current_status == 'InService':
                print(f"\n✅ Endpoint updated successfully!")
                print(f"\nEndpoint: {ENDPOINT_NAME}")
                print(f"Config: {new_config_name}")
                print(f"Model: {new_model_name}")
                print(f"Inference script: inference_working.py")
                break

            elif current_status == 'Failed':
                print(f"\n❌ Endpoint update failed")
                if 'FailureReason' in status:
                    print(f"Reason: {status['FailureReason']}")
                break

            time.sleep(30)

        except Exception as e:
            print(f"\n❌ Error checking status: {e}")
            break

if __name__ == '__main__':
    update_endpoint()
