#!/usr/bin/env python3
"""Monitor SageMaker endpoint status"""
import boto3
import time
import sys

sagemaker = boto3.client('sagemaker', region_name='us-east-1')
ENDPOINT_NAME = 'arima-endpoint-1760093864'

print(f"📊 Monitoring endpoint: {ENDPOINT_NAME}\n")

last_status = None
while True:
    try:
        response = sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = response['EndpointStatus']
        config = response.get('EndpointConfigName', 'N/A')

        if status != last_status:
            print(f"Status: {status}")
            print(f"Config: {config}")
            print(f"Time: {response.get('LastModifiedTime', 'N/A')}")

            if status == 'InService':
                print(f"\n✅ Endpoint is InService!")
                # Get model details
                try:
                    config_desc = sagemaker.describe_endpoint_config(EndpointConfigName=config)
                    model_name = config_desc['ProductionVariants'][0]['ModelName']
                    print(f"Model: {model_name}")

                    model_desc = sagemaker.describe_model(ModelName=model_name)
                    if 'Environment' in model_desc['PrimaryContainer']:
                        env = model_desc['PrimaryContainer']['Environment']
                        print(f"Inference script: {env.get('SAGEMAKER_PROGRAM', 'N/A')}")
                        print(f"Submit dir: {env.get('SAGEMAKER_SUBMIT_DIRECTORY', 'N/A')}")
                except Exception as e:
                    print(f"Could not get model details: {e}")

                sys.exit(0)

            elif status == 'Failed':
                print(f"\n❌ Endpoint update failed!")
                if 'FailureReason' in response:
                    print(f"Reason: {response['FailureReason']}")
                sys.exit(1)

            print()
            last_status = status

        time.sleep(10)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(10)
