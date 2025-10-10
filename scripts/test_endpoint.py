#!/usr/bin/env python3
"""Test the updated SageMaker endpoint"""
import boto3
import json

REGION = 'us-east-1'
ENDPOINT_NAME = 'arima-endpoint-1760093864'

sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=REGION)

print(f"🧪 Testing endpoint: {ENDPOINT_NAME}\n")

# Test with different forecast horizons
test_cases = [
    {"steps": 7, "description": "1 week forecast"},
    {"steps": 14, "description": "2 week forecast"},
    {"steps": 30, "description": "1 month forecast"}
]

for test in test_cases:
    print(f"Test: {test['description']} ({test['steps']} steps)")
    print(f"Request: {json.dumps(test)}")

    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=json.dumps({"steps": test["steps"]})
        )

        result = json.loads(response['Body'].read().decode('utf-8'))

        if 'error' in result:
            print(f"❌ Error: {result['error']}\n")
        elif 'forecast' in result:
            forecast = result['forecast']
            print(f"✅ Success!")
            print(f"   Forecast length: {len(forecast)}")
            print(f"   First 3 values: {forecast[:3]}")
            print(f"   Has confidence intervals: {('lower_bound' in result) and ('upper_bound' in result)}")
            print()
        else:
            print(f"⚠️ Unexpected response format: {result}\n")

    except Exception as e:
        print(f"❌ Exception: {type(e).__name__}: {str(e)}\n")

print("✅ All tests completed!")
