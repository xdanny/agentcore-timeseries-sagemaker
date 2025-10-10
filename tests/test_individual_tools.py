#!/usr/bin/env python3
"""
Test individual agent tools independently
"""
import os
import boto3
import json
import time

# Configuration - Auto-detect from environment or AWS account
REGION = os.environ.get('AWS_REGION', 'us-east-1')
ACCOUNT_ID = boto3.client('sts').get_caller_identity()['Account']
BUCKET = os.environ.get('BUCKET_NAME', f'sagemaker-forecasting-{REGION}-{ACCOUNT_ID}')

# Note: AGENTCORE_ENDPOINT_ARN must be set after deployment
AGENTCORE_ENDPOINT_ARN = os.environ.get(
    'AGENTCORE_ENDPOINT_ARN',
    f"arn:aws:bedrock-agentcore:{REGION}:{ACCOUNT_ID}:runtime/agent-PLACEHOLDER"
)
AGENT_ID = AGENTCORE_ENDPOINT_ARN.split('/')[-1]
TEST_DATASET = f"s3://{BUCKET}/training-data/train.csv"

agentcore = boto3.client('bedrock-agentcore-runtime', region_name='us-east-1')

def invoke_agent(prompt: str, test_name: str) -> dict:
    """Invoke agent with a prompt"""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}\n")

    try:
        response = agentcore.invoke_agent(
            agentId=AGENT_ID,
            sessionId=f"test-{int(time.time())}",
            inputText=prompt
        )

        result_text = ""
        for event in response.get('completion', []):
            if 'chunk' in event:
                result_text += event['chunk'].get('bytes', b'').decode('utf-8')

        print(f"✅ SUCCESS\n")
        print(f"Response: {result_text[:500]}...")

        # Try to parse as JSON
        try:
            result_json = json.loads(result_text)
            return {"status": "success", "data": result_json, "raw": result_text}
        except:
            return {"status": "success", "data": None, "raw": result_text}

    except Exception as e:
        print(f"❌ FAILED: {e}\n")
        return {"status": "error", "error": str(e)}


def test_1_advanced_eda():
    """Test: Advanced EDA Agent"""
    prompt = f"run advanced eda on {TEST_DATASET}"
    result = invoke_agent(prompt, "Advanced EDA")

    if result["status"] == "success" and result["data"]:
        eda = result["data"]
        print(f"📊 Stationarity: {eda.get('stationarity', {}).get('recommended_d')}")
        print(f"📊 Seasonal Strength: {eda.get('decomposition', {}).get('seasonal_strength')}")
        print(f"📊 Report URL: {eda.get('report_url')}")

    return result


def test_2_feature_recommendations():
    """Test: Feature Recommendation Agent"""
    # First get EDA results
    eda_result = test_1_advanced_eda()

    if eda_result["status"] != "success":
        print("❌ Skipping feature recommendations test - EDA failed")
        return

    prompt = f"recommend features for {TEST_DATASET} with eda results {json.dumps(eda_result['data'])}"
    result = invoke_agent(prompt, "Feature Recommendations")

    if result["status"] == "success" and result["data"]:
        recs = result["data"]
        print(f"💡 Total Features: {recs.get('summary', {}).get('total_recommended_features')}")
        print(f"💡 High Priority: {recs.get('summary', {}).get('high_priority_categories')}")

        if 'lag_features' in recs:
            print(f"💡 Lag Features: {recs['lag_features']['lags']}")
            print(f"   Justification: {recs['lag_features']['justification']}")

    return result


def test_3_feature_creation():
    """Test: Feature Creation Agent"""
    # Get recommendations first
    rec_result = test_2_feature_recommendations()

    if rec_result["status"] != "success":
        print("❌ Skipping feature creation test - recommendations failed")
        return

    # Create simple config
    feature_config = {
        "lag_features": {"enabled": True, "lags": [1, 7]},
        "rolling_features": {"enabled": True, "windows": [7], "statistics": ["mean"]},
        "calendar_features": {"enabled": True, "features": ["day_of_week", "month"]}
    }

    prompt = f"create features for {TEST_DATASET} with config {json.dumps(feature_config)}"
    result = invoke_agent(prompt, "Feature Creation")

    if result["status"] == "success" and result["data"]:
        feat = result["data"]
        print(f"✨ Features Created: {feat.get('feature_count')}")
        print(f"✨ Output Path: {feat.get('output_path')}")
        print(f"✨ Cleaned Rows: {feat.get('cleaned_rows')}")

    return result


def test_4_hyperparameter_tuning():
    """Test: SageMaker Hyperparameter Tuning"""
    # Need featured dataset
    feat_result = test_3_feature_creation()

    if feat_result["status"] != "success" or not feat_result.get("data"):
        print("❌ Skipping tuning test - feature creation failed")
        return

    featured_path = feat_result["data"].get("output_path")

    prompt = f"create sagemaker tuning job for {featured_path} with max_jobs 3 and parallel_jobs 1"
    result = invoke_agent(prompt, "Hyperparameter Tuning")

    if result["status"] == "success" and result["data"]:
        tuning = result["data"]
        print(f"🎯 Tuning Job: {tuning.get('job_name')}")
        print(f"🎯 Status: {tuning.get('status')}")
        print(f"🎯 Max Jobs: {tuning.get('max_jobs')}")

    return result


def test_5_tuning_status():
    """Test: Get Tuning Job Status"""
    # Use existing tuning job
    tuning_job = "arima-tuning-1759773637"  # From your running job

    prompt = f"get status of tuning job {tuning_job}"
    result = invoke_agent(prompt, "Tuning Job Status")

    if result["status"] == "success" and result["data"]:
        status = result["data"]
        print(f"📊 Status: {status.get('status')}")
        print(f"📊 Completed Jobs: {status.get('completed_jobs')}")
        print(f"📊 Best Hyperparameters: {status.get('best_hyperparameters')}")

    return result


def test_6_comprehensive_report():
    """Test: Comprehensive Report Generation"""
    # Need all previous results
    print("⏭️  Skipping comprehensive report test - requires full workflow")
    print("   This should be tested after tuning completes and training is done")
    return {"status": "skipped"}


# Run all tests
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║        INTELLIGENT FORECASTING AGENT TESTS               ║
║                                                          ║
║  Testing individual tools independently via AgentCore   ║
╚══════════════════════════════════════════════════════════╝
""")

    tests = [
        ("1️⃣  Advanced EDA", test_1_advanced_eda),
        ("2️⃣  Feature Recommendations", test_2_feature_recommendations),
        ("3️⃣  Feature Creation", test_3_feature_creation),
        ("4️⃣  Hyperparameter Tuning", test_4_hyperparameter_tuning),
        ("5️⃣  Tuning Status", test_5_tuning_status),
        ("6️⃣  Comprehensive Report", test_6_comprehensive_report)
    ]

    results = {}
    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = result
        except Exception as e:
            print(f"❌ Test {name} crashed: {e}")
            results[name] = {"status": "crashed", "error": str(e)}

        time.sleep(2)  # Pause between tests

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    for name, result in results.items():
        status_emoji = "✅" if result.get("status") == "success" else "❌" if result.get("status") == "error" else "⏭️"
        print(f"{status_emoji} {name}: {result.get('status', 'unknown').upper()}")

    print(f"\n{'='*60}\n")
