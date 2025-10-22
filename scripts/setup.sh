#!/bin/bash
# Complete setup script for Intelligent Forecasting System
# This script automates all manual steps for reproducibility

set -e  # Exit on error

echo "🚀 Intelligent Forecasting System - Setup Script"
echo "================================================"
echo ""

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install: https://aws.amazon.com/cli/"
    exit 1
fi
echo "✅ AWS CLI installed"

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Run: aws configure"
    exit 1
fi
echo "✅ AWS credentials configured"

# Check Terraform
if ! command -v terraform &> /dev/null; then
    echo "❌ Terraform not found. Please install: https://www.terraform.io/downloads"
    exit 1
fi
echo "✅ Terraform installed"

# Check Python/UV
if ! command -v uv &> /dev/null; then
    echo "❌ UV not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
echo "✅ UV package manager ready"

echo ""
echo "================================================"
echo "🔧 Configuration"
echo "================================================"
echo ""

# Get AWS account ID and region
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=${AWS_REGION:-us-east-1}

echo "📍 AWS Account ID: $AWS_ACCOUNT_ID"
echo "🌍 AWS Region: $AWS_REGION"
echo ""

# Set bucket name (unique per account)
BUCKET_NAME="sagemaker-forecasting-${AWS_REGION}-${AWS_ACCOUNT_ID}"

read -p "📦 Use bucket name: $BUCKET_NAME? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter custom bucket name: " BUCKET_NAME
fi

echo ""
echo "================================================"
echo "🪣 Step 1: Create S3 Bucket"
echo "================================================"
echo ""

# Create S3 bucket
if aws s3 ls "s3://$BUCKET_NAME" 2>&1 | grep -q 'NoSuchBucket'; then
    echo "Creating S3 bucket: $BUCKET_NAME"

    if [ "$AWS_REGION" == "us-east-1" ]; then
        aws s3 mb "s3://$BUCKET_NAME" --region "$AWS_REGION"
    else
        aws s3 mb "s3://$BUCKET_NAME" --region "$AWS_REGION" --create-bucket-configuration LocationConstraint="$AWS_REGION"
    fi

    echo "✅ Bucket created"
else
    echo "✅ Bucket already exists: $BUCKET_NAME"
fi

# Create bucket structure
echo "Creating bucket folder structure..."
aws s3api put-object --bucket "$BUCKET_NAME" --key uploads/ --region "$AWS_REGION" || true
aws s3api put-object --bucket "$BUCKET_NAME" --key datasets/ --region "$AWS_REGION" || true
aws s3api put-object --bucket "$BUCKET_NAME" --key features/ --region "$AWS_REGION" || true
aws s3api put-object --bucket "$BUCKET_NAME" --key reports/ --region "$AWS_REGION" || true
aws s3api put-object --bucket "$BUCKET_NAME" --key sagemaker/models/ --region "$AWS_REGION" || true
aws s3api put-object --bucket "$BUCKET_NAME" --key sagemaker/code/ --region "$AWS_REGION" || true
echo "✅ Folder structure created"

echo ""
echo "================================================"
echo "📦 Step 2: Upload SageMaker Scripts"
echo "================================================"
echo ""

# Package and upload inference scripts
cd sagemaker_scripts

echo "Packaging ARIMA inference script..."
tar -czf inference_working.tar.gz inference_working.py requirements.txt
aws s3 cp inference_working.tar.gz "s3://$BUCKET_NAME/sagemaker/code/" --region "$AWS_REGION"
echo "✅ ARIMA inference uploaded"

echo "Packaging ARIMA training script..."
tar -czf training_arima.tar.gz training_arima.py
aws s3 cp training_arima.tar.gz "s3://$BUCKET_NAME/sagemaker/code/" --region "$AWS_REGION"
echo "✅ ARIMA training uploaded"

# LSTM scripts (optional)
if [ -f "lstm_inference.py" ]; then
    echo "Packaging LSTM scripts..."
    tar -czf lstm_inference.tar.gz lstm_inference.py
    tar -czf lstm_training.tar.gz training_lstm.py
    aws s3 cp lstm_inference.tar.gz "s3://$BUCKET_NAME/sagemaker/code/" --region "$AWS_REGION"
    aws s3 cp lstm_training.tar.gz "s3://$BUCKET_NAME/sagemaker/code/" --region "$AWS_REGION"
    echo "✅ LSTM scripts uploaded"
fi

cd ..

echo ""
echo "================================================"
echo "📊 Step 3: Upload Example Dataset"
echo "================================================"
echo ""

# Create example dataset if it doesn't exist
if [ ! -f "data/example_timeseries.csv" ]; then
    mkdir -p data
    echo "Creating example time series dataset..."
    python3 << 'PYTHON'
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate synthetic time series data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
trend = np.linspace(100, 200, len(dates))
seasonality = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
noise = np.random.normal(0, 5, len(dates))
sales = trend + seasonality + noise

df = pd.DataFrame({
    'date': dates,
    'sales': sales,
    'day_of_week': dates.dayofweek,
    'month': dates.month
})

df.to_csv('data/example_timeseries.csv', index=False)
print(f"✅ Created example dataset with {len(df)} records")
PYTHON
fi

# Upload example dataset
aws s3 cp data/example_timeseries.csv "s3://$BUCKET_NAME/datasets/example_timeseries.csv" --region "$AWS_REGION"
echo "✅ Example dataset uploaded"

echo ""
echo "================================================"
echo "🏗️ Step 4: Deploy Infrastructure with Terraform"
echo "================================================"
echo ""

cd terraform

# Create terraform.tfvars
cat > terraform.tfvars <<EOF
aws_region = "$AWS_REGION"
aws_account_id = "$AWS_ACCOUNT_ID"
bucket_name = "$BUCKET_NAME"
EOF

echo "✅ Created terraform.tfvars"

# Initialize Terraform
echo "Initializing Terraform..."
terraform init

# Plan
echo ""
echo "📋 Terraform Plan:"
terraform plan -out=tfplan

# Apply
echo ""
read -p "🚀 Apply Terraform configuration? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    terraform apply tfplan
    echo "✅ Infrastructure deployed"
else
    echo "⏭️ Skipping Terraform apply"
fi

cd ..

echo ""
echo "================================================"
echo "🐍 Step 5: Install Python Dependencies"
echo "================================================"
echo ""

uv sync
echo "✅ Python dependencies installed"

echo ""
echo "================================================"
echo "🤖 Step 6: Configure AgentCore (Optional)"
echo "================================================"
echo ""

read -p "🚀 Deploy to AgentCore? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Configuring AgentCore..."
    uv run agentcore configure --entrypoint agent.py

    echo ""
    read -p "Launch AgentCore now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        uv run agentcore launch
        echo "✅ AgentCore deployed"
    fi
else
    echo "⏭️ Skipping AgentCore deployment"
fi

echo ""
echo "================================================"
echo "✅ Setup Complete!"
echo "================================================"
echo ""
echo "📦 S3 Bucket: $BUCKET_NAME"
echo "🌍 Region: $AWS_REGION"
echo "📊 Example dataset: s3://$BUCKET_NAME/datasets/example_timeseries.csv"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "1. Run the dashboard:"
echo "   uv run streamlit run scripts/intelligent_dashboard_v2.py"
echo ""
echo "2. Or interact via AgentCore CLI:"
echo "   agentcore status"
echo ""
echo "3. Or run end-to-end tests:"
echo "   python tests/test_end_to_end.py"
echo ""
echo "📚 Documentation: See README.md for detailed usage"
echo ""
