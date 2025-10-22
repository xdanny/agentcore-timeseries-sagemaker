#!/bin/bash
# Verification script to ensure setup completed successfully

set -e

echo "🔍 Intelligent Forecasting System - Setup Verification"
echo "======================================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verification functions
verify_check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $1${NC}"
        return 0
    else
        echo -e "${RED}❌ $1${NC}"
        return 1
    fi
}

echo "1️⃣ Checking AWS Configuration..."
aws sts get-caller-identity > /dev/null 2>&1
verify_check "AWS credentials configured"

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=${AWS_REGION:-us-east-1}
BUCKET_NAME="sagemaker-forecasting-${AWS_REGION}-${AWS_ACCOUNT_ID}"

echo -e "   Account: ${YELLOW}${AWS_ACCOUNT_ID}${NC}"
echo -e "   Region: ${YELLOW}${AWS_REGION}${NC}"
echo ""

echo "2️⃣ Checking S3 Bucket..."
aws s3 ls "s3://$BUCKET_NAME" > /dev/null 2>&1
verify_check "S3 bucket exists: $BUCKET_NAME"

# Check folder structure
for folder in uploads datasets features reports sagemaker/models sagemaker/code; do
    aws s3 ls "s3://$BUCKET_NAME/$folder" > /dev/null 2>&1
    verify_check "Folder exists: $folder/"
done
echo ""

echo "3️⃣ Checking SageMaker Scripts..."
aws s3 ls "s3://$BUCKET_NAME/sagemaker/code/inference_working.tar.gz" > /dev/null 2>&1
verify_check "ARIMA inference script uploaded"

aws s3 ls "s3://$BUCKET_NAME/sagemaker/code/training_arima.tar.gz" > /dev/null 2>&1
verify_check "ARIMA training script uploaded"
echo ""

echo "4️⃣ Checking Example Dataset..."
aws s3 ls "s3://$BUCKET_NAME/datasets/example_timeseries.csv" > /dev/null 2>&1
verify_check "Example dataset uploaded"
echo ""

echo "5️⃣ Checking Terraform State..."
if [ -f "terraform/terraform.tfstate" ]; then
    echo -e "${GREEN}✅ Terraform state exists${NC}"

    # Check for key resources
    cd terraform
    terraform show -json > /dev/null 2>&1
    verify_check "Terraform state valid"

    # Check IAM role
    terraform state list | grep "aws_iam_role.sagemaker_execution_role" > /dev/null 2>&1
    verify_check "SageMaker IAM role created"

    # Check Code Interpreter role
    terraform state list | grep "aws_iam_role.code_interpreter_role" > /dev/null 2>&1
    verify_check "Code Interpreter role created"

    cd ..
else
    echo -e "${YELLOW}⚠️  Terraform not applied yet${NC}"
fi
echo ""

echo "6️⃣ Checking Python Environment..."
if command -v uv &> /dev/null; then
    echo -e "${GREEN}✅ UV installed${NC}"
else
    echo -e "${RED}❌ UV not installed${NC}"
fi

if [ -f ".venv/bin/python" ] || [ -f "venv/bin/python" ]; then
    echo -e "${GREEN}✅ Virtual environment exists${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment not created (run: uv sync)${NC}"
fi
echo ""

echo "7️⃣ Checking Project Files..."
required_files=(
    "README.md"
    "agent.py"
    "pyproject.toml"
    ".env.example"
    ".gitignore"
    "LICENSE"
    "CONTRIBUTING.md"
    "scripts/setup.sh"
    "scripts/intelligent_dashboard_v2.py"
    "terraform/main.tf"
    "terraform/variables.tf"
    "agents/dataset_analysis_agent.py"
    "agents/advanced_eda_agent.py"
    "agents/intelligent_feature_engineering_agent.py"
    "agents/comprehensive_report_agent.py"
    "sagemaker_scripts/inference_working.py"
    "sagemaker_scripts/training_arima.py"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅${NC} $file"
    else
        echo -e "${RED}❌${NC} $file"
    fi
done
echo ""

echo "8️⃣ Optional: AgentCore Status..."
if command -v agentcore &> /dev/null; then
    echo -e "${GREEN}✅ AgentCore CLI installed${NC}"

    if agentcore status > /dev/null 2>&1; then
        echo -e "${GREEN}✅ AgentCore deployed${NC}"
        agentcore status
    else
        echo -e "${YELLOW}⚠️  AgentCore not deployed (optional)${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  AgentCore not installed (optional)${NC}"
fi
echo ""

echo "======================================================"
echo "🎉 Verification Complete!"
echo ""
echo "Next Steps:"
echo "1. Run the dashboard:"
echo "   uv run streamlit run scripts/intelligent_dashboard_v2.py"
echo ""
echo "2. Or test the endpoint:"
echo "   python scripts/test_endpoint.py"
echo ""
echo "3. Or run end-to-end tests:"
echo "   python tests/test_end_to_end.py"
echo ""
