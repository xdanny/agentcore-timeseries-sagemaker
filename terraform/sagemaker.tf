# SageMaker IAM Role for Model Training
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "SageMaker-ForecastingPipeline-ExecutionRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      },
      {
        Effect = "Allow"
        Principal = {
          Service = "bedrock-agentcore.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Project = "forecasting-pipeline"
    ManagedBy = "terraform"
  }
}

# Policy for S3 access to timeseries data lake
resource "aws_iam_role_policy" "sagemaker_s3_access" {
  name = "SageMaker-S3-DataAccess"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.bucket_name}",
          "arn:aws:s3:::${var.bucket_name}/*"
        ]
      }
    ]
  })
}

# Policy for CloudWatch Logs
resource "aws_iam_role_policy" "sagemaker_cloudwatch_logs" {
  name = "SageMaker-CloudWatch-Logs"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:us-east-1:*:log-group:/aws/sagemaker/*"
      }
    ]
  })
}

# Policy for ECR access (for custom training containers)
resource "aws_iam_role_policy" "sagemaker_ecr_access" {
  name = "SageMaker-ECR-Access"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })
}

# Policy for Bedrock Code Interpreter access
resource "aws_iam_role_policy" "sagemaker_code_interpreter" {
  name = "SageMaker-CodeInterpreter-Access"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock-agentcore:StartCodeInterpreterSession",
          "bedrock-agentcore:InvokeCodeInterpreter",
          "bedrock-agentcore:StopCodeInterpreterSession"
        ]
        Resource = "*"
      }
    ]
  })
}

# Allow AgentCore execution role to assume SageMaker role
resource "aws_iam_role_policy" "agentcore_assume_sagemaker" {
  name = "AgentCore-AssumeSageMaker"
  role = "AmazonBedrockAgentCoreSDKRuntime-us-east-1-d4f0bc5a29"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "sts:AssumeRole"
        Resource = aws_iam_role.sagemaker_execution_role.arn
      }
    ]
  })
}

# Allow SageMaker execution role to assume Code Interpreter role
resource "aws_iam_role_policy" "sagemaker_assume_code_interpreter" {
  name = "SageMaker-AssumeCodeInterpreter"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      Resource = "arn:aws:iam::${var.aws_account_id}:role/BedrockAgentCore-CodeInterpreter-Role"
    }]
  })
}

# Allow AgentCore to invoke SageMaker training jobs
resource "aws_iam_role_policy" "agentcore_sagemaker_training" {
  name = "AgentCore-SageMaker-Training"
  role = "AmazonBedrockAgentCoreSDKRuntime-us-east-1-d4f0bc5a29"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:StopTrainingJob",
          "sagemaker:ListTrainingJobs",
          "sagemaker:CreateHyperParameterTuningJob",
          "sagemaker:DescribeHyperParameterTuningJob",
          "sagemaker:StopHyperParameterTuningJob",
          "sagemaker:ListHyperParameterTuningJobs",
          "sagemaker:AddTags"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateModel",
          "sagemaker:DescribeModel",
          "sagemaker:DeleteModel"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.bucket_name}",
          "arn:aws:s3:::${var.bucket_name}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = "iam:PassRole"
        Resource = aws_iam_role.sagemaker_execution_role.arn
        Condition = {
          StringEquals = {
            "iam:PassedToService" = "sagemaker.amazonaws.com"
          }
        }
      }
    ]
  })
}

# Output SageMaker execution role ARN
output "sagemaker_execution_role_arn" {
  description = "SageMaker execution role ARN for training jobs"
  value       = aws_iam_role.sagemaker_execution_role.arn
}
