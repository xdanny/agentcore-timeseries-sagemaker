# Code Interpreter IAM Role with S3 Access
resource "aws_iam_role" "code_interpreter_role" {
  name = "BedrockAgentCore-CodeInterpreter-Role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "bedrock-agentcore.amazonaws.com"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
          ArnLike = {
            "aws:SourceArn" = "arn:aws:bedrock-agentcore:us-east-1:${data.aws_caller_identity.current.account_id}:*"
          }
        }
      }
    ]
  })

  tags = {
    Name = "BedrockAgentCore-CodeInterpreter-Role"
  }
}

# S3 Access Policy for Code Interpreter
resource "aws_iam_role_policy" "code_interpreter_s3_access" {
  name = "CodeInterpreterS3Access"
  role = aws_iam_role.code_interpreter_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::bedrock-agentcore-data-${var.aws_region}",
          "arn:aws:s3:::bedrock-agentcore-data-${var.aws_region}/*",
          "arn:aws:s3:::${var.bucket_name}",
          "arn:aws:s3:::${var.bucket_name}/*"
        ]
      }
    ]
  })
}

# Bedrock Code Interpreter Permissions
resource "aws_iam_role_policy" "code_interpreter_bedrock_access" {
  name = "CodeInterpreterBedrockAccess"
  role = aws_iam_role.code_interpreter_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = "arn:aws:bedrock:us-east-1::foundation-model/*"
      }
    ]
  })
}

# ECR Access for AgentCore Runtime
resource "aws_iam_role_policy" "code_interpreter_ecr_access" {
  name = "CodeInterpreterECRAccess"
  role = aws_iam_role.code_interpreter_role.id

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

# SageMaker Access for AgentCore Runtime
resource "aws_iam_role_policy" "code_interpreter_sagemaker_access" {
  name = "CodeInterpreterSageMakerAccess"
  role = aws_iam_role.code_interpreter_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:InvokeEndpoint",
          "sagemaker:DescribeEndpoint",
          "sagemaker:ListEndpoints"
        ]
        Resource = "*"
      }
    ]
  })
}

output "code_interpreter_role_arn" {
  description = "ARN of the Code Interpreter IAM role"
  value       = aws_iam_role.code_interpreter_role.arn
}