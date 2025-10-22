terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name
}

# Create new S3 bucket for us-east-1 region
resource "aws_s3_bucket" "reports_bucket" {
  bucket = "agentcore-reports-${local.account_id}-${local.region}"
}

# ECR repository for container
resource "aws_ecr_repository" "agent_repo" {
  name                 = "data-analysis-agent"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = false
  }
}

# IAM role for AgentCore
resource "aws_iam_role" "agentcore_role" {
  name = "agentcore-data-analysis-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "bedrock-agentcore.amazonaws.com"
        }
      }
    ]
  })
}

# IAM policy for S3 access
resource "aws_iam_role_policy" "agentcore_s3_policy" {
  name = "agentcore-s3-access"
  role = aws_iam_role.agentcore_role.id

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
          aws_s3_bucket.reports_bucket.arn,
          "${aws_s3_bucket.reports_bucket.arn}/*"
        ]
      }
    ]
  })
}

# IAM policy for basic execution permissions
resource "aws_iam_role_policy" "agentcore_execution_policy" {
  name = "agentcore-execution-access"
  role = aws_iam_role.agentcore_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
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

# Build and push container
resource "null_resource" "build_and_deploy" {
  triggers = {
    app_code_hash = filemd5("${path.root}/../agent.py")
    dockerfile_hash = filemd5("${path.root}/../Dockerfile")
  }

  provisioner "local-exec" {
    command = <<-EOT
      cd ${path.root}/..

      echo "🔨 Building container..."
      docker build -t data-analysis-agent:latest .

      echo "🏷️ Tagging for ECR..."
      docker tag data-analysis-agent:latest ${aws_ecr_repository.agent_repo.repository_url}:latest

      echo "🔑 ECR login..."
      aws ecr get-login-password --region ${local.region} | docker login --username AWS --password-stdin ${aws_ecr_repository.agent_repo.repository_url}

      echo "📤 Pushing to ECR..."
      docker push ${aws_ecr_repository.agent_repo.repository_url}:latest

      echo "🤖 Deploying to AgentCore..."
      agentcore configure --image-uri ${aws_ecr_repository.agent_repo.repository_url}:latest \
        --execution-role-arn ${aws_iam_role.agentcore_role.arn} \
        --region ${local.region}

      agentcore launch

      echo "✅ Deployment complete!"
    EOT
  }

  depends_on = [
    aws_ecr_repository.agent_repo,
    aws_iam_role.agentcore_role,
    aws_iam_role_policy.agentcore_s3_policy
  ]
}