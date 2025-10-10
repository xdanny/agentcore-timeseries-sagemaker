output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.agent_repo.repository_url
}

output "agentcore_role_arn" {
  description = "AgentCore execution role ARN"
  value       = aws_iam_role.agentcore_role.arn
}

output "sagemaker_role_arn" {
  description = "SageMaker execution role ARN (same as AgentCore role with SageMaker permissions)"
  value       = aws_iam_role.agentcore_role.arn
}

output "s3_bucket_name" {
  description = "S3 bucket for reports"
  value       = aws_s3_bucket.reports_bucket.bucket
}

output "agentcore_endpoint" {
  description = "Command to invoke the AgentCore endpoint"
  value       = "aws bedrock-agentcore invoke-actor --actor-id data-analysis-agent --region ${local.region} --payload '{\"num_points\": 100}'"
}

output "deployment_status" {
  description = "Deployment completion status"
  value       = "Complete - Use the agentcore_endpoint command to test"
}