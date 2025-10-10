# EventBridge rule to trigger when SageMaker tuning jobs complete
resource "aws_cloudwatch_event_rule" "sagemaker_tuning_complete" {
  name        = "sagemaker-tuning-complete"
  description = "Trigger when SageMaker hyperparameter tuning job completes"

  event_pattern = jsonencode({
    source      = ["aws.sagemaker"]
    detail-type = ["SageMaker HyperParameter Tuning Job State Change"]
    detail = {
      HyperParameterTuningJobStatus = ["Completed"]
    }
  })
}

# SNS topic for tuning completion notifications
resource "aws_sns_topic" "tuning_complete" {
  name = "sagemaker-tuning-complete"
}

# EventBridge target: Send to SNS
resource "aws_cloudwatch_event_target" "tuning_complete_sns" {
  rule      = aws_cloudwatch_event_rule.sagemaker_tuning_complete.name
  target_id = "SendToSNS"
  arn       = aws_sns_topic.tuning_complete.arn
}

# Lambda function to trigger Phase 2 of pipeline
resource "aws_lambda_function" "continue_pipeline" {
  filename      = "lambda_continue_pipeline.zip"
  function_name = "continue-forecasting-pipeline"
  role          = aws_iam_role.lambda_continue_pipeline.arn
  handler       = "continue_pipeline.handler"
  runtime       = "python3.11"
  timeout       = 60
  source_code_hash = filebase64sha256("lambda_continue_pipeline.zip")

  environment {
    variables = {
      AGENTCORE_RUNTIME_ARN = var.agentcore_runtime_arn
      S3_BUCKET             = aws_s3_bucket.reports_bucket.id
    }
  }
}

# IAM role for Lambda
resource "aws_iam_role" "lambda_continue_pipeline" {
  name = "lambda-continue-pipeline-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

# Lambda permissions
resource "aws_iam_role_policy" "lambda_continue_pipeline" {
  name = "lambda-continue-pipeline-policy"
  role = aws_iam_role.lambda_continue_pipeline.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock-agentcore:InvokeAgent",
          "bedrock-agentcore:InvokeAgentRuntime"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = "${aws_s3_bucket.reports_bucket.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# SNS subscription: Lambda
resource "aws_sns_topic_subscription" "tuning_complete_lambda" {
  topic_arn = aws_sns_topic.tuning_complete.arn
  protocol  = "lambda"
  endpoint  = aws_lambda_function.continue_pipeline.arn
}

# Allow SNS to invoke Lambda
resource "aws_lambda_permission" "allow_sns" {
  statement_id  = "AllowExecutionFromSNS"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.continue_pipeline.function_name
  principal     = "sns.amazonaws.com"
  source_arn    = aws_sns_topic.tuning_complete.arn
}

# SNS topic policy to allow EventBridge
resource "aws_sns_topic_policy" "tuning_complete" {
  arn = aws_sns_topic.tuning_complete.arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "events.amazonaws.com"
      }
      Action   = "SNS:Publish"
      Resource = aws_sns_topic.tuning_complete.arn
    }]
  })
}

# Variable for AgentCore runtime ARN
variable "agentcore_runtime_arn" {
  description = "ARN of the AgentCore runtime"
  type        = string
  default     = ""
}

# Output SNS topic ARN
output "tuning_complete_topic_arn" {
  value = aws_sns_topic.tuning_complete.arn
}
