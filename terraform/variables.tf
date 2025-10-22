# Terraform Variables for Intelligent Forecasting System
# Customize these values via terraform.tfvars

variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "aws_account_id" {
  description = "AWS account ID"
  type        = string
}

variable "bucket_name" {
  description = "S3 bucket name for SageMaker artifacts"
  type        = string
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "forecasting-pipeline"
}

variable "code_interpreter_name" {
  description = "Name for Bedrock Code Interpreter"
  type        = string
  default     = "forecasting_code_interpreter"
}

variable "sagemaker_instance_type" {
  description = "Default SageMaker instance type for training"
  type        = string
  default     = "ml.m5.xlarge"
}

variable "sagemaker_inference_instance_type" {
  description = "SageMaker instance type for inference endpoints"
  type        = string
  default     = "ml.m5.xlarge"
}

variable "enable_vpc" {
  description = "Enable VPC isolation for SageMaker"
  type        = bool
  default     = false
}

variable "tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Project   = "intelligent-forecasting"
    ManagedBy = "terraform"
  }
}