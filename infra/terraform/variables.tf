variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project" {
  type    = string
  default = "rag-platform"
}

variable "db_username" {
  type      = string
  default   = "rag_admin"
  sensitive = true
}

variable "db_password" {
  type      = string
  sensitive = true
}
