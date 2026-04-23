# Production RDS Postgres with pgvector extension enabled.
# For local dev, use docker-compose (postgres-pgvector service).

resource "aws_db_parameter_group" "pgvector" {
  name   = "${var.project}-pgvector"
  family = "postgres16"

  parameter {
    name         = "shared_preload_libraries"
    value        = "vector"
    apply_method = "pending-reboot"
  }
}

resource "aws_db_instance" "pgvector" {
  identifier              = "${var.project}-pgvector"
  engine                  = "postgres"
  engine_version          = "16.3"
  instance_class          = "db.t4g.medium"
  allocated_storage       = 50
  storage_encrypted       = true
  db_name                 = "rag"
  username                = var.db_username
  password                = var.db_password
  parameter_group_name    = aws_db_parameter_group.pgvector.name
  publicly_accessible     = false
  backup_retention_period = 7
  skip_final_snapshot     = false
  deletion_protection     = true
  multi_az                = false

  tags = {
    Project   = var.project
    Component = "vector-store"
  }
}

output "pgvector_endpoint" {
  value = aws_db_instance.pgvector.endpoint
}
