# Data Retention Policy

Raw bronze data is retained for 90 days then expired by S3 lifecycle. Silver chunks are retained for 2 years. Gold evaluation snapshots are retained indefinitely.

All Delta tables have `delta.logRetentionDuration = 'interval 30 days'` and `VACUUM` runs weekly with `RETAIN 168 HOURS`.
