# Data Engineering track

## Pipeline architecture

```text
Extract (CSV / generator)
   → Transform (features, ratios, train/test)
   → Train / score
   → Load (SQLite audit + CSV batch export)
```

## Implemented

| Piece | Location |
|-------|----------|
| Batch ETL scoring | `src/etl_batch.py` |
| Prediction audit log | `src/database.py` |
| Metrics artifacts | `src/metrics_export.py` |
| SQL analytics | `scripts/sql_analytics.sql` |
| Docker batch | `Dockerfile` / `docker-compose.yml` |

## PostgreSQL upgrade (TODO — not live)

- Replace SQLite with PostgreSQL DSN via env  
- Table `predictions` with indexes on `created_at`, `predicted_label`  
- Incremental load key: `account_id` + content hash  

## Scheduling (TODO)

- GitHub Actions `schedule:` cron  
- Or Airflow/Prefect DAG calling `python src/etl_batch.py`  

## API ingestion (TODO)

- REST endpoint accepting feature JSON batches  
- Validate with Pydantic, append to store  
