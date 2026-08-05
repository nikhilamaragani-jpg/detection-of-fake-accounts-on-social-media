# Data Engineering angle

## Pipeline stages

| Stage | Implementation |
|-------|----------------|
| Extract | CSV / synthetic generator |
| Transform | Feature ratios, train/test split |
| Load | SQLite prediction log + CSV batch export |

## Incremental loads (roadmap)

- Track `account_id` + `scored_at`  
- Score only new/changed rows  
- Partition output by date  

## Scheduling

- Local: cron / Task Scheduler calling `python src/etl_batch.py`  
- Cloud: GitHub Actions scheduled workflow or Airflow DAG (not included)  

## Logging

Console metrics + SQLite audit + file artifacts under `data/outputs/`.
