# Data Analyst track

## Stack signals

SQL · Pandas · NumPy · EDA · Power BI · data cleaning · feature engineering · business insights

## Run EDA

```bash
python scripts/eda_report.py
# → data/outputs/eda_report.md
```

## SQL

```bash
python src/etl_batch.py
sqlite3 data/predictions.db < scripts/sql_analytics.sql
```

Queries cover label mix, high-risk cohorts, and daily volume.

## Power BI

1. Import `data/sample_social_accounts.csv`  
2. Import `data/outputs/batch_predictions.csv` (after ETL)  
3. Import `data/outputs/metrics.json`  
4. Build:  
   - % fake vs genuine  
   - p_fake histogram  
   - followers vs following scatter by label  
   - feature importance bar chart  

## Business insights language

- Young accounts + high following + missing profile photo → elevated risk segment  
- Completeness features (bio, photo) improve precision of review queues  
- F1 preferred when genuine accounts dominate traffic  
