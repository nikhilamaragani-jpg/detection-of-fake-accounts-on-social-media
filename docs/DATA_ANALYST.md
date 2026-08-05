# Data Analyst angle

## SQL-friendly questions (SQLite after runs)

```sql
-- After predictions are logged, explore counts by label (example schema)
-- Adapt to your local table from database.py
SELECT predicted_label, COUNT(*) FROM predictions GROUP BY predicted_label;
```

## Pandas EDA checklist

1. Class balance of `is_fake`  
2. Distributions of followers / following / account age  
3. Correlation of `follower_following_ratio` with label  
4. Missingness (sample data is clean by design)  

## Power BI

1. Run `python src/main.py` and `python src/etl_batch.py`  
2. Import `data/outputs/batch_predictions.csv` and `metrics.json`  
3. Build visuals: predicted label share, p_fake histogram, feature importance bar chart  

## Business insights language

- Low age + high following + missing profile assets → elevated risk cohort  
- F1 preferred over accuracy when genuine accounts dominate  
