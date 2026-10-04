# Data Analyst track

## Stack signals

SQL · Pandas · NumPy · EDA · Power BI · data cleaning · feature engineering · business insights

## Run EDA

```bash
python scripts/eda_report.py
# → data/outputs/eda_report.md
```

The report includes class balance, missingness, duplicate rows, numeric
summaries, and medians by label. Treat these as descriptive statistics: they
do not prove causation or real-world detection performance. See
[`data/README.md`](../data/README.md) before presenting model results.

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

## Responsible interpretation

- Use precision to discuss the share of flagged accounts that are fake, and
  recall to discuss the share of fake-labeled accounts detected.
- Compare false positives and false negatives before proposing an operational
  review threshold; a model score is not proof that an account is fake.
- Avoid claims about "risk segments" or production performance unless supported
  by sufficiently large, representative, independently labeled data.
