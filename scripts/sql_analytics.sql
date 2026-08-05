-- SQL analytics for prediction audit table (SQLite)
-- Run after: python src/main.py or python src/etl_batch.py
-- Example: sqlite3 data/predictions.db < scripts/sql_analytics.sql

.headers on
.mode column

-- Label distribution
SELECT predicted_label,
       COUNT(*) AS n,
       ROUND(AVG(predicted_probability), 3) AS avg_p_fake
FROM predictions
GROUP BY predicted_label
ORDER BY predicted_label;

-- High-risk cohort: young accounts with aggressive following
SELECT id, account_age_days, followers, following, predicted_label, predicted_probability
FROM predictions
WHERE account_age_days < 30 AND following > 500
ORDER BY predicted_probability DESC
LIMIT 20;

-- Daily volume (if multiple runs)
SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS scored_rows
FROM predictions
GROUP BY substr(created_at, 1, 10)
ORDER BY day DESC;
