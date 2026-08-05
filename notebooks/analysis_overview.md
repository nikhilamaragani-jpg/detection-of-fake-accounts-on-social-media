# Fake Account Detection - Analysis Overview

## Dataset

- File: `data/sample_social_accounts.csv`
- Purpose: Demo dataset for training and evaluating a basic fake-account classifier

## Features

- `account_age_days`
- `followers`
- `following`
- `posts_count`
- `has_profile_pic`
- `has_bio`
- `follower_following_ratio`

## Label

- `is_fake` = 1 for suspicious accounts, 0 for genuine accounts

## Pipeline Steps

1. Load CSV dataset
2. Split into train/test sets
3. Train RandomForest and Logistic Regression
4. Compare model performance
5. Run one sample prediction
6. Store prediction log in SQLite

## How to Run

```bash
pip install -r requirements.txt
python src/main.py
```

## Notes

This is an academic prototype. For stronger results, replace the sample CSV with a real labeled social-media dataset and expand feature engineering.
