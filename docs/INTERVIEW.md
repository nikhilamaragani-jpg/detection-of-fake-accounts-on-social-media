# Project walkthrough — Fake Account Detection

## 60-second summary

Fake accounts undermine platform trust. I built an end-to-end ML pipeline: load data, engineer features, train Random Forest / Logistic Regression / Gradient Boosting, evaluate with accuracy and F1, predict a suspicious sample, and log results to SQLite.

## Demo

```bash
pip install -r requirements.txt
python src/main.py
```

Point to printed metrics and the sample prediction.

## Questions

**Why F1?** Class imbalance can make accuracy misleading.  
**What features?** Account age, followers/following, posts, profile completeness, ratios.  
**Prototype honesty?** Demo uses sample data; production would need continuous retraining and privacy-aware data handling.
