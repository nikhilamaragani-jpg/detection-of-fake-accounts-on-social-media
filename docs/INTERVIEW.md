# Project walkthrough — Fake Account Detection

## 60-second summary

I built an end-to-end ML pipeline: load data, engineer features, train Random Forest / Logistic Regression / Gradient Boosting, evaluate with accuracy and F1, predict a suspicious sample, and log results to SQLite.

## Demo

```bash
pip install -r requirements.txt
python src/main.py
```

## Questions

**Why F1?** Class imbalance can make accuracy misleading.  
**What features?** Account age, followers/following, posts, profile completeness, ratios.  
**Prototype honesty?** Demo uses sample data; larger real datasets and continuous retraining are future work.
