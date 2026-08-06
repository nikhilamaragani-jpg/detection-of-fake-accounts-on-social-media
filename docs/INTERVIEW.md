# Project walkthrough — Fake Account Detection

## 60-second pitch

My B.Tech real-time project (2023–24, guide Mrs. J. Pujitha, roll 22X31A0513) is *Detection of Fake Accounts on Social Media*. On GitHub I ship a complete supervised pipeline: load features, train Random Forest / Logistic Regression / Gradient Boosting, compare with accuracy and F1, score a sample profile, and log results to SQLite.

## Demo

```bash
pip install -r requirements.txt
python src/main.py
```

## Questions

**Why F1?** Imbalance can make accuracy alone misleading.  
**Features?** Account age, followers/following, posts, profile completeness, ratios.  
**Honesty?** Sample data demo — not live platform APIs.
