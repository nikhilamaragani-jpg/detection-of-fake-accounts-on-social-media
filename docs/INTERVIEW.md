# Interview Guide — Fake Account Detection

## 60-second pitch

> Fake social accounts drive spam, scams, and misinformation. Manual review does not scale. I built an end-to-end ML pipeline that prepares profile/activity features, trains and compares classifiers (Random Forest, Logistic Regression, Gradient Boosting), evaluates with accuracy/F1, predicts on a suspicious sample, and writes an audit log to SQLite.

## Problem → Solution → Impact

| | |
|--|--|
| **Problem** | Fake accounts at scale; weak filters; false positives/negatives |
| **Solution** | Multi-feature supervised classification + model comparison |
| **Impact** | Faster triage, measurable metrics, auditable predictions |

## Pipeline (say this out loud)

1. Load CSV / sample social account data
2. Clean + engineer features (followers, following, posts, bio, ratios…)
3. Train multiple models
4. Select best by **F1** (trust & safety imbalance awareness)
5. Predict + log for review

## Expected questions

**Q: Why F1 over accuracy?**  
A: Fake-class imbalance makes accuracy misleading; F1 balances precision/recall for the minority risk class.

**Q: What features matter most?**  
A: Account age, follower/following ratio, profile completeness, posting activity; report also discusses engagement/spam signals.

**Q: How do attackers evade?**  
A: They mimic real ratios and buy engagement. Need continuous retraining, graph features, and human review for edge cases.

**Q: Privacy concerns?**  
A: Minimize personal content analysis; prefer behavioral aggregates; respect platform ToS and regional privacy laws.

## Demo script

```bash
pip install -r requirements.txt
python src/main.py
```

Point to printed model metrics and the sample fake-profile prediction.

## Resume bullets

- Built an end-to-end **fake-account detection** ML pipeline with feature engineering, multi-model comparison, and SQLite prediction logging.
- Evaluated classifiers using accuracy and F1; selected models with imbalance-aware metrics suitable for trust-and-safety workflows.
- Framed detection as a scalable alternative to manual review on social platforms.
