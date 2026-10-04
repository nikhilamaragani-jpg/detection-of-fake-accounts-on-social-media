# Project walkthrough — Fake Account Detection

## 60-second pitch

My B.Tech project is *Detection of Fake Accounts on Social Media*. This
repository contains a reproducible offline prototype: it validates tabular
profile features, compares Random Forest, Logistic Regression, and Gradient
Boosting using training-fold cross-validation, reports class-aware holdout
metrics, and logs sample and batch predictions to SQLite. The bundled dataset
has only 30 illustrative rows with undocumented provenance, so its scores are
not evidence of real-world performance. The report proposes platform
collection, behavioral signals, dashboards, and real-time monitoring; those
capabilities are not implemented here.

## Demo

```bash
pip install -r requirements.txt
python src/main.py
```

## Questions

**Why cross-validation?** Model selection uses stratified folds within the
training data; the holdout is reserved for a final exploratory evaluation.
**Why F1?** It balances precision and recall for the fake label, but I also
review each metric and the confusion matrix because false positives and false
negatives have different costs.

**Features?** Account age, follower/following/post counts, profile photo and
bio indicators, and the derived follower/following ratio.

**What is real-time?** It is the academic project title. This repository runs
local inference and CSV batch scoring; it does not ingest live platform events
or provide a streaming service.
