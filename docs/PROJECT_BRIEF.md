# Project brief — Fake Account Detection

| Field | Detail |
|-------|--------|
| Type | B.Tech real-time / research project (**2023–2024**) |
| Author | Amaragani Nikhil Sai (**22X31A0513**) |
| Guide | Mrs. J. Pujitha |
| Institution | SIIET (JNTUH) |
| Stack | Python, Pandas, NumPy, scikit-learn, SQLite, Docker |

## Goal

Build an automated classification pipeline for suspicious social accounts using profile/activity features, multi-model comparison, and auditable predictions.

## Prototype vs report

- **Repo:** end-to-end train / evaluate / predict on sample CSV (+ synthetic fallback).  
- **Report:** full academic write-up for the real-time project submission.  

## Implementation alignment

The academic report describes a proposed social-platform system. Those
requirements should not be presented as implemented features of this
repository.

| Report concept | Repository status |
| --- | --- |
| Profile and activity data collection from social platforms | Not implemented; the pipeline reads a local CSV or creates synthetic data |
| Behavior signals such as spam comments, engagement, and artificial activity | Not present in the bundled dataset or model; current inputs are profile/count features |
| Gradient boosting and model comparison | Random Forest, Logistic Regression, and Gradient Boosting are trained and compared |
| Account prediction | Implemented as local sample inference and batch scoring |
| Admin dashboard, alerts, and real-time monitoring | Not implemented |
| Privacy, consent, and platform authorization | No user data collection is implemented; use only authorized data if extending the project |

The next evidence-based extension is to obtain a properly licensed dataset
with documented collection and independently reviewed labels, then evaluate
additional behavioral features without claiming live detection. Do not invent
report-described signals in the sample data or infer platform access that the
repository does not have.

See [Dataset notes](../data/README.md) and [the report-to-repository summary](REPORT_SUMMARY.md).
