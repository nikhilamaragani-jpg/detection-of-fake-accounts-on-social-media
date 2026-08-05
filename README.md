<div align="center">

# Detection of Fake Accounts on Social Media

### B.Tech Project · Machine Learning · Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Runnable%20ML%20Pipeline-success)](https://github.com/nikhilamaragani-jpg/detection-of-fake-accounts-on-social-media)

**Amaragani Nikhil Sai** · 22X31A0513 · SIIET (JNTUH) · Guide: Mrs. J. Pujitha · 2023–2024

[Quick start](#quick-start) · [Pipeline](#ml-pipeline) · [Scope](#implementation-status) · [Docs](#documentation)

</div>

---

## Problem

Fake profiles can spread spam and misinformation. Manual review does not scale well. This project implements a **complete supervised learning pipeline**: prepare profile/activity features, train and compare classifiers, evaluate metrics, run a sample prediction, and log results.

| Challenge | Approach |
|-----------|----------|
| Weak few-feature filters | Multi-signal feature preparation |
| Single-model habit | Compare Random Forest, Logistic Regression, Gradient Boosting |
| Accuracy-only thinking | Also report **F1** |
| No audit trail | SQLite prediction log |

---

## ML pipeline

```text
Sample account CSV
        |
        v
Preprocess & features → train/test split
        |
        v
Train models (RF · LogReg · GradientBoosting)
        |
        v
Evaluate (accuracy, F1, classification report)
        |
        v
Predict sample profile + log to SQLite
```

---

## Tech stack

| Area | Technology |
|------|------------|
| Language | Python 3 |
| Data | pandas, NumPy |
| ML | scikit-learn |
| Storage | SQLite |
| Data file | `data/sample_social_accounts.csv` |

---

## Quick start

```bash
git clone https://github.com/nikhilamaragani-jpg/detection-of-fake-accounts-on-social-media.git
cd detection-of-fake-accounts-on-social-media
pip install -r requirements.txt
python src/main.py
```

---

## Skills demonstrated

| Skill | Evidence |
|-------|----------|
| ML fundamentals | Train / evaluate / predict loop |
| Feature thinking | Profile & activity signals |
| Experimentation | Multi-model comparison |
| Metrics literacy | Accuracy + F1 |
| Engineering hygiene | Modular `src/`, requirements, docs |

---

## Implementation status

**Runnable prototype**
- [x] End-to-end supervised pipeline  
- [x] Feature preparation  
- [x] Multi-model training & evaluation  
- [x] Sample prediction + SQLite log  

**Full report / future extensions**
- [ ] Live social API ingestion  
- [ ] Graph / network features at scale  
- [ ] Continuous monitoring loop  

---

## Documentation

| File | Purpose |
|------|---------|
| [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md) | Brief |
| [docs/DEMO.md](docs/DEMO.md) | Demo |
| [docs/INTERVIEW.md](docs/INTERVIEW.md) | Walkthrough |
| [docs/RESUME_BULLETS.md](docs/RESUME_BULLETS.md) | Bullets |
| [docs/ABOUT_TOPICS.md](docs/ABOUT_TOPICS.md) | Topics |

**Suggested topics:** `python` · `machine-learning` · `scikit-learn` · `classification` · `trust-and-safety`

---

## Author

**Amaragani Nikhil Sai** · B.Tech CSE  
Portfolio: https://nikhilamaragani-jpg.github.io/  
Email: nikhilamaragani@gmail.com

## License

MIT — see [LICENSE](LICENSE).
