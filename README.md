<div align="center">

# Detection of Fake Accounts on Social Media

### B.Tech Research / Real-Time Project · Machine Learning · Trust & Safety

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Runnable%20ML%20Pipeline-success)](https://github.com/nikhilamaragani-jpg/detection-of-fake-accounts-on-social-media)

**Author:** Amaragani Nikhil Sai (22X31A0513)  
**Institution:** Sri Indu Institute of Engineering and Technology (JNTUH)  
**Guide:** Mrs. J. Pujitha · Department of CSE · 2023–2024

[Run](#quick-start) · [Pipeline](#ml-pipeline) · [Skills](#skills-recruiters-care-about) · [Docs](docs/PROJECT_BRIEF.md)

</div>

---

## Executive Summary (for recruiters)

Fake social accounts enable spam, scams, misinformation, and platform abuse. Manual review does not scale, and simplistic filters are easy to evade.

This project implements a **complete machine learning detection pipeline**:

1. Load social-account style data  
2. Clean and engineer features (profile/activity signals)  
3. Train and **compare classifiers**  
4. Evaluate performance  
5. Run sample predictions  
6. **Log outcomes** for auditability  

Report themes include behavioral analysis, engagement-related signals, and moving beyond weak single-feature heuristics toward stronger automated detection (including gradient-boosting style thinking in the full documentation).

---

## Problem Statement

| Challenge | Project response |
|-----------|------------------|
| Sybil / fake profiles at scale | Automated classification pipeline |
| Too few features in legacy filters | Multi-signal feature prep |
| Manual review bottlenecks | Train → predict → log workflow |
| False positives / false negatives | Model comparison + evaluation metrics |
| Need for transparency | SQLite prediction audit log |

---

## Objectives (from project report)

- Detect fraudulent / misleading accounts created for malicious purposes
- Improve user trust and platform integrity
- Balance false positives vs false negatives
- Demonstrate ML methods (beyond brittle manual rules)
- Support awareness of cybercrime and social-media abuse patterns

---

## ML Pipeline

```text
Sample / collected account data (CSV)
              |
              v
┌─────────────────────────────┐
│ Preprocessing               │  cleaning · encoding · feature prep
└─────────────────────────────┘
              |
              v
┌─────────────────────────────┐
│ Feature signals             │  followers, following, posts, bio, ratios…
└─────────────────────────────┘
              |
              v
┌─────────────────────────────┐
│ Models                      │  Random Forest · Logistic Regression
│                             │  (extendable to Gradient Boosting / hybrid)
└─────────────────────────────┘
              |
              v
┌─────────────────────────────┐
│ Evaluation + Prediction     │  metrics · sample inference
└─────────────────────────────┘
              |
              v
┌─────────────────────────────┐
│ Audit log (SQLite)          │  prediction history for review
└─────────────────────────────┘
```

---

## Tech Stack

| Area | Technology |
|------|------------|
| Language | Python 3 |
| Data | pandas, NumPy |
| ML | scikit-learn |
| Storage | SQLite |
| Dataset | Sample CSV included |
| Report tooling concepts | Scrapy / APIs / MongoDB / Spark (scope vision) |

---

## Repository Structure

```text
detection-of-fake-accounts-on-social-media/
├── data/
│   └── sample_social_accounts.csv
├── docs/
│   └── PROJECT_BRIEF.md
├── notebooks/
│   └── analysis_overview.md
├── src/
│   ├── main.py
│   ├── preprocess.py
│   ├── model.py
│   └── database.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/nikhilamaragani-jpg/detection-of-fake-accounts-on-social-media.git
cd detection-of-fake-accounts-on-social-media
pip install -r requirements.txt
python src/main.py
```

Expected behavior: train models → print evaluation metrics → run a sample prediction → write log rows to SQLite.

---

## Features

- [x] End-to-end supervised classification pipeline
- [x] Profile / activity feature preparation
- [x] Multi-model comparison
- [x] Metrics-based evaluation
- [x] Prediction logging
- [ ] Live social API ingestion
- [ ] Graph / network features at production scale
- [ ] Continuous learning feedback loop

---

## Skills Recruiters Care About

| Skill | Evidence |
|-------|----------|
| ML fundamentals | Train/eval/predict loop |
| Feature thinking | Behavioral & profile signals |
| Experimentation | Model comparison |
| Trust & safety domain awareness | Problem framing from OSN abuse |
| Data hygiene | Preprocessing module |
| Engineering hygiene | Modular `src/`, requirements, docs |

---

## Academic Context

- **Type:** Real-time / research project report + runnable ML prototype
- **College:** Sri Indu Institute of Engineering and Technology (Autonomous)
- **Student:** A. Nikhil Sai · 22X31A0513
- **Declaration date (report):** 03/07/2024

---

## Author

**Amaragani Nikhil Sai**  
B.Tech CSE · Applied ML / Detection Systems

- GitHub: [nikhilamaragani-jpg](https://github.com/nikhilamaragani-jpg)
- LinkedIn: [nikhil-sai-amaragani](https://www.linkedin.com/in/nikhil-sai-amaragani-219115382)
- Email: nikhilamaragani@gmail.com

---

## License

MIT License — see [LICENSE](LICENSE).
