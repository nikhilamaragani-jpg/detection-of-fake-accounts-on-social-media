<div align="center">

# Fake Account Detection on Social Media

### Production-style ML Application · Data Analysis · ETL

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Amaragani Nikhil Sai** · Portfolio system for Junior ML / Data Engineer / Data Analyst roles  
Runnable on sample data. Not a live social-network integration.

</div>

---

## Problem

Fake profiles scale faster than manual review. Trust & safety teams need a repeatable path from raw account features → models → metrics → auditable predictions.

---

## Solution

A **production-style machine learning application**:

- Feature engineering with Pandas / NumPy  
- Multi-model training (RF, LogReg, Gradient Boosting)  
- F1-aware selection, confusion matrices, feature importance  
- Hyperparameter search (GridSearchCV)  
- ETL-style batch scoring + SQLite audit log  
- SQL analytics scripts + EDA report for analyst interviews  

---

## Features

- End-to-end train → evaluate → infer  
- Metrics export for dashboards  
- Batch ETL load  
- SQL queries for risk cohorts  
- Dockerized pipeline  
- pytest for preprocessing  

---

## Architecture

![Pipeline](images/architecture.svg)

---

## Tech stack

Python · Pandas · NumPy · scikit-learn · SQL (SQLite) · Docker · pytest  
Analytics: EDA, Power BI export path, business insights docs

---

## Folder structure

```text
src/  tests/  docs/  data/  models/  notebooks/  scripts/  images/
Dockerfile  docker-compose.yml  requirements.txt  .github/workflows/ci.yml
```

---

## Installation

```bash
git clone https://github.com/nikhilamaragani-jpg/detection-of-fake-accounts-on-social-media.git
cd detection-of-fake-accounts-on-social-media
pip install -r requirements.txt
```

---

## Usage

```bash
python src/main.py
python src/tune_hyperparams.py
python src/etl_batch.py
python scripts/eda_report.py
pytest -q
docker compose up --build
```

---

## Project workflow

1. Load sample CSV  
2. Engineer features / split  
3. Train & compare models  
4. Export metrics  
5. Score batches → SQLite + CSV  
6. Analyze with SQL / Power BI  

---

## Screenshots / results

| Asset | Path |
|-------|------|
| Architecture | [images/architecture.svg](images/architecture.svg) |
| Sample metrics | [data/outputs/metrics.sample.json](data/outputs/metrics.sample.json) |
| EDA output | `data/outputs/eda_report.md` (after script) |

---

## Results

Sample metrics on the included small dataset are illustrative only — see `metrics.sample.json`. Always re-run locally for current numbers.

---

## Future improvements

- [ ] Larger public datasets + imbalance techniques  
- [ ] PostgreSQL sink + incremental loads  
- [ ] Model registry / versioning  
- [ ] Streaming feature ingestion API  

---

## Skills demonstrated

Machine Learning · feature engineering · evaluation · hyperparameter tuning · ETL · SQL analytics · Data Analysis · Docker · documentation

---

## Documentation

[PROJECT_BRIEF](docs/PROJECT_BRIEF.md) · [DATA_ANALYST](docs/DATA_ANALYST.md) · [DATA_ENGINEERING](docs/DATA_ENGINEERING.md) · [ML_NOTES](docs/ML_NOTES.md) · [DEMO](docs/DEMO.md) · [INTERVIEW](docs/INTERVIEW.md) · [RESUME_BULLETS](docs/RESUME_BULLETS.md)

## License

MIT

**Author:** Amaragani Nikhil Sai · https://nikhilamaragani-jpg.github.io/
