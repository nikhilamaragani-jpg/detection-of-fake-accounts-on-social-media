# Detection of Fake Accounts on Social Media

**B.Tech Project** | Machine Learning | Classification | Social Media Analysis

A practical machine learning prototype for detecting potentially fake social media accounts using behavioral features, model comparison, and prediction logging.

---

## Overview

Pipeline:

1. Load sample CSV dataset (`data/sample_social_accounts.csv`)
2. Preprocess and split data
3. Train RandomForest + Logistic Regression
4. Evaluate models
5. Run sample prediction
6. Store prediction logs in SQLite

**Status:** Runnable ML prototype with sample dataset

---

## Architecture

```text
CSV Dataset
    |
    v
Preprocessing
    |
    v
Model Training (RF + LogReg)
    |
    v
Evaluation + Best Model Selection
    |
    v
Prediction + SQLite Logging
```

---

## Project Structure

```text
detection-of-fake-accounts-on-social-media/
├── data/sample_social_accounts.csv
├── notebooks/analysis_overview.md
├── src/
│   ├── main.py
│   ├── preprocess.py
│   ├── model.py
│   └── database.py
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
pip install -r requirements.txt
python src/main.py
```

---

## Author

**Amaragani Nikhil Sai**  
B.Tech in Computer Science and Engineering

- GitHub: [nikhilamaragani-jpg](https://github.com/nikhilamaragani-jpg)
- LinkedIn: [Amaragani Nikhil Sai](https://linkedin.com/in/amaraganinikhilsai)
- Email: nikhilamaragani@gmail.com

---

## License

MIT License
