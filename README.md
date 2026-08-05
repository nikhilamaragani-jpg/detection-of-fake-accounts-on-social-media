# Detection of Fake Accounts on Social Media

**B.Tech Project** | Machine Learning | Classification | Social Media Analysis

A practical machine learning prototype for detecting potentially fake social media accounts using behavioral and profile features, model training/comparison, and prediction logging. Inspired by research on OSNs, feature selection, and hybrid classifiers (including concepts from SVM-NN and gradient boosting approaches).

---

## Overview

Fake accounts spread misinformation, engage in fraud, and undermine trust. This project implements a complete pipeline:

1. Load and explore a sample social accounts dataset
2. Preprocess features (followers, following, posts, bio signals, etc.)
3. Train and compare classifiers (Random Forest + Logistic Regression)
4. Evaluate performance
5. Run sample predictions
6. Log results in SQLite for auditability

**Status:** Runnable end-to-end ML prototype with sample CSV dataset  
**Focus:** Feature engineering, classification, and practical detection workflow

---

## System Architecture (Aligned with Project Report)

```text
Data Collection / Sample CSV
        |
        v
+---------------------------+
| Preprocessing Layer       |  Cleaning, encoding, feature prep
+---------------------------+
        |
        v
+---------------------------+
| Feature Extraction        |  Profile completeness, activity, ratios
+---------------------------+
        |
        v
+---------------------------+
| Detection Models          |  RandomForest, Logistic Regression
|                           |  (concepts extendable to Gradient Boosting / hybrid)
+---------------------------+
        |
        v
+---------------------------+
| Decision + Logging        |  Prediction + SQLite audit log
+---------------------------+
```

---

## Tech Stack

| Area              | Technology                          |
|-------------------|-------------------------------------|
| Language          | Python 3                            |
| ML Framework      | scikit-learn                        |
| Data Handling     | pandas, NumPy                       |
| Storage           | SQLite                              |
| Dataset           | Sample CSV (150+ synthetic rows)    |

---

## Project Structure

```text
detection-of-fake-accounts-on-social-media/
├── data/
│   └── sample_social_accounts.csv
├── src/
│   ├── main.py           # End-to-end pipeline
│   ├── preprocess.py     # Data loading & feature prep
│   ├── model.py          # Training, evaluation, prediction
│   └── database.py       # SQLite prediction logging
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/nikhilamaragani-jpg/detection-of-fake-accounts-on-social-media.git
cd detection-of-fake-accounts-on-social-media
pip install -r requirements.txt
python src/main.py
```

The script will train models, print evaluation metrics, run a sample prediction, and store the log in SQLite.

---

## Key Ideas from Project Scope

- Behavioral & profile features for anomaly detection
- Model comparison to select stronger classifiers
- Handling missing/incomplete signals (robustness focus)
- Logging for transparency and review
- Extensible toward real-time monitoring and larger feature sets (engagement rate, spam signals, graph features)

---

## Author

**Amaragani Nikhil Sai**  
B.Tech in Computer Science and Engineering  
Sri Indu Institute of Engineering and Technology

- GitHub: [nikhilamaragani-jpg](https://github.com/nikhilamaragani-jpg)
- LinkedIn: [Amaragani Nikhil Sai](https://linkedin.com/in/amaraganinikhilsai)
- Email: nikhilamaragani@gmail.com

---

## License

MIT License
