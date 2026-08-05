# Detection of Fake Accounts on Social Media

**B.Tech Project** | Machine Learning | Classification | Social Media Analysis

A practical machine learning prototype for detecting potentially fake social media accounts using behavioral features, model comparison, and prediction logging.

---

## Overview

This project builds an end-to-end classification pipeline:

1. Generate/prepare account feature data
2. Preprocess and split the dataset
3. Train multiple classifiers
4. Evaluate and select the better model
5. Run a sample prediction
6. Store prediction logs in SQLite

**Project Type:** Academic Prototype with working ML pipeline  
**Status:** Runnable training + prediction + database logging

---

## Architecture

```text
Raw/Synthetic Account Data
            |
            v
+----------------------+
| Preprocessing Module |  (cleaning, feature prep, train/test split)
+----------------------+
            |
            v
+----------------------+
| Model Training       |  (RandomForest, Logistic Regression)
+----------------------+
            |
            v
+----------------------+
| Evaluation           |  (accuracy + classification report)
+----------------------+
            |
            v
+----------------------+
| Prediction + SQLite  |  (log predicted label & probability)
+----------------------+
```

---

## Features

- Synthetic demo dataset for local testing
- Feature engineering (including follower/following ratio)
- Multiple model comparison
- Probability-based prediction output
- SQLite logging of predictions

---

## Tech Stack

| Area | Technology |
|------|------------|
| Language | Python |
| Data | Pandas, NumPy |
| ML | Scikit-learn |
| Storage | SQLite |
| Tools | Git |

---

## Project Structure

```text
detection-of-fake-accounts-on-social-media/
├── README.md
├── requirements.txt
├── data/                 # predictions.db created at runtime
├── src/
│   ├── main.py
│   ├── preprocess.py
│   ├── model.py
│   └── database.py
└── LICENSE
```

---

## How to Run

```bash
git clone https://github.com/nikhilamaragani-jpg/detection-of-fake-accounts-on-social-media.git
cd detection-of-fake-accounts-on-social-media

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python src/main.py
```

---

## Current Status

- [x] Problem definition
- [x] Feature-based pipeline
- [x] Multi-model training and evaluation
- [x] Sample prediction
- [x] SQLite prediction logging
- [ ] Real-world dataset integration
- [ ] Advanced feature set and model tuning

---

## Learning Outcomes

- Supervised classification workflow
- Feature preparation for tabular ML
- Model comparison and evaluation
- Lightweight database logging for ML outputs

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
