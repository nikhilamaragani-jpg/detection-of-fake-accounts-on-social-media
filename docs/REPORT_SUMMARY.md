# Academic Report Summary — Fake Account Detection

**Title:** Detection of Fake Accounts on Social Media  
**Student:** A. Nikhil Sai · 22X31A0513  
**Institution:** Sri Indu Institute of Engineering and Technology (JNTUH)  
**Guide:** Mrs. J. Pujitha  
**Year:** 2023–2024 · Declaration date: 03/07/2024

## Motivation

Online social networks concentrate personal data and attention, attracting Sybil attacks, spam, scams, and influence operations. Fake profiles damage user trust and create operational risk for platforms and brands.

## Problem definition

Develop an automated system to identify and flag fake accounts while balancing false positives (real accounts marked fake) and false negatives (fakes missed).

## Objectives

- Enhance user trust and safety
- Protect authentic user data
- Maintain platform integrity
- Reduce cyber-crime risk via better detection methods

## Proposed approach (report)

- Move beyond weak few-feature filters and pure manual review
- Use richer signals (spam commenting, engagement rate, artificial activity concepts)
- Prefer algorithms robust under missing inputs (Gradient Boosting discussed vs Random Forest limitations)
- Continuous improvement via feedback loops

## Technologies discussed

Scrapy / BeautifulSoup / Selenium, social APIs, SQL/NoSQL storage, Spark/Hadoop-style processing concepts, ML models (SVM, RF, Neural Nets, Gradient Boosting)

## Repository mapping

| Report concept | Repo module |
|----------------|-------------|
| Preprocessing | `src/preprocess.py` |
| Models incl. boosting | `src/model.py` |
| Audit / logging | `src/database.py` |
| End-to-end run | `src/main.py` |
| Sample data | `data/sample_social_accounts.csv` |

## Full PDF

Place official report at:

`docs/reports/REAL_TIME_PROJECT_FAKE_ACCOUNT_DETECTION.pdf`

(Local source: `OneDrive/Documents/B.TECH PROJECTS/REAL TIME PROJECT OF SIIET.pdf`)
